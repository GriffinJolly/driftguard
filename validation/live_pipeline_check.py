"""
validation/live_pipeline_check.py

End-to-end check of DriftGuard on LIVE API calls.

Runs the production evaluation path -- driftguard.evalsuite.runner.run_once,
the same function the scheduled job uses -- against a real provider, writing
every call and every per-run metric point into its own SQLite database. After
a healthy phase, a known fault is injected for a second phase, then the whole
detection + classification stack is run over what was collected:

    live calls -> runner -> store (log_rows + metric_series)
               -> per-call Page-Hinkley + PELT/permutation/BH  (detection/pipeline.py)
               -> rules.py labels -> incidents.py -> mapper.py plans -> evidence.py flip sets

The fault: during the fault phase a fraction of calls is sent with
max_tokens=16 instead of the normal budget. A reasoning model cut off that early
cannot finish its answer, so accuracy AND format should both degrade -- a
multi-signal incident. Every recorded outcome is still a real API response; the
fault only changes the request, the same way the paper's induced-drift trials did.
The fault start time is written to a ground-truth file that the detectors never
see; it is only used to score the result afterwards.

Usage:
    python -m validation.live_pipeline_check                      # collect, then analyse
    python -m validation.live_pipeline_check --analyse-only       # re-analyse an existing database
    streamlit run driftguard/dashboard/app.py                     # with DRIFTGUARD_DB pointed at the database
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DB = Path("validation/data/live_check.db")
OUT = Path("validation/results/live_pipeline_check")


def _ground_truth_path(db: Path) -> Path:
    return db.with_suffix(".ground_truth.json")


def collect(provider: str, model: str, db: Path, healthy_runs: int, fault_runs: int,
            fault_frac: float, max_tokens: int, rpm: int, seed: int) -> None:
    from driftguard.evalsuite import format_check, refusal_probe, runner, tasks
    from driftguard.providers import clients
    from driftguard.storage.store import DriftStore

    clients.PROVIDERS[provider].rate_limiter.max_calls = rpm   # stay inside the free tier's token budget
    rng = random.Random(seed)
    state = {"fault": False}
    real_call = clients.call_model

    def call_with_params(*args, **kwargs):
        params = {"temperature": 0, "max_tokens": max_tokens}
        if state["fault"] and rng.random() < fault_frac:
            params["max_tokens"] = 16                          # the injected fault
        kwargs["request_params"] = {**params, **(kwargs.get("request_params") or {})}
        return real_call(*args, **kwargs)

    for module in (tasks, format_check, refusal_probe):        # the eval tasks import call_model by name
        module.call_model = call_with_params

    db.parent.mkdir(parents=True, exist_ok=True)
    store = DriftStore(db_path=str(db))
    gt_path = _ground_truth_path(db)
    total = healthy_runs + fault_runs
    print(f"Collecting {total} runs ({healthy_runs} healthy, {fault_runs} with {fault_frac:.0%} of calls "
          f"cut to 16 tokens) from {provider} {model} at <= {rpm} calls/min into {db}")
    for i in range(total):
        if i == healthy_runs:
            state["fault"] = True
            start = datetime.now(timezone.utc)
            gt_path.write_text(json.dumps({"fault_start_utc": start.isoformat(), "fault_frac": fault_frac,
                                           "fault": "max_tokens=16", "provider": provider, "model": model,
                                           "healthy_runs": healthy_runs, "fault_runs": fault_runs}, indent=2))
            print(f"--- fault phase starts {start:%H:%M:%S} UTC (ground truth written to {gt_path})")
        t0 = time.time()
        p = runner.run_once(provider, model, store=store)
        print(f"run {i + 1:>2}/{total} {'FAULT ' if state['fault'] else 'healthy'} "
              f"acc={p.accuracy} fmt={p.format_adherence} refusal={p.refusal_rate} "
              f"p95={p.latency_p95_ms and round(p.latency_p95_ms)}ms calls={p.n_calls} ({time.time() - t0:.0f}s)",
              flush=True)


def analyse(db: Path, provider: str, model: str) -> None:
    from driftguard.dashboard import data as D
    from driftguard.detection.signals import SIGNALS

    gt = json.loads(_ground_truth_path(db).read_text()) if _ground_truth_path(db).exists() else None
    fault_start = pd.Timestamp(gt["fault_start_utc"]) if gt else None
    calls = D.load_calls(db)
    calls = calls[(calls["provider"] == provider) & (calls["model_id"] == model)].reset_index(drop=True)
    a = D.analyse_scope(calls, provider, model, D.load_settings(), sigmas=D.load_sigmas())
    results = a.results
    shifts = D.shift_table(a)

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, r in results.items():
        st = r.stream
        onset = int(st.timestamp.searchsorted(fault_start)) if fault_start is not None else None
        conf = [t["index"] for t in r.tested if t["significant"]]
        before = st.values[:onset].mean() if onset else np.nan
        after = st.values[onset:].mean() if onset is not None and onset < len(st.values) else np.nan
        rows.append(dict(signal=SIGNALS[key].label, metric=SIGNALS[key].metric, calls=len(st.values),
                         fault_onset_index=onset, bad_rate_before=before, bad_rate_after=after,
                         confirmed_at=conf, confirmed_offset=[c - onset for c in conf] if onset is not None else None,
                         ph_alerts_at=r.alerts,
                         ph_offset=[i - onset for i in r.alerts] if onset is not None else None,
                         ph_alerts_before_fault=sum(i < onset for i in r.alerts) if onset is not None else None,
                         status=D.signal_status(key, a)))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "signals.csv", index=False)
    shifts.to_csv(OUT / "detections.csv", index=False)

    lines = [f"# Live pipeline check: {provider} {model}", "",
             f"- Database: `{db}` · {len(calls):,} calls · {calls['timestamp'].min():%Y-%m-%d %H:%M} – "
             f"{calls['timestamp'].max():%H:%M} UTC",
             f"- Ground truth: {('fault ' + gt['fault'] + ' on ' + format(gt['fault_frac'], '.0%') + ' of calls from ' + gt['fault_start_utc']) if gt else 'none'}",
             "", "## Signals", "", summary.round(3).to_markdown(index=False), "", "## Incidents (incidents.py + mapper.py)", ""]
    for inc, plan in zip(a.incidents, a.plans):
        ev = D.incident_evidence(db, inc)
        lines += [f"- **{plan.urgency.upper()}** · {plan.headline}"]
        lines += [f"  - possible: {h}" for h in plan.possible_explanations]
        lines += [f"  - evidence ({m}): {e.summary}" for m, e in ev.items()] or ["  - evidence: none available"]
    if not a.incidents:
        lines.append("- none")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    pd.set_option("display.width", 220)
    print("\n" + summary.round(3).to_string(index=False))
    print("\nIncidents:")
    for inc, plan in zip(a.incidents, a.plans):
        print(f"  [{plan.urgency}] {plan.headline}")
        for h in plan.possible_explanations:
            print(f"      possible: {h}")
        for m, e in D.incident_evidence(db, inc).items():
            print(f"      evidence {m}: {e.summary}")
    print(f"\nWrote {OUT}/REPORT.md, signals.csv, detections.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end check of DriftGuard on live API calls")
    ap.add_argument("--provider", default="groq")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--healthy-runs", type=int, default=12)
    ap.add_argument("--fault-runs", type=int, default=8)
    ap.add_argument("--fault-frac", type=float, default=0.4)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--rpm", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args()
    if not args.analyse_only:
        collect(args.provider, args.model, args.db, args.healthy_runs, args.fault_runs,
                args.fault_frac, args.max_tokens, args.rpm, args.seed)
    analyse(args.db, args.provider, args.model)


if __name__ == "__main__":
    main()
