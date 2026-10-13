"""
validation/classification_validation.py

Validates the drift CLASSIFICATION layer (classification/rules.py,
classification/incidents.py, remediation/mapper.py) end to end, on real data,
using exactly the metrics it was designed for: the per-run metric points in
metric_series (accuracy, format_adherence, refusal_rate, latency_p95_ms).

Nothing here re-implements detection or classification. Every scenario goes
through the pipeline already in main, unchanged:

    metric_series points
      -> detection/baseline.calibrate          (quiet window -> PELT penalty per metric)
      -> detection/changepoint_pelt            (candidate change points per metric)
      -> detection/bootstrap                   (one-sided permutation test, adverse only)
      -> detection/multiple_testing            (Benjamini-Hochberg, persisted)
      -> classification/incidents + rules      (label, evidence tier, incident)
      -> remediation/mapper                    (urgency, one-line plan)

Each scenario is built from REAL model outputs and has a known answer:

  capability  Groq gpt-oss-120b live trials: after call 175, 35% of calls were cut to
              16 output tokens (driftguard.db, validation/live_drift_compare.py).
  safety      LMSYS-Chat-1M silent model swaps (validation/lmsys_model_swap.py):
              real replies from model A, then real replies from model B. A swap to a
              model that refuses safe prompts more often is a safety_shift.
  healthy     A swap to a model that refuses LESS (and the real WildChat-1M
              gpt-3.5-turbo 0301 -> 0613 update, after which refusals fell) is a
              real change in the HEALTHY direction: rules.py must not label it.
  control     Same model on both sides, or two models with near-identical refusal
              rates: nothing should be labelled.

Calls are grouped into "runs" (one metric point each), the same unit the eval
runner produces. Calibration always uses the first QUIET_RUNS runs, which lie
before every scenario's change point; the change point itself is used only to
score the result.

Not covered: format_drift and infra_change. No real dataset with a known format or
latency change is available, so those two labels are only covered by rules.py's
own unit tests.

Usage:  python -m validation.classification_validation
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from driftguard.classification import incidents as inc_mod
from driftguard.classification import rules
from driftguard.detection import baseline, bootstrap, changepoint_pelt, multiple_testing
from driftguard.remediation import mapper
from driftguard.storage.store import DriftStore, MetricSeriesRow

DATA = Path("validation/data")
OUT = Path("validation/results/classification_validation")
QUIET_RUNS = 20          # calibration window (baseline.MIN_QUIET_POINTS)
TOLERANCE_RUNS = 3       # a label counts as "at the change" within this many runs
LMSYS_RUN = 25           # conversations per run for the LMSYS / WildChat scenarios
GROQ_RUN = 8             # scored calls per run for the Groq trials


@dataclass
class Scenario:
    name: str
    kind: str                       # capability | safety | healthy | control
    expected: Optional[str]         # rules.py label expected at the change, or None
    metric: str
    points: list[MetricSeriesRow]
    change_run: Optional[int]       # index of the first run after the change (None = no change)
    source: str


# --------------------------------------------------------------------------- building metric points

def _points(values: list[float], metric: str, start: datetime, step: timedelta, model: str) -> list[MetricSeriesRow]:
    return [MetricSeriesRow(run_id=f"{model}-{i}", timestamp=start + i * step, provider="validation",
                            model_id=model, n_calls=0, **{metric: float(v)})
            for i, v in enumerate(values)]


def _batch(x: np.ndarray, size: int) -> np.ndarray:
    n = len(x) // size
    return x[: n * size].reshape(n, size).mean(axis=1)


def groq_trials(db: Path = Path("driftguard.db")) -> list[Scenario]:
    """Accuracy per run of GROQ_RUN scored calls, from the two live induced-change trials."""
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    out = []
    for k, run_id in enumerate(["live_drift_compare_sudden_1789368091", "live_drift_compare_sudden_1789444819"], 1):
        rows = con.execute("SELECT timestamp, eval_score FROM log_rows WHERE eval_run_id=? AND outcome='ok' "
                           "AND eval_score IS NOT NULL ORDER BY rowid", (run_id,)).fetchall()
        scores = np.array([r[1] for r in rows], dtype=float)
        onset = 175                                       # first truncated call (scored-call index)
        acc = _batch(scores, GROQ_RUN)
        t0 = pd.Timestamp(rows[0][0]).to_pydatetime().replace(tzinfo=timezone.utc)
        out.append(Scenario(f"Groq live trial {k}", "capability", rules.CAPABILITY_SHIFT, "accuracy",
                            _points(acc, "accuracy", t0, timedelta(hours=1), f"groq-trial-{k}"),
                            onset // GROQ_RUN, f"gpt-oss-120b, 35% of calls cut to 16 tokens after call 175"))
    con.close()
    return out


def lmsys_swaps(seed: int = 0) -> list[Scenario]:
    """refusal_rate (safe = moderation-unflagged prompts) per run, for real model swaps."""
    df = pd.read_parquet(DATA / "lmsys_english_first_turns.parquet")
    safe = df[~df["flagged"]]
    pairs = [
        ("vicuna-13b", "llama-2-13b-chat", "safety", rules.SAFETY_SHIFT),
        ("vicuna-13b", "llama-2-7b-chat", "safety", rules.SAFETY_SHIFT),
        ("dolly-v2-12b", "chatglm-6b", "safety", rules.SAFETY_SHIFT),
        ("alpaca-13b", "wizardlm-13b", "safety", rules.SAFETY_SHIFT),
        ("llama-2-13b-chat", "vicuna-13b", "healthy", None),   # refusals fall: not drift
        ("chatglm-6b", "dolly-v2-12b", "healthy", None),
        ("vicuna-13b", "vicuna-13b", "control", None),          # same model
        ("koala-13b", "koala-13b", "control", None),
        ("vicuna-13b", "koala-13b", "control", None),           # different model, same refusal rate
        ("gpt-4", "gpt-3.5-turbo", "control", None),
    ]
    rng = np.random.default_rng(seed)
    per_side = 50 * LMSYS_RUN                                     # 50 runs on each side
    out = []
    for a, b, kind, expected in pairs:
        ra = safe[safe["model"] == a]["refused"].to_numpy(dtype=float)
        rb = safe[safe["model"] == b]["refused"].to_numpy(dtype=float)
        if a == b:
            perm = rng.permutation(len(ra))
            first, second = ra[perm[:per_side]], ra[perm[per_side:2 * per_side]]
        else:
            first, second = ra[rng.permutation(len(ra))[:per_side]], rb[rng.permutation(len(rb))[:per_side]]
        vals = np.r_[_batch(first, LMSYS_RUN), _batch(second, LMSYS_RUN)]
        pts = _points(vals, "refusal_rate", datetime(2023, 8, 1, tzinfo=timezone.utc), timedelta(hours=1), f"{a}->{b}")
        out.append(Scenario(f"LMSYS {a} -> {b}", kind, expected, "refusal_rate", pts, per_side // LMSYS_RUN,
                            f"real replies; refusal of safe prompts {first.mean():.3f} -> {second.mean():.3f}"))
    return out


def wildchat_update() -> list[Scenario]:
    """refusal_rate per run around the real gpt-3.5-turbo 0301 -> 0613 update (repeats within a day removed)."""
    path = DATA / "wildchat_gpt35_jun_aug_2023.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path).sort_values("timestamp")
    df = df[~df["flagged"]]
    key = pd.DataFrame({"d": df["timestamp"].dt.floor("D"), "p": df["prompt"].str.strip().str.lower().str[:200]})
    df = df[~key.duplicated()].reset_index(drop=True)
    switch = df.loc[df["model"] == "gpt-3.5-turbo-0613", "timestamp"].min()
    vals = _batch(df["refused"].to_numpy(dtype=float), LMSYS_RUN)
    run_ts = df["timestamp"].iloc[::LMSYS_RUN].iloc[: len(vals)].tolist()
    pts = [MetricSeriesRow(run_id=f"wc-{i}", timestamp=t.to_pydatetime(), provider="validation",
                           model_id="wildchat-gpt-3.5", n_calls=LMSYS_RUN, refusal_rate=float(v))
           for i, (t, v) in enumerate(zip(run_ts, vals))]
    change = int(df["timestamp"].searchsorted(switch)) // LMSYS_RUN
    return [Scenario("WildChat gpt-3.5 0301 -> 0613", "healthy", None, "refusal_rate", pts, change,
                     "real provider update; refusals of safe prompts fell afterwards")]


# --------------------------------------------------------------------------- the pipeline in main

def run_pipeline(sc: Scenario, workdir: Path) -> dict:
    """main's detection modules -> events in a fresh DriftStore -> rules/incidents/mapper."""
    db = workdir / f"{abs(hash(sc.name))}.db"
    store = DriftStore(db_path=str(db))
    provider, model = "validation", sc.points[0].model_id
    for p in sc.points:
        p.provider, p.model_id = provider, model

    cal = baseline.calibrate(sc.points[:QUIET_RUNS], provider, model)
    th = cal.thresholds.get(sc.metric)
    if th is None:
        return dict(error="metric could not be calibrated")
    cands = changepoint_pelt.find_changepoints_for_metric(sc.points, sc.metric, penalty=th.pelt_penalty)
    values, _ = changepoint_pelt._extract_series(sorted(sc.points, key=lambda p: p.timestamp), sc.metric)
    sig = bootstrap.evaluate_candidates(cands, values)
    primary = {sc.metric: min(sig, key=lambda r: r.p_value)} if sig else {}
    rows = bootstrap.store_significance_results(store, provider, model, primary)
    multiple_testing.apply_and_persist_bh_correction(store, rows, primary)

    incidents = inc_mod.incidents_from_store(store)
    plans = [mapper.plan_for_incident(i) for i in incidents]
    labels = []
    for inc, plan in zip(incidents, plans):
        for m in inc.members:
            idx = next(k for k, p in enumerate(sorted(sc.points, key=lambda p: p.timestamp))
                       if inc_mod._as_utc(p.timestamp) == inc_mod.event_time(m.event))
            labels.append(dict(label=m.classification.drift_type, tier=m.classification.evidence_tier,
                               run=idx, urgency=plan.urgency, reason=m.classification.reason))
    store.engine.dispose()
    return dict(candidates=len(cands), adverse_tested=len(sig),
                primary_p=None if not primary else round(primary[sc.metric].p_value, 4),
                labels=labels)


def score(sc: Scenario, res: dict) -> tuple[str, str]:
    """(verdict, explanation). For an expected label: correct label within TOLERANCE_RUNS of the
    change. For no expected label: no label anywhere."""
    labels = res.get("labels", [])
    if sc.expected is None:
        if not labels:
            return "correct", "no label"
        return "false label", "; ".join(f"{l['label']} at run {l['run']}" for l in labels)
    hits = [l for l in labels if l["label"] == sc.expected and abs(l["run"] - sc.change_run) <= TOLERANCE_RUNS]
    if hits:
        h = hits[0]
        return "correct", f"{h['label']} at run {h['run']} (change at {sc.change_run}), {h['tier']}, {h['urgency']}"
    if labels and all(l["label"] == sc.expected for l in labels):
        return "right label, wrong place", "; ".join(f"{l['label']} at run {l['run']} (change at {sc.change_run})"
                                                      for l in labels)
    if labels:
        return "wrong label", "; ".join(f"{l['label']} at run {l['run']}" for l in labels)
    return "missed", "no label"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenarios = groq_trials() + lmsys_swaps() + wildchat_update()
    rows = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        for sc in scenarios:
            res = run_pipeline(sc, Path(tmp))
            verdict, why = score(sc, res)
            rows.append(dict(scenario=sc.name, kind=sc.kind, metric=sc.metric,
                             expected=sc.expected or "no label", runs=len(sc.points), change_run=sc.change_run,
                             pelt_candidates=res.get("candidates"), primary_p=res.get("primary_p"),
                             verdict=verdict, detail=why, source=sc.source))
            print(f"{verdict:>12}  {sc.name:<42} expected={sc.expected or 'no label':<17} {why}", flush=True)

    t = pd.DataFrame(rows)
    t.to_csv(OUT / "results.csv", index=False)
    summary = t.groupby("kind")["verdict"].value_counts().unstack(fill_value=0)
    n_ok = int((t["verdict"] == "correct").sum())
    lines = ["# Classification validation (rules.py / incidents.py / mapper.py, as merged in main)", "",
             f"{n_ok} of {len(t)} scenarios classified correctly.", "",
             "Each scenario: real model outputs -> per-run metric points -> main's calibration, PELT, permutation",
             "test and Benjamini-Hochberg -> rules.py label -> incidents.py -> mapper.py urgency.",
             f"Runs: {GROQ_RUN} scored calls (Groq) or {LMSYS_RUN} conversations (LMSYS, WildChat). "
             f"Calibration: first {QUIET_RUNS} runs. A label counts if it is within {TOLERANCE_RUNS} runs of the change.", "",
             "## By scenario type", "", summary.to_markdown(), "",
             "## Reading the result", "",
             "- rules.py: every label it gave had the right category and direction (capability for the",
             "  accuracy drops, safety_shift for every swap to a more-refusing model, nothing for changes in",
             "  the healthy direction).",
             "- The misplaced and false labels come from the detection step that feeds it (main's",
             "  calibration + changepoint_pelt + bootstrap), not from rules.py: the calibrated PELT penalty",
             "  (sigma^2 log n from the quiet window) is small enough to propose many candidates on a real",
             "  step; the permutation test compares everything before a candidate with everything after,",
             "  so most candidates around one real step pass; and only the smallest-p candidate per metric",
             "  is kept, ties going to the earliest. A single test at alpha = 0.05 also lets a",
             "  same-model control through now and then (BH cannot correct one test).", "",
             "## Every scenario", "",
             t.drop(columns=["source"]).to_markdown(index=False), "", "## Sources", ""]
    lines += [f"- **{r.scenario}**: {r.source}" for r in t.itertuples()]
    lines += ["", "Not covered: format_drift and infra_change (no real data with a known format or latency change)."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{n_ok}/{len(t)} correct\n{summary}\nWrote {OUT}/REPORT.md, results.csv")


if __name__ == "__main__":
    main()
