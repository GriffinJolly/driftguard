"""
validation/fault_injection_check.py

Controlled faults for the two drift categories with no real-data test yet:
format_drift and infra_change (step 4 of the completion plan). Each fault runs
through the production path -- per-call signals, the live Page-Hinkley monitor,
the retrospective PELT path, and the classification layer -- and has a control
with no fault. Pass = the right label near the fault, and no label at all on the
control.

FORMAT (live API calls)
  The real format tasks (driftguard/evalsuite/format_check.py) run against a live
  model. During the fault phase, a share of calls is sent as the same request in
  plain English WITHOUT the "Return ONLY a JSON ..." instruction -- standing in for
  a provider update that stops honouring format instructions. Every reply is a
  real model response, scored against the same schema as always. The fault
  changes the request, as the earlier truncation trials did; say so when
  reporting it.

INFRA (replay, no API calls)
  A provider cannot be made slower on demand, so this part replays REAL response
  latencies of gpt-oss-120b from driftguard.db, shuffled so the baseline is
  stable, and adds a fixed delay to a share of calls after the fault point. The
  delay is injected on our side; report it as a simulated infrastructure slowdown.

Usage:
    python -m validation.fault_injection_check --part infra
    python -m validation.fault_injection_check --part format      # ~480 Groq calls, ~35 min
    python -m validation.fault_injection_check --part report      # score both from their databases
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sqlmodel import Session

from driftguard.classification import rules
from driftguard.detection import pipeline
from driftguard.storage.store import DriftStore, LogRow

DATA = Path("validation/data")
OUT = Path("validation/results/fault_injection")
PROVIDER, MODEL = "groq", "openai/gpt-oss-120b"

# the same requests without the JSON instruction
PLAIN_PROMPTS = {
    "format_flat_object": 'Describe a person named "Ada Lovelace" who is 28 years old and works as an "engineer".',
    "format_array_of_strings": "Name 3 primary colors.",
    "format_nested_object": "Describe a plausible book: its title, its author, and the year and publisher of its publication.",
    "format_enum_field": 'Pick one status out of "active", "inactive" and "pending".',
    "format_array_of_objects": "Name 3 fruits and the color of each.",
    "format_boolean_nullable": "Describe a task that is not completed and has no due date.",
}
FORMAT_HEALTHY_RUNS, FORMAT_FAULT_RUNS, FORMAT_FAULT_FRAC = 20, 20, 0.5      # 6 calls per run
INFRA_N, INFRA_FAULT_AT, INFRA_FAULT_FRAC, INFRA_DELAY_MS = 1200, 600, 0.4, 3000.0
TOLERANCE = {"format": 24, "infra": 75}                                       # calls either side


# --------------------------------------------------------------------------- format: live

def run_format(db: Path, fault: bool, seed: int, rpm: int = 14) -> dict:
    """Run the real format tasks live, feeding the live monitor as in the runner, then the
    retrospective path. With fault=False the whole stream is the control."""
    from driftguard.detection.live import LiveMonitor
    from driftguard.evalsuite import format_check
    from driftguard.providers import clients

    clients.PROVIDERS[PROVIDER].rate_limiter.max_calls = rpm
    if db.exists():
        db.unlink()
    store = DriftStore(db_path=str(db))
    monitor = LiveMonitor(store, PROVIDER, MODEL)
    rng = random.Random(seed)
    state = {"fault": False}
    real_call = clients.call_model

    def call(*args, **kw):
        kw["request_params"] = {"temperature": 0, "max_tokens": 1536}
        if state["fault"] and rng.random() < FORMAT_FAULT_FRAC:
            # the stored prompt itself shows the fault; nothing extra goes in the request,
            # since every request_params key is sent to the provider
            kw["prompt"] = PLAIN_PROMPTS[kw["eval_task_id"]]
        return real_call(*args, **kw)

    format_check.call_model = call
    fault_call = fault_time = None
    calls = 0
    for i in range(FORMAT_HEALTHY_RUNS + FORMAT_FAULT_RUNS):
        if fault and i == FORMAT_HEALTHY_RUNS:
            state["fault"], fault_call, fault_time = True, calls, datetime.now(timezone.utc).isoformat()
        entries = format_check.run_format_tasks(PROVIDER, MODEL, f"fault-format-{uuid.uuid4().hex[:8]}",
                                                on_entry=monitor.observe)
        for e in entries:
            store.write_log(e)
        calls += len(entries)
        ok = [e for e in entries if e.eval_score is not None]
        print(f"  run {i + 1:>2}: {'FAULT' if state['fault'] else 'healthy'}  "
              f"passed {sum(e.eval_score for e in ok):.0f}/{len(ok)}", flush=True)
    format_check.call_model = real_call
    report = pipeline.run_detection(store, PROVIDER, MODEL)
    truth = {"part": "format", "fault": fault, "fault_at_call": fault_call, "fault_time_utc": fault_time,
             "calls": calls, "live_alarms": len(monitor.alarms), "written": report.written}
    db.with_suffix(".ground_truth.json").write_text(json.dumps(truth, indent=2))
    return truth


# --------------------------------------------------------------------------- infra: replay

def run_infra(db: Path, fault: bool, seed: int) -> dict:
    """Replay real gpt-oss-120b latencies (shuffled), with a delay injected after the fault point."""
    import sqlite3
    con = sqlite3.connect("file:driftguard.db?mode=ro", uri=True)
    lat = np.array([r[0] for r in con.execute(
        "SELECT latency_ms FROM log_rows WHERE provider='groq' AND model_id=? AND outcome='ok' "
        "AND latency_ms IS NOT NULL", (MODEL,))], dtype=float)
    con.close()
    rng = np.random.default_rng(seed)
    sample = lat[rng.permutation(len(lat))[:INFRA_N]]
    if fault:
        hit = (np.arange(INFRA_N) >= INFRA_FAULT_AT) & (rng.random(INFRA_N) < INFRA_FAULT_FRAC)
        sample = np.where(hit, sample + INFRA_DELAY_MS, sample)
    if db.exists():
        db.unlink()
    store = DriftStore(db_path=str(db))
    t0 = datetime(2026, 10, 1, tzinfo=timezone.utc)
    rows = [LogRow(log_id=uuid.uuid4().hex, timestamp=t0 + timedelta(minutes=i), integration_path="eval_suite",
                   provider=PROVIDER, model_id=MODEL, prompt="", outcome="ok", latency_ms=float(v),
                   eval_run_id="infra-replay", eval_task_type="accuracy", eval_task_id=f"q{i}", eval_score=1.0)
            for i, v in enumerate(sample)]
    with Session(store.engine) as s:
        s.add_all(rows)
        s.commit()
    report = pipeline.run_detection(store, PROVIDER, MODEL)
    truth = {"part": "infra", "fault": fault, "fault_at_call": INFRA_FAULT_AT if fault else None,
             "calls": INFRA_N, "delay_ms": INFRA_DELAY_MS, "fault_frac": INFRA_FAULT_FRAC,
             "latency_pool": int(len(lat)), "written": report.written}
    db.with_suffix(".ground_truth.json").write_text(json.dumps(truth, indent=2))
    store.engine.dispose()
    return truth


# --------------------------------------------------------------------------- scoring

def score(db: Path, part: str, expected: str) -> dict:
    truth = json.loads(db.with_suffix(".ground_truth.json").read_text())
    store = DriftStore(db_path=str(db))
    calls = pipeline.load_calls(store, PROVIDER, MODEL)
    ev = pd.read_sql_query("SELECT * FROM changepoint_events WHERE significant = 1 AND drift_type IS NOT NULL",
                           store.engine)
    store.engine.dispose()
    signal = "format_error" if part == "format" else "slow_call"
    stream = pipeline.build_streams(calls).get(signal)
    if len(ev) and stream is not None:
        ts = pd.to_datetime(stream.timestamp, utc=True)
        ev["call"] = [int(ts.searchsorted(t)) for t in pd.to_datetime(ev["changepoint_timestamp"], utc=True, format="mixed")]
    else:
        ev["call"] = pd.Series(dtype=int)
    labels = "; ".join(f"{r.drift_type} ({r.detector}) at call {r.call}" for r in ev.itertuples()) or "no label"
    if not truth["fault"]:
        verdict = "correct" if ev.empty else "false label"
    else:
        # position of the fault within the signal's own stream
        if part == "format":
            fault_at = truth["fault_at_call"] * len(stream.values) // truth["calls"]
        else:
            fault_at = truth["fault_at_call"]
        tol = TOLERANCE[part]
        placed = ev[(ev["detector"] == "pelt") & (ev["drift_type"] == expected) & ((ev["call"] - fault_at).abs() <= tol)]
        early = ev[ev["call"] < fault_at - tol]
        verdict = ("correct" if len(placed) and early.empty else
                   "right label, wrong place" if len(ev) and (ev["drift_type"] == expected).all() else
                   "wrong label" if len(ev) else "missed")
    before = after = None
    if stream is not None and truth["fault"]:
        cut = fault_at
        before, after = float(stream.values[:cut].mean()), float(stream.values[cut:].mean())
    return {"part": part, "stream": "fault" if truth["fault"] else "control", "expected": expected if truth["fault"] else "no label",
            "calls": truth["calls"], "bad_rate_before": before, "bad_rate_after": after,
            "verdict": verdict, "labels": labels}


def report() -> None:
    rows = []
    for part, expected in (("format", rules.FORMAT_DRIFT), ("infra", rules.INFRA_CHANGE)):
        for kind in ("fault", "control"):
            db = DATA / f"fault_{part}_{kind}.db"
            if db.exists():
                rows.append(score(db, part, expected))
    t = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "results.csv", index=False)
    lines = ["# Controlled faults: format drift and infra change", "",
             f"**{int((t['verdict'] == 'correct').sum())} of {len(t)} correct.** Production path: per-call signals, "
             "live Page-Hinkley during the run (format), retrospective PELT → permutation → BH, classification "
             "rules / incidents / mapper. Correct = a confirmed change with the expected label near the fault and "
             "nothing earlier; on a control, no label at all.", "",
             "- **Format** (live gpt-oss-120b on Groq): during the fault phase half the format requests were sent "
             "without the \"Return ONLY a JSON ...\" instruction. Real replies, same schema check.",
             f"- **Infra** (replay): real gpt-oss-120b latencies from driftguard.db, shuffled; after call "
             f"{INFRA_FAULT_AT}, {INFRA_FAULT_FRAC:.0%} of calls get +{INFRA_DELAY_MS / 1000:.0f} s injected on our side.",
             "", t.to_markdown(index=False), ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    pd.set_option("display.width", 220)
    print(t.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="Controlled format and infra faults")
    ap.add_argument("--part", choices=["format", "infra", "report"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", choices=["fault", "control"], help="format: run just one of the two streams")
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if args.part == "infra":
        print(run_infra(DATA / "fault_infra_fault.db", True, args.seed))
        print(run_infra(DATA / "fault_infra_control.db", False, args.seed + 1))
    elif args.part == "format":
        if args.only != "control":
            print("Format fault stream:")
            print(run_format(DATA / "fault_format_fault.db", True, args.seed))
        if args.only != "fault":
            print("Format control stream:")
            print(run_format(DATA / "fault_format_control.db", False, args.seed + 1))
    report()


if __name__ == "__main__":
    main()
