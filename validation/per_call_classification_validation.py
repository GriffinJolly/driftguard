"""
validation/per_call_classification_validation.py

Re-runs the classification validation through the per-call detection pipeline
(driftguard/detection/pipeline.py), so the result can be compared with the
per-run run in validation/results/classification_validation/ (8 of 13 correct).

Same 13 scenarios, same real model outputs, same classification layer
(rules.py / incidents.py / mapper.py, unchanged). The only difference is the
input the classification layer receives: one detection per change on per-call
streams, instead of per-run metric points.

  * every call is written to a fresh store as a log row (accuracy or refusal task)
  * pipeline.run_detection(store, ...) runs Page-Hinkley + PELT/permutation/BH and
    then remediation/mapper.apply_remediation, which labels and plans the events
  * the labelled, significant events in changepoint_events are scored:
      correct                 a confirmed (PELT) event with the expected label within
                              TOLERANCE of the change, and no labelled event earlier
      right label, wrong place  the expected label, but not placed at the change
      false label             any labelled event on a scenario that should have none
Tolerance = 3 runs' worth of calls, the same as the per-run validation
(24 calls for Groq, 75 for LMSYS / WildChat).

Needs driftguard.db (Groq trials) and the caches in validation/data/ built by
validation/lmsys_model_swap.py and validation/wildchat_version_switch.py.

Usage:  python -m validation.per_call_classification_validation
"""

from __future__ import annotations

import sqlite3
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlmodel import Session

from driftguard.classification import rules
from driftguard.detection import pipeline
from driftguard.storage.store import DriftStore, LogRow

DATA = Path("validation/data")
OUT = Path("validation/results/per_call_classification_validation")
GROQ_TOL, LMSYS_TOL = 3 * 8, 3 * 25


@dataclass
class Scenario:
    name: str
    kind: str                    # capability | safety | healthy | control
    expected: Optional[str]
    task: str                    # accuracy | refusal
    scores: np.ndarray           # per-call eval_score (1 = correct / 1 = refused)
    timestamps: list[datetime]
    change: Optional[int]        # first call after the change
    tol: int


def _times(n: int, start=datetime(2023, 8, 1, tzinfo=timezone.utc)) -> list[datetime]:
    return [start + timedelta(minutes=i) for i in range(n)]


def groq() -> list[Scenario]:
    con = sqlite3.connect("file:driftguard.db?mode=ro", uri=True)
    out = []
    for k, rid in enumerate(["live_drift_compare_sudden_1789368091", "live_drift_compare_sudden_1789444819"], 1):
        rows = con.execute("SELECT timestamp, eval_score FROM log_rows WHERE eval_run_id=? AND outcome='ok' "
                           "AND eval_score IS NOT NULL ORDER BY rowid", (rid,)).fetchall()
        ts = [pd.Timestamp(r[0]).tz_localize("UTC").to_pydatetime() for r in rows]
        out.append(Scenario(f"Groq live trial {k}", "capability", rules.CAPABILITY_SHIFT, "accuracy",
                            np.array([r[1] for r in rows], dtype=float), ts, 175, GROQ_TOL))
    con.close()
    return out


def lmsys(seed: int = 0) -> list[Scenario]:
    df = pd.read_parquet(DATA / "lmsys_english_first_turns.parquet")
    safe = df[~df["flagged"]]
    pairs = [
        ("vicuna-13b", "llama-2-13b-chat", "safety", rules.SAFETY_SHIFT),
        ("vicuna-13b", "llama-2-7b-chat", "safety", rules.SAFETY_SHIFT),
        ("dolly-v2-12b", "chatglm-6b", "safety", rules.SAFETY_SHIFT),
        ("alpaca-13b", "wizardlm-13b", "safety", rules.SAFETY_SHIFT),
        ("llama-2-13b-chat", "vicuna-13b", "healthy", None),
        ("chatglm-6b", "dolly-v2-12b", "healthy", None),
        ("vicuna-13b", "vicuna-13b", "control", None),
        ("koala-13b", "koala-13b", "control", None),
        ("vicuna-13b", "koala-13b", "control", None),
        ("gpt-4", "gpt-3.5-turbo", "control", None),
    ]
    rng = np.random.default_rng(seed)
    per_side = 50 * 25                                   # same sizes as the per-run validation
    out = []
    for a, b, kind, expected in pairs:
        ra = safe[safe["model"] == a]["refused"].to_numpy(dtype=float)
        rb = safe[safe["model"] == b]["refused"].to_numpy(dtype=float)
        if a == b:
            perm = rng.permutation(len(ra))
            first, second = ra[perm[:per_side]], ra[perm[per_side:2 * per_side]]
        else:
            first, second = ra[rng.permutation(len(ra))[:per_side]], rb[rng.permutation(len(rb))[:per_side]]
        s = np.r_[first, second]
        out.append(Scenario(f"LMSYS {a} -> {b}", kind, expected, "refusal", s, _times(len(s)), per_side, LMSYS_TOL))
    return out


def wildchat() -> list[Scenario]:
    path = DATA / "wildchat_gpt35_jun_aug_2023.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path).sort_values("timestamp")
    df = df[~df["flagged"]]
    key = pd.DataFrame({"d": df["timestamp"].dt.floor("D"), "p": df["prompt"].str.strip().str.lower().str[:200]})
    df = df[~key.duplicated()].reset_index(drop=True)
    switch = df.loc[df["model"] == "gpt-3.5-turbo-0613", "timestamp"].min()
    return [Scenario("WildChat gpt-3.5 0301 -> 0613", "healthy", None, "refusal",
                     df["refused"].to_numpy(dtype=float), [t.to_pydatetime() for t in df["timestamp"]],
                     int(df["timestamp"].searchsorted(switch)), LMSYS_TOL)]


def run(sc: Scenario, workdir: Path) -> pd.DataFrame:
    """Write the calls, run the pipeline, return the labelled open events with their call index."""
    store = DriftStore(db_path=str(workdir / f"{uuid.uuid4().hex}.db"))
    rows = [LogRow(log_id=uuid.uuid4().hex, timestamp=t, integration_path="eval_suite", provider="validation",
                   model_id="m", prompt="", outcome="ok", eval_run_id="v", eval_task_type=sc.task,
                   eval_task_id=f"{sc.task}_safe_{i}" if sc.task == "refusal" else f"q{i}", eval_score=float(v))
            for i, (t, v) in enumerate(zip(sc.timestamps, sc.scores))]
    with Session(store.engine) as s:
        s.add_all(rows)
        s.commit()
    pipeline.run_detection(store, "validation", "m")
    ev = pd.read_sql_query("SELECT * FROM changepoint_events WHERE significant = 1 AND drift_type IS NOT NULL",
                           store.engine)
    store.engine.dispose()
    ts = pd.to_datetime(pd.Series(sc.timestamps), utc=True)
    when = pd.to_datetime(ev["changepoint_timestamp"], utc=True, format="mixed")
    ev["call"] = [int(ts.searchsorted(t)) for t in when]
    return ev


def score(sc: Scenario, ev: pd.DataFrame) -> tuple[str, str]:
    desc = "; ".join(f"{r.drift_type} ({r.detector}) at call {r.call}" for r in ev.itertuples()) or "no label"
    if sc.expected is None:
        return ("correct", desc) if ev.empty else ("false label", desc)
    early = ev[ev["call"] < sc.change - sc.tol]
    placed = ev[(ev["detector"] == "pelt") & (ev["drift_type"] == sc.expected)
                & ((ev["call"] - sc.change).abs() <= sc.tol)]
    if len(placed) and early.empty:
        return "correct", desc
    if len(ev) and (ev["drift_type"] == sc.expected).all():
        return "right label, wrong place", desc
    if len(ev):
        return "wrong label", desc
    return "missed", desc


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenarios = groq() + lmsys() + wildchat()
    rows = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        for sc in scenarios:
            ev = run(sc, Path(tmp))
            verdict, why = score(sc, ev)
            rows.append(dict(scenario=sc.name, kind=sc.kind, expected=sc.expected or "no label", calls=len(sc.scores),
                             change_call=sc.change, verdict=verdict, detail=why))
            print(f"{verdict:>24}  {sc.name:<40} change at {sc.change}: {why}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "results.csv", index=False)
    summary = t.groupby("kind")["verdict"].value_counts().unstack(fill_value=0)
    n_ok = int((t["verdict"] == "correct").sum())
    lines = ["# Classification validation through the per-call detection pipeline", "",
             f"**{n_ok} of {len(t)} scenarios correct** (per-run pipeline on the same scenarios: 8 of 13).", "",
             "Same scenarios and the same classification layer as validation/results/classification_validation/;",
             "detection now runs per call (driftguard/detection/pipeline.py). Correct = a confirmed (PELT) event",
             "with the expected label within 3 runs' worth of calls of the change and no labelled event earlier;",
             "no labelled event at all where none is expected.", "",
             "## By scenario type", "", summary.to_markdown(), "", "## Every scenario", "",
             t.to_markdown(index=False), ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{n_ok}/{len(t)} correct\n{summary}")


if __name__ == "__main__":
    main()
