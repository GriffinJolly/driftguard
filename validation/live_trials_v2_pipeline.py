"""
validation/live_trials_v2_pipeline.py

The second round of live trials on NVIDIA's API, scored through the deployed
pipeline (per-call signals, noise-scaled Page-Hinkley, PELT with the permutation
test and Benjamini-Hochberg, classification rules). Compared with round one
(nvidia_trials_pipeline.py), the trials are longer and use the 100-question suite:

  long  (validation/data/nvidia_long_<n>.db)
      Nemotron-3-super-120b. After 175 normal calls, 15% of calls go out with a
      16-token budget, so the model is cut off before it answers. 700 calls after
      the fault; gradual trials ramp the faulty share up over 100 calls.

  swap  (validation/data/nvidia_swap_<n>.db)
      A silent model swap. Nemotron-3-super-120b answers the first 175 calls; then
      Nemotron-3.5-lightning-30b answers instead, under the same model name, at
      once (sudden) or with a share rising over 300 calls (gradual). Every reply is
      a genuine response; nothing is cut short. 700 calls after the swap.

Correct = a confirmed capability_shift inside the change window (the fault or swap
start, or the ramp for gradual trials, widened by 24 calls on each side) and no
label before it.

Usage:
    python -m validation.live_trials_v2_pipeline
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from driftguard.classification import rules
from driftguard.detection import pipeline
from driftguard.storage.store import DriftStore

DATA = Path("validation/data")
OUT = Path("validation/results/live_trials_v2")
PROVIDER, MODEL = "nvidia", "nvidia/nemotron-3-super-120b-a12b"
ONSET, TOL = 175, 24
TRIALS = [
    ("long", 1, "gradual", 100), ("long", 2, "sudden", 0), ("long", 3, "gradual", 100),
    ("swap", 1, "sudden", 0), ("swap", 2, "gradual", 300),
]


def served_share_after(db: Path) -> float:
    """Swap trials: share of scored calls after the onset that the replacement model answered."""
    con = sqlite3.connect(db)
    rows = con.execute("SELECT request_params_json FROM log_rows WHERE outcome = 'ok' "
                       "AND eval_score IS NOT NULL ORDER BY rowid").fetchall()
    con.close()
    after = rows[ONSET:]
    return sum('"served_by"' in (r[0] or "") for r in after) / max(1, len(after))


def score(kind: str, n: int, scen: str, ramp: int) -> dict:
    db = DATA / f"nvidia_{kind}_{n}.db"
    store = DriftStore(db_path=str(db))
    calls = pipeline.load_calls(store, PROVIDER, MODEL)
    store.engine.dispose()
    det = pipeline.detect(calls, PROVIDER, MODEL)
    v = det.results["accuracy_error"].stream.values
    labels = [(d.event.detector, d.index, c.drift_type) for d in det.events if d.signal == "accuracy_error"
              for c in [rules.classify_event(d.event)] if c.drift_type]
    pelt = [i for det_, i, t in labels if det_ == "pelt" and t == rules.CAPABILITY_SHIFT]
    live = [i for det_, i, t in labels if det_ == "sequential"]
    hi = ONSET + ramp + TOL
    placed = [i for i in pelt if ONSET - TOL <= i <= hi]
    early = [i for _, i, _ in labels if i < ONSET - TOL]
    other = sorted({f"{c.drift_type} on {d.signal}" for d in det.events for c in [rules.classify_event(d.event)]
                    if c.drift_type and d.signal != "accuracy_error"})
    return {
        "trial": f"{kind} {n}", "change": "silent swap" if kind == "swap" else "15% cut short", "kind": scen,
        "calls": len(v), "error_before": round(float(v[:ONSET].mean()), 3), "error_after": round(float(v[ONSET:].mean()), 3),
        "replacement_share_after": round(served_share_after(db), 2) if kind == "swap" else None,
        "confirmed_at": placed[0] if placed else (pelt[0] if pelt else None),
        "live_alarm_at": next((i for i in live if i >= ONSET), None),
        "live_alarms_before": sum(i < ONSET for i in live),
        "verdict": "correct" if placed and not early else ("label, wrong place" if pelt else "missed"),
        "other_labels": "; ".join(other) or "none",
    }


def main() -> None:
    t = pd.DataFrame([score(*x) for x in TRIALS if (DATA / f"nvidia_{x[0]}_{x[1]}.db").exists()])
    t["live_delay"] = t["live_alarm_at"] - ONSET
    t["confirm_offset"] = t["confirmed_at"] - ONSET
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "results.csv", index=False)
    lines = ["# Live trials, round two: longer streams and a silent model swap", "",
             "NVIDIA's API, 100-question MMLU suite, 175 normal calls then 700 after the change. Deployed settings: "
             "noise-scaled Page-Hinkley (alpha 0.998), PELT, permutation test, Benjamini-Hochberg, classification rules. "
             f"Correct = confirmed capability_shift inside the change window (± {TOL} calls; the ramp for gradual trials) "
             "and no earlier label.", "",
             "- **long**: Nemotron-3-super-120b, 15% of calls cut to a 16-token budget after call 175.",
             "- **swap**: Nemotron-3-super-120b replaced by Nemotron-3.5-lightning-30b under the same name after call 175, "
             "at once or over 300 calls. Genuine replies; on 50 questions the two models scored 90% and 80%.", "",
             f"**{int((t['verdict'] == 'correct').sum())} of {len(t)} correct.**", "",
             t.to_markdown(index=False), ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    pd.set_option("display.width", 240)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
