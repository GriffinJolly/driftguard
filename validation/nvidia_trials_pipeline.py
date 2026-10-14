"""
validation/nvidia_trials_pipeline.py

The five live NVIDIA trials (gpt-oss-20b, 20-question MMLU suite) through the
deployed pipeline: per-call signals, noise-scaled Page-Hinkley, PELT with the
permutation test and Benjamini-Hochberg, and the classification rules.

Each trial was collected by validation/live_drift_compare.py into its own
database (validation/data/nvidia_trial_<n>.db): 175 normal calls, then 175 calls
of which up to 35% go out with a 16-token output budget, so the model is cut off
before it gives a letter. Sudden trials start the fault at call 176; gradual
ones ramp the faulty share up over 100 calls.

Correct = a confirmed capability_shift inside the change window, and no label
before it. The window is the fault start (sudden) or the 100-call ramp (gradual),
widened by 24 calls on each side, the tolerance used for the Groq trials in
per_call_classification_validation.py.

Usage:
    python -m validation.nvidia_trials_pipeline
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from driftguard.classification import rules
from driftguard.detection import pipeline
from driftguard.storage.store import DriftStore

DATA = Path("validation/data")
OUT = Path("validation/results/live_drift_compare_nvidia")
PROVIDER, MODEL = "nvidia", "openai/gpt-oss-20b"
ONSET, TOL, RAMP = 175, 24, 100
KINDS = {1: "gradual", 2: "sudden", 3: "gradual", 4: "sudden", 5: "gradual"}


def score(n: int, kind: str) -> dict:
    store = DriftStore(db_path=str(DATA / f"nvidia_trial_{n}.db"))
    calls = pipeline.load_calls(store, PROVIDER, MODEL)
    store.engine.dispose()
    det = pipeline.detect(calls, PROVIDER, MODEL)
    r = det.results["accuracy_error"]
    v = r.stream.values
    labels = []
    for d in det.events:
        if d.signal != "accuracy_error":
            continue
        c = rules.classify_event(d.event)
        if c.drift_type:
            labels.append((d.event.detector, d.index, c.drift_type))
    pelt = [i for det_, i, t in labels if det_ == "pelt" and t == rules.CAPABILITY_SHIFT]
    live = [i for det_, i, t in labels if det_ == "sequential"]
    hi = ONSET + (RAMP if kind == "gradual" else 0) + TOL
    placed = [i for i in pelt if ONSET - TOL <= i <= hi]
    early = [i for _, i, _ in labels if i < ONSET - TOL]
    other = sorted({(k, d.signal) for d in det.events for k in [rules.classify_event(d.event).drift_type]
                    if k and d.signal != "accuracy_error"})
    return {
        "trial": n, "kind": kind, "calls": len(v),
        "error_before": round(float(v[:ONSET].mean()), 3), "error_after": round(float(v[ONSET:].mean()), 3),
        "confirmed_at": placed[0] if placed else (pelt[0] if pelt else None),
        "live_alarm_at": next((i for i in live if i >= ONSET), None),
        "live_alarms_before_fault": sum(i < ONSET for i in live),
        "verdict": "correct" if placed and not early else ("label, wrong place" if pelt else "missed"),
        "other_labels": "; ".join(f"{k} on {s}" for k, s in other) or "none",
    }


def main() -> None:
    t = pd.DataFrame([score(n, k) for n, k in KINDS.items()])
    t["live_delay"] = t["live_alarm_at"] - ONSET
    t["confirm_offset"] = t["confirmed_at"] - ONSET
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "pipeline_results.csv", index=False)
    lines = ["# Live NVIDIA trials through the deployed pipeline", "",
             f"gpt-oss-20b on NVIDIA's API, 20-question MMLU suite, 350 scored calls per trial, fault from call {ONSET + 1}. "
             "Deployed settings: noise-scaled Page-Hinkley (alpha 0.998), PELT, permutation test, Benjamini-Hochberg, "
             f"classification rules. Correct = confirmed capability_shift inside the change window (fault start, or the {RAMP}-call ramp for gradual, ± {TOL} calls) and no earlier label.",
             "", f"**{int((t['verdict'] == 'correct').sum())} of {len(t)} correct.**", "",
             t.to_markdown(index=False), "",
             "## Notes", "",
             "- The fixed-threshold Page-Hinkley used in the detector benchmark (`nvidia_raw_results.csv`) caught all "
             "five faults, 38-169 calls after they began, with no earlier alarm. The deployed detector is scaled to "
             "signal noise (results/ph_noise_scaling/) and is less sensitive on short, noisy streams.",
             "- PELT confirmed nothing in trial 4 (sudden, +26 points) because the penalty 5 sigma^2 log n, tuned on "
             "long recorded histories, is too strict at 350 calls; with a multiplier of 3 or less it places the change "
             "at call 187. The deployed setting is reported unchanged.",
             "- In trial 2 only about a fifth of the first 50 post-fault calls were cut short (random draw at 35%), "
             "so the error rate reached its new level around call 225 and PELT placed the change at 238.",
             "- Every trial got an infra_change early warning on slow calls: median latency roughly doubled (to 14-19 s) "
             "while the five trials ran in parallel against one free-tier endpoint. It was never confirmed by PELT.",
             ""]
    (OUT / "PIPELINE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    pd.set_option("display.width", 220)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
