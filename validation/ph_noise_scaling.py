"""
validation/ph_noise_scaling.py

Why the live detector's threshold is scaled to each signal's noise.

lambda = 12 was tuned on 0/1 streams with a 5% bad-call rate (run_validation.py).
Real signals are noisier: accuracy errors run 12-30%. With a fixed threshold the
running Page-Hinkley sum wanders further by chance on a noisier stream, so it
alarms on noise, and more often the longer it runs. This script compares the
previous setting (fixed lambda, alpha = 0.9999) with the current one from
configs/detectors.yaml (lambda and delta scaled by the baseline noise, alpha =
0.998), both through pipeline.page_hinkley_alerts, on:

  1. long streams (10,000 calls) at base rates 5-50%, stable or shifted at call 5,000;
  2. the original synthetic benchmark design (800 samples, 5% base rate);
  3. LMSYS-Chat-1M staged rollouts and controls (real replies, held-out seed 1).

No API calls. Usage:
    python -m validation.ph_noise_scaling            # ~10 minutes
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from driftguard.detection import pipeline
from validation import lmsys_gradual_drift as G
from validation.ground_truth import make_gradual, make_stable, make_sudden, make_transient

OUT = Path("validation/results/ph_noise_scaling")
LONG, SHIFT_AT, LONG_TRIALS = 10_000, 5_000, 60
SYN_TRIALS, LMSYS_TRIALS = 100, 100

CURRENT = pipeline.load_settings()
PREVIOUS = dataclasses.replace(CURRENT, ph_alpha=0.9999, ph_noise_ref=None)
SETTINGS = {"previous (fixed λ=12, α=0.9999)": PREVIOUS, "current (noise-scaled, α=0.998)": CURRENT}


def _long_streams() -> list[dict]:
    rows = []
    for p0 in (0.05, 0.15, 0.30, 0.50):
        for shift in (0.0, 0.03, 0.05, 0.10, 0.20):
            for name, s in SETTINGS.items():
                fa, hit, delays = 0, 0, []
                for t in range(LONG_TRIALS):
                    rng = np.random.default_rng([7, int(p0 * 100), int(shift * 100), t])
                    probs = np.full(LONG, p0)
                    probs[SHIFT_AT:] += shift
                    a = pipeline.page_hinkley_alerts((rng.random(LONG) < probs).astype(float), s)
                    fa += any(i < SHIFT_AT for i in a) if shift else bool(a)
                    post = next((i for i in a if i >= SHIFT_AT), None)
                    if shift and post is not None:
                        hit += 1
                        delays.append(post - SHIFT_AT)
                rows.append(dict(test="long stream", base_rate=p0, shift=shift, setting=name,
                                 false_alarm=fa / LONG_TRIALS,
                                 detected=hit / LONG_TRIALS if shift else np.nan,
                                 median_delay=float(np.median(delays)) if delays else np.nan))
    return rows


def _synthetic() -> list[dict]:
    rows = []
    makers = {
        "stable": lambda r, m: make_stable(r, 800, 0.05),
        "sudden": lambda r, m: make_sudden(r, 800, 400, 0.05, 0.05 + m),
        "gradual": lambda r, m: make_gradual(r, 800, 400, 120, 0.05, 0.05 + m),
        "transient": lambda r, m: make_transient(r, 800, 400, 50, 0.05, 0.05 + m),
    }
    for scen, make in makers.items():
        for m in ((0.0,) if scen == "stable" else (0.10, 0.20, 0.35)):
            for name, s in SETTINGS.items():
                fa, hit, delays = 0, 0, []
                for t in range(SYN_TRIALS):
                    rng = np.random.default_rng([11, ["stable", "sudden", "gradual", "transient"].index(scen),
                                                 int(m * 100), t])
                    a = pipeline.page_hinkley_alerts(make(rng, m).stream, s)
                    first = a[0] if a else None
                    if scen in ("stable", "transient"):
                        fa += first is not None
                    elif first is not None and first >= 400:
                        hit += 1
                        delays.append(first - 400)
                    elif first is not None:
                        fa += 1
                rows.append(dict(test=f"synthetic {scen}", base_rate=0.05, shift=m or np.nan, setting=name,
                                 false_alarm=fa / SYN_TRIALS,
                                 detected=hit / SYN_TRIALS if scen in ("sudden", "gradual") else np.nan,
                                 median_delay=float(np.median(delays)) if delays else np.nan))
    return rows


def _lmsys() -> list[dict]:
    pool = pd.read_parquet(G.DATA)
    safe = pool[~pool["flagged"]]
    pa = safe.loc[safe["model"] == G.MODEL_A, "refused"].to_numpy()
    pb = safe.loc[safe["model"] == G.MODEL_B, "refused"].to_numpy()
    rng = np.random.default_rng(1)
    streams = {k: [G.make_stream(pa, pb, k, rng) for _ in range(LMSYS_TRIALS)]
               for k in ("control", "sudden", "gradual")}
    onset, rows = G.N_HEALTHY, []
    for name, s in SETTINGS.items():
        ctrl = np.mean([bool(pipeline.page_hinkley_alerts(v, s)) for v in streams["control"]])
        rows.append(dict(test="LMSYS control", base_rate=np.nan, shift=0.0, setting=name,
                         false_alarm=ctrl, detected=np.nan, median_delay=np.nan))
        for kind in ("sudden", "gradual"):
            al = [pipeline.page_hinkley_alerts(v, s) for v in streams[kind]]
            post = [next((i for i in a if i >= onset), None) for a in al]
            rows.append(dict(test=f"LMSYS {kind}", base_rate=np.nan, shift=np.nan, setting=name,
                             false_alarm=np.mean([any(i < onset for i in a) for a in al]),
                             detected=np.mean([p is not None for p in post]),
                             median_delay=float(np.median([p - onset for p in post if p is not None]))))
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("long streams ...", flush=True)
    rows = _long_streams()
    print("synthetic benchmark ...", flush=True)
    rows += _synthetic()
    print("LMSYS ...", flush=True)
    rows += _lmsys()
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "summary.csv", index=False)

    stable = t[(t["test"] == "long stream") & (t["shift"] == 0)]
    wide = stable.pivot(index="base_rate", columns="setting", values="false_alarm")
    lines = ["# Page-Hinkley: threshold scaled to each signal's noise", "",
             "Previous setting: fixed λ = 12, α = 0.9999 (tuned on a 5% bad-call rate). Current setting "
             "(configs/detectors.yaml): λ and δ scaled up by the signal's baseline standard deviation "
             "relative to a 5% rate, detector starts after the 100-call baseline, α = 0.998. "
             "Both run through pipeline.page_hinkley_alerts. False alarm = any alarm on a stream (or before "
             "the change); detected = an alarm at or after the change.", "",
             "## False alarms on stable 10,000-call streams", "", wide.to_markdown(floatfmt=".2f"), "",
             "## All results", "", t.to_markdown(index=False, floatfmt=".3f"), ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
