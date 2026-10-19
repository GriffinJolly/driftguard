"""
validation/baseline_comparison.py

DriftGuard against what a team would do without it, on streams where the change
point is known. No API calls: every stream is real data already collected, or
synthetic with a planted change.

Methods (each sees the same per-call 0/1 stream, 1 = bad outcome):

  static threshold      alarm when the rate over the last 100 calls exceeds the
                        rate over the first 100 calls by 10 points
  periodic re-eval      the stream is cut into runs of 100 calls (one run of a
                        100-question suite); each run is compared with the first
                        by a one-sided two-proportion z-test, alarm at p < 0.05.
                        This is the snapshot comparison of Chen et al.
  periodic, corrected   the same with a Bonferroni correction over the runs
  DriftGuard live       noise-scaled Page-Hinkley (pipeline.page_hinkley_alerts)
  DriftGuard confirmed  PELT -> permutation test -> Benjamini-Hochberg, keeping
                        changes in the bad direction (the deployed batch path)

Scoring: a false alarm is any alarm before the change (or any alarm at all on a
stream without one); a detection is an alarm at or after it. PELT locates a
change after the fact on a grid, with a minimum segment length (200 calls on
long streams, a fifth of a short one), so a confirmed change counts as a
detection if it lies within one minimum segment before the onset, and its delay
is reported as the distance to the onset (location error).

Data:
  LMSYS-Chat-1M   real replies, 30 staged rollouts, 30 sudden swaps, 30 controls
                  (validation/lmsys_gradual_drift.py), refusals of safe prompts
  synthetic       10,000-call streams at 5%, 15% and 30% error, 30 with a
                  10-point rise at call 5,000 and 30 without, per rate
  live            the 2 Groq and 5 NVIDIA trial streams (results/live_drift_compare*)

Usage:
    python -m validation.baseline_comparison
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from driftguard.detection import pipeline
from driftguard.detection.bootstrap import confirm_changepoints
from driftguard.detection.changepoint_pelt import detect_changepoints
from validation import lmsys_gradual_drift as G

OUT = Path("validation/results/baseline_comparison")
S = pipeline.load_settings()
WIN, RISE, RUN, ALPHA, TOL = 100, 0.10, 100, 0.05, 24
METHODS = ["Static threshold", "Periodic re-eval", "Periodic, corrected", "DriftGuard live", "DriftGuard confirmed"]


# --------------------------------------------------------------------------- methods

def static_threshold(v: np.ndarray) -> list[int]:
    base = v[:WIN].mean()
    c = np.cumsum(np.insert(v.astype(float), 0, 0.0))
    for i in range(WIN, len(v)):
        if (c[i + 1] - c[i + 1 - WIN]) / WIN > base + RISE:
            return [i]
    return []


def _z_one_sided(x1: int, n1: int, x0: int, n0: int) -> float:
    p = (x1 + x0) / (n1 + n0)
    if p in (0.0, 1.0):
        return 1.0
    z = (x1 / n1 - x0 / n0) / math.sqrt(p * (1 - p) * (1 / n1 + 1 / n0))
    return 0.5 * math.erfc(z / math.sqrt(2))


def periodic(v: np.ndarray, corrected: bool) -> list[int]:
    runs = len(v) // RUN
    if runs < 2:
        return []
    x0 = int(v[:RUN].sum())
    alpha = ALPHA / (runs - 1) if corrected else ALPHA
    for k in range(1, runs):
        seg = v[k * RUN:(k + 1) * RUN]
        if _z_one_sided(int(seg.sum()), RUN, x0, RUN) < alpha:
            return [(k + 1) * RUN - 1]          # known when the run finishes
    return []


def dg_live(v: np.ndarray) -> list[int]:
    return pipeline.page_hinkley_alerts(v, S)


def dg_confirmed(v: np.ndarray) -> list[int]:
    cps = detect_changepoints(v, pipeline.effective_pelt(len(v), S))
    if not cps:
        return []
    return [r["index"] for r in confirm_changepoints(v, cps, S.significance)
            if r["significant"] and r["post_mean"] > r["pre_mean"]]


def run_all(v: np.ndarray) -> dict[str, list[int]]:
    return {"Static threshold": static_threshold(v), "Periodic re-eval": periodic(v, False),
            "Periodic, corrected": periodic(v, True), "DriftGuard live": dg_live(v),
            "DriftGuard confirmed": dg_confirmed(v)}


def score(alarms: list[int], onset: int | None, method: str, n: int) -> dict:
    slack = pipeline.effective_pelt(n, S).min_segment_length if method == "DriftGuard confirmed" else 0
    if onset is None:
        return {"false_alarm": bool(alarms), "detected": None, "delay": None}
    early = [a for a in alarms if a < onset - slack]
    late = [a for a in alarms if a >= onset - slack]
    delay = None
    if late:
        delay = abs(late[0] - onset) if method == "DriftGuard confirmed" else late[0] - onset
    return {"false_alarm": bool(early), "detected": bool(late), "delay": delay}


# --------------------------------------------------------------------------- data

def lmsys_streams():
    data = G.DATA if Path(G.DATA).exists() else Path(os.environ.get("LMSYS_PARQUET", G.DATA))
    pool = pd.read_parquet(data)
    safe = pool[~pool["flagged"]]
    pa = safe.loc[safe["model"] == G.MODEL_A, "refused"].to_numpy()
    pb = safe.loc[safe["model"] == G.MODEL_B, "refused"].to_numpy()
    rng = np.random.default_rng(0)
    for kind in ("gradual", "sudden", "control"):
        for t in range(30):
            yield (f"LMSYS {kind}", G.make_stream(pa, pb, kind, rng).astype(float),
                   None if kind == "control" else G.N_HEALTHY)


def synthetic_streams(n_each: int = 30):
    for p0 in (0.05, 0.15, 0.30):
        for shift in (0.0, 0.10):
            for t in range(n_each):
                rng = np.random.default_rng([31, int(p0 * 100), int(shift * 100), t])
                probs = np.full(10_000, p0)
                probs[5_000:] += shift
                v = (rng.random(10_000) < probs).astype(float)
                label = f"Synthetic {int(p0 * 100)}% {'+10 pts' if shift else 'stable'}"
                yield label, v, (5_000 if shift else None)


def live_streams():
    for folder, label in (("live_drift_compare", "Live Groq"), ("live_drift_compare_nvidia", "Live NVIDIA")):
        for f in sorted(Path("validation/results", folder, "streams").glob("*_stream.csv")):
            v = pd.read_csv(f)["error"].to_numpy().astype(float)
            yield label, v, 175


# --------------------------------------------------------------------------- run

def main() -> None:
    rows = []
    for source in (lmsys_streams(), synthetic_streams(), live_streams()):
        for group, v, onset in source:
            for method, alarms in run_all(v).items():
                rows.append({"group": group, "method": method, "n": len(v), **score(alarms, onset, method, len(v))})
        print("done", group, flush=True)
    raw = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    raw.to_csv(OUT / "raw.csv", index=False)

    summary = raw.groupby(["group", "method"], sort=False).agg(
        streams=("false_alarm", "size"),
        false_alarm=("false_alarm", "mean"),
        detected=("detected", lambda s: s.dropna().astype(float).mean() if s.notna().any() else np.nan),
        median_delay=("delay", lambda s: s.dropna().median() if s.notna().any() else np.nan),
    ).reset_index()
    summary.to_csv(OUT / "summary.csv", index=False)

    lines = ["# DriftGuard against practical baselines", "",
             "Every method sees the same per-call 0/1 stream with a known change point. False alarm = any alarm "
             "before the change, or any alarm on a stream without one. Detected = an alarm at or after the change "
             "(a confirmed PELT change within one minimum segment before it also counts, since PELT locates changes "
             "on a grid; its delay is the distance to the onset).", "",
             f"- Static threshold: rate over the last {WIN} calls > first {WIN} calls + {int(RISE * 100)} points.",
             f"- Periodic re-eval: runs of {RUN} calls vs the first run, one-sided two-proportion z-test, p < {ALPHA}.",
             "- Periodic, corrected: the same with a Bonferroni correction over the runs.",
             "- DriftGuard live: noise-scaled Page-Hinkley. DriftGuard confirmed: PELT, permutation test, Benjamini-Hochberg.", ""]
    for g, t in summary.groupby("group", sort=False):
        lines += [f"## {g}", "", t.drop(columns="group").to_markdown(index=False, floatfmt=".2f"), ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    print(summary.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
