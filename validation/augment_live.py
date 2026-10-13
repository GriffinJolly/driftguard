"""
validation/augment_live.py

Data augmentation for the live drift-detector comparison.

Collecting many independent live trials is rate-limited (the gradual scenario in
particular kept starving on the provider's free-tier caps). This module instead
AUGMENTS the trials we could not finish collecting live, by resampling the REAL
per-call outcomes we already have in driftguard.db.

How it stays honest:
  * Two real outcome pools are read straight from the database --
      STRONG   = calls made at full max_tokens (the healthy model), and
      DEGRADED = calls truncated to a tiny max_tokens (the induced fault).
    Every 0/1 in an augmented stream is therefore a genuine, graded API result;
    nothing is invented.
  * Only the COMPOSITION follows the drift schedule (same schedule the live
    harness uses): a synthetic stream draws each position from the strong pool,
    except a scheduled fraction of post-onset positions that draw from the
    degraded pool. This is a block/bootstrap resample, standard for getting
    statistical power out of limited real data.

Caveats, stated plainly:
  * Resampling with replacement makes samples i.i.d. within each regime -- it
    reproduces the real error RATES and the drift shape, not any fine-grained
    autocorrelation. For these detectors (which respond to rate shifts) that is
    the property that matters, but augmented trials are NOT a substitute for
    independent live trials -- report them as augmented.
  * The degraded pool is small (~135 real calls); it is sampled with
    replacement.

Usage:
    python -m validation.augment_live --n-trials 40
    python -m validation.augment_live --n-trials 40 --scenarios sudden,gradual,stable
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from validation.live_drift_compare import (
    DEGRADE_FRAC_DEFAULT,
    GRADUAL_RAMP_DEFAULT,
    ONSET_DEFAULT,
    POST_DEFAULT,
    _realized_rates,
    _registry,
    _score_trial,
    aggregate,
)
from validation.detectors_registry import run_detector_on_stream

_DEGRADE_MAXTOK = 64  # calls with max_tokens <= this are the degraded (truncated) pool


def load_pools(db_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read the real strong-model and degraded outcome pools from driftguard.db.

    Returns (strong, degraded) as 0/1 arrays (1 = wrong). A call is 'degraded'
    if its stored request had a tiny max_tokens (the induced truncation fault)."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT request_params_json, eval_score FROM log_rows "
            "WHERE eval_run_id LIKE 'live_drift_compare_%' AND eval_score IS NOT NULL"
        ).fetchall()
    finally:
        con.close()
    strong, degraded = [], []
    for rp, score in rows:
        try:
            mt = json.loads(rp).get("max_tokens")
        except Exception:
            mt = None
        err = 0 if score == 1.0 else 1
        (degraded if (mt is not None and mt <= _DEGRADE_MAXTOK) else strong).append(err)
    if not strong or not degraded:
        raise SystemExit("Not enough real data in driftguard.db to build both pools. "
                         "Run at least one live_drift_compare trial first.")
    return np.array(strong, dtype=np.int8), np.array(degraded, dtype=np.int8)


def synth_stream(scenario: str, onset: int, post: int, ramp: int, degrade_frac: float,
                 strong: np.ndarray, degraded: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One augmented 0/1 error stream, resampled from the real pools per the schedule."""
    n = onset + post
    out = np.empty(n, dtype=np.int8)
    for i in range(n):
        if i < onset or scenario == "stable":
            frac = 0.0
        elif scenario == "sudden":
            frac = degrade_frac
        elif scenario == "gradual":
            frac = degrade_frac * min(1.0, (i - onset) / max(1, ramp))
        else:
            raise ValueError(f"Unknown scenario: {scenario!r}")
        pool = degraded if rng.random() < frac else strong
        out[i] = pool[rng.integers(len(pool))]
    return out


def _stable_false_alarm(strong: np.ndarray, onset: int, post: int, n_trials: int,
                        rng: np.random.Generator) -> pd.DataFrame:
    """Control: on a NO-DRIFT stream (all strong), how often does each detector fire at all?"""
    order = [spec.name for spec in _registry()]
    fired = {name: 0 for name in order}
    for _ in range(n_trials):
        stream = np.array([strong[rng.integers(len(strong))] for _ in range(onset + post)], dtype=np.int8)
        for spec in _registry():
            if run_detector_on_stream(spec, stream)["first_drift_index"] is not None:
                fired[spec.name] += 1
    return pd.DataFrame({"detector": order,
                         "stable_false_alarm_rate": [fired[n] / n_trials for n in order]}).set_index("detector")


def run(db_path: str, out_dir: Path, n_trials: int, scenarios: list[str], onset: int,
        post: int, ramp: int, degrade_frac: float, seed: int) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    strong, degraded = load_pools(db_path)
    print(f"Real pools from driftguard.db:  STRONG n={len(strong)} err={strong.mean():.3f}  |  "
          f"DEGRADED n={len(degraded)} err={degraded.mean():.3f}")
    print(f"Augmenting: {n_trials} trials x {scenarios}  (onset={onset}, post={post}, "
          f"degrade_frac={degrade_frac}) -- resampled from real outcomes, no API calls.\n")

    rng = np.random.default_rng(seed)
    rows = []
    drift_scenarios = [s for s in scenarios if s in ("sudden", "gradual")]
    for scenario in drift_scenarios:
        for t in range(n_trials):
            stream = synth_stream(scenario, onset, post, ramp, degrade_frac, strong, degraded, rng).astype(float)
            p0, p1 = _realized_rates(stream, onset)
            mag = (p1 - p0) if not np.isnan(p1) else 0.0
            rows.extend(_score_trial(stream, scenario, onset, mag, f"{scenario}_aug_{t}",
                                     "augmented", "resampled-from-real", "resampled-degraded"))
    pd.DataFrame(rows).to_csv(out_dir / "augmented_raw_results.csv", index=False)

    summary = aggregate(Namespace(provider="augmented", out=str(out_dir),
                                  model="resampled-from-real-outcomes", onset=onset))

    if "stable" in scenarios:
        stab = _stable_false_alarm(strong, onset, post, n_trials, rng)
        stab.to_csv(out_dir / "augmented_stable_false_alarm.csv")
        print("\nSTABLE control (no-drift stream, resampled strong pool) -- false-alarm rate:")
        print(stab.round(3).to_string())
    return summary


def _main() -> None:
    p = argparse.ArgumentParser(description="Augment live drift trials by resampling real collected outcomes")
    p.add_argument("--db-path", dest="db_path", default="driftguard.db")
    p.add_argument("--out", default="validation/results/live_drift_compare_augmented")
    p.add_argument("--n-trials", dest="n_trials", type=int, default=40, help="augmented trials per scenario")
    p.add_argument("--scenarios", default="sudden,gradual,stable")
    p.add_argument("--onset", type=int, default=ONSET_DEFAULT)
    p.add_argument("--post", type=int, default=POST_DEFAULT)
    p.add_argument("--ramp-len", dest="ramp_len", type=int, default=GRADUAL_RAMP_DEFAULT)
    p.add_argument("--degrade-frac", dest="degrade_frac", type=float, default=DEGRADE_FRAC_DEFAULT)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args.db_path, Path(args.out), args.n_trials, [s.strip() for s in args.scenarios.split(",") if s.strip()],
        args.onset, args.post, args.ramp_len, args.degrade_frac, args.seed)


if __name__ == "__main__":
    _main()
