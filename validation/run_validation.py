"""
validation/run_validation.py

Reproduces -- and substantially extends -- the "Detection Rate Across All
Drift Detectors" comparison from the project deck, as committed, runnable
code rather than a one-off chart nobody can regenerate.

What this adds beyond the original comparison
-----------------------------------------------
The original chart reported one number per detector per scenario (sudden /
gradual), with no stated sample size and no false-alarm measurement. That's
not enough to justify "virtually no false alarms" or "better than ADWIN at
ignoring temporary fluctuations" -- both claims from the deck. This harness:

  1. Runs many independent trials per (detector, scenario) cell -- not one
     -- so reported rates come with a real sample size (see --trials).
  2. Sweeps three drift magnitudes (small/medium/large), not just one, so
     "detection rate" isn't a single lucky (or unlucky) number.
  3. Adds a STABLE scenario (no drift, ever) to measure false alarm rate --
     the claim the slide makes but never actually measured in the repo.
  4. Adds a TRANSIENT scenario (a temporary blip that reverts) to directly
     test the "distinguishes persistent degradation from noise" claim,
     instead of asserting it from the sudden/gradual numbers alone.
  5. Records detection DELAY (samples from true change point to alarm),
     not just whether detection happened -- relevant since the deck
     specifically pitches Page-Hinkley as the *low-latency* detector.

Usage (run from the repo root, so `validation` resolves as a package):
    python -m validation.run_validation
    python -m validation.run_validation --trials 100 --out validation/results/

Output: results/raw_results.csv -- one row per (detector, scenario,
magnitude, trial). driftguard/validation/report.py consumes this file to
produce the summary tables, charts, and the written verdict.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from validation.detectors_registry import default_registry, run_detector_on_stream, tuned_registry
from validation.ground_truth import (
    make_gradual,
    make_stable,
    make_sudden,
    make_transient,
)

# ---------------------------------------------------------------------------
# Benchmark parameters
# ---------------------------------------------------------------------------
STREAM_LENGTH = 800
CHANGE_POINT = 400          # where degradation begins (sudden/gradual/transient)
GRADUAL_RAMP_LEN = 120      # gradual scenario ramps over this many samples
TRANSIENT_BLIP_LEN = 50     # transient scenario reverts after this many samples

BASELINE_ERROR_RATE = 0.05  # p0 -- a "healthy" ~95%-correct model
DRIFT_MAGNITUDES = {
    "small": 0.10,   # p1 = 0.15 -- subtle regression
    "medium": 0.20,  # p1 = 0.25 -- moderate regression
    "large": 0.35,   # p1 = 0.40 -- severe regression
}

DEFAULT_TRIALS = 60
DEFAULT_SEED = 42  # matches the project's existing "frozen, seeded" convention


def _build_scenario(scenario_name: str, magnitude_name: str, rng: np.random.Generator):
    p0 = BASELINE_ERROR_RATE
    if scenario_name == "stable":
        return make_stable(rng, STREAM_LENGTH, p0)

    delta = DRIFT_MAGNITUDES[magnitude_name]
    p1 = p0 + delta

    if scenario_name == "sudden":
        return make_sudden(rng, STREAM_LENGTH, CHANGE_POINT, p0, p1)
    if scenario_name == "gradual":
        return make_gradual(rng, STREAM_LENGTH, CHANGE_POINT, GRADUAL_RAMP_LEN, p0, p1)
    if scenario_name == "transient":
        return make_transient(rng, STREAM_LENGTH, CHANGE_POINT, TRANSIENT_BLIP_LEN, p0, p1)

    raise ValueError(f"Unknown scenario: {scenario_name}")


def run_all(trials: int = DEFAULT_TRIALS, seed: int = DEFAULT_SEED, registry=None) -> pd.DataFrame:
    if registry is None:
        registry = default_registry()
    rows: list[dict] = []

    # stable has no "magnitude" -- it's one condition, run with its own trial budget
    scenario_magnitude_pairs = [("stable", None)]
    for scenario_name in ("sudden", "gradual", "transient"):
        for magnitude_name in DRIFT_MAGNITUDES:
            scenario_magnitude_pairs.append((scenario_name, magnitude_name))

    total_cells = len(scenario_magnitude_pairs) * trials * len(registry)
    done = 0
    t_start = time.monotonic()

    for scenario_name, magnitude_name in scenario_magnitude_pairs:
        for trial in range(trials):
            # unique, reproducible seed per (scenario, magnitude, trial)
            trial_seed = hash((seed, scenario_name, magnitude_name, trial)) % (2**32)
            rng = np.random.default_rng(trial_seed)
            scenario = _build_scenario(scenario_name, magnitude_name, rng)

            for spec in registry:
                result = run_detector_on_stream(spec, scenario.stream)
                first_idx = result["first_drift_index"]

                detected = first_idx is not None
                # "correct" detection: flagged AT OR AFTER the true change point
                # (for stable, change_point is None -> any detection at all is a false alarm)
                if scenario.change_point is None:
                    false_alarm = detected
                    correct_detection = False
                    delay = None
                else:
                    false_alarm = detected and first_idx < scenario.change_point
                    correct_detection = detected and first_idx >= scenario.change_point
                    delay = (first_idx - scenario.change_point) if correct_detection else None

                rows.append(
                    {
                        "detector": spec.name,
                        "scenario": scenario_name,
                        "magnitude": magnitude_name,
                        "trial": trial,
                        "is_real_drift": scenario.is_real_drift,
                        "detected_at_all": detected,
                        "correct_detection": correct_detection,
                        "false_alarm": false_alarm,
                        "delay": delay,
                    }
                )
                done += 1

        elapsed = time.monotonic() - t_start
        print(
            f"[{elapsed:6.1f}s] finished scenario={scenario_name} "
            f"magnitude={magnitude_name} ({done}/{total_cells} detector-trials)"
        )

    return pd.DataFrame(rows)


def _main() -> None:
    parser = argparse.ArgumentParser(description="DriftGuard detector validation benchmark")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=str, default="validation/results")
    parser.add_argument(
        "--registry", choices=["default", "tuned"], default="default",
        help="'default' = library defaults + Page-Hinkley (tuned) only (original comparison). "
             "'tuned' = ALL 8 detectors individually tuned (validation/tuning_sweep_all.py) -- the fair fight.",
    )
    parser.add_argument(
        "--filename", type=str, default="raw_results.csv",
        help="Output CSV filename (use a distinct name for --registry tuned so it doesn't overwrite the default run).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    registry = tuned_registry() if args.registry == "tuned" else None
    df = run_all(trials=args.trials, seed=args.seed, registry=registry)
    out_path = out_dir / args.filename
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    _main()
