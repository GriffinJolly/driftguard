"""
validation/live_replay_groq.py

The actual "real-time dataset" version: calls a REAL LLM provider (Groq by
default, OpenRouter also supported) right now, live, using DriftGuard's own
production eval-suite code (driftguard.evalsuite.tasks.run_accuracy_tasks --
the same frozen-MMLU accuracy task runner.py uses every hour in production),
and feeds the resulting REAL per-call correctness stream through the same
9-detector registry used everywhere else in validation/.

This is different from live_replay.py (which validates against a real but
generic public ML benchmark) and different from run_validation.py (which
validates against synthetic, injected drift with a known ground truth).
This script validates against DriftGuard's own real signal, from a real
model, called right now over the network -- the actual thing the finished
system watches in production, just compressed into a few minutes instead
of spread across weeks of hourly runs.

Requires:
  - A GROQ_API_KEY (or OPENROUTER_API_KEY) in a .env file at the repo root,
    same as manual_test.py and the rest of the project already expect.
  - Being run from the repo root, so `driftguard` and `validation` both
    resolve as packages: `python -m validation.live_replay_groq`

What it does, step by step:
  1. Calls driftguard.evalsuite.tasks.run_accuracy_tasks(...) once per
     "round" -- this runs the real, frozen 20-question MMLU subset
     against the real provider (20 real API calls per round).
  2. Every individual call's real outcome (correct/incorrect) is fed,
     live, into all 9 drift detectors AND written to driftguard.db via
     DriftStore.write_log -- so running this script also populates your
     actual database with real traffic, exactly like a production run of
     runner.py would (just compressed in time).
  3. Any detector that fires is printed immediately, live, as it happens.
  4. At the end: the same alert-summary CSV + detection-timeline chart
     that live_replay.py produces, so results from both are directly
     comparable.

Cost/time note: each round is 20 real API calls. Groq's free tier is
~25 req/min (see driftguard/providers/clients.py's RateLimiter) -- a few
rounds run almost immediately, but --rounds above ~5 will start hitting
the rate limiter and take a few minutes. This is deliberate: it's what
gives the detectors real elapsed time to operate over, rather than firing
all 20*rounds calls in a single instant.

Usage (from the repo root):
    python -m validation.live_replay_groq --rounds 10
    python -m validation.live_replay_groq --rounds 20 --provider groq --model openai/gpt-oss-120b
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# These are DriftGuard's own, already-built modules -- this script is
# meant to live at validation/live_replay_groq.py inside the real repo,
# where `driftguard` is a sibling top-level package.
from driftguard.evalsuite.tasks import run_accuracy_tasks
from driftguard.storage.store import DriftStore

from validation.detectors_registry import default_registry, tuned_registry
from validation.live_replay import make_timeline_chart


def run(provider: str, model_id: str, rounds: int, out_dir: Path, db_path: str, registry=None) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    store = DriftStore(db_path=db_path)

    if registry is None:
        registry = default_registry(include_tuned_variants=True)
    detectors = {spec.name: spec.factory() for spec in registry}
    alerts_by_detector: dict[str, list[int]] = {name: [] for name in detectors}

    errors: list[int] = []
    call_index = 0
    t_start = time.monotonic()

    print(f"Calling {provider}/{model_id} live -- {rounds} round(s) x 20 real MMLU calls each.")
    print("(Ctrl+C at any point is safe -- results-so-far are still saved.)\n")

    try:
        for round_i in range(rounds):
            eval_run_id = f"live_replay_groq_{int(time.time())}_{round_i}"
            entries = run_accuracy_tasks(provider, model_id, eval_run_id)

            for entry in entries:
                store.write_log(entry)  # real traffic -> real driftguard.db, same as production

                if entry.eval_score is None:
                    # the call itself failed (timeout/error/rate-limited) -- skip,
                    # same convention run_validation.py's synthetic benchmark uses
                    continue

                error = 0 if entry.eval_score == 1.0 else 1
                errors.append(error)

                for name, detector in detectors.items():
                    detector.update(value=error)
                    if detector.status.get("drift"):
                        alerts_by_detector[name].append(call_index)
                        elapsed = time.monotonic() - t_start
                        print(f"  [{elapsed:6.1f}s] call #{call_index:>4}  DRIFT flagged by {name}")

                call_index += 1

            elapsed = time.monotonic() - t_start
            rolling_err = np.mean(errors[-20:]) if errors else float("nan")
            print(
                f"[{elapsed:6.1f}s] round {round_i + 1}/{rounds} done "
                f"({call_index} real calls so far, last-20 error rate={rolling_err:.2f})"
            )
    except KeyboardInterrupt:
        print("\nInterrupted -- saving results collected so far.")

    if not errors:
        print("No successful calls were recorded -- nothing to report. Check your API key / network.")
        sys.exit(1)

    errors_arr = np.array(errors, dtype=np.int8)
    rows = [
        {
            "detector": name,
            "provider": provider,
            "model_id": model_id,
            "n_real_calls": len(errors_arr),
            "n_alerts": len(alerts),
            "first_alert_index": alerts[0] if alerts else None,
            "alerts_per_1000_calls": len(alerts) / len(errors_arr) * 1000,
        }
        for name, alerts in alerts_by_detector.items()
    ]
    summary = pd.DataFrame(rows).sort_values("n_alerts")
    # Sanitize every filesystem-unsafe character, not just "/". OpenRouter's
    # free model ids end in ":free", and on Windows a ":" in a path is the
    # NTFS alternate-data-stream separator -- left unsanitized it silently
    # routes the CSV/PNG into ADS streams instead of real files.
    tag = re.sub(r'[^A-Za-z0-9._-]', "-", f"{provider}_{model_id}")
    summary.to_csv(out_dir / f"{tag}_alert_summary.csv", index=False)
    pd.Series(errors_arr).to_frame("error").to_csv(out_dir / f"{tag}_error_stream.csv", index=False)
    make_timeline_chart(errors_arr, alerts_by_detector, f"{provider}/{model_id} (LIVE)", out_dir / f"{tag}_timeline.png")

    print("\n" + summary.to_string(index=False))
    print(f"\nWrote alert summary, error stream, and timeline chart to {out_dir}/")
    print(f"Real traffic from this run was also written to {db_path} -- inspect it with:")
    print(f"    python -m driftguard.scripts.inspect_production_logs --db {db_path} --integration-path eval_suite")
    return summary


def _main() -> None:
    parser = argparse.ArgumentParser(description="Live, real-time drift detector validation against a real LLM API")
    parser.add_argument("--provider", choices=["groq", "openrouter"], default="groq")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b")
    parser.add_argument("--rounds", type=int, default=10, help="Each round = 20 real API calls (the frozen MMLU subset)")
    parser.add_argument("--out", type=str, default="validation/results/live_replay_groq")
    parser.add_argument("--db-path", type=str, default="driftguard.db")
    parser.add_argument(
        "--registry", choices=["default", "tuned"], default="default",
        help="'default': library defaults + Page-Hinkley (tuned) only. "
             "'tuned': ALL 8 detectors individually tuned (validation/tuning_sweep_all.py), "
             "so Page-Hinkley (tuned) and DDM (tuned) are validated live alongside the other 6. "
             "The Page-Hinkley+DDM OR-hybrid is excluded (rejected -- see HYBRID_VERDICT.md; "
             "the project uses Page-Hinkley alone). Mirrors validation/live_replay.py's --registry flag.",
    )
    args = parser.parse_args()
    registry = tuned_registry() if args.registry == "tuned" else None
    out_dir = Path(args.out) / "tuned" if args.registry == "tuned" else Path(args.out)
    run(args.provider, args.model, args.rounds, out_dir, args.db_path, registry=registry)


if __name__ == "__main__":
    _main()
