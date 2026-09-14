"""
detection_check.py

Run from the project root: python detection_check.py

UPDATED after: (1) calibration.py -> baseline.py rename, old baseline.py
-> naive_detector.py, and (2) changepoint_sequential.py rewritten to wrap
frouros's tuned Page-Hinkley (configs/detectors.yaml), which requires
PER-CALL binary error data -- different from PELT/bootstrap/naive_detector,
which all operate on PER-RUN aggregated MetricPoint values. This script
builds BOTH data shapes correctly so each detector is tested on the
granularity it actually needs -- feeding Page-Hinkley the wrong shape
(per-run aggregates) silently produces zero alerts, which is the exact
mismatch documented in changepoint_sequential.py's module docstring.
"""

import random
from datetime import datetime, timedelta, timezone

from driftguard.detection import baseline as cal
from driftguard.detection import changepoint_pelt as pelt
from driftguard.detection import bootstrap as bs
from driftguard.detection import multiple_testing as mt
from driftguard.detection.changepoint_sequential import run_sequential_detection
from driftguard.detection.naive_detector import run_baseline_detection
from driftguard.storage.store import DriftStore, MetricSeriesRow


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def build_per_run_scenario(seed=11, n_quiet=30, n_drift=30):
    """PER-RUN aggregated data -- what PELT, bootstrap, and naive_detector consume."""
    random.seed(seed)
    base_time = datetime.now(timezone.utc)
    points = []
    for i in range(n_quiet + n_drift):
        acc = max(0, min(1, random.gauss(0.65, 0.05))) if i < n_quiet else max(0, min(1, random.gauss(0.40, 0.05)))
        points.append(MetricSeriesRow(
            run_id=f"synthetic-{i}", timestamp=base_time + timedelta(hours=i),
            provider="synthetic", model_id="test-model",
            accuracy=acc,
            format_adherence=max(0, min(1, random.gauss(0.95, 0.03))),
            refusal_rate=max(0, min(1, random.choice([0.0, 0.0, 0.0, 0.2]))),
            latency_p95_ms=random.gauss(400, 40),
            n_calls=20,
        ))
    return points, base_time


def build_per_call_scenario(seed=11, n_quiet=300, n_drift=300, baseline_error=0.05, drift_error=0.35):
    """PER-CALL binary error data -- what the frouros-based Page-Hinkley needs (see module docstring)."""
    random.seed(seed)
    base_time = datetime.now(timezone.utc)
    quiet_calls = [1 if random.random() < baseline_error else 0 for _ in range(n_quiet)]
    drift_calls = [1 if random.random() < drift_error else 0 for _ in range(n_drift)]
    full_calls = quiet_calls + drift_calls
    timestamps = [base_time + timedelta(minutes=i) for i in range(len(full_calls))]
    return full_calls, timestamps


def main():
    section("PART 1: Per-run scenario (PELT / bootstrap / naive_detector) -- known injected drift in 'accuracy'")

    points, base_time = build_per_run_scenario()
    quiet_points = points[:30]

    calibration_result = cal.calibrate(quiet_points, "synthetic", "test-model", k=3.0)
    for name, mt_thr in calibration_result.thresholds.items():
        print(f"  {name}: baseline_threshold={mt_thr.baseline_threshold:.4f}, pelt_penalty={mt_thr.pelt_penalty:.6f}")

    section("PART 2: PELT + bootstrap significance (full 60-point per-run series)")

    bootstrap_results = {}
    for metric_name, mt_thr in calibration_result.thresholds.items():
        candidates = pelt.find_changepoints_for_metric(points, metric_name, penalty=mt_thr.pelt_penalty)
        print(f"\n{metric_name}: {len(candidates)} PELT candidate(s)")
        if not candidates:
            continue
        ordered = sorted(points, key=lambda p: p.timestamp)
        values = [getattr(p, metric_name) for p in ordered]
        sig_results = bs.evaluate_candidates(candidates, values, n_permutations=2000)
        if sig_results:
            best = min(sig_results, key=lambda r: r.p_value)
            bootstrap_results[metric_name] = best
            print(f"  most significant: index={best.candidate.index} delta={best.observed_delta:.4f} raw_p={best.p_value:.4f}")

    section("PART 3: Benjamini-Hochberg correction across metrics tested this run")

    if bootstrap_results:
        corrected = mt.apply_bh_correction(bootstrap_results, alpha=0.05)
        for name, cr in corrected.items():
            flag = "SIGNIFICANT" if cr.significant else "not significant"
            print(f"  {name}: raw_p={cr.result.p_value:.4f} -> corrected_p={cr.p_value_corrected:.4f} [{flag}]")
        print(f"\n  Real injected drift (accuracy) correctly flagged: "
              f"{corrected.get('accuracy') and corrected['accuracy'].significant}")
    else:
        print("  No adverse candidates found on any metric.")

    section("PART 4: Naive baseline detector (comparison point) -- same per-run accuracy series")

    acc_thr = calibration_result.thresholds["accuracy"]
    ordered = sorted(points, key=lambda p: p.timestamp)
    acc_values = [p.accuracy for p in ordered]
    acc_timestamps = [p.timestamp for p in ordered]
    offset = abs(acc_thr.baseline_threshold - acc_thr.based_on.median)
    naive_alerts = run_baseline_detection(acc_values, acc_timestamps, "accuracy", "lower_is_bad", offset=offset)
    print(f"  {len(naive_alerts)} alert(s) fired out of 30 post-drift points "
          f"(expect this to under-perform vs. real detectors, per its documented purpose).")

    section("PART 5: Page-Hinkley (frouros, tuned) -- REQUIRES per-call data, NOT the per-run series above")

    per_call_values, per_call_timestamps = build_per_call_scenario()
    ph_alerts = run_sequential_detection(per_call_values, per_call_timestamps, "accuracy_error_rate")
    print(f"  {len(ph_alerts)} alert(s) fired on the per-call error stream "
          f"(real shift injected at call index 300).")
    if ph_alerts:
        print(f"  First alert at index {ph_alerts[0].index} -- detection delay: {ph_alerts[0].index - 300} calls")

    print("\n  Sanity check -- feeding it the WRONG granularity (per-run data from Part 1) on purpose:")
    wrong_shape_alerts = run_sequential_detection(acc_values, acc_timestamps, "accuracy")
    print(f"  {len(wrong_shape_alerts)} alert(s) (expected: 0 or near-0 -- confirms the documented "
          f"granularity requirement in changepoint_sequential.py's module docstring)")

    section("PART 6: Where your REAL data currently stands")

    try:
        store = DriftStore(db_path="driftguard.db")
        real_series = store.get_metric_series("groq", "openai/gpt-oss-120b")
        print(f"  Real per-run quiet-window points collected so far: {len(real_series)}")
        if len(real_series) < cal.MIN_QUIET_POINTS:
            print(f"  Need >= {cal.MIN_QUIET_POINTS} for per-run calibration to be considered reliable.")
    except Exception as e:
        print(f"  Could not read driftguard.db: {e}")

    section("DONE")
    print("PART 3 should flag the real injected drift significant with no/few false positives.")
    print("PART 5's first block should fire quickly near call index 300; the second block should")
    print("stay near-silent -- that contrast IS the granularity-mismatch finding, working as documented.")


if __name__ == "__main__":
    main()