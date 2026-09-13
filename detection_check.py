"""
detection_check.py

Run from the project root: python detection_check.py

Tests the detection layer (calibration, PELT, bootstrap, multiple_testing,
Page-Hinkley, baseline) as a WHOLE pipeline, using a synthetic injected-drift
scenario -- because your real driftguard.db almost certainly doesn't have
enough quiet-window points yet for real calibration to be meaningful. This
script proves the six modules work correctly TOGETHER, using controlled data
where we know exactly where the "real" drift is, so we can check whether
each detector finds it.

It also separately reports what calibration looks like against your
actual real accumulated data, so you can see how close you are to having
enough for genuine calibration (recommended >= 20 quiet-window points).

Nothing here writes to your real driftguard.db -- the synthetic part uses
an in-memory scenario only.
"""

import random
from datetime import datetime, timedelta, timezone

from driftguard.detection import calibration as cal
from driftguard.detection import changepoint_pelt as pelt
from driftguard.detection import bootstrap as bs
from driftguard.detection import multiple_testing as mt
from driftguard.detection.changepoint_sequential import run_sequential_detection
from driftguard.detection.baseline import run_baseline_detection
from driftguard.storage.store import DriftStore, MetricSeriesRow


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def build_synthetic_scenario(seed=11, n_quiet=30, n_drift=30):
    """
    Builds a controlled scenario: n_quiet stable points, then a REAL
    injected capability regression (accuracy drops) for n_drift points.
    Other metrics stay stable throughout, so we can also confirm the
    detectors correctly stay quiet on the metrics that DIDN'T change.
    """
    random.seed(seed)
    base_time = datetime.now(timezone.utc)
    points = []
    for i in range(n_quiet + n_drift):
        if i < n_quiet:
            acc = max(0, min(1, random.gauss(0.65, 0.05)))
        else:
            acc = max(0, min(1, random.gauss(0.40, 0.05)))  # the real, injected drift
        points.append(MetricSeriesRow(
            run_id=f"synthetic-{i}",
            timestamp=base_time + timedelta(hours=i),
            provider="synthetic", model_id="test-model",
            accuracy=acc,
            format_adherence=max(0, min(1, random.gauss(0.95, 0.03))),  # stays stable
            refusal_rate=max(0, min(1, random.choice([0.0, 0.0, 0.0, 0.2]))),  # stays stable (lumpy)
            latency_p95_ms=random.gauss(400, 40),  # stays stable
            n_calls=20,
        ))
    return points, base_time


def main():
    section("PART 1: Synthetic scenario -- known injected drift in 'accuracy' only")

    points, base_time = build_synthetic_scenario()
    quiet_points = points[:30]  # calibrate ONLY on the quiet portion, as would happen in reality

    print("Calibrating from the quiet 30 points...")
    calibration_result = cal.calibrate(quiet_points, "synthetic", "test-model", k=3.0)
    for name, mt_thr in calibration_result.thresholds.items():
        print(f"  {name}: baseline_threshold={mt_thr.baseline_threshold:.4f}, "
              f"pelt_penalty={mt_thr.pelt_penalty:.6f}, ph_delta={mt_thr.page_hinkley_delta:.4f}, "
              f"ph_threshold={mt_thr.page_hinkley_threshold:.4f}")

    section("PART 2: PELT + bootstrap significance (run on the FULL 60-point series)")

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
            # take the most significant adverse candidate as this metric's "primary" result for this run
            best = min(sig_results, key=lambda r: r.p_value)
            bootstrap_results[metric_name] = best
            print(f"  most significant: index={best.candidate.index} delta={best.observed_delta:.4f} "
                  f"raw_p={best.p_value:.4f}")

    section("PART 3: Benjamini-Hochberg correction across all metrics tested this run")

    if bootstrap_results:
        corrected = mt.apply_bh_correction(bootstrap_results, alpha=0.05)
        for name, cr in corrected.items():
            flag = "SIGNIFICANT" if cr.significant else "not significant"
            print(f"  {name}: raw_p={cr.result.p_value:.4f} -> corrected_p={cr.p_value_corrected:.4f} [{flag}]")

        real_drift_flagged = corrected.get("accuracy") and corrected["accuracy"].significant
        false_positives = [name for name, cr in corrected.items() if cr.significant and name != "accuracy"]
        print(f"\n  Real injected drift (accuracy) correctly flagged: {real_drift_flagged}")
        print(f"  False positives on unchanged metrics: {false_positives if false_positives else 'none'}")
    else:
        print("  No adverse candidates were found on any metric.")

    section("PART 4: Page-Hinkley (live/sequential) on the same accuracy series")

    ordered = sorted(points, key=lambda p: p.timestamp)
    acc_values = [p.accuracy for p in ordered]
    acc_timestamps = [p.timestamp for p in ordered]
    acc_thr = calibration_result.thresholds["accuracy"]

    ph_alerts = run_sequential_detection(
        acc_values, acc_timestamps, "accuracy", "lower_is_bad",
        delta=acc_thr.page_hinkley_delta, threshold=acc_thr.page_hinkley_threshold,
    )
    print(f"  {len(ph_alerts)} alert(s) fired. First alert at index: "
          f"{ph_alerts[0].index if ph_alerts else 'none'} (drift actually starts at index 30)")

    section("PART 5: Naive baseline (comparison point) on the same accuracy series")

    offset = abs(acc_thr.baseline_threshold - acc_thr.based_on.median)
    baseline_alerts = run_baseline_detection(acc_values, acc_timestamps, "accuracy", "lower_is_bad", offset=offset)
    print(f"  {len(baseline_alerts)} alert(s) fired out of 30 post-drift points.")
    print("  (Expect this to be noticeably fewer/quieter over time than Page-Hinkley's persistent")
    print("   detection -- the baseline's rolling window tends to absorb sustained drift as 'normal'.)")

    section("PART 6: Where your REAL data currently stands")

    try:
        store = DriftStore(db_path="driftguard.db")
        real_series = store.get_metric_series("groq", "openai/gpt-oss-120b")
        print(f"  Real quiet-window points collected so far: {len(real_series)}")
        if len(real_series) < cal.MIN_QUIET_POINTS:
            print(f"  Need >= {cal.MIN_QUIET_POINTS} for calibration to be considered reliable "
                  f"-- keep the runner going to accumulate more.")
        else:
            print("  You have enough real data for a first real calibration attempt!")
    except Exception as e:
        print(f"  Could not read driftguard.db: {e}")

    section("DONE")
    print("If PART 3 showed the real injected drift flagged significant and no/few false positives,")
    print("and PART 4 fired quickly near index 30, the detection layer is working correctly together.")


if __name__ == "__main__":
    main()