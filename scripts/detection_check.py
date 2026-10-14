import random
from datetime import datetime, timedelta, timezone

import pandas as pd

from driftguard.detection import baseline as cal
from driftguard.detection import pipeline
from driftguard.detection.changepoint_sequential import run_sequential_detection
from driftguard.detection.naive_detector import run_baseline_detection
from driftguard.storage.store import DriftStore, MetricSeriesRow


def line(title=""):
    print()
    print(title)
    print("-" * len(title) if title else "-" * 40)


def build_per_run_data(seed=11, n_quiet=30, n_drift=30):
    random.seed(seed)
    base_time = datetime.now(timezone.utc)
    points = []
    for i in range(n_quiet + n_drift):
        acc = max(0, min(1, random.gauss(0.65, 0.05))) if i < n_quiet else max(0, min(1, random.gauss(0.40, 0.05)))
        points.append(MetricSeriesRow(
            run_id=f"run-{i}", timestamp=base_time + timedelta(hours=i),
            provider="synthetic", model_id="test-model",
            accuracy=acc,
            format_adherence=max(0, min(1, random.gauss(0.95, 0.03))),
            refusal_rate=max(0, min(1, random.choice([0.0, 0.0, 0.0, 0.2]))),
            latency_p95_ms=random.gauss(400, 40),
            n_calls=20,
        ))
    return points


def build_per_call_data(seed=11, n_quiet=300, n_drift=300, baseline_error=0.05, drift_error=0.35):
    random.seed(seed)
    base_time = datetime.now(timezone.utc)
    quiet = [1 if random.random() < baseline_error else 0 for _ in range(n_quiet)]
    drift = [1 if random.random() < drift_error else 0 for _ in range(n_drift)]
    values = quiet + drift
    timestamps = [base_time + timedelta(minutes=i) for i in range(len(values))]
    return values, timestamps


def main():
    points = build_per_run_data()
    quiet_points = points[:30]

    line("1. Calibration (baseline.py)")
    calibration = cal.calibrate(quiet_points, "synthetic", "test-model", k=3.0)
    for name, t in calibration.thresholds.items():
        print(f"{name:<18} threshold={t.baseline_threshold:.4f}   pelt_penalty={t.pelt_penalty:.6f}")

    line("2-4. Detection pipeline on per-call data (pipeline.py: PELT + permutation + BH, Page-Hinkley)")
    values, timestamps = build_per_call_data()
    calls = pd.DataFrame({
        "seq_id": range(1, len(values) + 1), "timestamp": pd.to_datetime(timestamps, utc=True),
        "outcome": "ok", "latency_ms": 400.0, "eval_task_type": "accuracy", "eval_task_id": "q",
        "eval_score": [1.0 - v for v in values],
    })
    det = pipeline.detect(calls, "synthetic", "test-model")
    acc_result = det.results["accuracy_error"]
    for t in acc_result.tested:
        verdict = "SIGNIFICANT" if t["significant"] else "not significant"
        print(f"PELT candidate at call {t['index']}: error {t['pre_mean']:.3f} -> {t['post_mean']:.3f}   "
              f"p={t['p_value_raw']:.4f}  corrected_p={t['p_value_corrected']:.4f}   {verdict}")
    print(f"Page-Hinkley alarms at calls: {acc_result.alerts}")
    print(f"{len(det.events)} event(s) for the classification layer (true change at call 300)")

    line("5. Naive detector, for comparison (naive_detector.py)")
    acc = calibration.thresholds["accuracy"]
    ordered = sorted(points, key=lambda p: p.timestamp)
    acc_values = [p.accuracy for p in ordered]
    acc_timestamps = [p.timestamp for p in ordered]
    offset = abs(acc.baseline_threshold - acc.based_on.median)
    naive_alerts = run_baseline_detection(acc_values, acc_timestamps, "accuracy", "lower_is_bad", offset=offset)
    print(f"{len(naive_alerts)} alert(s) out of 30 post-drift points")

    line("6. Page-Hinkley, live detector (changepoint_sequential.py)")
    call_values, call_timestamps = build_per_call_data()
    alerts = run_sequential_detection(call_values, call_timestamps, "accuracy_error_rate")
    print(f"per-call data (correct input):  {len(alerts)} alert(s), first at call {alerts[0].index if alerts else '-'}")
    wrong_alerts = run_sequential_detection(acc_values, acc_timestamps, "accuracy")
    print(f"per-run data (wrong input):     {len(wrong_alerts)} alert(s)")

    line("7. Real data collected so far")
    try:
        store = DriftStore(db_path="driftguard.db")
        real_series = store.get_metric_series("groq", "openai/gpt-oss-120b")
        print(f"{len(real_series)} quiet-window points saved (need 20+ for calibration)")
    except Exception as e:
        print(f"could not read driftguard.db: {e}")

    print()


if __name__ == "__main__":
    main()