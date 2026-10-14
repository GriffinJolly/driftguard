"""End to end: calls in log_rows -> per-call detection (pipeline.py) -> the classification
layer (rules.py / incidents.py / mapper.py) -> labelled events in changepoint_events."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from driftguard.classification import rules
from driftguard.detection import pipeline as P
from driftguard.detection.signals import SIGNALS, to_metric_value
from driftguard.ingest.log_schema import CallOutcome, DriftLogEntry, IntegrationPath
from driftguard.storage.store import DriftStore

T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)


def _store(tmp_path, n=800, onset=400, p0=0.05, p1=0.40, task="accuracy", seed=0, outcomes=None):
    rng = np.random.default_rng(seed)
    bad = np.r_[rng.random(onset) < p0, rng.random(n - onset) < p1]
    store = DriftStore(db_path=str(tmp_path / "t.db"))
    for i in range(n):
        outcome = outcomes[i] if outcomes is not None else CallOutcome.OK
        score = None if outcome != CallOutcome.OK else (0.0 if bad[i] else 1.0)
        if task == "refusal" and score is not None:
            score = 1.0 - score                                  # 1 = refused, which is the bad outcome
        store.write_log(DriftLogEntry(
            integration_path=IntegrationPath.EVAL_SUITE, provider="groq", model_id="m", prompt="q",
            response_text="a", outcome=outcome, latency_ms=800.0, eval_run_id="r",
            eval_task_type=task, eval_task_id=f"{task}_safe_{i}" if task == "refusal" else f"q{i}",
            eval_score=score, timestamp=T0 + timedelta(minutes=i)))
    return store


def _events(store):
    return pd.read_sql_query("SELECT * FROM changepoint_events", store.engine)


def test_signals_map_only_onto_the_taxonomy():
    for sig in SIGNALS.values():
        assert sig.metric is None or sig.metric in rules.TAXONOMY
    assert to_metric_value("accuracy_error", 0.2) == pytest.approx(0.8)   # error rate -> accuracy
    assert to_metric_value("safe_refusal", 0.2) == pytest.approx(0.2)


def test_accuracy_drop_is_detected_and_labelled_capability(tmp_path):
    store = _store(tmp_path)
    report = P.run_detection(store, "groq", "m")
    acc = report.detection.results["accuracy_error"]
    kept = [t["index"] for t in acc.tested if t["significant"]]
    assert kept and abs(kept[0] - 400) <= 60                  # PELT places it near the true change
    assert acc.alerts and acc.alerts[0] > 400                 # Page-Hinkley notices it after the change

    ev = _events(store)
    open_ev = ev[ev["significant"] == 1]
    assert set(open_ev["metric_name"]) == {"accuracy"}
    assert set(open_ev["drift_type"]) == {rules.CAPABILITY_SHIFT}
    pelt = open_ev[open_ev["detector"] == "pelt"].iloc[0]
    assert pelt["pre_mean"] > pelt["post_mean"]               # accuracy, in its own units, fell
    assert open_ev["remediation"].str.startswith("[ACT NOW]").any()   # live + batch agree


def test_refusal_rise_is_labelled_safety(tmp_path):
    store = _store(tmp_path, task="refusal")
    P.run_detection(store, "groq", "m")
    ev = _events(store)
    assert set(ev.loc[ev["significant"] == 1, "drift_type"]) == {rules.SAFETY_SHIFT}


def test_stable_calls_produce_no_open_event(tmp_path):
    store = _store(tmp_path, p1=0.05)
    P.run_detection(store, "groq", "m")
    ev = _events(store)
    assert ev.empty or not (ev["significant"] == 1).any()


def test_quota_exhaustion_is_never_labelled(tmp_path):
    outcomes = [CallOutcome.OK] * 500 + [CallOutcome.RATE_LIMITED] * 300
    store = _store(tmp_path, p1=0.05, outcomes=outcomes)
    report = P.run_detection(store, "groq", "m")
    assert "quota_exhausted" in report.detection.results
    ev = _events(store)
    quota = ev[ev["metric_name"] == "quota_exhausted"]
    assert len(quota) and quota["drift_type"].isna().all()


def test_rerun_adds_nothing_twice(tmp_path):
    store = _store(tmp_path)
    first = P.run_detection(store, "groq", "m")
    second = P.run_detection(store, "groq", "m")
    assert first.written > 0 and second.written == 0


# --------------------------------------------------------------------------- noise-scaled Page-Hinkley

def test_noise_scale_only_raises_the_threshold():
    s = P.DetectionSettings(ph_noise_ref=0.05)
    assert P.ph_noise_scale(np.zeros(100), s) == 1.0                 # quieter than the tuning stream
    assert P.ph_noise_scale(np.r_[np.ones(5), np.zeros(95)], s) == pytest.approx(1.0)
    noisy = P.ph_noise_scale(np.r_[np.ones(30), np.zeros(70)], s)     # 30% bad: ~2.1x the noise
    assert noisy == pytest.approx(np.sqrt(0.3 * 0.7 / (0.05 * 0.95)))
    assert P.ph_noise_scale(np.ones(100), P.DetectionSettings()) == 1.0   # scaling off


def test_scaled_page_hinkley_waits_for_the_baseline_window():
    s = P.DetectionSettings(ph_noise_ref=0.05, baseline_window=100)
    v = np.r_[np.zeros(150), np.ones(100)]                           # bad from call 150 onwards
    alerts = P.page_hinkley_alerts(v, s)
    assert alerts and min(alerts) >= 150
    early = np.r_[np.zeros(40), np.ones(60)]                         # inside the baseline: no alarm
    assert P.page_hinkley_alerts(early, s) == []


def test_noise_scaling_stops_alarms_on_a_noisy_stable_signal():
    rng = np.random.default_rng(1)
    stable = (rng.random(10_000) < 0.30).astype(float)               # 30% wrong, never changes
    tuned_only = P.DetectionSettings(ph_alpha=0.9999)
    scaled = P.DetectionSettings(ph_alpha=0.998, ph_noise_ref=0.05)
    assert len(P.page_hinkley_alerts(stable, tuned_only)) > 0        # the old setting alarms on noise
    assert P.page_hinkley_alerts(stable, scaled) == []
    shifted = np.r_[stable[:5000], (rng.random(5000) < 0.45).astype(float)]
    alerts = P.page_hinkley_alerts(shifted, scaled)
    assert alerts and 5000 <= alerts[0] < 6500                       # a real 15-point rise is still caught
