"""Detection + mapping layer behind the dashboard (no Streamlit needed)."""

import numpy as np
import pandas as pd
import pytest

from driftguard.classification.rules import SIGNALS, classify
from driftguard.dashboard import data as D
from driftguard.remediation.mapper import remediation_for


def _calls(n=800, onset=400, p0=0.05, p1=0.40, seed=0):
    rng = np.random.default_rng(seed)
    wrong = np.r_[rng.random(onset) < p0, rng.random(n - onset) < p1]
    return pd.DataFrame({
        "seq_id": np.arange(1, n + 1),
        "timestamp": pd.date_range("2026-09-01", periods=n, freq="min", tz="UTC"),
        "outcome": "ok", "eval_task_type": "accuracy", "eval_task_id": "q",
        "eval_score": np.where(wrong, 0.0, 1.0), "latency_ms": 800.0,
    })


def test_every_signal_has_a_category_and_action():
    for key in SIGNALS:
        assert classify(key) in ("Capability", "Safety", "Format", "Infrastructure")
        assert remediation_for(key) != remediation_for("unknown_signal")


def test_classify_rejects_unknown_signal():
    with pytest.raises(ValueError):
        classify("nope")


def test_sudden_accuracy_shift_is_found_by_both_paths():
    s = D.DetectionSettings()
    results = D.analyse(D.build_streams(_calls(), s.baseline_window), s)
    acc = results["accuracy_error"]
    assert acc.alerts and acc.alerts[0] > 400              # online alert comes after the onset
    kept = [t["index"] for t in acc.confirmed if t["significant"]]
    assert kept and abs(kept[0] - 400) <= 60              # PELT locates it near the onset
    shifts = D.shift_table(results)
    assert set(shifts["category"]) == {"Capability"}
    assert shifts["action"].str.len().gt(0).all()
    assert D.signal_status(acc) == "critical"


def test_stable_stream_raises_nothing():
    s = D.DetectionSettings()
    results = D.analyse(D.build_streams(_calls(p1=0.05), s.baseline_window), s)
    assert D.signal_status(results["accuracy_error"]) == "good"
    assert D.shift_table(results).empty


def test_rate_limits_are_client_side_not_drift():
    calls = _calls(p1=0.05)
    calls.loc[500:, "outcome"] = "rate_limited"          # our quota runs out halfway
    calls.loc[500:, "eval_score"] = np.nan
    s = D.DetectionSettings()
    results = D.analyse(D.build_streams(calls, s.baseline_window), s)
    assert D.signal_status(results["quota_exhausted"]) == "notice"
    assert "call_failure" not in results or D.signal_status(results["call_failure"]) == "good"
    shifts = D.shift_table(results)
    assert len(shifts) and not shifts["provider_drift"].any()


def test_default_run_picks_latest_long_trial():
    a, b, c = _calls(n=200, onset=100), _calls(n=200, onset=100), _calls(n=40, onset=20)
    a["eval_run_id"], b["eval_run_id"], c["eval_run_id"] = "old", "new", "tiny"
    b["seq_id"] += 1000
    c["seq_id"] += 2000
    assert D.default_run(pd.concat([a, b, c])) == "new"
