"""Dashboard data layer: per-call detection, handed to the classification layer
(classification/rules.py, incidents.py, remediation/mapper.py). No Streamlit needed."""

import numpy as np
import pandas as pd

from driftguard.classification import rules
from driftguard.dashboard import data as D
from driftguard.dashboard.signals import SIGNALS, to_metric_value
from driftguard.remediation import mapper


def _calls(n=800, onset=400, p0=0.05, p1=0.40, seed=0):
    rng = np.random.default_rng(seed)
    wrong = np.r_[rng.random(onset) < p0, rng.random(n - onset) < p1]
    return pd.DataFrame({
        "seq_id": np.arange(1, n + 1),
        "timestamp": pd.date_range("2026-09-01", periods=n, freq="min", tz="UTC"),
        "outcome": "ok", "eval_task_type": "accuracy", "eval_task_id": "q",
        "eval_score": np.where(wrong, 0.0, 1.0), "latency_ms": 800.0,
    })


def _analyse(calls):
    s = D.DetectionSettings()
    results = D.analyse(D.build_streams(calls, s.baseline_window), s)
    return results, D.classify_detections(results, "groq", "m", s)


def test_signals_map_only_onto_the_taxonomy():
    for sig in SIGNALS.values():
        assert sig.metric is None or sig.metric in rules.TAXONOMY
    # an error rate becomes accuracy in the metric's own (higher = better) units
    assert to_metric_value("accuracy_error", 0.2) == 0.8
    assert to_metric_value("safe_refusal", 0.2) == 0.2


def test_sudden_accuracy_drop_becomes_a_capability_incident():
    results, a = _analyse(_calls())
    acc = results["accuracy_error"]
    assert acc.alerts and acc.alerts[0] > 400                 # online alert after the onset
    kept = [t["index"] for t in acc.confirmed if t["significant"]]
    assert kept and abs(kept[0] - 400) <= 60                 # PELT locates it near the onset

    shifts = D.shift_table(a)
    adverse = shifts[shifts["adverse"] == True]                # noqa: E712
    assert set(adverse["drift_type"]) == {rules.CAPABILITY_SHIFT}
    pelt = adverse[adverse["detector"] == "PELT"].iloc[0]
    assert pelt["before"] > pelt["after"]                     # accuracy fell, in accuracy units
    # live + batch agree -> corroborated -> act now; a later alert after the detector's reset,
    # more than incidents.DEFAULT_WINDOW away, is its own early-warning incident
    assert a.plans[0].urgency == mapper.ACT_NOW
    assert all(p.urgency == mapper.WATCH for p in a.plans[1:])
    assert D.signal_status("accuracy_error", a) == mapper.ACT_NOW


def test_stable_stream_raises_nothing():
    _, a = _analyse(_calls(p1=0.05))
    assert not a.incidents
    assert D.signal_status("accuracy_error", a) == "clear"


def test_signals_outside_the_taxonomy_never_become_incidents():
    calls = _calls(p1=0.05)
    calls.loc[500:, "outcome"] = "rate_limited"              # our quota runs out halfway
    calls.loc[500:, "eval_score"] = np.nan
    results, a = _analyse(calls)
    assert "quota_exhausted" in results
    shifts = D.shift_table(a)
    quota = shifts[shifts["signal"] == "quota_exhausted"]
    assert len(quota) and quota["drift_type"].isna().all()
    assert D.signal_status("quota_exhausted", a) == "context"
    assert not a.incidents


def test_record_shifts_writes_open_events_with_mapper_plans(tmp_path):
    from driftguard.storage.store import DriftStore

    db = tmp_path / "t.db"
    DriftStore(db_path=str(db))                               # create the tables
    _, a = _analyse(_calls())
    added = D.record_shifts(db, a)
    assert added >= 2                                         # PELT + Page-Hinkley
    ev = D.load_events(db)
    assert set(ev["metric_name"]) == {"accuracy"}
    assert set(ev["drift_type"]) == {rules.CAPABILITY_SHIFT}
    assert ev["remediation"].str.startswith("[ACT NOW]").any()
    assert ev["remediation"].str.match(r"^\[(ACT NOW|WATCH)\]").all()
    assert D.record_shifts(db, a) == 0                        # no duplicates on a second click


def test_default_run_picks_latest_long_trial():
    a, b, c = _calls(n=200, onset=100), _calls(n=200, onset=100), _calls(n=40, onset=20)
    a["eval_run_id"], b["eval_run_id"], c["eval_run_id"] = "old", "new", "tiny"
    b["seq_id"] += 1000
    c["seq_id"] += 2000
    assert D.default_run(pd.concat([a, b, c])) == "new"
