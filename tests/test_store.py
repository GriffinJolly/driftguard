"""Unit tests for driftguard/storage/store.py -- round-trip writes/reads
against a throwaway SQLite file (tmp_path), no network."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from driftguard.ingest.log_schema import DriftLogEntry, IntegrationPath, MetricPoint
from driftguard.storage.store import ChangePointEventRow, DriftStore


def _entry(**kw) -> DriftLogEntry:
    base = dict(
        integration_path=IntegrationPath.EVAL_SUITE,
        provider="Groq",  # exercises the provider lower-casing validator
        model_id="m1",
        prompt="p",
    )
    base.update(kw)
    return DriftLogEntry(**base)


@pytest.fixture
def store(tmp_path):
    return DriftStore(db_path=str(tmp_path / "t.db"))


def test_write_log_and_get_logs_for_run(store):
    e1 = _entry(eval_run_id="run1", eval_score=1.0, request_params={"temperature": 0})
    e2 = _entry(eval_run_id="run1", eval_score=0.0)
    e3 = _entry(eval_run_id="run2")
    for e in (e1, e2, e3):
        store.write_log(e)

    rows = store.get_logs_for_run("run1")
    assert {r.log_id for r in rows} == {e1.log_id, e2.log_id}
    assert all(r.provider == "groq" for r in rows)  # validator lowercased "Groq"
    got = {r.log_id: r for r in rows}
    assert got[e1.log_id].request_params_json == '{"temperature": 0}'
    assert store.get_logs_for_run("does_not_exist") == []


def test_metric_series_ordered_and_filtered(store):
    ts = datetime(2024, 1, 1)
    store.write_metric_point(MetricPoint(run_id="r2", timestamp=ts + timedelta(hours=1),
                                         provider="groq", model_id="m1", accuracy=0.8, n_calls=20))
    store.write_metric_point(MetricPoint(run_id="r1", timestamp=ts,
                                         provider="groq", model_id="m1", accuracy=0.9, n_calls=20))
    store.write_metric_point(MetricPoint(run_id="r3", timestamp=ts,
                                         provider="groq", model_id="OTHER", accuracy=0.1, n_calls=1))

    series = store.get_metric_series("groq", "m1")
    assert [round(s.accuracy, 1) for s in series] == [0.9, 0.8]  # timestamp-ordered, model-filtered
    assert store.get_metric_series("groq", "missing") == []


def test_open_events_only_significant_and_unreviewed(store):
    now = datetime(2024, 1, 1)

    def _event(**kw):
        base = dict(detected_at=now, provider="groq", model_id="m",
                    metric_name="accuracy", detector="pelt")
        base.update(kw)
        return ChangePointEventRow(**base)

    open_evt = _event(significant=True, reviewed=False)
    reviewed = _event(significant=True, reviewed=True)
    insignificant = _event(significant=False, reviewed=False)
    for e in (open_evt, reviewed, insignificant):
        store.write_changepoint_event(e)

    opens = store.get_open_events()
    assert len(opens) == 1
    assert opens[0].significant and not opens[0].reviewed


def test_write_changepoint_event_assigns_id(store):
    saved = store.write_changepoint_event(
        ChangePointEventRow(detected_at=datetime(2024, 1, 1), provider="groq",
                            model_id="m", metric_name="accuracy", detector="sequential")
    )
    assert saved.id is not None


def test_get_exemplars_respects_window_and_limit(store):
    base = datetime(2024, 1, 1, 12, 0, 0)
    for i in range(5):
        store.write_log(_entry(timestamp=base + timedelta(minutes=i)))

    window = store.get_exemplars("groq", "m1", base + timedelta(minutes=1), base + timedelta(minutes=3))
    assert len(window) == 3  # minutes 1, 2, 3

    assert len(store.get_exemplars("groq", "m1", base, base + timedelta(minutes=10), limit=2)) == 2
