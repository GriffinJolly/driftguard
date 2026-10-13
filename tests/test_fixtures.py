"""
Phase 0 checks: the hand-written events match the real database contract.

If these pass, the fixtures can be saved to and read back from a real
DriftStore exactly like events written by Person 1's detectors, so every
later phase can trust them.

Run with:  pytest tests/test_fixtures.py -v
"""

from __future__ import annotations

import pytest

from driftguard.storage.store import DriftStore
from tests.fixtures import events as fx


@pytest.fixture
def store(tmp_path):
    """A brand-new, empty database file in a temporary folder, deleted after each test."""
    return DriftStore(db_path=str(tmp_path / "test.db"))


def test_every_fixture_has_an_expected_answer():
    assert len(fx.ALL_FIXTURES) == 11
    for make in fx.ALL_FIXTURES:
        assert make in fx.EXPECTED_LABEL


def test_factories_return_fresh_objects():
    # Saving an event changes it in place, so two calls must never share one object.
    assert fx.refusal_up() is not fx.refusal_up()


def test_events_save_and_get_ids(store):
    saved = fx.load_into_store(store)
    assert len(saved) == 11
    assert all(e.id is not None for e in saved)


def test_open_events_are_significant_and_unreviewed(store):
    fx.load_into_store(store)
    open_events = store.get_open_events()
    # Only failed_correction has significant=False, so 10 of the 11 are "open".
    assert len(open_events) == 10
    assert all(e.significant and not e.reviewed for e in open_events)


def test_live_events_match_the_sequential_detector_shape():
    e = fx.format_down_live()
    assert e.detector == "sequential"
    assert e.pre_mean is None and e.p_value_raw is None and e.p_value_corrected is None
    assert e.significant is True


def test_metric_names_match_the_detectors():
    known = {"accuracy", "format_adherence", "refusal_rate", "latency_p95_ms"}
    names = {make().metric_name for make in fx.ALL_FIXTURES}
    assert names - known == {"latency_p50_ms"}  # the deliberate unknown_metric case


def test_incident_events_are_close_in_time_and_other_model_is_separate():
    a = fx.update_accuracy_down()
    f = fx.update_format_down()
    b = fx.other_model_latency_up()
    gap_minutes = (f.changepoint_timestamp - a.changepoint_timestamp).total_seconds() / 60
    assert gap_minutes == pytest.approx(40)
    assert b.model_id != a.model_id
