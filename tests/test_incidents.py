"""
Phase 2 checks for driftguard/classification/incidents.py.

Run with:  python -m pytest tests/test_incidents.py -v
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from driftguard.classification import incidents as inc
from driftguard.classification import rules
from driftguard.storage.store import DriftStore
from tests.fixtures import events as fx

SIGMAS = {"accuracy": 0.02, "format_adherence": 1e-6, "refusal_rate": 0.015, "latency_p95_ms": 400.0}


def the_model_update():
    """The simulated model update from the fixtures: three events on model A in one hour."""
    return [fx.update_accuracy_down(), fx.update_format_down(), fx.update_accuracy_live()]


# ---------------------------------------------------------------------------
# 1. Grouping
# ---------------------------------------------------------------------------

def test_model_update_becomes_one_incident():
    found = inc.group_into_incidents(the_model_update(), SIGMAS)
    assert len(found) == 1
    assert len(found[0].members) == 3


def test_other_model_in_the_same_hour_stays_separate():
    found = inc.group_into_incidents(the_model_update() + [fx.other_model_latency_up()], SIGMAS)
    assert len(found) == 2
    assert {i.model_id for i in found} == {fx.MODEL_A, fx.MODEL_B}


def test_all_fixtures_group_as_expected():
    # Model A has labelled events at hours 10, 20, 30, 40, 60 and 100 (the update trio): 6 incidents.
    # accuracy_up and unknown_metric have no label, so they are left out.
    # Model B has 1. Total 7.
    found = inc.group_into_incidents(fx.all_events(), SIGMAS)
    assert len(found) == 7
    assert sum(len(i.members) for i in found) == 9


def test_improvements_and_unknown_metrics_never_join_an_incident():
    assert inc.group_into_incidents([fx.accuracy_up(), fx.unknown_metric()]) == []


def test_event_exactly_at_the_window_edge_joins():
    first = fx._event("accuracy", hours_after_base=0, pre_mean=0.9, post_mean=0.7, p_value_corrected=0.01)
    edge = fx._event("format_adherence", hours_after_base=2, pre_mean=1.0, post_mean=0.8, p_value_corrected=0.01)
    assert len(inc.group_into_incidents([first, edge])) == 1


def test_event_just_past_the_window_starts_a_new_incident():
    first = fx._event("accuracy", hours_after_base=0, pre_mean=0.9, post_mean=0.7, p_value_corrected=0.01)
    late = fx._event("format_adherence", hours_after_base=2.01, pre_mean=1.0, post_mean=0.8, p_value_corrected=0.01)
    assert len(inc.group_into_incidents([first, late])) == 2


def test_window_is_adjustable():
    first = fx._event("accuracy", hours_after_base=0, pre_mean=0.9, post_mean=0.7, p_value_corrected=0.01)
    later = fx._event("format_adherence", hours_after_base=3, pre_mean=1.0, post_mean=0.8, p_value_corrected=0.01)
    assert len(inc.group_into_incidents([first, later], window=timedelta(hours=4))) == 1


def test_order_of_the_input_list_does_not_matter():
    shuffled = list(reversed(the_model_update()))
    found = inc.group_into_incidents(shuffled, SIGMAS)
    assert len(found) == 1
    assert found[0].first_channel == rules.CAPABILITY_SHIFT


def test_timestamps_with_and_without_timezone_can_be_mixed():
    """Older sqlmodel stores times without a timezone; newer requires one. Both must work together."""
    aware = fx.update_accuracy_down()
    naive = fx.update_format_down()
    naive.changepoint_timestamp = naive.changepoint_timestamp.replace(tzinfo=None)
    assert len(inc.group_into_incidents([aware, naive])) == 1


# ---------------------------------------------------------------------------
# 2. What the incident reports
# ---------------------------------------------------------------------------

def test_channels_first_and_largest():
    incident = inc.group_into_incidents(the_model_update(), SIGMAS)[0]
    assert incident.channels == [rules.CAPABILITY_SHIFT, rules.FORMAT_DRIFT]
    assert incident.is_multi_channel
    assert incident.first_channel == rules.CAPABILITY_SHIFT
    # accuracy 0.89 -> 0.74 = 7.5 noise units; format has no measurable noise, so accuracy is largest
    assert incident.largest.classification.drift_type == rules.CAPABILITY_SHIFT
    assert incident.largest.classification.noise_units == pytest.approx(7.5)


def test_largest_is_none_when_nothing_was_measurable():
    incident = inc.group_into_incidents(the_model_update())[0]  # no calibration given
    assert incident.largest is None


def test_start_end_and_id():
    incident = inc.group_into_incidents(the_model_update())[0]
    assert (incident.end - incident.start) == timedelta(minutes=40)
    assert incident.incident_id == "groq/openai/gpt-oss-120b@2026-09-24T16:00"


# ---------------------------------------------------------------------------
# 3. Evidence and corroboration
# ---------------------------------------------------------------------------

def test_live_plus_passing_batch_is_corroborated():
    incident = inc.group_into_incidents(the_model_update())[0]
    assert incident.corroborated_metrics == ["accuracy"]
    assert incident.metric_evidence("accuracy") == inc.CORROBORATED
    assert incident.metric_evidence("format_adherence") == rules.CONFIRMED
    assert incident.evidence == inc.CORROBORATED


def test_failed_batch_result_does_not_corroborate():
    weak = fx.failed_correction()  # PELT result that failed correction
    live = fx._event("accuracy", detector="sequential", hours_after_base=60.5, post_mean=1.0)
    incident = inc.group_into_incidents([weak, live])[0]
    assert incident.corroborated_metrics == []
    assert incident.evidence == rules.EARLY_WARNING  # the stronger of unconfirmed and early_warning


def test_live_alert_alone_is_an_early_warning_incident():
    incident = inc.group_into_incidents([fx.format_down_live()])[0]
    assert incident.evidence == rules.EARLY_WARNING


def test_tier_order_weakest_to_strongest():
    assert inc.tier_rank(rules.UNCONFIRMED) < inc.tier_rank(rules.EARLY_WARNING) \
        < inc.tier_rank(rules.PROVISIONAL) < inc.tier_rank(rules.CONFIRMED) < inc.tier_rank(inc.CORROBORATED)


# ---------------------------------------------------------------------------
# 4. Headlines
# ---------------------------------------------------------------------------

def test_headline_for_a_multi_signal_incident():
    incident = inc.group_into_incidents(the_model_update(), SIGMAS)[0]
    assert incident.headline == (
        "2 signals moved together within 40 minutes: capability shift, format drift. "
        "First: capability shift. Largest: capability shift (7.5 noise units). "
        "Evidence: corroborated (live and batch detectors agree on accuracy)."
    )


def test_headline_for_a_single_signal_incident():
    incident = inc.group_into_incidents([fx.refusal_up()])[0]
    assert incident.headline == "Safety shift on groq / openai/gpt-oss-120b. Evidence: confirmed."


# ---------------------------------------------------------------------------
# 5. Reading from the database
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return DriftStore(db_path=str(tmp_path / "test.db"))


def test_incidents_from_store(store):
    fx.load_into_store(store)
    found = inc.incidents_from_store(store, SIGMAS)
    # Same as test_all_fixtures_group_as_expected, minus failed_correction (not "open"): 6 incidents.
    assert len(found) == 6
    update = [i for i in found if i.is_multi_channel]
    assert len(update) == 1 and update[0].evidence == inc.CORROBORATED
