"""
Phase 1 checks for driftguard/classification/rules.py.

Every test follows the same pattern: take a hand-written event from
tests/fixtures/events.py, run it through rules.py, and assert that the
answer matches the one decided in advance.

Run with:  python -m pytest tests/test_rules.py -v
"""

from __future__ import annotations

import pytest

from driftguard.classification import rules
from driftguard.storage.store import DriftStore
from tests.fixtures import events as fx

# A made-up calibration result: each metric's normal spread.
# Accuracy normally wobbles by 0.02; format adherence never moved (0 spread).
SIGMAS = {
    "accuracy": 0.02,
    "format_adherence": 1e-6,
    "refusal_rate": 0.015,
    "latency_p95_ms": 400.0,
}


# ---------------------------------------------------------------------------
# 1. The label: the four-way taxonomy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("make", fx.ALL_FIXTURES, ids=lambda f: f.__name__)
def test_every_fixture_gets_its_expected_label(make):
    """Runs once per practice event: rules.py must give the pre-decided answer."""
    result = rules.classify_event(make())
    assert result.drift_type == fx.EXPECTED_LABEL[make]


def test_taxonomy_agrees_with_person_1_directions():
    """If Person 1 ever changes which direction is 'bad' for a metric, this fails loudly."""
    from driftguard.detection.baseline import METRIC_DIRECTIONS
    assert {name: rule.bad_direction for name, rule in rules.TAXONOMY.items()} == METRIC_DIRECTIONS


def test_improvement_is_not_drift():
    result = rules.classify_event(fx.accuracy_up())
    assert result.adverse is False
    assert result.drift_type is None
    assert "healthy direction" in result.reason


def test_unknown_metric_is_left_alone_without_crashing():
    result = rules.classify_event(fx.unknown_metric())
    assert result.drift_type is None
    assert result.adverse is None
    assert "outside the four-way taxonomy" in result.reason


def test_pelt_event_missing_its_averages_is_not_labelled():
    event = fx.accuracy_down()
    event.pre_mean = None
    result = rules.classify_event(event)
    assert result.adverse is None
    assert result.drift_type is None


# ---------------------------------------------------------------------------
# 2. Evidence tier
# ---------------------------------------------------------------------------

def test_live_alert_is_an_early_warning():
    assert rules.evidence_tier(fx.format_down_live()) == rules.EARLY_WARNING


def test_corrected_and_significant_is_confirmed():
    assert rules.evidence_tier(fx.refusal_up()) == rules.CONFIRMED


def test_significant_before_correction_is_provisional():
    event = fx.refusal_up()
    event.p_value_corrected = None  # Benjamini-Hochberg has not run yet
    assert rules.evidence_tier(event) == rules.PROVISIONAL


def test_failed_correction_keeps_its_label_but_is_unconfirmed():
    """The label says WHAT moved; the tier says how sure we are. They are separate."""
    result = rules.classify_event(fx.failed_correction())
    assert result.drift_type == rules.CAPABILITY_SHIFT
    assert result.evidence_tier == rules.UNCONFIRMED


# ---------------------------------------------------------------------------
# 3. Size in noise units
# ---------------------------------------------------------------------------

def test_noise_units_is_change_divided_by_normal_spread():
    # accuracy 0.88 -> 0.71 is a change of 0.17; normal spread 0.02; 0.17 / 0.02 = 8.5
    result = rules.classify_event(fx.accuracy_down(), SIGMAS)
    assert result.change == pytest.approx(-0.17)
    assert result.noise_units == pytest.approx(8.5)


def test_no_noise_units_without_calibration():
    assert rules.classify_event(fx.accuracy_down()).noise_units is None


def test_no_noise_units_for_live_alerts():
    assert rules.classify_event(fx.format_down_live(), SIGMAS).noise_units is None


def test_no_noise_units_when_metric_never_varied():
    """Format adherence was perfectly constant during calibration: dividing by ~0 would be meaningless."""
    result = rules.classify_event(fx.update_format_down(), SIGMAS)
    assert result.drift_type == rules.FORMAT_DRIFT
    assert result.noise_units is None


def test_sigmas_read_from_a_calibration_file_use_person_1_rule():
    calibration = {
        "thresholds": {
            "accuracy": {"based_on": {
                "metric_name": "accuracy", "n": 30, "mean": 0.86, "std": 0.03, "median": 0.86,
                "mad": 0.0135, "mad_scaled": 0.02, "p05": 0.82, "p95": 0.9, "min_val": 0.8, "max_val": 0.92,
            }},
            "format_adherence": {"based_on": {
                "metric_name": "format_adherence", "n": 30, "mean": 1.0, "std": 0.0, "median": 1.0,
                "mad": 0.0, "mad_scaled": 0.0, "p05": 1.0, "p95": 1.0, "min_val": 1.0, "max_val": 1.0,
            }},
        }
    }
    sigmas = rules.noise_sigmas_from_calibration(calibration)
    assert sigmas["accuracy"] == pytest.approx(0.02)   # uses mad_scaled when it is above zero
    assert sigmas["format_adherence"] == pytest.approx(1e-6)  # baseline.py's floor for a constant metric


# ---------------------------------------------------------------------------
# 4. The sentence shown to people
# ---------------------------------------------------------------------------

def test_reason_reads_like_a_sentence():
    reason = rules.classify_event(fx.accuracy_down(), SIGMAS).reason
    assert reason == (
        "Accuracy fell from 0.88 to 0.71 (8.5 noise units): capability shift. "
        "Evidence: confirmed (corrected p = 0.006)."
    )


def test_latency_reason_uses_milliseconds():
    reason = rules.classify_event(fx.latency_up()).reason
    assert "5,200 ms" in reason and "9,800 ms" in reason


# ---------------------------------------------------------------------------
# 5. Writing labels into the database
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return DriftStore(db_path=str(tmp_path / "test.db"))


def test_label_event_fills_in_drift_type():
    event = fx.refusal_up()
    assert event.drift_type is None
    rules.label_event(event)
    assert event.drift_type == rules.SAFETY_SHIFT


def test_classify_open_events_saves_labels(store):
    fx.load_into_store(store)
    results = rules.classify_open_events(store)
    assert len(results) == 10  # failed_correction is not significant, so not "open"

    saved = {e.id: e.drift_type for e in store.get_open_events()}
    for r in results:
        assert saved[r.event_id] == r.drift_type


def test_classify_open_events_does_not_overwrite_a_human_label(store):
    event = fx.refusal_up()
    event.drift_type = "format_drift"  # pretend a reviewer set this by hand
    fx.load_into_store(store, [event])
    assert rules.classify_open_events(store) == []
    assert store.get_open_events()[0].drift_type == "format_drift"
