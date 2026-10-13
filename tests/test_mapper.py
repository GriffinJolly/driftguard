"""
Phase 3 checks for driftguard/remediation/mapper.py (what to do).

Run with:  python -m pytest tests/test_mapper.py -v
"""

from __future__ import annotations

import pytest

from driftguard.classification import rules
from driftguard.classification.incidents import CORROBORATED, group_into_incidents
from driftguard.remediation import mapper
from driftguard.storage.store import DriftStore
from tests.fixtures import events as fx


def plan(*events):
    return mapper.plan_for_incident(group_into_incidents(list(events))[0])


# ---------------------------------------------------------------------------
# Every label has advice
# ---------------------------------------------------------------------------

def test_every_drift_type_has_advice():
    labels = {rule.drift_type for rule in rules.TAXONOMY.values()}
    assert set(mapper.ADVICE) == labels
    assert all(len(a.checks) >= 2 for a in mapper.ADVICE.values())


def test_every_tier_has_an_urgency_and_every_urgency_a_step():
    for tier in [rules.UNCONFIRMED, rules.EARLY_WARNING, rules.PROVISIONAL, rules.CONFIRMED, CORROBORATED]:
        assert mapper.URGENCY_BY_TIER[tier] in mapper.URGENCY_STEP


# ---------------------------------------------------------------------------
# Urgency follows the evidence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "make, expected",
    [
        (fx.failed_correction, mapper.NOTE),
        (fx.format_down_live, mapper.WATCH),
        (fx.refusal_up, mapper.ACT),
    ],
    ids=["unconfirmed->note", "early_warning->watch", "confirmed->act"],
)
def test_urgency_matches_evidence(make, expected):
    assert plan(make()).urgency == expected


def test_provisional_means_investigate():
    event = fx.refusal_up()
    event.p_value_corrected = None
    assert plan(event).urgency == mapper.INVESTIGATE


def test_corroborated_incident_means_act_now():
    p = plan(fx.update_accuracy_down(), fx.update_format_down(), fx.update_accuracy_live())
    assert p.urgency == mapper.ACT_NOW


# ---------------------------------------------------------------------------
# The steps
# ---------------------------------------------------------------------------

def test_urgency_step_comes_first_then_the_checks():
    p = plan(fx.refusal_up())
    assert p.steps[0] == mapper.URGENCY_STEP[mapper.ACT]
    assert p.steps[1:] == mapper.ADVICE[rules.SAFETY_SHIFT].checks


def test_multi_signal_plan_includes_checks_for_each_signal_once():
    p = plan(fx.update_accuracy_down(), fx.update_format_down(), fx.update_accuracy_live())
    for check in mapper.ADVICE[rules.CAPABILITY_SHIFT].checks + mapper.ADVICE[rules.FORMAT_DRIFT].checks:
        assert p.steps.count(check) == 1
    assert len(p.meanings) == 2


def test_one_line_version():
    assert plan(fx.format_down_live()).one_line.startswith("[WATCH] Run the eval suite again now")


# ---------------------------------------------------------------------------
# Possible explanations
# ---------------------------------------------------------------------------

def test_single_signal_has_no_explanations():
    assert mapper.possible_explanations([rules.CAPABILITY_SHIFT]) == []


def test_capability_and_format_suggest_a_model_swap():
    hints = mapper.possible_explanations([rules.CAPABILITY_SHIFT, rules.FORMAT_DRIFT])
    assert len(hints) == 1 and "swapping the model" in hints[0]


def test_latency_is_ruled_out_first():
    hints = mapper.possible_explanations([rules.CAPABILITY_SHIFT, rules.FORMAT_DRIFT, rules.INFRA_CHANGE])
    assert hints[0].startswith("Latency moved too")
    assert not any("swapping the model" in h for h in hints)  # latency moved, so not "steady"


def test_safety_and_capability_suggest_safety_tuning():
    hints = mapper.possible_explanations([rules.SAFETY_SHIFT, rules.CAPABILITY_SHIFT])
    assert any("safety-tuning" in h for h in hints)


def test_unmatched_combination_gets_a_general_hint():
    hints = mapper.possible_explanations([rules.SAFETY_SHIFT, rules.FORMAT_DRIFT])
    assert hints == ["Several signals moved together; treat them as one cause until the checks show otherwise."]


def test_explanations_are_never_stated_as_certain():
    for combo in [
        [rules.CAPABILITY_SHIFT, rules.FORMAT_DRIFT],
        [rules.CAPABILITY_SHIFT, rules.INFRA_CHANGE],
        [rules.SAFETY_SHIFT, rules.CAPABILITY_SHIFT],
    ]:
        for hint in mapper.possible_explanations(combo):
            assert "consistent with" in hint or "Rule out" in hint


# ---------------------------------------------------------------------------
# Writing to the database
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return DriftStore(db_path=str(tmp_path / "test.db"))


def test_apply_remediation_fills_both_columns(store):
    fx.load_into_store(store)
    updated = mapper.apply_remediation(store)
    assert updated == 8  # 10 open events minus accuracy_up and unknown_metric (no label)
    labelled = [e for e in store.get_open_events() if e.drift_type is not None]
    assert len(labelled) == 8
    assert all(e.remediation and e.remediation.startswith("[") for e in labelled)


def test_corroboration_raises_urgency_for_every_event_in_the_incident(store):
    fx.load_into_store(store, [fx.update_accuracy_down(), fx.update_format_down(), fx.update_accuracy_live()])
    mapper.apply_remediation(store)
    assert all(e.remediation.startswith("[ACT NOW]") for e in store.get_open_events())


def test_apply_remediation_keeps_a_reviewer_note(store):
    event = fx.refusal_up()
    event.remediation = "Checked with provider: known issue, fix due Friday."
    fx.load_into_store(store, [event])
    mapper.apply_remediation(store)
    assert store.get_open_events()[0].remediation == "Checked with provider: known issue, fix due Friday."
