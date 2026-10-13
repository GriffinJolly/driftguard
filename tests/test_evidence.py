"""
Phase 3 checks for driftguard/remediation/evidence.py (the flip set).

Each test writes a small eval history with known per-question results
(tests/fixtures/logs.py), then checks the flip set finds exactly the
questions that were designed to change.

Run with:  python -m pytest tests/test_evidence.py -v
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from driftguard.classification.incidents import group_into_incidents
from driftguard.remediation import evidence as ev
from driftguard.storage.store import DriftStore
from tests.fixtures import events as fx
from tests.fixtures.logs import Task, write_eval_history


@pytest.fixture
def store(tmp_path):
    return DriftStore(db_path=str(tmp_path / "test.db"))


# Four MMLU questions: two break at the change point, one is always right,
# one was wrong once by chance before (must NOT count as a flip).
ACCURACY_TASKS = [
    Task("mmlu_a", "accuracy", before=1.0, after=0.0),
    Task("mmlu_b", "accuracy", before=1.0, after=[1.0, 0.0, 0.0, 0.0, 0.0]),
    Task("mmlu_c", "accuracy", before=1.0, after=1.0),
    Task("mmlu_d", "accuracy", before=[1.0, 0.0, 1.0, 1.0, 1.0], after=1.0),
]


# ---------------------------------------------------------------------------
# Choosing the runs
# ---------------------------------------------------------------------------

def test_runs_around_takes_n_each_side(store):
    steady = [Task("mmlu_a", "accuracy", before=1.0, after=0.0)]  # same score every run, so any run count works
    change = write_eval_history(store, steady, n_before=8, n_after=6)
    before, after = ev.runs_around(store, fx.PROVIDER, fx.MODEL_A, change, n_runs=5)
    assert len(before) == 5 and len(after) == 5
    assert all(r.timestamp < after[0].timestamp for r in before)


# ---------------------------------------------------------------------------
# Accuracy flips
# ---------------------------------------------------------------------------

def test_flip_set_finds_exactly_the_broken_questions(store):
    change = write_eval_history(store, ACCURACY_TASKS)
    fs = ev.flip_set(store, fx.PROVIDER, fx.MODEL_A, "accuracy", change)
    assert fs.n_tasks_compared == 4
    assert [f.task_id for f in fs.flips] == ["mmlu_a", "mmlu_b"]  # mmlu_a moved more, so it is first
    assert all(f.worse for f in fs.flips)


def test_one_wrong_answer_by_chance_is_not_a_flip(store):
    change = write_eval_history(store, ACCURACY_TASKS)
    fs = ev.flip_set(store, fx.PROVIDER, fx.MODEL_A, "accuracy", change)
    assert "mmlu_d" not in [f.task_id for f in fs.flips]  # 0.8 -> 1.0 is a move of 0.2, below 0.5


def test_flip_description_and_examples(store):
    change = write_eval_history(store, ACCURACY_TASKS)
    flip = ev.flip_set(store, fx.PROVIDER, fx.MODEL_A, "accuracy", change).flips[1]  # mmlu_b
    assert flip.description == "correct in 5/5 runs before, 1/5 after"
    assert flip.response_before.startswith("before answer 4")   # last answer before the change
    assert flip.response_after.startswith("after answer 0")     # first answer after it


def test_summary_sentence(store):
    change = write_eval_history(store, ACCURACY_TASKS)
    fs = ev.flip_set(store, fx.PROVIDER, fx.MODEL_A, "accuracy", change)
    assert fs.summary == (
        "2 of 4 questions changed: 2 got worse, 0 got better "
        "(last 5 runs before vs first 5 after)."
    )


def test_improvements_are_listed_after_problems(store):
    tasks = [
        Task("mmlu_up", "accuracy", before=0.0, after=1.0),
        Task("mmlu_down", "accuracy", before=1.0, after=0.0),
    ]
    change = write_eval_history(store, tasks)
    fs = ev.flip_set(store, fx.PROVIDER, fx.MODEL_A, "accuracy", change)
    assert [f.task_id for f in fs.flips] == ["mmlu_down", "mmlu_up"]
    assert [f.task_id for f in fs.better] == ["mmlu_up"]


def test_failed_calls_without_a_score_are_ignored(store):
    tasks = [Task("mmlu_a", "accuracy", before=1.0, after=[None, None, 1.0, 1.0, 1.0])]
    change = write_eval_history(store, tasks)
    fs = ev.flip_set(store, fx.PROVIDER, fx.MODEL_A, "accuracy", change)
    assert fs.flips == []  # the three scored runs after are all correct


# ---------------------------------------------------------------------------
# Refusal flips: direction depends on safe vs unsafe prompts
# ---------------------------------------------------------------------------

def test_safe_prompt_newly_refused_is_worse(store):
    change = write_eval_history(store, [Task("xstest_safe_1", "refusal", before=0.0, after=1.0)])
    flip = ev.flip_set(store, fx.PROVIDER, fx.MODEL_A, "refusal_rate", change).flips[0]
    assert flip.worse
    assert flip.description == "refused in 0/5 runs before, 5/5 after"


def test_unsafe_prompt_no_longer_refused_is_worse(store):
    """Safety loosening: the refusal_rate metric (safe prompts only) would never show this."""
    change = write_eval_history(store, [Task("xstest_unsafe_0", "refusal", before=1.0, after=0.0)])
    assert ev.flip_set(store, fx.PROVIDER, fx.MODEL_A, "refusal_rate", change).flips[0].worse


def test_unsafe_prompt_newly_refused_is_not_worse(store):
    change = write_eval_history(store, [Task("xstest_unsafe_0", "refusal", before=0.0, after=1.0)])
    assert not ev.flip_set(store, fx.PROVIDER, fx.MODEL_A, "refusal_rate", change).flips[0].worse


# ---------------------------------------------------------------------------
# Only the matching task type is compared
# ---------------------------------------------------------------------------

def test_format_flip_set_ignores_accuracy_questions(store):
    tasks = ACCURACY_TASKS + [Task("fmt_json_1", "format", before=1.0, after=0.0)]
    change = write_eval_history(store, tasks)
    fs = ev.flip_set(store, fx.PROVIDER, fx.MODEL_A, "format_adherence", change)
    assert fs.task_type == "format"
    assert fs.n_tasks_compared == 1
    assert [f.task_id for f in fs.flips] == ["fmt_json_1"]


# ---------------------------------------------------------------------------
# Not enough data
# ---------------------------------------------------------------------------

def test_no_flip_set_for_latency(store):
    change = write_eval_history(store, ACCURACY_TASKS)
    assert ev.flip_set(store, fx.PROVIDER, fx.MODEL_A, "latency_p95_ms", change) is None


def test_no_flip_set_without_runs_after_the_change(store):
    change = write_eval_history(store, ACCURACY_TASKS, n_after=0)
    assert ev.flip_set(store, fx.PROVIDER, fx.MODEL_A, "accuracy", change) is None


def test_no_flip_set_for_a_model_with_no_history(store):
    change = write_eval_history(store, ACCURACY_TASKS)
    assert ev.flip_set(store, fx.PROVIDER, fx.MODEL_B, "accuracy", change) is None


# ---------------------------------------------------------------------------
# Infra evidence
# ---------------------------------------------------------------------------

def test_infra_evidence_compares_latency_and_failures(store):
    tasks = [Task("mmlu_a", "accuracy", before=1.0, after=1.0)]
    change = write_eval_history(store, tasks, latency_before_ms=1500, latency_after_ms=6000, failed_calls_after=1)
    infra = ev.infra_evidence(store, fx.PROVIDER, fx.MODEL_A, change)
    assert infra.median_latency_before_ms == pytest.approx(1500)
    assert infra.median_latency_after_ms == pytest.approx(6000)
    assert infra.outcomes_after == {"ok": 5, "timeout": 5}
    assert infra.failure_rate_before == 0.0
    assert infra.failure_rate_after == pytest.approx(0.5)
    assert infra.summary == (
        "Median latency 1,500 ms -> 6,000 ms; failed calls (error, timeout, rate-limited) 0% -> 50% "
        "(last 5 runs before vs first 5 after)."
    )


# ---------------------------------------------------------------------------
# Evidence for a whole incident
# ---------------------------------------------------------------------------

def test_evidence_for_incident_gives_one_item_per_metric(store):
    tasks = ACCURACY_TASKS + [Task("fmt_json_1", "format", before=1.0, after=0.0)]
    change = write_eval_history(store, tasks)
    # Two events at the change time: accuracy and format both fell.
    acc = fx._event("accuracy", pre_mean=1.0, post_mean=0.5, p_value_corrected=0.01,
                    hours_after_base=(change - fx.BASE_TIME) / timedelta(hours=1))
    fmt = fx._event("format_adherence", pre_mean=1.0, post_mean=0.0, p_value_corrected=0.01,
                    hours_after_base=(change - fx.BASE_TIME) / timedelta(hours=1))
    incident = group_into_incidents([acc, fmt])[0]
    found = ev.evidence_for_incident(store, incident)
    assert set(found) == {"accuracy", "format_adherence"}
    assert len(found["accuracy"].flips) == 2
    assert len(found["format_adherence"].flips) == 1
