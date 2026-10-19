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
from tests.fixtures.logs import Task, write_call_trial, write_eval_history


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
    assert found["accuracy"].basis == ev.RUNS  # hourly runs -> compared run by run


# ===========================================================================
# Per-call trials: compare each question's ANSWERS, not runs
# ===========================================================================

# Ten passes before and after the change. mmlu_a and mmlu_b break, mmlu_c
# never changes, mmlu_d is wrong once before by chance (not a flip).
TRIAL_TASKS = [
    Task("mmlu_a", "accuracy", before=1.0, after=0.0),
    Task("mmlu_b", "accuracy", before=1.0, after=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    Task("mmlu_c", "accuracy", before=1.0, after=1.0),
    Task("mmlu_d", "accuracy", before=[1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], after=1.0),
]


def test_a_single_run_trial_has_no_run_based_flip_set(store):
    change = write_call_trial(store, TRIAL_TASKS)
    assert ev.flip_set(store, fx.PROVIDER, fx.MODEL_A, "accuracy", change) is None


def test_answer_based_flip_set_finds_exactly_the_broken_questions(store):
    change = write_call_trial(store, TRIAL_TASKS)
    fs = ev.flip_set_by_answers(store, fx.PROVIDER, fx.MODEL_A, "accuracy", change)
    assert fs.basis == ev.ANSWERS
    assert fs.n_tasks_compared == 4
    assert [f.task_id for f in fs.flips] == ["mmlu_a", "mmlu_b"]
    assert all(f.worse for f in fs.flips)


def test_answer_based_description_and_summary(store):
    change = write_call_trial(store, TRIAL_TASKS)
    fs = ev.flip_set_by_answers(store, fx.PROVIDER, fx.MODEL_A, "accuracy", change)
    assert fs.flips[1].description == "correct in 10/10 answers before, 2/10 after"   # mmlu_b
    assert fs.summary == (
        "2 of 4 questions changed: 2 got worse, 0 got better "
        "(last 10 answers per question before vs first 10 after)."
    )


def test_only_the_nearest_answers_on_each_side_are_used(store):
    # 15 passes before the change: only the last 10 answers per question count.
    tasks = [Task("mmlu_a", "accuracy", before=[0.0] * 5 + [1.0] * 10, after=0.0)]
    change = write_call_trial(store, tasks, passes_before=15, passes_after=12)
    flip = ev.flip_set_by_answers(store, fx.PROVIDER, fx.MODEL_A, "accuracy", change).flips[0]
    assert (flip.n_before, flip.n_after) == (10, 10)
    assert flip.rate_before == 1.0                       # the 5 old wrong answers are outside the window
    assert flip.response_before.startswith("before answer 14")
    assert flip.response_after.startswith("after answer 0")


def test_too_few_answers_after_the_change_means_no_judgement(store):
    change = write_call_trial(store, TRIAL_TASKS, passes_after=2)   # only 2 answers per question after
    assert ev.flip_set_by_answers(store, fx.PROVIDER, fx.MODEL_A, "accuracy", change) is None


def test_answer_based_unsafe_prompt_no_longer_refused_is_worse(store):
    tasks = [Task("xstest_unsafe_0", "refusal", before=1.0, after=0.0),
             Task("xstest_safe_1", "refusal", before=0.0, after=0.0)]
    change = write_call_trial(store, tasks)
    fs = ev.flip_set_by_answers(store, fx.PROVIDER, fx.MODEL_A, "refusal_rate", change)
    assert [f.task_id for f in fs.flips] == ["xstest_unsafe_0"]
    assert fs.flips[0].worse


def test_answer_based_ignores_other_models_and_old_calls(store):
    change = write_call_trial(store, TRIAL_TASKS)
    # Another model breaking at the same moment must not leak in.
    write_call_trial(store, [Task("mmlu_c", "accuracy", before=1.0, after=0.0)], model_id=fx.MODEL_B)
    fs = ev.flip_set_by_answers(store, fx.PROVIDER, fx.MODEL_A, "accuracy", change)
    assert "mmlu_c" not in [f.task_id for f in fs.flips]
    # With a 5-minute look-around, too few answers per question remain on each side.
    assert ev.flip_set_by_answers(store, fx.PROVIDER, fx.MODEL_A, "accuracy", change,
                                  lookaround=timedelta(minutes=5)) is None


def test_infra_evidence_by_calls(store):
    tasks = [Task("mmlu_a", "accuracy", before=1.0, after=1.0)]
    change = write_call_trial(store, tasks, latency_before_ms=1500, latency_after_ms=6000, failed_calls_after=1)
    infra = ev.infra_evidence_by_calls(store, fx.PROVIDER, fx.MODEL_A, change)
    assert infra.basis == ev.CALLS
    assert infra.median_latency_before_ms == pytest.approx(1500)
    assert infra.median_latency_after_ms == pytest.approx(6000)
    assert infra.outcomes_after == {"ok": 10, "timeout": 10}
    assert infra.summary.endswith("(last 10 calls before vs first 20 after).")


def test_evidence_for_incident_switches_to_answers_for_a_per_call_trial(store):
    tasks = TRIAL_TASKS + [Task("fmt_json_1", "format", before=1.0, after=0.0)]
    change = write_call_trial(store, tasks)
    hours = (change - fx.BASE_TIME) / timedelta(hours=1)
    acc = fx._event("accuracy", pre_mean=1.0, post_mean=0.5, p_value_corrected=0.01, hours_after_base=hours)
    fmt = fx._event("format_adherence", pre_mean=1.0, post_mean=0.0, p_value_corrected=0.01, hours_after_base=hours)
    lat = fx._event("latency_p95_ms", pre_mean=1500.0, post_mean=9000.0, p_value_corrected=0.01,
                    hours_after_base=hours)
    incident = group_into_incidents([acc, fmt, lat])[0]
    found = ev.evidence_for_incident(store, incident)
    assert found["accuracy"].basis == ev.ANSWERS
    assert [f.task_id for f in found["accuracy"].flips] == ["mmlu_a", "mmlu_b"]
    assert [f.task_id for f in found["format_adherence"].flips] == ["fmt_json_1"]
    assert found["latency_p95_ms"].basis == ev.CALLS


# ---------------------------------------------------------------------------
# The demo database (driftguard/demo/make_demo_db.py): the flip set must find
# what the script broke on purpose (give or take random slips), and nothing else.
# ---------------------------------------------------------------------------

def test_demo_database_flip_set_finds_exactly_the_injected_breakage(tmp_path):
    from driftguard.demo import make_demo_db as demo

    path = demo.build(tmp_path / "test_demo.db")
    store = DriftStore(db_path=str(path))
    calls, _ = ev.calls_around(store, demo.PROVIDER, demo.MODEL_ID, fx.BASE_TIME.replace(year=2100),
                               lookaround=timedelta(days=365 * 200))
    onset = demo.ONSET_PASS * len(demo._tasks())
    change = calls[onset].timestamp

    broken_mmlu = {t["id"] for t in demo._tasks() if t["type"] == "accuracy" and t["broken"]}
    acc = ev.flip_set_by_answers(store, demo.PROVIDER, demo.MODEL_ID, "accuracy", change)
    found = {f.task_id for f in acc.worse}
    assert found <= broken_mmlu and acc.better == []   # nothing it should not have found, and nothing got better
    # The simulated answers are random, so with about a third of the questions broken one can slip through.
    assert len(found) >= 0.9 * len(broken_mmlu)

    fmt = ev.flip_set_by_answers(store, demo.PROVIDER, demo.MODEL_ID, "format_adherence", change)
    assert {f.task_id for f in fmt.worse} == demo.BROKEN_FORMAT and fmt.better == []

    ref = ev.flip_set_by_answers(store, demo.PROVIDER, demo.MODEL_ID, "refusal_rate", change)
    assert ref.flips == []   # refusals were not changed
