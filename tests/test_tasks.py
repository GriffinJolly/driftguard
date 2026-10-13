"""Unit tests for driftguard/evalsuite/tasks.py -- the frozen-MMLU accuracy
task: answer-letter extraction, prompt formatting, grading, and the
run_accuracy_tasks loop with call_model monkeypatched (no network)."""

from __future__ import annotations

import pytest

from driftguard.evalsuite import tasks
from driftguard.evalsuite.tasks import (
    _extract_answer_letter,
    _format_prompt,
    accuracy_rate,
    run_accuracy_tasks,
)
from driftguard.ingest.log_schema import CallOutcome, DriftLogEntry, IntegrationPath


@pytest.mark.parametrize("text,expected", [
    ("B", "B"),
    ("The answer is C.", "C"),
    ("(A)", "A"),
    ("d", "D"),
    ("Answer: A", "A"),
    ("I think it's B, because reasons", "B"),
    ("", None),
    (None, None),
    ("The letter E is not an option", None),
])
def test_extract_answer_letter(text, expected):
    assert _extract_answer_letter(text) == expected


def test_format_prompt_lists_all_choices():
    prompt = _format_prompt({"question": "Q?", "choices": ["ww", "xx", "yy", "zz"]})
    assert "Q?" in prompt
    for letter, choice in zip("ABCD", ["ww", "xx", "yy", "zz"]):
        assert f"{letter}. {choice}" in prompt
    assert "only the letter" in prompt.lower()


def _scored(score):
    e = DriftLogEntry(integration_path=IntegrationPath.EVAL_SUITE, provider="g", model_id="m", prompt="p")
    e.eval_score = score
    return e


def test_accuracy_rate():
    assert accuracy_rate([_scored(1.0), _scored(0.0), _scored(1.0)]) == pytest.approx(2 / 3)
    assert accuracy_rate([_scored(None)]) is None  # unscored calls excluded
    assert accuracy_rate([]) is None


def _fake_frozen():
    return [
        {"task_id": "t1", "subject": "s", "question": "Q1", "choices": ["a", "b", "c", "d"], "correct_index": 1},
        {"task_id": "t2", "subject": "s", "question": "Q2", "choices": ["a", "b", "c", "d"], "correct_index": 0},
    ]


def test_run_accuracy_tasks_grades_correct_and_wrong(monkeypatch):
    monkeypatch.setattr(tasks, "_ensure_frozen_subset", _fake_frozen)

    def fake_call(provider, model_id, prompt, **kwargs):
        # answer "B" to everything: correct for t1 (index 1), wrong for t2 (index 0)
        return DriftLogEntry(integration_path=IntegrationPath.EVAL_SUITE, provider=provider,
                             model_id=model_id, prompt=prompt, response_text="B", outcome=CallOutcome.OK)

    monkeypatch.setattr(tasks, "call_model", fake_call)
    entries = run_accuracy_tasks("groq", "m", "run1")
    assert [e.eval_score for e in entries] == [1.0, 0.0]
    assert accuracy_rate(entries) == pytest.approx(0.5)


def test_run_accuracy_tasks_failed_call_is_unscored(monkeypatch):
    monkeypatch.setattr(tasks, "_ensure_frozen_subset", lambda: _fake_frozen()[:1])

    def fake_call(provider, model_id, prompt, **kwargs):
        return DriftLogEntry(integration_path=IntegrationPath.EVAL_SUITE, provider=provider,
                             model_id=model_id, prompt=prompt, outcome=CallOutcome.RATE_LIMITED)

    monkeypatch.setattr(tasks, "call_model", fake_call)
    entries = run_accuracy_tasks("groq", "m", "run1")
    assert entries[0].eval_score is None  # failed calls don't count toward accuracy
