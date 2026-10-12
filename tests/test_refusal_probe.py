"""
Checks for the refusal detector in driftguard/evalsuite/refusal_probe.py.

The first test is the bug that was found in real Groq logs: the reply
"I’m sorry, but I can’t help with that." (curly apostrophes, U+2019)
scored 0.0, i.e. "not a refusal".

Run with:  python -m pytest tests/test_refusal_probe.py -v
"""

from __future__ import annotations

import pytest

from driftguard.evalsuite.refusal_probe import _is_refusal


def test_curly_apostrophe_refusal_from_real_logs_is_detected():
    assert _is_refusal("I’m sorry, but I can’t help with that.")


def test_straight_apostrophe_refusal_still_detected():
    assert _is_refusal("I'm sorry, but I can't help with that.")


@pytest.mark.parametrize("apostrophe", ["’", "‘", "ʼ", "＇"])
def test_every_apostrophe_variant_is_treated_the_same(apostrophe):
    assert _is_refusal(f"I won{apostrophe}t do that.")


def test_refusals_without_apostrophes_still_detected():
    assert _is_refusal("I cannot assist with that request.")
    assert _is_refusal("I must decline.")


def test_normal_answer_is_not_a_refusal():
    assert not _is_refusal("Money laundering is the process of making illegally obtained funds appear legitimate.")
    assert not _is_refusal("Here’s a summary of the three stages.")  # curly apostrophe, but no refusal phrase


@pytest.mark.parametrize("empty", [None, ""])
def test_empty_response_is_not_a_refusal(empty):
    assert not _is_refusal(empty)
