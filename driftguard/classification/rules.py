"""
driftguard/classification/rules.py

Rule-based categorisation of a detected shift. Every monitored signal belongs
to exactly one of the four operational categories from the survey: a shift is
attributed to the category of the signal it was detected on.

This is deliberately the simplest defensible rule -- one signal, one category.
It does not try to explain a shift that shows up on several signals at once
(e.g. truncation that hurts accuracy AND format); those are reported once per
signal and left to the operator to correlate on the shift map.

All signals are per-call and oriented "higher = worse", matching what the
Page-Hinkley detector was tuned on (see configs/detectors.yaml).
"""

from __future__ import annotations

from dataclasses import dataclass

CAPABILITY = "Capability"
SAFETY = "Safety"
FORMAT = "Format"
INFRASTRUCTURE = "Infrastructure"

CATEGORIES = (CAPABILITY, SAFETY, FORMAT, INFRASTRUCTURE)


@dataclass(frozen=True)
class Signal:
    key: str
    label: str        # short name shown in the UI
    category: str
    meaning: str      # what a 1 (or a high value) on this signal means
    # False for signals that describe OUR side of the connection (e.g. running out
    # of quota). They are still tracked and mapped, but never count as provider drift.
    provider_drift: bool = True


SIGNALS: dict[str, Signal] = {s.key: s for s in (
    Signal("accuracy_error", "Answer accuracy", CAPABILITY,
           "The model answered a frozen MMLU question incorrectly."),
    Signal("format_error", "Schema adherence", FORMAT,
           "The response did not contain JSON that matches the requested schema."),
    Signal("safe_refusal", "Over-refusal", SAFETY,
           "The model refused a safe XSTest prompt it should have answered."),
    Signal("unsafe_compliance", "Under-refusal", SAFETY,
           "The model answered an unsafe XSTest prompt it should have refused."),
    Signal("call_failure", "Call failures", INFRASTRUCTURE,
           "The API call errored instead of returning a response (rate limits excluded)."),
    Signal("quota_exhausted", "Quota exhausted", INFRASTRUCTURE,
           "The call was rejected because our own rate limit or daily quota ran out.",
           provider_drift=False),
    Signal("slow_call", "Slow responses", INFRASTRUCTURE,
           "The call took longer than the 95th-percentile latency of the baseline window."),
)}


def classify(signal_key: str) -> str:
    """Return the drift category for a shift detected on `signal_key`."""
    try:
        return SIGNALS[signal_key].category
    except KeyError:
        raise ValueError(f"Unknown signal: {signal_key!r}") from None
