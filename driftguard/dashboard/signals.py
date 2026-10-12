"""
driftguard/dashboard/signals.py

The per-call signals the dashboard runs its detectors on, and how each one
maps onto the classification layer (driftguard/classification/rules.py).

Detection stays per call -- the tuned Page-Hinkley detector only works on
per-call input (see changepoint_sequential.py). Every signal is therefore a
0/1 stream oriented "higher = worse". Classification, incident grouping,
urgency and remediation all happen in rules.py / incidents.py / mapper.py,
which speak in the per-run metric vocabulary (accuracy, format_adherence,
refusal_rate, latency_p95_ms) with each metric's OWN direction. `metric`
below is the bridge: a shift found on a signal is handed to rules.py as an
event on that metric, with its before/after values converted to the
metric's units (e.g. error rate 0.13 -> accuracy 0.87).

Signals whose `metric` is None are outside the four-way taxonomy. They are
still detected and shown for context, but rules.py never labels them, so
they never become incidents (call failures, quota exhaustion, and answers
to unsafe prompts -- the last is covered by flip-set evidence instead).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Signal:
    key: str
    label: str                 # short name shown in the UI
    meaning: str               # what a 1 (bad outcome) on this signal means
    metric: Optional[str]      # metric name in rules.TAXONOMY, or None if outside it
    context_note: str = ""     # why an unlabelled signal is still shown


SIGNALS: dict[str, Signal] = {s.key: s for s in (
    Signal("accuracy_error", "Answer accuracy",
           "The model answered a frozen MMLU question incorrectly.", "accuracy"),
    Signal("format_error", "Schema adherence",
           "The response did not contain JSON that matches the requested schema.", "format_adherence"),
    Signal("safe_refusal", "Over-refusal",
           "The model refused a safe XSTest prompt it should have answered.", "refusal_rate"),
    Signal("slow_call", "Slow responses",
           "The call took longer than the 95th-percentile latency of the baseline window.", "latency_p95_ms"),
    Signal("unsafe_compliance", "Under-refusal",
           "The model answered an unsafe XSTest prompt it should have refused.", None,
           "Outside the taxonomy; unsafe prompts are covered by flip-set evidence."),
    Signal("call_failure", "Call failures",
           "The API call errored instead of returning a response (rate limits excluded).", None,
           "Outside the taxonomy; used as infrastructure evidence."),
    Signal("quota_exhausted", "Quota exhausted",
           "The call was rejected because our own rate limit or daily quota ran out.", None,
           "Client-side: our quota, not a change in the provider."),
)}


def to_metric_value(signal_key: str, rate: float) -> float:
    """Convert a 'higher = worse' per-call rate into the metric's own units
    (latency is converted separately, from the calls' measured latencies)."""
    metric = SIGNALS[signal_key].metric
    if metric in ("accuracy", "format_adherence"):
        return 1.0 - rate
    return rate
