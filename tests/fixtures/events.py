"""
tests/fixtures/events.py

Hand-written ChangePointEventRow objects for testing the human-facing layer
(classification/rules.py, remediation/mapper.py, dashboard/app.py) without
waiting for the detectors to find real drift.

Why these exist
---------------
The classifier's input is a change-point event. Real events only appear after
the eval suite has built up a long metric history, a real drift has happened,
and the detectors have found it. Until then the changepoint_events table is
empty. These fixtures are practice questions with known answers: each one
checks one specific behaviour, and EXPECTED_LABEL records the answer we
decided in advance. They are never reported as experimental results.

How the two detectors fill in an event (copied from Person 1's code)
--------------------------------------------------------------------
- detector="pelt" (bootstrap.py, then multiple_testing.py):
    pre_mean / post_mean = metric average before / after the change point,
    p_value_raw = permutation-test p-value,
    p_value_corrected = Benjamini-Hochberg value (None until correction runs),
    significant = provisional verdict, overwritten after correction.
- detector="sequential" (changepoint_sequential.py, live Page-Hinkley):
    pre_mean = None (frouros does not expose it),
    post_mean = the single direction-normalised value at the alert,
    p_value_raw = p_value_corrected = None,
    significant = True (crossing the threshold IS the detection).

Each factory returns a NEW object every call, because SQLModel objects are
changed in place when saved to a database.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from driftguard.storage.store import ChangePointEventRow, DriftStore

# One fixed starting point so every test sees identical timestamps.
BASE_TIME = datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)  # UTC: newer sqlmodel rejects timestamps without a timezone

PROVIDER = "groq"
MODEL_A = "openai/gpt-oss-120b"
MODEL_B = "llama-3.3-70b-versatile"


def _event(
    metric_name: str,
    *,
    detector: str = "pelt",
    model_id: str = MODEL_A,
    hours_after_base: float = 0.0,
    pre_mean: Optional[float] = None,
    post_mean: Optional[float] = None,
    p_value_raw: Optional[float] = None,
    p_value_corrected: Optional[float] = None,
    significant: bool = True,
) -> ChangePointEventRow:
    """Build one event with sensible defaults; each fixture overrides what it tests."""
    shift_time = BASE_TIME + timedelta(hours=hours_after_base)
    return ChangePointEventRow(
        detected_at=shift_time + timedelta(minutes=30),  # detector runs a little after the shift
        provider=PROVIDER,
        model_id=model_id,
        metric_name=metric_name,
        detector=detector,
        changepoint_timestamp=shift_time,
        pre_mean=pre_mean,
        post_mean=post_mean,
        p_value_raw=p_value_raw,
        p_value_corrected=p_value_corrected,
        significant=significant,
    )


# ---------------------------------------------------------------------------
# One clear case for each of the four drift types
# ---------------------------------------------------------------------------

def refusal_up() -> ChangePointEventRow:
    """Safe prompts refused more often (2% -> 20%). Expected: safety shift."""
    return _event("refusal_rate", hours_after_base=10, pre_mean=0.02, post_mean=0.20,
                  p_value_raw=0.004, p_value_corrected=0.008)


def accuracy_down() -> ChangePointEventRow:
    """MMLU accuracy falls (0.88 -> 0.71). Expected: capability shift."""
    return _event("accuracy", hours_after_base=20, pre_mean=0.88, post_mean=0.71,
                  p_value_raw=0.002, p_value_corrected=0.006)


def format_down_live() -> ChangePointEventRow:
    """Live Page-Hinkley alert on format checks: no pre_mean, no p-value. Expected: format drift."""
    return _event("format_adherence", detector="sequential", hours_after_base=30,
                  pre_mean=None, post_mean=1.0)


def latency_up() -> ChangePointEventRow:
    """p95 latency jumps (5.2 s -> 9.8 s). Expected: infra change."""
    return _event("latency_p95_ms", hours_after_base=40, pre_mean=5200.0, post_mean=9800.0,
                  p_value_raw=0.001, p_value_corrected=0.004)


# ---------------------------------------------------------------------------
# Edge cases: where code usually breaks
# ---------------------------------------------------------------------------

def accuracy_up() -> ChangePointEventRow:
    """Accuracy IMPROVES (0.80 -> 0.90). Getting better is not drift. Expected: no label."""
    return _event("accuracy", hours_after_base=50, pre_mean=0.80, post_mean=0.90,
                  p_value_raw=0.01, p_value_corrected=0.02)


def failed_correction() -> ChangePointEventRow:
    """Raw p = 0.03 passes, but after Benjamini-Hochberg p = 0.12 fails.
    The detector stored significant=False. Expected: capability shift, weak evidence."""
    return _event("accuracy", hours_after_base=60, pre_mean=0.86, post_mean=0.82,
                  p_value_raw=0.03, p_value_corrected=0.12, significant=False)


def unknown_metric() -> ChangePointEventRow:
    """A metric outside the four-way taxonomy (p50 latency). Expected: no label, no crash."""
    return _event("latency_p50_ms", hours_after_base=70, pre_mean=1400.0, post_mean=2600.0,
                  p_value_raw=0.01, p_value_corrected=0.02)


# ---------------------------------------------------------------------------
# Several channels moving together: one incident, not separate problems
# ---------------------------------------------------------------------------

def update_accuracy_down() -> ChangePointEventRow:
    """Part 1 of a simulated model update: accuracy falls at hour 100."""
    return _event("accuracy", hours_after_base=100, pre_mean=0.89, post_mean=0.74,
                  p_value_raw=0.001, p_value_corrected=0.004)


def update_format_down() -> ChangePointEventRow:
    """Part 2 of the same update: format adherence falls 40 minutes later."""
    return _event("format_adherence", hours_after_base=100 + 40 / 60, pre_mean=1.0, post_mean=0.83,
                  p_value_raw=0.003, p_value_corrected=0.006)


def update_accuracy_live() -> ChangePointEventRow:
    """The live detector also caught the accuracy fall, 20 minutes after PELT's estimate.
    Two detectors agreeing on the same metric = the strongest evidence."""
    return _event("accuracy", detector="sequential", hours_after_base=100 + 20 / 60,
                  pre_mean=None, post_mean=1.0)


def other_model_latency_up() -> ChangePointEventRow:
    """Same hour, DIFFERENT model. Must not be merged into model A's incident."""
    return _event("latency_p95_ms", model_id=MODEL_B, hours_after_base=100, pre_mean=3000.0,
                  post_mean=7400.0, p_value_raw=0.002, p_value_corrected=0.005)


# ---------------------------------------------------------------------------
# Registry used by the tests: fixture -> the answer decided in advance
# ---------------------------------------------------------------------------

# Label strings follow the four-way taxonomy in the task description.
# None means "this event must not be given a drift type".
EXPECTED_LABEL: dict[Callable[[], ChangePointEventRow], Optional[str]] = {
    refusal_up: "safety_shift",
    accuracy_down: "capability_shift",
    format_down_live: "format_drift",
    latency_up: "infra_change",
    accuracy_up: None,
    failed_correction: "capability_shift",
    unknown_metric: None,
    update_accuracy_down: "capability_shift",
    update_format_down: "format_drift",
    update_accuracy_live: "capability_shift",
    other_model_latency_up: "infra_change",
}

ALL_FIXTURES: list[Callable[[], ChangePointEventRow]] = list(EXPECTED_LABEL)


def all_events() -> list[ChangePointEventRow]:
    """A fresh copy of every fixture event."""
    return [make() for make in ALL_FIXTURES]


def load_into_store(store: DriftStore, events: Optional[list[ChangePointEventRow]] = None) -> list[ChangePointEventRow]:
    """Save events (default: all fixtures) into a DriftStore, returning them with ids filled in."""
    return [store.write_changepoint_event(e) for e in (events if events is not None else all_events())]
