"""
driftguard/detection/baseline.py

The naive, fixed-threshold baseline detector -- deliberately simple,
built ONLY as the comparison point the paper's validation results
benchmark PELT+bootstrap and Page-Hinkley against. This is not meant to
be a good detector; its weaknesses (no significance testing, no false-
positive control across repeated checks, no localization of when a
change actually started) are the entire point -- showing that the
statistically-grounded methods in this project outperform what a team
would build with an afternoon and a threshold is the argument the
paper's validation section needs to make.

Matches the original design description: "flag if metric drops >X% from
a rolling mean." Concretely: maintain a rolling window of the last N
values, compute their mean, and flag the current value if it deviates
from that rolling mean by more than a fixed offset -- no bootstrap
significance test, no multiple-testing correction, no PELT-style
localization of a specific change point. Every new observation is
checked independently against the current rolling mean; there's no
memory of "have I already flagged this same ongoing shift" the way
Page-Hinkley's cumulative sum or a proper change-point algorithm would
have, which is exactly the kind of naive alert-fatigue behavior a real
naive threshold system exhibits in practice.

The fixed offset uses the same calibrated spread (k * robust_sigma) that
calibration.py already derived -- so the comparison to PELT/Page-Hinkley
is fair (same underlying noise estimate, same k), and the ONLY thing
different is the detection METHOD, not the sensitivity calibration.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from driftguard.detection.calibration import Direction

DEFAULT_ROLLING_WINDOW = 10


@dataclass
class BaselineAlert:
    metric_name: str
    index: int
    timestamp: Optional[datetime]
    value: float
    rolling_mean: float
    deviation: float  # value - rolling_mean (signed)
    offset_used: float


class NaiveBaselineDetector:
    """
    Stateful, incremental, deliberately simple. Maintains a rolling
    window of recent values; each new value is checked against the
    window's mean using a fixed absolute offset. No smoothing of the
    alert itself, no cooldown, no significance test -- every point that
    breaches the threshold fires its own independent alert, which is
    intentional: this is what a naive comparison baseline looks like in
    practice, alert fatigue included.
    """

    def __init__(
        self,
        metric_name: str,
        direction: Direction,
        offset: float,
        window_size: int = DEFAULT_ROLLING_WINDOW,
    ):
        self.metric_name = metric_name
        self.direction = direction
        self.offset = offset
        self.window_size = window_size
        self._window: deque[float] = deque(maxlen=window_size)
        self._index = -1

    def update(self, value: float, timestamp: Optional[datetime] = None) -> Optional[BaselineAlert]:
        self._index += 1

        # need a full window before the rolling mean means anything --
        # otherwise the first few points would be compared against a
        # mean computed from almost no data, which isn't a fair "naive"
        # baseline, just a broken one.
        if len(self._window) < self.window_size:
            self._window.append(value)
            return None

        rolling_mean = sum(self._window) / len(self._window)
        deviation = value - rolling_mean

        # THEN append the new value, so the current point is compared
        # against the window BEFORE it, not including itself
        self._window.append(value)

        breached = (
            (self.direction == "lower_is_bad" and deviation < -self.offset)
            or (self.direction == "upper_is_bad" and deviation > self.offset)
        )

        if breached:
            return BaselineAlert(
                metric_name=self.metric_name,
                index=self._index,
                timestamp=timestamp,
                value=value,
                rolling_mean=rolling_mean,
                deviation=deviation,
                offset_used=self.offset,
            )

        return None


def run_baseline_detection(
    values: list[float],
    timestamps: list[Optional[datetime]],
    metric_name: str,
    direction: Direction,
    offset: float,
    window_size: int = DEFAULT_ROLLING_WINDOW,
) -> list[BaselineAlert]:
    """Batch-mode convenience runner, same pattern as run_sequential_detection in changepoint_sequential.py."""
    detector = NaiveBaselineDetector(metric_name=metric_name, direction=direction, offset=offset, window_size=window_size)
    alerts = []
    for i, value in enumerate(values):
        ts = timestamps[i] if i < len(timestamps) else None
        alert = detector.update(value, timestamp=ts)
        if alert is not None:
            alerts.append(alert)
    return alerts