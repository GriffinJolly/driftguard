"""
driftguard/detection/changepoint_sequential.py

Page-Hinkley test: an ONLINE/sequential change detector, processing one
new metric value at a time and maintaining a small amount of running
state, rather than requiring a full batch series like PELT.

Why this exists alongside PELT (recalling the original design
rationale): PELT is retrospective -- it needs a reasonably dense,
already-collected series to localize a change point confidently. With an
hourly eval cadence, that means waiting for many post-change points to
accumulate before PELT can confirm anything. Page-Hinkley instead
maintains a running cumulative-deviation score and can fire the moment
new values consistently sit on the "bad" side of what's been seen so
far, using only a handful of post-change points, at the cost of being
less precise about exactly WHERE the change happened and using more
naive noise-handling. In this pipeline, Page-Hinkley is the "raise an
alert now" mechanism; PELT + bootstrap remain the "confirm and localize
after the fact, with a rigorous p-value" mechanism.

Algorithm (direction-aware, mirroring calibration.py's per-metric
direction so both increases and decreases can be the "adverse" case):

  For a lower_is_bad metric (a drop is bad, e.g. accuracy):
    running_mean_t = mean of all values seen so far, including x_t
    deviation_t = running_mean_t - x_t - delta
      (delta is a small allowed slack -- ordinary values don't
      accumulate score just for sitting a little below the mean)
    cumulative_sum_t = cumulative_sum_{t-1} + deviation_t
    min_seen = running minimum of cumulative_sum over all t so far
    PH_t = cumulative_sum_t - min_seen
    ALERT if PH_t > threshold

  For an upper_is_bad metric (a rise is bad, e.g. refusal_rate or
  latency), the deviation term is mirrored: (x_t - running_mean_t - delta).

delta and threshold come from calibration.py's page_hinkley_delta /
page_hinkley_threshold, derived empirically from quiet-window noise --
NOT hardcoded constants, same principle as everywhere else in this
pipeline.

After an alert fires, the detector's cumulative state is reset (a common
convention for Page-Hinkley) so it can detect a SUBSEQUENT change later,
rather than staying permanently triggered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from driftguard.detection.calibration import Direction


@dataclass
class PageHinkleyAlert:
    metric_name: str
    index: int  # position (0-indexed, among values fed to this detector) where the alert fired
    timestamp: Optional[datetime]
    value_at_alert: float
    running_mean_at_alert: float
    ph_statistic: float  # the cumulative score value that crossed threshold
    threshold: float


class PageHinkleyDetector:
    """
    Stateful, incremental detector. Call `update(value, timestamp)` once
    per new observation, in order. Returns a PageHinkleyAlert if this
    observation triggered a detection, else None. Internal state persists
    across calls -- this is meant to be created once per (provider,
    model_id, metric) and fed values as they arrive in production, not
    recreated from scratch each time (recreating it each time would lose
    the whole point of an online detector).
    """

    def __init__(
        self,
        metric_name: str,
        direction: Direction,
        delta: float,
        threshold: float,
    ):
        self.metric_name = metric_name
        self.direction = direction
        self.delta = delta
        self.threshold = threshold

        self._n = 0
        self._running_mean = 0.0
        self._cumulative_sum = 0.0
        self._min_cumulative_sum = 0.0
        self._index = -1  # incremented before use, so first update is index 0

    def update(self, value: float, timestamp: Optional[datetime] = None) -> Optional[PageHinkleyAlert]:
        self._index += 1
        self._n += 1

        # incremental running mean (Welford-style update, avoids needing to store full history)
        self._running_mean += (value - self._running_mean) / self._n

        if self.direction == "lower_is_bad":
            deviation = self._running_mean - value - self.delta
        else:  # upper_is_bad
            deviation = value - self._running_mean - self.delta

        self._cumulative_sum += deviation
        self._min_cumulative_sum = min(self._min_cumulative_sum, self._cumulative_sum)
        ph_statistic = self._cumulative_sum - self._min_cumulative_sum

        if ph_statistic > self.threshold:
            alert = PageHinkleyAlert(
                metric_name=self.metric_name,
                index=self._index,
                timestamp=timestamp,
                value_at_alert=value,
                running_mean_at_alert=self._running_mean,
                ph_statistic=ph_statistic,
                threshold=self.threshold,
            )
            self.reset()
            return alert

        return None

    def reset(self) -> None:
        """
        Clear accumulated state so the detector can find a SUBSEQUENT
        change later. Deliberately does NOT reset _n / _running_mean to
        zero -- the running mean should keep incorporating all history
        (the "new normal" after a detected shift becomes part of the
        baseline going forward), only the cumulative deviation tracker
        needs to restart.
        """
        self._cumulative_sum = 0.0
        self._min_cumulative_sum = 0.0


def run_sequential_detection(
    values: list[float],
    timestamps: list[Optional[datetime]],
    metric_name: str,
    direction: Direction,
    delta: float,
    threshold: float,
) -> list[PageHinkleyAlert]:
    """
    Convenience batch-mode runner: feeds an entire historical series
    through a fresh PageHinkleyDetector in order, collecting every alert
    that would have fired along the way. Useful for validation (replaying
    real historical data to see when Page-Hinkley WOULD have fired) even
    though the detector itself is designed for live, one-value-at-a-time
    use in production.
    """
    detector = PageHinkleyDetector(metric_name=metric_name, direction=direction, delta=delta, threshold=threshold)
    alerts = []
    for i, value in enumerate(values):
        ts = timestamps[i] if i < len(timestamps) else None
        alert = detector.update(value, timestamp=ts)
        if alert is not None:
            alerts.append(alert)
    return alerts