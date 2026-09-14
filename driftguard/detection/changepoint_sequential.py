"""
driftguard/detection/changepoint_sequential.py

Live/low-latency change detector -- WRAPS frouros's PageHinkley
(frouros.detectors.concept_drift.PageHinkley), using the tuned
configuration validated in configs/detectors.yaml (lambda_=12.0 chosen
via a sweep across 8 detectors, benchmarked against real Elec2/Insects
drift datasets in addition to synthetic data -- see that file's comments
for the full methodology and why Page-Hinkley was kept over synthetically
higher-scoring alternatives like ECDD/HDDM_W).

*** IMPORTANT -- REQUIRED DATA GRANULARITY, CONFIRMED BY TESTING ***
frouros's tuned lambda_=12.0 was validated against PER-CALL binary error
indicators (1 = this individual call was wrong, 0 = it was right), around
a baseline error rate of ~5% (see detectors.yaml's
calibration.baseline_error_rate_assumption and
calibration.min_window_size, which is explicitly "samples (individual
eval_score calls, not eval runs)").

This does NOT operate correctly on runner.py's aggregated per-run
MetricPoint accuracy values (e.g. one 0.65/0.90/etc. per hourly eval
suite run). Confirmed directly: feeding this detector our own aggregated
per-run accuracy series (60 points, one real injected drift at index 30,
the same scenario used throughout this project's testing) produced ZERO
alerts. Feeding it a per-call binary error stream at the SAME baseline
error rate and shift magnitude this config was tuned for fired correctly,
~49 calls after the real shift.

*** INTEGRATION REQUIREMENT, NOT YET WIRED ***
For this detector to work as intended, whatever code produces individual
eval_score values per call (tasks.py's per-question grading,
format_check.py's per-task pass/fail, refusal_probe.py's per-prompt
refusal flag) needs to feed each one into this detector's .update() as
it happens, in ADDITION to being aggregated into the per-run MetricPoint
that baseline.py/changepoint_pelt.py/bootstrap.py consume. That call-site
wiring is outside this file's scope -- flagged here, and in the project's
notes, for whoever owns the eval-suite runner integration.

Direction handling: frouros's PageHinkley assumes "higher value = worse"
(it's built around error rates, where an increase is the drift being
watched for). For metrics in this project where a DECREASE is bad (e.g.
per-call correctness, where 1=correct is GOOD), invert at the call site
-- feed `1 - eval_score` (i.e. an error indicator) rather than
`eval_score` itself. For metrics where an increase is already bad (e.g.
a per-call refusal flag, 1=refused), feed the value directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from frouros.detectors.concept_drift import PageHinkley as _FrourosPageHinkley
from frouros.detectors.concept_drift import PageHinkleyConfig as _FrourosPageHinkleyConfig

# Tuned via a sweep across 8 detectors, validated against real Elec2/
# Insects drift datasets in addition to synthetic data. See
# configs/detectors.yaml for the full methodology and rationale -- do not
# change without re-running that validation.
DEFAULT_LAMBDA = 12.0
DEFAULT_DELTA = 0.005   # frouros default -- not yet independently tuned per detectors.yaml
DEFAULT_ALPHA = 0.9999  # frouros default -- not yet independently tuned per detectors.yaml
DEFAULT_MIN_NUM_INSTANCES = 30


@dataclass
class PageHinkleyAlert:
    metric_name: str
    index: int  # position (0-indexed, among values fed to this detector) where the alert fired
    timestamp: Optional[datetime]
    value_at_alert: float  # the raw value passed to update() (already direction-normalized by the caller)


class PageHinkleyDetector:
    """
    Thin wrapper around frouros.detectors.concept_drift.PageHinkley,
    adding: (1) this project's ChangePointEventRow persistence contract,
    (2) alert metadata (metric name, index, timestamp) frouros itself
    doesn't track.

    Call `update(value, timestamp)` once per new PER-CALL observation, in
    the direction-normalized form described in the module docstring
    (error indicator / "higher = worse"). Do NOT feed aggregated per-run
    metric values -- see the module docstring for why that doesn't work
    with this tuned configuration.
    """

    def __init__(
        self,
        metric_name: str,
        store: Optional["object"] = None,
        provider: Optional[str] = None,
        model_id: Optional[str] = None,
        lambda_: float = DEFAULT_LAMBDA,
        delta: float = DEFAULT_DELTA,
        alpha: float = DEFAULT_ALPHA,
        min_num_instances: int = DEFAULT_MIN_NUM_INSTANCES,
    ):
        self.metric_name = metric_name
        self._store = store
        self._provider = provider
        self._model_id = model_id

        self._detector = _FrourosPageHinkley(
            config=_FrourosPageHinkleyConfig(
                lambda_=lambda_, delta=delta, alpha=alpha, min_num_instances=min_num_instances,
            )
        )
        self._index = -1

    def update(self, value: float, timestamp: Optional[datetime] = None) -> Optional[PageHinkleyAlert]:
        self._index += 1
        self._detector.update(value=value)

        if self._detector.status["drift"]:
            alert = PageHinkleyAlert(
                metric_name=self.metric_name,
                index=self._index,
                timestamp=timestamp,
                value_at_alert=value,
            )
            if self._store is not None:
                self._persist_alert(alert)
            return alert

        return None

    def _persist_alert(self, alert: PageHinkleyAlert) -> None:
        from driftguard.storage.store import ChangePointEventRow

        event = ChangePointEventRow(
            detected_at=datetime.now(timezone.utc),
            provider=self._provider or "unknown",
            model_id=self._model_id or "unknown",
            metric_name=alert.metric_name,
            detector="sequential",
            changepoint_timestamp=alert.timestamp,
            pre_mean=None,   # frouros doesn't expose running pre/post means directly
            post_mean=alert.value_at_alert,
            p_value_raw=None,
            p_value_corrected=None,
            significant=True,  # threshold crossing IS the detection criterion -- no separate significance test
        )
        self._store.write_changepoint_event(event)


def run_sequential_detection(
    values: list[float],
    timestamps: list[Optional[datetime]],
    metric_name: str,
    lambda_: float = DEFAULT_LAMBDA,
    delta: float = DEFAULT_DELTA,
    alpha: float = DEFAULT_ALPHA,
    min_num_instances: int = DEFAULT_MIN_NUM_INSTANCES,
) -> list[PageHinkleyAlert]:
    """
    Batch-mode convenience runner -- replays a historical PER-CALL series
    (already direction-normalized, see module docstring) through a fresh
    detector, collecting every alert that would have fired. Useful for
    validation/replay; production use is via PageHinkleyDetector.update()
    fed one call at a time as they actually happen.
    """
    detector = PageHinkleyDetector(
        metric_name=metric_name, lambda_=lambda_, delta=delta, alpha=alpha, min_num_instances=min_num_instances,
    )
    alerts = []
    for i, value in enumerate(values):
        ts = timestamps[i] if i < len(timestamps) else None
        alert = detector.update(value, timestamp=ts)
        if alert is not None:
            alerts.append(alert)
    return alerts