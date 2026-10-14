"""
driftguard/detection/live.py

The live alarm: Page-Hinkley fed each score the moment it is produced.

The eval runner creates a LiveMonitor at the start of a run and hands it every
scored call as it comes back (run_accuracy_tasks / run_format_tasks /
run_refusal_tasks take an `on_entry` callback for this). The monitor:

  1. picks up where the detectors left off -- it replays the target's stored
     calls through the same per-call signals (pipeline.call_signals) and the
     same Page-Hinkley detectors, reset after each alarm, that the batch path
     uses, so a live alarm falls on exactly the call the batch path would put it
     on (and pipeline.save_events never stores it twice);
  2. updates the detector of every signal the new call touches;
  3. on an alarm, builds a ChangePointEventRow in the classification layer's
     vocabulary (detector="sequential") and saves it through
     pipeline.save_events, which lets remediation/mapper.apply_remediation label
     and plan it -- so it shows on the dashboard straight away.

A live alarm marks where the change was NOTICED. Confirmation and location come
from the retrospective path (pipeline.run_detection), which the runner runs at
the end of each run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from driftguard.detection import pipeline
from driftguard.detection.changepoint_sequential import PageHinkleyDetector
from driftguard.detection.signals import SIGNALS, to_metric_value
from driftguard.ingest.log_schema import DriftLogEntry
from driftguard.storage.store import ChangePointEventRow, DriftStore

RECENT = 50   # calls that make up the "after" value of a live alarm (the window it was noticed in)


@dataclass
class _SignalState:
    detector: Optional[PageHinkleyDetector]   # None until the baseline window is complete
    scale: float = 1.0
    values: list[float] = field(default_factory=list)
    latencies: list[Optional[float]] = field(default_factory=list)


@dataclass
class LiveAlarm:
    signal: str
    event: ChangePointEventRow


class LiveMonitor:
    """Per-target live Page-Hinkley over every per-call signal."""

    def __init__(self, store: DriftStore, provider: str, model_id: str,
                 settings: Optional[pipeline.DetectionSettings] = None):
        self.store, self.provider, self.model_id = store, provider, model_id
        self.s = settings or pipeline.load_settings()
        self._should = pipeline._should_refuse_map()
        history = pipeline.load_calls(store, provider, model_id)
        self._slow_ms = pipeline.slow_threshold(history, self.s.baseline_window)
        self._signals: dict[str, _SignalState] = {}
        self.alarms: list[LiveAlarm] = []
        for r in history.itertuples(index=False):           # catch up, without raising alarms
            latency = None if pd.isna(r.latency_ms) else float(r.latency_ms)
            score = None if pd.isna(r.eval_score) else float(r.eval_score)
            self._update(r.outcome, r.eval_task_type, r.eval_task_id, score, latency, r.timestamp, live=False)

    # ------------------------------------------------------------------ feeding

    def observe(self, entry: DriftLogEntry) -> list[LiveAlarm]:
        """Feed one scored call; returns (and saves) any alarm it triggers."""
        outcome = getattr(entry.outcome, "value", entry.outcome)
        return self._update(outcome, entry.eval_task_type, entry.eval_task_id, entry.eval_score,
                            entry.latency_ms, entry.timestamp, live=True)

    def _fresh(self, scale: float = 1.0) -> PageHinkleyDetector:
        return pipeline.new_page_hinkley(self.s, scale)

    def _update(self, outcome, task_type, task_id, score, latency, timestamp, live: bool) -> list[LiveAlarm]:
        raised = []
        for key, value in pipeline.call_signals(outcome, task_type, task_id, score, latency,
                                                self._should, self._slow_ms).items():
            start = pipeline.ph_start(self.s)
            st = self._signals.setdefault(key, _SignalState(None if start else self._fresh()))
            st.values.append(value)
            st.latencies.append(latency)
            if st.detector is None:                          # still filling the baseline window
                if len(st.values) == start:
                    st.detector, st.scale = pipeline.warmed_page_hinkley(np.asarray(st.values, dtype=float), self.s)
                continue
            if st.detector.update(value) is None:
                continue
            st.detector = self._fresh(st.scale)              # reset after each alarm, as in the batch path
            if live:
                raised.append(self._raise(key, st, timestamp))
        if raised:
            pipeline.save_events(self.store, [a.event for a in raised])
            self.alarms.extend(raised)
        return raised

    # ------------------------------------------------------------------ alarms

    def _raise(self, key: str, st: _SignalState, timestamp) -> LiveAlarm:
        v = np.asarray(st.values, dtype=float)
        i = len(v) - 1
        recent = v[max(0, i - RECENT + 1): i + 1]
        before = v[max(0, i - RECENT + 1 - pipeline.COMPARE_WINDOW): max(0, i - RECENT + 1)]
        metric = SIGNALS[key].metric or key
        if metric == "latency_p95_ms":
            lat = np.asarray([x for x in st.latencies if x is not None], dtype=float)
            pre = float(np.percentile(lat[:-RECENT], 95)) if len(lat) > RECENT else None
            post = float(np.percentile(lat[-RECENT:], 95)) if len(lat) else None
        else:
            pre = to_metric_value(key, float(before.mean())) if len(before) else None
            post = to_metric_value(key, float(recent.mean()))
        t = pd.Timestamp(timestamp)
        t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        event = ChangePointEventRow(
            detected_at=datetime.now(timezone.utc), provider=self.provider, model_id=self.model_id,
            metric_name=metric, detector="sequential", changepoint_timestamp=t.to_pydatetime(),
            pre_mean=pre, post_mean=post, significant=True)
        return LiveAlarm(key, event)
