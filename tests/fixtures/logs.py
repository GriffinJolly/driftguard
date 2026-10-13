"""
tests/fixtures/logs.py

Writes a small, hand-designed eval history into a DriftStore: a series of
hourly eval runs, each with one log row per frozen question, exactly the
shape runner.py produces (LogRow per call, MetricPoint per run, linked by
run_id == eval_run_id).

Each question's result before and after the change point is chosen by
the test, so the flip set has a known right answer. Used only in tests.
"""

from __future__ import annotations

import statistics
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Sequence, Union

from driftguard.ingest.log_schema import DriftLogEntry, MetricPoint
from driftguard.storage.store import DriftStore
from tests.fixtures.events import BASE_TIME, MODEL_A, PROVIDER

Scores = Union[float, Sequence[Optional[float]]]


@dataclass
class Task:
    """One frozen question and its 0/1 score in each run before and after the change."""
    task_id: str
    task_type: str        # "accuracy" | "format" | "refusal"
    before: Scores        # one score for every run, or a list with one score per run
    after: Scores

    def score(self, side: str, run_index: int) -> Optional[float]:
        value = self.before if side == "before" else self.after
        return value if isinstance(value, (int, float)) else value[run_index]


def _rate(entries: list[DriftLogEntry], task_type: str, only_safe: bool = False) -> Optional[float]:
    scores = [
        e.eval_score for e in entries
        if e.eval_task_type == task_type and e.eval_score is not None
        and not (only_safe and "_unsafe_" in (e.eval_task_id or ""))
    ]
    return sum(scores) / len(scores) if scores else None


def write_eval_history(
    store: DriftStore,
    tasks: list[Task],
    n_before: int = 5,
    n_after: int = 5,
    provider: str = PROVIDER,
    model_id: str = MODEL_A,
    start: datetime = BASE_TIME,
    latency_before_ms: float = 1500.0,
    latency_after_ms: float = 1500.0,
    failed_calls_after: int = 0,
) -> datetime:
    """
    Write n_before + n_after hourly runs. Returns the change time
    (the timestamp of the first run after the change).

    failed_calls_after: in each run after the change, this many extra
    calls end in a timeout (no answer, no score), for infra tests.
    """
    for i in range(n_before + n_after):
        side, index = ("before", i) if i < n_before else ("after", i - n_before)
        run_id = str(uuid.uuid4())
        run_time = start + timedelta(hours=i)
        latency = latency_before_ms if side == "before" else latency_after_ms
        entries: list[DriftLogEntry] = []

        for task in tasks:
            score = task.score(side, index)
            entries.append(DriftLogEntry(
                integration_path="eval_suite",
                provider=provider,
                model_id=model_id,
                timestamp=run_time,
                prompt=f"Prompt for {task.task_id}",
                response_text=f"{side} answer {index} to {task.task_id} (score {score})",
                latency_ms=latency,
                eval_run_id=run_id,
                eval_task_type=task.task_type,
                eval_task_id=task.task_id,
                eval_score=score,
            ))

        if side == "after":
            for k in range(failed_calls_after):
                entries.append(DriftLogEntry(
                    integration_path="eval_suite",
                    provider=provider,
                    model_id=model_id,
                    timestamp=run_time,
                    prompt=f"Timed-out call {k}",
                    outcome="timeout",
                    eval_run_id=run_id,
                    eval_task_type="accuracy",
                    eval_task_id=f"timeout_{k}",
                    eval_score=None,
                ))

        for entry in entries:
            store.write_log(entry)

        latencies = sorted(e.latency_ms for e in entries if e.latency_ms is not None)
        store.write_metric_point(MetricPoint(
            run_id=run_id,
            timestamp=run_time,
            provider=provider,
            model_id=model_id,
            accuracy=_rate(entries, "accuracy"),
            format_adherence=_rate(entries, "format"),
            refusal_rate=_rate(entries, "refusal", only_safe=True),
            latency_p50_ms=statistics.median(latencies) if latencies else None,
            latency_p95_ms=latencies[-1] if latencies else None,
            n_calls=len(entries),
        ))

    return start + timedelta(hours=n_before)


def write_call_trial(
    store: DriftStore,
    tasks: list[Task],
    passes_before: int = 10,
    passes_after: int = 10,
    provider: str = PROVIDER,
    model_id: str = MODEL_A,
    start: datetime = BASE_TIME,
    latency_before_ms: float = 1500.0,
    latency_after_ms: float = 1500.0,
    failed_calls_after: int = 0,
    run_id: str = "trial",
) -> datetime:
    """
    A per-call trial, the shape the detection pipeline and dashboard work on:
    ONE run, one call per minute, cycling through every task once per pass.
    Task.before / Task.after give the score for each pass (one value for
    every pass, or a list with one score per pass). No MetricPoint rows are
    written, because a per-call trial has no per-run metrics.

    Returns the change time: the timestamp of the first call after the change.
    failed_calls_after: in each pass after the change, this many extra calls
    end in a timeout (no answer, no score), for infra tests.
    """
    minute = 0
    change_time: Optional[datetime] = None
    for p in range(passes_before + passes_after):
        side, index = ("before", p) if p < passes_before else ("after", p - passes_before)
        latency = latency_before_ms if side == "before" else latency_after_ms
        calls = [(task, task.score(side, index)) for task in tasks]
        if side == "after":
            calls += [(None, None)] * failed_calls_after
        for task, score in calls:
            when = start + timedelta(minutes=minute)
            if side == "after" and change_time is None:
                change_time = when
            if task is None:
                entry = DriftLogEntry(
                    integration_path="eval_suite", provider=provider, model_id=model_id, timestamp=when,
                    prompt="Timed-out call", outcome="timeout", eval_run_id=run_id,
                    eval_task_type="accuracy", eval_task_id=f"timeout_{minute}", eval_score=None,
                )
            else:
                entry = DriftLogEntry(
                    integration_path="eval_suite", provider=provider, model_id=model_id, timestamp=when,
                    prompt=f"Prompt for {task.task_id}",
                    response_text=f"{side} answer {index} to {task.task_id} (score {score})",
                    latency_ms=latency, eval_run_id=run_id,
                    eval_task_type=task.task_type, eval_task_id=task.task_id, eval_score=score,
                )
            store.write_log(entry)
            minute += 1
    return change_time if change_time is not None else start + timedelta(minutes=minute)
