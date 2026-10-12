"""
driftguard/remediation/evidence.py

The evidence a person can actually read: which exact eval questions
changed their result across a change point.

The flip set
------------
The eval suite asks the SAME frozen questions every run (MMLU for
accuracy, the structured-output tasks for format, XSTest for refusals),
and every answer is stored in log_rows with its eval_task_id and a 0/1
eval_score. So for any change point we can line up each question's
results before and after:

  - take the last N runs before the change point and the first N after
    (default N = 5),
  - for each question, compute its rate on each side (fraction correct,
    fraction passing the schema, or fraction refused),
  - the question FLIPPED if that rate moved by at least MIN_SHIFT
    (default 0.5): e.g. right in 5/5 runs before, right in 1/5 after.

Requiring half the runs to change on purpose ignores a question that
was wrong once by chance.

"Worse" depends on the question:
  accuracy, format            : the rate falling is worse.
  refusal, SAFE XSTest prompt : refusing MORE is worse (over-refusal).
  refusal, UNSAFE prompt      : refusing LESS is worse (safety loosening).
The last case matters: the refusal_rate metric only tracks safe prompts,
so a model that stops refusing harmful prompts never moves that metric.
The flip set still catches it.

Infra changes do not change answers, so for latency the evidence is a
before/after comparison of call outcomes (ok / error / timeout /
rate_limited) and typical latency instead.

Reads only through DriftStore's existing methods (get_metric_series,
get_logs_for_run); changes nothing in the database.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Union

from driftguard.classification.incidents import Incident, _as_utc
from driftguard.storage.store import DriftStore, LogRow, MetricSeriesRow

DEFAULT_RUNS_EACH_SIDE = 5
MIN_SHIFT = 0.5

# Which eval_task_type in log_rows feeds which metric.
METRIC_TASK_TYPE = {
    "accuracy": "accuracy",
    "format_adherence": "format",
    "refusal_rate": "refusal",
}

TASK_NOUN = {"accuracy": "questions", "format": "format tasks", "refusal": "prompts"}


# ---------------------------------------------------------------------------
# Picking the runs on each side of a change point
# ---------------------------------------------------------------------------

def runs_around(
    store: DriftStore,
    provider: str,
    model_id: str,
    change_time: datetime,
    n_runs: int = DEFAULT_RUNS_EACH_SIDE,
) -> tuple[list[MetricSeriesRow], list[MetricSeriesRow]]:
    """The last n_runs eval runs before change_time and the first n_runs at or after it."""
    change_time = _as_utc(change_time)
    runs = sorted(store.get_metric_series(provider, model_id), key=lambda r: _as_utc(r.timestamp))
    before = [r for r in runs if _as_utc(r.timestamp) < change_time][-n_runs:]
    after = [r for r in runs if _as_utc(r.timestamp) >= change_time][:n_runs]
    return before, after


def _logs(store: DriftStore, runs: list[MetricSeriesRow]) -> list[LogRow]:
    rows: list[LogRow] = []
    for run in runs:
        rows.extend(store.get_logs_for_run(run.run_id))
    return rows


# ---------------------------------------------------------------------------
# The flip set
# ---------------------------------------------------------------------------

@dataclass
class Flip:
    task_id: str
    task_type: str
    rate_before: float
    rate_after: float
    n_before: int
    n_after: int
    worse: bool
    prompt: str
    response_before: Optional[str]
    response_after: Optional[str]

    @property
    def shift(self) -> float:
        return self.rate_after - self.rate_before

    @property
    def description(self) -> str:
        """e.g. 'refused in 0/5 runs before, 4/5 after'."""
        verb = {"accuracy": "correct", "format": "passed the schema", "refusal": "refused"}[self.task_type]
        before = round(self.rate_before * self.n_before)
        after = round(self.rate_after * self.n_after)
        return f"{verb} in {before}/{self.n_before} runs before, {after}/{self.n_after} after"


@dataclass
class FlipSet:
    metric_name: str
    task_type: str
    n_runs_before: int
    n_runs_after: int
    n_tasks_compared: int
    flips: list[Flip] = field(default_factory=list)

    @property
    def worse(self) -> list[Flip]:
        return [f for f in self.flips if f.worse]

    @property
    def better(self) -> list[Flip]:
        return [f for f in self.flips if not f.worse]

    @property
    def summary(self) -> str:
        noun = TASK_NOUN[self.task_type]
        return (
            f"{len(self.flips)} of {self.n_tasks_compared} {noun} changed: "
            f"{len(self.worse)} got worse, {len(self.better)} got better "
            f"(last {self.n_runs_before} runs before vs first {self.n_runs_after} after)."
        )


def _is_worse(task_type: str, task_id: str, shift: float) -> bool:
    if task_type == "refusal" and "_unsafe_" in task_id:
        return shift < 0      # stopped refusing a harmful prompt
    if task_type == "refusal":
        return shift > 0      # started refusing a harmless prompt
    return shift < 0          # accuracy / format: fewer passes


def flip_set(
    store: DriftStore,
    provider: str,
    model_id: str,
    metric_name: str,
    change_time: datetime,
    n_runs: int = DEFAULT_RUNS_EACH_SIDE,
    min_shift: float = MIN_SHIFT,
) -> Optional[FlipSet]:
    """
    Compare every frozen question's result before and after change_time.
    Returns None when the metric has no per-question logs (latency) or
    there is no run on one side of the change point.
    """
    task_type = METRIC_TASK_TYPE.get(metric_name)
    if task_type is None:
        return None
    before_runs, after_runs = runs_around(store, provider, model_id, change_time, n_runs)
    if not before_runs or not after_runs:
        return None

    def by_task(rows: list[LogRow]) -> dict[str, list[LogRow]]:
        grouped: dict[str, list[LogRow]] = {}
        for row in sorted(rows, key=lambda r: _as_utc(r.timestamp)):
            if row.eval_task_type == task_type and row.eval_score is not None and row.eval_task_id:
                grouped.setdefault(row.eval_task_id, []).append(row)
        return grouped

    before = by_task(_logs(store, before_runs))
    after = by_task(_logs(store, after_runs))
    shared = [task_id for task_id in before if task_id in after]

    flips: list[Flip] = []
    for task_id in shared:
        b, a = before[task_id], after[task_id]
        rate_b = sum(r.eval_score for r in b) / len(b)
        rate_a = sum(r.eval_score for r in a) / len(a)
        if abs(rate_a - rate_b) < min_shift:
            continue
        flips.append(Flip(
            task_id=task_id,
            task_type=task_type,
            rate_before=rate_b,
            rate_after=rate_a,
            n_before=len(b),
            n_after=len(a),
            worse=_is_worse(task_type, task_id, rate_a - rate_b),
            prompt=b[-1].prompt,
            response_before=b[-1].response_text,   # the last answer before the change
            response_after=a[0].response_text,     # the first answer after it
        ))

    # Worse flips first, biggest moves first.
    flips.sort(key=lambda f: (not f.worse, -abs(f.shift), f.task_id))
    return FlipSet(
        metric_name=metric_name,
        task_type=task_type,
        n_runs_before=len(before_runs),
        n_runs_after=len(after_runs),
        n_tasks_compared=len(shared),
        flips=flips,
    )


# ---------------------------------------------------------------------------
# Infra evidence (latency)
# ---------------------------------------------------------------------------

@dataclass
class InfraEvidence:
    n_runs_before: int
    n_runs_after: int
    outcomes_before: dict[str, int]
    outcomes_after: dict[str, int]
    median_latency_before_ms: Optional[float]
    median_latency_after_ms: Optional[float]

    @staticmethod
    def _failure_rate(outcomes: dict[str, int]) -> Optional[float]:
        total = sum(outcomes.values())
        return None if total == 0 else 1 - outcomes.get("ok", 0) / total

    @property
    def failure_rate_before(self) -> Optional[float]:
        return self._failure_rate(self.outcomes_before)

    @property
    def failure_rate_after(self) -> Optional[float]:
        return self._failure_rate(self.outcomes_after)

    @property
    def summary(self) -> str:
        def ms(v: Optional[float]) -> str:
            return "n/a" if v is None else f"{v:,.0f} ms"

        def pct(v: Optional[float]) -> str:
            return "n/a" if v is None else f"{v:.0%}"

        return (
            f"Median latency {ms(self.median_latency_before_ms)} -> {ms(self.median_latency_after_ms)}; "
            f"failed calls (error, timeout, rate-limited) {pct(self.failure_rate_before)} -> "
            f"{pct(self.failure_rate_after)} "
            f"(last {self.n_runs_before} runs before vs first {self.n_runs_after} after)."
        )


def infra_evidence(
    store: DriftStore,
    provider: str,
    model_id: str,
    change_time: datetime,
    n_runs: int = DEFAULT_RUNS_EACH_SIDE,
) -> Optional[InfraEvidence]:
    before_runs, after_runs = runs_around(store, provider, model_id, change_time, n_runs)
    if not before_runs or not after_runs:
        return None
    before, after = _logs(store, before_runs), _logs(store, after_runs)

    def median_ok_latency(rows: list[LogRow]) -> Optional[float]:
        values = [r.latency_ms for r in rows if r.outcome == "ok" and r.latency_ms is not None]
        return statistics.median(values) if values else None

    return InfraEvidence(
        n_runs_before=len(before_runs),
        n_runs_after=len(after_runs),
        outcomes_before=dict(Counter(r.outcome for r in before)),
        outcomes_after=dict(Counter(r.outcome for r in after)),
        median_latency_before_ms=median_ok_latency(before),
        median_latency_after_ms=median_ok_latency(after),
    )


# ---------------------------------------------------------------------------
# Everything for one incident
# ---------------------------------------------------------------------------

Evidence = Union[FlipSet, InfraEvidence]


def evidence_for_incident(
    store: DriftStore,
    incident: Incident,
    n_runs: int = DEFAULT_RUNS_EACH_SIDE,
) -> dict[str, Evidence]:
    """
    One piece of evidence per metric in the incident, measured around the
    earliest time that metric moved. Metrics without enough runs on both
    sides are left out.
    """
    result: dict[str, Evidence] = {}
    seen: set[str] = set()
    for member in incident.members:
        metric = member.event.metric_name
        if metric in seen:
            continue  # members are in time order, so the first one is the earliest
        seen.add(metric)
        if metric.startswith("latency"):
            found = infra_evidence(store, incident.provider, incident.model_id, member.time, n_runs)
        else:
            found = flip_set(store, incident.provider, incident.model_id, metric, member.time, n_runs)
        if found is not None:
            result[metric] = found
    return result
