"""
driftguard/classification/incidents.py

Groups related drift events into incidents.

Why this exists
---------------
rules.py labels each change-point event on its own. But one real-world
cause, such as a provider quietly swapping the model behind an endpoint,
can move several signals at once: accuracy falls, format adherence falls
an hour later, and the live detector raises its own alert on accuracy.
Read one event at a time, that looks like three separate problems. This
module reads them together as one incident and reports:

  - which channels (drift types) moved together,
  - which moved first,
  - which moved most (in noise units, from rules.py),
  - which metrics both detectors flagged (live + batch), and
  - the strongest evidence tier in the incident.

It reports facts only. Turning a pattern into a suggested explanation
or action is remediation/mapper.py's job.

The grouping rule
-----------------
An incident is every LABELLED event on one (provider, model) whose
change-point time falls within WINDOW (default 2 hours) of the
incident's first event. Events are processed in time order; an event
that is too late starts a new incident. Unlabelled events (improvements,
unknown metrics) are left out because they are not problems.

Window choice: the eval suite runs about once an hour, so PELT can only
place a change to the nearest run, and one cause can surface a run late
on some signals. Two hours absorbs that slack without merging problems
from different days. It is a parameter, not a constant.

The "corroborated" tier
-----------------------
A metric is corroborated when, inside one incident, the live detector
(Page-Hinkley) fired on it AND PELT found it with passing statistics
(confirmed or provisional). A PELT result that failed its significance
test never upgrades anything: two weak signals do not make a strong one.

Evidence order, weakest to strongest:
    unconfirmed < early_warning < provisional < confirmed < corroborated
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from driftguard.classification import rules
from driftguard.storage.store import ChangePointEventRow, DriftStore

DEFAULT_WINDOW = timedelta(hours=2)

CORROBORATED = "corroborated"

TIER_ORDER = [
    rules.UNKNOWN_TIER,
    rules.UNCONFIRMED,
    rules.EARLY_WARNING,
    rules.PROVISIONAL,
    rules.CONFIRMED,
    CORROBORATED,
]


def tier_rank(tier: str) -> int:
    """Position of a tier in TIER_ORDER (higher = stronger evidence)."""
    return TIER_ORDER.index(tier) if tier in TIER_ORDER else 0


def _as_utc(moment: datetime) -> datetime:
    """
    Make every timestamp comparable. Older sqlmodel versions store times
    without a timezone; newer ones require one. Python refuses to compare
    the two kinds, so times without a timezone are treated as UTC (which
    is what the eval runner and detectors write).
    """
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def event_time(event: ChangePointEventRow) -> datetime:
    """When the change happened: the detector's estimate, else when it was detected."""
    return _as_utc(event.changepoint_timestamp or event.detected_at)


@dataclass
class IncidentEvent:
    """One event inside an incident, with its rules.py reading attached."""
    event: ChangePointEventRow
    classification: rules.Classification
    time: datetime


@dataclass
class Incident:
    provider: str
    model_id: str
    members: list[IncidentEvent] = field(default_factory=list)

    # ----- basic facts ------------------------------------------------------

    @property
    def start(self) -> datetime:
        return self.members[0].time

    @property
    def end(self) -> datetime:
        return self.members[-1].time

    @property
    def incident_id(self) -> str:
        return f"{self.provider}/{self.model_id}@{self.start.strftime('%Y-%m-%dT%H:%M')}"

    @property
    def channels(self) -> list[str]:
        """Drift types involved, in the order they first appeared."""
        seen: list[str] = []
        for m in self.members:
            if m.classification.drift_type not in seen:
                seen.append(m.classification.drift_type)
        return seen

    @property
    def is_multi_channel(self) -> bool:
        return len(self.channels) > 1

    # ----- order and size ---------------------------------------------------

    @property
    def first_channel(self) -> str:
        """Drift type of the earliest event (members are kept in time order)."""
        return self.members[0].classification.drift_type

    @property
    def largest(self) -> Optional[IncidentEvent]:
        """The member with the biggest move in noise units, or None if none were measurable."""
        measured = [m for m in self.members if m.classification.noise_units is not None]
        if not measured:
            return None
        return max(measured, key=lambda m: m.classification.noise_units)

    # ----- evidence ---------------------------------------------------------

    @property
    def corroborated_metrics(self) -> list[str]:
        """Metrics flagged by the live detector AND by a passing PELT result."""
        result: list[str] = []
        for metric in dict.fromkeys(m.event.metric_name for m in self.members):
            same = [m for m in self.members if m.event.metric_name == metric]
            live = any(m.event.detector == "sequential" for m in same)
            batch_passed = any(
                m.event.detector == "pelt"
                and m.classification.evidence_tier in (rules.CONFIRMED, rules.PROVISIONAL)
                for m in same
            )
            if live and batch_passed:
                result.append(metric)
        return result

    def metric_evidence(self, metric_name: str) -> str:
        """Strongest evidence for one metric inside this incident."""
        if metric_name in self.corroborated_metrics:
            return CORROBORATED
        tiers = [m.classification.evidence_tier for m in self.members if m.event.metric_name == metric_name]
        return max(tiers, key=tier_rank) if tiers else rules.UNKNOWN_TIER

    @property
    def evidence(self) -> str:
        """Strongest evidence tier anywhere in the incident."""
        metrics = dict.fromkeys(m.event.metric_name for m in self.members)
        return max((self.metric_evidence(name) for name in metrics), key=tier_rank)

    # ----- the sentence shown to people -------------------------------------

    @property
    def headline(self) -> str:
        names = [_display(c) for c in self.channels]
        evidence = self.evidence.replace("_", " ")
        if self.corroborated_metrics:
            evidence += f" (live and batch detectors agree on {', '.join(self.corroborated_metrics)})"

        if not self.is_multi_channel:
            return f"{names[0]} on {self.provider} / {self.model_id}. Evidence: {evidence}."

        minutes = round((self.end - self.start).total_seconds() / 60)
        parts = [
            f"{len(names)} signals moved together within {minutes} minutes: {', '.join(names).lower()}.",
            f"First: {_display(self.first_channel).lower()}.",
        ]
        if self.largest is not None:
            parts.append(
                f"Largest: {_display(self.largest.classification.drift_type).lower()} "
                f"({self.largest.classification.noise_units:.1f} noise units)."
            )
        parts.append(f"Evidence: {evidence}.")
        return " ".join(parts)


def _display(drift_type: str) -> str:
    for rule in rules.TAXONOMY.values():
        if rule.drift_type == drift_type:
            return rule.display_name
    return drift_type


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def group_into_incidents(
    events: list[ChangePointEventRow],
    sigmas: Optional[dict[str, float]] = None,
    window: timedelta = DEFAULT_WINDOW,
) -> list[Incident]:
    """
    Apply the grouping rule to a list of events. Returns incidents sorted
    by start time (oldest first). Does not change the events.
    """
    # Step 1: classify, keep only labelled events.
    labelled: list[IncidentEvent] = []
    for event in events:
        c = rules.classify_event(event, sigmas)
        if c.drift_type is not None:
            labelled.append(IncidentEvent(event=event, classification=c, time=event_time(event)))

    # Step 2: split by (provider, model).
    by_model: dict[tuple[str, str], list[IncidentEvent]] = {}
    for item in labelled:
        by_model.setdefault((item.event.provider, item.event.model_id), []).append(item)

    # Steps 3 and 4: sort by time, then walk through opening incidents.
    incidents: list[Incident] = []
    for (provider, model_id), items in by_model.items():
        items.sort(key=lambda m: m.time)
        current: Optional[Incident] = None
        for item in items:
            if current is None or item.time - current.start > window:
                current = Incident(provider=provider, model_id=model_id)
                incidents.append(current)
            current.members.append(item)

    incidents.sort(key=lambda inc: inc.start)
    return incidents


def incidents_from_store(
    store: DriftStore,
    sigmas: Optional[dict[str, float]] = None,
    window: timedelta = DEFAULT_WINDOW,
) -> list[Incident]:
    """Group every open event (significant, not yet reviewed) in the database."""
    return group_into_incidents(store.get_open_events(), sigmas=sigmas, window=window)
