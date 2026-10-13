"""
driftguard/remediation/mapper.py

What to do about an incident.

Phases 1 and 2 (classification/rules.py, classification/incidents.py)
say WHAT happened. This module turns that into a plan a person can
follow. It has three parts:

1. Advice per drift type: what the label means and concrete checks.

2. Urgency, set ONLY by the incident's evidence tier:
       unconfirmed   -> note        (record it; may be chance)
       early_warning -> watch       (re-run the eval suite, wait for PELT)
       provisional   -> investigate (start the checks)
       confirmed     -> act         (do the checks, warn users)
       corroborated  -> act now     (live and batch detectors agree)

3. Possible explanations when several signals moved together. These are
   phrased as things to check, never as proven causes. They are ordered
   by one principle: rule out the cheapest explanation first. An
   infrastructure problem is checked before a model change because it
   is quick to confirm (status page, error counts) and because timeouts
   and cut-off answers can themselves lower accuracy and format scores.

The full plan is built when it is needed (for the dashboard). A one-line
version is also written into ChangePointEventRow.remediation so the
database row carries the recommended action on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

from driftguard.classification import rules
from driftguard.classification.incidents import (
    CORROBORATED,
    DEFAULT_WINDOW,
    Incident,
    incidents_from_store,
)
from driftguard.storage.store import DriftStore

# ---------------------------------------------------------------------------
# 1. Advice per drift type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Advice:
    meaning: str
    checks: tuple[str, ...]


ADVICE: dict[str, Advice] = {
    rules.SAFETY_SHIFT: Advice(
        meaning="The model refuses harmless requests more often than before.",
        checks=(
            "Read the flip set: which safe prompts are now refused, and did any unsafe prompts stop being refused?",
            "Check the provider's changelog or announcements for safety or policy updates.",
            "Test your own product's prompts; if they are affected, adjust the system prompt or pin an earlier model version.",
        ),
    ),
    rules.CAPABILITY_SHIFT: Advice(
        meaning="The model answers the fixed benchmark questions correctly less often.",
        checks=(
            "Read the flip set: which questions went from right to wrong, and do they share a subject?",
            "Check whether the provider updated, quantized or re-routed the model behind this name (changelog, model-version fields in responses).",
            "Run the same questions against another provider serving the same model to see whether the drop follows the model or the provider.",
        ),
    ),
    rules.FORMAT_DRIFT: Advice(
        meaning="Responses fail the structured-output schema checks more often.",
        checks=(
            "Read the flip set to see how the outputs now break (extra text, missing fields, invalid JSON).",
            "Check whether your application's parsing code would fail on these outputs.",
            "Tighten the format instructions, or use the provider's structured-output / JSON mode if it has one.",
        ),
    ),
    rules.INFRA_CHANGE: Advice(
        meaning="Responses take noticeably longer (p95 latency).",
        checks=(
            "Check the provider's status page for incidents in this time window.",
            "Compare error, timeout and rate-limit counts before and after the change (see the evidence).",
            "Check whether the slowdown affects every task type or only long answers.",
        ),
    ),
}

# ---------------------------------------------------------------------------
# 2. Urgency from evidence
# ---------------------------------------------------------------------------

NOTE, WATCH, INVESTIGATE, ACT, ACT_NOW = "note", "watch", "investigate", "act", "act now"

URGENCY_BY_TIER: dict[str, str] = {
    rules.UNKNOWN_TIER: NOTE,
    rules.UNCONFIRMED: NOTE,
    rules.EARLY_WARNING: WATCH,
    rules.PROVISIONAL: INVESTIGATE,
    rules.CONFIRMED: ACT,
    CORROBORATED: ACT_NOW,
}

URGENCY_STEP: dict[str, str] = {
    NOTE: "No action needed yet: the change did not pass its significance test and may be chance. Keep monitoring.",
    WATCH: "Run the eval suite again now instead of waiting for the schedule, then check whether the batch detector (PELT) confirms the change.",
    INVESTIGATE: "The change passed its significance test but the multiple-testing correction has not run yet. Start the checks below.",
    ACT: "The change is statistically confirmed. Do the checks below and tell the people who depend on this model.",
    ACT_NOW: "The live and batch detectors both flagged this. Treat it as real: do the checks below now and tell the people who depend on this model.",
}

# ---------------------------------------------------------------------------
# 3. Possible explanations for multi-signal incidents
# ---------------------------------------------------------------------------


def possible_explanations(channels: list[str]) -> list[str]:
    """
    Ordered hypotheses to check for an incident whose signals moved
    together. Empty for a single-signal incident: one signal alone says
    nothing about co-movement.
    """
    c = set(channels)
    if len(c) < 2:
        return []

    hints: list[str] = []
    if rules.INFRA_CHANGE in c:
        hints.append(
            "Latency moved too. Rule out provider infrastructure first: timeouts and cut-off answers "
            "can also lower accuracy and format scores."
        )
    if {rules.CAPABILITY_SHIFT, rules.FORMAT_DRIFT} <= c and rules.INFRA_CHANGE not in c:
        hints.append(
            "Accuracy and format moved together while latency stayed steady: consistent with the "
            "provider updating or swapping the model behind the same name."
        )
    if {rules.SAFETY_SHIFT, rules.CAPABILITY_SHIFT} <= c:
        hints.append(
            "Refusals rose as accuracy fell: consistent with a safety-tuning update; extra refusals "
            "can also cost points on borderline questions."
        )
    if not hints:
        hints.append("Several signals moved together; treat them as one cause until the checks show otherwise.")
    return hints


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RemediationPlan:
    incident_id: str
    urgency: str
    headline: str                       # from incidents.py
    meanings: tuple[str, ...]           # one line per drift type involved
    possible_explanations: tuple[str, ...]
    steps: tuple[str, ...]              # urgency step first, then each drift type's checks

    @property
    def one_line(self) -> str:
        """Short form stored in ChangePointEventRow.remediation."""
        return f"[{self.urgency.upper()}] {self.steps[0]}"


def plan_for_incident(incident: Incident) -> RemediationPlan:
    urgency = URGENCY_BY_TIER.get(incident.evidence, NOTE)
    steps: list[str] = [URGENCY_STEP[urgency]]
    meanings: list[str] = []
    for drift_type in incident.channels:
        advice = ADVICE[drift_type]
        meanings.append(advice.meaning)
        for check in advice.checks:
            if check not in steps:
                steps.append(check)
    return RemediationPlan(
        incident_id=incident.incident_id,
        urgency=urgency,
        headline=incident.headline,
        meanings=tuple(meanings),
        possible_explanations=tuple(possible_explanations(incident.channels)),
        steps=tuple(steps),
    )


def apply_remediation(
    store: DriftStore,
    sigmas: Optional[dict[str, float]] = None,
    window: timedelta = DEFAULT_WINDOW,
    overwrite: bool = False,
) -> int:
    """
    Group the open events into incidents and write each incident's
    one-line plan into its events' remediation column (and the drift
    label into drift_type, if missing). Existing text is kept unless
    overwrite=True, so a note a reviewer edited is never replaced.
    Returns how many events were updated.
    """
    updated = 0
    for incident in incidents_from_store(store, sigmas=sigmas, window=window):
        line = plan_for_incident(incident).one_line
        for member in incident.members:
            event = member.event
            changed = False
            if event.drift_type is None:
                event.drift_type = member.classification.drift_type
                changed = True
            if event.remediation is None or overwrite:
                event.remediation = line
                changed = True
            if changed:
                store.update_changepoint_event(event)
                updated += 1
    return updated
