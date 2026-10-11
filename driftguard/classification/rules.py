"""
driftguard/classification/rules.py

Turns a detected change point into a named kind of behavioural drift.

The four-way taxonomy (from the proposal's monitoring-flow slide)
-----------------------------------------------------------------
    refusal rate  goes UP    ->  safety_shift
    accuracy      goes DOWN  ->  capability_shift
    format        goes DOWN  ->  format_drift
    p95 latency   goes UP    ->  infra_change

A change in the healthy direction (accuracy up, latency down, ...) is NOT
drift and never gets a label. A metric outside these four never gets a
label either.

Two extra readings, added on top of the label
---------------------------------------------
The label says WHAT kind of change happened. Two more readings say how
much to trust it and how big it is, so the dashboard can rank alerts
instead of treating every one the same:

1. Evidence tier: how much statistical backing the event has.
     early_warning : live Page-Hinkley alert. Fast, but only a threshold
                     crossing, with no p-value behind it.
     unconfirmed   : PELT found a change but it failed the significance
                     test (or failed after multiple-testing correction).
     provisional   : PELT + permutation test passed, but the
                     Benjamini-Hochberg correction has not run yet.
     confirmed     : passed the permutation test AND the correction.
   (Phase 2 adds one more tier, "corroborated", for when the live and
   batch detectors both flag the same metric close in time; that needs
   to look at several events together, so it does not live here.)

2. Size in noise units: the change divided by how much that metric
   normally wobbles, using the robust spread Person 1's calibration
   already measured (baseline.py). "Accuracy fell 0.17" means little on
   its own; "accuracy fell 8.5 noise units" says it is far outside
   normal variation. This matters because on long histories even tiny
   wiggles can test "significant" (the deck's slide 16 caveat).

Nothing here changes the database schema: the label is written to the
existing drift_type column; tier and size are computed when needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from driftguard.storage.store import ChangePointEventRow, DriftStore

Direction = Literal["lower_is_bad", "upper_is_bad"]

# ---------------------------------------------------------------------------
# Drift labels (stored in ChangePointEventRow.drift_type)
# ---------------------------------------------------------------------------

SAFETY_SHIFT = "safety_shift"
CAPABILITY_SHIFT = "capability_shift"
FORMAT_DRIFT = "format_drift"
INFRA_CHANGE = "infra_change"


@dataclass(frozen=True)
class DriftRule:
    """One row of the taxonomy: which metric, which direction is bad, which label."""
    metric_name: str
    bad_direction: Direction
    drift_type: str
    display_name: str   # human-readable label for the dashboard
    metric_label: str   # how to name the metric in a sentence


TAXONOMY: dict[str, DriftRule] = {
    "refusal_rate": DriftRule("refusal_rate", "upper_is_bad", SAFETY_SHIFT, "Safety shift", "Refusal rate"),
    "accuracy": DriftRule("accuracy", "lower_is_bad", CAPABILITY_SHIFT, "Capability shift", "Accuracy"),
    "format_adherence": DriftRule("format_adherence", "lower_is_bad", FORMAT_DRIFT, "Format drift", "Format adherence"),
    "latency_p95_ms": DriftRule("latency_p95_ms", "upper_is_bad", INFRA_CHANGE, "Infra change", "p95 latency"),
}

# ---------------------------------------------------------------------------
# Evidence tiers
# ---------------------------------------------------------------------------

EARLY_WARNING = "early_warning"
UNCONFIRMED = "unconfirmed"
PROVISIONAL = "provisional"
CONFIRMED = "confirmed"
UNKNOWN_TIER = "unknown"

# Below this spread a metric was effectively constant during calibration,
# so "noise units" would divide by ~zero and give a meaningless huge number.
# baseline.py floors a perfectly constant metric's spread at 1e-6.
MIN_USABLE_SIGMA = 1e-5


@dataclass(frozen=True)
class Classification:
    """Everything rules.py concludes about one event."""
    event_id: Optional[int]
    metric_name: str
    detector: str
    drift_type: Optional[str]       # one of the four labels, or None
    adverse: Optional[bool]         # True = moved the bad way, False = good way, None = cannot tell
    evidence_tier: str
    change: Optional[float]         # post_mean - pre_mean, in the metric's own units
    noise_units: Optional[float]    # |change| / normal spread, or None if not measurable
    reason: str                     # one plain-English sentence for the dashboard


# ---------------------------------------------------------------------------
# Step 1: did the metric move the bad way?
# ---------------------------------------------------------------------------

def is_adverse(event: ChangePointEventRow) -> Optional[bool]:
    """
    True if the event moved its metric in the bad direction, False if in
    the good direction, None if we cannot tell (unknown metric, or a
    PELT event missing its before/after averages).
    """
    rule = TAXONOMY.get(event.metric_name)
    if rule is None:
        return None

    # The live detector only ever fires on a "higher = worse" stream
    # (changepoint_sequential.py flips accuracy to 1 - accuracy before
    # feeding it), so any live alert is in the bad direction by design.
    if event.detector == "sequential":
        return True

    if event.pre_mean is None or event.post_mean is None:
        return None
    if rule.bad_direction == "lower_is_bad":
        return event.post_mean < event.pre_mean
    return event.post_mean > event.pre_mean


# ---------------------------------------------------------------------------
# Step 2: how much statistical backing does it have?
# ---------------------------------------------------------------------------

def evidence_tier(event: ChangePointEventRow) -> str:
    if event.detector == "sequential":
        return EARLY_WARNING
    if event.detector == "pelt":
        if not event.significant:
            return UNCONFIRMED
        if event.p_value_corrected is None:
            return PROVISIONAL
        return CONFIRMED
    return UNKNOWN_TIER


# ---------------------------------------------------------------------------
# Step 3: how big is it compared with normal noise?
# ---------------------------------------------------------------------------

def noise_sigmas_from_calibration(calibration: dict) -> dict[str, float]:
    """
    Read each metric's normal spread out of a calibration dict (the
    contents of configs/detectors_calibrated.json, as returned by
    baseline.load_calibration()).

    Uses baseline.py's own _robust_sigma rule, so "noise" means exactly
    the same thing here as it does inside the detectors.
    """
    from driftguard.detection.baseline import NoiseProfile, _robust_sigma

    sigmas: dict[str, float] = {}
    for metric_name, info in calibration.get("thresholds", {}).items():
        profile = NoiseProfile(**info["based_on"])
        sigmas[metric_name] = _robust_sigma(profile)
    return sigmas


def noise_units(event: ChangePointEventRow, sigmas: Optional[dict[str, float]]) -> Optional[float]:
    """
    |post_mean - pre_mean| divided by the metric's normal spread.
    None when it cannot be measured honestly: no calibration, a live
    event (no before value), or a metric that never varied at all.
    """
    if not sigmas or event.pre_mean is None or event.post_mean is None:
        return None
    sigma = sigmas.get(event.metric_name)
    if sigma is None or sigma < MIN_USABLE_SIGMA:
        return None
    return abs(event.post_mean - event.pre_mean) / sigma


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------

def _fmt(value: float, metric_name: str) -> str:
    if metric_name.startswith("latency"):
        return f"{value:,.0f} ms"
    return f"{value:.2f}"


def _evidence_phrase(event: ChangePointEventRow, tier: str) -> str:
    if tier == EARLY_WARNING:
        return "early warning from the live detector (no p-value)"
    if tier == CONFIRMED:
        return f"confirmed (corrected p = {event.p_value_corrected:.3g})"
    if tier == PROVISIONAL:
        return f"provisional (p = {event.p_value_raw:.3g}, correction not yet applied)"
    if tier == UNCONFIRMED:
        if event.p_value_corrected is not None:
            return f"unconfirmed (corrected p = {event.p_value_corrected:.3g})"
        if event.p_value_raw is not None:
            return f"unconfirmed (p = {event.p_value_raw:.3g})"
        return "unconfirmed"
    return f"unknown detector '{event.detector}'"


def classify_event(
    event: ChangePointEventRow,
    sigmas: Optional[dict[str, float]] = None,
) -> Classification:
    """Classify one event. Pure: reads the event, changes nothing."""
    rule = TAXONOMY.get(event.metric_name)
    tier = evidence_tier(event)
    adverse = is_adverse(event)
    change = (
        event.post_mean - event.pre_mean
        if event.pre_mean is not None and event.post_mean is not None
        else None
    )
    size = noise_units(event, sigmas)

    def result(drift_type: Optional[str], reason: str) -> Classification:
        return Classification(
            event_id=event.id,
            metric_name=event.metric_name,
            detector=event.detector,
            drift_type=drift_type,
            adverse=adverse,
            evidence_tier=tier,
            change=change,
            noise_units=size,
            reason=reason,
        )

    if rule is None:
        return result(None, f"Metric '{event.metric_name}' is outside the four-way taxonomy; not classified.")

    if adverse is None:
        return result(None, f"{rule.metric_label}: cannot tell which way it moved (missing before/after values).")

    if adverse is False:
        return result(
            None,
            f"{rule.metric_label} moved in the healthy direction "
            f"({_fmt(event.pre_mean, event.metric_name)} -> {_fmt(event.post_mean, event.metric_name)}); not drift.",
        )

    # Adverse: describe the move, then the label, then the evidence.
    if change is not None:
        verb = "rose" if change > 0 else "fell"
        move = (
            f"{rule.metric_label} {verb} from {_fmt(event.pre_mean, event.metric_name)} "
            f"to {_fmt(event.post_mean, event.metric_name)}"
        )
        if size is not None:
            move += f" ({size:.1f} noise units)"
    else:
        move = f"{rule.metric_label} crossed the live alert threshold"

    reason = f"{move}: {rule.display_name.lower()}. Evidence: {_evidence_phrase(event, tier)}."
    return result(rule.drift_type, reason)


def label_event(
    event: ChangePointEventRow,
    sigmas: Optional[dict[str, float]] = None,
) -> Classification:
    """Classify an event AND write the label into its drift_type field (not saved yet)."""
    c = classify_event(event, sigmas)
    event.drift_type = c.drift_type
    return c


def classify_open_events(
    store: DriftStore,
    sigmas: Optional[dict[str, float]] = None,
    overwrite: bool = False,
) -> list[Classification]:
    """
    Label every open event (significant and not yet reviewed) in the
    database and save the labels. Events that already have a drift_type
    are left alone unless overwrite=True, so a label a person corrected
    by hand is never silently replaced.
    """
    results: list[Classification] = []
    for event in store.get_open_events():
        if event.drift_type is not None and not overwrite:
            continue
        c = label_event(event, sigmas)
        store.update_changepoint_event(event)
        results.append(c)
    return results
