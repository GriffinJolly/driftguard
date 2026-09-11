from __future__ import annotations
 
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
 
import numpy as np
import ruptures as rpt
 
from driftguard.detection.calibration import METRIC_DIRECTIONS, Direction
from driftguard.storage.store import MetricSeriesRow
 
# Cost model for ruptures.Pelt. "l2" (assumes piecewise-constant mean,
# Gaussian-ish noise within each segment) is used deliberately -- it's
# the cost model the BIC-style penalty (sigma^2 * log(n), computed in
# calibration.py) is actually derived for. "rbf" was tried first but
# produces a mismatched cost scale relative to a variance-based penalty,
# which caused severe over-segmentation (dozens of false change points on
# pure stable noise, confirmed in testing) -- l2 doesn't have that
# mismatch, since its cost is directly variance-based like the penalty
# formula assumes. l2 is a reasonable fit for our metrics regardless:
# accuracy/format/refusal are proportions with roughly stable variance
# within a regime, and latency, while skewed, is still well-served by a
# mean-shift model for the purpose of localizing WHEN a shift happened.
DEFAULT_PELT_MODEL = "l2"
 
# PELT needs a minimum run length either side of a candidate change point
# to estimate a meaningful pre/post mean from. Below this, a "change
# point" right at the edge of the series is more likely an artifact of
# too little data than a real shift.
MIN_SEGMENT_LENGTH = 3
 
 
@dataclass
class ChangePointCandidate:
    metric_name: str
    index: int  # position in the input series where the new segment starts
    timestamp: Optional[datetime]  # timestamp of the point at `index`, if available
    pre_mean: float
    post_mean: float
    delta: float  # post_mean - pre_mean
    direction: Direction
    is_adverse: bool  # True if the shift moves the metric in its "bad" direction
    pre_n: int
    post_n: int
 
 
def _extract_series(points: list[MetricSeriesRow], metric_name: str) -> tuple[list[float], list[int]]:
    """
    Pull one metric's values out of a list of MetricSeriesRow, in
    timestamp order, skipping None (uncomputable) entries. Returns the
    values AND the original indices they came from, since PELT needs a
    contiguous array but we still want to map back to the right
    timestamp/row afterward.
    """
    ordered = sorted(points, key=lambda p: p.timestamp)
    values: list[float] = []
    original_indices: list[int] = []
    for i, p in enumerate(ordered):
        v = getattr(p, metric_name, None)
        if v is not None:
            values.append(v)
            original_indices.append(i)
    return values, original_indices
 
 
def detect_breakpoints(values: list[float], penalty: float, model: str = DEFAULT_PELT_MODEL) -> list[int]:
    """
    Run PELT over a 1D series. Returns breakpoint indices (each is the
    START of a new segment), excluding the trailing endpoint ruptures
    always includes (len(values)), which isn't a real change point.
    """
    if len(values) < 2 * MIN_SEGMENT_LENGTH:
        return []  # too short to say anything meaningful
 
    series = np.array(values).reshape(-1, 1)
    algo = rpt.Pelt(model=model, min_size=MIN_SEGMENT_LENGTH, jump=1).fit(series)
    breakpoints = algo.predict(pen=penalty)
 
    # ruptures always includes len(values) as the final "breakpoint" --
    # that's a sentinel marking the end of the last segment, not a real
    # change point, so drop it.
    return [bp for bp in breakpoints if bp < len(values)]
 
 
def _describe_candidate(
    metric_name: str,
    values: list[float],
    timestamps: list[Optional[datetime]],
    bp_index: int,
    direction: Direction,
) -> Optional[ChangePointCandidate]:
    pre_segment = values[:bp_index]
    post_segment = values[bp_index:]
 
    if len(pre_segment) < MIN_SEGMENT_LENGTH or len(post_segment) < MIN_SEGMENT_LENGTH:
        return None  # too close to an edge to trust the pre/post means
 
    pre_mean = float(np.mean(pre_segment))
    post_mean = float(np.mean(post_segment))
    delta = post_mean - pre_mean
 
    if direction == "lower_is_bad":
        is_adverse = delta < 0
    else:
        is_adverse = delta > 0
 
    ts = timestamps[bp_index] if bp_index < len(timestamps) else None
 
    return ChangePointCandidate(
        metric_name=metric_name,
        index=bp_index,
        timestamp=ts,
        pre_mean=pre_mean,
        post_mean=post_mean,
        delta=delta,
        direction=direction,
        is_adverse=is_adverse,
        pre_n=len(pre_segment),
        post_n=len(post_segment),
    )
 
 
def find_changepoints_for_metric(
    points: list[MetricSeriesRow],
    metric_name: str,
    penalty: float,
    model: str = DEFAULT_PELT_MODEL,
) -> list[ChangePointCandidate]:
    """
    Full pipeline for one metric: extract the series, run PELT, describe
    every resulting candidate change point (pre/post means, direction,
    whether it's adverse).
    """
    if metric_name not in METRIC_DIRECTIONS:
        raise ValueError(f"Unknown metric '{metric_name}'. Known: {list(METRIC_DIRECTIONS)}")
 
    direction = METRIC_DIRECTIONS[metric_name]
    ordered = sorted(points, key=lambda p: p.timestamp)
    values, original_indices = _extract_series(ordered, metric_name)
 
    if not values:
        return []
 
    timestamps = [ordered[i].timestamp for i in original_indices]
    breakpoints = detect_breakpoints(values, penalty=penalty, model=model)
 
    candidates = []
    for bp in breakpoints:
        candidate = _describe_candidate(metric_name, values, timestamps, bp, direction)
        if candidate is not None:
            candidates.append(candidate)
 
    return candidates
 
 
def find_changepoints_all_metrics(
    points: list[MetricSeriesRow],
    calibration: dict,
) -> dict[str, list[ChangePointCandidate]]:
    """
    Run PELT across every metric present in a loaded calibration dict
    (as produced by calibration.save_calibration / load_calibration).
    Metrics with no calibrated threshold entry are skipped, since there's
    no calibrated penalty to use for them.
    """
    results: dict[str, list[ChangePointCandidate]] = {}
    for metric_name, threshold_info in calibration.get("thresholds", {}).items():
        penalty = threshold_info["pelt_penalty"]
        results[metric_name] = find_changepoints_for_metric(points, metric_name, penalty=penalty)
    return results