from __future__ import annotations
 
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional
 
from driftguard.storage.store import DriftStore, MetricSeriesRow
 
DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "detectors_calibrated.json"
 
Direction = Literal["lower_is_bad", "upper_is_bad"]
 
METRIC_DIRECTIONS: dict[str, Direction] = {
    "accuracy": "lower_is_bad",
    "format_adherence": "lower_is_bad",
    "refusal_rate": "upper_is_bad",       # tracked on the SAFE subset -- rising = safety over-triggering
    "latency_p95_ms": "upper_is_bad",
}
 
MIN_QUIET_POINTS = 20  # below this, warn -- thresholds derived from very few points are unreliable
 
 
@dataclass
class NoiseProfile:
    metric_name: str
    n: int
    mean: float
    std: float
    median: float
    mad: float  # median absolute deviation, NOT scaled -- see mad_scaled
    mad_scaled: float  # MAD * 1.4826, makes it comparable to std under normality
    p05: float
    p95: float
    min_val: float
    max_val: float
 
 
@dataclass
class MetricThresholds:
    metric_name: str
    direction: Direction
    baseline_threshold: float          # the actual cutoff value: below/above this = flagged (baseline detector)
    pelt_penalty: float                # penalty term for ruptures.Pelt
    page_hinkley_delta: float          # allowed slack before Page-Hinkley starts accumulating
    page_hinkley_threshold: float      # cumulative deviation that triggers a live alert
    based_on: NoiseProfile
 
 
@dataclass
class CalibrationResult:
    provider: str
    model_id: str
    n_points: int
    k: float  # the sensitivity multiplier used (see calibrate())
    thresholds: dict[str, MetricThresholds]
    warnings: list[str]
 
 
def _mad(values: list[float], center: float) -> float:
    deviations = [abs(v - center) for v in values]
    return statistics.median(deviations) if deviations else 0.0
 
 
def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * pct
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)
 
 
def compute_noise_profile(metric_name: str, values: list[Optional[float]]) -> Optional[NoiseProfile]:
    """
    Build a NoiseProfile from a list of quiet-window values for one metric.
    None values (failed/uncomputable runs) are dropped. Returns None if
    fewer than 2 usable points remain -- can't estimate spread from 0-1 points.
    """
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if len(clean) < 2:
        return None
 
    sorted_vals = sorted(clean)
    mean = statistics.mean(clean)
    std = statistics.stdev(clean)
    median = statistics.median(clean)
    mad = _mad(clean, median)
 
    return NoiseProfile(
        metric_name=metric_name,
        n=len(clean),
        mean=mean,
        std=std,
        median=median,
        mad=mad,
        mad_scaled=mad * 1.4826,  # consistency constant for normally-distributed data
        p05=_percentile(sorted_vals, 0.05),
        p95=_percentile(sorted_vals, 0.95),
        min_val=sorted_vals[0],
        max_val=sorted_vals[-1],
    )
 
 
def _robust_sigma(profile: NoiseProfile) -> float:
    """
    The spread estimate used everywhere below. Falls back away from
    mad_scaled when it collapses to (near) zero -- this happens for
    lumpy, mostly-constant metrics with a dominant mode (e.g. refusal
    rate on a small safe-prompt subset, where "0 refusals" is the most
    common outcome by far). A zero-width threshold there would flag any
    single non-zero observation as significant drift, which is too
    sensitive to be useful. Falling back to a fraction of std (which
    still reflects the occasional non-zero values, unlike MAD) gives a
    less degenerate, more usable threshold; the final 1e-6 floor only
    kicks in for metrics that are PERFECTLY constant across the whole
    quiet window (no variance to fall back on at all).
    """
    if profile.mad_scaled > 1e-9:
        return profile.mad_scaled
    if profile.std > 1e-9:
        return 0.5 * profile.std
    return 1e-6
 
 
def _pelt_penalty(profile: NoiseProfile, n_points: int) -> float:
    """
    Standard BIC-style penalty for ruptures.Pelt: sigma^2 * log(n).
    Uses the robust spread estimate (see _robust_sigma) rather than raw
    std, for the same lumpy/skewed-data reasons noted in the module
    docstring.
    """
    sigma = _robust_sigma(profile)
    return max(sigma**2 * math.log(max(n_points, 2)), 1e-6)
 
 
def _page_hinkley_params(profile: NoiseProfile, k: float) -> tuple[float, float]:
    """
    delta: allowed slack per observation before it counts toward the
    cumulative deviation sum -- set to half the robust spread, a common
    starting heuristic (small enough to stay sensitive, large enough to
    absorb ordinary noise).
    threshold: cumulative deviation that triggers an alert -- scaled by
    k (the same sensitivity multiplier used for the baseline threshold),
    so tightening/loosening k affects both detectors consistently.
    """
    sigma = _robust_sigma(profile)
    delta = 0.5 * sigma
    threshold = k * sigma
    return delta, threshold
 
 
def _baseline_threshold(profile: NoiseProfile, direction: Direction, k: float) -> float:
    """
    The actual cutoff value for the naive fixed-threshold baseline
    detector (the one intentionally kept simple, used as the comparison
    point PELT/Page-Hinkley are benchmarked against). Uses median +/- k *
    robust_sigma rather than mean +/- k * std, per the module's robust-
    statistics rationale.
    """
    sigma = _robust_sigma(profile)
    if direction == "lower_is_bad":
        return profile.median - k * sigma
    return profile.median + k * sigma
 
 
def calibrate(
    quiet_points: list[MetricSeriesRow],
    provider: str,
    model_id: str,
    k: float = 3.0,
) -> CalibrationResult:
    """
    Core calibration entry point. Takes a list of MetricSeriesRow objects
    from a quiet (known-stable) window and derives thresholds for every
    metric in METRIC_DIRECTIONS.
 
    k: sensitivity multiplier (in robust-sigma units) applied uniformly
    across baseline threshold and Page-Hinkley threshold. Higher k =
    fewer false alarms but slower/weaker detection of real drift. 3.0 is
    a reasonable starting point (roughly analogous to "3-sigma" but
    computed from MAD instead of std); this is exactly the kind of
    number that should be revisited once real validation-window results
    (recall/FPR against known transitions) are available -- calibrate()
    fixes the FORM of the thresholds from real noise, k is the one
    remaining tunable knob, and it should be tuned looking at the
    validation results, not guessed twice.
    """
    warnings: list[str] = []
    if len(quiet_points) < MIN_QUIET_POINTS:
        warnings.append(
            f"Only {len(quiet_points)} quiet-window points available (recommended >= "
            f"{MIN_QUIET_POINTS}). Thresholds derived from this window may be unreliable "
            f"until more data accumulates."
        )
 
    thresholds: dict[str, MetricThresholds] = {}
 
    for metric_name, direction in METRIC_DIRECTIONS.items():
        values = [getattr(p, metric_name) for p in quiet_points]
        profile = compute_noise_profile(metric_name, values)
 
        if profile is None:
            warnings.append(f"Not enough usable data to calibrate '{metric_name}' -- skipped.")
            continue
 
        pelt_penalty = _pelt_penalty(profile, len(quiet_points))
        ph_delta, ph_threshold = _page_hinkley_params(profile, k)
        baseline_thr = _baseline_threshold(profile, direction, k)
 
        thresholds[metric_name] = MetricThresholds(
            metric_name=metric_name,
            direction=direction,
            baseline_threshold=baseline_thr,
            pelt_penalty=pelt_penalty,
            page_hinkley_delta=ph_delta,
            page_hinkley_threshold=ph_threshold,
            based_on=profile,
        )
 
    return CalibrationResult(
        provider=provider,
        model_id=model_id,
        n_points=len(quiet_points),
        k=k,
        thresholds=thresholds,
        warnings=warnings,
    )
 
 
def calibrate_from_store(
    store: DriftStore,
    provider: str,
    model_id: str,
    k: float = 3.0,
) -> CalibrationResult:
    """Convenience wrapper: pulls the full metric series for (provider, model_id) and calibrates on it."""
    quiet_points = store.get_metric_series(provider, model_id)
    return calibrate(quiet_points, provider=provider, model_id=model_id, k=k)
 
 
def save_calibration(result: CalibrationResult, path: Path = DEFAULT_CALIBRATION_PATH) -> None:
    """Persist a CalibrationResult to JSON so changepoint_pelt.py / changepoint_sequential.py / baseline.py can load locked-in thresholds without recomputing them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "provider": result.provider,
        "model_id": result.model_id,
        "n_points": result.n_points,
        "k": result.k,
        "warnings": result.warnings,
        "thresholds": {name: asdict(mt) for name, mt in result.thresholds.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
 
 
def load_calibration(path: Path = DEFAULT_CALIBRATION_PATH) -> dict:
    """Load a previously saved calibration file as a plain dict (detectors read whatever fields they need)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)