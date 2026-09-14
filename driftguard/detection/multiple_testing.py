"""
driftguard/detection/multiple_testing.py

Benjamini-Hochberg (BH) correction across the p-values bootstrap.py
produces for the several metrics tested together in one run.

Why this is needed (the problem this fixes): each of the 4 metrics
(accuracy, format_adherence, refusal_rate, latency_p95_ms) gets its own
independent significance test via bootstrap.py. If each test uses an
uncorrected threshold of alpha=0.05, then even on completely stable data
where NOTHING is actually drifting, there's roughly a 1-(1-0.05)^4 ~= 18.5%
chance that AT LEAST ONE of the 4 tests comes back "significant" purely by
chance on any given evaluation window. That's a real, measured effect --
we saw it directly in bootstrap.py's own test run, where 1 of 4 candidates
on a pure-noise series crossed the uncorrected 0.05 line. Across many
windows over a multi-week calibration period, that adds up to a lot of
false alarms unless corrected for.

BH controls the FALSE DISCOVERY RATE (not quite the same as controlling
the family-wise error rate, but appropriate here and less conservative
than e.g. a Bonferroni correction, which would needlessly reduce our
ability to detect real drift on any single metric). Procedure:
  1. Sort all candidate p-values ascending: p(1) <= p(2) <= ... <= p(m)
  2. Find the largest k such that p(k) <= (k/m) * alpha
  3. Reject (call significant) all hypotheses with rank <= k, equivalently:
     a hypothesis is significant if its BH-adjusted p-value is <= alpha.

This module operates on SignificanceResult objects from bootstrap.py --
it doesn't know or care which metric each came from, it just corrects
the whole batch of p-values from one evaluation run together, which is
exactly the right scope: the "aggregate more than alpha overall" problem
happens because multiple tests ran in the SAME run, so correction has to
happen across that same set.
"""

from __future__ import annotations

from dataclasses import dataclass

from driftguard.detection.bootstrap import SignificanceResult

DEFAULT_ALPHA = 0.05


@dataclass
class CorrectedResult:
    result: SignificanceResult
    metric_name: str
    rank: int  # 1-indexed rank among the sorted p-values in this batch
    p_value_corrected: float  # BH-adjusted p-value (see benjamini_hochberg_adjusted_pvalues)
    significant: bool  # True if this passed the BH procedure at the given alpha


def benjamini_hochberg_adjusted_pvalues(p_values: list[float]) -> list[float]:
    """
    Compute BH-adjusted p-values (sometimes called q-values) for a list
    of raw p-values, in the SAME order they were given. This is the
    standard "step-up" BH adjustment: adjusted p(i) = min over all j>=i of
    (p(j) * m / j), ensuring adjusted values are monotonically
    non-decreasing when read in sorted order (a technical requirement of
    the procedure -- without the running-minimum step, adjusted p-values
    could be non-monotonic, which doesn't make sense for a p-value).

    Returns adjusted values clipped to [0, 1] and in the original input
    order, so callers can zip them back against whatever they passed in.
    """
    m = len(p_values)
    if m == 0:
        return []

    # sort by p-value, keeping track of original positions
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    adjusted = [0.0] * m
    # step-up: work from the largest p-value down, tracking the running minimum
    running_min = 1.0
    for rank_from_end, (original_idx, p) in enumerate(reversed(indexed)):
        rank = m - rank_from_end  # 1-indexed rank from the SORTED-ascending order
        candidate = p * m / rank
        running_min = min(running_min, candidate)
        adjusted[original_idx] = min(running_min, 1.0)

    return adjusted


def apply_bh_correction(
    results: dict[str, SignificanceResult],
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, CorrectedResult]:
    """
    Apply BH correction across a dict of {metric_name: SignificanceResult}
    -- one result per metric tested in a single evaluation run. This is
    the intended call shape: bootstrap.py produces at most one
    "primary" significance result per metric per run (the most
    significant adverse candidate, if any), and those get corrected
    together here.

    Returns a dict in the same shape, with each result's corrected
    p-value and final significant/not-significant verdict attached.
    """
    if not results:
        return {}

    metric_names = list(results.keys())
    raw_p_values = [results[name].p_value for name in metric_names]
    adjusted_p_values = benjamini_hochberg_adjusted_pvalues(raw_p_values)

    # ranks, for transparency/debugging -- 1 = smallest raw p-value
    sorted_order = sorted(range(len(raw_p_values)), key=lambda i: raw_p_values[i])
    ranks = [0] * len(raw_p_values)
    for rank, idx in enumerate(sorted_order, start=1):
        ranks[idx] = rank

    corrected: dict[str, CorrectedResult] = {}
    for i, name in enumerate(metric_names):
        corrected[name] = CorrectedResult(
            result=results[name],
            metric_name=name,
            rank=ranks[i],
            p_value_corrected=adjusted_p_values[i],
            significant=adjusted_p_values[i] <= alpha,
        )

    return corrected


def apply_and_persist_bh_correction(
    store: "DriftStore",
    event_rows: dict[str, "ChangePointEventRow"],
    results: dict[str, SignificanceResult],
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, CorrectedResult]:
    """
    Full correct-and-persist step for one evaluation cycle: takes the
    ChangePointEventRow rows bootstrap.py already wrote provisionally
    (via store_significance_results -- p_value_corrected=None,
    significant=uncorrected verdict), computes the BH-corrected p-values
    across this batch, and patches the corrected p-value + FINAL
    significance verdict back into each row via update_changepoint_event.

    `event_rows` and `results` must share the same metric-name keys (as
    produced together by bootstrap.store_significance_results, which
    returns event_rows, and the `results` dict passed into it).

    This is multiple_testing.py's half of the "every detector populates
    ChangePointEventRow directly" contract -- after this call, the rows
    in the store reflect the final, corrected verdict, and nothing
    downstream needs to re-run or know about the correction step.
    """
    corrected = apply_bh_correction(results, alpha=alpha)

    for metric_name, cr in corrected.items():
        if metric_name not in event_rows:
            continue  # defensive -- shouldn't happen if called with matching dicts
        row = event_rows[metric_name]
        row.p_value_corrected = cr.p_value_corrected
        row.significant = cr.significant
        event_rows[metric_name] = store.update_changepoint_event(row)

    return corrected