"""
driftguard/detection/bootstrap.py

Statistical significance filter for PELT's candidate change points
(changepoint_pelt.py). PELT proposes candidates liberally by design --
this module decides which ones are bigger than ordinary noise would
explain, using a permutation test rather than a parametric one (no
assumption that any metric is normally distributed -- important, since
we already know accuracy/format/refusal are small-n proportions and
latency is skewed).

Method: permutation test on the difference of means.
  1. Take the observed pre-segment and post-segment values around a
     candidate change point.
  2. Compute the observed delta (post_mean - pre_mean).
  3. Pool both segments together, then repeatedly (n_permutations times)
     shuffle the pooled values and re-split them into two groups of the
     SAME SIZES as the real pre/post segments, computing the delta each
     time.
  4. The p-value is the fraction of permuted deltas at least as extreme
     (in the adverse direction) as the observed one. If shuffling the
     data essentially never produces a gap this big, the observed split
     probably isn't due to chance -- if shuffled data regularly produces
     gaps this big, the observed split doesn't tell us much.

This module only computes p-values for candidates already flagged
`is_adverse=True` by changepoint_pelt.py -- a candidate where the metric
moved in the GOOD direction is never something we'd want to alert on,
regardless of its p-value, so there's no reason to spend permutations on
it.

Multiple-testing correction (Benjamini-Hochberg) across the several
metrics tested per run is intentionally NOT done here -- that's
multiple_testing.py's job, applied on top of the p-values this module
produces. Keeping them separate means this module answers one honest
question ("is this one shift bigger than shuffling would produce") and
the correction step answers a different one ("given we ran 4 such tests,
how many of the 'significant' ones are we confident are not false
alarms").
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from driftguard.detection.changepoint_pelt import ChangePointCandidate

DEFAULT_N_PERMUTATIONS = 2000
DEFAULT_SEED = 42  # fixed for reproducibility -- same rationale as the frozen eval-suite subsets


@dataclass
class SignificanceResult:
    candidate: ChangePointCandidate
    observed_delta: float
    p_value: float
    n_permutations: int
    significant_at_05: bool  # convenience flag at the conventional (uncorrected) 0.05 level


def permutation_test(
    pre_values: list[float],
    post_values: list[float],
    direction: str,  # "lower_is_bad" or "upper_is_bad" -- from METRIC_DIRECTIONS
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    """
    Core permutation test. Returns (observed_delta, p_value).

    p_value is one-sided, in the ADVERSE direction only -- we don't care
    whether the metric improved by an implausibly large margin, only
    whether it got worse by one. This roughly doubles statistical power
    for the direction that actually matters, compared to a two-sided
    test, at no extra cost in permutations.
    """
    if len(pre_values) < 2 or len(post_values) < 2:
        raise ValueError("Need at least 2 values in each segment to run a permutation test")

    rng = random.Random(seed)
    observed_delta = (sum(post_values) / len(post_values)) - (sum(pre_values) / len(pre_values))

    pooled = list(pre_values) + list(post_values)
    n_pre = len(pre_values)
    n_total = len(pooled)

    count_as_extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(pooled)
        perm_pre = pooled[:n_pre]
        perm_post = pooled[n_pre:]
        perm_delta = (sum(perm_post) / len(perm_post)) - (sum(perm_pre) / len(perm_pre))

        if direction == "lower_is_bad":
            # adverse = delta is very negative; "as extreme" means <= observed (which is negative)
            if perm_delta <= observed_delta:
                count_as_extreme += 1
        else:  # upper_is_bad
            if perm_delta >= observed_delta:
                count_as_extreme += 1

    p_value = count_as_extreme / n_permutations
    return observed_delta, p_value


def evaluate_candidate(
    candidate: ChangePointCandidate,
    pre_values: list[float],
    post_values: list[float],
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
) -> SignificanceResult:
    """
    Run the permutation test for one candidate, using its own pre/post
    raw values (NOT just the means already stored on the candidate --
    the test needs the actual per-run values to shuffle).
    """
    observed_delta, p_value = permutation_test(
        pre_values, post_values, direction=candidate.direction,
        n_permutations=n_permutations, seed=seed,
    )
    return SignificanceResult(
        candidate=candidate,
        observed_delta=observed_delta,
        p_value=p_value,
        n_permutations=n_permutations,
        significant_at_05=p_value < 0.05,
    )


def evaluate_candidates(
    candidates: list[ChangePointCandidate],
    full_series_values: list[float],
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
    only_adverse: bool = True,
) -> list[SignificanceResult]:
    """
    Evaluate a list of candidates (as produced by
    changepoint_pelt.find_changepoints_for_metric) against the FULL
    series they were detected in. `full_series_values` must be the exact
    same values array passed to PELT, in the same order -- candidate.index
    is used to re-slice it into pre/post segments for the permutation test.

    only_adverse=True (default) skips candidates where is_adverse is
    False, per the module's rationale above -- no reason to test
    something we'd never alert on regardless of significance.
    """
    results = []
    for candidate in candidates:
        if only_adverse and not candidate.is_adverse:
            continue

        pre_values = full_series_values[: candidate.index]
        post_values = full_series_values[candidate.index :]

        if len(pre_values) < 2 or len(post_values) < 2:
            continue  # too close to an edge -- changepoint_pelt.py should already filter these, but be defensive

        results.append(
            evaluate_candidate(candidate, pre_values, post_values, n_permutations=n_permutations, seed=seed)
        )

    return results