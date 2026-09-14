"""
driftguard/detection/bootstrap.py

Bootstrap / permutation significance testing for candidate change points.

PELT (changepoint_pelt.py) will always return *some* split points, even in pure
noise. This module decides which of them are real: for each candidate it runs a
permutation test on the two adjacent segments -- if the observed jump in the
mean (e.g. error rate) is no bigger than what random re-labelling of the same
samples routinely produces, the change point is not significant.

Because a whole history is searched for change points, many are tested at once,
so raw p-values are corrected for multiple testing with Benjamini-Hochberg
(config: configs/detectors.yaml -> significance). The output maps 1:1 onto the
ChangePointEventRow fields (pre_mean, post_mean, p_value_raw, p_value_corrected,
significant) in driftguard/storage/store.py.

Honest caveat: on very long segments even a tiny mean difference is
"statistically significant" (large-n), so `delta` (the effect size -- how big
the jump actually is) matters as much as the p-value. The permutation test takes
a bounded `window` around each change point for exactly this reason: it asks
"did the rate change *locally and sharply here*", not "do two thousand-sample
halves differ at all".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SignificanceConfig:
    alpha: float = 0.05
    n_bootstrap_samples: int = 1000
    window: int | None = 2000  # samples each side of the change point to test; None = full segments


def changepoint_pvalue(
    signal: np.ndarray, cp: int, left_bound: int, right_bound: int,
    n_bootstrap: int = 1000, window: int | None = 2000, seed: int = 0,
) -> dict:
    """Permutation test that the mean of [left_bound, cp) differs from [cp, right_bound).

    Returns pre_mean, post_mean, delta (|post-pre|), and p_value_raw =
    P(a random re-labelling gives a jump >= the observed one).
    """
    left = np.asarray(signal[left_bound:cp], dtype=float)
    right = np.asarray(signal[cp:right_bound], dtype=float)
    if window is not None:
        left = left[-window:]
        right = right[:window]

    pre_mean, post_mean = float(left.mean()), float(right.mean())
    observed = abs(post_mean - pre_mean)

    pooled = np.concatenate([left, right])
    n_left = len(left)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_bootstrap):
        perm = rng.permutation(pooled)
        if abs(perm[n_left:].mean() - perm[:n_left].mean()) >= observed - 1e-12:
            count += 1
    # +1 smoothing so a p-value is never exactly 0 (Davison & Hinkley convention)
    p_value_raw = (1 + count) / (1 + n_bootstrap)
    return {"index": cp, "pre_mean": pre_mean, "post_mean": post_mean,
            "delta": observed, "p_value_raw": p_value_raw}


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Return (significant_mask, adjusted_p_values) for a set of raw p-values.

    Standard Benjamini-Hochberg step-up procedure controlling the false
    discovery rate at `alpha`.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)

    order = np.argsort(p)
    ranked = p[order]
    ranks = np.arange(1, m + 1)

    # BH-adjusted p-values (monotone from the largest rank down)
    adj_sorted = np.minimum.accumulate((ranked * m / ranks)[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adj_sorted

    # significance: largest k with p_(k) <= alpha*k/m, then reject all up to it
    below = ranked <= alpha * ranks / m
    significant = np.zeros(m, dtype=bool)
    if below.any():
        kmax = np.max(np.where(below)[0])
        significant[order[: kmax + 1]] = True
    return significant, adjusted


def confirm_changepoints(
    signal: np.ndarray, changepoints: list[int], config: SignificanceConfig = SignificanceConfig(),
) -> list[dict]:
    """Run the full confirm-or-reject step for a list of PELT change points.

    Each change point is tested against its two neighbouring segments; then all
    raw p-values are BH-corrected together. Returns one dict per change point
    with pre/post means, effect size, raw + corrected p-values, and `significant`.
    """
    n = len(signal)
    bounds = [0, *changepoints, n]
    results = []
    for i, cp in enumerate(changepoints):
        left_bound, right_bound = bounds[i], bounds[i + 2]
        results.append(
            changepoint_pvalue(signal, cp, left_bound, right_bound,
                               n_bootstrap=config.n_bootstrap_samples, window=config.window, seed=i)
        )
    if results:
        significant, adjusted = benjamini_hochberg([r["p_value_raw"] for r in results], config.alpha)
        for r, s, a in zip(results, significant, adjusted):
            r["p_value_corrected"] = float(a)
            r["significant"] = bool(s)
    return results
