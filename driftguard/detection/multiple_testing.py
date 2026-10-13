"""
driftguard/detection/multiple_testing.py

Benjamini-Hochberg (BH) correction for the p-values of change points tested
together.

Why it is needed: PELT searches a whole per-call history and proposes several
candidate change points, on several signals at once, and each gets its own
permutation test (bootstrap.py). Testing many candidates at alpha = 0.05 makes
it likely that at least one passes by chance even on stable data. BH controls
the FALSE DISCOVERY RATE across the batch -- less conservative than Bonferroni,
so real drift on a single signal is not needlessly missed. Procedure:

  1. sort the p-values ascending: p(1) <= p(2) <= ... <= p(m)
  2. find the largest k with p(k) <= (k / m) * alpha
  3. call significant every hypothesis ranked <= k; equivalently, a hypothesis
     is significant when its BH-adjusted p-value is <= alpha.

Used by bootstrap.confirm_changepoints, whose output fills the p_value_raw,
p_value_corrected and significant fields of ChangePointEventRow.
"""

from __future__ import annotations

import numpy as np

DEFAULT_ALPHA = 0.05


def benjamini_hochberg(p_values: list[float], alpha: float = DEFAULT_ALPHA) -> tuple[np.ndarray, np.ndarray]:
    """Return (significant_mask, adjusted_p_values) for raw p-values, in input order."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)

    order = np.argsort(p)
    ranked = p[order]
    ranks = np.arange(1, m + 1)

    # adjusted p-values, made monotone from the largest rank down
    adj_sorted = np.minimum.accumulate((ranked * m / ranks)[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adj_sorted

    # significance: largest k with p_(k) <= alpha * k / m, then reject all up to it
    below = ranked <= alpha * ranks / m
    significant = np.zeros(m, dtype=bool)
    if below.any():
        kmax = np.max(np.where(below)[0])
        significant[order[: kmax + 1]] = True
    return significant, adjusted
