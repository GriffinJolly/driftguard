"""
driftguard/detection/changepoint_pelt.py

Retrospective / batch change-point detection via PELT (Pruned Exact Linear
Time; Killick, Fearnhead & Eckley 2012), backed by the `ruptures` library.

This is the deck's "PELT: the historical confirmation" path. Unlike the online
sequential detectors (changepoint_sequential.py / Page-Hinkley), which alert the
moment drift appears in a live stream, PELT runs over a WHOLE recorded metric
history at once and returns every sustained change point it can find. Those
candidates are then confirmed or rejected by bootstrap significance testing
(driftguard/detection/bootstrap.py), so the two modules are meant to be used
together: PELT proposes, the bootstrap disposes.

Config: configs/detectors.yaml -> retrospective_detector.

Cost model note: the config originally listed `rbf`, but the RBF kernel cost
builds an n x n Gram matrix -- ~20 GB for a 52k-sample stream -- so it is not
usable on real metric histories. `l2` (piecewise-constant mean, i.e. detect a
shift in the error/accuracy RATE) is the right model here: it is O(n), and a
metric drift IS a shift in the mean, which is exactly what l2 segments on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PeltConfig:
    cost_model: str = "l2"          # "l2" (mean shift) | "normal" | "rbf" (small streams only)
    penalty: float | None = None    # None -> BIC-style default derived from the signal
    penalty_scale: float = 3.0      # multiplies the BIC default; higher -> fewer change points
    min_segment_length: int = 200   # minimum samples between change points
    jump: int = 25                  # grid resolution PELT considers; larger = much faster on long
                                    # histories, at coarser change-point precision (25 is plenty for
                                    # metric drift, and keeps a 50k-sample stream to ~15s not minutes)


def default_penalty(signal: np.ndarray, cost_model: str, penalty_scale: float) -> float:
    """A BIC-style penalty: bigger streams and noisier data need a larger
    penalty to avoid over-segmenting. For l2 the natural scale is the signal
    variance times log(n); kernel costs are ~O(1) so just log(n)."""
    n = len(signal)
    base = math.log(n)
    if cost_model in ("l2", "normal"):
        base *= float(np.var(signal)) or 1.0
    return penalty_scale * base


def detect_changepoints(signal, config: PeltConfig = PeltConfig()) -> list[int]:
    """Return the sorted interior change-point indices PELT finds in `signal`.

    Indices are positions in the stream where a new segment begins; the trailing
    endpoint (len(signal)) that ruptures always appends is dropped.
    """
    import ruptures as rpt

    x = np.asarray(signal, dtype=float).reshape(-1, 1)
    n = len(x)
    if n < 2 * config.min_segment_length:
        return []
    pen = config.penalty if config.penalty is not None else default_penalty(
        x.ravel(), config.cost_model, config.penalty_scale
    )
    algo = rpt.Pelt(model=config.cost_model, min_size=config.min_segment_length, jump=config.jump).fit(x)
    bkps = algo.predict(pen=pen)
    return [int(b) for b in bkps if 0 < b < n]


def segments_from_changepoints(n: int, changepoints: list[int]) -> list[tuple[int, int]]:
    """Turn interior change points into [start, end) segment bounds covering [0, n)."""
    bounds = [0, *changepoints, n]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
