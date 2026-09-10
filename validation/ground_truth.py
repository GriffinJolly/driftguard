"""
validation/ground_truth.py

Synthetic drift-injection framework for validating DriftGuard's detectors.

Why binary error streams
-------------------------
Every one of the 8 detectors compared on the project's "detector evaluation"
slide (Page-Hinkley, DDM, EDDM, ADWIN, HDDM_A, HDDM_W, RDDM, ECDD) is, in its
standard form, a streaming monitor over a sequence of 0/1 "was this call
wrong" outcomes -- that is exactly the shape of DriftGuard's own eval_score
field on accuracy, format-adherence, and refusal-probe calls (see
driftguard/ingest/log_schema.py: DriftLogEntry.eval_score, and
driftguard/evalsuite/*.py, which already compute a 1.0/0.0 score per call).
This module lives at the repo root alongside driftguard/ (not inside the
package) to match the existing validation/ directory layout.
So the synthetic benchmark below is not an abstract textbook exercise: a
"stream" here corresponds 1:1 to the sequence of per-call eval_score values
DriftGuard already produces for one (provider, model, metric) over time.

Four scenario families
-----------------------
1. STABLE     -- constant error rate throughout. No real drift ever occurs.
                 Used to measure the FALSE ALARM RATE: any flagged drift
                 here is, by construction, wrong.
2. SUDDEN     -- error rate jumps from p0 to p1 at a single known index.
                 This is what the original 8-detector comparison measured
                 (detection rate on "sudden drift").
3. GRADUAL    -- error rate ramps linearly from p0 to p1 over a window.
                 This is the original comparison's "gradual drift" case.
4. TRANSIENT  -- error rate rises from p0 to p1 for a short window and then
                 REVERTS back to p0. This is NOT drift -- it's the kind of
                 temporary fluctuation (a bad hour, a noisy batch) that a
                 good detector must learn to ignore. The project's own
                 pitch claims Page-Hinkley "distinguished persistent
                 degradation from temporary fluctuations more effectively
                 than ADWIN" -- this scenario is what actually tests that
                 claim instead of just asserting it.

Every scenario is parameterized by a fixed seed per trial so results are
reproducible, in keeping with the project's "frozen, versioned" eval
philosophy used elsewhere in the codebase (see evalsuite/tasks.py,
evalsuite/refusal_probe.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Scenario:
    """One synthetic trial's ground truth."""

    name: str                     # "stable" | "sudden" | "gradual" | "transient"
    stream: np.ndarray            # 0/1 array, 1 == "error"/"failed call"
    change_point: Optional[int]   # index the metric begins degrading (None for stable)
    is_real_drift: bool           # True for sudden/gradual, False for stable/transient
    magnitude: Optional[float]    # p1 - p0, None for stable
    p0: float
    p1: Optional[float]


def _bernoulli_stream(rng: np.random.Generator, probs: np.ndarray) -> np.ndarray:
    return rng.binomial(1, probs).astype(np.int8)


def make_stable(
    rng: np.random.Generator, length: int, p0: float
) -> Scenario:
    probs = np.full(length, p0)
    return Scenario(
        name="stable",
        stream=_bernoulli_stream(rng, probs),
        change_point=None,
        is_real_drift=False,
        magnitude=None,
        p0=p0,
        p1=None,
    )


def make_sudden(
    rng: np.random.Generator, length: int, change_point: int, p0: float, p1: float
) -> Scenario:
    probs = np.full(length, p0)
    probs[change_point:] = p1
    return Scenario(
        name="sudden",
        stream=_bernoulli_stream(rng, probs),
        change_point=change_point,
        is_real_drift=True,
        magnitude=p1 - p0,
        p0=p0,
        p1=p1,
    )


def make_gradual(
    rng: np.random.Generator,
    length: int,
    change_point: int,
    ramp_len: int,
    p0: float,
    p1: float,
) -> Scenario:
    probs = np.full(length, p0)
    ramp_end = min(change_point + ramp_len, length)
    if ramp_end > change_point:
        probs[change_point:ramp_end] = np.linspace(p0, p1, ramp_end - change_point)
    probs[ramp_end:] = p1
    return Scenario(
        name="gradual",
        stream=_bernoulli_stream(rng, probs),
        change_point=change_point,
        is_real_drift=True,
        magnitude=p1 - p0,
        p0=p0,
        p1=p1,
    )


def make_transient(
    rng: np.random.Generator,
    length: int,
    change_point: int,
    blip_len: int,
    p0: float,
    p1: float,
) -> Scenario:
    probs = np.full(length, p0)
    blip_end = min(change_point + blip_len, length)
    probs[change_point:blip_end] = p1
    # reverts back to p0 after the blip -- no real, sustained drift
    return Scenario(
        name="transient",
        stream=_bernoulli_stream(rng, probs),
        change_point=change_point,
        is_real_drift=False,
        magnitude=p1 - p0,
        p0=p0,
        p1=p1,
    )


SCENARIO_BUILDERS = {
    "stable": make_stable,
    "sudden": make_sudden,
    "gradual": make_gradual,
    "transient": make_transient,
}
