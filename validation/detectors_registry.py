"""
validation/detectors_registry.py

Uniform wrapper around the 8 detectors compared on the project's
"detector evaluation" slide, all sourced from the `frouros` library
(pip install frouros) -- confirmed to be the exact set used originally:
frouros.detectors.concept_drift.streaming exposes ADWIN, DDM, EDDM, ECDDWT
(== "ECDD" on the slide), HDDMA (== "HDDM_A"), HDDMW (== "HDDM_W"),
PageHinkley, and RDDM, with no other detector family matching those eight
names. Centralizing the construction here means run_validation.py and any
future caller build every detector the same way, with the same defaults,
so a "which config did you use" question has one answer.

`driftguard/detection/changepoint_sequential.py` (Person 1's module) should
end up depending on this file too -- Page-Hinkley there and PageHinkley
here should be the same detector with the same config, not two
reimplementations that can quietly disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from frouros.detectors.concept_drift import (
    ADWIN,
    ADWINConfig,
    DDM,
    DDMConfig,
    ECDDWT,
    ECDDWTConfig,
    EDDM,
    EDDMConfig,
    HDDMA,
    HDDMAConfig,
    HDDMW,
    HDDMWConfig,
    PageHinkley,
    PageHinkleyConfig,
    RDDM,
    RDDMConfig,
)


@dataclass(frozen=True)
class DetectorSpec:
    """A named, re-instantiable detector factory.

    `factory()` must return a *fresh* detector instance -- these objects
    are stateful (they accumulate statistics across .update() calls), so
    one instance can only ever be used for one trial/stream.
    """

    name: str
    factory: Callable[[], object]


def default_registry(include_tuned_variants: bool = True) -> list[DetectorSpec]:
    """The 8 detectors from the project's evaluation slide, default configs,
    plus (by default) one tuned variant discovered by tuning_sweep.py.

    Defaults are frouros's own library defaults -- deliberately NOT hand-
    tuned per detector, so the comparison reflects out-of-the-box behavior
    rather than whichever detector we happened to tune hardest. If/when
    the team tunes configs for production use (configs/detectors.yaml),
    that tuning should happen in driftguard/detection/, informed by these
    results -- not by silently changing what "default" means here.

    include_tuned_variants=True additionally includes "Page-Hinkley (tuned
    lambda=12)". tuning_sweep.py found the library default (lambda_=50.0)
    is calibrated for a different signal scale than DriftGuard's actual
    error rates (~5-40%), making default Page-Hinkley look artificially
    slow and unreliable in a naive out-of-the-box comparison. Including
    the tuned variant here means the main comparison doesn't quietly bury
    that finding -- it shows up directly next to the untuned default. The
    other 7 detectors have NOT had an equivalent tuning pass yet (see
    REPORT.md's "Known limitations" section) -- this is a deliberate,
    disclosed asymmetry, not a claim that Page-Hinkley was fairly compared
    against 7 untuned competitors.
    """
    registry = [
        DetectorSpec("Page-Hinkley", lambda: PageHinkley(config=PageHinkleyConfig())),
        DetectorSpec("DDM", lambda: DDM(config=DDMConfig())),
        DetectorSpec("EDDM", lambda: EDDM(config=EDDMConfig())),
        DetectorSpec("ADWIN", lambda: ADWIN(config=ADWINConfig())),
        DetectorSpec("HDDM_A", lambda: HDDMA(config=HDDMAConfig())),
        DetectorSpec("HDDM_W", lambda: HDDMW(config=HDDMWConfig())),
        DetectorSpec("RDDM", lambda: RDDM(config=RDDMConfig())),
        DetectorSpec("ECDD", lambda: ECDDWT(config=ECDDWTConfig())),
    ]
    if include_tuned_variants:
        registry.append(
            DetectorSpec(
                "Page-Hinkley (tuned)",
                lambda: PageHinkley(config=PageHinkleyConfig(lambda_=12.0)),
            )
        )
    return registry


# Detector name -> tuned config, discovered by validation/tuning_sweep_all.py
# (Page-Hinkley's by validation/tuning_sweep.py, run separately/first). Same
# methodology for every detector: sweep the parameter(s) that most directly
# control alarm sensitivity, over the SAME false-alarm-rate (stable stream)
# + detection-rate/delay (sudden, medium-magnitude drift) evaluation, same
# trial count and seeds, same reliability gate (false_alarm_rate <= 5%)
# used to pick Page-Hinkley's lambda_ -- so this is a fair fight, not one
# tuned detector compared against seven untuned ones. See
# validation/results/all_detector_tuning_recommendations.csv for the full
# sweep results and validation/results/07_all_detector_tuning.png for the
# per-detector curves.
#
# DDM needed a 2-parameter tune, not one: a drift_level-only sweep hit a
# hard 24% false-alarm floor that never improved no matter how conservative
# the threshold got. Root cause (confirmed by isolating the variable):
# frouros's default min_num_instances=30 lets DDM start evaluating drift
# after only 30 samples, before its running mean/std have stabilized --
# raising it to 150 fixed this independent of drift_level. Every other
# detector's floor was found via its single primary sensitivity parameter.
TUNED_CONFIGS = {
    "Page-Hinkley": lambda: PageHinkley(config=PageHinkleyConfig(lambda_=12.0)),
    "DDM": lambda: DDM(config=DDMConfig(warning_level=4.0 * (2.0 / 3.0), drift_level=4.0, min_num_instances=150)),
    "EDDM": lambda: EDDM(config=EDDMConfig(beta=0.56)),
    "ADWIN": lambda: ADWIN(config=ADWINConfig(delta=0.1)),
    "HDDM_A": lambda: HDDMA(config=HDDMAConfig(alpha_d=0.005, alpha_w=0.025)),
    "HDDM_W": lambda: HDDMW(config=HDDMWConfig(alpha_d=0.15, alpha_w=0.75)),
    "RDDM": lambda: RDDM(config=RDDMConfig(warning_level=4.5 * (1.773 / 2.258), drift_level=4.5)),
    "ECDD": lambda: ECDDWT(config=ECDDWTConfig(average_run_length=400, lambda_=0.02)),
}


class OrHybridDetector:
    """OR-logic ensemble: reports drift the moment EITHER of two wrapped
    detectors reports drift at that same sample.

    Built specifically to test, empirically rather than by reasoning
    alone, whether combining Page-Hinkley (tuned) and DDM (tuned) -- the
    two detectors whose real-data alarm timing was actually trustworthy
    on both Elec2 and Insects, per validation/results/FINAL_VERDICT.md
    section 3 -- catches drift faster than either alone, without paying
    an unacceptable false-alarm cost. Nothing here assumes the hybrid is
    better; it exists so the hybrid can be run through the exact same
    synthetic (run_validation.py) and real-data (live_replay.py)
    validation as every other detector in this repo.

    Deliberately does NOT latch "drift" permanently true once either
    constituent has ever fired -- it re-evaluates both constituents'
    OWN current `.status` every sample and ORs that. This matters for
    live_replay.py's run_all_alerts, which records every index where
    `.status['drift']` is true (not just the first) -- a hybrid that
    latched forever after one alarm would report ~every remaining
    sample as an alert and make its real-data alert-frequency numbers
    incomparable to every other detector in this suite. Each wrapped
    detector already manages its own post-alarm reset/continuation
    internally (that's why, e.g., Page-Hinkley alone shows a bounded
    alerts_per_1000_samples rather than 1000); ORing their live status
    each step preserves that same pulsing behavior for the hybrid.

    Implements the same minimal interface every caller in this suite
    relies on -- `.update(value=...)` and `.status` (a dict with
    'drift' / 'warning' keys) -- so it's a drop-in DetectorSpec factory
    result, indistinguishable from a single frouros detector to
    run_detector_on_stream, run_all_alerts, or anything else here.
    """

    def __init__(self, factory_a: Callable[[], object], factory_b: Callable[[], object]):
        self._a = factory_a()
        self._b = factory_b()

    def update(self, value: int) -> None:
        self._a.update(value=value)
        self._b.update(value=value)

    @property
    def status(self) -> dict:
        status_a, status_b = self._a.status, self._b.status
        return {
            "drift": bool(status_a.get("drift")) or bool(status_b.get("drift")),
            "warning": bool(status_a.get("warning")) or bool(status_b.get("warning")),
        }


# Name kept consistent everywhere (run_validation.py's raw_results,
# live_replay.py's alert_summary, report.py's detector_order lists, etc.)
# so every downstream table/chart can key off this one constant.
HYBRID_NAME = "Page-Hinkley+DDM (OR hybrid)"


def hybrid_registry() -> list[DetectorSpec]:
    """Just the OR-hybrid itself, as a single-entry registry -- callers
    that want it alongside the individually-tuned detectors it's built
    from should concatenate this with tuned_registry(), e.g.
    `tuned_registry() + hybrid_registry()`, so results are directly
    comparable (same trial seeds -- seeding in run_validation.py depends
    only on scenario/magnitude/trial/seed, not on which detectors are in
    the registry)."""
    return [
        DetectorSpec(
            HYBRID_NAME,
            lambda: OrHybridDetector(TUNED_CONFIGS["Page-Hinkley"], TUNED_CONFIGS["DDM"]),
        )
    ]


def tuned_registry() -> list[DetectorSpec]:
    """All 8 detectors from the deck's comparison, each individually tuned
    by validation/tuning_sweep_all.py (Page-Hinkley by tuning_sweep.py) --
    the fair-fight comparison: every detector got the same tuning pass,
    not just Page-Hinkley. Detector order matches default_registry()'s
    first 8 so composite scores are directly comparable to the untuned
    baseline in REPORT.md.
    """
    return [DetectorSpec(f"{name} (tuned)", factory) for name, factory in TUNED_CONFIGS.items()]


def run_detector_on_stream(spec: DetectorSpec, stream) -> dict:
    """
    Feed one binary error stream through a fresh instance of `spec`,
    sample by sample, and record the first index (if any) at which the
    detector's status reports drift=True.

    Returns a dict with:
      first_drift_index: int | None -- index of the first flagged drift
      any_warning: bool             -- whether a "warning" state was ever hit
    """
    detector = spec.factory()
    first_drift_index = None
    any_warning = False

    for i, value in enumerate(stream):
        detector.update(value=int(value))
        status = detector.status
        if status.get("warning"):
            any_warning = True
        if status.get("drift") and first_drift_index is None:
            first_drift_index = i
            # Keep going only if we care about later re-detections; for
            # this benchmark the first flag is what matters (matches how
            # a live monitor would behave -- it alerts once, a human
            # reviews, then the detector is typically reset).
            break

    return {"first_drift_index": first_drift_index, "any_warning": any_warning}
