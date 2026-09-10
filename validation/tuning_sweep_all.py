"""
validation/tuning_sweep_all.py

Generalizes tuning_sweep.py's methodology (which tuned Page-Hinkley alone,
finding its library default was miscalibrated for DriftGuard's signal
scale) to the other 7 detectors. This exists because the honest caveat on
every result up to this point has been: "only Page-Hinkley got a tuning
pass -- the other 7 are compared at library defaults, which may be
similarly mismatched." This is what closes that gap, so the eventual
"Page-Hinkley wins" claim is a fair fight, not a tuned detector beating
7 untuned ones.

Same methodology as tuning_sweep.py, per detector: false-alarm rate over
N trials of a STABLE (no-drift) stream, and detection-rate + median delay
over N trials of a SUDDEN, medium-magnitude (p0=0.05 -> p1=0.25) drift
stream -- the same stream generator, same seeds, same trial count, same
"reliability gate" selection rule (RELIABILITY_GATE = 0.05 false-alarm
rate) as tuning_sweep.py, so results are directly comparable across
detectors and consistent with everything already in REPORT.md.

One parameter is swept per detector -- the single parameter that most
directly controls its alarm threshold/sensitivity (confirmed by
inspecting each frouros *Config class's fields, not guessed):
  DDM     -> drift_level             (# std devs past min error rate)
  EDDM    -> beta                    (drift ratio threshold)
  ADWIN   -> delta                   (allowed false-positive probability)
  HDDM_A  -> alpha_d                 (drift significance level)
  HDDM_W  -> alpha_d                 (drift significance level)
  RDDM    -> drift_level             (# std devs, DDM-family)
  ECDD    -> average_run_length      (target run length -> control limit)
Other fields are left at frouros's own defaults -- one axis at a time,
same as the original Page-Hinkley sweep, not a full multi-parameter grid
search (that's a reasonable further extension, not done here -- see
REPORT.md's "Known limitations").

Usage:
    python -m validation.tuning_sweep_all
    python -m validation.tuning_sweep_all --trials 150
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from frouros.detectors.concept_drift import (
    ADWIN, ADWINConfig, DDM, DDMConfig, ECDDWT, ECDDWTConfig,
    EDDM, EDDMConfig, HDDMA, HDDMAConfig, HDDMW, HDDMWConfig, RDDM, RDDMConfig,
)

from validation.ground_truth import make_stable, make_sudden
from validation.report import DETECTOR_COLOR, TEXT_PRIMARY, TEXT_SECONDARY, GRID_COLOR, _style_axis

LENGTH, CHANGE_POINT, BASELINE_P0 = 800, 400, 0.05
MEDIUM_MAGNITUDE = 0.20  # matches run_validation.py's "medium" drift magnitude, and tuning_sweep.py
RELIABILITY_GATE = 0.05  # same gate report.py's recommendation logic uses

# name -> (frouros default value, grid of values to try, factory(value) -> detector)
#
# Where a detector's Config enforces a relationship between two thresholds
# (frouros raises ValueError otherwise -- confirmed by testing, not
# assumed: DDM/RDDM require warning_level < drift_level, EDDM requires
# beta < alpha, HDDM_A/HDDM_W require alpha_d < alpha_w), the secondary
# threshold is scaled proportionally with the swept one, preserving
# frouros's own default ratio between them, rather than left fixed (which
# would make large parts of the intended grid raise an error).
DETECTOR_GRIDS: dict[str, dict] = {
    "DDM": {
        # A drift_level-only sweep (1.5 -> 30.0) hit a hard false-alarm
        # FLOOR of exactly 24% from drift_level=8.0 onward -- it never
        # improved further no matter how conservative the threshold got,
        # which is itself a real finding, not noise (confirmed: identical
        # 0.24 at 8.0, 10.0, 15.0, 20.0, 30.0). Root-caused by testing
        # min_num_instances in isolation: frouros's default (30) lets DDM
        # start evaluating drift after only 30 samples, before its running
        # mean/std have stabilized on an 800-sample stream, causing
        # spurious early alarms independent of drift_level entirely.
        # Raising min_num_instances alone (drift_level held at 8.0)
        # collapsed the false-alarm rate from 19% to 0% by
        # min_num_instances=100. So DDM needed a 2-parameter tune, not
        # drift_level alone -- "value" here is a (drift_level,
        # min_num_instances) tuple.
        "param": "(drift_level, min_num_instances)",
        "default": (3.0, 30),
        "grid": [
            (dl, mn)
            for dl in (2.5, 3.0, 3.5, 4.0, 5.0, 6.0)
            for mn in (60, 100, 150, 200)
        ],
        # default ratio warning_level/drift_level = 2.0/3.0
        "factory": lambda v: DDM(config=DDMConfig(warning_level=v[0] * (2.0 / 3.0), drift_level=v[0], min_num_instances=v[1])),
    },
    "EDDM": {
        "param": "beta",
        "default": 0.9,
        # first pass showed the reliability gate is crossed somewhere
        # between 0.5 (0% FA, 85% detection) and 0.6 (9% FA, 100%
        # detection) -- refined with finer steps in that exact window.
        "grid": [0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.93],
        "factory": lambda v: EDDM(config=EDDMConfig(beta=v)),
    },
    "ADWIN": {
        "param": "delta",
        "default": 0.002,
        # false_alarm_rate stayed at 0.0 for every value up to the first
        # pass's ceiling (0.01, 92% detection) -- extended upward to find
        # where detection actually crosses 95%, since reliability was
        # never the bottleneck here.
        "grid": [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
        "factory": lambda v: ADWIN(config=ADWINConfig(delta=v)),
    },
    "HDDM_A": {
        "param": "alpha_d",
        "default": 0.001,
        "grid": [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.05],
        # default ratio alpha_w/alpha_d = 0.005/0.001 = 5
        "factory": lambda v: HDDMA(config=HDDMAConfig(alpha_d=v, alpha_w=v * 5)),
    },
    "HDDM_W": {
        "param": "alpha_d",
        "default": 0.001,
        # first pass's ceiling (0.05) was already 0% FA / 100% detection /
        # lowest delay in-grid -- extended further to see if delay keeps
        # improving before reliability breaks down.
        "grid": [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15],
        "factory": lambda v: HDDMW(config=HDDMWConfig(alpha_d=v, alpha_w=v * 5)),
    },
    "RDDM": {
        "param": "drift_level",
        "default": 2.258,
        "grid": [1.2, 1.5, 1.773, 2.0, 2.258, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        # default ratio warning_level/drift_level = 1.773/2.258
        "factory": lambda v: RDDM(config=RDDMConfig(warning_level=v * (1.773 / 2.258), drift_level=v)),
    },
    "ECDD": {
        # average_run_length is NOT a free continuous parameter -- frouros
        # enforces it must be exactly 100, 400, or 1000 (confirmed by
        # testing). lambda_ (the EWMA weighting factor) is continuous, so
        # this sweeps lambda_ x each allowed ARL as a small 2D grid instead
        # of one axis -- "value" here is an (arl, lambda_) tuple. First
        # pass found (400, 0.05) best in-grid (22% FA) but still short of
        # the gate -- refined with smaller lambda_ around that point.
        "param": "(average_run_length, lambda_)",
        "default": (400, 0.2),
        "grid": [
            (arl, lam)
            for arl in (100, 400, 1000)
            for lam in (0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2)
        ],
        "factory": lambda v: ECDDWT(config=ECDDWTConfig(average_run_length=v[0], lambda_=v[1])),
    },
}


def _eval_param(detector_name: str, value: float, n_trials: int) -> dict:
    factory = DETECTOR_GRIDS[detector_name]["factory"]

    false_alarms = 0
    for t in range(n_trials):
        rng = np.random.default_rng(1000 + t)
        scenario = make_stable(rng, LENGTH, BASELINE_P0)
        det = factory(value)
        for v in scenario.stream:
            det.update(value=int(v))
            if det.status.get("drift"):
                false_alarms += 1
                break
    false_alarm_rate = false_alarms / n_trials

    hits, delays = 0, []
    for t in range(n_trials):
        rng = np.random.default_rng(2000 + t)
        scenario = make_sudden(rng, LENGTH, CHANGE_POINT, BASELINE_P0, BASELINE_P0 + MEDIUM_MAGNITUDE)
        det = factory(value)
        first = None
        for i, v in enumerate(scenario.stream):
            det.update(value=int(v))
            if det.status.get("drift"):
                first = i
                break
        if first is not None and first >= CHANGE_POINT:
            hits += 1
            delays.append(first - CHANGE_POINT)

    return {
        "detector": detector_name,
        "param": DETECTOR_GRIDS[detector_name]["param"],
        "value": value,
        "false_alarm_rate": false_alarm_rate,
        "detection_rate_medium_drift": hits / n_trials,
        "median_delay": float(np.median(delays)) if delays else None,
        "is_library_default": value == DETECTOR_GRIDS[detector_name]["default"],
    }


def run_sweep(n_trials: int) -> pd.DataFrame:
    rows = []
    for name, spec in DETECTOR_GRIDS.items():
        print(f"Sweeping {name}.{spec['param']} over {len(spec['grid'])} values...")
        for value in spec["grid"]:
            rows.append(_eval_param(name, value, n_trials))
    return pd.DataFrame(rows)


def pick_recommended(df: pd.DataFrame, detector_name: str) -> dict:
    """
    Same selection rule as tuning_sweep.py's _pick_recommended, generalized:
      1. Prefer configs with false_alarm_rate <= RELIABILITY_GATE AND
         detection_rate >= 0.95, tiebroken by lowest median delay.
      2. If none qualify, relax to false_alarm_rate <= RELIABILITY_GATE only,
         pick highest detection rate (tiebreak lowest delay).
      3. If NO grid point reaches the reliability gate, this detector
         couldn't be tuned to DriftGuard's reliability bar within the
         tested range -- report the least-bad point and flag it, rather
         than silently picking something unreliable.
    """
    d = df[df["detector"] == detector_name]

    ok = d[(d["false_alarm_rate"] <= RELIABILITY_GATE) & (d["detection_rate_medium_drift"] >= 0.95)]
    if not ok.empty:
        row = ok.loc[ok["median_delay"].idxmin()]
        return {"detector": detector_name, "value": row["value"], "status": "reliable_and_fast", **row.to_dict()}

    reliable = d[d["false_alarm_rate"] <= RELIABILITY_GATE]
    if not reliable.empty:
        row = reliable.loc[reliable["detection_rate_medium_drift"].idxmax()]
        return {"detector": detector_name, "value": row["value"], "status": "reliable_but_low_detection", **row.to_dict()}

    row = d.loc[d["false_alarm_rate"].idxmin()]
    return {"detector": detector_name, "value": row["value"], "status": "COULD_NOT_REACH_RELIABILITY_GATE", **row.to_dict()}


def make_charts(df: pd.DataFrame, recommendations: pd.DataFrame, out_dir: Path) -> None:
    detectors = list(DETECTOR_GRIDS.keys())
    fig, axes = plt.subplots(4, 2, figsize=(13, 15))
    axes = axes.flatten()

    for ax, name in zip(axes, detectors):
        d = df[df["detector"] == name].reset_index(drop=True)
        color = DETECTOR_COLOR.get(name, "#2a78d6")
        is_tuple_valued = isinstance(DETECTOR_GRIDS[name]["default"], tuple)

        if is_tuple_valued:
            # ECDD: (arl, lambda_) tuples -- plot against categorical index,
            # sorted by (arl, lambda_) so each ARL forms a contiguous block.
            d = d.sort_values("value", key=lambda col: col.map(str))
            x = np.arange(len(d))
            rec_value = recommendations.loc[recommendations["detector"] == name, "value"].iloc[0]
            default_value = DETECTOR_GRIDS[name]["default"]
            rec_x = [i for i, v in enumerate(d["value"]) if tuple(v) == tuple(rec_value)]
            def_x = [i for i, v in enumerate(d["value"]) if tuple(v) == tuple(default_value)]
            ax.set_xticks(x)
            ax.set_xticklabels([f"{v[0]}/{v[1]:g}" for v in d["value"]], rotation=60, fontsize=6.5)
        else:
            d = d.sort_values("value")
            x = d["value"]
            rec_value = recommendations.loc[recommendations["detector"] == name, "value"].iloc[0]
            default_value = DETECTOR_GRIDS[name]["default"]
            rec_x, def_x = [rec_value], [default_value]
            if d["value"].max() / max(d["value"].min(), 1e-12) > 20:
                ax.set_xscale("log")

        ax.plot(x, d["detection_rate_medium_drift"] * 100, "o-", color=color, label="Detection rate")
        ax.plot(x, d["false_alarm_rate"] * 100, "o--", color="#e34948", label="False alarm rate")
        for rx in rec_x:
            ax.axvline(rx, color="#1baf7a", linestyle=":", linewidth=1.3)
        for dx in def_x:
            ax.axvline(dx, color="#9a9890", linestyle=":", linewidth=1)
        ax.set_title(f"{name}  ({DETECTOR_GRIDS[name]['param']})", fontsize=10.5, color=TEXT_PRIMARY, loc="left")
        ax.set_ylim(-5, 112)
        ax.tick_params(labelsize=8)
        _style_axis(ax)

    axes[-1].axis("off")  # 7 detectors in an 8-slot grid
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.98, 0.06), frameon=False, fontsize=9.5)
    fig.suptitle(
        "Per-detector tuning sweeps — dotted grey = library default, dotted green = recommended",
        fontsize=13, color=TEXT_PRIMARY, x=0.02, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "07_all_detector_tuning.png", dpi=150, facecolor="white")
    plt.close(fig)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Tune the other 7 detectors, same methodology as Page-Hinkley")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--out", type=str, default="validation/results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = run_sweep(args.trials)
    df.to_csv(out_dir / "all_detector_tuning_sweep.csv", index=False)

    recs = pd.DataFrame([pick_recommended(df, name) for name in DETECTOR_GRIDS])
    recs.to_csv(out_dir / "all_detector_tuning_recommendations.csv", index=False)
    make_charts(df, recs, out_dir)

    print("\n" + recs[["detector", "param", "value", "status", "false_alarm_rate", "detection_rate_medium_drift", "median_delay"]].to_string(index=False))
    print(f"\nWrote all_detector_tuning_sweep.csv, all_detector_tuning_recommendations.csv, 07_all_detector_tuning.png to {out_dir}/")


if __name__ == "__main__":
    _main()
