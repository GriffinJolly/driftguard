"""
validation/tuning_sweep.py

Companion to run_validation.py, for one specific question that the main
8-detector comparison surfaced: is a detector's poor showing real, or an
artifact of frouros's library-default threshold not matching the scale of
DriftGuard's actual signals (error rates in the ~5-40% range, on streams
built from ~20-30 calls per eval run -- see evalsuite/runner.py)?

Concretely: the main comparison (report.py) found Page-Hinkley, under its
frouros default (lambda_=50.0), had the WORST detection delay of all 8
detectors and a below-average detection rate -- apparently contradicting
the project's own pitch of Page-Hinkley as the fast, low-latency detector.
A quick lambda sweep (below) shows this is a tuning artifact, not a real
weakness: lambda_=50 is calibrated for a different signal scale than a
Bernoulli(0.05-0.40) error stream. This script is what found that, and
should be the first thing Person 1 runs before hand-picking values for
configs/detectors.yaml -- don't trust a library default without checking
it against the actual data regime first, for ANY of the 8 detectors, not
just this one.

Usage:
    python -m validation.tuning_sweep
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from frouros.detectors.concept_drift import PageHinkley, PageHinkleyConfig

from validation.ground_truth import make_stable, make_sudden

LENGTH, CHANGE_POINT, BASELINE_P0 = 800, 400, 0.05
MEDIUM_MAGNITUDE = 0.20  # matches run_validation.py's "medium" drift magnitude
LAMBDA_GRID = [1, 2, 3, 5, 8, 12, 16, 20, 30, 50]  # 50 is the frouros library default


def _eval_lambda(lam: float, n_trials: int) -> dict:
    false_alarms = 0
    for t in range(n_trials):
        rng = np.random.default_rng(1000 + t)
        scenario = make_stable(rng, LENGTH, BASELINE_P0)
        det = PageHinkley(config=PageHinkleyConfig(lambda_=lam))
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
        det = PageHinkley(config=PageHinkleyConfig(lambda_=lam))
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
        "lambda": lam,
        "false_alarm_rate": false_alarm_rate,
        "detection_rate_medium_drift": hits / n_trials,
        "median_delay": float(np.median(delays)) if delays else None,
        "is_library_default": lam == 50,
    }


def run_sweep(n_trials: int = 80) -> pd.DataFrame:
    return pd.DataFrame([_eval_lambda(lam, n_trials) for lam in LAMBDA_GRID])


def _pick_recommended(df: pd.DataFrame) -> float:
    """Smallest lambda that hits 0% false alarms and >=95% detection rate."""
    ok = df[(df["false_alarm_rate"] == 0.0) & (df["detection_rate_medium_drift"] >= 0.95)]
    if ok.empty:
        return float(df.loc[df["detection_rate_medium_drift"].idxmax(), "lambda"])
    return float(ok.loc[ok["median_delay"].idxmin(), "lambda"])


def make_chart(df: pd.DataFrame, recommended: float, out_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    color_rate = "#2a78d6"
    color_delay = "#eb6834"

    ax1.plot(df["lambda"], df["detection_rate_medium_drift"] * 100, "o-", color=color_rate, label="Detection rate (medium drift)")
    ax1.plot(df["lambda"], df["false_alarm_rate"] * 100, "o--", color="#e34948", label="False alarm rate (stable)")
    ax1.set_xlabel("Page-Hinkley λ threshold", color="#52514e")
    ax1.set_ylabel("Rate (%)", color="#52514e")
    ax1.set_xscale("log")
    ax1.spines["top"].set_visible(False)
    ax1.grid(True, color="#e3e2dd", linewidth=0.8)
    ax1.set_axisbelow(True)

    ax2 = ax1.twinx()
    ax2.plot(df["lambda"], df["median_delay"], "s-", color=color_delay, label="Median detection delay")
    ax2.set_ylabel("Median delay (samples)", color=color_delay)
    ax2.spines["top"].set_visible(False)

    ax1.set_ylim(-5, 112)
    ax1.axvline(50, color="#9a9890", linestyle=":", linewidth=1)
    ax1.text(50, 96, "library\ndefault", fontsize=8, color="#9a9890", ha="center")
    ax1.axvline(recommended, color="#1baf7a", linestyle=":", linewidth=1.3)
    ax1.text(recommended, 4, f"recommended\nλ={recommended:g}", fontsize=8, color="#1baf7a", ha="center", va="bottom")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", frameon=False, fontsize=8.5)

    ax1.set_title(
        "Page-Hinkley is only 'low-latency' once λ is tuned to DriftGuard's signal scale",
        fontsize=11.5, color="#0b0b0b", loc="left", pad=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor="white")
    plt.close(fig)


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--out", type=str, default="validation/results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = run_sweep(args.trials)
    recommended = _pick_recommended(df)
    df.to_csv(out_dir / "page_hinkley_lambda_sweep.csv", index=False)
    make_chart(df, recommended, out_dir / "05_page_hinkley_tuning.png")

    print(df.to_string(index=False))
    print(f"\nRecommended lambda_ for configs/detectors.yaml: {recommended:g}")
    print(f"(library default is 50.0 -- untuned, it is the SLOWEST of all 8 detectors on this benchmark)")


if __name__ == "__main__":
    _main()
