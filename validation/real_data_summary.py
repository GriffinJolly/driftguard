"""
validation/real_data_summary.py

Combines the three validation/results/live_replay/*_alert_summary.csv files
(phishing, elec2, insects -- produced by `python -m validation.live_replay
--dataset <name>`) into one cross-dataset comparison: a single table and
chart answering "does each detector's real-data behavior match what the
synthetic benchmark (run_validation.py / report.py) predicted?"

Why this is a separate script instead of folding into report.py: report.py
scores detectors against a KNOWN ground truth (synthetic, injected drift --
"was this actually the changepoint, yes/no"). Real data here has no such
ground truth for elec2/insects beyond "this is a well-documented drift
benchmark" -- so this script deliberately does NOT compute a detection-rate
or composite score. It reports what's actually measurable on real data:
how often each detector fired, and on phishing/elec2 specifically, whether
that firing rate tracks the real, visible rolling-error-rate regime shown
in each dataset's *_timeline.png (see those charts to check that visually --
this script only tabulates the alert counts).

Usage (from the repo root, after running live_replay.py for all three):
    python -m validation.real_data_summary
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from validation.report import DETECTOR_ORDER, TUNED_DETECTOR_ORDER, TEXT_PRIMARY, TEXT_SECONDARY, GRID_COLOR, _style_axis

DATASETS = ["phishing", "elec2", "insects"]
# Same dataviz-skill 8-slot categorical palette, a fresh 3-color subset
# (distinct from DETECTOR_COLOR, since this chart colors by dataset not by
# detector) -- blue/orange/green, high-contrast, colorblind-safe pairing.
DATASET_COLOR = {"phishing": "#2a78d6", "elec2": "#eb6834", "insects": "#1baf7a"}
DATASET_LABEL = {
    "phishing": "Phishing (1,250 samples, no known drift)",
    "elec2": "Elec2 (45,312 samples, real recurring drift)",
    "insects": "Insects-abrupt (52,848 samples, real abrupt drift)",
}


def load_combined(results_dir: Path) -> pd.DataFrame:
    frames = []
    for name in DATASETS:
        path = results_dir / f"{name}_alert_summary.csv"
        if not path.exists():
            print(f"[skip] {path} not found -- run `python -m validation.live_replay --dataset {name}` first.")
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit("No *_alert_summary.csv files found in " + str(results_dir))
    return pd.concat(frames, ignore_index=True)


def _detector_order_for(present: set[str]) -> list[str]:
    """DETECTOR_ORDER covers untuned names (+ 'Page-Hinkley (tuned)' only);
    TUNED_DETECTOR_ORDER covers all 8 '<Name> (tuned)' names. Use whichever
    actually matches what's present, so this works for both
    --registry default and --registry tuned live_replay runs without the
    caller having to say which."""
    if any(d in TUNED_DETECTOR_ORDER for d in present):
        return [d for d in TUNED_DETECTOR_ORDER if d in present]
    return [d for d in DETECTOR_ORDER if d in present]


def make_chart(combined: pd.DataFrame, out_path: Path) -> None:
    present_datasets = [d for d in DATASETS if d in combined["dataset"].unique()]
    detectors = _detector_order_for(set(combined["detector"].unique()))

    fig, ax = plt.subplots(figsize=(12, 6))
    n_groups = len(present_datasets)
    x = np.arange(len(detectors))
    width = 0.8 / n_groups

    for i, dataset in enumerate(present_datasets):
        sub = combined[combined["dataset"] == dataset].set_index("detector")
        vals = np.array([sub.loc[d, "alerts_per_1000_samples"] if d in sub.index else 0.0 for d in detectors])
        # log1p scale so 0-alert detectors (correctly quiet) are visible as
        # a real zero bar, not lost off a log axis or crushed by a 966
        # neighbor -- this is a rate that spans 0 to ~1000 per 1000 samples.
        plotted = np.log1p(vals)
        offset = (i - (n_groups - 1) / 2) * width
        bars = ax.bar(
            x + offset, plotted, width * 0.92, label=DATASET_LABEL[dataset],
            color=DATASET_COLOR[dataset], edgecolor="white", linewidth=0.5,
        )
        for rect, v in zip(bars, vals):
            ax.text(
                rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.05,
                f"{v:.0f}" if v >= 1 else ("0" if v == 0 else f"{v:.1f}"),
                ha="center", va="bottom", fontsize=7.5, color=TEXT_SECONDARY, rotation=0,
            )

    ax.set_title(
        "Alert frequency on real data — log(1+x) scale (lower is not automatically\n"
        "better: 0 alerts on elec2/insects would mean missing real, documented drift)",
        fontsize=12, color=TEXT_PRIMARY, loc="left", pad=14,
    )
    ax.set_ylabel("log(1 + alerts per 1,000 samples)", fontsize=10, color=TEXT_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels(detectors, rotation=20, ha="right", fontsize=9.5, color=TEXT_PRIMARY)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)  # headroom so the legend never overlaps the tallest bars' labels
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor="white")
    plt.close(fig)


def write_markdown(combined: pd.DataFrame, out_path: Path) -> None:
    lines = ["# Real-data validation — cross-dataset summary\n"]
    lines.append(
        "Produced by `python -m validation.real_data_summary`, combining "
        "`{phishing,elec2,insects}_alert_summary.csv` from `python -m "
        "validation.live_replay --dataset <name>`. Elec2 and Insects here "
        "were run from user-supplied local CSVs "
        "(`validation/data/electricity.csv.gz`, "
        "`validation/data/insects_abrupt_balanced.csv.gz`) — real, published "
        "concept-drift benchmarks, not synthetic data and not this repo's "
        "own injected-drift scenarios.\n"
    )
    for name in DATASETS:
        sub = combined[combined["dataset"] == name]
        if sub.empty:
            continue
        lines.append(f"### {DATASET_LABEL[name]}\n")
        lines.append(
            sub.sort_values("n_alerts")[
                ["detector", "n_alerts", "alerts_per_1000_samples", "first_alert_index"]
            ].round(2).to_markdown(index=False) + "\n"
        )
    out_path.write_text("\n".join(lines))


def _main() -> None:
    parser = argparse.ArgumentParser(description="Cross-dataset real-data validation summary")
    parser.add_argument("--results-dir", type=str, default="validation/results/live_replay")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    combined = load_combined(results_dir)
    combined.to_csv(results_dir / "combined_real_data_summary.csv", index=False)
    make_chart(combined, results_dir / "06_real_data_alert_frequency.png")
    write_markdown(combined, results_dir / "REAL_DATA_SUMMARY.md")
    print(combined.sort_values(["dataset", "n_alerts"]).to_string(index=False))
    print(f"\nWrote combined_real_data_summary.csv, 06_real_data_alert_frequency.png, "
          f"REAL_DATA_SUMMARY.md to {results_dir}/")


if __name__ == "__main__":
    _main()
