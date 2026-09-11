"""
validation/report.py

Turns validation/results/raw_results.csv (produced by run_validation.py)
into: a console summary, results/summary.csv, three charts, and a written
markdown verdict -- specifically answering the question DriftGuard's own
pipeline needs answered: which detector(s) should
driftguard/detection/changepoint_sequential.py and changepoint_pelt.py
actually use, and does the deck's "virtually no false alarms" claim hold up
under many trials rather than one run.

Scoring, aligned to DriftGuard's actual goals (not generic ML-drift-lit
goals): the project pitch (see the deck's "Page-Hinkley: the live warning"
and "detector evaluation" slides) explicitly wants (a) high detection rate
on real drift, (b) low false-alarm rate -- both on pure noise and on
temporary blips that aren't real drift, and (c) low detection delay, since
Page-Hinkley is specifically pitched as the low-latency live-alert path.
The composite score below is a transparent weighted average of exactly
those three things -- change the weights in COMPOSITE_WEIGHTS if the
team's priorities differ; don't just trust the ranking blindly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fixed categorical order (dataviz skill reference palette, light mode,
# 8 slots -- one per detector, assigned in the SAME order everywhere so a
# detector's color never changes chart to chart).
DETECTOR_ORDER = [
    "Page-Hinkley", "DDM", "EDDM", "ADWIN", "HDDM_A", "HDDM_W", "RDDM", "ECDD",
    "Page-Hinkley (tuned)",
]
DETECTOR_COLOR = dict(zip(
    DETECTOR_ORDER,
    # first 8 = dataviz skill's validated 8-slot categorical order (fixed,
    # never cycled). The 9th, "Page-Hinkley (tuned)", is a lighter tint of
    # Page-Hinkley's own blue rather than a new hue -- it's a configuration
    # variant of that same detector, not a new categorical entity.
    ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948", "#9ec7ee"],
))
# "<Name> (tuned)" variants (validation/tuning_sweep_all.py, all 8
# detectors) reuse their base detector's own hue -- same categorical
# entity, a different configuration, not a new color slot.
TUNED_DETECTOR_ORDER = [f"{name} (tuned)" for name in DETECTOR_ORDER[:8]]
for _base_name in DETECTOR_ORDER[:8]:
    DETECTOR_COLOR[f"{_base_name} (tuned)"] = DETECTOR_COLOR[_base_name]

# The OR-hybrid (Page-Hinkley (tuned) + DDM (tuned), validation/
# detectors_registry.py's OrHybridDetector) is a new categorical entity --
# not a config variant of an existing detector -- so it gets its own color,
# not a tint of Page-Hinkley's or DDM's blue/orange.
HYBRID_DETECTOR_NAME = "Page-Hinkley+DDM (OR hybrid)"
DETECTOR_COLOR[HYBRID_DETECTOR_NAME] = "#7d5ba6"

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e3e2dd"

COMPOSITE_WEIGHTS = {
    "detection_rate": 0.40,   # catches real drift (sudden + gradual)
    "no_false_alarm": 0.40,   # stays quiet on pure noise AND transient blips
    "low_delay": 0.20,        # flags real drift quickly (Page-Hinkley's pitch)
}


def _style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)


def compute_summary(df: pd.DataFrame, detector_order: list[str] | None = None) -> pd.DataFrame:
    if detector_order is None:
        detector_order = DETECTOR_ORDER
    rows = []
    for detector in detector_order:
        d = df[df["detector"] == detector]

        real_drift = d[d["is_real_drift"]]
        detection_rate = real_drift["correct_detection"].mean() if len(real_drift) else np.nan

        sudden = d[d["scenario"] == "sudden"]
        sudden_rate = sudden["correct_detection"].mean() if len(sudden) else np.nan
        gradual = d[d["scenario"] == "gradual"]
        gradual_rate = gradual["correct_detection"].mean() if len(gradual) else np.nan

        stable = d[d["scenario"] == "stable"]
        stable_false_alarm = stable["false_alarm"].mean() if len(stable) else np.nan
        transient = d[d["scenario"] == "transient"]
        transient_false_alarm = transient["false_alarm"].mean() if len(transient) else np.nan
        # Unweighted mean of the two conditions, NOT a pooled mean over all
        # no-drift trials. Pooling would let the scenario with more trials
        # (transient has 3x as many, one per magnitude) dilute the other's
        # result -- e.g. a detector that's terrible on pure noise (stable)
        # but fine on brief blips (transient) must not look "mostly fine"
        # just because transient trials outnumber stable ones 3:1.
        false_alarm_rate = np.nanmean([stable_false_alarm, transient_false_alarm])
        no_drift = d[~d["is_real_drift"]]  # stable + transient, for the trial count only

        delays = real_drift.loc[real_drift["correct_detection"], "delay"]
        median_delay = delays.median() if len(delays) else np.nan

        rows.append(
            {
                "detector": detector,
                "detection_rate_sudden": sudden_rate,
                "detection_rate_gradual": gradual_rate,
                "detection_rate_overall": detection_rate,
                "false_alarm_rate_stable": stable_false_alarm,
                "false_alarm_rate_transient": transient_false_alarm,
                "false_alarm_rate_overall": false_alarm_rate,
                "median_delay_samples": median_delay,
                "n_real_drift_trials": len(real_drift),
                "n_no_drift_trials": len(no_drift),
            }
        )

    summary = pd.DataFrame(rows).set_index("detector").loc[detector_order]

    # Composite score, each component normalized to [0, 1], higher = better.
    max_delay = summary["median_delay_samples"].max()
    delay_score = 1 - (summary["median_delay_samples"] / max_delay)
    delay_score = delay_score.fillna(0.0)  # never detected in time -> worst possible

    summary["composite_score"] = (
        COMPOSITE_WEIGHTS["detection_rate"] * summary["detection_rate_overall"].fillna(0.0)
        + COMPOSITE_WEIGHTS["no_false_alarm"] * (1 - summary["false_alarm_rate_overall"].fillna(1.0))
        + COMPOSITE_WEIGHTS["low_delay"] * delay_score
    )

    return summary.sort_values("composite_score", ascending=False)


def _grouped_bar(
    summary: pd.DataFrame, cols: list[str], col_labels: list[str], title: str,
    ylabel: str, out_path: Path, as_percent: bool = True, value_fmt: str = "{:.0f}",
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    n_groups = len(cols)
    x = np.arange(len(summary.index))
    width = 0.8 / n_groups

    for i, (col, label) in enumerate(zip(cols, col_labels)):
        vals = summary[col].to_numpy(dtype=float)
        vals = np.nan_to_num(vals, nan=0.0)
        if as_percent:
            vals = vals * 100
        offset = (i - (n_groups - 1) / 2) * width
        bars = ax.bar(
            x + offset, vals, width * 0.92, label=label,
            color=[DETECTOR_COLOR[d] for d in summary.index],
            alpha=1.0 if n_groups == 1 else (0.95 - 0.35 * i),
            edgecolor="white", linewidth=0.5,
        )
        for rect, v in zip(bars, vals):
            ax.text(
                rect.get_x() + rect.get_width() / 2, rect.get_height() + (1.5 if as_percent else 0.015 * max(vals.max(), 1e-9)),
                value_fmt.format(v) + ("%" if as_percent else ""),
                ha="center", va="bottom", fontsize=7.5, color=TEXT_SECONDARY,
            )

    ax.set_title(title, fontsize=13, color=TEXT_PRIMARY, loc="left", pad=14)
    ax.set_ylabel(ylabel, fontsize=10, color=TEXT_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels(summary.index, rotation=20, ha="right", fontsize=9.5, color=TEXT_PRIMARY)
    if as_percent:
        ax.set_ylim(0, 112)
    if n_groups > 1:
        ax.legend(frameon=False, fontsize=9, loc="upper right")
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor="white")
    plt.close(fig)


def make_charts(summary: pd.DataFrame, out_dir: Path) -> None:
    _grouped_bar(
        summary,
        cols=["detection_rate_sudden", "detection_rate_gradual"],
        col_labels=["Sudden drift", "Gradual drift"],
        title="Detection rate on real drift (higher is better)",
        ylabel="Detection rate (%)",
        out_path=out_dir / "01_detection_rate.png",
    )
    _grouped_bar(
        summary,
        cols=["false_alarm_rate_stable", "false_alarm_rate_transient"],
        col_labels=["Stable stream (pure noise)", "Transient blip (reverts, not real drift)"],
        title="False alarm rate on non-drift (lower is better)",
        ylabel="False alarm rate (%)",
        out_path=out_dir / "02_false_alarm_rate.png",
    )
    _grouped_bar(
        summary,
        cols=["median_delay_samples"],
        col_labels=["Median detection delay"],
        title="Detection delay on sudden + gradual drift, correct detections only (lower is better)",
        ylabel="Samples after true change point",
        out_path=out_dir / "03_detection_delay.png",
        as_percent=False,
    )
    _grouped_bar(
        summary,
        cols=["composite_score"],
        col_labels=["Composite score"],
        title=(
            f"Composite score for DriftGuard's use case "
            f"({int(COMPOSITE_WEIGHTS['detection_rate']*100)}% detection, "
            f"{int(COMPOSITE_WEIGHTS['no_false_alarm']*100)}% no-false-alarm, "
            f"{int(COMPOSITE_WEIGHTS['low_delay']*100)}% low-delay)"
        ),
        ylabel="Score (0-1, higher is better)",
        out_path=out_dir / "04_composite_score.png",
        as_percent=False,
        value_fmt="{:.2f}",
    )


def write_markdown_report(
    summary: pd.DataFrame, n_trials: int, out_path: Path,
    highlight_detector: str = "Page-Hinkley", title: str = "DriftGuard detector validation — results",
    generated_by_note: str = "Generated by `validation/run_validation.py` + `report.py`",
    all_tuned: bool = False,
) -> None:
    best = summary.index[0]
    worst_false_alarm = summary["false_alarm_rate_overall"].idxmax()
    # highlight_detector may not be an exact index match (e.g. "Page-Hinkley"
    # vs. this summary's "Page-Hinkley (tuned)") -- match by prefix so the
    # same call works for both the default and tuned comparisons.
    highlight_matches = [d for d in summary.index if d == highlight_detector or d.startswith(highlight_detector + " ")]
    highlight_key = highlight_matches[0] if highlight_matches else best

    lines = []
    lines.append(f"# {title}\n")
    lines.append(
        f"{generated_by_note}, "
        f"{n_trials} trials per (detector, scenario, magnitude) cell, "
        f"3 drift magnitudes (Δ = 0.10 / 0.20 / 0.35 over a 5% baseline error rate), "
        f"stream length 800, change point at sample 400.\n"
    )
    lines.append("## Summary table\n")
    lines.append(
        summary.round(3)[
            [
                "detection_rate_sudden", "detection_rate_gradual",
                "false_alarm_rate_stable", "false_alarm_rate_transient",
                "median_delay_samples", "composite_score",
            ]
        ].to_markdown()
    )
    lines.append("\n\n## Verdict\n")
    lines.append(
        f"**`{best}` ranks first** on the composite score defined above "
        f"(40% detection rate on real drift, 40% avoiding false alarms on pure "
        f"noise and on transient blips, 20% low detection delay).\n"
    )

    ph_false_alarm = summary.loc[highlight_key, "false_alarm_rate_overall"]
    ph_transient_false_alarm = summary.loc[highlight_key, "false_alarm_rate_transient"]
    lines.append(
        f"\n**On the deck's specific claims:** {highlight_key}'s measured false-alarm "
        f"rate here is {ph_false_alarm*100:.1f}% overall "
        f"({ph_transient_false_alarm*100:.1f}% on transient blips specifically) across "
        f"{n_trials} trials per condition — "
        + (
            "consistent with the deck's 'virtually no false alarms' claim."
            if ph_false_alarm < 0.05
            else "NOT as clean as the deck's 'virtually no false alarms' claim suggests — "
            "worth re-running the original comparison's exact methodology to see where the "
            "discrepancy comes from before repeating that claim in the final report."
        )
    )
    lines.append(
        f"\n\nThe detector with the worst false-alarm rate in this benchmark is "
        f"`{worst_false_alarm}` at {summary.loc[worst_false_alarm, 'false_alarm_rate_overall']*100:.1f}%.\n"
    )

    # "Fastest, among detectors that are actually reliable" -- never just
    # "fastest," since the fastest detector here (ECDD) is fast because it
    # is trigger-happy, not because it is good. A false-alarm-rate gate
    # comes first; delay is the tiebreaker only among survivors.
    RELIABILITY_GATE = 0.05
    reliable = summary[summary["false_alarm_rate_overall"] <= RELIABILITY_GATE]
    fastest_reliable = (
        reliable["median_delay_samples"].idxmin()
        if not reliable.empty and reliable["median_delay_samples"].notna().any()
        else None
    )

    lines.append("\n## Recommendation for the codebase\n")
    global_fastest = summary["median_delay_samples"].idxmin()
    if fastest_reliable:
        caveat = (
            f"Do not pick by delay alone — the single fastest detector overall in this run "
            f"(`{global_fastest}`) is only fast because it also has "
            f"one of the worst false-alarm rates; a detector that fires constantly will always "
            f"look fast.\n"
            if global_fastest != fastest_reliable
            else (
                f"Here the fastest detector overall and the fastest RELIABLE detector are the "
                f"same (`{fastest_reliable}`) — its speed is not a false-alarm-rate artifact.\n"
            )
        )
        lines.append(
            f"- `driftguard/detection/changepoint_sequential.py` (the live/low-latency path): "
            f"among detectors with a false-alarm rate at or below {RELIABILITY_GATE*100:.0f}% "
            f"(`{'`, `'.join(reliable.index)}`), **{fastest_reliable}** has the lowest median "
            f"delay ({summary.loc[fastest_reliable, 'median_delay_samples']:.0f} samples). "
            + caveat
        )
    lines.append(
        "- `driftguard/detection/changepoint_pelt.py` + `bootstrap.py` (the retrospective, "
        "confirmatory path): false-alarm rate matters more than speed here, since this "
        "runs after the fact — weight the false-alarm columns above more heavily than delay.\n"
        "- Re-run this benchmark (`python -m validation.run_validation --trials 200`) "
        "once real DriftGuard log data exists, using `live_replay.py` to validate against actual "
        "provider traffic rather than only synthetic streams — synthetic Bernoulli streams are a "
        "reasonable first pass but are not a substitute for validating against real data.\n"
    )

    lines.append("\n## Known limitations of this benchmark\n")
    if all_tuned:
        lines.append(
            "- **This is the fair-fight comparison**: all 8 detectors were individually tuned "
            "(`validation/tuning_sweep_all.py`, same methodology, same reliability gate, same "
            "trial count and seeds used for Page-Hinkley's own tuning pass), so this ranking is "
            "not just a tuned Page-Hinkley beating 7 untuned competitors — see "
            "`validation/results/all_detector_tuning_recommendations.csv` for every detector's "
            "chosen config and `07_all_detector_tuning.png` for the underlying sweep curves. "
            "One caveat that still applies: each detector's tuning swept its single primary "
            "sensitivity parameter (two, for DDM, after its first-pass floor was root-caused to "
            "`min_num_instances`) — a full multi-parameter grid search per detector was not run, "
            "so a further, smaller improvement per detector is still possible in principle.\n"
        )
    else:
        lines.append(
            "- Only Page-Hinkley received a tuning pass (`tuning_sweep.py`); the other 7 detectors "
            "are compared at frouros's library defaults. Page-Hinkley's default (λ=50) turned out to "
            "be badly mismatched to DriftGuard's ~5-40% error-rate scale — the other 7 detectors' "
            "defaults have NOT been checked for the same problem, so this comparison likely "
            "understates all of them, not just Page-Hinkley. Do not treat the ranking above as final "
            "until each detector gets the same tuning-sweep treatment (now done — see "
            "`validation/results/tuned/REPORT.md`).\n"
        )
    lines.append(
        "- All scenarios are synthetic Bernoulli streams, not real DriftGuard traffic — they test "
        "whether a detector can find a KNOWN, well-defined change, not whether it behaves well on "
        "the messier statistics of real model outputs. `live_replay.py` (unbuilt) is meant to "
        "close this gap.\n"
        "- One baseline error rate (5%) and three magnitudes were tested. Detector ranking can "
        "shift at different baselines — worth sweeping baseline too before locking in a default "
        "detector per metric type.\n"
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Summarize DriftGuard validation results")
    parser.add_argument("--results-dir", type=str, default="validation/results")
    parser.add_argument(
        "--registry", choices=["default", "tuned"], default="default",
        help="'default': raw_results.csv, library defaults + Page-Hinkley (tuned) only. "
             "'tuned': raw_results_tuned.csv, ALL 8 detectors individually tuned -- writes to results-dir/tuned/.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if args.registry == "tuned":
        df = pd.read_csv(results_dir / "raw_results_tuned.csv")
        detector_order = TUNED_DETECTOR_ORDER
        out_dir = results_dir / "tuned"
        title = "DriftGuard detector validation — ALL detectors tuned (fair-fight comparison)"
        generated_by_note = (
            "Generated by `validation/tuning_sweep_all.py` (per-detector tuning) + "
            "`validation/run_validation.py --registry tuned` + `report.py --registry tuned`"
        )
        highlight_detector = "Page-Hinkley (tuned)"
    else:
        df = pd.read_csv(results_dir / "raw_results.csv")
        detector_order = DETECTOR_ORDER
        out_dir = results_dir
        title = "DriftGuard detector validation — results"
        generated_by_note = "Generated by `validation/run_validation.py` + `report.py`"
        highlight_detector = "Page-Hinkley"

    out_dir.mkdir(parents=True, exist_ok=True)
    n_trials = df.groupby(["detector", "scenario", "magnitude"], dropna=False).size().max()

    summary = compute_summary(df, detector_order=detector_order)
    summary.to_csv(out_dir / "summary.csv")

    print(summary.round(3).to_string())

    make_charts(summary, out_dir)
    write_markdown_report(
        summary, int(n_trials), out_dir / "REPORT.md",
        highlight_detector=highlight_detector, title=title, generated_by_note=generated_by_note,
        all_tuned=(args.registry == "tuned"),
    )
    print(f"\nCharts + REPORT.md written to {out_dir}/")


if __name__ == "__main__":
    _main()
