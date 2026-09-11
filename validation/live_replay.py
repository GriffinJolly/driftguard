"""
validation/live_replay.py

Real-data validation, as opposed to the synthetic Bernoulli benchmark in
run_validation.py. This is the module `codebase-review-and-workplan.md`
flagged as a gap ("everything is synthetic ... not a substitute for
validating against real data") -- this fills it.

Two real-data modes, chosen because they answer two different questions:

1. PUBLIC BENCHMARK MODE (--dataset phishing / elec2 / insects)
   Runs an actual online classifier (river) against a real, publicly
   published dataset in PREQUENTIAL fashion (river's own standard
   protocol: for each new example, predict first, THEN learn on it --
   this is the honest online-learning setting, not train/test split).
   The resulting correct/incorrect sequence is a REAL error stream, not
   a synthetic Bernoulli draw. This is the standard way concept-drift
   detectors are validated in the literature your own project cites
   (Page, Hinkley, Killick et al., etc. all validate against real
   streams, not only synthetic ones).

     --dataset phishing  is BUNDLED inside the `river` package (no
     network needed) -- this is the one this script was actually run
     against to produce validation/results/live_replay/ in this repo.

     --dataset elec2 / insects are real, well-known CONCEPT DRIFT
     benchmarks (not just "real data" -- these specific datasets are
     used throughout the concept-drift literature because they contain
     genuine, documented distribution shift). By default they download
     from river's hosted mirrors on first use -- but if a local copy is
     present at validation/data/electricity.csv.gz or
     validation/data/insects_abrupt_balanced.csv.gz, that local file is
     used instead and NO network call is made at all (see
     _local_elec2_stream / _local_insects_stream below). This is how
     this repo's own validation/results/live_replay/elec2_*.csv and
     insects_*.csv were produced -- in a network-restricted environment,
     using manually-obtained copies of the standard Elec2
     (https://maxhalford.github.io/files/datasets/electricity.zip) and
     INSECTS-abrupt_balanced (Souza et al., the standard OpenML/USP DCC
     concept-drift benchmark) datasets, gzipped and committed under
     validation/data/.

2. LIVE GROQ MODE -- see live_replay_groq.py instead. This script is
   public-benchmark-only; live_replay_groq.py is the "call your actual
   provider right now and watch detectors react to real production
   traffic" mode, which needs a GROQ_API_KEY and wall-clock time to
   accumulate a stream, so it's kept as a separate, explicitly-opt-in
   script rather than bundled into this one.

Unlike run_validation.py (which stops at the FIRST detected drift,
because in production you alert once and a human reviews), this script
records EVERY alert across the whole stream. There's no injected,
known change point in real data (except by construction, for a spliced
stream -- see --dataset insects), so the interesting output here is
"how did each detector actually behave on real, messy, non-synthetic
data" -- how often did it fire, and does that look sane -- not "did it
find the changepoint," which the synthetic benchmark already covers
with a known ground truth.

Usage:
    python -m validation.live_replay --dataset phishing
    python -m validation.live_replay --dataset elec2      # needs network
    python -m validation.live_replay --dataset insects    # needs network
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from validation.detectors_registry import DetectorSpec, default_registry, tuned_registry
from validation.report import DETECTOR_COLOR, DETECTOR_ORDER, _style_axis  # reuse the same palette/theme


def _require_river():
    try:
        import river  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "river is required for live_replay.py (pip install river). "
            "It's already listed in requirements.txt."
        ) from exc


_DATA_DIR = Path(__file__).resolve().parent / "data"
_ELEC2_LOCAL = _DATA_DIR / "electricity.csv.gz"
_INSECTS_LOCAL = _DATA_DIR / "insects_abrupt_balanced.csv.gz"


def _local_elec2_stream() -> Iterator[tuple[dict, bool]]:
    """
    validation/data/electricity.csv.gz -- the standard Elec2 (Australian
    New South Wales electricity market) dataset: 45,312 real half-hourly
    samples, 1996-1998, 8 numeric features + a binary UP/DOWN price-move
    label. Same schema river.datasets.Elec2() would hand you (date, day,
    period, nswprice, nswdemand, vicprice, vicdemand, transfer, class) --
    this just reads it from a local file instead of downloading it, so it
    matches river's own target convention: y = True when price moved UP.
    Elec2 is one of the most-cited real concept-drift benchmarks in the
    literature (recurring seasonal + non-seasonal drift in electricity
    price/demand) -- exactly the kind of real, documented drift the
    synthetic benchmark in run_validation.py can't speak to.
    """
    import pandas as pd

    df = pd.read_csv(_ELEC2_LOCAL)
    y = df["class"].astype(str).str.strip().str.upper() == "UP"
    x_records = df.drop(columns=["class"]).to_dict(orient="records")
    for x, y_val in zip(x_records, y):
        yield x, bool(y_val)


def _local_insects_stream() -> Iterator[tuple[dict, str]]:
    """
    validation/data/insects_abrupt_balanced.csv.gz -- the standard
    INSECTS-abrupt_balanced dataset (Souza et al., a purpose-built real
    concept-drift benchmark used throughout the streaming-ML literature):
    52,848 real optical-sensor flight readings, 33 numeric features, no
    header row, final column is a 6-way species label (balanced 8,808
    samples per class). The "abrupt" variant is built by concatenating
    blocks recorded at different ambient temperatures, so the SAME
    species' feature distribution abruptly shifts at each block boundary
    -- real, engineered concept drift with known approximate change
    points (every ~8,808 samples), not injected/synthetic drift.
    """
    import pandas as pd

    df = pd.read_csv(_INSECTS_LOCAL, header=None)
    n_features = df.shape[1] - 1
    feature_cols = [f"f{i}" for i in range(n_features)]
    df.columns = feature_cols + ["class"]
    y = df["class"].astype(str)
    x_records = df[feature_cols].to_dict(orient="records")
    for x, y_val in zip(x_records, y):
        yield x, y_val


def _prequential_error_stream(dataset_name: str) -> tuple[np.ndarray, str]:
    """
    Run a real river online classifier prequentially (predict-then-learn,
    the standard online-learning protocol -- no data leakage) against a
    real dataset, and return the resulting real 0/1 error indicator
    stream: 1 == the model got that example wrong.

    Returns (error_stream, description).
    """
    _require_river()
    from river import datasets, linear_model, metrics, preprocessing, tree

    if dataset_name == "phishing":
        stream = datasets.Phishing()
        model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
        desc = "river.datasets.Phishing (1,250 real webpage samples, bundled, no download)"
    elif dataset_name == "elec2":
        model = preprocessing.StandardScaler() | tree.HoeffdingTreeClassifier()
        if _ELEC2_LOCAL.exists():
            stream = _local_elec2_stream()
            desc = (
                f"LOCAL {_ELEC2_LOCAL.relative_to(_DATA_DIR.parent.parent)} -- real Elec2 "
                "(45,312 real NSW electricity-market samples, 1996-1998, documented "
                "recurring concept drift). No network used."
            )
        else:
            stream = datasets.Elec2()
            desc = (
                "river.datasets.Elec2 (45,312 real NSW electricity-market samples, "
                "1996-1998, documented recurring concept drift -- REQUIRES NETWORK "
                "on first run, downloads from river's mirror)"
            )
    elif dataset_name == "insects":
        model = preprocessing.StandardScaler() | tree.HoeffdingTreeClassifier()
        if _INSECTS_LOCAL.exists():
            stream = _local_insects_stream()
            desc = (
                f"LOCAL {_INSECTS_LOCAL.relative_to(_DATA_DIR.parent.parent)} -- real "
                "INSECTS-abrupt_balanced (52,848 real samples, purpose-built abrupt "
                "concept-drift benchmark, 6 balanced classes). No network used."
            )
        else:
            stream = datasets.Insects()
            desc = (
                "river.datasets.Insects (52,848 real samples, purpose-built concept-"
                "drift benchmark with labeled abrupt/gradual/incremental variants -- "
                "REQUIRES NETWORK on first run)"
            )
    else:
        raise ValueError(f"Unknown --dataset {dataset_name!r}. Choose phishing, elec2, or insects.")

    acc = metrics.Accuracy()
    errors: list[int] = []
    for x, y in stream:
        y_pred = model.predict_one(x)
        correct = (y_pred == y) if y_pred is not None else False
        errors.append(0 if correct else 1)
        acc.update(y, y_pred if y_pred is not None else False)
        model.learn_one(x, y)

    print(f"[{dataset_name}] {len(errors)} real samples, final prequential accuracy = {acc.get():.4f}")
    return np.array(errors, dtype=np.int8), desc


def run_all_alerts(spec: DetectorSpec, stream: np.ndarray) -> list[int]:
    """Like detectors_registry.run_detector_on_stream, but records EVERY
    drift alert across the stream instead of stopping at the first one --
    real data has no single known change point to stop at."""
    detector = spec.factory()
    alerts = []
    for i, value in enumerate(stream):
        detector.update(value=int(value))
        if detector.status.get("drift"):
            alerts.append(i)
    return alerts


def _rolling_error_rate(errors: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(errors, kernel, mode="valid")


def make_timeline_chart(
    errors: np.ndarray, alerts_by_detector: dict[str, list[int]], dataset_name: str, out_path: Path,
) -> None:
    window = max(10, len(errors) // 40)
    rolling = _rolling_error_rate(errors, window)

    n_detectors = len(alerts_by_detector)
    fig, axes = plt.subplots(
        n_detectors + 1, 1, figsize=(12, 1.35 * n_detectors + 3.2), sharex=True,
        gridspec_kw={"height_ratios": [2.6] + [1] * n_detectors},
    )

    ax0 = axes[0]
    ax0.plot(np.arange(window - 1, len(errors)), rolling, color="#2a78d6", linewidth=1.4)
    ax0.set_ylabel(f"Rolling error rate\n(window={window})", fontsize=8.5, color="#52514e")
    ax0.set_title(
        f"Real-data detector validation — {dataset_name} (river, prequential online learning)",
        fontsize=12.5, color="#0b0b0b", loc="left", pad=10,
    )
    _style_axis(ax0)

    for ax, (name, alerts) in zip(axes[1:], alerts_by_detector.items()):
        color = DETECTOR_COLOR.get(name, "#52514e")
        for a in alerts:
            ax.axvline(a, color=color, linewidth=1.1, alpha=0.75)
        ax.set_yticks([])
        ax.set_ylabel(name, fontsize=8, color="#0b0b0b", rotation=0, ha="right", va="center")
        ax.set_xlim(0, len(errors))
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#e3e2dd")

    axes[-1].set_xlabel("Sample index (real, time-ordered)", fontsize=9, color="#52514e")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor="white")
    plt.close(fig)


def run(dataset_name: str, out_dir: Path, registry=None) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    errors, desc = _prequential_error_stream(dataset_name)
    print(f"Dataset: {desc}")

    if registry is None:
        registry = default_registry(include_tuned_variants=True)
    alerts_by_detector: dict[str, list[int]] = {}
    rows = []
    for spec in registry:
        alerts = run_all_alerts(spec, errors)
        alerts_by_detector[spec.name] = alerts
        rows.append(
            {
                "detector": spec.name,
                "dataset": dataset_name,
                "n_samples": len(errors),
                "n_alerts": len(alerts),
                "first_alert_index": alerts[0] if alerts else None,
                "alerts_per_1000_samples": len(alerts) / len(errors) * 1000,
            }
        )

    summary = pd.DataFrame(rows).sort_values("n_alerts")
    summary.to_csv(out_dir / f"{dataset_name}_alert_summary.csv", index=False)

    pd.Series(errors).to_frame("error").to_csv(out_dir / f"{dataset_name}_error_stream.csv", index=False)

    make_timeline_chart(errors, alerts_by_detector, dataset_name, out_dir / f"{dataset_name}_timeline.png")

    print("\n" + summary.to_string(index=False))
    print(f"\nWrote alert summary, error stream, and timeline chart to {out_dir}/")
    return summary


def _main() -> None:
    parser = argparse.ArgumentParser(description="Real-data drift detector validation")
    parser.add_argument("--dataset", choices=["phishing", "elec2", "insects"], default="phishing")
    parser.add_argument("--out", type=str, default="validation/results/live_replay")
    parser.add_argument(
        "--registry", choices=["default", "tuned"], default="default",
        help="'default': library defaults + Page-Hinkley (tuned) only. "
             "'tuned': ALL 8 detectors individually tuned (validation/tuning_sweep_all.py).",
    )
    args = parser.parse_args()
    registry = tuned_registry() if args.registry == "tuned" else None
    out_dir = Path(args.out) / "tuned" if args.registry == "tuned" else Path(args.out)
    run(args.dataset, out_dir, registry=registry)


if __name__ == "__main__":
    _main()
