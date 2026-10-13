"""
validation/retrospective_validation.py

Validates the RETROSPECTIVE (batch / historical-confirmation) drift path --
PELT change-point detection + bootstrap significance -- on real HISTORICAL data.

This complements the online/sequential validation (run_validation.py,
live_replay.py): those alert on a live stream, this one runs over a whole
recorded metric history at once, the way DriftGuard's PELT path is meant to be
used after the fact. The Insects dataset is ideal here because it is genuinely
historical (52,848 pre-recorded samples) and contains a documented, sharp drift
whose onset was independently estimated at approximately sample 14,476 in this
repo's earlier work (validation/results/FINAL_VERDICT.md) -- so we have a real
answer to check the detector against.

Pipeline:
  1. Load the real Insects error stream (0 = correct, 1 = wrong), produced by an
     online classifier learning prequentially over the dataset (live_replay.py).
  2. PELT (driftguard/detection/changepoint_pelt.py) proposes candidate change
     points over the whole history.
  3. Bootstrap significance (driftguard/detection/bootstrap.py) confirms or
     rejects each candidate, with Benjamini-Hochberg multiple-testing correction.
  4. Report the confirmed change points, whether one brackets the known ~14,476
     drift, and the size of each jump; save a CSV + an annotated timeline chart.

Usage:
    python -m validation.retrospective_validation
    python -m validation.retrospective_validation --penalty-scale 5 --dataset insects
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from driftguard.detection.bootstrap import SignificanceConfig, confirm_changepoints
from driftguard.detection.changepoint_pelt import PeltConfig, detect_changepoints
from validation.report import _style_axis

# Independently-estimated true drift onset for Insects-abrupt_balanced, from
# validation/results/FINAL_VERDICT.md (sustained crossing of baseline+0.30 on a
# 200-sample rolling error rate). A well-supported estimate, not an official label.
INSECTS_ESTIMATED_ONSET = 14476

_SAVED_STREAMS = {
    "insects": Path("validation/results/live_replay/insects_error_stream.csv"),
    "elec2": Path("validation/results/live_replay/elec2_error_stream.csv"),
}


def _load_error_stream(dataset: str) -> np.ndarray:
    """Prefer the error stream already saved by live_replay.py; otherwise build
    it fresh (needs the local dataset + river)."""
    saved = _SAVED_STREAMS.get(dataset)
    if saved and saved.exists():
        print(f"Loaded saved error stream: {saved}")
        return pd.read_csv(saved)["error"].to_numpy().astype(np.int8)
    from validation.live_replay import _prequential_error_stream
    errors, desc = _prequential_error_stream(dataset)
    print(f"Built error stream: {desc}")
    return errors


def _rolling(errors: np.ndarray, window: int) -> np.ndarray:
    return np.convolve(errors, np.ones(window) / window, mode="valid")


def make_chart(errors, results, dataset, onset, out_path: Path) -> None:
    window = max(50, len(errors) // 120)
    roll = _rolling(errors, window)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(np.arange(window - 1, len(errors)), roll, color="#2a78d6", linewidth=1.2, zorder=3)
    ax.set_ylabel(f"Rolling error rate (window={window})", fontsize=9, color="#52514e")
    ax.set_title(f"Retrospective PELT + bootstrap significance — {dataset} (historical, n={len(errors)})",
                 fontsize=12.5, color="#0b0b0b", loc="left", pad=10)
    for r in results:
        sig = r.get("significant", False)
        ax.axvline(r["index"], color="#1baf7a" if sig else "#b8b6b0",
                   linewidth=1.6 if sig else 0.9, alpha=0.9 if sig else 0.6, zorder=2)
    if onset is not None:
        ax.axvline(onset, color="#e34948", linewidth=1.4, linestyle="--", zorder=4,
                   label=f"estimated true onset (~{onset})")
        ax.legend(frameon=False, fontsize=9, loc="upper right")
    _style_axis(ax)
    ax.set_xlabel("Sample index (historical, time-ordered)", fontsize=9, color="#52514e")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor="white")
    plt.close(fig)


def run(dataset: str, pelt_cfg: PeltConfig, sig_cfg: SignificanceConfig, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    errors = _load_error_stream(dataset).astype(float)
    print(f"[{dataset}] n={len(errors)}  overall error rate={errors.mean():.3f}")

    changepoints = detect_changepoints(errors, pelt_cfg)
    print(f"PELT ({pelt_cfg.cost_model}, penalty_scale={pelt_cfg.penalty_scale}, "
          f"min_seg={pelt_cfg.min_segment_length}) proposed {len(changepoints)} candidate change point(s).")

    results = confirm_changepoints(errors, changepoints, sig_cfg)
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("index").reset_index(drop=True)
        df["magnitude"] = (df["post_mean"] - df["pre_mean"]).round(3)

    onset = INSECTS_ESTIMATED_ONSET if dataset == "insects" else None
    out_csv = out_dir / f"{dataset}_pelt_bootstrap.csv"
    df.to_csv(out_csv, index=False)
    make_chart(errors, results, dataset, onset, out_dir / f"{dataset}_pelt_bootstrap.png")

    if not df.empty:
        show = df[["index", "pre_mean", "post_mean", "magnitude", "p_value_raw", "p_value_corrected", "significant"]]
        print("\n" + show.round(4).to_string(index=False))
        n_sig = int(df["significant"].sum())
        print(f"\n{n_sig} of {len(df)} candidate change point(s) are significant "
              f"(bootstrap + Benjamini-Hochberg at alpha={sig_cfg.alpha}).")
        if onset is not None:
            sig = df[df["significant"]]
            if len(sig):
                nearest = sig.iloc[(sig["index"] - onset).abs().argmin()]
                print(f"Nearest SIGNIFICANT change point to the estimated onset (~{onset}): "
                      f"sample {int(nearest['index'])} ({int(nearest['index'] - onset):+d} samples), "
                      f"error rate {nearest['pre_mean']:.3f} -> {nearest['post_mean']:.3f}.")
    else:
        print("No candidate change points found.")

    print(f"\nWrote {out_csv} and the timeline chart to {out_dir}/")
    return df


def _main() -> None:
    p = argparse.ArgumentParser(description="Retrospective PELT + bootstrap drift validation on historical data")
    p.add_argument("--dataset", choices=["insects", "elec2"], default="insects")
    p.add_argument("--cost-model", dest="cost_model", default="l2",
                   help="'l2' (mean shift, recommended) | 'normal' | 'rbf' (small streams only -- O(n^2) memory)")
    p.add_argument("--penalty-scale", dest="penalty_scale", type=float, default=3.0,
                   help="higher -> fewer candidate change points")
    p.add_argument("--min-segment", dest="min_segment", type=int, default=200)
    p.add_argument("--jump", type=int, default=25, help="PELT grid resolution; larger = faster, coarser")
    p.add_argument("--n-bootstrap", dest="n_bootstrap", type=int, default=1000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--window", type=int, default=2000, help="samples each side of a change point to test")
    p.add_argument("--out", default="validation/results/retrospective")
    args = p.parse_args()

    pelt_cfg = PeltConfig(cost_model=args.cost_model, penalty_scale=args.penalty_scale,
                          min_segment_length=args.min_segment, jump=args.jump)
    sig_cfg = SignificanceConfig(alpha=args.alpha, n_bootstrap_samples=args.n_bootstrap, window=args.window)
    run(args.dataset, pelt_cfg, sig_cfg, Path(args.out))


if __name__ == "__main__":
    _main()
