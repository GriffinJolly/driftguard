"""
validation/lmsys_gradual_drift.py

Gradual drift from REAL responses: a staged model rollout.

Providers often move traffic to a new model a little at a time. This replays
real English conversations from LMSYS-Chat-1M (lmsys/lmsys-chat-1m) in that
pattern:

    calls     1 .. 3000   all answered by model A           (vicuna-13b)
    calls  3001 .. 4500   share answered by model B rises linearly 0% -> 100%
    calls  4501 .. 6000   all answered by model B           (llama-2-13b-chat)

Model B refuses safe prompts about four times as often (~5.5% -> ~21%), so the
over-refusal signal ramps up gradually. Each reply is a genuine response from
that model to a genuine user prompt; only the mixing schedule is constructed.
For comparison the same trials are run with a SUDDEN swap at call 3001, and as
CONTROLS with model A throughout.

Both detection paths are scored, through driftguard/detection/pipeline.py:
  * live path       : tuned Page-Hinkley, fed one 0/1 refusal per call; its first alarm
  * retrospective   : PELT -> permutation test on adjacent segments -> Benjamini-Hochberg
                      on the same per-call stream; each confirmed change is labelled by
                      classification/rules.py

Safe prompts = prompts OpenAI moderation did not flag. Refusal = the eval suite's
keyword heuristic. Trials draw fresh random conversations each time from the same
two model pools (~360k / ~22k safe prompts), so they are resamples of one dataset,
not independent deployments -- report them that way.

Needs the English cache built by validation/lmsys_model_swap.py
(validation/data/lmsys_english_first_turns.parquet).

Usage:  python -m validation.lmsys_gradual_drift [--trials 30]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from driftguard.classification import rules
from driftguard.detection import pipeline

DATA = Path("validation/data/lmsys_english_first_turns.parquet")
OUT = Path("validation/results/lmsys_gradual_drift")
MODEL_A, MODEL_B = "vicuna-13b", "llama-2-13b-chat"
N_HEALTHY, N_RAMP, N_AFTER = 3000, 1500, 1500
WINDOW_SLACK = 75                          # calls either side of the change window a label may sit
T0 = datetime(2023, 8, 1, tzinfo=timezone.utc)


def make_stream(pool_a: np.ndarray, pool_b: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    a = pool_a[rng.permutation(len(pool_a))]
    b = pool_b[rng.permutation(len(pool_b))]
    n = N_HEALTHY + N_RAMP + N_AFTER
    if kind == "control":
        return a[:n].astype(float)
    if kind == "sudden":
        return np.r_[a[:N_HEALTHY], b[: N_RAMP + N_AFTER]].astype(float)
    ia = ib = 0
    out = np.empty(n)
    for i in range(n):
        if i < N_HEALTHY:
            p_b = 0.0
        elif i < N_HEALTHY + N_RAMP:
            p_b = (i - N_HEALTHY + 1) / N_RAMP
        else:
            p_b = 1.0
        if rng.random() < p_b:
            out[i], ib = b[ib], ib + 1
        else:
            out[i], ia = a[ia], ia + 1
    return out


def detect_trial(values: np.ndarray) -> dict:
    """Run the detection pipeline on one stream of per-call refusals (1 = refused a safe prompt)."""
    calls = pd.DataFrame({
        "seq_id": np.arange(1, len(values) + 1),
        "timestamp": [T0 + timedelta(minutes=i) for i in range(len(values))],
        "outcome": "ok", "latency_ms": np.nan, "eval_run_id": "trial", "eval_task_type": "refusal",
        "eval_task_id": [f"lmsys_safe_{i}" for i in range(len(values))], "eval_score": values,
    })
    det = pipeline.detect(calls, "lmsys-replay", "served-model")
    r = det.results["safe_refusal"]
    labelled = []
    for d in det.events:
        if d.event.detector == "pelt" and d.event.significant:
            c = rules.classify_event(d.event)
            if c.drift_type:
                labelled.append((d.index, c))
    first = labelled[0] if labelled else None
    return dict(ph_first_alert=r.alerts[0] if r.alerts else None, ph_alerts=len(r.alerts),
                label=first[1].drift_type if first else None, label_call=first[0] if first else None,
                label_tier=first[1].evidence_tier if first else None, n_labels=len(labelled))


def main() -> None:
    ap = argparse.ArgumentParser(description="Gradual drift from real LMSYS responses (staged model rollout)")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA)
    safe = df[~df["flagged"]]
    pool_a = safe.loc[safe["model"] == MODEL_A, "refused"].to_numpy()
    pool_b = safe.loc[safe["model"] == MODEL_B, "refused"].to_numpy()
    print(f"Pools of safe prompts: {MODEL_A} n={len(pool_a):,} refusal={pool_a.mean():.3f} | "
          f"{MODEL_B} n={len(pool_b):,} refusal={pool_b.mean():.3f}")

    rng = np.random.default_rng(args.seed)
    rows = []
    example = {}
    ramp_start, ramp_end = N_HEALTHY, N_HEALTHY + N_RAMP
    for kind in ("gradual", "sudden", "control"):
        for t in range(args.trials):
            v = make_stream(pool_a, pool_b, kind, rng)
            res = detect_trial(v)
            rows.append(dict(kind=kind, trial=t, rate_before=v[:ramp_start].mean(),
                             rate_end=v[ramp_end:].mean(), **res))
            if t == 0:
                example[kind] = (v, res)
        print(f"  {kind}: {args.trials} trials done", flush=True)

    r = pd.DataFrame(rows)
    r.to_csv(OUT / "trials.csv", index=False)

    def summarise(g: pd.DataFrame, kind: str) -> dict:
        fa = g["ph_first_alert"]
        drift = kind != "control"
        pre = fa.notna() & (fa < ramp_start)
        hit = fa.notna() & (fa >= ramp_start)
        delay = (fa[hit] - ramp_start)
        lab_ok = (g["label"] == rules.SAFETY_SHIFT)
        # retrospective label placed within the change window (the ramp, or the step), +- slack
        lo, hi = ramp_start - WINDOW_SLACK, (ramp_end if kind == "gradual" else ramp_start) + WINDOW_SLACK
        placed = lab_ok & g["label_call"].between(lo, hi)
        return {
            "trials": len(g),
            "PH detection rate": f"{hit.mean():.0%}" if drift else "–",
            "PH false alarm (before change / any on control)": f"{(pre if drift else fa.notna()).mean():.0%}",
            "PH delay, median (calls after change starts)": f"{delay.median():.0f}" if drift and len(delay) else "–",
            "PH delay, IQR": f"{delay.quantile(.25):.0f}–{delay.quantile(.75):.0f}" if drift and len(delay) else "–",
            "safety_shift label (any place)": f"{lab_ok.mean():.0%}",
            "safety_shift label in change window": f"{placed.mean():.0%}" if drift else "–",
        }

    summary = pd.DataFrame({k: summarise(g, k) for k, g in r.groupby("kind", sort=False)})
    summary = summary[["gradual", "sudden", "control"]]
    pd.set_option("display.width", 200)
    print("\n" + summary.to_string())
    _plot(example)

    lines = ["# Gradual drift from real LMSYS-Chat-1M responses (staged model rollout)", "",
             f"{MODEL_A} → {MODEL_B}, safe (moderation-unflagged) English prompts. "
             f"Calls 1–{N_HEALTHY} all A; calls {N_HEALTHY + 1}–{ramp_end} B's share rises 0→100%; "
             f"then all B until call {ramp_end + N_AFTER}. Sudden trials swap at call {N_HEALTHY + 1}; "
             f"controls are all A. {args.trials} trials each, fresh random conversations per trial "
             "(resamples of one dataset, not independent deployments).", "",
             f"Over-refusal: {pool_a.mean():.3f} ({MODEL_A}) → {pool_b.mean():.3f} ({MODEL_B}).", "",
             "Both paths run per call through driftguard/detection/pipeline.py: Page-Hinkley (λ = 12, first",
             "alarm) and PELT → permutation test on adjacent segments → Benjamini-Hochberg, each confirmed",
             f"change labelled by classification/rules.py. A label counts as placed if it lies in the change",
             f"window (the ramp, or the step) ± {WINDOW_SLACK} calls.", "", summary.to_markdown(), "",
             "![example trials](example_trials.png)", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT}/REPORT.md, trials.csv, example_trials.png")


def _plot(example: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for ax, kind in zip(axes, ("gradual", "sudden", "control")):
        v, res = example[kind]
        roll = pd.Series(v).rolling(200, min_periods=1).mean()
        ax.plot(roll.to_numpy(), color="#2a78d6", lw=1.3, label="Rolling over-refusal (200 calls)")
        if kind == "gradual":
            ax.axvspan(N_HEALTHY, N_HEALTHY + N_RAMP, color="#898781", alpha=0.15, label="rollout: B share 0→100%")
        elif kind == "sudden":
            ax.axvline(N_HEALTHY, color="#898781", lw=1, label="swap")
        if res["ph_first_alert"] is not None:
            ax.axvline(res["ph_first_alert"], color="#d03b3b", lw=1.2, ls="--", label="Page-Hinkley first alert")
        if res["label_call"] is not None:
            ax.axvline(res["label_call"], color="#0b0b0b", lw=1.2,
                       label=f"PELT confirmed change, labelled {res['label']}")
        ax.set_title(f"{kind.capitalize()} (trial 0)", loc="left", fontsize=10)
        ax.set_ylabel("refusal rate")
        ax.legend(fontsize=7, loc="upper left")
    axes[-1].set_xlabel("call")
    fig.tight_layout()
    fig.savefig(OUT / "example_trials.png", dpi=150)


if __name__ == "__main__":
    main()
