"""
validation/wildchat_version_switch.py

Drift in the wild: does DriftGuard notice a REAL provider model update?

WildChat-1M (allenai/WildChat-1M, ODC-BY) logs ~1M real user conversations with
OpenAI's chat models through one public app, and records which model VERSION
answered each one. Behind the same app, the GPT-3.5 service moved from
gpt-3.5-turbo-0301 to gpt-3.5-turbo-0613 on 28 June 2023. That is a genuine,
provider-side update with a known date -- the scenario DriftGuard exists for --
so it is used here as ground truth.

What is measured (English conversations, first assistant reply only):
  * safe_refusal       -- refused a prompt that OpenAI's moderation did NOT flag
                          (over-refusal, Safety)
  * unsafe_compliance  -- did NOT refuse a prompt that moderation flagged
                          (under-refusal, Safety)
Refusal is detected with the same keyword heuristic as the eval suite
(driftguard/evalsuite/refusal_probe.py), so the same caveats apply.

Both detection paths and the attribution rules run exactly as in the dashboard
(driftguard/dashboard/data.py). The model-version field is used ONLY to score
the result afterwards; the detectors never see it.

Caveats, stated plainly:
  * These are real users, not a frozen suite: the prompt mix itself can change
    over time. To separate "the model changed" from "the users changed", the
    script also compares IDENTICAL prompts that were sent both before and after
    the switch (paired check).
  * The keyword refusal heuristic is English-only and imperfect.

Usage:
    python -m validation.wildchat_version_switch            # shards 3-5 (7 Jun - 13 Aug 2023)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from driftguard.dashboard import data as D
from driftguard.detection.signals import SIGNALS
from driftguard.evalsuite.refusal_probe import _is_refusal

REPO = "datasets/allenai/WildChat-1M/data/train-{k:05d}-of-00014.parquet"
OLD, NEW = "gpt-3.5-turbo-0301", "gpt-3.5-turbo-0613"
OUT = Path("validation/results/wildchat_version_switch")
CACHE = Path("validation/data/wildchat_gpt35_jun_aug_2023.parquet")
TEXT_CAP = 2000  # characters kept of each prompt / reply


def extract(shards: list[int]) -> pd.DataFrame:
    """One row per English GPT-3.5 conversation: time, version, first prompt/reply,
    whether moderation flagged the prompt, whether the reply is a refusal."""
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    rows = []
    for k in shards:
        pf = pq.ParquetFile(fs.open(REPO.format(k=k)))
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["conversation_hash", "model", "timestamp", "language", "conversation",
                                               "openai_moderation"]).to_pandas()
            t = t[t["model"].isin([OLD, NEW]) & (t["language"] == "English")]
            for _, r in t.iterrows():
                conv = list(r["conversation"])
                user = next((m for m in conv if m["role"] == "user"), None)
                bot = next((m for m in conv if m["role"] == "assistant"), None)
                mod = list(r["openai_moderation"]) if r["openai_moderation"] is not None else []
                if user is None or bot is None or not mod:
                    continue
                rows.append(dict(timestamp=r["timestamp"], model=r["model"], conversation_id=r["conversation_hash"],
                                 prompt=user["content"][:TEXT_CAP], reply=(bot["content"] or "")[:TEXT_CAP],
                                 flagged=bool(mod[0]["flagged"]), refused=_is_refusal(bot["content"])))
        print(f"  shard {k}: {len(rows):,} conversations so far")
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE, index=False)
    return df


def to_calls(frame: pd.DataFrame) -> pd.DataFrame:
    """The conversations as per-call refusal tasks on safe prompts, in the log_rows shape the
    detection pipeline reads ("_safe_" task ids mark prompts that should be answered)."""
    return pd.DataFrame({
        "seq_id": np.arange(1, len(frame) + 1), "timestamp": frame["timestamp"].to_numpy(),
        "outcome": "ok", "latency_ms": np.nan, "eval_run_id": "wildchat",
        "eval_task_type": "refusal", "eval_task_id": [f"wildchat_safe_{i}" for i in range(len(frame))],
        "eval_score": frame["refused"].astype(float).to_numpy(),
    })


def paired_check(df: pd.DataFrame, switch: pd.Timestamp) -> pd.DataFrame:
    """Refusal rate on IDENTICAL prompts asked both before and after the switch."""
    d = df.assign(norm=df["prompt"].str.strip().str.lower(), after=df["timestamp"] >= switch)
    both = d.groupby("norm")["after"].nunique()
    shared = d[d["norm"].isin(both[both == 2].index)]
    out = []
    for flagged, g in shared.groupby("flagged"):
        per = g.groupby(["norm", "after"])["refused"].mean().unstack()
        out.append(dict(prompts="moderation-flagged" if flagged else "not flagged", n_prompts=len(per),
                        refusal_before=per[False].mean(), refusal_after=per[True].mean(),
                        prompts_refused_more=int((per[True] > per[False]).sum()),
                        prompts_refused_less=int((per[True] < per[False]).sum())))
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--shards", default="3,4,5")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print("Extracting WildChat-1M conversations (cached after the first run)...")
    df = extract([int(s) for s in args.shards.split(",")])
    switch = df.loc[df["model"] == NEW, "timestamp"].min()
    print(f"{len(df):,} English GPT-3.5 conversations, {df.timestamp.min():%d %b} - {df.timestamp.max():%d %b %Y}")
    print(f"Ground truth: first {NEW} reply at {switch:%Y-%m-%d %H:%M} UTC\n")

    # WildChat-1M ships only the non-toxic subset, so moderation-flagged prompts (needed for
    # the under-refusal signal) are absent; that signal is reported as untestable here.
    safe = df[~df["flagged"]]
    if df["flagged"].any():
        print(f"note: {int(df['flagged'].sum())} flagged prompts found but not analysed")
    # Two views of the same over-refusal signal:
    #   raw      -- every conversation, as passive traffic monitoring would see it
    #   deduped  -- a prompt repeated within the same UTC day is kept once. Real users resend
    #               the same prompt in bursts (scripts, retries); this removes those bursts
    #               WITHOUT using the model-version field or the switch date.
    day = safe["timestamp"].dt.floor("D")
    norm = safe["prompt"].str.strip().str.lower().str[:200]
    dedup = safe[~pd.DataFrame({"d": day, "n": norm}).duplicated()]
    views = {"raw": safe, "deduped": dedup}
    cfg = D.load_settings()
    all_results, rows, shift_tables = {}, [], []
    for view, frame in views.items():
        a = D.analyse_scope(to_calls(frame), "openai (WildChat)", "gpt-3.5-turbo", cfg)
        all_results[view] = a.results["safe_refusal"]
        shift_tables.append(D.shift_table(a).assign(view=view))
    shifts = pd.concat(shift_tables, ignore_index=True)

    for view, r in all_results.items():
        key = "safe_refusal"
        st = r.stream
        onset = int(st.timestamp.searchsorted(switch))
        conf = [t for t in r.tested if t["significant"]]
        nearest = min(conf, key=lambda t: abs(t["index"] - onset)) if conf else None
        before, after = st.values[:onset], st.values[onset:]
        w = pd.Timedelta(days=14)
        tsr = st.timestamp
        b14 = st.values[((tsr >= switch - w) & (tsr < switch)).to_numpy()]
        a14 = st.values[((tsr >= switch) & (tsr < switch + w)).to_numpy()]
        pool = (b14.sum() + a14.sum()) / (len(a14) + len(b14))
        z14 = (a14.mean() - b14.mean()) / np.sqrt(pool * (1 - pool) * (1 / len(a14) + 1 / len(b14)))
        rows.append(dict(
            view=view, signal=SIGNALS[key].label, metric=SIGNALS[key].metric, n=len(st.values), onset_index=onset,
            rate_before=before.mean(), rate_after=after.mean(),
            confirmed_changes=len(conf),
            nearest_confirmed_index=None if nearest is None else nearest["index"],
            nearest_offset_calls=None if nearest is None else nearest["index"] - onset,
            nearest_offset_hours=None if nearest is None else
            (st.timestamp.iloc[nearest["index"]] - switch).total_seconds() / 3600,
            nearest_pre=None if nearest is None else nearest["pre_mean"],
            nearest_post=None if nearest is None else nearest["post_mean"],
            ph_alerts=len(r.alerts),
            ph_first_after_switch=next((i - onset for i in r.alerts if i >= onset), None),
            ph_alerts_before_switch=sum(i < onset for i in r.alerts),
            rate_14d_before=b14.mean(), rate_14d_after=a14.mean(), z_14d=z14,
        ))
    summary = pd.DataFrame(rows)
    paired = paired_check(df, switch)

    summary.to_csv(OUT / "summary.csv", index=False)
    shifts.to_csv(OUT / "all_shifts.csv", index=False)
    paired.to_csv(OUT / "paired_identical_prompts.csv", index=False)
    pd.set_option("display.width", 220)
    print(summary.round(4).to_string(index=False))
    print("\nPaired check (identical prompts before vs after):")
    print(paired.round(4).to_string(index=False))
    _plot(all_results, switch)


def _plot(results, switch) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(results), 1, figsize=(9, 2.6 * len(results)), sharex=True)
    for ax, (view, r) in zip(np.atleast_1d(axes), results.items()):
        st = r.stream
        ts = st.timestamp
        ax.plot(ts, D.rolling_rate(st.values, 300), color="#2a78d6", lw=1.2, label="Rolling refusal rate (300 conversations)")
        for a, b, m in r.segments:
            ax.hlines(m, ts.iloc[a], ts.iloc[min(b, len(ts)) - 1], color="#0b0b0b", lw=2)
        for t in r.tested:
            if t["significant"]:
                ax.axvline(ts.iloc[t["index"]], color="#898781", lw=0.8)
        ax.axvline(switch, color="#d03b3b", lw=1.2, ls="--", label=f"{OLD} -> {NEW}")
        title = {"raw": "Over-refusal, all conversations (passive traffic)",
                 "deduped": "Over-refusal, prompts repeated within a day kept once"}[view]
        ax.set_title(f"{title}  ·  {len(st.values):,} conversations", loc="left", fontsize=10)
        ax.set_ylabel("rate")
        ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "wildchat_switch_timeline.png", dpi=150)
    print(f"\nWrote {OUT}/ (summary.csv, all_shifts.csv, paired_identical_prompts.csv, timeline png)")


if __name__ == "__main__":
    main()
