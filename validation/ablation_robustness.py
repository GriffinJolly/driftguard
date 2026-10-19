"""
validation/ablation_robustness.py

Two analyses for the paper, on the same streams as baseline_comparison.py
(90 LMSYS-Chat-1M trials, 180 synthetic streams, 7 live trials).

1. Ablation: what each part of the detection pipeline contributes.
     PH fixed          Page-Hinkley, lambda = 12, alpha = 0.9999 (no noise scaling)
     PH scaled         the deployed live detector (noise-scaled, alpha = 0.998)
     PELT alone        every PELT change point in the bad direction, no test
     PELT + perm       ... kept if the permutation test gives raw p < 0.05
     PELT + perm + BH  ... after Benjamini-Hochberg correction (deployed batch path)
     Full system       first of PH scaled and PELT + perm + BH (what an operator sees)
   Rates come with Wilson 95% confidence intervals.

2. Serial dependence.
   a) How dependent are real per-call outcomes? For the live trials the frozen
      suite repeats the same questions, so outcomes cluster by question. We
      report the lag-1 autocorrelation and the intraclass correlation of errors
      by question, and the design effect it implies.
   b) Does the permutation test, which assumes exchangeable observations, keep
      its false-confirmation rate when calls are serially dependent? Stable
      two-state Markov streams with lag-1 correlation 0, 0.3 and 0.6 are run
      through PELT + permutation + BH with the ordinary (i.i.d.) permutation and
      with a block permutation that shuffles contiguous blocks.
   c) Do the confirmed changes on real data survive the block permutation?

Usage:
    python -m validation.ablation_robustness
"""

from __future__ import annotations

import dataclasses
import math
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from driftguard.detection import pipeline
from driftguard.detection.changepoint_pelt import detect_changepoints
from driftguard.detection.multiple_testing import benjamini_hochberg
from validation.baseline_comparison import lmsys_streams, live_streams, synthetic_streams

OUT = Path("validation/results/ablation_robustness")
# where the trial databases live (they are git-ignored, so a fresh checkout may not have them)
DATA_ROOT = Path(os.environ.get("DRIFTGUARD_DATA_ROOT", "."))
S = pipeline.load_settings()
S_FIXED = dataclasses.replace(S, ph_alpha=0.9999, ph_noise_ref=None)
ALPHA = 0.05
VARIANTS = ["PH fixed", "PH scaled", "PELT alone", "PELT + perm", "PELT + perm + BH", "Full system"]
PELT_BASED = {"PELT alone", "PELT + perm", "PELT + perm + BH"}


# --------------------------------------------------------------------------- statistics helpers

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def perm_pvalue(left: np.ndarray, right: np.ndarray, n_perm: int, rng: np.random.Generator,
                block: int | None = None) -> float:
    """Two-sided permutation p-value for a difference in means. With `block`, contiguous
    blocks of that length are shuffled instead of single observations."""
    observed = abs(right.mean() - left.mean())
    pooled = np.concatenate([left, right])
    n_left = len(left)
    if block:
        blocks = [pooled[i:i + block] for i in range(0, len(pooled), block)]
    count = 0
    for _ in range(n_perm):
        if block:
            perm = np.concatenate([blocks[j] for j in rng.permutation(len(blocks))])
        else:
            perm = rng.permutation(pooled)
        if abs(perm[n_left:].mean() - perm[:n_left].mean()) >= observed - 1e-12:
            count += 1
    return (1 + count) / (1 + n_perm)


def tested_changes(v: np.ndarray, block: int | None = None, n_perm: int | None = None,
                   settings: pipeline.DetectionSettings = S) -> list[dict]:
    """PELT change points with pre/post means and raw and BH-adjusted p-values, using the
    same windows as the deployed path; `block` switches to the block permutation."""
    cps = detect_changepoints(v, pipeline.effective_pelt(len(v), settings))
    if not cps:
        return []
    n_perm = n_perm or S.significance.n_bootstrap_samples
    win = S.significance.window
    bounds = [0, *cps, len(v)]
    out = []
    for i, cp in enumerate(cps):
        left = v[bounds[i]:cp][-win:] if win else v[bounds[i]:cp]
        right = v[cp:bounds[i + 2]][:win] if win else v[cp:bounds[i + 2]]
        p = perm_pvalue(left, right, n_perm, np.random.default_rng(i), block)
        out.append({"index": cp, "pre": float(left.mean()), "post": float(right.mean()), "p_raw": p})
    sig, adj = benjamini_hochberg([r["p_raw"] for r in out], ALPHA)
    for r, s_, a in zip(out, sig, adj):
        r["p_adj"], r["sig_bh"] = float(a), bool(s_)
    return out


def variants(v: np.ndarray) -> dict[str, list[int]]:
    t = tested_changes(v)
    bad = [r for r in t if r["post"] > r["pre"]]
    ph_scaled = pipeline.page_hinkley_alerts(v, S)
    confirmed = [r["index"] for r in bad if r["sig_bh"]]
    return {
        "PH fixed": pipeline.page_hinkley_alerts(v, S_FIXED),
        "PH scaled": ph_scaled,
        "PELT alone": [r["index"] for r in bad],
        "PELT + perm": [r["index"] for r in bad if r["p_raw"] < ALPHA],
        "PELT + perm + BH": confirmed,
        "Full system": sorted(ph_scaled[:1] + confirmed[:1]),
    }


def score(alarms: list[int], onset: int | None, slack: int) -> dict:
    if onset is None:
        return {"false_alarm": bool(alarms), "found": None}
    early = [a for a in alarms if a < onset - slack]
    late = [a for a in alarms if a >= onset - slack]
    return {"false_alarm": bool(early), "found": bool(late) and not early}


# --------------------------------------------------------------------------- 1. ablation

def ablation() -> pd.DataFrame:
    rows = []
    for source in (lmsys_streams(), synthetic_streams(), live_streams()):
        for group, v, onset in source:
            slack = pipeline.effective_pelt(len(v), S).min_segment_length
            for name, alarms in variants(v).items():
                # the full system's confirmed part is located on PELT's grid, its live part is not
                sl = slack if name in PELT_BASED or name == "Full system" else 0
                rows.append({"group": group, "variant": name, **score(alarms, onset, sl)})
        print("ablation done", group, flush=True)
    return pd.DataFrame(rows)


def pooled_table(raw: pd.DataFrame, label_col: str, names: list[str]) -> pd.DataFrame:
    if "found" not in raw:   # baseline_comparison rows: found = detected with no earlier alarm
        raw = raw.assign(found=raw["detected"].where(raw["detected"].isna(),
                                                     raw["detected"].astype(bool) & ~raw["false_alarm"].astype(bool)))
    stable = raw[raw.group.isin(["LMSYS control", "Synthetic 5% stable", "Synthetic 15% stable", "Synthetic 30% stable"])]
    change = raw[raw.group.isin(["LMSYS gradual", "LMSYS sudden", "Synthetic 5% +10 pts",
                                 "Synthetic 15% +10 pts", "Synthetic 30% +10 pts"])]
    noisy = change[change.group.isin(["Synthetic 15% +10 pts", "Synthetic 30% +10 pts"])]
    live = raw[raw.group.str.startswith("Live")]
    out = []
    for m in names:
        def rate(d, col):
            x = d[d[label_col] == m][col].dropna().astype(bool)
            k, n = int(x.sum()), len(x)
            lo, hi = wilson(k, n)
            return k, n, lo, hi
        fa = rate(stable, "false_alarm")
        found = rate(change, "found")
        nz = rate(noisy, "found")
        lv = rate(live, "found")
        out.append({label_col: m,
                    "false_alarm": f"{fa[0]}/{fa[1]} = {fa[0]/fa[1]:.0%} [{fa[2]:.0%}, {fa[3]:.0%}]",
                    "found": f"{found[0]}/{found[1]} = {found[0]/found[1]:.0%} [{found[2]:.0%}, {found[3]:.0%}]",
                    "found_noisy": f"{nz[0]}/{nz[1]} = {nz[0]/nz[1]:.0%} [{nz[2]:.0%}, {nz[3]:.0%}]",
                    "live": f"{lv[0]}/{lv[1]}"})
    return pd.DataFrame(out)


def penalty_ablation(scale: float = 1.0) -> pd.DataFrame:
    """The batch path with PELT's penalty lowered to scale x sigma^2 log n (the deployed value
    is 5), where PELT on its own proposes many more change points, to show what the
    permutation test and the Benjamini-Hochberg correction remove."""
    low = dataclasses.replace(S, pelt=dataclasses.replace(S.pelt, penalty_scale=scale))
    rows = []
    for source in (lmsys_streams(), synthetic_streams()):
        for group, v, onset in source:
            t = [r for r in tested_changes(v, settings=low) if r["post"] > r["pre"]]
            slack = pipeline.effective_pelt(len(v), low).min_segment_length
            for name, alarms in ((f"PELT alone (x{scale:g})", [r["index"] for r in t]),
                                 (f"PELT + perm (x{scale:g})", [r["index"] for r in t if r["p_raw"] < ALPHA]),
                                 (f"PELT + perm + BH (x{scale:g})", [r["index"] for r in t if r["sig_bh"]])):
                rows.append({"group": group, "variant": name, **score(alarms, onset, slack)})
        print("penalty ablation done", group, flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / f"penalty_x{scale:g}_raw.csv", index=False)
    return pooled_table(raw, "variant", list(dict.fromkeys(raw["variant"])))


# --------------------------------------------------------------------------- 2. serial dependence

def live_trials() -> list[tuple[str, pd.DataFrame]]:
    """The live trials with question ids: (name, scored accuracy calls in order)."""
    sources = [(DATA_ROOT / f"validation/data/nvidia_trial_{n}.db", f"NVIDIA trial {n}", "openai/gpt-oss-20b", None)
               for n in range(1, 6)]
    sources += [(DATA_ROOT / "driftguard.db", f"Groq trial {k}", "openai/gpt-oss-120b", rid)
                for k, rid in ((1, "live_drift_compare_sudden_1789368091"), (2, "live_drift_compare_sudden_1789444819"))]
    # round two: Nemotron-3-super-120b on the 100-question suite (live_trials_v2_pipeline.py)
    sources += [(DATA_ROOT / f"validation/data/nvidia_{kind}_{n}.db", f"Round 2 {kind} {n}",
                 "nvidia/nemotron-3-super-120b-a12b", None)
                for kind, n in (("long", 1), ("long", 2), ("long", 3), ("swap", 1), ("swap", 2))]
    out = []
    for db, name, model, rid in sources:
        if not Path(db).exists():
            continue
        con = sqlite3.connect(db)
        q = ("SELECT eval_task_id, eval_score FROM log_rows WHERE outcome='ok' AND eval_score IS NOT NULL "
             "AND eval_task_type='accuracy' AND model_id=?")
        params = [model]
        if rid:
            q += " AND eval_run_id=?"
            params.append(rid)
        df = pd.read_sql_query(q + " ORDER BY rowid", con, params=params)
        con.close()
        if len(df) >= 200:
            out.append((name, df.assign(e=1.0 - df["eval_score"])))
    return out


def live_dependence() -> pd.DataFrame:
    """Lag-1 autocorrelation and intraclass correlation by question, before the change."""
    rows = []
    for name, df in live_trials():
        pre = df.iloc[:175]
        e = 1.0 - pre["eval_score"].to_numpy()
        lag1 = float(np.corrcoef(e[:-1], e[1:])[0, 1]) if e.std() > 0 else 0.0
        g = pre.assign(e=e).groupby("eval_task_id")["e"]
        k, m = g.ngroups, len(e) / g.ngroups
        msb = (g.size() * (g.mean() - e.mean()) ** 2).sum() / max(1, k - 1)
        msw = ((pre.assign(e=e)["e"] - g.transform("mean")) ** 2).sum() / max(1, len(e) - k)
        icc = float((msb - msw) / (msb + (m - 1) * msw)) if (msb + (m - 1) * msw) > 0 else 0.0
        icc = max(0.0, icc)
        deff = 1 + (m - 1) * icc
        rows.append({"trial": name, "calls_before": len(e), "questions": k, "repeats_per_question": round(m, 1),
                     "error_rate": round(float(e.mean()), 3), "lag1_autocorr": round(lag1, 3),
                     "icc_by_question": round(icc, 3), "design_effect": round(deff, 2),
                     "effective_n": round(len(e) / deff, 1)})
    return pd.DataFrame(rows)


def question_level_tests(onset: int = 175, n_perm: int = 2000) -> pd.DataFrame:
    """Live trials: is the change still significant when calls on the same question are not
    treated as independent? Three tests of 'error rate after > before', one-sided:
      call-level     ordinary permutation of all calls (what the pipeline assumes)
      within-question permutation of before/after labels within each question only
      paired         sign test over questions whose error rate went up vs down"""
    rows = []
    for name, df in live_trials():
        before, after = df.iloc[:onset], df.iloc[onset:]
        obs = after["e"].mean() - before["e"].mean()
        rng = np.random.default_rng(7)
        e, is_after = df["e"].to_numpy(), np.r_[np.zeros(len(before)), np.ones(len(after))].astype(bool)
        c_call = sum((lambda m: e[m].mean() - e[~m].mean())(rng.permutation(is_after)) >= obs - 1e-12
                     for _ in range(n_perm))
        groups = [np.flatnonzero(df["eval_task_id"].to_numpy() == q) for q in df["eval_task_id"].unique()]
        c_q = 0
        for _ in range(n_perm):
            lab = is_after.copy()
            for g in groups:
                lab[g] = rng.permutation(lab[g])
            c_q += (e[lab].mean() - e[~lab].mean()) >= obs - 1e-12
        per_q = df.assign(after=is_after).groupby(["eval_task_id", "after"])["e"].mean().unstack()
        per_q = per_q.dropna()
        up, down = int((per_q[True] > per_q[False]).sum()), int((per_q[True] < per_q[False]).sum())
        from scipy.stats import binomtest
        p_sign = binomtest(up, up + down, 0.5, alternative="greater").pvalue if up + down else 1.0
        rows.append({"trial": name, "error_before": round(before["e"].mean(), 3), "error_after": round(after["e"].mean(), 3),
                     "p_call_level": round((1 + c_call) / (1 + n_perm), 4),
                     "p_within_question": round((1 + c_q) / (1 + n_perm), 4),
                     "questions_up_down": f"{up} / {down}", "p_sign_test": round(float(p_sign), 4)})
    return pd.DataFrame(rows)


def markov_stream(n: int, p: float, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Stationary two-state Markov chain with P(1) = p and lag-1 correlation rho."""
    p11 = p + rho * (1 - p)
    p01 = p * (1 - rho)
    x = np.empty(n)
    x[0] = rng.random() < p
    for i in range(1, n):
        x[i] = rng.random() < (p11 if x[i - 1] else p01)
    return x


def dependence_false_confirmations(n_streams: int = 60, n: int = 2000, p: float = 0.15) -> pd.DataFrame:
    rows = []
    for rho in (0.0, 0.3, 0.6):
        for block in (None, 20, 50):
            hits = 0
            for t in range(n_streams):
                v = markov_stream(n, p, rho, np.random.default_rng([41, int(rho * 10), t]))
                tested = tested_changes(v, block=block, n_perm=500)
                hits += any(r["sig_bh"] for r in tested)
            lo, hi = wilson(hits, n_streams)
            rows.append({"lag1_corr": rho, "permutation": "i.i.d." if block is None else f"blocks of {block}",
                         "false_confirmation": f"{hits}/{n_streams} = {hits / n_streams:.0%} [{lo:.0%}, {hi:.0%}]"})
        print("dependence done rho", rho, flush=True)
    return pd.DataFrame(rows)


def real_changes_under_blocks() -> pd.DataFrame:
    """Confirmed changes on the LMSYS trials and live trials, re-tested with block permutations."""
    rows = []
    for source in (lmsys_streams(), live_streams()):
        for group, v, onset in source:
            if onset is None:
                continue
            for block in (None, 20, 50):
                t = tested_changes(v, block=block, n_perm=500)
                slack = pipeline.effective_pelt(len(v), S).min_segment_length
                ok = any(r["sig_bh"] and r["post"] > r["pre"] and r["index"] >= onset - slack for r in t)
                rows.append({"group": group.split(" ")[0] + " " + group.split(" ")[1],
                             "permutation": "i.i.d." if block is None else f"blocks of {block}", "confirmed": ok})
    d = pd.DataFrame(rows)
    out = d.groupby(["group", "permutation"], sort=False)["confirmed"].agg(["sum", "size"]).reset_index()
    out["confirmed"] = out["sum"].astype(int).astype(str) + "/" + out["size"].astype(str)
    return out[["group", "permutation", "confirmed"]]


# --------------------------------------------------------------------------- main

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = ablation()
    raw.to_csv(OUT / "ablation_raw.csv", index=False)
    abl = pooled_table(raw, "variant", VARIANTS)
    abl.to_csv(OUT / "ablation.csv", index=False)

    base_raw = pd.read_csv("validation/results/baseline_comparison/raw.csv")
    base = pooled_table(base_raw, "method", ["Static threshold", "Periodic re-eval", "Periodic, corrected",
                                             "DriftGuard live", "DriftGuard confirmed"])
    base.to_csv(OUT / "baselines_with_ci.csv", index=False)

    dep = live_dependence()
    dep.to_csv(OUT / "live_dependence.csv", index=False)
    ql = question_level_tests()
    ql.to_csv(OUT / "live_question_level_tests.csv", index=False)
    fc = dependence_false_confirmations()
    fc.to_csv(OUT / "dependence_false_confirmations.csv", index=False)
    blocks = real_changes_under_blocks()
    blocks.to_csv(OUT / "real_changes_block_permutation.csv", index=False)

    lines = ["# Ablation and statistical robustness", "",
             "Rates are k/n with Wilson 95% confidence intervals. False alarm: 120 stable streams (30 LMSYS controls, "
             "90 synthetic). Found: 150 streams with a change (60 LMSYS, 90 synthetic), counted only if no alarm came "
             "before the change; noisy = synthetic at 15% and 30% error. Live: the 7 Groq and NVIDIA trials.", "",
             "## 1. Ablation", "", abl.to_markdown(index=False), "",
             "## Baselines with confidence intervals", "", base.to_markdown(index=False), "",
             "## 2a. Dependence of live per-call outcomes (before the change)", "",
             "The frozen suite cycles the same questions, so outcomes cluster by question. Design effect = "
             "1 + (m - 1) ICC for m repeats per question; effective n = n / design effect.", "",
             dep.to_markdown(index=False), "",
             "### Live changes tested at the level of questions", "",
             "One-sided tests that the error rate rose after call 175. Call-level permutation treats calls as "
             "independent (as the pipeline does); the within-question permutation shuffles before/after labels only "
             "among calls on the same question; the sign test counts questions whose error rate went up versus down.", "",
             ql.to_markdown(index=False), "",
             "## 2b. False confirmations on stable, serially dependent streams", "",
             "60 two-state Markov streams of 2,000 calls at a 15% error rate per row; PELT + permutation (500) + BH.", "",
             fc.to_markdown(index=False), "",
             "## 2c. Real changes re-tested with block permutations", "", blocks.to_markdown(index=False), ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
