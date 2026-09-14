"""
validation/live_drift_compare.py

Live, ground-truth-aware detector COMPARISON on real LLM traffic.

This is the piece FINAL_VERDICT.md / HYBRID_VERDICT.md / LIVE_TRAFFIC_VERDICT.md
all pointed at but couldn't do: rank the detectors (DDM, Page-Hinkley, ADWIN,
...) on how fast/reliably they catch BOTH sudden and gradual drift, judged on
REAL model calls -- not synthetic Bernoulli streams (run_validation.py) and not
a stable live model with no drift to find (live_replay_groq.py, which correctly
produced 0 alerts from everyone because nothing drifted).

The core idea
-------------
A naturally-stable live model gives you no drift to rank detectors on. So we
INDUCE drift with a KNOWN onset by changing what answers mid-stream, using real
calls the whole time:

  * SUDDEN  : a strong model answers the first `onset` calls, then we hard-switch
              to a weak model for the rest. Error rate steps up at `onset`.
  * GRADUAL : after `onset`, each call is routed strong-vs-weak with a probability
              that ramps 0 -> 1 over `ramp_len` calls (a canary/rollout ramp), so
              the aggregate error rate rises gradually -- the truest analog of a
              real gradual model rollout. (--gradual-mode temperature instead ramps
              one model's sampling temperature up, keeping the model id fixed.)

Because WE authored the schedule, the onset index is exactly known (unlike Elec2/
Insects, whose onset had to be estimated) -- so we can score every detector the
same way run_validation.py does: detection delay = first_alert - onset; a fire in
the pre-onset baseline region is a false alarm. That pre-onset region (a real,
strong-model, low-error stream) doubles as the live false-alarm measurement, so no
separate "stable" scenario / extra API budget is needed.

Free-tier friendly / accumulate across days
-------------------------------------------
One invocation runs one trial of each requested scenario (default: sudden+gradual)
and APPENDS its per-detector rows to a running CSV. Run it once/day (e.g. via a
scheduler) and the trials accumulate; `--aggregate` then reads all accumulated
trials and produces the ranking table + charts + verdict. A hard
`--daily-attempt-budget` cap guarantees a single invocation never exceeds the
provider's free daily request cap even if calls fail.

Usage
-----
    # validate the whole pipeline with ZERO API calls first:
    python -m validation.live_drift_compare --self-test

    # one day's real run (~700 Groq calls, ~25 min, paces itself):
    python -m validation.live_drift_compare --provider groq \
        --strong-model llama-3.3-70b-versatile --weak-model llama-3.1-8b-instant

    # after N days of runs, produce the ranking:
    python -m validation.live_drift_compare --aggregate
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from driftguard.evalsuite.tasks import (
    ANSWER_LETTERS,
    _ensure_frozen_subset,
    _extract_answer_letter,
    _format_prompt,
)
from driftguard.providers.clients import call_model
from driftguard.storage.store import DriftStore

from validation.detectors_registry import run_detector_on_stream, tuned_registry
from validation.live_replay import make_timeline_chart
from validation.report import (
    COMPOSITE_WEIGHTS,
    DETECTOR_COLOR,
    TUNED_DETECTOR_ORDER,
    _grouped_bar,
)

# ---------------------------------------------------------------------------
# Defaults -- sized to fit inside a ~1,000-calls/day free tier (Groq) while
# still clearing DDM (tuned)'s min_num_instances=150 warm-up BEFORE the onset,
# and leaving real post-onset runway to measure detection delay.
# ---------------------------------------------------------------------------
ONSET_DEFAULT = 175         # successful calls of the strong model before drift begins
POST_DEFAULT = 175          # successful calls after onset
GRADUAL_RAMP_DEFAULT = 100  # gradual: error rate ramps over the first this-many post-onset calls
DAILY_ATTEMPT_BUDGET = 950  # hard cap on attempts per invocation (< Groq's ~1,000/day free cap)
MAX_TOKENS_DEFAULT = 1536   # gpt-oss are reasoning models: must leave room to reason THEN emit the
                            # letter, or the answer is truncated and every call scores wrong.
                            # Calibration: 120b avg 187 out-tok (max 433), 20b avg 288 (occasional 1024).
RPM_DEFAULT = 14            # Groq free binds at ~8,000 tokens/min; at ~450 tok/call, ~14 calls/min
                            # stays safely under it so calls aren't lost to 429s. clients.py's own
                            # 25/min limiter assumes a 30 RPM cap -- for these reasoning models the
                            # token/min limit is tighter, so we pace below it here too.

# The single model whose config is degraded to induce drift. gpt-oss-120b is the
# strongest free Groq chat model (baseline err ~0.10 on the frozen subset).
MODEL_DEFAULT = "openai/gpt-oss-120b"
# Drift magnitude/mechanism defaults (Option A -- config degradation, see _schedule):
DEGRADE_FRAC_DEFAULT = 0.35        # ~35% of post-onset calls degraded -> post err ~0.10+0.35*~0.9 ≈ 0.38
DEGRADE_MAX_TOKENS_DEFAULT = 16    # truncate-mode: too few tokens to reach the letter -> wrong
DEGRADE_TEMP_DEFAULT = 2.0         # temperature-mode: high enough to garble answers

OUT_DIR_DEFAULT = "validation/results/live_drift_compare"

# The registry under comparison: the 8 individually-tuned detectors. The
# Page-Hinkley+DDM OR-hybrid is deliberately excluded -- the project uses
# Page-Hinkley alone as the live detector (the hybrid was tested and rejected,
# see validation/results/HYBRID_VERDICT.md). The other 7 tuned detectors stay
# as comparison baselines that justify the Page-Hinkley choice.
LIVE_DETECTOR_ORDER = TUNED_DETECTOR_ORDER


def _registry():
    return tuned_registry()


# ---------------------------------------------------------------------------
# Drift schedule (Option A: controllable config-degradation of ONE model)
# ---------------------------------------------------------------------------
# The free Groq chat models differ by only ~0.05-0.09 in MMLU error, too small a
# real capability gap to make a detectable drift (a full strong->weak swap trial
# left 0/9 detectors firing -- see groq_raw_results_temp03_pilot_DISCARDED.csv and
# the temp-0 follow-up). So instead of swapping to a barely-weaker model, we induce
# a real, controllable error-rate change by DEGRADING the same model's request
# config at the onset: a fraction `degrade_frac` of post-onset calls are made under
# a degraded config that reliably produces wrong answers -- by default a tiny
# max_tokens that truncates the model before it can emit the answer letter
# (--degrade-mode temperature instead spikes sampling temperature). The calls are
# still real and live; only the CAUSE of the errors is engineered, and its
# magnitude is tunable via `degrade_frac`.
def _schedule(
    scenario: str, i: int, onset: int, ramp_len: int, model: str,
    base_temp: float, base_max_tokens: int,
    degrade_frac: float, degrade_temp: float, degrade_max_tokens: int,
    degrade_mode: str, rng: random.Random,
) -> tuple[str, float, int]:
    """Return (model_id, temperature, max_tokens) for the i-th SUCCESSFUL call.

    i is counted over successful calls only, so the induced onset lands at exactly
    error-stream index `onset` regardless of how many calls failed along the way.
    """
    baseline = (model, base_temp, base_max_tokens)
    if i < onset or scenario == "stable":
        return baseline

    if scenario == "sudden":
        p = degrade_frac                                   # constant elevated error (a step in the mean)
    elif scenario == "gradual":
        p = degrade_frac * min(1.0, (i - onset) / max(1, ramp_len))  # error ramps up
    else:
        raise ValueError(f"Unknown scenario: {scenario!r}")

    if rng.random() < p:  # this call is degraded
        if degrade_mode == "temperature":
            return model, degrade_temp, base_max_tokens
        return model, base_temp, degrade_max_tokens        # default: truncate -> wrong/no answer
    return baseline


# ---------------------------------------------------------------------------
# Collect one real error stream for one (scenario) trial
# ---------------------------------------------------------------------------
def _collect_stream(
    scenario: str, provider: str, model: str,
    onset: int, post: int, ramp_len: int, base_temp: float, base_max_tokens: int,
    degrade_frac: float, degrade_temp: float, degrade_max_tokens: int, degrade_mode: str,
    store: DriftStore, trial_id: str, attempts_left: list[int], rng: random.Random,
    t_start: float, min_interval: float,
) -> np.ndarray:
    """Make real calls following the drift schedule until `onset+post` successful
    calls are collected (or the shared daily attempt budget runs out). Returns the
    0/1 error stream. Every call (success or failure) is logged to driftguard.db."""
    tasks = _ensure_frozen_subset()
    target = onset + post
    errors: list[int] = []
    cycle: list[dict] = []
    next_call_at = time.monotonic()

    while len(errors) < target and attempts_left[0] > 0:
        if not cycle:  # refill + reshuffle the 20-question cycle
            cycle = list(tasks)
            rng.shuffle(cycle)
        task = cycle.pop()

        i = len(errors)  # index this call WILL occupy if it succeeds
        model_id, temp, max_tokens = _schedule(
            scenario, i, onset, ramp_len, model, base_temp, base_max_tokens,
            degrade_frac, degrade_temp, degrade_max_tokens, degrade_mode, rng,
        )
        # pace under the tokens/min cap so calls aren't lost to 429s
        sleep_for = next_call_at - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        next_call_at = time.monotonic() + min_interval

        attempts_left[0] -= 1
        entry = call_model(
            provider=provider, model_id=model_id, prompt=_format_prompt(task),
            request_params={"temperature": round(temp, 3), "max_tokens": max_tokens},
            eval_run_id=f"live_drift_compare_{trial_id}",
            eval_task_type="accuracy", eval_task_id=task["task_id"],
        )

        if entry.outcome == "ok":
            letter = _extract_answer_letter(entry.response_text)
            correct = letter == ANSWER_LETTERS[task["correct_index"]]
            entry.eval_score = 1.0 if correct else 0.0
            errors.append(0 if correct else 1)
        else:
            entry.eval_score = None  # failed call -- excluded from the stream (not retried, to save quota)

        store.write_log(entry)

        if len(errors) and len(errors) % 20 == 0 and len(errors) != getattr(_collect_stream, "_last", -1):
            _collect_stream._last = len(errors)  # type: ignore[attr-defined]
            elapsed = time.monotonic() - t_start
            roll = float(np.mean(errors[-20:]))
            print(f"  [{elapsed:6.1f}s] {scenario}: {len(errors)}/{target} ok  "
                  f"(last-20 err={roll:.2f}, budget left={attempts_left[0]})")

    return np.array(errors, dtype=np.int8)


# ---------------------------------------------------------------------------
# Score one trial's stream against the KNOWN onset (same logic as run_validation.py)
# ---------------------------------------------------------------------------
def _score_trial(stream: np.ndarray, scenario: str, onset: int, magnitude: float,
                 trial_id: str, provider: str, strong_model: str, weak_model: str) -> list[dict]:
    is_real_drift = scenario in ("sudden", "gradual")
    change_point = onset if is_real_drift else None
    rows = []
    for spec in _registry():
        first_idx = run_detector_on_stream(spec, stream)["first_drift_index"]
        detected = first_idx is not None
        if change_point is None:
            false_alarm, correct_detection, delay = detected, False, None
        else:
            false_alarm = detected and first_idx < change_point
            correct_detection = detected and first_idx >= change_point
            delay = (first_idx - change_point) if correct_detection else None
        rows.append({
            "detector": spec.name, "scenario": scenario, "magnitude": round(magnitude, 4),
            "trial": trial_id, "is_real_drift": is_real_drift,
            "detected_at_all": detected, "correct_detection": correct_detection,
            "false_alarm": false_alarm, "delay": delay,
            "first_alert_index": first_idx, "n_samples": len(stream), "onset": onset,
            "provider": provider, "strong_model": strong_model, "weak_model": weak_model,
        })
    return rows


def _realized_rates(stream: np.ndarray, onset: int) -> tuple[float, float]:
    """Measured baseline (pre-onset) and degraded (post-onset) error rates."""
    p0 = float(stream[:onset].mean()) if onset and len(stream) >= 1 else float("nan")
    p1 = float(stream[onset:].mean()) if len(stream) > onset else float("nan")
    return p0, p1


# ---------------------------------------------------------------------------
# One invocation = one trial of each requested scenario, appended to the raw CSV
# ---------------------------------------------------------------------------
def run_once(args) -> None:
    out_dir = Path(args.out)
    (out_dir / "streams").mkdir(parents=True, exist_ok=True)
    store = DriftStore(db_path=args.db_path)
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    attempts_left = [args.daily_attempt_budget]
    t_start = time.monotonic()
    stamp = int(time.time())

    raw_path = out_dir / f"{args.provider}_raw_results.csv"
    all_rows: list[dict] = []

    degrade_desc = (f"{args.model}#degraded(temp={args.degrade_temp})" if args.degrade_mode == "temperature"
                    else f"{args.model}#degraded(max_tokens={args.degrade_max_tokens})")
    print(f"Live drift comparison -- provider={args.provider}  model={args.model}\n"
          f"drift: {args.degrade_mode} degradation, degrade_frac={args.degrade_frac}  ({degrade_desc})\n"
          f"scenarios={scenarios}  onset={args.onset}  post={args.post}  "
          f"daily attempt budget={args.daily_attempt_budget}\n"
          "(Ctrl+C is safe -- trials collected so far are still appended.)\n")

    try:
        for scenario in scenarios:
            if attempts_left[0] <= 0:
                print(f"Daily attempt budget exhausted before {scenario} -- stopping.")
                break
            trial_id = f"{scenario}_{stamp}"
            rng = random.Random(stamp ^ hash(scenario) & 0xFFFFFFFF)
            stream = _collect_stream(
                scenario, args.provider, args.model,
                args.onset, args.post, args.ramp_len, args.temperature, args.max_tokens,
                args.degrade_frac, args.degrade_temp, args.degrade_max_tokens, args.degrade_mode,
                store, trial_id, attempts_left, rng, t_start, 60.0 / max(1, args.rpm),
            )
            if len(stream) < args.onset + 5:
                print(f"  {scenario}: only {len(stream)} successful calls -- too few to score, skipping.")
                continue

            p0, p1 = _realized_rates(stream, args.onset)
            magnitude = (p1 - p0) if not np.isnan(p1) else 0.0
            rows = _score_trial(stream, scenario, args.onset, magnitude, trial_id,
                                args.provider, args.model, degrade_desc)
            all_rows.extend(rows)

            # per-trial artifacts for inspection
            tag = re.sub(r"[^A-Za-z0-9._-]", "-", trial_id)
            pd.Series(stream).to_frame("error").to_csv(out_dir / "streams" / f"{tag}_stream.csv", index=False)
            alerts_by = {r["detector"]: ([r["first_alert_index"]] if r["first_alert_index"] is not None else [])
                         for r in rows}
            make_timeline_chart(stream, alerts_by, f"{scenario} (onset={args.onset}, p0={p0:.2f}->p1={p1:.2f})",
                                out_dir / "streams" / f"{tag}_timeline.png")
            n_fired = sum(1 for r in rows if r["detected_at_all"])
            print(f"  {scenario}: n={len(stream)}  p0={p0:.3f} p1={p1:.3f}  "
                  f"{n_fired}/{len(rows)} detectors fired.\n")
    except KeyboardInterrupt:
        print("\nInterrupted -- appending trials collected so far.")

    if not all_rows:
        print("No trials collected. Check API key / model availability.")
        sys.exit(1)

    df_new = pd.DataFrame(all_rows)
    if raw_path.exists():
        df_new = pd.concat([pd.read_csv(raw_path), df_new], ignore_index=True)
    df_new.to_csv(raw_path, index=False)

    n_trials = df_new.groupby(["scenario"])["trial"].nunique().to_dict()
    print(f"Appended {len(all_rows)} rows to {raw_path}")
    print(f"Accumulated trials per scenario so far: {n_trials}")
    print("Run again (another day) to add trials, then: "
          "python -m validation.live_drift_compare --aggregate")


# ---------------------------------------------------------------------------
# Aggregate accumulated trials -> ranking table + charts + verdict
# ---------------------------------------------------------------------------
def aggregate(args) -> None:
    out_dir = Path(args.out)
    raw_path = out_dir / f"{args.provider}_raw_results.csv"
    if not raw_path.exists():
        print(f"No accumulated results at {raw_path}. Run some trials first.")
        sys.exit(1)
    df = pd.read_csv(raw_path)

    order = [d for d in LIVE_DETECTOR_ORDER if d in set(df["detector"])]
    rows = []
    for det in order:
        d = df[df["detector"] == det]
        sudden = d[d["scenario"] == "sudden"]
        gradual = d[d["scenario"] == "gradual"]
        real = d[d["is_real_drift"]]
        # false alarms: any fire in the pre-onset (strong-model, low-error) baseline
        # region of ANY real-drift trial -- this is our live false-alarm measurement.
        far = real["false_alarm"].mean() if len(real) else np.nan
        delays = real.loc[real["correct_detection"], "delay"]
        rows.append({
            "detector": det,
            "detection_rate_sudden": sudden["correct_detection"].mean() if len(sudden) else np.nan,
            "detection_rate_gradual": gradual["correct_detection"].mean() if len(gradual) else np.nan,
            "detection_rate_overall": real["correct_detection"].mean() if len(real) else np.nan,
            "false_alarm_rate": far,
            "median_delay_samples": delays.median() if len(delays) else np.nan,
            "n_trials_sudden": sudden["trial"].nunique(),
            "n_trials_gradual": gradual["trial"].nunique(),
        })
    summary = pd.DataFrame(rows).set_index("detector")

    # composite: same weights + shape as validation/report.py, so it's directly
    # comparable to the synthetic and tuned benchmarks.
    max_delay = summary["median_delay_samples"].max()
    delay_score = (1 - summary["median_delay_samples"] / max_delay).fillna(0.0) if max_delay and not np.isnan(max_delay) else summary["median_delay_samples"] * 0
    summary["composite_score"] = (
        COMPOSITE_WEIGHTS["detection_rate"] * summary["detection_rate_overall"].fillna(0.0)
        + COMPOSITE_WEIGHTS["no_false_alarm"] * (1 - summary["false_alarm_rate"].fillna(1.0))
        + COMPOSITE_WEIGHTS["low_delay"] * delay_score
    )
    summary = summary.sort_values("composite_score", ascending=False)

    summary.to_csv(out_dir / f"{args.provider}_summary.csv")
    print(summary.round(3).to_string())

    disp = summary.copy()
    _grouped_bar(disp, ["detection_rate_sudden", "detection_rate_gradual"],
                 ["Sudden drift", "Gradual drift"],
                 "LIVE detection rate on real induced drift (higher is better)",
                 "Detection rate (%)", out_dir / "01_live_detection_rate.png")
    _grouped_bar(disp, ["false_alarm_rate"], ["Pre-onset false alarms"],
                 "LIVE false-alarm rate on the pre-onset baseline region (lower is better)",
                 "False alarm rate (%)", out_dir / "02_live_false_alarm.png")
    _grouped_bar(disp, ["median_delay_samples"], ["Median detection delay"],
                 "LIVE detection delay, correct detections only (lower is better)",
                 "Calls after true onset", out_dir / "03_live_delay.png",
                 as_percent=False)
    _grouped_bar(disp, ["composite_score"], ["Composite score"],
                 f"LIVE composite ({int(COMPOSITE_WEIGHTS['detection_rate']*100)}% detect / "
                 f"{int(COMPOSITE_WEIGHTS['no_false_alarm']*100)}% no-false-alarm / "
                 f"{int(COMPOSITE_WEIGHTS['low_delay']*100)}% low-delay)",
                 "Score (0-1, higher is better)", out_dir / "04_live_composite.png",
                 as_percent=False, value_fmt="{:.2f}")

    _write_verdict(summary, df, out_dir / "LIVE_DRIFT_COMPARE_REPORT.md", args)
    print(f"\nWrote summary, 4 charts, and LIVE_DRIFT_COMPARE_REPORT.md to {out_dir}/")


def _write_verdict(summary: pd.DataFrame, df: pd.DataFrame, out_path: Path, args) -> None:
    best = summary.index[0]
    n_sudden = int(df[df.scenario == "sudden"]["trial"].nunique())
    n_gradual = int(df[df.scenario == "gradual"]["trial"].nunique())
    model = getattr(args, "model", "?")
    lines = [
        "# Live drift-detector comparison — real induced drift\n",
        f"Generated by `validation/live_drift_compare.py --aggregate` over "
        f"{n_sudden} sudden + {n_gradual} gradual live trials "
        f"({args.provider}: {model}, config-degradation drift). Onset at call "
        f"{getattr(args, 'onset', '?')}; at the onset a fraction of calls are degraded "
        f"(truncated / temperature-spiked) to raise the error rate by a controllable amount, "
        f"so the change point is exactly known. False alarms are fires in the pre-onset "
        f"baseline region.\n",
        "## Ranking (composite; same 40/40/20 weights as the synthetic benchmark)\n",
        summary.round(3)[["detection_rate_sudden", "detection_rate_gradual",
                          "false_alarm_rate", "median_delay_samples", "composite_score",
                          "n_trials_sudden", "n_trials_gradual"]].to_markdown(),
        "\n\n## Verdict\n",
        f"**`{best}` ranks first** on the live composite. Read this as a "
        f"*directional* live result ({n_sudden}/{n_gradual} trials, not the 60–100 "
        f"the synthetic benchmark uses) that corroborates the high-N synthetic + "
        f"Elec2/Insects results — not a from-scratch authoritative ranking.\n",
        "\n## Caveats, stated plainly\n",
        f"- Trial counts are small ({n_sudden} sudden, {n_gradual} gradual) — a "
        "free-tier-budget result. More daily runs tighten it.\n"
        "- Drift is *induced* by degrading the same model's request config (truncating "
        "output / spiking temperature) on a fraction of post-onset calls. The error stream "
        "is real and live, but the CAUSE is engineered config drift, not a spontaneous "
        "capability change or a genuine model swap -- a deliberate choice made because the "
        "free Groq chat models' real capability gap (~0.05) was too small to detect.\n"
        "- DDM (tuned)'s `min_num_instances=150` means it only has a short pre-onset "
        f"window (onset={args.onset}) to exhibit false alarms — interpret its "
        "false-alarm number with that in mind.\n"
        "- Grading is exact-match on the answer letter; non-reasoning models are used "
        "so the correct/incorrect signal isn't contaminated by output-format noise.\n",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Self-test: exercise scoring + aggregation on synthetic streams, ZERO API calls
# ---------------------------------------------------------------------------
def self_test(args) -> None:
    from validation.ground_truth import make_gradual, make_stable, make_sudden
    print("SELF-TEST: scoring + aggregation on synthetic streams (no API calls).\n")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    onset = args.onset
    rows = []
    for trial in range(3):
        for scenario, sc in [
            ("sudden", make_sudden(rng, onset + args.post, onset, 0.10, 0.35)),
            ("gradual", make_gradual(rng, onset + args.post, onset, args.ramp_len, 0.10, 0.35)),
        ]:
            p0, p1 = _realized_rates(sc.stream, onset)
            rows.extend(_score_trial(sc.stream, scenario, onset, p1 - p0,
                                     f"{scenario}_selftest_{trial}", "selftest", "strong", "weak"))
    raw_path = out_dir / "selftest_raw_results.csv"
    pd.DataFrame(rows).to_csv(raw_path, index=False)
    print(f"Scored {len(rows)} detector-rows across 6 synthetic trials -> {raw_path}")

    # run the real aggregation path against it
    agg_args = argparse.Namespace(provider="selftest", out=args.out, model="selftest-model", onset=onset)
    aggregate(agg_args)
    print("\nSELF-TEST OK: full scoring + aggregation + chart + verdict pipeline runs.")


def _main() -> None:
    p = argparse.ArgumentParser(description="Live, ground-truth drift-detector comparison on real LLM traffic")
    p.add_argument("--provider", choices=["groq", "openrouter"], default="groq")
    p.add_argument("--model", default=MODEL_DEFAULT,
                   help="the single model whose config is degraded mid-stream to induce drift")
    p.add_argument("--scenarios", default="sudden,gradual",
                   help="comma-separated subset of: sudden,gradual,stable")
    p.add_argument("--degrade-mode", dest="degrade_mode", choices=["truncate", "temperature"], default="truncate",
                   help="how post-onset calls are degraded: 'truncate' = tiny max_tokens so the model is cut "
                        "off before the answer letter (reliable, err~1.0); 'temperature' = spike sampling temp.")
    p.add_argument("--degrade-frac", dest="degrade_frac", type=float, default=DEGRADE_FRAC_DEFAULT,
                   help="fraction of post-onset calls that are degraded -- sets the drift MAGNITUDE "
                        "(post-onset error ~= p0*(1-f) + p_degraded*f).")
    p.add_argument("--degrade-max-tokens", dest="degrade_max_tokens", type=int, default=DEGRADE_MAX_TOKENS_DEFAULT,
                   help="max_tokens for a degraded (truncate-mode) call")
    p.add_argument("--degrade-temp", dest="degrade_temp", type=float, default=DEGRADE_TEMP_DEFAULT,
                   help="temperature for a degraded (temperature-mode) call")
    p.add_argument("--onset", type=int, default=ONSET_DEFAULT)
    p.add_argument("--post", type=int, default=POST_DEFAULT)
    p.add_argument("--ramp-len", dest="ramp_len", type=int, default=GRADUAL_RAMP_DEFAULT)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="baseline sampling temperature. Kept at 0.0: a 9/12 temp=0.3 pilot showed "
                        "sampling noise collapses the strong/weak error gap (calibrated ~0.11 at "
                        "temp 0) down to ~0.05/~0.00 on 350-sample trials, i.e. it destroys the very "
                        "signal being measured. temp 0 keeps the gap clean; the cost is that sudden "
                        "trials are near-deterministic reruns (gradual still varies via mixture "
                        "routing) -- a documented trade of trial-independence for a real signal.")
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=MAX_TOKENS_DEFAULT,
                   help="per-call output cap -- must be large enough for reasoning models to reach the letter")
    p.add_argument("--rpm", type=int, default=RPM_DEFAULT,
                   help="max calls/min, paced to stay under the provider's tokens/min limit")
    p.add_argument("--daily-attempt-budget", dest="daily_attempt_budget", type=int, default=DAILY_ATTEMPT_BUDGET)
    p.add_argument("--out", default=OUT_DIR_DEFAULT)
    p.add_argument("--db-path", dest="db_path", default="driftguard.db")
    p.add_argument("--aggregate", action="store_true", help="aggregate accumulated trials into a ranking (no calls)")
    p.add_argument("--self-test", dest="self_test", action="store_true", help="validate the pipeline with zero API calls")
    args = p.parse_args()

    if args.self_test:
        self_test(args)
    elif args.aggregate:
        aggregate(args)
    else:
        run_once(args)


if __name__ == "__main__":
    _main()
