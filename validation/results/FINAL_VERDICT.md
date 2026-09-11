# DriftGuard detector validation — final verdict (all 8 detectors tuned, synthetic + real)

This closes out the validation effort: every detector from the deck's slide-12 comparison
was individually tuned (not just Page-Hinkley), the fully-tuned set was compared on the
same 9,000-trial synthetic ground-truth benchmark used throughout this repo, and then —
separately — compared again on two real, independently-obtained, documented concept-drift
benchmarks (Elec2, Insects-abrupt_balanced). The two comparisons disagree on the winner,
and that disagreement is itself the headline finding.

Full detail lives in three places this file ties together:
- `validation/results/all_detector_tuning_recommendations.csv` + `07_all_detector_tuning.png`
  — the per-detector tuning sweeps.
- `validation/results/tuned/REPORT.md` — the fully-tuned synthetic benchmark.
- `validation/results/live_replay/tuned/REAL_DATA_SUMMARY.md` — the fully-tuned real-data run.

## 1. Every detector got a fair tuning pass, not just Page-Hinkley

`validation/tuning_sweep_all.py` swept each detector's primary sensitivity parameter
(two, for DDM — see below) against the same false-alarm-rate / detection-rate / delay
evaluation, same trial count and seeds, same reliability gate (≤5% false-alarm rate) used
to pick Page-Hinkley's λ=12. All 7 reached the gate:

| detector | tuned parameter(s) | false-alarm rate | detection rate | delay |
|---|---|---|---|---|
| Page-Hinkley | λ=12.0 | ~0% | 100% | fast |
| DDM | drift_level=4.0, min_num_instances=150 | 4% | 97.3% | 62 |
| EDDM | beta=0.56 | 3.3% | 98% | 149 |
| ADWIN | delta=0.1 | 0% | 100% | 111 |
| HDDM_A | alpha_d=0.005 | 4% | 96.7% | 68 |
| HDDM_W | alpha_d=0.15 | 4.7% | 97.3% | 26 |
| RDDM | drift_level=4.5 | 1.3% | 98% | 79 |
| ECDD | average_run_length=400, λ=0.02 | 4.7% | 100% | 32 |

DDM is worth flagging on its own: a drift_level-only sweep hit a hard **24% false-alarm
floor** that never improved no matter how conservative the threshold got (identical 0.24
at drift_level 8, 10, 15, 20, 30). Root-caused by isolating the variable: frouros's
default `min_num_instances=30` lets DDM start evaluating drift after only 30 samples,
before its running mean/std have stabilized on an 800-sample stream — a genuine bug-shaped
finding, not a tuning artifact. Raising it to 150 (alongside drift_level=4.0) fixed it.

## 2. On synthetic data, fully tuned, Page-Hinkley is no longer #1

| detector | composite score |
|---|---|
| ECDD (tuned) | **0.905** |
| HDDM_W (tuned) | 0.902 |
| **Page-Hinkley (tuned)** | 0.846 |
| DDM (tuned) | 0.829 |
| RDDM (tuned) | 0.806 |
| HDDM_A (tuned) | 0.783 |
| ADWIN (tuned) | 0.734 |
| EDDM (tuned) | 0.713 |

Once every detector gets the same tuning effort Page-Hinkley got, ECDD and HDDM_W both
edge it out — both are faster (65–67 samples vs. 110) at a comparably low false-alarm
rate, on this synthetic i.i.d. Bernoulli benchmark. Taken alone, this would mean the
earlier "Page-Hinkley wins" verdict does not survive a truly fair fight.

## 3. On real data, the synthetic winners fail — and Page-Hinkley/DDM are the only ones that generalize

The fully-tuned registry was re-run against the same real Elec2 (45,312 samples) and
Insects-abrupt_balanced (52,848 samples) datasets used earlier, with no synthetic
ground truth to game. The result reverses the synthetic ranking's top two:

- **ECDD (tuned)** — the synthetic #1 — fires on **20.9% of Elec2 samples and 13.4% of
  Insects samples**, including well *before* Insects' one unambiguous real drift event.
  That event's onset is independently estimable from the raw error stream at
  approximately **sample 14,476** (sustained crossing of baseline + 0.30 on a 200-sample
  rolling error rate — see `validation/results/FINAL_VERDICT.md` methodology below).
  ECDD's first alert on Insects comes at sample **1,530 — nearly 13,000 samples before**
  the real drift starts. That is a false trigger, not early detection.
- **HDDM_W (tuned)** — the synthetic #2 — fires at low overall volume (0.5–0.6% of
  samples) but its alerts are **scattered across the entire stream**, not concentrated
  near the real drift region. First alert on Insects: sample 261, over 14,000 samples
  before the real event.
- **Page-Hinkley (tuned)** and **DDM (tuned)** are the only two whose alarm timing
  actually brackets the real degraded regions on both datasets. On Insects specifically —
  the one real dataset with an unambiguous single drift event — their first alerts land
  at samples **14,847 and 14,667 respectively: 191–371 samples AFTER the estimated true
  onset**, and then stay alarmed through the elevated-error period that follows. RDDM
  (tuned) is close behind (first alert at 14,725, 249 samples after onset). Every other
  detector (ADWIN, ECDD, HDDM_A, HDDM_W) fired 12,000+ samples *before* the real event,
  and EDDM (tuned) never fired on Insects at all — a clean miss, despite 84.8% detection
  on the synthetic benchmark.

Onset-relative timing on Insects, summarized:

| detector | first alert index | samples relative to estimated onset (14,476) |
|---|---|---|
| DDM (tuned) | 14,667 | **+191 (correct, fast)** |
| RDDM (tuned) | 14,725 | **+249 (correct, fast)** |
| Page-Hinkley (tuned) | 14,847 | **+371 (correct, fast)** |
| HDDM_A (tuned) | 1,629 | −12,847 (false early trigger) |
| ECDD (tuned) | 1,530 | −12,946 (false early trigger) |
| HDDM_W (tuned) | 261 | −14,215 (false early trigger) |
| ADWIN (tuned) | 127 | −14,349 (false early trigger) |
| EDDM (tuned) | never fired | miss |

## 4. Verdict

**Page-Hinkley (λ=12) remains the recommended `live_detector` in `configs/detectors.yaml`
— not because it won the synthetic benchmark (it didn't, once every detector was tuned
fairly), but because it is one of only two detectors (with DDM) whose real-data alarm
timing was actually trustworthy on both independent real benchmarks tested.** A synthetic
ranking that disagrees with real-data behavior should lose to the real-data result, not
the other way around — that is the entire point of having done both.

This is also a genuine, reportable methodological finding in its own right, independent
of which detector "wins": **tuning against a synthetic i.i.d. benchmark does not
guarantee the tuned detector generalizes to real, non-stationary data** — it overfit two
of the eight detectors here (ECDD, HDDM_W) to the specific noise structure of the
synthetic Bernoulli streams. Page-Hinkley's classical CUSUM-family statistic was the most
robust across both regimes, which is consistent with — not merely coincidental with — its
classical theoretical grounding (Page 1954; Lorden 1971; Moustakides 1986: CUSUM-type
sequential tests are minimax-optimal for detecting a shift in mean under known pre-/post-
change distributions). See the "how to prove it's optimal" discussion earlier in this
project's history for the full theoretical-vs-empirical distinction — this real-data
result is empirical evidence consistent with that theory, not a substitute for it.

DDM (tuned) is the natural second choice — its real-data timing was nearly as clean as
Page-Hinkley's on Insects (191 vs. 371 samples after onset) — and is documented in
`configs/detectors.yaml`'s new `alternative_tuned_detectors` section as the candidate for
a future second/ensemble live detector.

**Update: an OR-hybrid of the two was built and tested, and did not change this
verdict.** See `validation/results/HYBRID_VERDICT.md` for the full result — short
version: an ensemble that alarms whenever EITHER Page-Hinkley (tuned) or DDM (tuned)
fires scored a small, genuine improvement over either alone on the synthetic benchmark
(composite 0.859 vs. 0.850/0.848), but on both real datasets its alert count and
first-alert timing were identical to DDM (tuned) alone, to the sample — every
Page-Hinkley alert on real data already coincided with a DDM alert, so the hybrid added
zero real-world benefit over DDM by itself while running two detectors instead of one.
Page-Hinkley (tuned) remains `live_detector`; DDM (tuned) remains the documented
second choice, not a combined default.

## Caveats, stated plainly

- The Insects "estimated true onset" (sample 14,476) is derived independently from the
  raw error stream (sustained crossing of baseline + 0.30 on a 200-sample rolling error
  rate), not read from an official per-sample ground-truth label shipped with the
  dataset — treat it as a well-supported estimate, not an authoritative timestamp.
- Each detector's tuning swept its single primary sensitivity parameter (two for DDM). A
  full multi-parameter grid search per detector was not run — a further, smaller
  improvement per detector remains possible in principle, on both the synthetic and real
  comparisons.
- Elec2's real signal is noisier and more genuinely non-stationary than Insects' single
  clean event, so an onset-relative timing table like the one above isn't as clean to
  construct for it — the qualitative timeline comparison (`elec2_timeline.png` in both
  `validation/results/live_replay/` and `.../tuned/`) is the right way to inspect it.
