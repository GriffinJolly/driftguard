# Live real-provider traffic — the detectors held quiet on a stable stream (partial run)

This is the one validation stage `FINAL_VERDICT.md` and `HYBRID_VERDICT.md` had not yet
covered: not synthetic injected drift (`run_validation.py`), not a real-but-generic public
ML benchmark (`live_replay.py` on Elec2/Insects), but DriftGuard's *own* production signal —
the frozen 20-question MMLU accuracy suite called live against a real LLM API, right now,
over the network, with each real correct/incorrect outcome fed into the detectors as it
arrived. This is the thing the finished system actually watches in production, compressed
into a few minutes instead of spread across weeks of hourly runs.

Run via `validation/live_replay_groq.py --registry tuned` (the new `--registry` flag added
to mirror `live_replay.py`; `tuned` = `tuned_registry() + hybrid_registry()`, so all 8
individually-tuned detectors **plus** the Page-Hinkley+DDM OR-hybrid were validated live,
not just Page-Hinkley). Outputs:
`validation/results/live_replay_groq/tuned/openrouter_nvidia-nemotron-3-super-120b-a12b-free_{alert_summary,error_stream}.csv`
and `_timeline.png`.

## 0. What actually ran — stated up front, because it matters

- **Provider/model:** OpenRouter, `nvidia/nemotron-3-super-120b-a12b:free` — a currently-active
  `:free` model picked from the live `openrouter.ai/models` free-tier list on 2026-09-11
  (not hardcoded; free-model availability changes over time).
- **Intended:** 10 rounds × 20 calls = **200** real API calls.
- **Actually completed: 51 real, scored calls.** Rounds 1–3 succeeded (18 + 19 + 14 = 51
  `outcome=ok`); rounds 4–10 were **100% HTTP 429 (rate-limited)** — OpenRouter's free tier
  cut the account off after ~51 successful free-model requests. Those 149 failed calls carry
  `eval_score=None` and are correctly excluded from the error stream (same convention the
  synthetic benchmark uses for failed calls), so they did not enter any detector. A second
  attempt with a different free model (`nemotron-3.5-lightning:free`) returned HTTP 429 on the
  *very first* call, which confirms this is an **account-wide free-tier daily quota
  exhaustion, not a code fault** and not something a longer wait within the session fixes.
- All 200 attempts (successes and 429s alike) were written to `driftguard.db` via
  `DriftStore.write_log`, exactly as a production `runner.py` pass would; the 51-sample error
  stream and the CSV/PNG deliverables were reconstructed from those real logged outcomes.

So this is an **honest but partial** live run: 51 real samples, not 200. Read every
conclusion below through that lens.

## 1. The real error stream: stable and near-perfect

The model answered **50 of 51** MMLU questions correctly — a **1.96% error rate** (one wrong
answer, early in round 1). By construction this is a *stationary, near-zero-error* stream:
there was **no behavioral drift during the observation window** to detect. That is the
important thing to hold onto when reading the detector results — a correct detector's job on
a stream like this is to **stay silent**, not to fire.

## 2. Every detector stayed silent — including the two we trust and the hybrid

| detector | n_alerts | first_alert_index | alerts / 1000 calls |
|---|---|---|---|
| Page-Hinkley (tuned) | 0 | — | 0.0 |
| DDM (tuned) | 0 | — | 0.0 |
| EDDM (tuned) | 0 | — | 0.0 |
| ADWIN (tuned) | 0 | — | 0.0 |
| HDDM_A (tuned) | 0 | — | 0.0 |
| HDDM_W (tuned) | 0 | — | 0.0 |
| RDDM (tuned) | 0 | — | 0.0 |
| ECDD (tuned) | 0 | — | 0.0 |
| Page-Hinkley+DDM (OR hybrid) | 0 | — | 0.0 |

Zero alerts, all nine. On a stable stream with no drift, that is the **correct** result: it
is direct evidence that none of these tuned detectors false-fires on real, live LLM traffic
that isn't drifting. It matches this repo's own clean-stream baseline exactly — on the
**phishing** dataset (1,250 real samples, no engineered drift) every tuned detector also
fired 0 (`validation/results/live_replay/tuned/combined_real_data_summary.csv`).

## 3. Answering the three questions this run was meant to answer — honestly

**Did Page-Hinkley (tuned) and DDM (tuned) behave the way the Elec2/Insects results
predicted?** — Consistent with, but this run could not fully test them. On Elec2/Insects
their value was correct *timing on a real drift event*; here there was **no drift event**, so
the only prediction this run can check is "don't false-fire on a stable stream," which both
satisfied (0 alerts). One caveat that must be stated: **DDM (tuned) uses
`min_num_instances=150`**, and this stream is only 51 samples long — DDM had **not reached
the sample count at which it even begins evaluating drift**, so its silence here is partly
structural, not purely "correctly quiet." Page-Hinkley (tuned) (`min_num_instances=30`) *did*
have runway to fire and correctly did not.

**Did the OR-hybrid add anything over DDM alone?** — It was **identical to DDM (tuned)** here
(both 0), same as on Elec2 and Insects — so, no contradiction with `HYBRID_VERDICT.md`. But
this is a *trivial* identity: with no drift, everything was 0, so neither detector had
anything to contribute and the run does **not** constitute new evidence for or against the
hybrid. `HYBRID_VERDICT.md`'s real-data finding (hybrid ≡ DDM, so not worth two detectors)
still rests on Elec2/Insects, not on this run.

**Did the known synthetic-overfitters (ECDD, HDDM_W) misbehave here too?** — **No — and that
is not exoneration.** On Elec2/Insects, ECDD and HDDM_W false-fired thousands of samples
*before* the real drift (ECDD alerted at Insects sample 1,530, ~13k samples early). Here they
fired 0 — but a 51-sample, near-zero-error stream is far too short and too clean to reproduce
the long-noisy-stream conditions under which they overfit. This run does **not** trigger
their failure mode, and equally does **not** clear them; it simply doesn't stress them the
way the 45k–52k-sample real benchmarks did.

## 4. Verdict

**This live-traffic run reinforces, but does not by itself prove, the existing
recommendation.** The one genuinely new, positive fact it establishes: on real, live LLM
traffic that is *not* drifting, Page-Hinkley (tuned), DDM (tuned), the OR-hybrid, and the
other six tuned detectors all correctly **stayed silent** — no false alarms on a real
production-shaped stream, consistent with the clean-stream (phishing) baseline. That is worth
having: it closes the "but you never actually called a real model live" gap, and nothing in
it argues against Page-Hinkley (tuned) as `live_detector` or DDM (tuned) as the documented
second choice.

What it explicitly does **not** do — and I'm flagging this rather than dressing 51 quiet
samples up as a triumph: it does **not** test drift-*detection* timing (there was no drift),
it does **not** exercise DDM past its 150-sample warm-up floor, and it does **not** reproduce
or rule out ECDD/HDDM_W's real-data overfitting. The load-bearing evidence for the
recommendation remains the Elec2/Insects real-drift results in `FINAL_VERDICT.md` §3; this
run is a corroborating "no false alarms on stable live traffic" data point on top of that,
not a replacement for it.

`configs/detectors.yaml`'s `live_detector` recommendation is **unchanged** — Page-Hinkley
(tuned), DDM (tuned) second — and its comment block has been updated to record this
live-traffic result and its caveats.

## Caveats, stated plainly

- **51 of 200 calls.** OpenRouter's free-tier quota capped the run at ~51 real calls; the
  intended 200-call stream was not collected. A fuller run (200 calls) needs either the free
  daily quota to reset, a paid/credited OpenRouter key, or a different provider (e.g. Groq
  with a `GROQ_API_KEY`) — `live_replay_groq.py --provider groq` already supports this.
- **No drift in the window.** These 51 calls captured a stable model; the run therefore
  measures false-alarm behavior on stable real traffic, not detection latency. To exercise
  detection live, a future run would need an actual capability change (e.g. swapping the
  model mid-stream) or a longer window in which real degradation occurs.
- **DDM below its warm-up floor.** With `min_num_instances=150` and only 51 samples, DDM
  (tuned) never entered its drift-evaluation regime — its 0 alerts here are not a meaningful
  test of DDM.
- **Windows filename fix.** OpenRouter free model ids end in `:free`; on Windows a `:` in a
  path is the NTFS alternate-data-stream separator, which silently routed the first run's
  CSV/PNG into ADS streams instead of real files. `live_replay_groq.py`'s output-tag
  construction was changed to sanitize all filesystem-unsafe characters (not just `/`), so
  the deliverables are now real files. The results themselves are unaffected — they were
  reconstructed from the real per-call outcomes in `driftguard.db`.
