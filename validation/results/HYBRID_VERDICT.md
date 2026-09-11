# The OR-hybrid (Page-Hinkley + DDM) — tested, not adopted

This answers a direct follow-up question from the "isn't DDM better?" discussion in
`FINAL_VERDICT.md`: since Page-Hinkley (tuned) and DDM (tuned) were the only two
detectors whose real-data alarm timing was trustworthy on both Elec2 and Insects, would
an OR-logic hybrid — alarm the moment EITHER fires — beat running either one alone?

Built as `OrHybridDetector` in `validation/detectors_registry.py` (wraps both tuned
detectors, ORs their live `status['drift']` every sample — it does not latch "drift"
permanently true after one alarm, so its real-data alert-frequency numbers stay
comparable to every other detector's). Run through both stages of this repo's
methodology by `validation/hybrid_validation.py`:

- Synthetic: same protocol, trial count (100), and seeds as
  `validation/results/tuned/REPORT.md` (the existing 8-detector tuned comparison) — see
  `validation/results/hybrid/REPORT.md` and `raw_results_hybrid.csv`.
- Real data: Page-Hinkley (tuned), DDM (tuned), and the hybrid, against the same local
  Elec2/Insects CSVs used throughout this repo — see
  `validation/results/live_replay/hybrid/`.

## 1. Synthetic result: a small, real improvement over either detector alone

| detector | composite score | detection rate (overall) | false-alarm rate (overall) | median delay |
|---|---|---|---|---|
| HDDM_W (tuned) | 0.896 | 0.980 | 0.043 | 61.5 |
| ECDD (tuned) | 0.896 | 0.973 | 0.027 | 65.0 |
| **Page-Hinkley+DDM (OR hybrid)** | **0.859** | 0.978 | 0.045 | 90.0 |
| Page-Hinkley (tuned) | 0.850 | 0.983 | 0.002 | 112.0 |
| DDM (tuned) | 0.848 | 0.962 | 0.045 | 93.0 |

The hybrid genuinely beats both of its own constituents on the synthetic benchmark
(0.859 vs. 0.850 and 0.848) — it inherits close to Page-Hinkley's detection rate and a
median delay between the two. But it does **not** close the gap to HDDM_W/ECDD, the two
detectors already shown (in `FINAL_VERDICT.md`, section 3) to overfit the synthetic
benchmark and fail on real data. And the improvement over Page-Hinkley alone is small —
0.009 composite points — for the cost of running two detectors on every sample instead
of one.

## 2. Real-data result: on both Elec2 and Insects, the hybrid is *identical* to DDM alone

| detector | dataset | n_alerts | first_alert_index |
|---|---|---|---|
| DDM (tuned) | elec2 | 39,874 | 250 |
| **Page-Hinkley+DDM (OR hybrid)** | **elec2** | **39,874** | **250** |
| Page-Hinkley (tuned) | elec2 | 33,648 | 5,809 |
| DDM (tuned) | insects | 38,181 | 14,667 |
| **Page-Hinkley+DDM (OR hybrid)** | **insects** | **38,181** | **14,667** |
| Page-Hinkley (tuned) | insects | 31,850 | 14,847 |

On both real datasets, the hybrid's alert count and first-alert index match DDM (tuned)
exactly, to the sample — every one of Page-Hinkley's real-data alerts already lands on a
sample where DDM also alerts, so ORing the two adds **zero** new alerts on real data.
DDM's alarm timing simply dominates Page-Hinkley's here; the hybrid doesn't blend the
two, it just becomes DDM. On the one dataset with an independently-estimable true drift
onset (Insects, ≈ sample 14,476), the hybrid catches it exactly as fast as DDM alone
(191 samples after onset) — no faster, because there was nothing left for Page-Hinkley
to contribute.

That also means the hybrid inherits DDM's real-data alert *volume* in full: 88% of Elec2
samples and 72% of Insects samples flagged, the same as DDM alone (see
`validation/results/live_replay/tuned/combined_real_data_summary.csv`). Ensembling did
not dilute that — it's just DDM's behavior with an extra detector running for no
real-data benefit.

## Verdict

**Not adopted.** The hybrid is a measurable, small win on the synthetic benchmark and a
complete no-op on real data — DDM's alerts already contain every alert Page-Hinkley
would have added on both real datasets tested, so it buys real-world robustness from
neither detector that the other didn't already have alone, while doubling the number of
detector instances `changepoint_sequential.py` would need to maintain per monitored
metric. `configs/detectors.yaml` keeps Page-Hinkley (tuned) as `live_detector` for the
same real-data reasoning as `FINAL_VERDICT.md` section 4; DDM (tuned) remains the
documented second choice, not "Page-Hinkley OR DDM" as a combined default.

This is also, honestly, evidence against a suspicion raised earlier in this project
(the "isn't DDM better?" question) rather than for the hybrid: it shows DDM's real-data
alerts really do occur earlier and more often than Page-Hinkley's, consistently enough
that an OR-combination is indistinguishable from DDM by itself — which is exactly the
kind of thing "just look more carefully at DDM vs. Page-Hinkley" reasoning alone
couldn't have told us, and running the actual numbers did.

## Caveats, stated plainly

- Only the OR combination was built and tested here (per the choice made when this was
  scoped) — AND logic and the two-tier ("fast alert, second detector corroborates")
  design discussed earlier were not implemented or tested. Nothing here rules those out;
  they simply weren't measured.
- DDM's real-data alert volume (72-88% of samples) is high enough on its own to be a
  practical concern for a live monitor regardless of the hybrid question — a human
  reviewing "DDM says drift" on 4 out of 5 samples is not actually being alerted to
  anything specific. That's a pre-existing property of DDM (tuned), not something this
  hybrid experiment introduced.
- As with the rest of this repo's real-data results, Insects' "true onset" (sample
  14,476) is an independent estimate from the raw error stream, not an official labeled
  ground truth.
