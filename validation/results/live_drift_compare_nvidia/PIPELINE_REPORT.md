# Live NVIDIA trials through the deployed pipeline

gpt-oss-20b on NVIDIA's API, 20-question MMLU suite, 350 scored calls per trial, fault from call 176. Deployed settings: noise-scaled Page-Hinkley (alpha 0.998), PELT, permutation test, Benjamini-Hochberg, classification rules. Correct = confirmed capability_shift inside the change window (fault start, or the 100-call ramp for gradual, ± 24 calls) and no earlier label.

**1 of 5 correct.**

|   trial | kind    |   calls |   error_before |   error_after |   confirmed_at |   live_alarm_at |   live_alarms_before_fault | verdict            | other_labels              |   live_delay |   confirm_offset |
|--------:|:--------|--------:|---------------:|--------------:|---------------:|----------------:|---------------------------:|:-------------------|:--------------------------|-------------:|-----------------:|
|       1 | gradual |     350 |          0.137 |         0.394 |            238 |             305 |                          0 | correct            | infra_change on slow_call |          130 |               63 |
|       2 | sudden  |     350 |          0.16  |         0.446 |            238 |             295 |                          0 | label, wrong place | infra_change on slow_call |          120 |               63 |
|       3 | gradual |     350 |          0.154 |         0.326 |            nan |             nan |                          0 | missed             | infra_change on slow_call |          nan |              nan |
|       4 | sudden  |     350 |          0.171 |         0.434 |            nan |             292 |                          0 | missed             | infra_change on slow_call |          117 |              nan |
|       5 | gradual |     350 |          0.166 |         0.297 |            nan |             nan |                          0 | missed             | infra_change on slow_call |          nan |              nan |

## Notes

- The fixed-threshold Page-Hinkley used in the detector benchmark (`nvidia_raw_results.csv`) caught all five faults, 38-169 calls after they began, with no earlier alarm. The deployed detector is scaled to signal noise (results/ph_noise_scaling/) and is less sensitive on short, noisy streams.
- PELT confirmed nothing in trial 4 (sudden, +26 points) because the penalty 5 sigma^2 log n, tuned on long recorded histories, is too strict at 350 calls; with a multiplier of 3 or less it places the change at call 187. The deployed setting is reported unchanged.
- In trial 2 only about a fifth of the first 50 post-fault calls were cut short (random draw at 35%), so the error rate reached its new level around call 225 and PELT placed the change at 238.
- Every trial got an infra_change early warning on slow calls: median latency roughly doubled (to 14-19 s) while the five trials ran in parallel against one free-tier endpoint. It was never confirmed by PELT.
