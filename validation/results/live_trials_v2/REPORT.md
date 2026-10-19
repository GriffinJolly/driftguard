# Live trials, round two: longer streams and a silent model swap

NVIDIA's API, 100-question MMLU suite, 175 normal calls then 700 after the change. Deployed settings: noise-scaled Page-Hinkley (alpha 0.998), PELT, permutation test, Benjamini-Hochberg, classification rules. Correct = confirmed capability_shift inside the change window (± 24 calls; the ramp for gradual trials) and no earlier label.

- **long**: Nemotron-3-super-120b, 15% of calls cut to a 16-token budget after call 175.
- **swap**: Nemotron-3-super-120b replaced by Nemotron-3.5-lightning-30b under the same name after call 175, at once or over 300 calls. Genuine replies; on 50 questions the two models scored 90% and 80%.

**0 of 5 correct.**

| trial   | change        | kind    |   calls |   error_before |   error_after |   replacement_share_after | confirmed_at   |   live_alarm_at |   live_alarms_before | verdict   | other_labels              |   live_delay |   confirm_offset |
|:--------|:--------------|:--------|--------:|---------------:|--------------:|--------------------------:|:---------------|----------------:|---------------------:|:----------|:--------------------------|-------------:|-----------------:|
| long 1  | 15% cut short | gradual |     875 |          0.103 |         0.23  |                    nan    |                |             554 |                    0 | missed    | none                      |          379 |              nan |
| long 2  | 15% cut short | sudden  |     875 |          0.086 |         0.219 |                    nan    |                |             336 |                    0 | missed    | none                      |          161 |              nan |
| long 3  | 15% cut short | gradual |     875 |          0.091 |         0.204 |                    nan    |                |             475 |                    0 | missed    | none                      |          300 |              nan |
| swap 1  | silent swap   | sudden  |     875 |          0.08  |         0.107 |                      1    |                |             nan |                    0 | missed    | infra_change on slow_call |          nan |              nan |
| swap 2  | silent swap   | gradual |     875 |          0.091 |         0.109 |                      0.84 |                |             nan |                    0 | missed    | none                      |          nan |              nan |
