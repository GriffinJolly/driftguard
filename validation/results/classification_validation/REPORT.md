> **Superseded.** This result came from the per-run detection pipeline that was in `main` before
> PR #12 (script: `validation/classification_validation.py`, commit `ff52a1f` on branch
> `classification_validation`). Detection is now per call; the same 13 scenarios score 13/13 there:
> `validation/results/per_call_classification_validation/REPORT.md`. Kept as the before/after record.

# Classification validation (rules.py / incidents.py / mapper.py, as merged in main)

8 of 13 scenarios classified correctly.

Each scenario: real model outputs -> per-run metric points -> main's calibration, PELT, permutation
test and Benjamini-Hochberg -> rules.py label -> incidents.py -> mapper.py urgency.
Runs: 8 scored calls (Groq) or 25 conversations (LMSYS, WildChat). Calibration: first 20 runs. A label counts if it is within 3 runs of the change.

## By scenario type

| kind       |   correct |   false label |   right label, wrong place |
|:-----------|----------:|--------------:|---------------------------:|
| capability |         2 |             0 |                          0 |
| control    |         3 |             1 |                          0 |
| healthy    |         3 |             0 |                          0 |
| safety     |         0 |             0 |                          4 |

## Reading the result

- rules.py: every label it gave had the right category and direction (capability for the
  accuracy drops, safety_shift for every swap to a more-refusing model, nothing for changes in
  the healthy direction).
- The misplaced and false labels come from the detection step that feeds it (main's
  calibration + changepoint_pelt + bootstrap), not from rules.py: the calibrated PELT penalty
  (sigma^2 log n from the quiet window) is small enough to propose many candidates on a real
  step; the permutation test compares everything before a candidate with everything after,
  so most candidates around one real step pass; and only the smallest-p candidate per metric
  is kept, ties going to the earliest. A single test at alpha = 0.05 also lets a
  same-model control through now and then (BH cannot correct one test).

## Every scenario

| scenario                             | kind       | metric       | expected         |   runs |   change_run |   pelt_candidates |   primary_p | verdict                  | detail                                                    |
|:-------------------------------------|:-----------|:-------------|:-----------------|-------:|-------------:|------------------:|------------:|:-------------------------|:----------------------------------------------------------|
| Groq live trial 1                    | capability | accuracy     | capability_shift |     41 |           21 |                 8 |      0      | correct                  | capability_shift at run 22 (change at 21), confirmed, act |
| Groq live trial 2                    | capability | accuracy     | capability_shift |     43 |           21 |                 1 |      0      | correct                  | capability_shift at run 22 (change at 21), confirmed, act |
| LMSYS vicuna-13b -> llama-2-13b-chat | safety     | refusal_rate | safety_shift     |    100 |           50 |                14 |      0      | right label, wrong place | safety_shift at run 10 (change at 50)                     |
| LMSYS vicuna-13b -> llama-2-7b-chat  | safety     | refusal_rate | safety_shift     |    100 |           50 |                22 |      0      | right label, wrong place | safety_shift at run 24 (change at 50)                     |
| LMSYS dolly-v2-12b -> chatglm-6b     | safety     | refusal_rate | safety_shift     |    100 |           50 |                19 |      0      | right label, wrong place | safety_shift at run 21 (change at 50)                     |
| LMSYS alpaca-13b -> wizardlm-13b     | safety     | refusal_rate | safety_shift     |    100 |           50 |                11 |      0      | right label, wrong place | safety_shift at run 32 (change at 50)                     |
| LMSYS llama-2-13b-chat -> vicuna-13b | healthy    | refusal_rate | no label         |    100 |           50 |                 3 |    nan      | correct                  | no label                                                  |
| LMSYS chatglm-6b -> dolly-v2-12b     | healthy    | refusal_rate | no label         |    100 |           50 |                 4 |    nan      | correct                  | no label                                                  |
| LMSYS vicuna-13b -> vicuna-13b       | control    | refusal_rate | no label         |    100 |           50 |                 2 |      0.0615 | correct                  | no label                                                  |
| LMSYS koala-13b -> koala-13b         | control    | refusal_rate | no label         |    100 |           50 |                 2 |      0.037  | false label              | safety_shift at run 58                                    |
| LMSYS vicuna-13b -> koala-13b        | control    | refusal_rate | no label         |    100 |           50 |                 0 |    nan      | correct                  | no label                                                  |
| LMSYS gpt-4 -> gpt-3.5-turbo         | control    | refusal_rate | no label         |    100 |           50 |                 0 |    nan      | correct                  | no label                                                  |
| WildChat gpt-3.5 0301 -> 0613        | healthy    | refusal_rate | no label         |   2195 |          746 |               176 |      0.115  | correct                  | no label                                                  |

## Sources

- **Groq live trial 1**: gpt-oss-120b, 35% of calls cut to 16 tokens after call 175
- **Groq live trial 2**: gpt-oss-120b, 35% of calls cut to 16 tokens after call 175
- **LMSYS vicuna-13b -> llama-2-13b-chat**: real replies; refusal of safe prompts 0.062 -> 0.231
- **LMSYS vicuna-13b -> llama-2-7b-chat**: real replies; refusal of safe prompts 0.074 -> 0.250
- **LMSYS dolly-v2-12b -> chatglm-6b**: real replies; refusal of safe prompts 0.016 -> 0.115
- **LMSYS alpaca-13b -> wizardlm-13b**: real replies; refusal of safe prompts 0.030 -> 0.093
- **LMSYS llama-2-13b-chat -> vicuna-13b**: real replies; refusal of safe prompts 0.215 -> 0.053
- **LMSYS chatglm-6b -> dolly-v2-12b**: real replies; refusal of safe prompts 0.114 -> 0.016
- **LMSYS vicuna-13b -> vicuna-13b**: real replies; refusal of safe prompts 0.054 -> 0.043
- **LMSYS koala-13b -> koala-13b**: real replies; refusal of safe prompts 0.039 -> 0.058
- **LMSYS vicuna-13b -> koala-13b**: real replies; refusal of safe prompts 0.053 -> 0.046
- **LMSYS gpt-4 -> gpt-3.5-turbo**: real replies; refusal of safe prompts 0.050 -> 0.058
- **WildChat gpt-3.5 0301 -> 0613**: real provider update; refusals of safe prompts fell afterwards

Not covered: format_drift and infra_change (no real data with a known format or latency change).
