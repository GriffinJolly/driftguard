# Drift in the wild: WildChat-1M, gpt-3.5-turbo-0301 → 0613

Script: `validation/wildchat_version_switch.py` · Data: `allenai/WildChat-1M` (ODC-BY), shards 3–5

## Setup
- 90,225 English GPT-3.5 conversations from real users, 7 Jun – 13 Aug 2023.
- Ground truth: the logged model version switches from `gpt-3.5-turbo-0301` to
  `gpt-3.5-turbo-0613` at **2023-06-28 01:02 UTC**. The detectors never see this field.
- Signal: over-refusal (Safety), meaning a refusal of a prompt that OpenAI moderation did not
  flag, detected with the eval suite's keyword heuristic on the first reply.
- Detectors and attribution: identical to the dashboard (tuned Page-Hinkley, PELT →
  permutation test → Benjamini-Hochberg).

## Results

| View | Conversations | Refusal 14 d before | 14 d after | z | Confirmed changes | Page-Hinkley alerts |
|---|---|---|---|---|---|---|
| Raw passive traffic | 90,225 | 10.4% | 12.6% | +7.1 | 12 | 5 (4 before the switch) |
| Repeats within a day removed | 54,887 | 10.3% | 8.2% | −6.1 | 1 | 0 |

- **Raw traffic is dominated by user behaviour.**
  - The 12 confirmed changes come in up/down pairs a few hours apart (e.g. 9.8% → 31% → 11%).
  - Each pair is a burst of one user resending the same role-play prompt dozens of times.
  - These changes are statistically real, but they are not changes in the model. They even
    reverse the apparent direction of the effect.
- **With repeats removed, refusals fell after the update (10.3% → 8.2%, z = −6.1).**
  - PELT confirms one change, 9.6% → 6.3%, but places it about 7 days after the switch
    (5 Jul), not at it.
  - Page-Hinkley stays silent, as designed, because fewer refusals of safe prompts is not
    a degradation.
  - Removing repeats uses only the prompt text and the calendar day, never the model version.
- **Paired check:** 148 identical prompts were sent both before and after the switch. Their
  refusal rate went from 12.2% to 10.2% (15 refused more often, 19 less often). This is the
  same direction, but too few prompts to be conclusive.
- **Under-refusal could not be tested.** WildChat-1M contains no moderation-flagged prompts;
  the toxic conversations are only in the gated WildChat-1M-Full.

## What this does and does not show
- It **does** show a real, provider-side model update coinciding with a measurable drop in
  over-refusal, recovered from real traffic by DriftGuard's retrospective path.
- It **does not** pinpoint the update: the confirmed change lags it by about a week, and
  real user traffic changes for other reasons too.
- It **does** support the frozen-suite design. On passive traffic, change detection mostly
  finds changes in what users send, so a monitor of live traffic needs repeat suppression,
  or should rely on a fixed probe set as DriftGuard does.
