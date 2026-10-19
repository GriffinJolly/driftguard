# Ablation and statistical robustness

Rates are k/n with Wilson 95% confidence intervals. False alarm: 120 stable streams (30 LMSYS controls, 90 synthetic). Found: 150 streams with a change (60 LMSYS, 90 synthetic), counted only if no alarm came before the change; noisy = synthetic at 15% and 30% error. Live: the 7 Groq and NVIDIA trials.

## 1. Ablation

| variant          | false_alarm             | found                     | found_noisy             | live   |
|:-----------------|:------------------------|:--------------------------|:------------------------|:-------|
| PH fixed         | 39/120 = 32% [25%, 41%] | 118/150 = 79% [71%, 84%]  | 36/60 = 60% [47%, 71%]  | 7/7    |
| PH scaled        | 1/120 = 1% [0%, 5%]     | 147/150 = 98% [94%, 99%]  | 58/60 = 97% [89%, 99%]  | 5/7    |
| PELT alone       | 0/120 = 0% [0%, 3%]     | 149/150 = 99% [96%, 100%] | 59/60 = 98% [91%, 100%] | 4/7    |
| PELT + perm      | 0/120 = 0% [0%, 3%]     | 149/150 = 99% [96%, 100%] | 59/60 = 98% [91%, 100%] | 4/7    |
| PELT + perm + BH | 0/120 = 0% [0%, 3%]     | 149/150 = 99% [96%, 100%] | 59/60 = 98% [91%, 100%] | 4/7    |
| Full system      | 1/120 = 1% [0%, 5%]     | 146/150 = 97% [93%, 99%]  | 57/60 = 95% [86%, 98%]  | 5/7    |

### Batch path with PELT's penalty lowered to 1 x sigma^2 log n (deployed: 5)

| variant               | false_alarm         | found                    | found_noisy            | live   |
|:----------------------|:--------------------|:-------------------------|:-----------------------|:-------|
| PELT alone (x1)       | 5/120 = 4% [2%, 9%] | 147/150 = 98% [94%, 99%] | 57/60 = 95% [86%, 98%] | 0/0    |
| PELT + perm (x1)      | 5/120 = 4% [2%, 9%] | 147/150 = 98% [94%, 99%] | 57/60 = 95% [86%, 98%] | 0/0    |
| PELT + perm + BH (x1) | 5/120 = 4% [2%, 9%] | 147/150 = 98% [94%, 99%] | 57/60 = 95% [86%, 98%] | 0/0    |

## Baselines with confidence intervals

| method               | false_alarm             | found                     | found_noisy             | live   |
|:---------------------|:------------------------|:--------------------------|:------------------------|:-------|
| Static threshold     | 57/120 = 48% [39%, 56%] | 106/150 = 71% [63%, 77%]  | 25/60 = 42% [30%, 54%]  | 7/7    |
| Periodic re-eval     | 65/120 = 54% [45%, 63%] | 86/150 = 57% [49%, 65%]   | 31/60 = 52% [39%, 64%]  | 7/7    |
| Periodic, corrected  | 3/120 = 2% [1%, 7%]     | 100/150 = 67% [59%, 74%]  | 20/60 = 33% [23%, 46%]  | 7/7    |
| DriftGuard live      | 1/120 = 1% [0%, 5%]     | 147/150 = 98% [94%, 99%]  | 58/60 = 97% [89%, 99%]  | 5/7    |
| DriftGuard confirmed | 0/120 = 0% [0%, 3%]     | 149/150 = 99% [96%, 100%] | 59/60 = 98% [91%, 100%] | 4/7    |

## 2a. Dependence of live per-call outcomes (before the change)

The frozen suite cycles the same questions, so outcomes cluster by question. Design effect = 1 + (m - 1) ICC for m repeats per question; effective n = n / design effect.

| trial          |   calls_before |   questions |   repeats_per_question |   error_rate |   lag1_autocorr |   icc_by_question |   design_effect |   effective_n |
|:---------------|---------------:|------------:|-----------------------:|-------------:|----------------:|------------------:|----------------:|--------------:|
| NVIDIA trial 1 |            175 |          20 |                    8.8 |        0.137 |          -0.015 |             0.744 |            6.77 |          25.8 |
| NVIDIA trial 2 |            175 |          20 |                    8.8 |        0.16  |           0.064 |             0.888 |            7.88 |          22.2 |
| NVIDIA trial 3 |            175 |          20 |                    8.8 |        0.154 |          -0.096 |             0.875 |            7.78 |          22.5 |
| NVIDIA trial 4 |            175 |          20 |                    8.8 |        0.171 |           0     |             0.788 |            7.11 |          24.6 |
| NVIDIA trial 5 |            175 |          20 |                    8.8 |        0.166 |          -0.112 |             0.882 |            7.83 |          22.3 |
| Groq trial 1   |            175 |          20 |                    8.8 |        0.126 |          -0.041 |             0.875 |            7.78 |          22.5 |
| Groq trial 2   |            175 |          20 |                    8.8 |        0.131 |           0.098 |             0.807 |            7.26 |          24.1 |

### Live changes tested at the level of questions

One-sided tests that the error rate rose after call 175. Call-level permutation treats calls as independent (as the pipeline does); the within-question permutation shuffles before/after labels only among calls on the same question; the sign test counts questions whose error rate went up versus down.

| trial          |   error_before |   error_after |   p_call_level |   p_within_question | questions_up_down   |   p_sign_test |
|:---------------|---------------:|--------------:|---------------:|--------------------:|:--------------------|--------------:|
| NVIDIA trial 1 |          0.137 |         0.394 |         0.0005 |              0.0005 | 17 / 0              |        0      |
| NVIDIA trial 2 |          0.16  |         0.446 |         0.0005 |              0.0005 | 16 / 2              |        0.0007 |
| NVIDIA trial 3 |          0.154 |         0.326 |         0.0005 |              0.0005 | 15 / 1              |        0.0003 |
| NVIDIA trial 4 |          0.171 |         0.434 |         0.0005 |              0.0005 | 17 / 2              |        0.0004 |
| NVIDIA trial 5 |          0.166 |         0.297 |         0.0035 |              0.0005 | 15 / 3              |        0.0038 |
| Groq trial 1   |          0.126 |         0.506 |         0.0005 |              0.0005 | 17 / 0              |        0      |
| Groq trial 2   |          0.131 |         0.44  |         0.0005 |              0.0005 | 18 / 0              |        0      |

## 2b. False confirmations on stable, serially dependent streams

60 two-state Markov streams of 2,000 calls at a 15% error rate per row; PELT + permutation (500) + BH.

|   lag1_corr | permutation   | false_confirmation   |
|------------:|:--------------|:---------------------|
|         0   | i.i.d.        | 0/60 = 0% [0%, 6%]   |
|         0   | blocks of 20  | 0/60 = 0% [0%, 6%]   |
|         0   | blocks of 50  | 0/60 = 0% [0%, 6%]   |
|         0.3 | i.i.d.        | 0/60 = 0% [0%, 6%]   |
|         0.3 | blocks of 20  | 0/60 = 0% [0%, 6%]   |
|         0.3 | blocks of 50  | 0/60 = 0% [0%, 6%]   |
|         0.6 | i.i.d.        | 4/60 = 7% [3%, 16%]  |
|         0.6 | blocks of 20  | 4/60 = 7% [3%, 16%]  |
|         0.6 | blocks of 50  | 4/60 = 7% [3%, 16%]  |

## 2c. Real changes re-tested with block permutations

| group         | permutation   | confirmed   |
|:--------------|:--------------|:------------|
| LMSYS gradual | i.i.d.        | 30/30       |
| LMSYS gradual | blocks of 20  | 30/30       |
| LMSYS gradual | blocks of 50  | 30/30       |
| LMSYS sudden  | i.i.d.        | 30/30       |
| LMSYS sudden  | blocks of 20  | 30/30       |
| LMSYS sudden  | blocks of 50  | 30/30       |
| Live Groq     | i.i.d.        | 2/2         |
| Live Groq     | blocks of 20  | 2/2         |
| Live Groq     | blocks of 50  | 2/2         |
| Live NVIDIA   | i.i.d.        | 2/5         |
| Live NVIDIA   | blocks of 20  | 2/5         |
| Live NVIDIA   | blocks of 50  | 2/5         |
