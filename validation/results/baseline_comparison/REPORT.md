# DriftGuard against practical baselines

Every method sees the same per-call 0/1 stream with a known change point. False alarm = any alarm before the change, or any alarm on a stream without one. Detected = an alarm at or after the change (a confirmed PELT change within one minimum segment before it also counts, since PELT locates changes on a grid; its delay is the distance to the onset).

- Static threshold: rate over the last 100 calls > first 100 calls + 10 points.
- Periodic re-eval: runs of 100 calls vs the first run, one-sided two-proportion z-test, p < 0.05.
- Periodic, corrected: the same with a Bonferroni correction over the runs.
- DriftGuard live: noise-scaled Page-Hinkley. DriftGuard confirmed: PELT, permutation test, Benjamini-Hochberg.

## LMSYS gradual

| method               |   streams |   false_alarm |   detected |   median_delay |
|:---------------------|----------:|--------------:|-----------:|---------------:|
| Static threshold     |        30 |          0.03 |       0.97 |         732.00 |
| Periodic re-eval     |        30 |          0.40 |       0.60 |         799.00 |
| Periodic, corrected  |        30 |          0.00 |       0.93 |        1099.00 |
| DriftGuard live      |        30 |          0.00 |       1.00 |         730.00 |
| DriftGuard confirmed |        30 |          0.00 |       1.00 |         762.50 |

## LMSYS sudden

| method               |   streams |   false_alarm |   detected |   median_delay |
|:---------------------|----------:|--------------:|-----------:|---------------:|
| Static threshold     |        30 |          0.10 |       0.90 |          74.00 |
| Periodic re-eval     |        30 |          0.27 |       0.73 |          99.00 |
| Periodic, corrected  |        30 |          0.00 |       1.00 |         199.00 |
| DriftGuard live      |        30 |          0.03 |       1.00 |         120.50 |
| DriftGuard confirmed |        30 |          0.00 |       1.00 |           0.00 |

## LMSYS control

| method               |   streams |   false_alarm |   detected |   median_delay |
|:---------------------|----------:|--------------:|-----------:|---------------:|
| Static threshold     |        30 |          0.17 |        nan |            nan |
| Periodic re-eval     |        30 |          0.60 |        nan |            nan |
| Periodic, corrected  |        30 |          0.07 |        nan |            nan |
| DriftGuard live      |        30 |          0.00 |        nan |            nan |
| DriftGuard confirmed |        30 |          0.00 |        nan |            nan |

## Synthetic 5% stable

| method               |   streams |   false_alarm |   detected |   median_delay |
|:---------------------|----------:|--------------:|-----------:|---------------:|
| Static threshold     |        30 |          0.10 |        nan |            nan |
| Periodic re-eval     |        30 |          0.47 |        nan |            nan |
| Periodic, corrected  |        30 |          0.00 |        nan |            nan |
| DriftGuard live      |        30 |          0.00 |        nan |            nan |
| DriftGuard confirmed |        30 |          0.00 |        nan |            nan |

## Synthetic 5% +10 pts

| method               |   streams |   false_alarm |   detected |   median_delay |
|:---------------------|----------:|--------------:|-----------:|---------------:|
| Static threshold     |        30 |          0.17 |       0.83 |         121.00 |
| Periodic re-eval     |        30 |          0.50 |       0.50 |          99.00 |
| Periodic, corrected  |        30 |          0.03 |       0.73 |         349.00 |
| DriftGuard live      |        30 |          0.00 |       1.00 |         191.00 |
| DriftGuard confirmed |        30 |          0.00 |       1.00 |           0.00 |

## Synthetic 15% stable

| method               |   streams |   false_alarm |   detected |   median_delay |
|:---------------------|----------:|--------------:|-----------:|---------------:|
| Static threshold     |        30 |          0.67 |        nan |            nan |
| Periodic re-eval     |        30 |          0.53 |        nan |            nan |
| Periodic, corrected  |        30 |          0.00 |        nan |            nan |
| DriftGuard live      |        30 |          0.03 |        nan |            nan |
| DriftGuard confirmed |        30 |          0.00 |        nan |            nan |

## Synthetic 15% +10 pts

| method               |   streams |   false_alarm |   detected |   median_delay |
|:---------------------|----------:|--------------:|-----------:|---------------:|
| Static threshold     |        30 |          0.53 |       0.47 |         251.00 |
| Periodic re-eval     |        30 |          0.47 |       0.50 |         399.00 |
| Periodic, corrected  |        30 |          0.03 |       0.37 |         599.00 |
| DriftGuard live      |        30 |          0.07 |       1.00 |         334.00 |
| DriftGuard confirmed |        30 |          0.00 |       1.00 |          25.00 |

## Synthetic 30% stable

| method               |   streams |   false_alarm |   detected |   median_delay |
|:---------------------|----------:|--------------:|-----------:|---------------:|
| Static threshold     |        30 |          0.97 |        nan |            nan |
| Periodic re-eval     |        30 |          0.57 |        nan |            nan |
| Periodic, corrected  |        30 |          0.03 |        nan |            nan |
| DriftGuard live      |        30 |          0.00 |        nan |            nan |
| DriftGuard confirmed |        30 |          0.00 |        nan |            nan |

## Synthetic 30% +10 pts

| method               |   streams |   false_alarm |   detected |   median_delay |
|:---------------------|----------:|--------------:|-----------:|---------------:|
| Static threshold     |        30 |          0.63 |       0.37 |         196.00 |
| Periodic re-eval     |        30 |          0.37 |       0.53 |         499.00 |
| Periodic, corrected  |        30 |          0.00 |       0.30 |         999.00 |
| DriftGuard live      |        30 |          0.00 |       1.00 |         448.00 |
| DriftGuard confirmed |        30 |          0.03 |       0.97 |          50.00 |

## Live Groq

| method               |   streams |   false_alarm |   detected |   median_delay |
|:---------------------|----------:|--------------:|-----------:|---------------:|
| Static threshold     |         2 |          0.00 |       1.00 |          23.00 |
| Periodic re-eval     |         2 |          0.00 |       1.00 |          74.00 |
| Periodic, corrected  |         2 |          0.00 |       1.00 |          74.00 |
| DriftGuard live      |         2 |          0.00 |       1.00 |          75.50 |
| DriftGuard confirmed |         2 |          0.00 |       1.00 |           3.00 |

## Live NVIDIA

| method               |   streams |   false_alarm |   detected |   median_delay |
|:---------------------|----------:|--------------:|-----------:|---------------:|
| Static threshold     |         5 |          0.00 |       1.00 |          78.00 |
| Periodic re-eval     |         5 |          0.00 |       1.00 |         124.00 |
| Periodic, corrected  |         5 |          0.00 |       1.00 |         124.00 |
| DriftGuard live      |         5 |          0.00 |       0.60 |         120.00 |
| DriftGuard confirmed |         5 |          0.00 |       0.40 |          63.00 |
