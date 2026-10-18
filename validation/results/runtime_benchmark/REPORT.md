# Runtime overhead and scalability

Machine: AMD64 Family 25 Model 116 Stepping 1, AuthenticAMD, Python 3.12.10, Windows 11.

## Capture overhead (transport hook, mock provider, no network)

|   calls |   median_plain_ms |   median_hooked_ms |   median_overhead_ms |   p95_overhead_ms |   median_overhead_redact_ms |   fresh_db_bytes_per_call |
|--------:|------------------:|-------------------:|---------------------:|------------------:|----------------------------:|--------------------------:|
|    1000 |             0.421 |             12.776 |               12.356 |            14.853 |                      12.796 |                   3084.99 |

## Storage

|   project_db_calls |   project_db_bytes_per_call |   avg_prompt_plus_reply_chars |
|-------------------:|----------------------------:|------------------------------:|
|               3895 |                      1638.4 |                         771.7 |

At one 112-call suite run every 3 hours (896 calls a day) per target, storage grows by about 1.4 MB a day per target (0.50 GB a year).

## Detection

|   calls_in_history |   live_us_per_call |   batch_seconds |   batch_peak_mb |   change_points_tested |
|-------------------:|-------------------:|----------------:|----------------:|-----------------------:|
|               1000 |              4.153 |           0.295 |           1.847 |                      1 |
|              10000 |              3.475 |           5.962 |           0.212 |                      1 |
|             100000 |              3.572 |         787.511 |           2.106 |                      1 |

## Live monitor

|   history_calls |   startup_replay_seconds |   observe_median_ms |
|----------------:|-------------------------:|--------------------:|
|            5000 |                    0.182 |               0.022 |
