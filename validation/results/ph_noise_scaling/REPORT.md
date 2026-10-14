# Page-Hinkley: threshold scaled to each signal's noise

Previous setting: fixed λ = 12, α = 0.9999 (tuned on a 5% bad-call rate). Current setting (configs/detectors.yaml): λ and δ scaled up by the signal's baseline standard deviation relative to a 5% rate, detector starts after the 100-call baseline, α = 0.998. Both run through pipeline.page_hinkley_alerts. False alarm = any alarm on a stream (or before the change); detected = an alarm at or after the change.

## False alarms on stable 10,000-call streams

|   base_rate |   current (noise-scaled, α=0.998) |   previous (fixed λ=12, α=0.9999) |
|------------:|----------------------------------:|----------------------------------:|
|        0.05 |                              0.02 |                              0.12 |
|        0.15 |                              0.02 |                              0.40 |
|        0.30 |                              0.00 |                              0.62 |
|        0.50 |                              0.00 |                              0.63 |

## All results

| test                |   base_rate |   shift | setting                         |   false_alarm |   detected |   median_delay |
|:--------------------|------------:|--------:|:--------------------------------|--------------:|-----------:|---------------:|
| long stream         |       0.050 |   0.000 | previous (fixed λ=12, α=0.9999) |         0.117 |    nan     |        nan     |
| long stream         |       0.050 |   0.000 | current (noise-scaled, α=0.998) |         0.017 |    nan     |        nan     |
| long stream         |       0.050 |   0.030 | previous (fixed λ=12, α=0.9999) |         0.033 |      1.000 |       1405.500 |
| long stream         |       0.050 |   0.030 | current (noise-scaled, α=0.998) |         0.000 |      0.900 |        962.500 |
| long stream         |       0.050 |   0.050 | previous (fixed λ=12, α=0.9999) |         0.100 |      1.000 |        800.500 |
| long stream         |       0.050 |   0.050 | current (noise-scaled, α=0.998) |         0.000 |      1.000 |        494.000 |
| long stream         |       0.050 |   0.100 | previous (fixed λ=12, α=0.9999) |         0.000 |      1.000 |        367.500 |
| long stream         |       0.050 |   0.100 | current (noise-scaled, α=0.998) |         0.000 |      1.000 |        202.500 |
| long stream         |       0.050 |   0.200 | previous (fixed λ=12, α=0.9999) |         0.033 |      1.000 |        160.500 |
| long stream         |       0.050 |   0.200 | current (noise-scaled, α=0.998) |         0.033 |      1.000 |         84.000 |
| long stream         |       0.150 |   0.000 | previous (fixed λ=12, α=0.9999) |         0.400 |    nan     |        nan     |
| long stream         |       0.150 |   0.000 | current (noise-scaled, α=0.998) |         0.017 |    nan     |        nan     |
| long stream         |       0.150 |   0.030 | previous (fixed λ=12, α=0.9999) |         0.350 |      0.917 |       1558.000 |
| long stream         |       0.150 |   0.030 | current (noise-scaled, α=0.998) |         0.033 |      0.483 |       1671.000 |
| long stream         |       0.150 |   0.050 | previous (fixed λ=12, α=0.9999) |         0.350 |      1.000 |        647.000 |
| long stream         |       0.150 |   0.050 | current (noise-scaled, α=0.998) |         0.000 |      0.950 |        864.000 |
| long stream         |       0.150 |   0.100 | previous (fixed λ=12, α=0.9999) |         0.417 |      1.000 |        396.000 |
| long stream         |       0.150 |   0.100 | current (noise-scaled, α=0.998) |         0.033 |      1.000 |        294.500 |
| long stream         |       0.150 |   0.200 | previous (fixed λ=12, α=0.9999) |         0.400 |      1.000 |        154.000 |
| long stream         |       0.150 |   0.200 | current (noise-scaled, α=0.998) |         0.000 |      1.000 |        136.000 |
| long stream         |       0.300 |   0.000 | previous (fixed λ=12, α=0.9999) |         0.617 |    nan     |        nan     |
| long stream         |       0.300 |   0.000 | current (noise-scaled, α=0.998) |         0.000 |    nan     |        nan     |
| long stream         |       0.300 |   0.030 | previous (fixed λ=12, α=0.9999) |         0.450 |      0.900 |       1465.500 |
| long stream         |       0.300 |   0.030 | current (noise-scaled, α=0.998) |         0.000 |      0.317 |       2133.000 |
| long stream         |       0.300 |   0.050 | previous (fixed λ=12, α=0.9999) |         0.417 |      0.983 |        871.000 |
| long stream         |       0.300 |   0.050 | current (noise-scaled, α=0.998) |         0.017 |      0.717 |       1095.000 |
| long stream         |       0.300 |   0.100 | previous (fixed λ=12, α=0.9999) |         0.367 |      1.000 |        390.500 |
| long stream         |       0.300 |   0.100 | current (noise-scaled, α=0.998) |         0.000 |      1.000 |        484.000 |
| long stream         |       0.300 |   0.200 | previous (fixed λ=12, α=0.9999) |         0.467 |      1.000 |        226.000 |
| long stream         |       0.300 |   0.200 | current (noise-scaled, α=0.998) |         0.000 |      1.000 |        171.000 |
| long stream         |       0.500 |   0.000 | previous (fixed λ=12, α=0.9999) |         0.633 |    nan     |        nan     |
| long stream         |       0.500 |   0.000 | current (noise-scaled, α=0.998) |         0.000 |    nan     |        nan     |
| long stream         |       0.500 |   0.030 | previous (fixed λ=12, α=0.9999) |         0.467 |      0.900 |       1678.000 |
| long stream         |       0.500 |   0.030 | current (noise-scaled, α=0.998) |         0.000 |      0.133 |       1433.500 |
| long stream         |       0.500 |   0.050 | previous (fixed λ=12, α=0.9999) |         0.467 |      0.983 |        784.000 |
| long stream         |       0.500 |   0.050 | current (noise-scaled, α=0.998) |         0.000 |      0.567 |       1489.000 |
| long stream         |       0.500 |   0.100 | previous (fixed λ=12, α=0.9999) |         0.650 |      1.000 |        482.000 |
| long stream         |       0.500 |   0.100 | current (noise-scaled, α=0.998) |         0.017 |      1.000 |        523.500 |
| long stream         |       0.500 |   0.200 | previous (fixed λ=12, α=0.9999) |         0.550 |      1.000 |        212.500 |
| long stream         |       0.500 |   0.200 | current (noise-scaled, α=0.998) |         0.000 |      1.000 |        192.000 |
| synthetic stable    |       0.050 | nan     | previous (fixed λ=12, α=0.9999) |         0.010 |    nan     |        nan     |
| synthetic stable    |       0.050 | nan     | current (noise-scaled, α=0.998) |         0.000 |    nan     |        nan     |
| synthetic sudden    |       0.050 |   0.100 | previous (fixed λ=12, α=0.9999) |         0.000 |      0.930 |        181.000 |
| synthetic sudden    |       0.050 |   0.100 | current (noise-scaled, α=0.998) |         0.000 |      0.790 |        188.000 |
| synthetic sudden    |       0.050 |   0.200 | previous (fixed λ=12, α=0.9999) |         0.000 |      1.000 |         75.000 |
| synthetic sudden    |       0.050 |   0.200 | current (noise-scaled, α=0.998) |         0.000 |      1.000 |         84.000 |
| synthetic sudden    |       0.050 |   0.350 | previous (fixed λ=12, α=0.9999) |         0.000 |      1.000 |         42.000 |
| synthetic sudden    |       0.050 |   0.350 | current (noise-scaled, α=0.998) |         0.000 |      1.000 |         44.500 |
| synthetic gradual   |       0.050 |   0.100 | previous (fixed λ=12, α=0.9999) |         0.000 |      0.950 |        217.000 |
| synthetic gradual   |       0.050 |   0.100 | current (noise-scaled, α=0.998) |         0.000 |      0.880 |        239.000 |
| synthetic gradual   |       0.050 |   0.200 | previous (fixed λ=12, α=0.9999) |         0.000 |      1.000 |        136.500 |
| synthetic gradual   |       0.050 |   0.200 | current (noise-scaled, α=0.998) |         0.000 |      1.000 |        141.500 |
| synthetic gradual   |       0.050 |   0.350 | previous (fixed λ=12, α=0.9999) |         0.000 |      1.000 |        105.000 |
| synthetic gradual   |       0.050 |   0.350 | current (noise-scaled, α=0.998) |         0.000 |      1.000 |        110.000 |
| synthetic transient |       0.050 |   0.100 | previous (fixed λ=12, α=0.9999) |         0.080 |    nan     |        nan     |
| synthetic transient |       0.050 |   0.100 | current (noise-scaled, α=0.998) |         0.000 |    nan     |        nan     |
| synthetic transient |       0.050 |   0.200 | previous (fixed λ=12, α=0.9999) |         0.190 |    nan     |        nan     |
| synthetic transient |       0.050 |   0.200 | current (noise-scaled, α=0.998) |         0.110 |    nan     |        nan     |
| synthetic transient |       0.050 |   0.350 | previous (fixed λ=12, α=0.9999) |         0.670 |    nan     |        nan     |
| synthetic transient |       0.050 |   0.350 | current (noise-scaled, α=0.998) |         0.640 |    nan     |        nan     |
| LMSYS control       |     nan     |   0.000 | previous (fixed λ=12, α=0.9999) |         0.030 |    nan     |        nan     |
| LMSYS sudden        |     nan     | nan     | previous (fixed λ=12, α=0.9999) |         0.090 |      1.000 |        174.000 |
| LMSYS gradual       |     nan     | nan     | previous (fixed λ=12, α=0.9999) |         0.050 |      1.000 |        770.500 |
| LMSYS control       |     nan     |   0.000 | current (noise-scaled, α=0.998) |         0.020 |    nan     |        nan     |
| LMSYS sudden        |     nan     | nan     | current (noise-scaled, α=0.998) |         0.000 |      1.000 |        109.500 |
| LMSYS gradual       |     nan     | nan     | current (noise-scaled, α=0.998) |         0.010 |      1.000 |        701.000 |
