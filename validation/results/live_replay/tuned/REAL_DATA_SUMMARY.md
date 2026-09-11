# Real-data validation — cross-dataset summary

Produced by `python -m validation.real_data_summary`, combining `{phishing,elec2,insects}_alert_summary.csv` from `python -m validation.live_replay --dataset <name>`. Elec2 and Insects here were run from user-supplied local CSVs (`validation/data/electricity.csv.gz`, `validation/data/insects_abrupt_balanced.csv.gz`) — real, published concept-drift benchmarks, not synthetic data and not this repo's own injected-drift scenarios.

### Phishing (1,250 samples, no known drift)

| detector             |   n_alerts |   alerts_per_1000_samples |   first_alert_index |
|:---------------------|-----------:|--------------------------:|--------------------:|
| Page-Hinkley (tuned) |          0 |                         0 |                 nan |
| DDM (tuned)          |          0 |                         0 |                 nan |
| EDDM (tuned)         |          0 |                         0 |                 nan |
| ADWIN (tuned)        |          0 |                         0 |                 nan |
| HDDM_A (tuned)       |          0 |                         0 |                 nan |
| HDDM_W (tuned)       |          0 |                         0 |                 nan |
| RDDM (tuned)         |          0 |                         0 |                 nan |
| ECDD (tuned)         |          0 |                         0 |                 nan |

### Elec2 (45,312 samples, real recurring drift)

| detector             |   n_alerts |   alerts_per_1000_samples |   first_alert_index |
|:---------------------|-----------:|--------------------------:|--------------------:|
| RDDM (tuned)         |         17 |                      0.38 |                6748 |
| HDDM_W (tuned)       |        254 |                      5.61 |                 193 |
| HDDM_A (tuned)       |        324 |                      7.15 |                 193 |
| EDDM (tuned)         |       5672 |                    125.18 |               20428 |
| ECDD (tuned)         |       9487 |                    209.37 |                 251 |
| Page-Hinkley (tuned) |      33648 |                    742.58 |                5809 |
| DDM (tuned)          |      39874 |                    879.99 |                 250 |
| ADWIN (tuned)        |      44993 |                    992.96 |                 319 |

### Insects-abrupt (52,848 samples, real abrupt drift)

| detector             |   n_alerts |   alerts_per_1000_samples |   first_alert_index |
|:---------------------|-----------:|--------------------------:|--------------------:|
| EDDM (tuned)         |          0 |                      0    |                 nan |
| RDDM (tuned)         |          5 |                      0.09 |               14725 |
| HDDM_A (tuned)       |          8 |                      0.15 |                1629 |
| HDDM_W (tuned)       |        287 |                      5.43 |                 261 |
| ECDD (tuned)         |       7090 |                    134.16 |                1530 |
| Page-Hinkley (tuned) |      31850 |                    602.67 |               14847 |
| DDM (tuned)          |      38181 |                    722.47 |               14667 |
| ADWIN (tuned)        |      52721 |                    997.6  |                 127 |
