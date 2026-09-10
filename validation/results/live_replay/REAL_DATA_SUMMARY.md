# Real-data validation — cross-dataset summary

Produced by `python -m validation.real_data_summary`, combining `{phishing,elec2,insects}_alert_summary.csv` from `python -m validation.live_replay --dataset <name>`. Elec2 and Insects here were run from user-supplied local CSVs (`validation/data/electricity.csv.gz`, `validation/data/insects_abrupt_balanced.csv.gz`) — real, published concept-drift benchmarks, not synthetic data and not this repo's own injected-drift scenarios.

### Phishing (1,250 samples, no known drift)

| detector             |   n_alerts |   alerts_per_1000_samples |   first_alert_index |
|:---------------------|-----------:|--------------------------:|--------------------:|
| Page-Hinkley         |          0 |                         0 |                 nan |
| DDM                  |          0 |                         0 |                 nan |
| EDDM                 |          0 |                         0 |                 nan |
| ADWIN                |          0 |                         0 |                 nan |
| HDDM_A               |          0 |                         0 |                 nan |
| HDDM_W               |          0 |                         0 |                 nan |
| RDDM                 |          0 |                         0 |                 nan |
| ECDD                 |          0 |                         0 |                 nan |
| Page-Hinkley (tuned) |          0 |                         0 |                 nan |

### Elec2 (45,312 samples, real recurring drift)

| detector             |   n_alerts |   alerts_per_1000_samples |   first_alert_index |
|:---------------------|-----------:|--------------------------:|--------------------:|
| HDDM_W               |        111 |                      2.45 |                 229 |
| HDDM_A               |        225 |                      4.97 |                 195 |
| RDDM                 |        273 |                      6.02 |                 189 |
| ECDD                 |       4803 |                    106    |                 182 |
| EDDM                 |       9020 |                    199.06 |                 188 |
| Page-Hinkley         |      29369 |                    648.15 |                7262 |
| Page-Hinkley (tuned) |      33648 |                    742.58 |                5809 |
| DDM                  |      41204 |                    909.34 |                 196 |
| ADWIN                |      43777 |                    966.12 |                1535 |

### Insects-abrupt (52,848 samples, real abrupt drift)

| detector             |   n_alerts |   alerts_per_1000_samples |   first_alert_index |
|:---------------------|-----------:|--------------------------:|--------------------:|
| HDDM_A               |          7 |                      0.13 |               14361 |
| RDDM                 |         29 |                      0.55 |                1608 |
| HDDM_W               |         34 |                      0.64 |                1530 |
| ECDD                 |       2444 |                     46.25 |                1507 |
| EDDM                 |      18031 |                    341.19 |               16245 |
| Page-Hinkley         |      31280 |                    591.89 |               14917 |
| Page-Hinkley (tuned) |      31850 |                    602.67 |               14847 |
| DDM                  |      38293 |                    724.59 |               14555 |
| ADWIN                |      52465 |                    992.75 |                 383 |
