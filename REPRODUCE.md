# Reproducing the paper's results

Every result in the paper *DriftGuard: Black-Box Detection of Silent Behavioral Drift in
Hosted Large Language Model APIs* comes from a script in `validation/`. Each script writes
its tables and figures to `validation/results/<name>/`, where the committed outputs used in
the paper can be compared with a fresh run.

The paper's results correspond to the git tag **`v1.0-paper`**.

## Setup

Python 3.10, 3.11 or 3.12 (frouros does not install on 3.13).

```bash
git clone https://github.com/GriffinJolly/driftguard.git
cd driftguard
git checkout v1.0-paper
python -m venv .venv
.venv/bin/pip install -r requirements-lock.txt      # Windows: .venv\Scripts\pip ...
python -m pytest                                     # all tests should pass
```

`requirements-lock.txt` pins the exact versions used. Detector settings are in
`driftguard/configs/detectors.yaml`; the frozen question sets are in
`driftguard/configs/mmlu_frozen.json` and `driftguard/configs/xstest_frozen.json`.

## What needs what

| Kind of result | Network | API key | External data |
|---|---|---|---|
| Synthetic benchmarks, tuning, noise scaling, ablation, baselines, runtime | No | No | No |
| Elec2 and Insects | First run downloads them | No | Public benchmarks |
| LMSYS-Chat-1M trials | Download | No | Accept the dataset licence on Hugging Face, then `hf auth login` |
| WildChat-1M update | Download | No | Public (ODC-BY) |
| Live trials and controlled format fault | Yes | `GROQ_API_KEY` or `NVIDIA_API_KEY` in `.env` | No |

Live trials call a hosted model, so a rerun gives new responses, not the same ones. The
responses used in the paper are kept in the trial databases and the committed streams under
`validation/results/live_drift_compare*/streams/`, and every scoring script can be rerun on them.

## Results, in paper order

| Paper section | Result | Command | Output |
|---|---|---|---|
| IV-A, V-A | Detector tuning, λ sweep | `python -m validation.tuning_sweep_all` and `python -m validation.tuning_sweep` | `results/` |
| V-A | Synthetic benchmark (Table: synthetic results) | `python -m validation.run_validation --trials 100 --registry tuned` then `python -m validation.report --registry tuned` | `results/tuned/` |
| V-B | First alarm on Insects and Elec2 | `python -m validation.live_replay --dataset insects --registry tuned` (and `elec2`, `phishing`), then `python -m validation.real_data_summary` | `results/live_replay/` |
| V-C | Retrospective path on Insects | `python -m validation.retrospective_validation --penalty-scale 5 --dataset insects` | `results/retrospective/` |
| V-D, V-E | Live Groq trials, resampling | `python -m validation.live_drift_compare --provider groq` (live), `python -m validation.live_drift_compare --aggregate` | `results/live_drift_compare*/` |
| V-G | WildChat-1M provider update | `python -m validation.wildchat_version_switch` | `results/wildchat_version_switch/` |
| V-H | LMSYS swaps and staged rollouts | `python -m validation.lmsys_model_swap` and `python -m validation.lmsys_gradual_drift` | `results/lmsys_*` |
| V-I | Classification on 13 scenarios | `python -m validation.per_call_classification_validation` | `results/per_call_classification_validation/` |
| V-I | Controlled format and latency faults | `python -m validation.fault_injection_check --part infra` (offline), `--part format` (live), `--part report` | `results/fault_injection/` |
| V-J | Noise-scaled live threshold | `python -m validation.ph_noise_scaling` | `results/ph_noise_scaling/` |
| V-K | Live trials on NVIDIA | `python -m validation.live_drift_compare --provider nvidia --model openai/gpt-oss-20b` (live), then `python -m validation.nvidia_trials_pipeline` | `results/live_drift_compare_nvidia/` |
| V-L | Practical baselines | `python -m validation.baseline_comparison` | `results/baseline_comparison/` |
| V-M | Ablation and statistical robustness | `python -m validation.ablation_robustness` | `results/ablation_robustness/` |
| V-N | Runtime overhead and scalability | `python -m validation.runtime_benchmark` | `results/runtime_benchmark/` |

Scripts that read the git-ignored trial databases or the LMSYS cache take their location from
`DRIFTGUARD_DATA_ROOT` and `LMSYS_PARQUET` when these are not under the working directory.

## Randomness

Synthetic streams, resampled trials and permutation tests use fixed seeds, so offline results
reproduce exactly on the same library versions. Live model responses do not.
