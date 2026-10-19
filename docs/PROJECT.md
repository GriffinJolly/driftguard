# DriftGuard

DriftGuard watches a hosted LLM API for silent behavioral drift: changes in how the model behaves that the provider never announced. You can't see a hosted model's weights, training data or update history, so DriftGuard works only from the outside. It sends the same frozen eval questions over and over, logs every call, and runs change-point detection on the results. When something moves, it says what kind of change it was, how sure it is, which signals moved together, what to check first, and which exact questions changed their answers.

It tracks four signals:

| Signal | Measured with | A drop or rise means |
|---|---|---|
| Accuracy | 100 frozen MMLU questions (25 per subject) | Accuracy down: capability shift |
| Format adherence | 6 structured-output (JSON schema) tasks | Format down: format drift |
| Refusal rate | 6 frozen XSTest prompts (5 safe, 1 unsafe) | Refusals of safe prompts up: safety shift |
| Latency | p95 response time of every call | Latency up: infra change |

## How it works

```
your app / eval suite
        |
   ingest: wrapper or proxy logs every call          driftguard/ingest/
        |
   storage: SQLite, driftguard.db                     driftguard/storage/store.py
        |
   eval suite: frozen MMLU, format and XSTest tasks   driftguard/evalsuite/
        |
   detection, per call:                               driftguard/detection/
     Page-Hinkley (live warning)
     PELT + permutation test + Benjamini-Hochberg (confirmation)
        |
   classification: label, evidence tier, incidents    driftguard/classification/
        |
   remediation: urgency, plan, flip-set evidence      driftguard/remediation/
        |
   dashboard                                          driftguard/dashboard/
```

### Ingest

`ingest/transport_hook.py` captures calls at the httpx transport layer, so it works underneath any SDK built on httpx (OpenAI, Groq, raw httpx). It only records requests to known provider hosts. It fails open: if logging breaks, the real API call still goes through untouched. There are two ways to use it. `ingest/wrapper.py` gives you a drop-in client for your own code, and `ingest/proxy.py` runs a local reverse proxy that you point your app at. Both write the same `DriftLogEntry` shape (`ingest/log_schema.py`).

### Storage

`storage/store.py` wraps a SQLite file (default `driftguard.db`) with three tables. `log_rows` has one row per call (prompt, response, outcome, latency, and for eval calls the task id and a 0/1 score). `metric_series` has one row per eval run with that run's accuracy, format adherence, refusal rate and latency percentiles. `changepoint_events` holds what the detectors found and what the classification layer made of it. The database file is in `.gitignore`, so each person has their own.

### Eval suite

`evalsuite/runner.py` runs one pass of the suite against a provider and model: the MMLU questions (`tasks.py`), the format tasks (`format_check.py`) and the XSTest prompts (`refusal_probe.py`). The question sets are frozen in `driftguard/configs/mmlu_frozen.json` and `driftguard/configs/xstest_frozen.json`, so every run asks exactly the same things. Each call gets a 0/1 score: correct, passed the schema, or refused. `providers/clients.py` talks to Groq, OpenRouter and NVIDIA's API catalog through their OpenAI-compatible APIs, with a rate limiter set under each free tier's cap.

Refusal detection is a keyword heuristic ("I can't", "I'm sorry, but I", ...), not a classifier. It can miss or over-count individual refusals. That is acceptable for tracking change over time on the same model, but don't read it as an exact refusal rate.

### Detection

Detection runs on per-call 0/1 streams, because the tuned Page-Hinkley detector only works on per-call input. `detection/signals.py` builds seven streams from the logs, each oriented so that higher means worse. Four of them feed the four labels: answer errors, format errors, over-refusal (safe prompts refused) and slow calls. The other three are shown for context and never classified: under-refusal (unsafe prompts answered, which the flip set covers instead), call failures, and quota exhaustion. Quota exhaustion means we ran out of our own rate limit, which says nothing about the provider.

`detection/pipeline.py` runs two paths over each stream:

1. Page-Hinkley (`changepoint_sequential.py`, wrapping frouros) updates on every call and raises a fast early warning. Its threshold is λ = 12 for a signal that is wrong 5% of the time and is scaled up for noisier signals, using the noise of each signal's first 100 calls; old evidence fades slowly (α = 0.998). It has no p-value; an alarm means the threshold was crossed. During an eval run, `detection/live.py` feeds it every scored call as it arrives, so an alarm is saved and classified before the run ends.
2. PELT (`changepoint_pelt.py`, using ruptures) finds where a stored series changed. A permutation test (`bootstrap.py`) checks whether each candidate change could be chance, and Benjamini-Hochberg (`multiple_testing.py`) corrects for testing several signals at once.

`detection/baseline.py` sets each signal's thresholds from a quiet window using the median and MAD. The settings live in `driftguard/configs/detectors.yaml`. Page-Hinkley was picked over seven other detectors (DDM, EDDM, ADWIN, HDDM_A, HDDM_W, RDDM, ECDD) because of how it behaved on real data, not because of its synthetic score. Scaling its threshold to each signal's noise cut false alarms on stable 10,000-call streams from up to 63% to at most 2% (`validation/ph_noise_scaling.py`). See `validation/WORKFLOW.md` and `validation/results/`.

### Classification

`classification/rules.py` turns each detected change into one of four labels: safety shift, capability shift, format drift or infra change. A change in the healthy direction (accuracy going up, say) gets no label. Each event also gets an evidence tier: early warning (live detector only), unconfirmed (failed its significance test), provisional (passed, but not yet corrected for multiple testing) or confirmed. It also gets a size in noise units: the change divided by how much that metric normally wobbles.

`classification/incidents.py` groups events on the same model that start within 2 hours of each other into one incident. It reports which signals moved together, which moved first and which moved most. When the live detector and a passing PELT result both flag the same metric, that metric is marked corroborated, the strongest evidence tier.

### Remediation and evidence

`remediation/mapper.py` gives each incident an urgency (note, watch, investigate, act, act now) based only on its evidence. It adds checks for each kind of drift and, when several signals moved together, possible explanations to check first. Infrastructure is always ruled out first, since timeouts and cut-off answers can also lower accuracy and format scores. These are suggestions for a person. DriftGuard never acts on its own.

`remediation/evidence.py` builds the flip set: the frozen questions whose result changed across the change point, with one real answer from before and one from after. A question counts as flipped when its rate moves by at least half, for example "correct in 9/10 answers before, 1/10 after". The flip set also flags unsafe prompts that stop being refused, which the refusal-rate signal can't show because it only covers safe prompts. For latency incidents it compares failed calls and median latency instead.

## Repository layout

| Path | What's there |
|---|---|
| `driftguard/` | The package: ingest, storage, providers, evalsuite, detection, classification, remediation, dashboard |
| `driftguard/configs/` | Detector settings, calibration output, frozen MMLU and XSTest sets (shipped inside the package) |
| `tests/` | pytest suite. `tests/fixtures/` has hand-written change-point events and helpers that write fake eval histories |
| `validation/` | Detector benchmarks on synthetic, real (Elec2, Insects) and live data. Start with `validation/WORKFLOW.md` |
| `scripts/` | Tools, run with `python -m scripts.<name>`: `integration_check` (whole chain once), `check_runs`, `token_usage`, `inspect_production_logs`, `export_results`, `real_dataset_check`, `demo_phase3`, and small provider probes. `scripts/archive/` keeps one-off helpers |
| `run_cycle.py` | Thin wrapper around `driftguard collect`, kept so existing scheduled tasks keep working |

## Install

You need Python 3.10, 3.11 or 3.12. frouros doesn't install on Python 3.13 yet.

```bash
pip install llm-driftguard                   # core: eval suite, detection, classification, wrapper
pip install "llm-driftguard[dashboard]"      # adds the dashboard (streamlit, plotly)
pip install "llm-driftguard[proxy]"          # adds the reverse proxy (fastapi, uvicorn)
pip install "llm-driftguard[all]"            # both
```

Quickstart:

```bash
driftguard init                                                   # writes a .env template, add your key
driftguard eval --provider groq --model openai/gpt-oss-120b       # one eval run, saved to ./driftguard.db
driftguard detect --provider groq --model openai/gpt-oss-120b     # detect and classify on what's stored
driftguard dashboard                                              # open the dashboard
```

`driftguard --help` lists every command: `init`, `eval`, `collect`, `proxy`, `detect`, `dashboard`. The database is `driftguard.db` in the current directory unless you set `DRIFTGUARD_DB`.

### From a git checkout (development)

```bash
git clone https://github.com/GriffinJolly/driftguard.git
cd driftguard
python -m venv .venv                     # use a Python 3.10-3.12 interpreter
source .venv/bin/activate                # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev,all,validation]"
```

`pip install -r requirements-lock.txt` installs the exact versions that passed the test suite. `.venv/`, `.venv-*/` and `venv/` are ignored by git.

Put your API keys in a `.env` file in the directory you run driftguard from (`driftguard init` creates the template). It's in `.gitignore`.

```
GROQ_API_KEY=your-groq-key
OPENROUTER_API_KEY=your-openrouter-key
NVIDIA_API_KEY=your-nvidia-key
```

You only need the key for the provider you use.

## Running it

The `driftguard ...` commands work from any directory once the package is installed. Commands that start with `python -m scripts...` or `python -m validation...` need a git checkout and run from its root.

### Collect data

Run the eval suite once:

```bash
driftguard eval --provider groq --model openai/gpt-oss-120b
```

A run is 112 calls (100 MMLU, 6 format, 6 refusal). Run it every 3 hours until you stop it; Groq's free tier allows about 1,000 calls a day:

```bash
driftguard eval --provider groq --model openai/gpt-oss-120b --interval-seconds 10800
```

`driftguard collect` does one round for each target and exits. With no arguments it uses the built-in pair (Groq `openai/gpt-oss-120b` and NVIDIA `nvidia/nemotron-3-super-120b-a12b`); pass `--target provider:model` (repeatable) to choose your own. `python run_cycle.py` still works and does the same thing. Point cron or Task Scheduler at it, every 3 hours. It writes a log to `logs/run_cycle.log` and only counts a run as a success if at least half its calls came back OK.

Both use `driftguard.db` unless you pass `--db-path` or set `DRIFTGUARD_DB`; `collect` keeps its `logs/` folder in the current directory, or in `DRIFTGUARD_HOME` if set.

### Capture your own app's traffic

With the wrapper:

```python
import os
from driftguard.ingest.wrapper import DriftGuardClient

api_key = os.environ["GROQ_API_KEY"]
client = DriftGuardClient(provider="groq")   # base URL https://api.groq.com/openai/v1
resp = client.post(
    "/chat/completions",
    headers={"Authorization": f"Bearer {api_key}"},
    json={"model": "openai/gpt-oss-120b", "messages": [{"role": "user", "content": "hi"}]},
)
```

With the proxy:

```bash
driftguard proxy --port 8000
```

Then point your app at the proxy instead of the provider. The first part of the path picks the provider and the rest is passed on unchanged, so a Groq chat request goes to `http://127.0.0.1:8000/groq/openai/v1/chat/completions` and an OpenRouter one to `http://127.0.0.1:8000/openrouter/api/v1/chat/completions`. Traffic logged this way is tagged `wrapper` or `proxy`, so you can tell it apart from eval-suite calls.

#### Privacy: what is stored

By default the capture layer stores the full prompt and reply of every captured call, which can include personal data from your users. Detection only needs the outcome, timing and score of each call, so you can keep less:

```python
client = DriftGuardClient(provider="groq", text_policy="redact")   # or "hash" or "none"
```

`redact` replaces e-mail addresses, phone numbers, card-like numbers, IP addresses and API keys with placeholders; `hash` keeps only a SHA-256 digest; `none` keeps no text. The proxy takes the same choice as `--text-policy`. Redaction is pattern-based and does not remove names or other free-text details, so use `hash` or `none` when prompts may be sensitive. To apply a retention period, delete old captured traffic with `DriftStore().purge_logs(timedelta(days=30))`; eval-suite calls, which hold only the frozen public questions, are kept.

### Detect and classify

`scripts/integration_check.py` runs the whole chain once: imports, one real eval run, calibration, the detection pipeline over the stored per-call scores, and the classification layer.

```bash
python -m scripts.integration_check
```

To run only detection and classification on what's already stored:

```python
from driftguard.storage.store import DriftStore
from driftguard.detection.pipeline import run_detection

report = run_detection(DriftStore(), "groq", "openai/gpt-oss-120b")
print(report.written, "new events,", report.labelled, "labelled")
```

New events are saved to `changepoint_events` with their label and one-line plan. Re-running adds nothing twice.

### Dashboard

```bash
driftguard dashboard
```

It reads `driftguard.db` by default. Set `DRIFTGUARD_DB` to open a different file. With an empty database it just says there are no logged calls yet. It opens on the most recent run that has at least 150 scored calls.

### Look at the logs

```bash
python -m scripts.inspect_production_logs --limit 20
python -m scripts.inspect_production_logs --integration-path wrapper
python -m scripts.check_runs 6
python -m scripts.token_usage
```

### Validation

To reproduce every table and figure in the paper, follow [`REPRODUCE.md`](../REPRODUCE.md). The detector benchmarks also have their own guide in `validation/WORKFLOW.md`, and each result folder under `validation/results/` has its own `REPORT.md`. In short:

| Check | Result |
|---|---|
| Synthetic streams, 8 tuned detectors | Page-Hinkley: 97.7% / 97.0% detection (sudden / gradual), no false alarms |
| Insects benchmark, batch path | Largest confirmed change at sample 14,350; documented onset 14,476 |
| Live Groq trials (gpt-oss-120b, induced fault) | Change confirmed at the fault in both trials, labelled capability shift |
| Live NVIDIA trials (gpt-oss-20b, 2 sudden + 3 gradual) | Fixed-threshold Page-Hinkley caught 5 / 5; the deployed pipeline confirmed 2 / 5 within 350 calls (`results/live_drift_compare_nvidia/PIPELINE_REPORT.md`) |
| WildChat-1M, gpt-3.5-turbo 0301 → 0613 | Change confirmed about a week after the switch, once repeated prompts are removed |
| LMSYS-Chat-1M rollouts and swaps (30 each, plus 30 controls) | Safety shift labelled in every rollout and swap, none on controls |
| Classification on 13 real scenarios | 13 / 13 with per-call detection |
| Controlled format and latency faults | 4 / 4 correct, including controls |
| Against practical baselines (270 streams with a known answer) | Static threshold and periodic re-evaluation alarm on about half of stable streams; DriftGuard on at most 1 in 120, finding 98–99% of real changes (`results/baseline_comparison/`) |

The quickest benchmarks need no internet and no API key:

```bash
python -m validation.run_validation --trials 100 --out validation/results
python -m validation.report --results-dir validation/results
```

## Tests

```bash
python -m pytest tests/
```

Use `python -m pytest`, not plain `pytest`, so the tests can import the `driftguard` package. The tests make no API calls. Database tests use temporary files and never touch your `driftguard.db`.

## Configuration

| File | What it controls |
|---|---|
| `driftguard/configs/detectors.yaml` | Page-Hinkley settings, PELT settings, significance level, permutation samples, calibration window |
| `driftguard/configs/detectors_calibrated.json` | Thresholds written by `detection/baseline.py` from a quiet window |
| `driftguard/configs/mmlu_frozen.json` | The 100 MMLU questions (the original 20 plus 20 more per subject, same seed) |
| `driftguard/configs/xstest_frozen.json` | The 6 XSTest prompts and whether each should be refused |

## Known limitations

- Refusals are detected by keywords, not a classifier.
- `driftguard/configs/detectors_calibrated.json` was built from only 5 runs. Format adherence and refusal rate never varied in those runs, so their noise units can't be computed yet. Recalibrate once there are at least 20 quiet runs.
- Refusals written with curly apostrophes only count since the curly-apostrophe fix to the refusal detector. Data logged before it undercounts them, so re-score the stored refusal logs or recalibrate before trusting a safety shift around that date.
- Live results so far come from a small number of trials. A live alert means "alarm at call X", not proof that the provider changed the model at X.
- The only real provider change tested (WildChat) was located about a week late. The MMLU set grew from 20 to 100 questions after the live trials reported in the paper were run, so those trials measure accuracy in steps of 0.05.
- With its threshold scaled to signal noise, the live detector warns later on noisy signals, and shifts of 3 to 5 points on a signal that is wrong 15% to 50% of the time are below what either detector catches reliably.
- `requirements.txt` pins no versions; use `requirements-lock.txt` for exact ones.

## Data sources

- MMLU: Hendrycks et al., "Measuring Massive Multitask Language Understanding", ICLR 2021. [cais/mmlu](https://huggingface.co/datasets/cais/mmlu)
- XSTest: Röttger et al., "XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models", NAACL 2024. [walledai/XSTest](https://huggingface.co/datasets/walledai/XSTest)
- Elec2 and Insects concept-drift benchmarks, used in `validation/`.
- LLMDrift (Chen, Zaharia and Zou, 2023), the GPT-4 March vs June prime-number data used by `scripts/real_dataset_check.py` (downloaded on first run).
- WildChat-1M (Zhao et al., 2024) and LMSYS-Chat-1M (Zheng et al., 2024), real conversation logs used in `validation/`. LMSYS-Chat-1M is gated on Hugging Face; accept its terms there before downloading.

## Authors

Griffin Jolly, Siddharth Kumar and Srikumaran S

Guide: Dr. Pradeep K V
