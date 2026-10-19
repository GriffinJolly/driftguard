# DriftGuard

DriftGuard watches a hosted LLM API for silent behavioral drift: changes in how the model behaves that the provider never announced. You can't see a hosted model's weights, training data or update history, so DriftGuard works only from the outside. It sends the same frozen eval questions over and over, logs every call, and runs change-point detection on the results. When something moves, it says what kind of change it was, how sure it is, which signals moved together, what to check first, and which exact questions changed their answers.

It tracks four signals:

| Signal | Measured with | A drop or rise means |
|---|---|---|
| Accuracy | 100 frozen MMLU questions (25 per subject) | Accuracy down: capability shift |
| Format adherence | 6 structured-output (JSON schema) tasks | Format down: format drift |
| Refusal rate | 6 frozen XSTest prompts (5 safe, 1 unsafe) | Refusals of safe prompts up: safety shift |
| Latency | p95 response time of every call | Latency up: infra change |

**Where to read more**

- **Run it on a new machine:** the guide below.
- **How it works, repository layout, validation results and known limitations:** [docs/PROJECT.md](docs/PROJECT.md).
- **Reproducing the paper's results:** [REPRODUCE.md](REPRODUCE.md).
- **Presenting the demo and carrying it to another machine:** [DEMO.md](DEMO.md).

---

## Getting started on a new system

This guide takes a computer that has never seen DriftGuard to a working install: checking the
machine, installing, adding your API key, running the eval suite, detecting drift and opening the
dashboard. Every step is a command you can copy.

There are two routes. Pick one.

| Route | Use it when | You need |
|---|---|---|
| **A. Docker** | You want it to work regardless of the machine's Python | Docker, Git |
| **B. Python (`pip install`)** | You want the quickest install, with no repository to clone | Python **3.10, 3.11 or 3.12** (not 3.13). Git is optional |

DriftGuard does not run on Python 3.13 yet, because its detection library (`frouros`) does not support
it. Docker avoids the problem because Python 3.12 is inside the image.

Commands are written for **Windows PowerShell**. macOS and Linux differences are marked.

---

## 1. Check what the machine has

```powershell
docker --version      # Route A
python --version      # Route B: must say 3.10, 3.11 or 3.12
git --version         # optional: only the git form of the pip install command needs it
```

If a command is "not recognized", that tool is missing. Install it (Section 2 or 3) or choose the
other route.

You also need an **API key** from at least one provider: Groq, NVIDIA or OpenRouter. Create it in the
provider's web console. DriftGuard calls the model's API, so the machine needs internet access for
Sections 5.2, 5.5 and 5.6. The dashboard and the simulated data need no internet and no key.

---

## 2. Route A: Docker

### 2.1 Install Docker

- **Windows / macOS:** install Docker Desktop, start it, and wait until it says it is running.
  On Windows this needs administrator rights and virtualization enabled, and it sometimes needs a restart.
- **Linux:** install Docker Engine from your distribution, and add yourself to the `docker` group.

Check it:

```powershell
docker info
```

If this says it cannot connect to the daemon, Docker is not running yet.

### 2.2 Get the code and build the image

```powershell
git clone https://github.com/GriffinJolly/driftguard.git
cd driftguard
docker build -t driftguard-demo .
```

The first build downloads packages and takes several minutes. Python 3.12 is inside the image.

### 2.3 Make a data folder and add your API key

All databases live in a folder on your machine that is mounted into the container, so nothing is lost
when a container stops.

```powershell
mkdir demo-data -Force | Out-Null
```

Create a file named `.env` in the project folder with your key. Use Notepad, and when saving choose
"All files" so Windows does not name it `.env.txt`:

```
GROQ_API_KEY=paste_your_key_here
NVIDIA_API_KEY=
OPENROUTER_API_KEY=
```

No quotes and no spaces around `=`. Leave a provider's line empty if you do not use it.

### 2.4 Run DriftGuard (Docker)

Every command below starts a short-lived container. `--env-file .env` passes your keys in, and
`-v "${PWD}\demo-data:/data"` is the data folder. (macOS and Linux: use `-v "$PWD/demo-data:/data"`.)

Check the install:

```powershell
docker run --rm driftguard-demo driftguard --version
```

Run the eval suite once against a model (this calls the real API):

```powershell
docker run --rm -it --env-file .env -v "${PWD}\demo-data:/data" driftguard-demo driftguard eval --provider groq --model openai/gpt-oss-120b
```

Detect drift in what was collected:

```powershell
docker run --rm -it -v "${PWD}\demo-data:/data" driftguard-demo driftguard detect --provider groq --model openai/gpt-oss-120b
```

Open the dashboard, then browse to <http://localhost:8502> (not `0.0.0.0`):

```powershell
docker run --rm -it -p 8502:8502 -v "${PWD}\demo-data:/data" driftguard-demo driftguard dashboard --port 8502
```

Open the live demo page, then browse to <http://localhost:8501>:

```powershell
docker run --rm -it -p 8501:8501 -v "${PWD}\demo-data:/data" driftguard-demo
```

Stop any of these with Ctrl+C. The default database is `demo-data\driftguard.db`.

---

## 3. Route B: Python (`pip install`, no clone)

This is the quickest way to show DriftGuard's plug-and-play install: one `pip install`, no repository, no
Docker. The package already contains the dashboard, the simulated demo data and the live demo page.

### 3.1 Install Python 3.12

- **Windows:** download Python 3.12 from python.org. In the installer, tick **Add python.exe to PATH**.
- **macOS:** `brew install python@3.12`
- **Linux (Debian/Ubuntu):** `sudo apt install python3.12 python3.12-venv` (the package names vary by release).

Check it:

```powershell
python --version      # must say 3.10, 3.11 or 3.12
```

If it says 3.13 or newer, install 3.12 as well and use that interpreter in the next step.

### 3.2 Install DriftGuard with pip

```powershell
python -m venv dg
.\dg\Scripts\Activate.ps1
pip install "llm-driftguard[dashboard] @ git+https://github.com/GriffinJolly/driftguard.git"
driftguard --version
```

macOS / Linux: replace the activation line with `source dg/bin/activate`.

The quotes matter. The `[dashboard]` part adds the dashboard and the live page (streamlit, plotly). Use
`[dashboard,proxy]` to also get the reverse proxy.

| Situation | Use this instead |
|---|---|
| The machine has no Git | `pip install "llm-driftguard[dashboard] @ https://github.com/GriffinJolly/driftguard/archive/refs/heads/main.zip"` |
| The package is published on PyPI | `pip install "llm-driftguard[dashboard]"` |
| You want a fixed version | Add `@TAG` after the repository URL in the git form, once you have created a release tag |

These commands install the latest `main`. If PowerShell refuses to activate the environment ("running
scripts is disabled"), run this once in the same window and activate again:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

Your prompt should now start with `(dg)`, and `driftguard --version` should print `driftguard 0.1.0`.

**Developers** (editing the code or running the tests) clone the repository instead:

```powershell
git clone https://github.com/GriffinJolly/driftguard.git
cd driftguard
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,all]"
python -m pytest -q
```

### 3.3 Add your API key

```powershell
driftguard init
```

This writes a `.env` file in the current folder. Open it and fill in the key for the provider you use:

```
GROQ_API_KEY=paste_your_key_here
NVIDIA_API_KEY=
OPENROUTER_API_KEY=
```

No quotes and no spaces around `=`. DriftGuard reads `.env` from the folder you run it in, so run the
later commands from this same folder.

---

## 4. Smoke test (no API key, no internet)

This proves the install works before you spend any API calls. It builds a simulated database and opens
the dashboard on it.

**Route B (Python):**

```powershell
driftguard demo-data --out check_demo.db
driftguard dashboard --db-path check_demo.db
```

**Route A (Docker):**

```powershell
docker run --rm -v "${PWD}\demo-data:/data" driftguard-demo driftguard demo-data --out /data/check_demo.db
docker run --rm -it -p 8502:8502 -v "${PWD}\demo-data:/data" driftguard-demo driftguard dashboard --db-path /data/check_demo.db --port 8502
```

A browser tab opens (Route B at <http://localhost:8501>, Route A at <http://localhost:8502>). You should
see the target `demo/simulated-model` and 3,136 calls. This data is **simulated**. Stop with Ctrl+C.

The output file name must end in `_demo.db`. That is a safety rule, so this command can never overwrite
a real database.

---

## 5. Running DriftGuard

The commands in this section are for Route B. For Route A, use the matching `docker run` command from
Section 2.4.

### 5.1 The commands

```powershell
driftguard --help
```

| Command | What it does |
|---|---|
| `driftguard init` | Writes a `.env` template for your API keys |
| `driftguard eval` | Runs the eval suite against one model, once or on a repeating interval |
| `driftguard collect` | One run per target, then exit. Meant for a scheduler (cron, Task Scheduler) |
| `driftguard detect` | Runs detection and classification on what is stored |
| `driftguard dashboard` | Opens the dashboard |
| `driftguard live` | Opens the live demo page: your prompts, a real API, DriftGuard's detectors |
| `driftguard demo-data` | Builds a **simulated** database to try the dashboard (no key, no internet) |
| `driftguard proxy` | Runs a reverse proxy that logs an application's LLM traffic |

Add `--help` to any command for its options, for example `driftguard eval --help`.

### 5.2 Run the eval suite

```powershell
driftguard eval --provider groq --model openai/gpt-oss-120b
```

One run sends the frozen eval suite (about 112 calls: 100 knowledge questions, 6 format tasks and 6
safety-refusal prompts) to the model and stores every call in `driftguard.db` in the current folder.
Providers are `groq`, `nvidia` and `openrouter`. Use the model name exactly as the provider spells it.
OpenRouter needs a model you choose, because its free models change.

To repeat automatically (this example runs every 3 hours, and stops with Ctrl+C):

```powershell
driftguard eval --provider groq --model openai/gpt-oss-120b --interval-seconds 10800
```

### 5.3 Detect drift

```powershell
driftguard detect --provider groq --model openai/gpt-oss-120b
```

This runs the detectors on the stored calls and prints how many new events it found. The detectors
need history to compare against. One run is not enough to find a change. Collect many runs over time
first.

### 5.4 Open the dashboard

```powershell
driftguard dashboard
```

Browse to <http://localhost:8501>. Choose the model under "Monitored target" and the run window under
"Scope". Use `--db-path` to open a different database and `--port` to change the port.

### 5.5 Collect on a schedule

```powershell
driftguard collect
driftguard collect --target groq:openai/gpt-oss-120b --target nvidia:nvidia/nemotron-3-super-120b-a12b
```

`collect` does one round per target and exits, so it can run from a scheduler. It keeps a `logs/`
folder in the current folder, or in the folder named by the `DRIFTGUARD_HOME` environment variable.
Run it from a fixed folder, and set `DRIFTGUARD_HOME` explicitly if your scheduler starts it somewhere else.

### 5.6 The live demo page

Type your own prompts with expected answers, send them to a real API, and watch the detectors react:

```powershell
driftguard live
```

Browse to <http://localhost:8501>. Section 6.2 walks through it step by step. Route A runs this page by
default (Section 2.4).

### 5.7 Other tools (these need a git checkout)

```powershell
python -m scripts.real_dataset_check                       # reproduces a published GPT-4 drift result
driftguard proxy --port 8000                               # log an application's LLM traffic
python -m pytest -q                                        # runs the tests (needs the dev install)
```

---

## 6. See the demo, and what we have seen so far

Four demonstrations, from the quickest to the most involved. Use them in this order.

| Demo | Needs | Data |
|---|---|---|
| 6.1 Simulated dashboard | nothing | Simulated |
| 6.2 Live page | an API key and internet | Real API, simulated silent update |
| 6.3 Real collected data | a database from your own runs | Real |
| 6.4 Published GPT-4 drift, reproduced | a git checkout and internet | Real (public dataset) |

### 6.1 Demo 1: the simulated dashboard

```powershell
driftguard demo-data --out simulated_demo.db
driftguard dashboard --db-path simulated_demo.db
```

Browse to <http://localhost:8501>. Say clearly that this data is **simulated**: a made-up model,
3,136 calls, with a "model update" injected halfway (at call 1,568) so every detector can be shown on a
case where the answer is known. Walk through the tabs in order: Overview, Shift map, Signals, Shift log &
review, Call explorer.

**What you should see** (the data is generated with a fixed seed, so the numbers repeat):

- **Capability shift, confirmed.** The retrospective detector (PELT with a permutation test) places the
  change at the injected point, with accuracy falling from about 0.93 to about 0.66. The live detector
  (Page-Hinkley) agrees about 90 calls later. The plan is **ACT NOW**, with the evidence level
  "corroborated (live and batch detectors agree)".
- **Format drift, early warning.** Schema adherence falls from about 0.83 to 0.50, flagged as **Watch**.
- **A false alarm.** Latency was never changed, yet the live detector raises a **Watch** on "Slow
  responses" at about call 688. The retrospective detector does not confirm it. This is deliberate to
  keep: it shows why an early warning is a prompt to look, not proof.
- Three incidents in total.

### 6.2 Demo 2: the live page

```powershell
driftguard live
```

Browse to <http://localhost:8501>.

1. In the sidebar choose the provider and model, and paste your API key (hidden; used for this session
   only; never stored).
2. Edit or add watch prompts, each with the answer it should contain. Edit them to show they are yours.
3. **Build a baseline.** Leave "The provider silently changes the model now" **unticked**, set "Rounds
   to run" to 10 and press **Run**. That sends 100 calls and takes several minutes with a real API, so do it
   before you present. The data is kept in `live_demo.db` in the current folder.
4. Show the green "No drift detected" and the flat chart.
5. Tick the box, set Rounds to 3 and press **Run**.
6. Show the red banner, the orange triangles where the detectors fired, the "Every detection" table and
   the "What DriftGuard recommends" box. In "Latest calls", point at `max_tokens` 8 and the empty answers.

The API calls and answers are real. The **silent change is simulated** by the page (a tiny output limit,
or a secretly different model). Say so.

What to expect, and how sure we are: we checked the page end to end with a stand-in for the API, not yet
with a live provider, so run it once with your key before you present. With a 100-call baseline, a severe
change was flagged within about 10 to 20 calls and a moderate drop (97% to 60% correct) in about 30 to 40
calls. With a baseline of only about 50 calls the live detector never fired, which is why the page asks
for at least 100.

### 6.3 Demo 3: real collected data

Open the dashboard on a **copy** of your own database. The dashboard writes the events it detects back
into the file it opens, so never point it at the only copy.

```powershell
Copy-Item driftguard.db real_view.db
driftguard dashboard --db-path real_view.db
```

Choose the model under "Monitored target". Present only what the data shows. Our snapshot (Groq
`openai/gpt-oss-120b`, 564 calls logged between 18 Aug and 12 Oct 2026) said **"No drift detected in
this scope"**, with 0 incidents. Its numbers: call success rate 93.3% (4 errors and 34 rate-limited
calls), answer error rate 14.5% against a 14.0% baseline, p95 latency 5.30 s. The one online alert was
"Quota exhausted", which the dashboard shows as context only, because it is our own rate limit and not a
change in the provider.

### 6.4 Demo 4: a published GPT-4 drift, reproduced

This needs a git checkout of the repository (the script is not part of the pip package) and internet the
first time, to download the public LLMDrift dataset:

```powershell
git clone https://github.com/GriffinJolly/driftguard.git
cd driftguard
pip install -e ".[dev,all]"
python -m scripts.real_dataset_check
```

It re-grades 1,000 real GPT-4 answers from March 2023 and 1,000 from June 2023 on the same task ("is
this number prime?") and gets **84.0% against 51.1%**, matching the numbers in the original paper. Passed
through DriftGuard's own significance test, the drop of 32.9 percentage points gives **p = 0.0024**, so it is
flagged as significant. This is a two-snapshot comparison of a historical dataset, and the batches are
slices of the data file in file order, not a live time series. It shows that the test agrees with a
documented real regression.

### 6.5 Results so far

| What | Data | Result |
|---|---|---|
| Simulated dashboard | Simulated | Capability shift confirmed at the injected change (accuracy 0.93 to 0.66). Format drift flagged as an early warning. One false latency alarm, not confirmed |
| Live page | Real API calls, simulated update | Checked with a stand-in for the API: no drift on a 100-call baseline, then a confirmed capability shift ("ACT NOW") about 30 calls after the change began. Not yet run against a live provider |
| Real collected data | Real | No drift detected in 564 calls |
| Published GPT-4 drift | Real, public dataset | 84.0% to 51.1%, matching the paper. p = 0.0024 |

What this does and does not show:

- Simulated data demonstrates that the detectors find a known change. It is not a result about any real model.
- Early warnings come from the fast live detector and need confirming. The false latency alarm is the example.
- No real provider drift has been observed in our own collected data so far. The only real drift
  reproduced is the published GPT-4 case.
- The detectors need history. A baseline of at least 100 calls is needed before a change.

---

## 7. Where things are stored

| Item | Location |
|---|---|
| Database of every call | `driftguard.db` in the current folder. Set `DRIFTGUARD_DB` to change it, or pass `--db-path` |
| API keys | `.env` in the current folder (or environment variables) |
| Collector logs | `logs\` in the current folder, or in `DRIFTGUARD_HOME` |
| The live page's data | `live_demo.db` in the current folder |
| Docker data | `demo-data\` on the host, mounted at `/data` in the container |

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `python` is not recognized | Python is not installed or not on the PATH. Reinstall and tick "Add python.exe to PATH". |
| `requires a different Python version` | You are on Python 3.13 or newer. Use Python 3.12, or use Route A (Docker). |
| Cannot activate the virtual environment | Run `Set-ExecutionPolicy -Scope Process Bypass`, then activate again. |
| `driftguard` is not recognized | The virtual environment is not active. Activate it (the prompt shows `(dg)` or `(.venv)`). |
| `git` is not recognized during `pip install` | The git form of the install command needs Git. Use the zip form in Section 3.2 instead. |
| `driftguard live` or `dashboard` says it needs extra packages | Install the extra: `pip install "llm-driftguard[dashboard]"` (or repeat the install command in Section 3.2). |
| `Missing GROQ_API_KEY in environment` | The key is not set. Check `.env` is in the folder you run from, with no quotes. |
| HTTP 401 | The key is wrong or expired. |
| HTTP 404 | The model name is wrong for that provider. |
| HTTP 429 | Rate limit. Wait a minute, or run less often. Free tiers are limited. |
| Dashboard shows no data | The database is empty or the path is wrong. Run an eval first, or pass `--db-path`. |
| Port already in use | Use `--port 8503` (Python) or `-p 8602:8502` (Docker) and open that port. |
| `0.0.0.0` does not load in the browser | Use `http://localhost:PORT` instead. |
| `docker: command not found` | Docker is not installed. Use Route B (`pip install`). |
| `error during connect` (Docker) | Docker Desktop is not running. Start it and wait. |
| `permission denied` on `/data` (Linux) | Run `sudo chown -R 1000 demo-data`. |
| `no matching manifest for linux/arm64` | Build the image on the same type of machine you run it on, or add `--platform`. |
| Antivirus or a firewall blocks the dashboard | Allow Python or Docker on private networks, or run on another machine. |

---

## 9. Security

- Keep `.env` out of Git. The repository's `.gitignore` should list it. Never paste a key into an
  issue, a screenshot or a chat.
- On a shared or lab machine, use a throwaway key and revoke it afterwards.
- Databases store your prompts and the model's answers in plain text. Do not send anything private, and
  delete the database when you finish.
- The dashboard writes detected events back into the database it opens. Open a **copy** of any database
  you want to keep untouched.

For a full walkthrough of presenting the demo (including building the image offline and carrying it to
another machine), see `DEMO.md`.

---

## 10. Command cheat sheet

```powershell
# install (Python route, no clone)
python -m venv dg
.\dg\Scripts\Activate.ps1
pip install "llm-driftguard[dashboard] @ git+https://github.com/GriffinJolly/driftguard.git"
driftguard --version

# first use
driftguard init
driftguard eval --provider groq --model openai/gpt-oss-120b
driftguard detect --provider groq --model openai/gpt-oss-120b
driftguard dashboard

# demo
driftguard demo-data --out simulated_demo.db
driftguard dashboard --db-path simulated_demo.db
driftguard live

# docker route
docker build -t driftguard-demo .
docker run --rm -it --env-file .env -v "${PWD}\demo-data:/data" driftguard-demo driftguard eval --provider groq --model openai/gpt-oss-120b
docker run --rm -it -p 8502:8502 -v "${PWD}\demo-data:/data" driftguard-demo driftguard dashboard --port 8502
docker run --rm -it -p 8501:8501 -v "${PWD}\demo-data:/data" driftguard-demo
```
