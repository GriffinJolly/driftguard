# Running DriftGuard's detector validation — full workflow (VS Code)

This is the step-by-step guide to running everything in `validation/` yourself,
locally, in VS Code — the synthetic ground-truth benchmark, the Page-Hinkley
tuning sweep, real-data validation against a public dataset, and a live
validation run against a real LLM API. Every script below already exists in
your repo under `validation/`; this file is the "how do I actually run it"
companion to `validation/results/REPORT.md` (the findings) and
`codebase-review-and-workplan.md` (the project-level context).

## What each script answers

| Script | Question it answers | Needs internet? | Needs an API key? | Typical run time |
|---|---|---|---|---|
| `run_validation.py` + `report.py` | On **synthetic, injected** drift with a known answer, how good is each detector? | No | No | ~70s (100 trials) |
| `tuning_sweep.py` | Is Page-Hinkley's default threshold actually right for DriftGuard's signal scale? | No | No | ~15s |
| `live_replay.py --dataset phishing` | On **real** (not synthetic) data, do the detectors behave sanely — no excess false alarms? | No (bundled) | No | ~5s |
| `live_replay.py --dataset elec2` / `insects` | On a **real, documented concept-drift benchmark**, do detectors actually catch real drift? | Yes (first run only) | No | ~1-3 min |
| `live_replay_groq.py` | On **real, live LLM API traffic right now**, how do the detectors behave on DriftGuard's actual production signal? | Yes | Yes (Groq or OpenRouter) | ~1-10 min depending on `--rounds` |

Run them in this order the first time — each one is independent, but this order goes from "no setup needed" to "needs an API key," so you can sanity-check the environment before touching real credentials.

## 1. One-time setup

### 1.1 Prerequisites
- Python 3.11+ (check with `python3 --version`)
- VS Code with the **Python** extension (Microsoft) installed — search "Python" in the Extensions panel (`Ctrl+Shift+X` / `Cmd+Shift+X`) if you don't have it
- Your `driftguard` repo open as the VS Code workspace folder (`File > Open Folder…` → select the `driftguard` folder itself, not a parent folder — `validation/` and `driftguard/` need to be siblings at the workspace root)

### 1.2 Create and activate a virtual environment

Open the VS Code integrated terminal (`` Ctrl+` `` / `` Cmd+` ``) and run, from the repo root:

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
If PowerShell blocks the activation script, run this once first (as your normal user, not admin): `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Your terminal prompt should now start with `(.venv)`. In VS Code, also select this interpreter for the editor itself: `Ctrl+Shift+P` / `Cmd+Shift+P` → "Python: Select Interpreter" → pick the one at `.venv/bin/python` (or `.venv\Scripts\python.exe` on Windows). This makes Pylance/IntelliSense and the "Run" button use the same environment as your terminal.

### 1.3 Install dependencies

```bash
pip install -r requirements.txt
```

This installs everything, including the validation-specific packages (`frouros`, `pandas`, `matplotlib`, `river`) added at the bottom of `requirements.txt`.

### 1.4 Set up your `.env` (only needed for `live_replay_groq.py`)

Create a file named `.env` in the repo root (same folder as `requirements.txt`) — it's already gitignored, so this stays local:
```
GROQ_API_KEY=your_key_here
```
Get a free key at [console.groq.com](https://console.groq.com) if you don't have one — this is the same key `manual_test.py` already expects.

## 2. Running each script

Run every command below **from the repo root** (so `validation` and `driftguard` both resolve as Python packages) — not from inside the `validation/` folder itself.

### 2.1 Synthetic ground-truth benchmark

```bash
python -m validation.run_validation --trials 100 --out validation/results
python -m validation.report --results-dir validation/results
```
- First command: runs the 9-detector-variant × 4-scenario × 3-magnitude synthetic benchmark (9,000 trials), writes `validation/results/raw_results.csv`.
- Second command: summarizes it into `validation/results/summary.csv`, 4 charts (`01_detection_rate.png` … `04_composite_score.png`), and `validation/results/REPORT.md`.
- Bump `--trials` to 200+ for a more statistically stable result before a final writeup (linear in run time — 200 trials ≈ 2.5 min).

### 2.2 Page-Hinkley tuning sweep

```bash
python -m validation.tuning_sweep --trials 100 --out validation/results
```
Produces `validation/results/05_page_hinkley_tuning.png` and `page_hinkley_lambda_sweep.csv` — this is what justified `configs/detectors.yaml`'s `lambda_: 12.0`. If you change the baseline error rate assumption in `run_validation.py` (`BASELINE_ERROR_RATE`), re-run this too — the right λ depends on that scale.

### 2.3 Real-data validation (offline, no download)

```bash
python -m validation.live_replay --dataset phishing
```
Runs a real `river` online classifier against the real, bundled Phishing dataset in prequential (predict-then-learn) fashion, then checks all 9 detectors against that real error stream. Output: `validation/results/live_replay/phishing_timeline.png` (rolling real error rate + every detector's alert positions) and `phishing_alert_summary.csv`.

**Expect zero or very few alerts here** — Phishing isn't a labeled drift benchmark, it's just real data, and the model's error rate mostly *improves* as it learns (not the kind of degradation these detectors watch for). Zero alerts is the CORRECT result, not a failure — it's a real-data confirmation of the "no false alarms on non-drift" result from the synthetic benchmark, on genuinely real (not synthetic) data.

### 2.4 Real, documented concept-drift benchmark (needs internet)

```bash
python -m validation.live_replay --dataset elec2
```
or
```bash
python -m validation.live_replay --dataset insects
```
These download a real dataset on first run (Elec2: ~3MB, Australian electricity market data with well-documented real concept drift; Insects: purpose-built for concept-drift research, ~50k real samples). **This needs a normal internet connection** — if it fails with a network/timeout error, your connection or a firewall is blocking it, not a bug in the script.

> Why this step didn't run automatically for you already: the environment I ran the other scripts in has a restricted outbound network policy that blocks these specific hosts (confirmed by testing — even huggingface.co, which `evalsuite/tasks.py` already depends on, is blocked there). Your own machine almost certainly doesn't have that restriction. If it does, see "Troubleshooting" below.

This is the step that answers the question the synthetic benchmark can't: does a detector catch *real*, not injected, concept drift? Compare the alert timeline here against `phishing_timeline.png` — Elec2/Insects should show real alerts where Phishing showed none.

### 2.5 Live validation against a real LLM API right now

```bash
python -m validation.live_replay_groq --rounds 10
```
This calls Groq for real, 20 calls per round (the frozen MMLU accuracy subset — the same one `driftguard/evalsuite/tasks.py` uses in production), feeds each real outcome into all 9 detectors as it arrives, and prints any alert live as it happens. It also writes every call to `driftguard.db` via the real `DriftStore` — so this run also becomes real accumulated data your eventual dashboard can read.

- `--rounds 10` ≈ 200 real calls, a few minutes (Groq's free tier is ~25 req/min, so beyond the first ~25 calls it naturally paces itself).
- Increase `--rounds` for a longer, more realistic stream. Ctrl+C is safe at any point — it saves whatever was collected so far.
- Try `--provider openrouter` if you'd rather use an `OPENROUTER_API_KEY`.

Output: `validation/results/live_replay_groq/<provider>_<model>_timeline.png` + `_alert_summary.csv` + `_error_stream.csv`.

## 3. Viewing results in VS Code

- **PNG charts**: click any `.png` in the Explorer sidebar — VS Code previews images natively, no extension needed.
- **CSV files**: readable as-is, but install the **Rainbow CSV** extension (search in Extensions panel) for column-aligned, colorized viewing, or **Data Wrangler** (Microsoft) for a spreadsheet-like view with sorting/filtering.
- **REPORT.md**: right-click the file → "Open Preview" (or `Ctrl+Shift+V` / `Cmd+Shift+V`) to render it instead of reading raw markdown.

## 4. Optional: VS Code run configurations

Instead of typing commands in the terminal, add this to `.vscode/launch.json` (create the `.vscode` folder if it doesn't exist) to run/debug any script with F5:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Validation: run_validation",
      "type": "debugpy",
      "request": "launch",
      "module": "validation.run_validation",
      "args": ["--trials", "100"],
      "cwd": "${workspaceFolder}",
      "console": "integratedTerminal"
    },
    {
      "name": "Validation: report",
      "type": "debugpy",
      "request": "launch",
      "module": "validation.report",
      "cwd": "${workspaceFolder}",
      "console": "integratedTerminal"
    },
    {
      "name": "Validation: tuning_sweep",
      "type": "debugpy",
      "request": "launch",
      "module": "validation.tuning_sweep",
      "cwd": "${workspaceFolder}",
      "console": "integratedTerminal"
    },
    {
      "name": "Validation: live_replay (phishing)",
      "type": "debugpy",
      "request": "launch",
      "module": "validation.live_replay",
      "args": ["--dataset", "phishing"],
      "cwd": "${workspaceFolder}",
      "console": "integratedTerminal"
    },
    {
      "name": "Validation: live_replay_groq",
      "type": "debugpy",
      "request": "launch",
      "module": "validation.live_replay_groq",
      "args": ["--rounds", "10"],
      "cwd": "${workspaceFolder}",
      "console": "integratedTerminal"
    }
  ]
}
```
Pick a configuration from the "Run and Debug" panel (`Ctrl+Shift+D` / `Cmd+Shift+D`) dropdown at the top, then press F5. Breakpoints work normally — e.g. set one inside `detectors_registry.run_detector_on_stream` to step through exactly what each detector sees, call by call.

## 5. Troubleshooting

- **`ModuleNotFoundError: No module named 'validation'` or `'driftguard'`** — you're running from inside a subfolder. `cd` back to the repo root (where `requirements.txt` lives) first, or check `"cwd": "${workspaceFolder}"` is set if using `launch.json`.
- **`ImportError` for frouros / river / pandas / matplotlib** — the venv isn't activated, or `pip install -r requirements.txt` didn't run inside it. Check `which python` (macOS/Linux) or `where python` (Windows) points inside `.venv`.
- **`live_replay.py --dataset elec2/insects` hangs or errors with a network/timeout exception** — see the note in 2.4. Check you're not on a restrictive corporate/university network or VPN; try a different network if possible.
- **`live_replay_groq.py` fails immediately with a missing-API-key error** — check `.env` exists at the repo root (not inside `validation/`) and `GROQ_API_KEY=...` has no quotes or extra spaces.
- **`live_replay_groq.py` runs but every call has `outcome != ok`** — check your API key is valid and has remaining free-tier quota at [console.groq.com](https://console.groq.com).
- **Charts look different from the ones already in `validation/results/`** — expected if you changed `--trials`, the random seed, or your machine's real-time API responses differ run to run (for `live_replay_groq.py` specifically — a live model's actual answers will vary from what was captured before).
