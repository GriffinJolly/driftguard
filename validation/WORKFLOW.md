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
| `tuning_sweep_all.py` | Same question, for the other 7 detectors -- is the "Page-Hinkley wins" comparison a fair fight? | No | No | ~3-4 min (150 trials) |
| `run_validation.py --registry tuned` + `report.py --registry tuned` | With ALL 8 detectors tuned, who actually wins the synthetic benchmark? | No | No | ~40s |
| `live_replay.py --dataset phishing` | On **real** (not synthetic) data, do the detectors behave sanely — no excess false alarms? | No (bundled) | No | ~5s |
| `live_replay.py --dataset elec2` / `insects` [`--registry tuned`] | On a **real, documented concept-drift benchmark**, do detectors actually catch real drift? | Only if `validation/data/*.csv.gz` isn't present (see §2.4) | No | Elec2 ~3-4 min, Insects ~4-5 min |
| `real_data_summary.py [--results-dir validation/results/live_replay/tuned]` | Combines phishing/elec2/insects results into one cross-dataset comparison | No | No | <5s |
| `hybrid_validation.py` | Would an OR-hybrid of Page-Hinkley (tuned) + DDM (tuned) beat either alone? | Only if `validation/data/*.csv.gz` isn't present | No | ~8-10 min (synthetic + elec2 + insects) |
| `live_replay_groq.py` | On **real, live LLM API traffic right now**, how do the detectors behave on DriftGuard's actual production signal? | Yes | Yes (Groq or OpenRouter) | ~1-10 min depending on `--rounds` |

Run them in this order the first time — each one is independent, but this order goes from "no setup needed" to "needs an API key," so you can sanity-check the environment before touching real credentials.

**The headline finding from the tuned + real-data comparison** (full writeup:
`validation/results/FINAL_VERDICT.md`): once every detector gets the same tuning pass
Page-Hinkley got, two others (ECDD, HDDM_W) actually score higher on the synthetic
benchmark — but on real data, both of those fire well before the real drift event even
starts, while Page-Hinkley and DDM are the only two whose alarm timing actually tracks
the real degraded regions on both real datasets tested. Page-Hinkley stays the
recommended `live_detector` because of the real-data result, not the synthetic score.

**Follow-up: an OR-hybrid of Page-Hinkley + DDM was tested and not adopted** (full
writeup: `validation/results/HYBRID_VERDICT.md`). It scored a small, real improvement
over either detector alone on the synthetic benchmark, but on both real datasets its
alert count and first-alert timing were identical to DDM (tuned) alone, to the sample —
every Page-Hinkley real-data alert already coincided with a DDM alert, so the hybrid
added no real-world benefit over DDM by itself while running two detectors instead of
one.

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

### 2.2b Tune the other 7 detectors, then re-run the synthetic benchmark fully tuned

```bash
python -m validation.tuning_sweep_all --trials 150 --out validation/results
python -m validation.run_validation --trials 100 --registry tuned --filename raw_results_tuned.csv --out validation/results
python -m validation.report --results-dir validation/results --registry tuned
```
First command sweeps each detector's primary sensitivity parameter the same way
`tuning_sweep.py` did for Page-Hinkley's λ, writes
`all_detector_tuning_sweep.csv` + `all_detector_tuning_recommendations.csv` +
`07_all_detector_tuning.png`. Second and third commands re-run the full synthetic
benchmark using ALL 8 tuned configs (`validation.detectors_registry.tuned_registry()`)
instead of library defaults, writing to `validation/results/tuned/`. **This is the fair
fight** — see `validation/results/FINAL_VERDICT.md` for why the winner here (ECDD, on
this synthetic-only comparison) is NOT what ended up recommended in
`configs/detectors.yaml` once real data was checked too (next section).

### 2.3 Real-data validation (offline, no download)

```bash
python -m validation.live_replay --dataset phishing
```
Runs a real `river` online classifier against the real, bundled Phishing dataset in prequential (predict-then-learn) fashion, then checks all 9 detectors against that real error stream. Output: `validation/results/live_replay/phishing_timeline.png` (rolling real error rate + every detector's alert positions) and `phishing_alert_summary.csv`.

**Expect zero or very few alerts here** — Phishing isn't a labeled drift benchmark, it's just real data, and the model's error rate mostly *improves* as it learns (not the kind of degradation these detectors watch for). Zero alerts is the CORRECT result, not a failure — it's a real-data confirmation of the "no false alarms on non-drift" result from the synthetic benchmark, on genuinely real (not synthetic) data.

### 2.4 Real, documented concept-drift benchmark

```bash
python -m validation.live_replay --dataset elec2
```
or
```bash
python -m validation.live_replay --dataset insects
```

**This repo already includes both datasets locally** — `validation/data/electricity.csv.gz` (real Elec2, 45,312 Australian NSW electricity-market samples, 1996-1998, well-documented recurring concept drift) and `validation/data/insects_abrupt_balanced.csv.gz` (real INSECTS-abrupt_balanced, 52,848 samples, purpose-built abrupt concept-drift benchmark, 6 balanced classes). `live_replay.py` checks for these first and uses them automatically — **no network call, no download step, nothing else to do.** These are the actual files used to produce `validation/results/live_replay/elec2_*` and `insects_*` in this repo; they were obtained because this validation was originally run in a network-restricted sandbox that couldn't reach river's dataset mirrors, so the datasets were downloaded separately and committed here instead — you don't need to repeat that.

If you ever delete `validation/data/` (or want a fresh download), `live_replay.py` falls back to `river.datasets.Elec2()` / `Insects()`, which download from river's own hosted mirrors on first use — that path does need a normal internet connection.

This is the step that answers the question the synthetic benchmark can't: does a detector catch *real*, not injected, concept drift? Both real timelines show a clean answer:
- **Elec2**: the rolling real error rate genuinely swings between ~10% and ~35% over the stream (real, non-stationary electricity-market behavior). Page-Hinkley (both default and tuned) and DDM enter a sustained alarm state closely aligned with the sustained high-error region (roughly sample 15,000–43,000) and stay quiet in the low-error region before it — visually check this yourself against `elec2_timeline.png`.
- **Insects**: there's one dramatic, unmistakable real drift event — the rolling error rate jumps from ~28% to ~93% right around sample ~15,500. Page-Hinkley, DDM, and EDDM all fire within a few hundred samples of that jump and stay alarmed through the elevated-error regime that follows. ADWIN fires almost immediately and essentially never stops (not discriminating). HDDM_A/HDDM_W/RDDM fire rarely (single-digit to low-dozens of alerts across 45-53k samples) — a genuinely different, much more conservative alerting style. See `insects_timeline.png`.

Compare both against `phishing_timeline.png` (zero alerts, zero real drift) — the contrast is the point: the same 9 detectors stay silent on real non-drift data and react on real drift data, on genuinely real (not synthetic, not injected) streams.

Run `python -m validation.real_data_summary` after both to get one combined table/chart (`combined_real_data_summary.csv`, `06_real_data_alert_frequency.png`, `REAL_DATA_SUMMARY.md`) across all three real datasets.

### 2.4b Same real datasets, but with every detector tuned (not just Page-Hinkley)

```bash
python -m validation.live_replay --dataset elec2 --registry tuned
python -m validation.live_replay --dataset insects --registry tuned
python -m validation.live_replay --dataset phishing --registry tuned
python -m validation.real_data_summary --results-dir validation/results/live_replay/tuned
```
Same real data, but using `tuned_registry()` (all 8 detectors from §2.2b) instead of
library defaults. Writes to `validation/results/live_replay/tuned/`. **This is where the
"fair fight" synthetic winner (ECDD) falls apart**: on Insects, ECDD (tuned) fires at
sample 1,530 — nearly 13,000 samples *before* the real drift event (estimated onset
~14,476) — while Page-Hinkley (tuned) and DDM (tuned) fire at 14,847 and 14,667, within
a few hundred samples *after* the real onset. Full breakdown, including why Page-Hinkley
stays the `configs/detectors.yaml` recommendation despite not winning the synthetic
comparison once everything was tuned fairly: `validation/results/FINAL_VERDICT.md`.

### 2.4c OR-hybrid (Page-Hinkley + DDM): would combining them beat either alone?

```bash
python -m validation.hybrid_validation --trials 100 --seed 42 --out validation/results
```
Builds `OrHybridDetector` (alarms when EITHER Page-Hinkley (tuned) or DDM (tuned)
fires), then runs it through both stages: the same synthetic benchmark as §2.2b (writes
`validation/results/hybrid/`) and the same real Elec2/Insects data as §2.4b, but limited
to just Page-Hinkley (tuned), DDM (tuned), and the hybrid so the comparison stays
readable (writes `validation/results/live_replay/hybrid/`). Takes ~8-10 minutes total
because it re-runs the full Elec2 + Insects prequential classifiers; pass
`--skip-real-data` to only run the ~40s synthetic half.

**Result: not adopted.** The hybrid scores a small, genuine improvement over either
detector alone on the synthetic benchmark, but on both real datasets its alert count and
first-alert timing are identical to DDM (tuned) alone, to the sample — every
Page-Hinkley real-data alert already coincided with a DDM alert. Full breakdown:
`validation/results/HYBRID_VERDICT.md`.

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
      "name": "Validation: tuning_sweep_all",
      "type": "debugpy",
      "request": "launch",
      "module": "validation.tuning_sweep_all",
      "args": ["--trials", "150"],
      "cwd": "${workspaceFolder}",
      "console": "integratedTerminal"
    },
    {
      "name": "Validation: run_validation (tuned registry)",
      "type": "debugpy",
      "request": "launch",
      "module": "validation.run_validation",
      "args": ["--trials", "100", "--registry", "tuned", "--filename", "raw_results_tuned.csv"],
      "cwd": "${workspaceFolder}",
      "console": "integratedTerminal"
    },
    {
      "name": "Validation: report (tuned registry)",
      "type": "debugpy",
      "request": "launch",
      "module": "validation.report",
      "args": ["--registry", "tuned"],
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
      "name": "Validation: live_replay (elec2)",
      "type": "debugpy",
      "request": "launch",
      "module": "validation.live_replay",
      "args": ["--dataset", "elec2"],
      "cwd": "${workspaceFolder}",
      "console": "integratedTerminal"
    },
    {
      "name": "Validation: live_replay (insects)",
      "type": "debugpy",
      "request": "launch",
      "module": "validation.live_replay",
      "args": ["--dataset", "insects"],
      "cwd": "${workspaceFolder}",
      "console": "integratedTerminal"
    },
    {
      "name": "Validation: real_data_summary",
      "type": "debugpy",
      "request": "launch",
      "module": "validation.real_data_summary",
      "cwd": "${workspaceFolder}",
      "console": "integratedTerminal"
    },
    {
      "name": "Validation: hybrid_validation",
      "type": "debugpy",
      "request": "launch",
      "module": "validation.hybrid_validation",
      "args": ["--trials", "100", "--seed", "42"],
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
- **`live_replay.py --dataset elec2/insects` hangs or errors with a network/timeout exception** — this only happens if `validation/data/electricity.csv.gz` / `insects_abrupt_balanced.csv.gz` are missing (check `git status` didn't exclude them, e.g. via a stray `.gitignore` rule on `*.gz` or `data/`) and the script fell back to downloading. Check you're not on a restrictive corporate/university network or VPN; try a different network if possible, or re-obtain the CSVs and gzip them back into `validation/data/`.
- **`live_replay.py --dataset elec2/insects` takes several minutes** — expected. `HoeffdingTreeClassifier` runs 45k-53k samples through real prequential predict-then-learn, then 9 detectors each replay the resulting error stream. This isn't a hang; let it finish (~3-5 min).
- **`live_replay_groq.py` fails immediately with a missing-API-key error** — check `.env` exists at the repo root (not inside `validation/`) and `GROQ_API_KEY=...` has no quotes or extra spaces.
- **`live_replay_groq.py` runs but every call has `outcome != ok`** — check your API key is valid and has remaining free-tier quota at [console.groq.com](https://console.groq.com).
- **Charts look different from the ones already in `validation/results/`** — expected if you changed `--trials`, the random seed, or your machine's real-time API responses differ run to run (for `live_replay_groq.py` specifically — a live model's actual answers will vary from what was captured before).
