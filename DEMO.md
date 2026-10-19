# Running the DriftGuard demo on a new machine

This guide gets the demo running on a computer that has never seen the project. It assumes nothing
about the machine's Python. The recommended way is **Docker**, which carries its own Python 3.12.

## What you are demonstrating

| Demo | What it shows | Needs internet? | Real or simulated |
|---|---|---|---|
| **A. Live page** (port 8501) | You type prompts and expected answers, they go to a real LLM API, and DriftGuard's detectors flag a change. | Yes (the LLM API) | Real API and answers. The "silent provider update" is a simulated switch (tiny output limit, or a secretly different model). |
| **B. Real-data dashboard** (port 8502) | The runs collected from real hosted models, with DriftGuard's analysis. | No | Real |
| **C. Simulated dashboard** (port 8502) | A made-up model with an update injected at call 1,568, so every detector can be shown on a known case. | No | Simulated. Say so out loud. |

## 1. Pick your route

| The new machine... | Do this |
|---|---|
| already has Docker running | Route 1 (Sections 2 and 3) |
| has no Docker, but you may install software and virtualization is on | Install Docker Desktop **the day before** (it needs administrator rights and sometimes a restart), then Route 1 |
| has no Docker and you cannot install | Bring **your own laptop** and plug it into the projector (best), or use Route 2 (Section 4, Python without Docker) |
| allows nothing at all | Present from your recording and screenshots |

Ask the lab or your professor early: the operating system, whether Docker or Python are installed,
whether you can plug in your own laptop, and whether the internet (needed for Demo A) is available.

---

## 2. Prepare at home (the day before)

Do this on your own computer, in the project folder (`C:\dev\driftguard`), with Docker Desktop running.

**2.1 Build the image** (the first build downloads packages and takes several minutes):

```powershell
cd C:\dev\driftguard
docker build -t driftguard-demo .
```

**2.2 Make the data folder and put the databases in it:**

```powershell
mkdir demo-data -Force | Out-Null

# Demo B: a copy of your real collected data
Copy-Item driftguard.db demo-data\real_view.db -Force

# Demo C: generate the simulated database
docker run --rm -v "${PWD}\demo-data:/data" driftguard-demo python /app/scripts/make_demo_db.py --out /data/simulated_demo.db
```

**2.3 Build the baseline for Demo A.** The page needs at least 100 calls made before the "silent
change", so the detectors have something to compare against. This takes several minutes with a real
API, so do it now.

```powershell
docker run --rm -it -p 8501:8501 -v "${PWD}\demo-data:/data" driftguard-demo
```

Open <http://localhost:8501>. In the sidebar choose the provider and model and paste your API key
(typing is hidden and the key is never stored). Leave the "silently changes" box **unticked**, set
"Rounds to run" to **10**, and press **Run**. This makes 100 calls and saves them in
`demo-data\live_demo.db`. Stop the container with Ctrl+C.

**2.4 Export the image,** so the new machine does not need to build or download anything:

```powershell
docker save -o driftguard-demo.tar driftguard-demo
```

**2.5 Assemble the kit** on a USB drive (and keep a copy in cloud storage, since USB ports are
sometimes blocked):

```
kit\
  driftguard-demo.tar       the Docker image
  demo-data\                real_view.db, simulated_demo.db, live_demo.db
  DEMO.md                   this guide
  recording.mp4             a 2-minute screen recording of the whole flow (your fallback)
  screenshots\              one per dashboard tab
```

**2.6 Rehearse on a machine that has never seen the project,** using only the kit and Section 3.
Time it. You should be ready in under 10 minutes.

---

## 3. On the new machine

Open PowerShell (or a terminal) in the kit folder. In the commands below, `${PWD}\demo-data` is the
data folder inside the kit. On macOS or Linux use `-v "$PWD/demo-data:/data"` instead.

**3.1 Check that Docker is installed and running:**

```powershell
docker --version
docker info
```

If `docker info` says it cannot connect to the daemon, start Docker Desktop and wait until it reports
that it is running.

**3.2 Load the image (once):**

```powershell
docker load -i driftguard-demo.tar
```

This can take a few minutes. If you are on the same machine where you built the image, skip this step.

**3.3 Demo A: the live page.**

```powershell
docker run --rm -it -p 8501:8501 -v "${PWD}\demo-data:/data" driftguard-demo
```

Open <http://localhost:8501> in the browser. Streamlit prints `http://0.0.0.0:8501` in the terminal,
but that address does not work in a browser. Use `localhost`.

1. In the sidebar, choose the provider and model and paste your API key (hidden).
2. Show the green "No drift detected" and the flat chart. This is the baseline from step 2.3.
3. Edit or add a prompt to show that the prompts are yours.
4. Tick **The provider silently changes the model now**, set "Rounds to run" to **3**, and press **Run**.
5. Show the red banner, the orange triangles on the chart (where the detectors fired), the
   "What DriftGuard recommends" box and the "Every detection" table. In "Latest calls", point at
   `max_tokens` 8 and the empty answers.
6. Say clearly: the API and the answers are real; the silent update is simulated by me.

If a call fails, the page stops and shows the error. A 401 means the key is wrong, a 404 means the
model id is wrong, and a 429 means a rate limit (wait a minute).

Stop with Ctrl+C.

**3.4 Demo B: the real-data dashboard.**

```powershell
docker run --rm -it -p 8502:8502 -v "${PWD}\demo-data:/data" driftguard-demo driftguard dashboard --db-path /data/real_view.db --port 8502
```

Open <http://localhost:8502>. Present only what the data shows. If it shows stable runs with no
alarms, say that.

**3.5 Demo C: the simulated dashboard.**

```powershell
docker run --rm -it -p 8502:8502 -v "${PWD}\demo-data:/data" driftguard-demo driftguard dashboard --db-path /data/simulated_demo.db --port 8502
```

Say "this data is simulated" before you start. Walk through the tabs in order: Overview, Shift map,
Signals, Shift log & review, Call explorer.

The live page's own data (`live_demo.db`) can be opened in the full dashboard the same way, with
`--db-path /data/live_demo.db`.

**3.6 Clean up before you leave.**

```powershell
docker image rm driftguard-demo
Remove-Item demo-data\live_demo.db
```

Close the browser (or use a private window) and delete any copies of the kit from the machine.
`live_demo.db` stores your prompts and the model's answers in plain text.

---

## 4. Route 2: no Docker (Python on the machine)

Use this only if the machine has Python **3.10, 3.11 or 3.12** (check with `python --version`). Python
3.13 does not work, because the detection library `frouros` does not support it yet. That limit is the
reason for the Docker route.

At home, on a Windows machine with the same Python version, build an offline package folder:

```powershell
cd C:\dev\driftguard
.\.venv\Scripts\Activate.ps1
python -m build
$whl = (Get-ChildItem dist\*.whl | Sort-Object LastWriteTime | Select-Object -Last 1).FullName
mkdir kit-py -Force | Out-Null
Copy-Item $whl kit-py\
pip download "$whl[dashboard,proxy]" -d kit-py\wheelhouse
Copy-Item examples\live_page.py kit-py\
```

On the new machine, from the `kit-py` folder:

```powershell
python -m venv $env:TEMP\dg
$kit = (Get-Location).Path
& "$env:TEMP\dg\Scripts\python.exe" -m pip install --no-index --find-links "$kit\wheelhouse" (Get-ChildItem "$kit\*.whl").FullName
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"
$env:STREAMLIT_THEME_BASE = "light"
& "$env:TEMP\dg\Scripts\streamlit.exe" run "$kit\live_page.py"
```

For the dashboards, run `& "$env:TEMP\dg\Scripts\driftguard.exe" dashboard --db-path <file>`.

---

## 5. Troubleshooting

| Problem | Fix |
|---|---|
| `docker: command not found` | Docker is not installed or not on the PATH. Use another route. |
| `error during connect` / cannot connect to the Docker daemon | Docker Desktop is not running. Start it and wait. |
| `port is already allocated` | Use another host port, for example `-p 8601:8501`, and open `http://localhost:8601`. |
| Browser cannot reach the page | Use `http://localhost:PORT`, not `0.0.0.0`. Check the `-p` mapping. |
| `no matching manifest for linux/arm64` (or amd64) | The image was built on a different type of CPU. Build it on the same kind of machine as the target, or add `--platform`. |
| `permission denied` writing `/data` (Linux) | Run `sudo chown -R 1000 demo-data`. |
| `docker load` is slow | Normal, the image is large. Do it before the panel, not on stage. |
| The page says "No calls yet" | You have not pressed Run, or the data folder is not mounted. Check the `-v` argument. |
| `401` / `404` / `429` on the live page | Wrong key, wrong model id, or a rate limit. |
| The lab blocks the LLM API | Demo A cannot run. Use B and C, and your recording of A. |
| Dashboard text is hard to read | The image forces the light theme. If you still see the problem, use a private browser window. |

---

## 6. Security and hygiene

- **API key:** typed into the page, it lives only in memory for that session. The capture code never
  reads or stores the Authorization header. Use a throwaway key on a shared machine and revoke it afterwards.
- **Prompts and answers** are saved in plain text in `demo-data\live_demo.db`. Do not type anything private.
- **Do not point the demo at your real `driftguard.db`.** The kit uses a copy (`real_view.db`) on purpose,
  because the dashboard writes detected events back into the database it opens.
- Do not put keys or databases in the Git repository. The `.dockerignore` already keeps `*.db` files out
  of the image.

## 7. Reference

| Item | Value |
|---|---|
| Image name | `driftguard-demo` |
| Live page | port 8501 (the default command) |
| Dashboard | port 8502 (`driftguard dashboard --port 8502`) |
| Data folder in the container | `/data` (mount it from the host with `-v`) |
| Database the CLI uses by default | `/data/driftguard.db` (`DRIFTGUARD_DB`) |
| Python inside the image | 3.12 |
| Theme | forced light, through `STREAMLIT_THEME_*` environment variables |
| Other commands inside the image | `driftguard --help` (init, eval, collect, proxy, detect, dashboard) |

Examples:

```powershell
docker run --rm driftguard-demo driftguard --version
docker run --rm driftguard-demo driftguard --help
```
