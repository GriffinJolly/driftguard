"""
validation/runtime_benchmark.py

What DriftGuard costs to run, measured on this machine.

  capture overhead   extra time per API call added by the transport hook, which
                     parses the request and response and writes one row to SQLite
                     in the request path. Measured against an in-process mock
                     provider, so network time is excluded and the overhead is
                     the whole difference.
  storage            bytes per logged call, in the project database and in a fresh one
  detection          time for the live detector per call, and for the batch path
                     (PELT, permutation test, Benjamini-Hochberg) as history grows;
                     peak memory of the batch path
  live monitor       time to replay stored history at start-up, and per observed call

Usage:
    python -m validation.runtime_benchmark
"""

from __future__ import annotations

import json
import os
import platform
import sqlite3
import statistics
import tempfile
import time
import tracemalloc
from datetime import datetime, timedelta, timezone
from pathlib import Path

import logging
import httpx

logging.getLogger("httpx").setLevel(logging.WARNING)
import numpy as np
import pandas as pd

from driftguard.detection import pipeline
from driftguard.detection.bootstrap import confirm_changepoints
from driftguard.detection.changepoint_pelt import detect_changepoints
from driftguard.ingest.log_schema import CallOutcome, DriftLogEntry, IntegrationPath
from driftguard.ingest.transport_hook import DriftCaptureTransport
from driftguard.storage.store import DriftStore

OUT = Path("validation/results/runtime_benchmark")
DATA_ROOT = Path(os.environ.get("DRIFTGUARD_DATA_ROOT", "."))
S = pipeline.load_settings()

REPLY = {"id": "x", "object": "chat.completion", "model": "openai/gpt-oss-120b",
         "choices": [{"index": 0, "message": {"role": "assistant", "content": "The answer is B. " * 20}}],
         "usage": {"prompt_tokens": 245, "completion_tokens": 200, "total_tokens": 445}}
PROMPT = {"model": "openai/gpt-oss-120b", "temperature": 0, "max_tokens": 1536,
          "messages": [{"role": "user", "content": "Which of the following is true about X? " * 15}]}


def _mock(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=REPLY)


def capture_overhead(n: int = 1000) -> dict:
    url = "https://api.groq.com/openai/v1/chat/completions"
    plain = httpx.Client(transport=httpx.MockTransport(_mock))
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        store = DriftStore(db_path=str(Path(d) / "bench.db"))
        hooked = httpx.Client(transport=DriftCaptureTransport(store=store, wrapped=httpx.MockTransport(_mock)))
        hooked_hash = httpx.Client(transport=DriftCaptureTransport(store=store, wrapped=httpx.MockTransport(_mock),
                                                                   text_policy="redact"))
        for c in (plain, hooked):           # warm up
            for _ in range(50):
                c.post(url, json=PROMPT)
        t_plain, t_hook, t_red = [], [], []
        for _ in range(n):                  # interleave so drift in machine load affects all equally
            t = time.perf_counter(); plain.post(url, json=PROMPT); t_plain.append(time.perf_counter() - t)
            t = time.perf_counter(); hooked.post(url, json=PROMPT); t_hook.append(time.perf_counter() - t)
            t = time.perf_counter(); hooked_hash.post(url, json=PROMPT); t_red.append(time.perf_counter() - t)
        store.engine.dispose()
        db_bytes = (Path(d) / "bench.db").stat().st_size
        con = sqlite3.connect(Path(d) / "bench.db")
        rows = con.execute("SELECT COUNT(*) FROM log_rows").fetchone()[0]
        con.close()
    med_plain, med_hook = statistics.median(t_plain), statistics.median(t_hook)
    p95 = lambda xs: float(np.percentile(xs, 95))
    rows //= 2   # two hooked clients wrote to the same database
    return {"calls": n, "median_plain_ms": med_plain * 1e3, "median_hooked_ms": med_hook * 1e3,
            "median_overhead_ms": (med_hook - med_plain) * 1e3, "p95_overhead_ms": (p95(t_hook) - p95(t_plain)) * 1e3,
            "median_overhead_redact_ms": (statistics.median(t_red) - med_plain) * 1e3,
            "fresh_db_bytes_per_call": db_bytes / max(1, rows)}


def project_db_storage() -> dict:
    db = DATA_ROOT / "driftguard.db"
    if not db.exists():
        return {}
    con = sqlite3.connect(db)
    rows = con.execute("SELECT COUNT(*) FROM log_rows").fetchone()[0]
    text = con.execute("SELECT AVG(LENGTH(prompt) + LENGTH(COALESCE(response_text, ''))) FROM log_rows").fetchone()[0]
    con.close()
    return {"project_db_calls": rows, "project_db_bytes_per_call": db.stat().st_size / max(1, rows),
            "avg_prompt_plus_reply_chars": float(text or 0)}


def _stream(n: int, rng) -> np.ndarray:
    probs = np.full(n, 0.13)
    probs[n // 2:] = 0.40
    return (rng.random(n) < probs).astype(float)


def detection_cost() -> list[dict]:
    rows = []
    rng = np.random.default_rng(0)
    for n in (1_000, 10_000, 100_000):
        v = _stream(n, rng)
        t = time.perf_counter(); pipeline.page_hinkley_alerts(v, S); ph = time.perf_counter() - t
        tracemalloc.start()
        t = time.perf_counter()
        cps = detect_changepoints(v, pipeline.effective_pelt(n, S))
        confirm_changepoints(v, cps, S.significance)
        batch = time.perf_counter() - t
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        print("detection", n, "ph", round(ph, 3), "batch", round(batch, 2), flush=True)
        rows.append({"calls_in_history": n, "live_us_per_call": ph / n * 1e6, "batch_seconds": batch,
                     "batch_peak_mb": peak / 2**20, "change_points_tested": len(cps)})
    return rows


def live_monitor_cost(n_history: int = 5_000, n_live: int = 500) -> dict:
    from driftguard.detection.live import LiveMonitor
    t0 = datetime(2026, 9, 1, tzinfo=timezone.utc)
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        store = DriftStore(db_path=str(Path(d) / "lm.db"))

        def entry(i):
            return DriftLogEntry(integration_path=IntegrationPath.EVAL_SUITE, provider="groq", model_id="m", prompt="q",
                                 response_text="a", outcome=CallOutcome.OK, latency_ms=float(800 + rng.normal(0, 50)),
                                 eval_run_id="r", eval_task_type="accuracy", eval_task_id=f"q{i % 100}",
                                 eval_score=float(rng.random() > 0.13), timestamp=t0 + timedelta(seconds=i))
        for i in range(n_history):
            store.write_log(entry(i))
        t = time.perf_counter(); monitor = LiveMonitor(store, "groq", "m"); startup = time.perf_counter() - t
        times = []
        for i in range(n_history, n_history + n_live):
            e = entry(i)
            t = time.perf_counter(); monitor.observe(e); times.append(time.perf_counter() - t)
        store.engine.dispose()
    return {"history_calls": n_history, "startup_replay_seconds": startup,
            "observe_median_ms": statistics.median(times) * 1e3}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cap = capture_overhead(); print("capture", cap, flush=True)
    sto = project_db_storage()
    det = detection_cost()
    lm = live_monitor_cost(); print("live monitor", lm, flush=True)
    machine = f"{platform.processor() or platform.machine()}, Python {platform.python_version()}, {platform.system()} {platform.release()}"
    result = {"machine": machine, "capture": cap, "storage": sto, "detection": det, "live_monitor": lm}
    (OUT / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    calls_per_day = 112 * 8
    per_call = sto.get("project_db_bytes_per_call", cap["fresh_db_bytes_per_call"])
    lines = ["# Runtime overhead and scalability", "", f"Machine: {machine}.", "",
             "## Capture overhead (transport hook, mock provider, no network)", "",
             pd.DataFrame([cap]).round(3).to_markdown(index=False), "",
             "## Storage", "", pd.DataFrame([sto]).round(1).to_markdown(index=False) if sto else "(no project database)", "",
             f"At one 112-call suite run every 3 hours ({calls_per_day} calls a day) per target, storage grows by about "
             f"{per_call * calls_per_day / 2**20:.1f} MB a day per target ({per_call * calls_per_day * 365 / 2**30:.2f} GB a year).", "",
             "## Detection", "", pd.DataFrame(det).round(3).to_markdown(index=False), "",
             "## Live monitor", "", pd.DataFrame([lm]).round(3).to_markdown(index=False), ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
