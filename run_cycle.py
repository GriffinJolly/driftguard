#!/usr/bin/env python3
"""Single-shot collector for driftguard. Run on a schedule (cron / Task Scheduler).

Each invocation runs the eval suite once per target, stores the results via
the normal runner/store path, and exits. Collection only: no calibration or
detection here, so data keeps accumulating while those decisions are made.

A target only counts as a success (and only then updates the heartbeat) if
enough of its calls came back OK. A run where most calls errored is reported
as FAILED, so a dead model or a provider outage cannot look healthy.
"""
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
LOCK = ROOT / ".run_cycle.lock"
HEARTBEAT = ROOT / "logs" / "last_success.json"
STALE_LOCK_SECONDS = 50 * 60   # a hung run must not block the next hours forever
GAP_BETWEEN_TARGETS = 10       # seconds, eases free-tier rate limits
MIN_OK_FRACTION = 0.5          # below this share of OK calls, the run counts as failed
DB_PATH = os.environ.get("DRIFTGUARD_DB", str(ROOT / "driftguard.db"))

# (provider, model_id): the cross-provider validation pair.
# A run is 112 calls (100 MMLU, 6 format, 6 refusal), about 8 min on Groq and
# 15-20 min on NVIDIA. Groq's free tier allows ~1,000 calls/day, so schedule
# this every 3 hours, not hourly. Nemotron moved from OpenRouter's free tier
# (50 calls/day, too few for one run) to NVIDIA's API, which has no daily cap.
TARGETS = [
    ("groq", "openai/gpt-oss-120b"),
    ("nvidia", "nvidia/nemotron-3-super-120b-a12b"),
]

LOG_DIR.mkdir(exist_ok=True)
log = logging.getLogger("run_cycle")
log.setLevel(logging.INFO)
_h = RotatingFileHandler(LOG_DIR / "run_cycle.log", maxBytes=2_000_000, backupCount=5)
_h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
log.addHandler(_h)
log.addHandler(logging.StreamHandler(sys.stdout))


class RunFailed(Exception):
    """Raised when a target's run stored too few OK calls to count as a success."""


def acquire_lock() -> bool:
    """Portable lock: create-exclusive file, taking over if it is stale."""
    if LOCK.exists() and time.time() - LOCK.stat().st_mtime > STALE_LOCK_SECONDS:
        log.warning("stale lock found, taking over")
        LOCK.unlink(missing_ok=True)
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def _max_rowid() -> int:
    """Highest rowid in log_rows before a run (0 if the table does not exist yet)."""
    con = sqlite3.connect(DB_PATH)
    try:
        return con.execute("select coalesce(max(rowid), 0) from log_rows").fetchone()[0]
    except sqlite3.OperationalError:
        return 0
    finally:
        con.close()


def _outcome_counts(provider: str, since_rowid: int) -> tuple[int, int, bool]:
    """(total rows, OK rows, all-failures-were-404) for rows this run added."""
    con = sqlite3.connect(DB_PATH)
    try:
        total, ok = con.execute(
            "select count(*), coalesce(sum(outcome = 'ok'), 0) from log_rows "
            "where rowid > ? and provider = ?",
            (since_rowid, provider),
        ).fetchone()
        model_gone = False
        try:
            n404 = con.execute(
                "select count(*) from log_rows where rowid > ? and provider = ? "
                "and outcome != 'ok' and error_message like '%404%'",
                (since_rowid, provider),
            ).fetchone()[0]
            model_gone = n404 > 0 and n404 == total - ok
        except sqlite3.OperationalError:
            pass  # no error_message column; skip the 404 hint
        return total, ok, model_gone
    finally:
        con.close()


_store = None


def run_one(provider: str, model_id: str) -> None:
    """Run the full eval suite once for one target via runner.run_once, then
    verify that enough calls succeeded; raise RunFailed if not.

    run_once contains per-task failures itself: it always saves every call as a
    LogRow plus one MetricPoint (None for any metric it could not compute).
    Imports are lazy so the lock and logging are in place even if the package
    import itself fails.
    """
    global _store
    from driftguard.evalsuite.runner import run_once
    from driftguard.storage.store import DriftStore

    if _store is None:
        _store = DriftStore(db_path=DB_PATH)

    since = _max_rowid()
    run_once(provider, model_id, store=_store)

    total, ok, model_gone = _outcome_counts(provider, since)
    if total == 0:
        raise RunFailed(f"{provider} {model_id}: run stored no rows")
    if ok / total < MIN_OK_FRACTION:
        hint = " (all errors were 404: model likely removed or no longer free)" if model_gone else ""
        raise RunFailed(f"{provider} {model_id}: only {ok}/{total} calls OK{hint}")
    if ok < total:
        log.warning("%s %s: %d/%d calls OK", provider, model_id, ok, total)


def main() -> int:
    if not acquire_lock():
        log.warning("previous run still active, skipping this cycle")
        return 0
    failures = 0
    try:
        beat = {}
        if HEARTBEAT.exists():
            try:
                beat = json.loads(HEARTBEAT.read_text())
            except json.JSONDecodeError:
                beat = {}
        for i, (provider, model_id) in enumerate(TARGETS):
            if i:
                time.sleep(GAP_BETWEEN_TARGETS)
            start = time.time()
            try:
                run_one(provider, model_id)
                beat[f"{provider}:{model_id}"] = datetime.now(timezone.utc).isoformat()
                log.info("ok %s %s in %.0fs", provider, model_id, time.time() - start)
            except RunFailed as e:
                failures += 1
                log.error("FAILED %s", e)
            except Exception:  # one target failing must not stop the others
                failures += 1
                log.exception("FAILED %s %s", provider, model_id)
        HEARTBEAT.write_text(json.dumps(beat, indent=2))
    finally:
        LOCK.unlink(missing_ok=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())