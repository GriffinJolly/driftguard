"""
Eval suite runner for DriftGuard.

Ties tasks.py (accuracy/MMLU), format_check.py (format adherence),
and refusal_probe.py (refusal/XSTest) into one batch "run": call all
three, save every individual call as a LogRow, aggregate into one
MetricPoint per run, and save that too. This is what produces the
hourly time series everything downstream (calibration, detection,
classification) consumes.

Usage as a one-off:
    from driftguard.evalsuite.runner import run_once
    point = run_once("groq", "llama-3.3-70b-versatile")

Usage as a continuous scheduled job (what actually runs on the VM):
    python -m driftguard.evalsuite.runner --provider groq --model llama-3.3-70b-versatile --interval-seconds 3600

Design notes:
- One run = one call to run_once(). It runs accuracy, then format, then
  refusal tasks IN SEQUENCE (not concurrently) -- this keeps total calls
  per minute predictable and easy to reason about against the free-tier
  rate caps, since clients.py's RateLimiter is already handling the
  actual throttling underneath.
- latency_p50/p95 are computed from every successful call in the run
  (across all three task types), not just one task type, since latency
  is a property of the provider/infra, not of what's being asked.
- Individual call failures don't abort a run -- run_once always returns
  a MetricPoint, with None for any metric that couldn't be computed
  (e.g. every call in that category failed). This mirrors clients.py's
  own philosophy of never letting one bad call take down a batch.
"""

from __future__ import annotations

import argparse
import logging
import statistics
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from driftguard.evalsuite.tasks import run_accuracy_tasks, accuracy_rate
from driftguard.evalsuite.format_check import run_format_tasks, format_adherence_rate
from driftguard.evalsuite.refusal_probe import run_refusal_tasks, refusal_rate
from driftguard.ingest.log_schema import DriftLogEntry, MetricPoint
from driftguard.storage.store import DriftStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("driftguard.runner")


def _latency_percentiles(entries: list[DriftLogEntry]) -> tuple[Optional[float], Optional[float]]:
    """p50/p95 latency across every successful call in the run."""
    latencies = sorted(e.latency_ms for e in entries if e.latency_ms is not None)
    if not latencies:
        return None, None

    def _percentile(data: list[float], pct: float) -> float:
        if len(data) == 1:
            return data[0]
        k = (len(data) - 1) * pct
        f = int(k)
        c = min(f + 1, len(data) - 1)
        if f == c:
            return data[f]
        return data[f] + (data[c] - data[f]) * (k - f)

    return _percentile(latencies, 0.50), _percentile(latencies, 0.95)


def run_once(
    provider: str,
    model_id: str,
    store: Optional[DriftStore] = None,
    db_path: str = "driftguard.db",
) -> MetricPoint:
    """
    Run one full batch (accuracy + format + refusal) against
    (provider, model_id), save every call and the aggregated point to
    the store, and return the MetricPoint.
    """
    run_id = str(uuid.uuid4())
    store = store or DriftStore(db_path=db_path)

    logger.info(f"Starting eval run {run_id} for provider={provider} model={model_id}")

    all_entries: list[DriftLogEntry] = []

    try:
        acc_entries = run_accuracy_tasks(provider, model_id, run_id)
        all_entries.extend(acc_entries)
        logger.info(f"  accuracy: {len(acc_entries)} calls, rate={accuracy_rate(acc_entries)}")
    except Exception:
        logger.exception("Accuracy task batch failed entirely")
        acc_entries = []

    try:
        fmt_entries = run_format_tasks(provider, model_id, run_id)
        all_entries.extend(fmt_entries)
        logger.info(f"  format: {len(fmt_entries)} calls, rate={format_adherence_rate(fmt_entries)}")
    except Exception:
        logger.exception("Format task batch failed entirely")
        fmt_entries = []

    try:
        ref_entries = run_refusal_tasks(provider, model_id, run_id)
        all_entries.extend(ref_entries)
        logger.info(f"  refusal: {len(ref_entries)} calls, safe_rate={refusal_rate(ref_entries, 'safe')}")
    except Exception:
        logger.exception("Refusal task batch failed entirely")
        ref_entries = []

    # save every individual call, regardless of outcome
    for entry in all_entries:
        store.write_log(entry)

    p50, p95 = _latency_percentiles(all_entries)

    point = MetricPoint(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc),
        provider=provider,
        model_id=model_id,
        accuracy=accuracy_rate(acc_entries),
        format_adherence=format_adherence_rate(fmt_entries),
        refusal_rate=refusal_rate(ref_entries, subset="safe"),
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        n_calls=len(all_entries),
    )
    store.write_metric_point(point)

    logger.info(
        f"Run {run_id} complete: accuracy={point.accuracy}, format={point.format_adherence}, "
        f"refusal={point.refusal_rate}, p50={point.latency_p50_ms}, p95={point.latency_p95_ms}, "
        f"n_calls={point.n_calls}"
    )

    return point


def run_forever(
    provider: str,
    model_id: str,
    interval_seconds: int = 3600,
    db_path: str = "driftguard.db",
) -> None:
    """
    Run the eval suite on a fixed interval, forever, until interrupted.
    This is what a systemd service / cron-launched long-lived process
    on the always-on VM actually invokes.

    Each run's own exceptions are already contained inside run_once
    (per-task try/except), but this loop also guards the OUTER call so
    a totally unexpected error (e.g. a store/database issue) logs and
    waits for the next interval instead of killing the whole process --
    losing one hour of data is much better than losing the rest of the
    multi-week calibration run.
    """
    store = DriftStore(db_path=db_path)
    logger.info(
        f"Starting continuous run loop: provider={provider} model={model_id} "
        f"interval={interval_seconds}s db={db_path}"
    )

    while True:
        cycle_start = time.monotonic()
        try:
            run_once(provider, model_id, store=store)
        except Exception:
            logger.exception("Unexpected error in run_once -- will retry next interval")

        elapsed = time.monotonic() - cycle_start
        sleep_for = max(0.0, interval_seconds - elapsed)
        logger.info(f"Sleeping {sleep_for:.0f}s until next run")
        time.sleep(sleep_for)


def _main() -> None:
    parser = argparse.ArgumentParser(description="DriftGuard eval suite runner")
    parser.add_argument("--provider", required=True, choices=["groq", "openrouter"])
    parser.add_argument("--model", required=True, help="Model ID as the provider names it")
    parser.add_argument("--db-path", default="driftguard.db")
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=None,
        help="If set, run continuously on this interval. If omitted, run once and exit.",
    )
    args = parser.parse_args()

    if args.interval_seconds:
        run_forever(args.provider, args.model, interval_seconds=args.interval_seconds, db_path=args.db_path)
    else:
        run_once(args.provider, args.model, db_path=args.db_path)


if __name__ == "__main__":
    _main()