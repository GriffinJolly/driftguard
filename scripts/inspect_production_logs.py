"""
driftguard/scripts/inspect_production_logs.py

Quick visibility into captured wrapper/proxy traffic without hand-writing
SQL or trusting a possibly-stale DB-viewer GUI cache. Queries
driftguard.db directly with sqlite3 (same trick that confirmed
runner.py was actually writing data earlier).

Usage:
    python -m driftguard.scripts.inspect_production_logs
    python -m driftguard.scripts.inspect_production_logs --limit 20
    python -m driftguard.scripts.inspect_production_logs --provider groq
    python -m driftguard.scripts.inspect_production_logs --integration-path wrapper
    python -m driftguard.scripts.inspect_production_logs --db path/to/driftguard.db

Reads the real LogRow table (__tablename__ = "log_rows") from store.py.
Column names match LogRow's fields directly: log_id, timestamp,
integration_path, provider, model_id, prompt, response_text,
request_params_json (JSON-encoded, decoded below for display), outcome,
error_message, latency_ms, prompt_tokens, completion_tokens,
eval_run_id, eval_task_type, eval_task_id, eval_score.
"""

import argparse
import json
import sqlite3
import sys
import textwrap
from pathlib import Path

TABLE_NAME = "log_rows"


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect captured LogRow entries.")
    parser.add_argument(
        "--db",
        default="driftguard.db",
        help="Path to the SQLite DB file (default: driftguard.db in cwd)",
    )
    parser.add_argument("--limit", type=int, default=10, help="Number of rows to show")
    parser.add_argument("--provider", default=None, help="Filter by provider (e.g. groq)")
    parser.add_argument(
        "--integration-path",
        default=None,
        choices=["wrapper", "proxy", "eval_suite"],
        help="Filter by integration_path (default: no filter, shows all)",
    )
    parser.add_argument(
        "--outcome",
        default=None,
        choices=["ok", "error", "timeout", "rate_limited"],
        help="Filter by outcome",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB not found at {db_path.resolve()}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    query = f"SELECT * FROM {TABLE_NAME}"
    conditions = []
    params = []

    if args.integration_path:
        conditions.append("integration_path = ?")
        params.append(args.integration_path)
    if args.provider:
        conditions.append("provider = ?")
        params.append(args.provider)
    if args.outcome:
        conditions.append("outcome = ?")
        params.append(args.outcome)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(args.limit)

    try:
        rows = conn.execute(query, params).fetchall()
    except sqlite3.OperationalError as exc:
        print(
            f"Query failed ({exc}). Check TABLE_NAME/column names in this script "
            f"match your actual schema (run: sqlite3 {db_path} '.schema').",
            file=sys.stderr,
        )
        return 1

    if not rows:
        print("No matching rows found.")
        return 0

    for row in rows:
        d = dict(row)
        print("-" * 70)
        print(f"log_id:           {d.get('log_id')}")
        print(f"timestamp:        {d.get('timestamp')}")
        print(f"integration_path: {d.get('integration_path')}")
        print(f"provider:         {d.get('provider')}")
        print(f"model_id:         {d.get('model_id')}")
        print(f"outcome:          {d.get('outcome')}")
        print(f"latency_ms:       {d.get('latency_ms')}")
        if d.get("prompt_tokens") is not None or d.get("completion_tokens") is not None:
            print(f"tokens:           prompt={d.get('prompt_tokens')} completion={d.get('completion_tokens')}")
        if d.get("prompt"):
            print(f"prompt:           {textwrap.shorten(str(d['prompt']), width=120)}")
        if d.get("response_text"):
            print(f"response:         {textwrap.shorten(str(d['response_text']), width=120)}")
        raw_params = d.get("request_params_json")
        if raw_params and raw_params != "{}":
            try:
                parsed_params = json.loads(raw_params)
                print(f"request_params:   {parsed_params}")
            except Exception:
                print(f"request_params:   {raw_params}")
        if d.get("error_message"):
            print(f"error_message:    {d.get('error_message')}")
        if d.get("eval_run_id"):
            print(
                f"eval_run_id:      {d.get('eval_run_id')}  "
                f"task_type={d.get('eval_task_type')}  score={d.get('eval_score')}"
            )

    print("-" * 70)
    print(f"Showed {len(rows)} row(s) from {db_path.resolve()}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())