"""
validation/provider_variants.py

Cross-provider / cross-model drift-detector validation.

The synthetic benchmark (run_validation.py) and the real-data replays
(live_replay.py, live_replay_groq.py) each validate the detectors against ONE
error stream at a time. This module answers a different question: does a
detector behave the SAME across different (provider, model) variants, or does a
config tuned on one model misbehave on another? That matters because DriftGuard
watches many (provider, model, metric) series in production, and a live detector
that's trustworthy on one model but trigger-happy on another is a real risk.

It does NOT make any live API calls. It reads the REAL per-call correctness
already captured in driftguard.db (written by runner.py / live_replay_groq.py /
live_drift_compare.py), splits it by (provider, model) according to
validation/provider_config.yaml, and runs the detector registry over each
variant's error stream -- recording every alert, exactly like live_replay.py's
real-data mode -- so the variants are directly comparable.

Usage (from the repo root):
    python -m validation.provider_variants
    python -m validation.provider_variants --registry tuned --db-path driftguard.db
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from validation.detectors_registry import DetectorSpec, default_registry, tuned_registry
from validation.live_replay import run_all_alerts

_CONFIG_DEFAULT = Path(__file__).resolve().parent / "provider_config.yaml"


def load_variants(config_path: Path = _CONFIG_DEFAULT) -> list[dict]:
    """Read the [{name, provider, model}, ...] variant list from the YAML config."""
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    variants = data.get("variants", [])
    for v in variants:
        missing = {"name", "provider", "model"} - set(v)
        if missing:
            raise ValueError(f"variant {v!r} is missing keys: {sorted(missing)}")
    return variants


def error_stream_for_variant(db_path: str, provider: str, model: str) -> np.ndarray:
    """Real 0/1 error stream (1 == wrong) for one (provider, model), in call
    order, over every scored accuracy call captured in driftguard.db.

    eval_score is 1.0 for a correct call and 0.0 for an incorrect one; failed
    calls (eval_score IS NULL) never entered the stream, same as everywhere else
    in this suite. Ordered by log_id, which is insertion (time) order.
    """
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT eval_score FROM log_rows "
            "WHERE provider = ? AND model_id = ? AND eval_score IS NOT NULL "
            "ORDER BY log_id",
            (provider, model),
        ).fetchall()
    finally:
        con.close()
    return np.array([0 if r[0] == 1.0 else 1 for r in rows], dtype=np.int8)


def compare_variants(
    streams: dict[str, np.ndarray], registry: list[DetectorSpec]
) -> pd.DataFrame:
    """Run every detector over every variant's error stream and summarize.

    Pure function of (streams, registry) -- no DB, no network -- so it can be
    unit-tested with hand-built streams. One row per (variant, detector).
    """
    rows = []
    for variant_name, stream in streams.items():
        for spec in registry:
            alerts = run_all_alerts(spec, stream) if len(stream) else []
            rows.append(
                {
                    "variant": variant_name,
                    "detector": spec.name,
                    "n_samples": len(stream),
                    "error_rate": float(stream.mean()) if len(stream) else float("nan"),
                    "n_alerts": len(alerts),
                    "first_alert_index": alerts[0] if alerts else None,
                    "alerts_per_1000_samples": (len(alerts) / len(stream) * 1000) if len(stream) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def run(db_path: str, config_path: Path, registry: list[DetectorSpec], out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = load_variants(config_path)

    streams: dict[str, np.ndarray] = {}
    for v in variants:
        stream = error_stream_for_variant(db_path, v["provider"], v["model"])
        streams[v["name"]] = stream
        print(f"{v['name']:34} {v['provider']}/{v['model']}  "
              f"n={len(stream):>5}  error_rate={stream.mean():.3f}" if len(stream)
              else f"{v['name']:34} {v['provider']}/{v['model']}  n=0  (no captured traffic yet)")

    summary = compare_variants(streams, registry)
    out_path = out_dir / "provider_variants_summary.csv"
    summary.to_csv(out_path, index=False)
    print("\n" + summary.to_string(index=False))
    print(f"\nWrote {out_path}")

    empty = [name for name, s in streams.items() if len(s) == 0]
    if empty:
        print("\nNote: no captured traffic yet for: " + ", ".join(empty) +
              ".\nRun the eval suite / live replays against these variants first "
              "(they write to driftguard.db), then re-run this comparison.")
    return summary


def _main() -> None:
    p = argparse.ArgumentParser(description="Cross-provider/model drift-detector validation (reads driftguard.db)")
    p.add_argument("--db-path", dest="db_path", default="driftguard.db")
    p.add_argument("--config", default=str(_CONFIG_DEFAULT))
    p.add_argument("--registry", choices=["default", "tuned"], default="tuned",
                   help="'default' = library defaults + Page-Hinkley (tuned); 'tuned' = all 8 detectors tuned.")
    p.add_argument("--out", default="validation/results/provider_variants")
    args = p.parse_args()
    registry = tuned_registry() if args.registry == "tuned" else default_registry(include_tuned_variants=True)
    run(args.db_path, Path(args.config), registry, Path(args.out))


if __name__ == "__main__":
    _main()
