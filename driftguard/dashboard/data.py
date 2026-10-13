"""
driftguard/dashboard/data.py

Data layer behind the dashboard. No Streamlit in here, so it can be tested and
reused on its own. It adds no detection or classification logic of its own:

  * detection    -> driftguard/detection/pipeline.py (per-call Page-Hinkley and
                    PELT / permutation test / Benjamini-Hochberg, events in the
                    classification vocabulary)
  * classification, incidents, urgency, plans
                 -> driftguard/classification/rules.py, incidents.py and
                    driftguard/remediation/mapper.py, evidence.py

What lives here is what only the dashboard needs: reading the call log with
prompts and responses for the call explorer, picking a scope, reading the
classification layer's verdicts back per signal, and saving reviewed events.

Wording follows the project's reporting rule: a Page-Hinkley alarm is an
"alert observed at call N", not the true onset; PELT gives the estimated
location of a confirmed shift.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from driftguard.classification import rules
from driftguard.classification.incidents import DEFAULT_WINDOW, Incident, group_into_incidents
from driftguard.detection import pipeline
from driftguard.detection.pipeline import (  # noqa: F401  (re-exported for the dashboard)
    COMPARE_WINDOW,
    MIN_SIGNAL_CALLS,
    DetectionSettings,
    SignalResult,
    Stream,
    load_settings,
)
from driftguard.detection.signals import SIGNALS
from driftguard.remediation import mapper
from driftguard.storage.store import ChangePointEventRow

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB = ROOT / "driftguard.db"
CALIBRATION_JSON = ROOT / "configs" / "detectors_calibrated.json"


# --------------------------------------------------------------------------- loading

def load_calls(db_path: Path | str = DEFAULT_DB) -> pd.DataFrame:
    """Every logged call, oldest first, including prompts and responses. Read-only connection."""
    con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query(
            "SELECT rowid AS seq_id, log_id, timestamp, integration_path, provider, model_id, "
            "prompt, response_text, request_params_json, outcome, error_message, latency_ms, "
            "prompt_tokens, completion_tokens, eval_run_id, eval_task_type, eval_task_id, eval_score "
            "FROM log_rows ORDER BY rowid",
            con,
        )
    finally:
        con.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
    df["target"] = df["provider"] + " · " + df["model_id"]
    return df


def load_events(db_path: Path | str = DEFAULT_DB) -> pd.DataFrame:
    """Events already recorded in the changepoint_events table."""
    con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    try:
        return pd.read_sql_query("SELECT * FROM changepoint_events ORDER BY detected_at DESC", con)
    except Exception:
        return pd.DataFrame()
    finally:
        con.close()


def run_catalog(calls: pd.DataFrame) -> pd.DataFrame:
    """One row per evaluation run for a target, for the scope picker."""
    g = calls.groupby(calls["eval_run_id"].fillna("(no run id)"), sort=False)
    out = g.agg(calls=("seq_id", "size"), first=("timestamp", "min"), last=("timestamp", "max"),
                start=("seq_id", "min")).reset_index()
    return out.sort_values("start").rename(columns={"eval_run_id": "run_id"})


def default_run(calls: pd.DataFrame, min_scored: int = 150) -> str | None:
    """The most recent evaluation run with enough scored calls to show one clean
    before/after experiment. None if no single run is long enough."""
    scored = calls[(calls["outcome"] == "ok") & calls["eval_score"].notna() & calls["eval_run_id"].notna()]
    counts = scored.groupby("eval_run_id")["seq_id"].agg(["size", "max"])
    counts = counts[counts["size"] >= min_scored]
    return None if counts.empty else str(counts["max"].idxmax())


def rolling_rate(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean, defined from the first call (shorter window at the start)."""
    c = np.cumsum(np.insert(values.astype(float), 0, 0.0))
    idx = np.arange(1, len(values) + 1)
    lo = np.maximum(0, idx - window)
    return (c[idx] - c[lo]) / (idx - lo)


# --------------------------------------------------------------------------- reading the classification layer's verdicts

URGENCY_RANK = {"clear": 0, "context": 0, mapper.NOTE: 1, mapper.WATCH: 2,
                mapper.INVESTIGATE: 3, mapper.ACT: 4, mapper.ACT_NOW: 5}


@dataclass
class DashEvent:
    """One detection from the pipeline, with rules.py's reading attached."""
    signal: str
    index: int
    seq_id: int
    event: ChangePointEventRow
    classification: rules.Classification


@dataclass
class Analysis:
    results: dict[str, SignalResult]
    events: list[DashEvent]
    incidents: list[Incident]
    plans: list[mapper.RemediationPlan]          # aligned with incidents
    incident_of: dict[int, int]                  # id(event) -> index into incidents


def load_sigmas(path: Path = CALIBRATION_JSON) -> Optional[dict[str, float]]:
    """Normal spread per metric from the quiet-window calibration, for noise units.
    None when no calibration file exists (rules.py then reports no noise units)."""
    if not path.exists():
        return None
    from driftguard.detection.baseline import load_calibration
    return rules.noise_sigmas_from_calibration(load_calibration(path))


def analyse_scope(calls: pd.DataFrame, provider: str, model_id: str,
                  s: Optional[DetectionSettings] = None, window: timedelta = DEFAULT_WINDOW,
                  sigmas: Optional[dict[str, float]] = None) -> Analysis:
    """Detection (pipeline.detect) -> classification (rules / incidents / mapper) for one scope."""
    det = pipeline.detect(calls, provider, model_id, s)
    events = [DashEvent(d.signal, d.index, d.seq_id, d.event, rules.classify_event(d.event, sigmas))
              for d in det.events]
    incidents = group_into_incidents([d.event for d in events], sigmas=sigmas, window=window)
    plans = [mapper.plan_for_incident(inc) for inc in incidents]
    incident_of = {id(m.event): k for k, inc in enumerate(incidents) for m in inc.members}
    return Analysis(det.results, events, incidents, plans, incident_of)


def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def shift_table(a: Analysis) -> pd.DataFrame:
    """One row per detection with the classification layer's verdict on it."""
    rows = []
    for d in a.events:
        c, ev = d.classification, d.event
        rule = rules.TAXONOMY.get(ev.metric_name)
        k = a.incident_of.get(id(ev))
        rows.append(dict(
            signal=d.signal, signal_label=SIGNALS[d.signal].label, metric=ev.metric_name,
            label=rule.display_name if (rule and c.drift_type) else None, drift_type=c.drift_type,
            detector="PELT" if ev.detector == "pelt" else "Page-Hinkley",
            tier=c.evidence_tier, adverse=c.adverse, index=d.index, seq_id=d.seq_id,
            timestamp=_utc(ev.changepoint_timestamp),
            before=ev.pre_mean, after=ev.post_mean, change=c.change, noise_units=c.noise_units,
            p_adj=ev.p_value_corrected, reason=c.reason,
            incident=None if k is None else a.incidents[k].incident_id,
            urgency=None if k is None else a.plans[k].urgency,
            in_taxonomy=rule is not None,
        ))
    cols = ["signal", "signal_label", "metric", "label", "drift_type", "detector", "tier", "adverse", "index",
            "seq_id", "timestamp", "before", "after", "change", "noise_units", "p_adj", "reason", "incident",
            "urgency", "in_taxonomy"]
    return pd.DataFrame(rows, columns=cols)


def signal_status(key: str, a: Analysis) -> str:
    """Urgency the classification layer gives this signal: the strongest per-metric evidence
    across the incidents it appears in (mapper.URGENCY_BY_TIER). 'context' for signals outside
    the taxonomy that had detections, 'clear' for none."""
    mine = [d for d in a.events if d.signal == key]
    metric = SIGNALS[key].metric
    if metric is None:
        return "context" if any(d.event.detector == "sequential" or d.event.significant for d in mine) else "clear"
    best = "clear"
    for d in mine:
        k = a.incident_of.get(id(d.event))
        if k is None:
            continue
        u = mapper.URGENCY_BY_TIER.get(a.incidents[k].metric_evidence(metric), mapper.NOTE)
        if URGENCY_RANK[u] > URGENCY_RANK[best]:
            best = u
    return best


def top_incident(a: Analysis) -> Optional[int]:
    """Index of the most urgent incident (latest wins a tie)."""
    if not a.incidents:
        return None
    return max(range(len(a.incidents)), key=lambda k: (URGENCY_RANK[a.plans[k].urgency], a.incidents[k].start))


def incident_evidence(db_path: Path | str, incident: Incident) -> dict:
    """Flip-set / infrastructure evidence from remediation/evidence.py. It needs per-run eval
    data (metric_series); empty when the scope has none, e.g. per-call drift trials."""
    from driftguard.remediation.evidence import evidence_for_incident
    from driftguard.storage.store import DriftStore
    try:
        return evidence_for_incident(DriftStore(db_path=str(db_path)), incident)
    except Exception:
        return {}


# --------------------------------------------------------------------------- persistence

def record_shifts(db_path: Path | str, analysis: Analysis) -> int:
    """Save the scope's open events (labelled, adverse, live or passing PELT) through
    pipeline.save_events, which then lets mapper.apply_remediation label and plan them."""
    from driftguard.storage.store import DriftStore

    open_events = [d.event for d in analysis.events
                   if d.classification.drift_type and d.classification.adverse and d.event.significant]
    written, _ = pipeline.save_events(DriftStore(db_path=str(db_path)), open_events)
    return written


def set_reviewed(db_path: Path | str, event_ids: list[int], reviewed: bool = True) -> None:
    con = sqlite3.connect(str(db_path))
    try:
        con.executemany("UPDATE changepoint_events SET reviewed = ? WHERE id = ?",
                        [(int(reviewed), int(i)) for i in event_ids])
        con.commit()
    finally:
        con.close()
