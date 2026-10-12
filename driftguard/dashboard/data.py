"""
driftguard/dashboard/data.py

Data and detection layer behind the dashboard. No Streamlit in here, so it can
be tested and reused on its own.

Pipeline, per (provider, model, scope):
  1. load the per-call log rows from driftguard.db (ordered by insertion, rowid)
  2. turn them into per-call signal streams, each oriented "higher = worse"
  3. run the ONLINE detector (tuned Page-Hinkley) over every stream, one update
     per call, exactly as it would run live
  4. run the RETROSPECTIVE path (PELT -> permutation test -> Benjamini-Hochberg)
     over the same streams to confirm and locate sustained shifts
  5. hand every shift to the classification layer AS AN EVENT in its own
     vocabulary (see dashboard/signals.py): rules.classify_event labels it and
     sets its evidence tier, incidents.group_into_incidents groups events that
     moved together, and remediation/mapper.py turns each incident into an
     urgency and a plan. The dashboard adds no classification rules of its own.

Wording follows the project's reporting rule: a Page-Hinkley alarm is an
"alert observed at call N", not the true onset; PELT gives the estimated
location of a confirmed shift.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from datetime import datetime, timedelta, timezone
from typing import Optional

from driftguard.classification import rules
from driftguard.classification.incidents import DEFAULT_WINDOW, Incident, group_into_incidents
from driftguard.dashboard.signals import SIGNALS, to_metric_value
from driftguard.detection.bootstrap import SignificanceConfig, confirm_changepoints
from driftguard.detection.changepoint_pelt import PeltConfig, detect_changepoints, segments_from_changepoints
from driftguard.remediation import mapper
from driftguard.storage.store import ChangePointEventRow

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB = ROOT / "driftguard.db"
DETECTORS_YAML = ROOT / "configs" / "detectors.yaml"
CALIBRATION_JSON = ROOT / "configs" / "detectors_calibrated.json"
XSTEST_FROZEN = ROOT / "configs" / "xstest_frozen.json"

COMPARE_WINDOW = 200     # calls each side of an online alert used for the before/after values
MIN_SIGNAL_CALLS = 60    # below this a stream is too short to say anything about


# --------------------------------------------------------------------------- config

@dataclass(frozen=True)
class DetectionSettings:
    ph_lambda: float = 12.0
    ph_delta: float = 0.005
    ph_alpha: float = 0.9999
    ph_min_instances: int = 30
    pelt: PeltConfig = field(default_factory=PeltConfig)
    significance: SignificanceConfig = field(default_factory=SignificanceConfig)
    baseline_window: int = 100   # calls used as the quiet baseline (calibration.min_window_size)


def load_settings(path: Path = DETECTORS_YAML) -> DetectionSettings:
    """Read the tuned detector settings from configs/detectors.yaml."""
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return DetectionSettings()
    ph = (cfg.get("live_detector") or {}).get("params") or {}
    rp = (cfg.get("retrospective_detector") or {}).get("params") or {}
    sg = cfg.get("significance") or {}
    cal = cfg.get("calibration") or {}
    return DetectionSettings(
        ph_lambda=float(ph.get("lambda_", 12.0)),
        ph_delta=float(ph.get("delta", 0.005)),
        ph_alpha=float(ph.get("alpha", 0.9999)),
        ph_min_instances=int(ph.get("min_num_instances", 30)),
        pelt=PeltConfig(
            cost_model=rp.get("cost_model", "l2"),
            penalty=rp.get("penalty"),
            penalty_scale=float(rp.get("penalty_scale", 5)),
            min_segment_length=int(rp.get("min_segment_length", 200)),
            jump=int(rp.get("jump", 25)),
        ),
        significance=SignificanceConfig(
            alpha=float(sg.get("alpha", 0.05)),
            n_bootstrap_samples=int(sg.get("n_bootstrap_samples", 1000)),
            window=sg.get("window", 2000),
        ),
        baseline_window=int(cal.get("min_window_size", 100)),
    )


# --------------------------------------------------------------------------- loading

def load_calls(db_path: Path | str = DEFAULT_DB) -> pd.DataFrame:
    """Every logged call, oldest first. Read-only connection."""
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
    """Shifts already recorded in the changepoint_events table."""
    con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    try:
        return pd.read_sql_query("SELECT * FROM changepoint_events ORDER BY detected_at DESC", con)
    except Exception:
        return pd.DataFrame()
    finally:
        con.close()


def _should_refuse_map() -> dict[str, bool]:
    try:
        tasks = json.loads(XSTEST_FROZEN.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    return {t["task_id"]: bool(t["should_refuse"]) for t in tasks}


def run_catalog(calls: pd.DataFrame) -> pd.DataFrame:
    """One row per evaluation run for a target, for the scope picker."""
    g = calls.groupby(calls["eval_run_id"].fillna("(no run id)"), sort=False)
    out = g.agg(calls=("seq_id", "size"), first=("timestamp", "min"), last=("timestamp", "max"),
                start=("seq_id", "min")).reset_index()
    return out.sort_values("start").rename(columns={"eval_run_id": "run_id"})


# --------------------------------------------------------------------------- signals

@dataclass
class Stream:
    key: str
    values: np.ndarray           # per-call, higher = worse
    seq_id: np.ndarray           # rowid of each call, links back to the log
    timestamp: pd.Series
    note: str = ""
    latency_ms: Optional[np.ndarray] = None   # measured latency per call (slow_call only)


def build_streams(calls: pd.DataFrame, baseline_window: int = 100) -> dict[str, Stream]:
    """Turn the call log into per-call signal streams (only those with enough data)."""
    streams: dict[str, Stream] = {}
    ok = calls[(calls["outcome"] == "ok") & calls["eval_score"].notna()]

    def add(key, frame, values, note="", latency=None):
        if len(frame) >= MIN_SIGNAL_CALLS:
            streams[key] = Stream(key, np.asarray(values, dtype=float), frame["seq_id"].to_numpy(),
                                  frame["timestamp"].reset_index(drop=True), note, latency)

    acc = ok[ok["eval_task_type"] == "accuracy"]
    add("accuracy_error", acc, 1.0 - acc["eval_score"])

    fmt = ok[ok["eval_task_type"] == "format"]
    add("format_error", fmt, 1.0 - fmt["eval_score"])

    ref = ok[ok["eval_task_type"] == "refusal"]
    if len(ref):
        should = ref["eval_task_id"].map(_should_refuse_map())
        safe, unsafe = ref[should == False], ref[should == True]  # noqa: E712
        add("safe_refusal", safe, safe["eval_score"])
        add("unsafe_compliance", unsafe, 1.0 - unsafe["eval_score"])

    # Rate limits are OUR quota running out, not the provider changing: keep them on
    # their own signal so they never feed the drift verdict.
    limited = calls["outcome"] == "rate_limited"
    served = calls[~limited]
    add("call_failure", served, (served["outcome"] != "ok").astype(float))
    if limited.any():
        add("quota_exhausted", calls, limited.astype(float))

    lat = calls[(calls["outcome"] == "ok") & calls["latency_ms"].notna()]
    if len(lat) >= max(MIN_SIGNAL_CALLS, baseline_window + 20):
        p95 = float(np.percentile(lat["latency_ms"].iloc[:baseline_window], 95))
        add("slow_call", lat, (lat["latency_ms"] > p95).astype(float),
            note=f"slow = above {p95:,.0f} ms, the p95 of the first {baseline_window} calls",
            latency=lat["latency_ms"].to_numpy(dtype=float))
    return streams


# --------------------------------------------------------------------------- detection

def page_hinkley_alerts(values: np.ndarray, s: DetectionSettings) -> list[int]:
    """Indices where the tuned Page-Hinkley detector alarms, fed one call at a time.
    The detector is reset after each alarm, so later alerts are independent."""
    from frouros.detectors.concept_drift import PageHinkley, PageHinkleyConfig

    def fresh():
        return PageHinkley(config=PageHinkleyConfig(lambda_=s.ph_lambda, delta=s.ph_delta,
                                                    alpha=s.ph_alpha, min_num_instances=s.ph_min_instances))
    det, alerts = fresh(), []
    for i, v in enumerate(values):
        det.update(value=float(v))
        if det.status["drift"]:
            alerts.append(i)
            det = fresh()
    return alerts


def effective_pelt(n: int, s: DetectionSettings) -> PeltConfig:
    """The tuned minimum segment (200) assumes long histories. For short scopes it is
    capped at a fifth of the stream so PELT can still run; never below 30 calls."""
    p = s.pelt
    seg = max(30, min(p.min_segment_length, n // 5))
    jump = max(1, min(p.jump, seg // 4))
    return PeltConfig(cost_model=p.cost_model, penalty=p.penalty, penalty_scale=p.penalty_scale,
                      min_segment_length=seg, jump=jump)


@dataclass
class SignalResult:
    stream: Stream
    alerts: list[int]
    confirmed: list[dict]               # every PELT candidate with its test result
    segments: list[tuple[int, int, float]]  # (start, end, mean) of the PELT regimes
    pelt_config: PeltConfig


def analyse(streams: dict[str, Stream], s: DetectionSettings) -> dict[str, SignalResult]:
    results = {}
    for key, st in streams.items():
        v = st.values
        pcfg = effective_pelt(len(v), s)
        cps = detect_changepoints(v, pcfg)
        tested = confirm_changepoints(v, cps, s.significance) if cps else []
        kept = [r["index"] for r in tested if r["significant"]]
        segs = [(a, b, float(v[a:b].mean())) for a, b in segments_from_changepoints(len(v), kept)]
        results[key] = SignalResult(st, page_hinkley_alerts(v, s), tested, segs, pcfg)
    return results


def alert_effect(values: np.ndarray, i: int) -> tuple[float, float]:
    """Rate in the COMPARE_WINDOW calls before and after an online alert."""
    pre, post = values[max(0, i - COMPARE_WINDOW):i], values[i:i + COMPARE_WINDOW]
    return (float(pre.mean()) if len(pre) else np.nan, float(post.mean()) if len(post) else np.nan)


def _ts(stream: Stream, i: int):
    return stream.timestamp.iloc[min(i, len(stream.timestamp) - 1)]


# --------------------------------------------------------------------------- classification bridge
#
# Everything below hands the detections to the classification layer
# (classification/rules.py, classification/incidents.py, remediation/mapper.py)
# and reads its verdicts back. No labelling decisions are made here.

URGENCY_RANK = {"clear": 0, "context": 0, mapper.NOTE: 1, mapper.WATCH: 2,
                mapper.INVESTIGATE: 3, mapper.ACT: 4, mapper.ACT_NOW: 5}


@dataclass
class DashEvent:
    """One detection, as an event in the classification layer's vocabulary."""
    signal: str                       # dashboard signal key (signals.py)
    index: int                        # position in that signal's stream
    seq_id: int                       # log row of the call where it was placed
    event: ChangePointEventRow        # what rules.py / incidents.py read
    classification: rules.Classification


@dataclass
class Analysis:
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


def _p95(x: np.ndarray) -> float:
    return float(np.percentile(x, 95)) if len(x) else np.nan


def _metric_values(st: Stream, lo_pre: int, cut: int, hi_post: int, rate_pre: float, rate_post: float):
    """Before/after values in the metric's own units."""
    metric = SIGNALS[st.key].metric
    if metric == "latency_p95_ms" and st.latency_ms is not None:
        return _p95(st.latency_ms[lo_pre:cut]), _p95(st.latency_ms[cut:hi_post])
    return to_metric_value(st.key, rate_pre), to_metric_value(st.key, rate_post)


def _naive_utc(ts) -> datetime:
    return pd.Timestamp(ts).tz_convert("UTC").tz_localize(None).to_pydatetime()


def build_events(results: dict[str, SignalResult], provider: str, model_id: str,
                 s: DetectionSettings, sigmas: Optional[dict[str, float]] = None) -> list[DashEvent]:
    """Turn every detection into a ChangePointEventRow (not saved) and classify it with rules.py.
    PELT candidates that failed the significance test are included: rules.py calls them
    'unconfirmed', exactly as it would for stored events."""
    now = datetime.now(timezone.utc)
    win = s.significance.window or 10**9
    out: list[DashEvent] = []
    for key, r in results.items():
        st, v, n = r.stream, r.stream.values, len(r.stream.values)
        metric = SIGNALS[key].metric or key
        bounds = [0, *[t["index"] for t in r.confirmed], n]
        for j, t in enumerate(r.confirmed):
            i = t["index"]
            pre, post = _metric_values(st, max(bounds[j], i - win), i, min(bounds[j + 2], i + win),
                                       t["pre_mean"], t["post_mean"])
            ev = ChangePointEventRow(
                detected_at=now, provider=provider, model_id=model_id, metric_name=metric, detector="pelt",
                changepoint_timestamp=_naive_utc(_ts(st, i)), pre_mean=pre, post_mean=post,
                p_value_raw=t["p_value_raw"], p_value_corrected=t.get("p_value_corrected"),
                significant=bool(t.get("significant")))
            out.append(DashEvent(key, i, int(st.seq_id[i]), ev, rules.classify_event(ev, sigmas)))
        for i in r.alerts:
            rb, ra = alert_effect(v, i)
            pre, post = _metric_values(st, max(0, i - COMPARE_WINDOW), i, i + COMPARE_WINDOW, rb, ra)
            ev = ChangePointEventRow(
                detected_at=now, provider=provider, model_id=model_id, metric_name=metric, detector="sequential",
                changepoint_timestamp=_naive_utc(_ts(st, i)), pre_mean=pre, post_mean=post, significant=True)
            out.append(DashEvent(key, i, int(st.seq_id[i]), ev, rules.classify_event(ev, sigmas)))
    out.sort(key=lambda d: (d.seq_id, d.event.detector))
    return out


def classify_detections(results: dict[str, SignalResult], provider: str, model_id: str,
                        s: DetectionSettings, window: timedelta = DEFAULT_WINDOW,
                        sigmas: Optional[dict[str, float]] = None) -> Analysis:
    """Detections -> events -> incidents -> remediation plans, all via the classification layer."""
    events = build_events(results, provider, model_id, s, sigmas)
    incidents = group_into_incidents([d.event for d in events], sigmas=sigmas, window=window)
    plans = [mapper.plan_for_incident(inc) for inc in incidents]
    incident_of = {id(m.event): k for k, inc in enumerate(incidents) for m in inc.members}
    return Analysis(events, incidents, plans, incident_of)


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
            timestamp=pd.Timestamp(ev.changepoint_timestamp, tz="UTC"),
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


# --------------------------------------------------------------------------- persistence

def record_shifts(db_path: Path | str, analysis: Analysis) -> int:
    """Write the open events (labelled, adverse, and live or passing PELT) into changepoint_events,
    skipping any already recorded, then let remediation/mapper.apply_remediation fill drift_type
    and the one-line plan, exactly as it does for any stored event. Returns rows added."""
    from driftguard.storage.store import DriftStore

    store = DriftStore(db_path=str(db_path))
    existing = load_events(db_path)
    seen = set()
    if not existing.empty:
        for _, e in existing.iterrows():
            seen.add((e["provider"], e["model_id"], e["metric_name"], e["detector"],
                      str(e["changepoint_timestamp"])[:19]))
    added = 0
    for d in analysis.events:
        ev, c = d.event, d.classification
        if not (c.drift_type and c.adverse and ev.significant):
            continue
        key = (ev.provider, ev.model_id, ev.metric_name, ev.detector, str(ev.changepoint_timestamp)[:19])
        if key in seen:
            continue
        store.write_changepoint_event(ChangePointEventRow(**ev.model_dump(exclude={"id"})))
        seen.add(key)
        added += 1
    if added:
        mapper.apply_remediation(store)
    return added


def set_reviewed(db_path: Path | str, event_ids: list[int], reviewed: bool = True) -> None:
    con = sqlite3.connect(str(db_path))
    try:
        con.executemany("UPDATE changepoint_events SET reviewed = ? WHERE id = ?",
                        [(int(reviewed), int(i)) for i in event_ids])
        con.commit()
    finally:
        con.close()
