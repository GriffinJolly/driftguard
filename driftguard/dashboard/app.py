"""
DriftGuard operator dashboard.

    streamlit run driftguard/dashboard/app.py

Reads driftguard.db (set DRIFTGUARD_DB to point elsewhere), runs the tuned
online detector and the retrospective PELT path over every per-call signal,
and hands every detection to the classification layer -- rules.py labels it,
incidents.py groups it, remediation/mapper.py gives the urgency and the plan.
"""

from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # `streamlit run` only puts this file's folder on the path
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from driftguard.classification import rules
from driftguard.dashboard import data as D
from driftguard.dashboard.signals import SIGNALS
from driftguard.remediation import mapper

DB_PATH = Path(os.environ.get("DRIFTGUARD_DB", D.DEFAULT_DB))

# ----------------------------------------------------------------------------- design tokens
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e8e7e1", "#c3c2b7", "#ffffff"
SERIES_1 = "#2a78d6"
# Drift labels from classification/rules.py, in its taxonomy order.
DRIFT_TYPES = [r.drift_type for r in rules.TAXONOMY.values()]
DRIFT_NAME = {r.drift_type: r.display_name for r in rules.TAXONOMY.values()}
CATEGORY_COLOR = {rules.CAPABILITY_SHIFT: "#2a78d6", rules.SAFETY_SHIFT: "#eb6834",
                  rules.FORMAT_DRIFT: "#1baf7a", rules.INFRA_CHANGE: "#eda100"}
# Urgency levels from remediation/mapper.py, plus two dashboard-only states.
_X, _BANG, _DASH, _I, _OK = ("M6.5 6.5l7 7M13.5 6.5l-7 7", "M10 5.5v5.5M10 14.2v.3", "M5.5 10h9",
                             "M10 9v5M10 6v.3", "M5 10.5l3.2 3.2L15 7")
STATUS = {
    mapper.ACT_NOW: ("#d03b3b", "Act now", _X),
    mapper.ACT: ("#d03b3b", "Act", _X),
    mapper.INVESTIGATE: ("#ec835a", "Investigate", _BANG),
    mapper.WATCH: ("#d99400", "Watch", _BANG),
    mapper.NOTE: ("#898781", "Note", _DASH),
    "context": ("#898781", "Context only", _I),
    "clear": ("#0ca30c", "Clear", _OK),
}
SEQ_SCALE = [[0.0, "#cde2fb"], [0.17, "#9ec5f4"], [0.33, "#6da7ec"], [0.5, "#3987e5"],
             [0.67, "#256abf"], [0.83, "#184f95"], [1.0, "#0d366b"]]
FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
PLOT_CONFIG = {"displayModeBar": False, "responsive": True}

st.set_page_config(page_title="DriftGuard", page_icon=":material/monitoring:", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(f"""
<style>
  html, body, [class*="css"] {{ font-family: {FONT}; }}
  .block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1480px; }}
  #MainMenu, footer, [data-testid="stDecoration"] {{ visibility: hidden; height: 0; }}
  [data-testid="stSidebar"] {{ background: #ffffff; border-right: 1px solid rgba(11,11,11,.08); }}
  [data-testid="stSidebar"] .block-container {{ padding-top: 1.2rem; }}
  h1, h2, h3, h4 {{ letter-spacing: -0.01em; }}
  [data-testid="stVerticalBlockBorderWrapper"] {{ background: {SURFACE}; border-radius: 10px;
      border-color: rgba(11,11,11,.09) !important; }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid rgba(11,11,11,.09); }}
  .stTabs [data-baseweb="tab"] {{ padding: 10px 16px; font-weight: 500; color: {INK_2}; }}
  .stTabs [aria-selected="true"] {{ color: {INK}; }}

  .dg-head {{ display:flex; align-items:center; justify-content:space-between; gap:24px; flex-wrap:wrap;
      margin-bottom: 6px; }}
  .dg-brand {{ display:flex; align-items:center; gap:12px; }}
  .dg-brand h1 {{ font-size: 1.45rem; font-weight: 650; margin: 0; padding: 0; color:{INK}; }}
  .dg-brand p {{ margin: 0; color: {INK_2}; font-size: .88rem; }}
  .dg-meta {{ display:flex; gap:22px; flex-wrap:wrap; }}
  .dg-meta div {{ font-size:.78rem; color:{MUTED}; text-transform:uppercase; letter-spacing:.04em; }}
  .dg-meta b {{ display:block; font-size:.9rem; color:{INK}; text-transform:none; letter-spacing:0;
      font-weight:550; margin-top:2px; }}

  .dg-banner {{ display:flex; align-items:center; gap:14px; padding:14px 18px; border-radius:10px;
      background:{SURFACE}; border:1px solid rgba(11,11,11,.09); margin: 8px 0 18px; }}
  .dg-banner .t {{ font-weight:600; color:{INK}; font-size:1rem; }}
  .dg-banner .s {{ color:{INK_2}; font-size:.88rem; }}

  .dg-kpis {{ display:grid; grid-template-columns: repeat(5, minmax(0,1fr)); gap:12px; margin-bottom:18px; }}
  @media (max-width: 1100px) {{ .dg-kpis {{ grid-template-columns: repeat(2, minmax(0,1fr)); }} }}
  .dg-kpi {{ background:{SURFACE}; border:1px solid rgba(11,11,11,.09); border-radius:10px; padding:14px 16px; }}
  .dg-kpi .l {{ font-size:.8rem; color:{INK_2}; }}
  .dg-kpi .v {{ font-size:1.65rem; font-weight:600; color:{INK}; margin-top:4px; line-height:1.15; }}
  .dg-kpi .d {{ font-size:.78rem; color:{MUTED}; margin-top:4px; }}
  .dg-up-bad {{ color:#c02f2f !important; }} .dg-down-good {{ color:#006300 !important; }}

  .dg-pill {{ display:inline-flex; align-items:center; gap:6px; font-size:.78rem; font-weight:550;
      padding:3px 9px 3px 6px; border-radius:999px; background:#f4f4f1; color:{INK}; white-space:nowrap; }}
  .dg-cat {{ display:inline-flex; align-items:center; gap:7px; font-size:.78rem; color:{INK_2}; }}
  .dg-cat i {{ width:9px; height:9px; border-radius:3px; display:inline-block; }}
  .dg-sig-title {{ font-weight:600; font-size:1rem; color:{INK}; margin:6px 0 2px; }}
  .dg-sig-row {{ display:flex; justify-content:space-between; align-items:center; gap:8px; }}
  .dg-nums {{ display:flex; gap:22px; margin-top:8px; }}
  .dg-nums div {{ font-size:.75rem; color:{MUTED}; }}
  .dg-nums b {{ display:block; font-size:1.15rem; color:{INK}; font-weight:600; }}
  .dg-foot {{ font-size:.76rem; color:{MUTED}; margin-top:2px; }}

  .dg-section {{ font-size:1.02rem; font-weight:600; color:{INK}; margin: 18px 0 2px; }}
  .dg-sub {{ font-size:.85rem; color:{INK_2}; margin: 0 0 10px; }}

  .dg-shift {{ background:{SURFACE}; border:1px solid rgba(11,11,11,.09); border-left-width:4px;
      border-radius:10px; padding:12px 16px; margin-bottom:10px; }}
  .dg-shift .top {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; }}
  .dg-shift .what {{ font-weight:600; color:{INK}; }}
  .dg-shift .when {{ color:{MUTED}; font-size:.8rem; margin-left:auto; }}
  .dg-shift .num {{ color:{INK_2}; font-size:.86rem; margin-top:4px; }}
  .dg-shift .act {{ color:{INK}; font-size:.86rem; margin-top:6px; }}
  .dg-shift .act span {{ color:{MUTED}; }}

  .dg-catgrid {{ display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:12px; margin: 6px 0 14px; }}
  .dg-catgrid .c {{ background:{SURFACE}; border:1px solid rgba(11,11,11,.09); border-radius:10px; padding:12px 14px; }}
  .dg-catgrid .n {{ font-size:1.4rem; font-weight:600; color:{INK}; margin-top:4px; }}
  .dg-catgrid .x {{ font-size:.76rem; color:{MUTED}; }}
  .dg-empty {{ color:{INK_2}; font-size:.9rem; padding: 18px; text-align:center; background:{SURFACE};
      border:1px dashed rgba(11,11,11,.15); border-radius:10px; }}
  .dg-side-h {{ font-size:.72rem; font-weight:600; color:{MUTED}; text-transform:uppercase;
      letter-spacing:.06em; margin: 14px 0 4px; }}
  .dg-kv {{ font-size:.82rem; color:{INK_2}; line-height:1.7; }}
  .dg-kv b {{ color:{INK}; font-weight:550; }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- helpers

def esc(x) -> str:
    return html.escape(str(x))


def status_icon(level: str, size: int = 18) -> str:
    color, _, path = STATUS[level]
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 20 20" aria-hidden="true">'
            f'<circle cx="10" cy="10" r="9" fill="{color}"/>'
            f'<path d="{path}" stroke="#fff" stroke-width="2.2" fill="none" stroke-linecap="round" '
            f'stroke-linejoin="round"/></svg>')


def status_pill(level: str) -> str:
    return f'<span class="dg-pill">{status_icon(level, 14)}{STATUS[level][1]}</span>'


def cat_chip(drift_type) -> str:
    """Chip for a rules.py drift label; grey 'Outside taxonomy' for signals rules.py never labels."""
    if not drift_type:
        return f'<span class="dg-cat"><i style="background:{MUTED}"></i>Outside taxonomy</span>'
    return (f'<span class="dg-cat"><i style="background:{CATEGORY_COLOR[drift_type]}"></i>'
            f'{esc(DRIFT_NAME[drift_type])}</span>')


def signal_drift_type(key: str):
    metric = SIGNALS[key].metric
    return rules.TAXONOMY[metric].drift_type if metric else None


def fmt_metric(metric: str, v) -> str:
    """A before/after value in the metric's own units, as rules.py reports it."""
    if v is None or pd.isna(v):
        return "–"
    if metric.startswith("latency"):
        return f"{v:,.0f} ms"
    return f"{v:.3f}"


def pct(x, digits=1) -> str:
    return "–" if x is None or pd.isna(x) else f"{x * 100:.{digits}f}%"


def pp(x) -> str:
    return "–" if x is None or pd.isna(x) else f"{x * 100:+.1f} pp"


def fmt_ts(ts) -> str:
    return "–" if ts is None or pd.isna(ts) else pd.Timestamp(ts).strftime("%d %b %Y, %H:%M UTC")


def base_layout(fig: go.Figure, height: int, **kw) -> go.Figure:
    kw.setdefault("margin", dict(l=8, r=12, t=10, b=8))
    kw.setdefault("showlegend", False)
    fig.update_layout(
        height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=12, color=INK_2),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="rgba(11,11,11,.15)", font=dict(family=FONT, color=INK, size=12)),
        **kw,
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, ticks="outside", tickcolor=AXIS, ticklen=4,
                     tickfont=dict(color=MUTED), zeroline=False)
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, zeroline=False, linecolor=AXIS, tickfont=dict(color=MUTED))
    return fig


def run_label(row) -> str:
    rid = str(row["run_id"])
    day = pd.Timestamp(row["first"]).strftime("%d %b")
    if rid.startswith("live_drift_compare_"):
        kind = rid.split("_")[3] if len(rid.split("_")) > 3 else "trial"
        name = f"{kind.capitalize()} trial"
    elif rid.startswith("live_replay"):
        name = "Live replay round"
    elif rid == "(no run id)":
        name = "Untagged calls"
    else:
        name = rid[:18]
    return f"{name} · {day} · {row['calls']:,} calls"


# ----------------------------------------------------------------------------- data (cached)

def db_version() -> float:
    try:
        return DB_PATH.stat().st_mtime
    except FileNotFoundError:
        return 0.0


@st.cache_data(show_spinner=False)
def cached_calls(path: str, version: float) -> pd.DataFrame:
    return D.load_calls(path)


@st.cache_resource(show_spinner="Running Page-Hinkley and PELT, then classifying…")
def cached_analysis(path: str, version: float, target: str, run_id: str | None):
    calls = cached_calls(path, version)
    scope = calls[calls["target"] == target]
    if run_id is not None:
        scope = scope[scope["eval_run_id"].fillna("(no run id)") == run_id]
    scope = scope.reset_index(drop=True)
    settings = D.load_settings()
    streams = D.build_streams(scope, settings.baseline_window)
    results = D.analyse(streams, settings)
    prov, model = target.split(" · ", 1)
    analysis = D.classify_detections(results, prov, model, settings, sigmas=D.load_sigmas())
    shifts = D.shift_table(analysis)
    pos = pd.Series(np.arange(1, len(scope) + 1), index=scope["seq_id"].to_numpy())
    if len(shifts):
        shifts["call_no"] = shifts["seq_id"].map(pos).astype(int)
    else:
        shifts["call_no"] = pd.Series(dtype=int)
    return scope, results, analysis, shifts, pos, settings


# ----------------------------------------------------------------------------- sidebar

if not DB_PATH.exists():
    st.error(f"No database found at `{DB_PATH}`. Run the eval suite first, or set DRIFTGUARD_DB.")
    st.stop()

calls_all = cached_calls(str(DB_PATH), db_version())
if calls_all.empty:
    st.info("The database has no logged calls yet.")
    st.stop()

with st.sidebar:
    st.markdown(f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">'
                f'<svg width="26" height="26" viewBox="0 0 26 26"><rect width="26" height="26" rx="7" fill="{INK}"/>'
                f'<path d="M5 15h4l2.5-6 3 10 2.5-6H21" stroke="#fff" stroke-width="2" fill="none" '
                f'stroke-linecap="round" stroke-linejoin="round"/></svg>'
                f'<b style="font-size:1.05rem;color:{INK}">DriftGuard</b></div>', unsafe_allow_html=True)

    st.markdown('<div class="dg-side-h">Monitored target</div>', unsafe_allow_html=True)
    targets = calls_all.groupby("target")["seq_id"].size().sort_values(ascending=False)
    target = st.selectbox("Target", targets.index.tolist(), label_visibility="collapsed",
                          format_func=lambda t: t.split(" · ", 1)[1])
    st.caption(f"{target.split(' · ', 1)[0]} · {targets[target]:,} calls logged")

    st.markdown('<div class="dg-side-h">Scope</div>', unsafe_allow_html=True)
    catalog = D.run_catalog(calls_all[calls_all["target"] == target])
    scope_opts = [None] + catalog["run_id"].tolist()
    labels = {None: f"All runs (combined history) · {targets[target]:,} calls"}
    labels.update({r["run_id"]: run_label(r) for _, r in catalog.iterrows()})
    default = D.default_run(calls_all[calls_all["target"] == target])
    run_id = st.selectbox("Scope", scope_opts, format_func=lambda r: labels[r], label_visibility="collapsed",
                          index=scope_opts.index(default) if default in scope_opts else 0, key=f"scope_{target}")
    st.caption("Opens on the latest single trial with enough data. Combined history joins separate "
               "experiments, so its shifts can be the boundaries between them.")

    st.markdown('<div class="dg-side-h">Chart axis</div>', unsafe_allow_html=True)
    axis = st.segmented_control("Axis", ["Call sequence", "Time"], default="Call sequence",
                                label_visibility="collapsed") or "Call sequence"
    st.markdown('<div class="dg-side-h">Rolling window</div>', unsafe_allow_html=True)
    window = st.slider("Rolling window (calls)", 20, 300, 50, step=10, label_visibility="collapsed",
                       help="Smoothing for the rate charts only. Detectors always see every call.")

    s0 = D.load_settings()
    st.markdown('<div class="dg-side-h">Detectors</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="dg-kv"><b>Online</b> · Page-Hinkley, λ = {s0.ph_lambda:g}, δ = {s0.ph_delta:g}, '
        f'warm-up {s0.ph_min_instances} calls<br><b>Retrospective</b> · PELT ({s0.pelt.cost_model}, '
        f'penalty × {s0.pelt.penalty_scale:g}) → permutation test ({s0.significance.n_bootstrap_samples:,}) '
        f'→ Benjamini-Hochberg, α = {s0.significance.alpha:g}<br><b>Classification</b> · rules.py labels, '
        f'incidents.py groups events within {int(D.DEFAULT_WINDOW.total_seconds() // 3600)} h, '
        f'mapper.py sets urgency</div>', unsafe_allow_html=True)
    st.write("")
    if st.button("Reload data", icon=":material/refresh:", width="stretch"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    st.caption(f"Database: `{DB_PATH.name}`")

scope, results, analysis, shifts, pos, settings = cached_analysis(str(DB_PATH), db_version(), target, run_id)
provider, model_id = target.split(" · ", 1)


def xs(stream: D.Stream, idx) -> np.ndarray | pd.Series:
    """x positions for stream indices on the shared axis (call # in scope, or time)."""
    idx = np.asarray(idx, dtype=int)
    if axis == "Time":
        return stream.timestamp.iloc[idx].to_numpy()
    return pos.loc[stream.seq_id[idx]].to_numpy()


X_TITLE = "Time (UTC)" if axis == "Time" else "Call # in scope"

# ----------------------------------------------------------------------------- header

levels = {k: D.signal_status(k, analysis) for k in results}
top = D.top_incident(analysis)
worst = analysis.plans[top].urgency if top is not None else "clear"
st.markdown(f"""
<div class="dg-head">
  <div class="dg-brand"><div><h1>Behavioral drift monitor</h1>
    <p>Black-box monitoring of a hosted LLM API from its responses alone</p></div></div>
  <div class="dg-meta">
    <div>Provider<b>{esc(provider)}</b></div>
    <div>Model<b>{esc(model_id)}</b></div>
    <div>Scope<b>{esc(' · '.join(labels[run_id].split(' · ')[:2]) if run_id else 'Combined history')}</b></div>
    <div>Last call<b>{fmt_ts(scope['timestamp'].max())}</b></div>
  </div>
</div>""", unsafe_allow_html=True)

tab_over, tab_map, tab_sig, tab_log, tab_calls = st.tabs(
    ["Overview", "Shift map", "Signals", "Shift log & review", "Call explorer"])

labelled = shifts[shifts["drift_type"].notna()] if len(shifts) else shifts
context = (shifts[~shifts["in_taxonomy"] & ((shifts["detector"] == "Page-Hinkley") | (shifts["tier"] == rules.CONFIRMED))]
           if len(shifts) else shifts)


def recent_vs_base(values: np.ndarray, base_n: int, recent_n: int = 200):
    if len(values) == 0:
        return np.nan, np.nan
    return float(values[:base_n].mean()), float(values[-recent_n:].mean())


# ============================================================================= OVERVIEW
with tab_over:
    if top is None:
        banner_t = "No drift detected in this scope"
        parts = [f"{sum(SIGNALS[k].metric is not None for k in results)} classified signals monitored "
                 f"across {len(scope):,} calls."]
    else:
        plan = analysis.plans[top]
        banner_t = f"{STATUS[worst][1]} · {plan.headline}"
        parts = [plan.steps[0]]
    ctx = sorted({SIGNALS[k].label for k in context["signal"]}) if len(context) else []
    if ctx:
        parts.append(f"Also moved, outside the taxonomy and shown for context only: {', '.join(ctx)}.")
    if run_id is None:
        parts.append("This is the combined history of several experiments.")
    banner_s = " ".join(parts)
    st.markdown(f'<div class="dg-banner" style="border-left:4px solid {STATUS[worst][0]}">'
                f'{status_icon(worst, 26)}<div><div class="t">{esc(banner_t)}</div><div class="s">{esc(banner_s)}</div>'
                f'</div></div>', unsafe_allow_html=True)

    # KPI tiles
    ok_rate = float((scope["outcome"] == "ok").mean()) if len(scope) else np.nan
    acc = results.get("accuracy_error")
    acc_b, acc_r = recent_vs_base(acc.stream.values, settings.baseline_window) if acc else (np.nan, np.nan)
    lat = scope.loc[scope["outcome"] == "ok", "latency_ms"].dropna().to_numpy()
    lat_b = float(np.percentile(lat[:settings.baseline_window], 95)) if len(lat) >= 20 else np.nan
    lat_r = float(np.percentile(lat[-200:], 95)) if len(lat) >= 20 else np.nan

    def delta_html(diff, unit_fmt, up_is_bad=True):
        if pd.isna(diff) or abs(diff) < 1e-12:
            return "no change vs baseline"
        cls = "dg-up-bad" if (diff > 0) == up_is_bad else "dg-down-good"
        arrow = "▲" if diff > 0 else "▼"
        return f'<span class="{cls}">{arrow} {unit_fmt(abs(diff))}</span> vs baseline'

    kpis = [
        ("Calls in scope", f"{len(scope):,}", f"{scope['timestamp'].min():%d %b} – {scope['timestamp'].max():%d %b %Y}"),
        ("Call success rate", pct(ok_rate),
         f"{int((~scope['outcome'].isin(['ok', 'rate_limited'])).sum()):,} errors · "
         f"{int((scope['outcome'] == 'rate_limited').sum()):,} rate-limited"),
        ("Answer error rate · last 200", pct(acc_r), delta_html(acc_r - acc_b, lambda v: f"{v * 100:.1f} pp")),
        ("p95 latency · last 200", "–" if pd.isna(lat_r) else f"{lat_r / 1000:.2f} s",
         delta_html(lat_r - lat_b, lambda v: f"{v / 1000:.2f} s")),
        ("Incidents", f"{len(analysis.incidents)}",
         f"{len(labelled)} classified event{'s' if len(labelled) != 1 else ''}"
         + ("" if top is None else f" · top urgency: {STATUS[worst][1].lower()}")),
    ]
    st.markdown('<div class="dg-kpis">' + "".join(
        f'<div class="dg-kpi"><div class="l">{esc(l)}</div><div class="v">{v}</div><div class="d">{d}</div></div>'
        for l, v, d in kpis) + "</div>", unsafe_allow_html=True)

    st.markdown('<div class="dg-section">Signal health</div>'
                f'<div class="dg-sub">Each signal is a per-call rate of bad outcomes. Baseline = first '
                f'{settings.baseline_window} calls; recent = last 200.</div>', unsafe_allow_html=True)
    keys = list(results.keys())
    if not keys:
        st.markdown('<div class="dg-empty">Not enough calls in this scope to evaluate any signal '
                    f'(each needs at least {D.MIN_SIGNAL_CALLS}).</div>', unsafe_allow_html=True)
    for row_start in range(0, len(keys), 3):
        cols = st.columns(3)
        for col, key in zip(cols, keys[row_start:row_start + 3]):
            r, sig = results[key], SIGNALS[key]
            b, rc = recent_vs_base(r.stream.values, settings.baseline_window)
            n_conf = sum(1 for t in r.confirmed if t["significant"])
            with col.container(border=True):
                st.markdown(
                    f'<div class="dg-sig-row">{cat_chip(signal_drift_type(key))}{status_pill(levels[key])}</div>'
                    f'<div class="dg-sig-title">{esc(sig.label)}</div>'
                    f'<div class="dg-nums"><div>Recent<b>{pct(rc)}</b></div><div>Baseline<b>{pct(b)}</b></div>'
                    f'<div>Calls<b>{len(r.stream.values):,}</b></div></div>', unsafe_allow_html=True)
                roll = D.rolling_rate(r.stream.values, window)
                fig = go.Figure(go.Scatter(x=np.arange(len(roll)), y=roll * 100, mode="lines",
                                           line=dict(color=SERIES_1, width=2), fill="tozeroy",
                                           fillcolor="rgba(42,120,214,0.10)",
                                           hovertemplate="%{y:.1f}%<extra></extra>"))
                for t in r.confirmed:
                    if t["significant"]:
                        fig.add_vline(x=t["index"], line=dict(color=INK, width=1))
                base_layout(fig, 92, margin=dict(l=0, r=0, t=4, b=0))
                fig.update_xaxes(visible=False)
                fig.update_yaxes(visible=False, rangemode="tozero")
                st.plotly_chart(fig, config=PLOT_CONFIG, key=f"spark_{key}")
                st.markdown(f'<div class="dg-foot">{n_conf} confirmed shift{"s" if n_conf != 1 else ""} · '
                            f'{len(r.alerts)} online alert{"s" if len(r.alerts) != 1 else ""}'
                            f'{" · " + esc(r.stream.note) if r.stream.note else ""}'
                            f'{" · " + esc(sig.context_note) if sig.context_note else ""}</div>', unsafe_allow_html=True)

    st.markdown('<div class="dg-section">Incidents</div>'
                '<div class="dg-sub">Events that moved together, grouped by incidents.py; urgency and plan from '
                'remediation/mapper.py.</div>', unsafe_allow_html=True)
    if not analysis.incidents:
        st.markdown('<div class="dg-empty">No incident in this scope.</div>', unsafe_allow_html=True)
    order = sorted(range(len(analysis.incidents)), key=lambda k: analysis.incidents[k].start, reverse=True)
    for k in order[:4]:
        inc, plan = analysis.incidents[k], analysis.plans[k]
        chips = "".join(cat_chip(c) for c in inc.channels)
        meanings = "".join(f"<li>{esc(m)}</li>" for m in plan.meanings)
        hints = "".join(f"<li>{esc(h)}</li>" for h in plan.possible_explanations)
        steps = "".join(f"<li>{esc(x)}</li>" for x in plan.steps[:4])
        ev = D.incident_evidence(DB_PATH, inc)
        ev_html = ("".join(f"<li><b>{esc(m)}</b>: {esc(e.summary)}</li>" for m, e in ev.items()) if ev else
                   "<li>Flip-set evidence needs per-run eval data (<code>metric_series</code>); this scope has none.</li>")
        st.markdown(
            f'<div class="dg-shift" style="border-left-color:{STATUS[plan.urgency][0]}"><div class="top">'
            f'{status_pill(plan.urgency)}{chips}<span class="when">{fmt_ts(inc.start)}</span></div>'
            f'<div class="num" style="color:{INK}">{esc(plan.headline)}</div>'
            f'<div class="act"><span>What it means</span><ul style="margin:2px 0 4px 18px">{meanings}</ul></div>'
            + (f'<div class="act"><span>Possible explanations</span><ul style="margin:2px 0 4px 18px">{hints}</ul></div>'
               if hints else "")
            + f'<div class="act"><span>Plan</span><ol style="margin:2px 0 4px 18px">{steps}</ol></div>'
            f'<div class="act"><span>Evidence</span><ul style="margin:2px 0 0 18px">{ev_html}</ul></div></div>',
            unsafe_allow_html=True)


# ============================================================================= SHIFT MAP
with tab_map:
    st.markdown('<div class="dg-section">Shift map</div>'
                '<div class="dg-sub">One lane per signal. Shading is the bad-outcome rate of each regime PELT '
                'found (darker = worse); a new shade starts at every confirmed shift. ◆ marks a confirmed shift, '
                '▼ marks where the online detector raised its alert.</div>', unsafe_allow_html=True)
    if not results:
        st.markdown('<div class="dg-empty">Nothing to map in this scope.</div>', unsafe_allow_html=True)
    else:
        keys = list(results.keys())
        fig = go.Figure()
        for lane, key in enumerate(keys):
            r = results[key]
            st_ = r.stream
            n = len(st_.values)
            for a, b, mean in r.segments:
                x0, x1 = xs(st_, [a])[0], xs(st_, [b - 1])[0]
                if axis == "Time":
                    base, length = pd.Timestamp(x0), (pd.Timestamp(x1) - pd.Timestamp(x0)).total_seconds() * 1000
                else:
                    base, length = x0, max(1, x1 - x0 + 1)
                fig.add_trace(go.Bar(
                    orientation="h", y=[lane], base=[base], x=[length], width=0.56,
                    marker=dict(color=[mean], colorscale=SEQ_SCALE, cmin=0, cmax=1, line=dict(width=2, color=SURFACE)),
                    text=[pct(mean, 0)], textposition="inside", insidetextanchor="middle",
                    textfont=dict(color="#ffffff" if mean > 0.45 else INK, size=11),
                    hovertemplate=(f"<b>{esc(SIGNALS[key].label)}</b><br>Regime rate: {pct(mean)}<br>"
                                   f"Calls {a + 1:,}–{b:,} of this signal ({b - a:,})<extra></extra>"),
                ))
            conf = [t for t in r.confirmed if t["significant"]]
            if conf:
                idx = [t["index"] for t in conf]
                fig.add_trace(go.Scatter(
                    x=xs(st_, idx), y=[lane] * len(idx), mode="markers",
                    marker=dict(symbol="diamond", size=13, color=INK, line=dict(color=SURFACE, width=2)),
                    customdata=[[pct(t["pre_mean"]), pct(t["post_mean"]), f"{t['p_value_corrected']:.4f}",
                                 fmt_ts(st_.timestamp.iloc[t["index"]])] for t in conf],
                    hovertemplate=("<b>Confirmed shift</b><br>%{customdata[0]} → %{customdata[1]}<br>"
                                   "Adj. p = %{customdata[2]}<br>%{customdata[3]}<extra></extra>"),
                ))
            if r.alerts:
                fig.add_trace(go.Scatter(
                    x=xs(st_, r.alerts), y=[lane - 0.42] * len(r.alerts), mode="markers",
                    marker=dict(symbol="triangle-down", size=11, color=SURFACE, line=dict(color=INK, width=2)),
                    customdata=[[fmt_ts(st_.timestamp.iloc[i])] for i in r.alerts],
                    hovertemplate="<b>Page-Hinkley alert observed</b><br>%{customdata[0]}<extra></extra>",
                ))
        base_layout(fig, 90 + 74 * len(keys), barmode="overlay", bargap=0.0)
        fig.update_yaxes(tickvals=list(range(len(keys))), autorange="reversed", showgrid=False,
                         ticktext=[f"<b>{SIGNALS[k].label}</b>  <span style='color:{MUTED}'>"
                                   f"{DRIFT_NAME[signal_drift_type(k)] if signal_drift_type(k) else 'context only'}</span>"
                                   for k in keys], tickfont=dict(color=INK, size=12), range=[len(keys) - 0.4, -0.75])
        fig.update_xaxes(title=dict(text=X_TITLE, font=dict(color=MUTED, size=11)), showgrid=True, gridcolor=GRID)
        with st.container(border=True):
            st.plotly_chart(fig, config=PLOT_CONFIG, key="shift_map")
            st.markdown(
                f'<div style="display:flex;gap:22px;align-items:center;flex-wrap:wrap;font-size:.8rem;color:{INK_2};'
                f'padding:0 4px 4px">'
                f'<span>◆ Confirmed shift (PELT + permutation test, BH-adjusted)</span>'
                f'<span>▼ Online alert (Page-Hinkley)</span>'
                f'<span style="display:flex;align-items:center;gap:8px">Regime rate 0%'
                f'<span style="width:120px;height:9px;border-radius:3px;background:linear-gradient(90deg,'
                f'#cde2fb,#6da7ec,#256abf,#0d366b)"></span>100%</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="dg-section">Events by drift type</div>'
                '<div class="dg-sub">Adverse events labelled by rules.py, counted by evidence tier.</div>',
                unsafe_allow_html=True)
    cells = []
    adverse = labelled[labelled["adverse"] == True] if len(labelled) else labelled  # noqa: E712
    for dt in DRIFT_TYPES:
        sub = adverse[adverse["drift_type"] == dt] if len(adverse) else adverse
        n_c = int(sub["tier"].isin([rules.CONFIRMED, rules.PROVISIONAL]).sum()) if len(sub) else 0
        n_o = int((sub["tier"] == rules.EARLY_WARNING).sum()) if len(sub) else 0
        n_u = int((sub["tier"] == rules.UNCONFIRMED).sum()) if len(sub) else 0
        monitored = [SIGNALS[k].label for k in results if signal_drift_type(k) == dt]
        note = ", ".join(monitored) if monitored else "no data in this scope"
        cells.append(f'<div class="c">{cat_chip(dt)}<div class="n">{n_c}</div>'
                     f'<div class="x">confirmed · {n_o} early warning{"s" if n_o != 1 else ""} · {n_u} unconfirmed'
                     f'<br>{esc(note)}</div></div>')
    st.markdown('<div class="dg-catgrid">' + "".join(cells) + "</div>", unsafe_allow_html=True)

    st.markdown('<div class="dg-section">Every detection, as rules.py reads it</div>', unsafe_allow_html=True)
    if len(shifts) == 0:
        st.markdown('<div class="dg-empty">No detections in this scope.</div>', unsafe_allow_html=True)
    else:
        view = shifts.assign(
            when=shifts["timestamp"].dt.tz_convert(None),
            label=shifts["label"].fillna("—"),
            before_s=[fmt_metric(m, v) for m, v in zip(shifts["metric"], shifts["before"])],
            after_s=[fmt_metric(m, v) for m, v in zip(shifts["metric"], shifts["after"])],
            tier=shifts["tier"].str.replace("_", " "),
            urgency=shifts["urgency"].fillna("—"),
        )[["when", "call_no", "signal_label", "label", "detector", "tier", "before_s", "after_s", "urgency", "reason"]]
        st.dataframe(view, hide_index=True, column_config={
            "when": st.column_config.DatetimeColumn("Change at (UTC)", format="D MMM YYYY, HH:mm"),
            "call_no": st.column_config.NumberColumn("Call #", format="%d"),
            "signal_label": "Signal", "label": "Drift type", "detector": "Detector", "tier": "Evidence",
            "before_s": "Before", "after_s": "After", "urgency": "Urgency",
            "reason": st.column_config.TextColumn("rules.py reading", width="large"),
        })
        st.caption("Before/after are in each metric's own units (accuracy, not error rate). For a PELT change they "
                   f"come from the two regimes either side of it; for an online alert from the {D.COMPARE_WINDOW} calls "
                   "before and after the alert, which is when the detector noticed, not when the change began.")


# ============================================================================= SIGNALS
with tab_sig:
    if not results:
        st.markdown('<div class="dg-empty">No signal has enough calls in this scope.</div>', unsafe_allow_html=True)
    else:
        key = st.segmented_control("Signal", list(results.keys()), default=list(results.keys())[0],
                                   format_func=lambda k: SIGNALS[k].label, label_visibility="collapsed") \
            or list(results.keys())[0]
        r, sig = results[key], SIGNALS[key]
        st.markdown(f'<div class="dg-sig-row" style="justify-content:flex-start;gap:14px;margin-top:8px">'
                    f'{cat_chip(signal_drift_type(key))}{status_pill(levels[key])}</div>'
                    f'<div class="dg-sub" style="margin-top:6px">A bad outcome on this signal means: '
                    f'{esc(sig.meaning)}{" (" + esc(r.stream.note) + ")" if r.stream.note else ""}</div>',
                    unsafe_allow_html=True)

        v = r.stream.values
        roll = D.rolling_rate(v, window)
        x_all = xs(r.stream, np.arange(len(v)))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_all, y=roll * 100, mode="lines", name=f"Rolling rate ({window} calls)",
                                 line=dict(color=SERIES_1, width=2), fill="tozeroy", fillcolor="rgba(42,120,214,0.08)",
                                 hovertemplate="%{y:.1f}%<extra>rolling rate</extra>"))
        seg_x, seg_y = [], []
        for a, b, mean in r.segments:
            seg_x += [*xs(r.stream, [a, b - 1]), None]
            seg_y += [mean * 100, mean * 100, None]
        fig.add_trace(go.Scatter(x=seg_x, y=seg_y, mode="lines", name="Regime mean (PELT)",
                                 line=dict(color=INK, width=2), hovertemplate="%{y:.1f}%<extra>regime mean</extra>"))
        if r.alerts:
            fig.add_trace(go.Scatter(x=xs(r.stream, r.alerts), y=roll[r.alerts] * 100, mode="markers",
                                     name="Page-Hinkley alert",
                                     marker=dict(symbol="triangle-down", size=12, color=SURFACE, line=dict(color=INK, width=2)),
                                     hovertemplate="Alert observed · %{y:.1f}%<extra></extra>"))
        for t in r.confirmed:
            if t["significant"]:
                fig.add_vline(x=xs(r.stream, [t["index"]])[0], line=dict(color=MUTED, width=1))
        base_layout(fig, 360, showlegend=True, hovermode="x unified",
                    legend=dict(orientation="h", y=1.08, x=0, font=dict(color=INK_2)))
        fig.update_yaxes(title=dict(text="Bad-outcome rate (%)", font=dict(color=MUTED, size=11)), rangemode="tozero")
        fig.update_xaxes(title=dict(text=X_TITLE, font=dict(color=MUTED, size=11)))
        with st.container(border=True):
            st.plotly_chart(fig, config=PLOT_CONFIG, key="sig_main")

        tested = pd.DataFrame(r.confirmed)
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown('<div class="dg-section">Retrospective candidates</div>'
                        '<div class="dg-sub">Every change point PELT proposed, and whether the permutation test '
                        'kept it after Benjamini-Hochberg correction.</div>', unsafe_allow_html=True)
            if tested.empty:
                st.markdown('<div class="dg-empty">PELT proposed no change point.</div>', unsafe_allow_html=True)
            else:
                tv = tested.assign(call_no=pos.loc[r.stream.seq_id[tested["index"].to_numpy()]].to_numpy(),
                                   pre=tested["pre_mean"] * 100, post=tested["post_mean"] * 100,
                                   tier=np.where(tested["significant"], "confirmed", "unconfirmed"))
                st.dataframe(tv[["call_no", "pre", "post", "p_value_raw", "p_value_corrected", "tier"]],
                             hide_index=True, column_config={
                                 "call_no": st.column_config.NumberColumn("Call #", format="%d"),
                                 "pre": st.column_config.NumberColumn("Before", format="%.1f%%"),
                                 "post": st.column_config.NumberColumn("After", format="%.1f%%"),
                                 "p_value_raw": st.column_config.NumberColumn("Raw p", format="%.4f"),
                                 "p_value_corrected": st.column_config.NumberColumn("Adj. p", format="%.4f"),
                                 "tier": "Evidence (rules.py)",
                             })
        with c2:
            pc = r.pelt_config
            st.markdown('<div class="dg-section">Run details</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="dg-kv">Calls on this signal: <b>{len(v):,}</b><br>'
                f'Overall rate: <b>{pct(v.mean())}</b><br>'
                f'PELT minimum segment: <b>{pc.min_segment_length} calls</b>'
                f'{" (capped for this short scope; tuned value " + str(settings.pelt.min_segment_length) + ")" if pc.min_segment_length != settings.pelt.min_segment_length else ""}<br>'
                f'Online alerts: <b>{len(r.alerts)}</b> (detector resets after each)</div>', unsafe_allow_html=True)

        if len(lat):
            st.markdown('<div class="dg-section">Response latency</div>'
                        '<div class="dg-sub">Every successful call; the line is the rolling median.</div>',
                        unsafe_allow_html=True)
            okc = scope[(scope["outcome"] == "ok") & scope["latency_ms"].notna()]
            lx = okc["timestamp"] if axis == "Time" else pos.loc[okc["seq_id"]].to_numpy()
            med = okc["latency_ms"].rolling(window, min_periods=1).median() / 1000
            lf = go.Figure()
            lf.add_trace(go.Scattergl(x=lx, y=okc["latency_ms"] / 1000, mode="markers",
                                      marker=dict(size=4, color="rgba(137,135,129,0.35)"),
                                      hovertemplate="%{y:.2f} s<extra>call</extra>"))
            lf.add_trace(go.Scatter(x=lx, y=med, mode="lines", line=dict(color=SERIES_1, width=2),
                                    hovertemplate="%{y:.2f} s<extra>rolling median</extra>"))
            base_layout(lf, 260)
            lf.update_yaxes(title=dict(text="Latency (s)", font=dict(color=MUTED, size=11)), rangemode="tozero",
                            range=[0, float(np.percentile(okc["latency_ms"], 99)) / 1000 * 1.1])
            lf.update_xaxes(title=dict(text=X_TITLE, font=dict(color=MUTED, size=11)))
            with st.container(border=True):
                st.plotly_chart(lf, config=PLOT_CONFIG, key="latency")


# ============================================================================= SHIFT LOG & REVIEW
with tab_log:
    st.markdown('<div class="dg-section">Recorded shifts</div>'
                '<div class="dg-sub">Events saved to the <code>changepoint_events</code> table. Labels and plans are '
                'written by remediation/mapper.apply_remediation, as for any stored event. Tick a row once someone '
                'has looked at it.</div>', unsafe_allow_html=True)
    to_record = [d for d in analysis.events
                 if d.classification.drift_type and d.classification.adverse and d.event.significant]
    n_new = len(to_record)
    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button(f"Record {n_new} open event{'s' if n_new != 1 else ''} from this scope",
                     type="primary", disabled=n_new == 0, icon=":material/save:"):
            added = D.record_shifts(DB_PATH, analysis)
            st.cache_data.clear()
            st.cache_resource.clear()
            st.toast(f"{added} new event{'s' if added != 1 else ''} recorded"
                     + ("" if added == n_new else f", {n_new - added} already in the log"))
    events = D.load_events(DB_PATH)
    if events.empty:
        st.markdown('<div class="dg-empty">No shifts recorded yet. Use the button above to save the ones '
                    'found in the current scope.</div>', unsafe_allow_html=True)
    else:
        ev = events.copy()
        ev["reviewed"] = ev["reviewed"].astype(bool)
        for c in ("pre_mean", "post_mean"):
            ev[c] = [fmt_metric(m, v) for m, v in zip(ev["metric_name"], ev[c])]
        ev["metric_name"] = ev["metric_name"].map(
            lambda m: rules.TAXONOMY[m].metric_label if m in rules.TAXONOMY else (SIGNALS[m].label if m in SIGNALS else m))
        ev["drift_type"] = ev["drift_type"].map(lambda d: DRIFT_NAME.get(d, d) if isinstance(d, str) else d)
        ev["detector"] = ev["detector"].map({"pelt": "PELT", "sequential": "Page-Hinkley"}).fillna(ev["detector"])
        cols = ["id", "reviewed", "changepoint_timestamp", "provider", "model_id", "metric_name", "drift_type",
                "detector", "pre_mean", "post_mean", "p_value_corrected", "remediation"]
        edited = st.data_editor(ev[cols], hide_index=True, disabled=[c for c in cols if c != "reviewed"],
                                key="events_editor", column_config={
                                    "id": None,
                                    "reviewed": st.column_config.CheckboxColumn("Reviewed"),
                                    "changepoint_timestamp": st.column_config.DatetimeColumn("Shift at (UTC)", format="D MMM YYYY, HH:mm"),
                                    "provider": "Provider", "model_id": "Model", "metric_name": "Metric",
                                    "drift_type": "Drift type", "detector": "Detector",
                                    "pre_mean": "Before", "post_mean": "After",
                                    "p_value_corrected": st.column_config.NumberColumn("Adj. p", format="%.4f"),
                                    "remediation": st.column_config.TextColumn("Plan (mapper.py)", width="large"),
                                })
        changed = edited[edited["reviewed"] != ev["reviewed"]]
        open_n = int((~edited["reviewed"]).sum())
        st.caption(f"{len(edited)} recorded · {open_n} awaiting review")
        if st.button("Save review status", disabled=changed.empty, icon=":material/task_alt:"):
            for val in (True, False):
                ids = changed.loc[changed["reviewed"] == val, "id"].tolist()
                if ids:
                    D.set_reviewed(DB_PATH, ids, val)
            st.toast(f"Updated {len(changed)} shift{'s' if len(changed) != 1 else ''}")
            st.rerun()


# ============================================================================= CALL EXPLORER
with tab_calls:
    st.markdown('<div class="dg-section">Call explorer</div>'
                '<div class="dg-sub">Inspect the actual prompts and responses around a shift.</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns([3, 1, 1])
    shift_opts = [None] + list(shifts.index)
    with c1:
        focus = st.selectbox(
            "Centre on", shift_opts,
            format_func=lambda i: "All calls in scope" if i is None else
            f"{shifts.loc[i, 'signal_label']} · {shifts.loc[i, 'detector']} · "
            f"{shifts.loc[i, 'tier'].replace('_', ' ')} · call #{shifts.loc[i, 'call_no']:,}")
    with c2:
        span = st.number_input("Calls either side", 5, 500, 30, step=5, disabled=focus is None)
    with c3:
        outcome = st.selectbox("Outcome", ["All", "ok", "error", "rate_limited"])

    view = scope.assign(call_no=np.arange(1, len(scope) + 1))
    if focus is not None:
        centre = int(shifts.loc[focus, "call_no"])
        view = view[(view["call_no"] >= centre - span) & (view["call_no"] <= centre + span)]
    if outcome != "All":
        view = view[view["outcome"] == outcome]
    view = view.assign(
        max_tokens=view["request_params_json"].map(lambda s: json.loads(s or "{}").get("max_tokens")),
        result=np.select([view["eval_score"] == 1.0, view["eval_score"] == 0.0], ["Pass", "Fail"], "–"),
        latency_s=view["latency_ms"] / 1000,
        when=view["timestamp"].dt.tz_convert(None),
    )
    table = view[["call_no", "when", "eval_task_type", "outcome", "result", "latency_s", "max_tokens", "prompt"]]
    ev_sel = st.dataframe(table, hide_index=True, on_select="rerun", selection_mode="single-row", height=380,
                          key="calls_table", column_config={
                              "call_no": st.column_config.NumberColumn("Call #", format="%d"),
                              "when": st.column_config.DatetimeColumn("Time (UTC)", format="D MMM, HH:mm:ss"),
                              "eval_task_type": "Task", "outcome": "Outcome", "result": "Score",
                              "latency_s": st.column_config.NumberColumn("Latency", format="%.2f s"),
                              "max_tokens": st.column_config.NumberColumn("max_tokens", format="%d"),
                              "prompt": st.column_config.TextColumn("Prompt", width="large"),
                          })
    st.caption(f"{len(table):,} calls shown. Select a row to read the full exchange.")
    rows = ev_sel.selection.rows if ev_sel and ev_sel.selection else []
    if rows:
        call = view.iloc[rows[0]]
        with st.container(border=True):
            st.markdown(f'<div class="dg-sig-row"><span class="dg-sig-title">Call #{int(call["call_no"]):,}</span>'
                        f'<span class="dg-foot">{fmt_ts(call["timestamp"])} · {esc(call["outcome"])} · '
                        f'{"–" if pd.isna(call["latency_ms"]) else f"{call["latency_ms"] / 1000:.2f} s"}</span></div>',
                        unsafe_allow_html=True)
            a, b = st.columns(2)
            with a:
                st.markdown("**Prompt**")
                st.code(call["prompt"] or "", language=None, wrap_lines=True)
                st.markdown("**Request parameters**")
                st.code(call["request_params_json"] or "{}", language="json")
            with b:
                st.markdown("**Response**")
                st.code(call["response_text"] or call["error_message"] or "(empty)", language=None, wrap_lines=True)
                st.markdown(f"**Score** · {call['result']}")
