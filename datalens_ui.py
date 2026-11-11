"""Shared UI helpers for DataLens AI's Streamlit multipage interface.

Design system: navy sidebar, light workspace, Inter type, one indigo accent.
Data helpers at the bottom of this file (read_uploaded_dataframe,
load_uploaded_datasets, record_analysis) are unchanged from the original.
"""
from __future__ import annotations

from datetime import datetime
from html import escape
from io import BytesIO
from typing import Any

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
ACCENT = "#4F46E5"
CHART_COLORS = ["#4F46E5", "#0EA5A4", "#F59E0B", "#EC4899", "#64748B", "#22C55E", "#8B5CF6", "#EF4444"]

APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --dl-bg: #F5F6FA;
  --dl-surface: #FFFFFF;
  --dl-border: #E4E7EE;
  --dl-ink: #0F172A;
  --dl-muted: #64748B;
  --dl-accent: #4F46E5;
  --dl-accent-soft: #EEF0FF;
  --dl-nav: #0B1220;
  --dl-nav-2: #111A2E;
  --dl-ok: #15803D;   --dl-ok-bg: #E8F7EE;
  --dl-warn: #B45309; --dl-warn-bg: #FEF3E2;
  --dl-bad: #B91C1C;  --dl-bad-bg: #FDECEC;
  --dl-shadow: 0 1px 2px rgba(15,23,42,.05), 0 4px 14px rgba(15,23,42,.05);
}

html, body, .stApp, button, input, textarea, select,
[data-testid="stMarkdownContainer"], [data-testid="stMetric"], [data-testid="stSidebarNav"] span {
  font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
}
/* keep Streamlit's icon font intact */
[data-testid="stIconMaterial"], .material-symbols-rounded, span[class*="material"] {
  font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}
.stApp { background: var(--dl-bg); color: var(--dl-ink); }
.main .block-container, [data-testid="stMainBlockContainer"] {
  max-width: 1360px; padding: 2.2rem 2.6rem 4rem;
}
@media (max-width: 760px) {
  .main .block-container, [data-testid="stMainBlockContainer"] { padding: 1.2rem 1rem 3rem; }
}
[data-testid="stHeader"] { background: transparent; }
h1, h2, h3, h4 { color: var(--dl-ink); letter-spacing: -0.015em; }
h3 { font-size: 1.05rem !important; font-weight: 600 !important; margin-top: 1.4rem; padding-top: .9rem !important; }
[data-testid="stHorizontalBlock"] { margin-bottom: .35rem; }
[data-testid="stCaptionContainer"], .stCaption { color: var(--dl-muted); }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background: var(--dl-nav); border-right: 1px solid #1B263B; }
[data-testid="stSidebar"] > div:first-child { background: transparent; }
[data-testid="stSidebarNav"]::before {
  content: "DataLens AI"; display: block; color: #F8FAFC;
  font-weight: 700; font-size: 1.15rem; letter-spacing: -0.01em;
  padding: 1.3rem 1.1rem .9rem; margin-bottom: .3rem;
  border-bottom: 1px solid #1B263B;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] a { color: #C7D0E0; }
[data-testid="stSidebarNav"] a { border-radius: 9px; margin: 2px 8px; padding: .45rem .7rem; }
[data-testid="stSidebarNav"] a:hover { background: rgba(255,255,255,.06); }
[data-testid="stSidebarNav"] a[aria-current="page"] { background: rgba(99,102,241,.28); }
[data-testid="stSidebarNav"] a[aria-current="page"] span { color: #FFFFFF; font-weight: 600; }
[data-testid="stSidebar"] hr { border-color: #1B263B; }

/* ---------- Page header ---------- */
.dl-header { display:flex; align-items:center; gap:.9rem; margin: 0 0 1.5rem; }
.dl-header-icon {
  width: 44px; height: 44px; border-radius: 12px; background: var(--dl-accent-soft);
  display:flex; align-items:center; justify-content:center; font-size: 1.35rem; flex: none;
}
.dl-header h1 { margin:0; padding:0; font-size: 1.65rem; font-weight: 700; line-height: 1.2; }
.dl-header p { margin: .15rem 0 0; color: var(--dl-muted); font-size: .95rem; }

/* ---------- Cards ---------- */
.dl-card {
  background: var(--dl-surface); border: 1px solid var(--dl-border);
  border-radius: 14px; padding: 1.05rem 1.2rem; box-shadow: var(--dl-shadow);
}
.dl-card .dl-stat-label { min-height: 0; }
.dl-stat-card { min-height: 104px; }
.dl-stat-label { color: var(--dl-muted); font-size: .8rem; font-weight: 500; }
.dl-stat-value { font-size: 1.7rem; font-weight: 700; letter-spacing: -0.02em; margin-top: .15rem; line-height: 1.15; }
.dl-stat-hint { color: var(--dl-muted); font-size: .78rem; margin-top: .25rem; }
.dl-ds-name { font-weight: 600; font-size: 1rem; word-break: break-word; }
.dl-ds-meta { color: var(--dl-muted); font-size: .85rem; margin-top: .2rem; }
.dl-ds-stats { display:flex; gap:1.3rem; margin-top:.8rem; flex-wrap:wrap; }
.dl-ds-stats b { display:block; font-size: 1.15rem; font-weight:700; }
.dl-ds-stats span { color: var(--dl-muted); font-size: .75rem; }
.dl-answer { border-left: 4px solid var(--dl-accent); }
.dl-answer-text { font-size: 1.05rem; line-height: 1.6; margin: 0; }
.dl-q { font-weight: 600; }

/* ---------- Chips ---------- */
.dl-chip { display:inline-block; padding: .12rem .6rem; border-radius: 999px; font-size: .74rem; font-weight: 600; }
.dl-chip.ok   { background: var(--dl-ok-bg);   color: var(--dl-ok); }
.dl-chip.warn { background: var(--dl-warn-bg); color: var(--dl-warn); }
.dl-chip.bad  { background: var(--dl-bad-bg);  color: var(--dl-bad); }
.dl-chip.info { background: var(--dl-accent-soft); color: var(--dl-accent); }
.dl-chip.mute { background: #EEF1F6; color: #475569; }

/* ---------- Native widgets ---------- */
[data-testid="stMetric"] {
  background: var(--dl-surface); border: 1px solid var(--dl-border);
  padding: 1rem 1.15rem; border-radius: 14px; box-shadow: var(--dl-shadow);
}
[data-testid="stMetricLabel"] { color: var(--dl-muted); }
[data-testid="stMetricValue"] { font-weight: 700; letter-spacing: -0.02em; }
div[data-testid="stDataFrame"] {
  border: 1px solid var(--dl-border); border-radius: 12px; overflow: hidden; box-shadow: var(--dl-shadow);
}
.stButton > button, .stDownloadButton > button, [data-testid="stPageLink"] a, .stLinkButton a {
  border-radius: 10px; font-weight: 600; transition: background .15s, border-color .15s, box-shadow .15s;
}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
  background: var(--dl-accent); border-color: var(--dl-accent); color: #fff;
}
.stButton > button[kind="primary"]:hover { background: #4338CA; border-color: #4338CA; box-shadow: 0 4px 12px rgba(79,70,229,.3); }
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible, input:focus-visible {
  outline: 2px solid var(--dl-accent); outline-offset: 2px;
}
[data-testid="stPageLink"] a {
  background: var(--dl-surface); border: 1px solid var(--dl-border); padding: .7rem 1rem; box-shadow: var(--dl-shadow);
}
[data-testid="stPageLink"] a:hover { border-color: var(--dl-accent); background: var(--dl-accent-soft); }
div[data-testid="stFileUploader"] section {
  background: var(--dl-surface); border: 1.5px dashed #C3CBDA; border-radius: 14px; padding: 1.1rem;
}
div[data-testid="stFileUploader"] section:hover { border-color: var(--dl-accent); background: #FAFAFF; }
[data-testid="stTextInput"] input, [data-baseweb="select"] > div {
  border-radius: 10px !important; border-color: var(--dl-border) !important; background: #fff !important;
}
[data-testid="stTextInput"] input { padding: .75rem .9rem; font-size: 1rem; }
[data-testid="stTextInput"] input:focus { border-color: var(--dl-accent) !important; box-shadow: 0 0 0 3px rgba(79,70,229,.15) !important; }
[data-testid="stExpander"] { border: 1px solid var(--dl-border); border-radius: 12px; background: var(--dl-surface); }
[data-testid="stAlert"] { border-radius: 12px; }
[data-testid="stImage"] img, [data-testid="stImageContainer"] img { border-radius: 12px; border: 1px solid var(--dl-border); background:#fff; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px; }

[data-testid="stBaseButton-secondary"] p, .stButton button p { white-space: normal; overflow: visible; text-overflow: clip; line-height: 1.3; }
.stButton > button[kind="secondary"] { min-height: 3rem; background: var(--dl-surface); border: 1px solid var(--dl-border); }
.stButton > button[kind="secondary"]:hover { border-color: var(--dl-accent); color: var(--dl-accent); background: var(--dl-accent-soft); }
.stAppDeployButton { display: none; }
[data-testid="stHeading"], h3 { scroll-margin-top: 1rem; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
"""


def apply_brand_theme() -> None:
    """Apply the shared design system (light workspace + navy sidebar)."""
    st.markdown(APP_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Presentation helpers (pure UI; no data or engine logic)
# ---------------------------------------------------------------------------
def page_header(title: str, subtitle: str = "", icon: str = "") -> None:
    icon_html = f'<div class="dl-header-icon">{escape(icon)}</div>' if icon else ""
    sub_html = f"<p>{escape(subtitle)}</p>" if subtitle else ""
    st.markdown(
        f'<div class="dl-header">{icon_html}<div><h1>{escape(title)}</h1>{sub_html}</div></div>',
        unsafe_allow_html=True,
    )


def chip(text: str, tone: str = "mute") -> str:
    """Return an HTML status chip. tone: ok | warn | bad | info | mute."""
    return f'<span class="dl-chip {tone}">{escape(str(text))}</span>'


def stat_cards(items: list[tuple[str, str, str]]) -> None:
    """Render a row of stat cards from (label, value, hint) tuples."""
    cols = st.columns(len(items))
    for col, (label, value, hint) in zip(cols, items):
        hint_html = f'<div class="dl-stat-hint">{escape(hint)}</div>' if hint else ""
        col.markdown(
            f'<div class="dl-card dl-stat-card"><div class="dl-stat-label">{escape(label)}</div>'
            f'<div class="dl-stat-value">{escape(str(value))}</div>{hint_html}</div>',
            unsafe_allow_html=True,
        )


def dataset_card_html(name: str, df: pd.DataFrame, file_size: str = "") -> str:
    """HTML card summarising one dataset, with a data-quality chip."""
    rows, cols = len(df), len(df.columns)
    missing = int(df.isna().sum().sum())
    dupes = int(df.duplicated().sum())
    cells = max(rows * cols, 1)
    if missing == 0 and dupes == 0:
        status = chip("Clean", "ok")
    elif missing / cells < 0.05:
        status = chip("Minor gaps", "warn")
    else:
        status = chip("Needs review", "bad")
    meta = f"{file_size} · " if file_size else ""
    return (
        '<div class="dl-card">'
        f'<div style="display:flex;justify-content:space-between;gap:.6rem;align-items:flex-start">'
        f'<div class="dl-ds-name">{escape(str(name))}</div>{status}</div>'
        f'<div class="dl-ds-meta">{escape(meta)}{cols} columns</div>'
        '<div class="dl-ds-stats">'
        f'<div><b>{rows:,}</b><span>Rows</span></div>'
        f'<div><b>{missing:,}</b><span>Missing</span></div>'
        f'<div><b>{dupes:,}</b><span>Duplicates</span></div>'
        '</div></div>'
    )


def format_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"


def apply_chart_style() -> None:
    """Give matplotlib charts the same look as the app (colors, grid, fonts)."""
    import matplotlib as mpl
    from cycler import cycler

    mpl.rcParams.update({
        "axes.prop_cycle": cycler(color=CHART_COLORS),
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#CBD5E1",
        "axes.labelcolor": "#475569",
        "axes.titlecolor": "#0F172A",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#EEF1F6",
        "grid.linewidth": 1,
        "xtick.color": "#64748B",
        "ytick.color": "#64748B",
        "text.color": "#0F172A",
        "font.family": ["Inter", "DejaVu Sans"],
        "lines.linewidth": 2.2,
        "patch.edgecolor": "white",
        "patch.linewidth": 0.8,
        "legend.frameon": False,
    })


# ---------------------------------------------------------------------------
# Data helpers (unchanged)
# ---------------------------------------------------------------------------
def read_uploaded_dataframe(uploaded_file: Any) -> pd.DataFrame:
    """Read a CSV/XLSX upload without relying on the current file pointer."""
    data = uploaded_file.getvalue()
    name = str(uploaded_file.name).lower()
    if name.endswith(".csv"):
        return pd.read_csv(BytesIO(data))
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(BytesIO(data))
    raise ValueError("Only CSV and Excel files are supported.")


def load_uploaded_datasets(uploaded_files: list[Any]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    frames: dict[str, pd.DataFrame] = {}
    errors: list[str] = []
    for upload in uploaded_files:
        name = str(upload.name)
        table_name = name.rsplit(".", 1)[0]
        try:
            frame = read_uploaded_dataframe(upload)
            if table_name in frames:
                errors.append(f"Duplicate table name after removing extension: {table_name}")
            else:
                frames[table_name] = frame
        except Exception as exc:  # Show a useful per-file error in the UI.
            errors.append(f"{name}: {exc}")
    return frames, errors


def record_analysis(question: str, payload: dict[str, Any]) -> None:
    """Save a compact in-session copy of a completed analysis for History."""
    history = st.session_state.setdefault("analysis_history", [])
    result = payload.get("result", [])
    columns = payload.get("columns", [])
    if not isinstance(result, list):
        result = []
    if not isinstance(columns, list):
        columns = []
    entry = {
        "timestamp": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "question": question,
        "answer": str(payload.get("answer", "No answer returned.")),
        "chart_type": str(payload.get("chart_type", "NONE")).upper(),
        "columns": columns,
        "result": result,
    }
    history.insert(0, entry)
    del history[50:]
