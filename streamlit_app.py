from pathlib import Path
import platform
import subprocess
import shutil

import streamlit as st

from datalens_ui import apply_brand_theme, load_uploaded_datasets

st.set_page_config(page_title="DataLens AI | Dashboard", page_icon="📊", layout="wide")
apply_brand_theme()

ROOT = Path(__file__).resolve().parent
uploads = st.session_state.get("shared_uploaded_files", [])
history = st.session_state.get("analysis_history", [])
frames, load_errors = load_uploaded_datasets(uploads) if uploads else ({}, [])
rows = sum(len(df) for df in frames.values())
columns = sum(len(df.columns) for df in frames.values())
missing = sum(int(df.isna().sum().sum()) for df in frames.values())

st.title("DataLens AI")
st.caption("Your AI-powered workspace for exploring data with natural language.")

left, right = st.columns([1.5, 1], gap="large")
with left:
    st.markdown("### Turn questions into analysis")
    st.write("Upload one or more datasets, ask a question in plain English, and get an answer with a result table and an appropriate chart.")
    st.page_link("pages/1_Ask_Your_Data.py", label="Open Ask Your Data", icon="🔎", use_container_width=True)
    st.page_link("pages/2_Data_Sources.py", label="Manage Data Sources", icon="🗂️", use_container_width=True)
with right:
    st.markdown("### Workspace status")
    st.caption("This workspace keeps uploads and recent analysis history for the current browser session.")
    st.metric("Recent analyses", len(history))
    st.metric("Datasets loaded", len(frames))

st.markdown("### Your data at a glance")
a, b, c, d = st.columns(4)
a.metric("Datasets", f"{len(frames):,}")
b.metric("Total rows", f"{rows:,}")
c.metric("Total columns", f"{columns:,}")
d.metric("Missing values", f"{missing:,}")

if load_errors:
    for err in load_errors:
        st.warning(err)

if frames:
    st.markdown("### Connected datasets")
    st.dataframe(
        [{"Dataset": name, "Rows": len(df), "Columns": len(df.columns), "Missing values": int(df.isna().sum().sum())}
         for name, df in frames.items()],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No datasets are loaded yet. Open **Data Sources** to upload CSV or Excel files.")

if history:
    st.markdown("### Recent questions")
    st.dataframe(
        [{"Time": item.get("timestamp", ""), "Question": item.get("question", ""), "Chart": item.get("chart_type", "NONE")}
         for item in history[:5]],
        use_container_width=True,
        hide_index=True,
    )
    st.page_link("pages/3_Reports_History.py", label="View all analysis history", icon="🕘")

with st.expander("Engine / runtime details"):
    st.write(f"Python: {platform.python_version()}")
    st.write(f"Engine script found: { (ROOT / 'test_lim.py').is_file() }")
    st.write(f"Engine helper app.py found: { (ROOT / 'app.py').is_file() }")
    if shutil.which("ollama"):
        try:
            check = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=8)
            if check.returncode == 0:
                st.caption("Ollama CLI is available on this machine.")
            else:
                st.caption("Ollama CLI was found, but `ollama list` returned an error.")
        except Exception:
            st.caption("Ollama status could not be checked from this page.")
    else:
        st.caption("Ollama CLI was not found in this runtime PATH; your existing engine may run it in another environment.")
