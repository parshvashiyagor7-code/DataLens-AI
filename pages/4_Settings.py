from pathlib import Path
import os
import re
import shutil
import subprocess
import sys

import streamlit as st

from datalens_ui import apply_brand_theme, page_header, chip

st.set_page_config(page_title="DataLens AI | Settings", page_icon="⚙️", layout="wide")
apply_brand_theme()
ROOT = Path(__file__).resolve().parents[1]

page_header("Settings", "Connection and runtime details for this local DataLens AI workspace.", "⚙️")

st.markdown("### Analysis engine")
engine = ROOT / "test_lim.py"
helper = ROOT / "app.py"
a, b = st.columns(2)
a.markdown(f'<div class="dl-card"><div class="dl-stat-label">test_lim.py (analysis engine)</div><div style="margin-top:.4rem">{chip("Found", "ok") if engine.is_file() else chip("Missing", "bad")}</div></div>', unsafe_allow_html=True)
b.markdown(f'<div class="dl-card"><div class="dl-stat-label">app.py (helper)</div><div style="margin-top:.4rem">{chip("Found", "ok") if helper.is_file() else chip("Missing", "bad")}</div></div>', unsafe_allow_html=True)
if not engine.is_file():
    st.warning("Place your existing test_lim.py in this project root. This page package deliberately does not replace your engine files.")
if not helper.is_file():
    st.warning("Place your existing app.py helper in this project root. It is left untouched by this multipage setup.")

model_name = os.environ.get("OLLAMA_MODEL", "Not set in OLLAMA_MODEL environment variable")
if engine.is_file():
    try:
        text = engine.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"^\s*MODEL_NAME\s*=\s*['\"]([^'\"]+)['\"]", text, re.MULTILINE)
        if match:
            model_name = match.group(1)
    except OSError:
        pass
st.text_input("Configured model (detected; informational)", value=model_name, disabled=True)

st.markdown("### Ollama connection")
if not shutil.which("ollama"):
    st.info("Ollama CLI is not visible in this Python environment's PATH. The existing engine may still be configured externally.")
else:
    if st.button("Check installed models", type="primary"):
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                st.success("Ollama CLI responded successfully.")
                st.code(result.stdout.strip() or "No models listed.")
            else:
                st.error("Ollama returned an error.")
                st.code((result.stderr or result.stdout)[-4000:])
        except subprocess.TimeoutExpired:
            st.error("Ollama check timed out after 10 seconds.")
        except Exception as exc:
            st.error(f"Could not check Ollama: {exc}")

st.markdown("### Workspace")
st.markdown(
    '<div class="dl-card">'
    f'<div class="dl-ds-meta"><b>Theme</b> &nbsp; Navy navigation, light workspace</div>'
    f'<div class="dl-ds-meta" style="margin-top:.5rem"><b>Authentication</b> &nbsp; {chip("Not enabled", "mute")}</div>'
    f'<div class="dl-ds-meta" style="margin-top:.5rem"><b>Data retention</b> &nbsp; Uploads and history stay in the current session only</div>'
    '</div>', unsafe_allow_html=True)
