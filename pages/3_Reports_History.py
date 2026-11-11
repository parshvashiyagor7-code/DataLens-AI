import pandas as pd
import streamlit as st

from datalens_ui import apply_brand_theme, page_header, stat_cards, chip

st.set_page_config(page_title="DataLens AI | Reports & History", page_icon="🕘", layout="wide")
apply_brand_theme()

page_header("Reports & History", "Review analyses completed in this browser session.", "🕘")
history = st.session_state.get("analysis_history", [])

if not history:
    st.info("No analyses yet. Run a question in Ask Your Data and it will appear here during this session.")
    st.page_link("pages/1_Ask_Your_Data.py", label="Go to Ask Your Data", icon="🔎")
    st.stop()

stat_cards([
    ("Saved analyses", f"{len(history):,}", "This session"),
    ("With charts", f"{sum(1 for x in history if str(x.get('chart_type', 'NONE')).upper() != 'NONE'):,}", ""),
    ("With result tables", f"{sum(1 for x in history if x.get('result')):,}", ""),
])
st.markdown("### All analyses")
st.dataframe(
    [{"Time": x.get("timestamp", ""), "Question": x.get("question", ""), "Chart type": x.get("chart_type", "NONE"), "Result rows": len(x.get("result", []))}
     for x in history],
    use_container_width=True,
    hide_index=True,
)

labels = [f"{i + 1}. {item.get('question', 'Untitled question')[:90]}" for i, item in enumerate(history)]
selected_label = st.selectbox("Open an analysis", labels)
selected_index = labels.index(selected_label)
item = history[selected_index]

from html import escape as _esc
st.markdown("### Analysis")
st.markdown(
    f'<div class="dl-card dl-answer"><div class="dl-q">{_esc(str(item.get("question", "")))}</div>'
    f'<p class="dl-answer-text" style="margin-top:.6rem">{_esc(str(item.get("answer", "No answer saved.")))}</p>'
    f'<div style="margin-top:.8rem">{chip("Chart: " + str(item.get("chart_type", "NONE")), "info")} '
    f'{chip(str(item.get("timestamp", "unknown time")), "mute")}</div></div>',
    unsafe_allow_html=True,
)

rows = item.get("result", [])
columns = item.get("columns", [])
if rows:
    try:
        df = pd.DataFrame(rows, columns=columns or None)
    except Exception:
        df = pd.DataFrame(rows)
    st.markdown("### Result snapshot")
    st.dataframe(df, use_container_width=True, hide_index=True)
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download saved result as CSV",
        data=csv_bytes,
        file_name=f"datalens_analysis_{selected_index + 1}.csv",
        mime="text/csv",
    )
else:
    st.caption("This analysis did not return a tabular result to save.")

st.caption("History is kept for this session only and clears when the session resets.")
