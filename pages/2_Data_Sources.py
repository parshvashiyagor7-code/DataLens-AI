import streamlit as st

from datalens_ui import apply_brand_theme, load_uploaded_datasets, page_header, stat_cards, dataset_card_html, format_bytes

st.set_page_config(page_title="DataLens AI | Data Sources", page_icon="🗂️", layout="wide")
apply_brand_theme()

page_header("Data Sources", "Upload CSV or Excel tables once and use them across every page.", "🗂️")

st.markdown("### Upload or replace datasets")
st.caption("Uploading new files replaces the current set. Use Clear below to remove everything.")
fresh_uploads = st.file_uploader(
    "Upload datasets",
    type=["csv", "xlsx"],
    accept_multiple_files=True,
    key="data_sources_uploader",
    help="Upload multiple tables together when your questions require joins.",
)
if fresh_uploads:
    st.session_state["shared_uploaded_files"] = fresh_uploads

uploads = fresh_uploads or st.session_state.get("shared_uploaded_files", [])
if not uploads:
    st.info("Upload one or more CSV/Excel files to get started. Files added here will be available on Ask Your Data during this session.")
    st.stop()

frames, errors = load_uploaded_datasets(uploads)
for error in errors:
    st.error(error)

stat_cards([
    ("Datasets", f"{len(frames):,}", ""),
    ("Total rows", f"{sum(len(df) for df in frames.values()):,}", ""),
    ("Total columns", f"{sum(len(df.columns) for df in frames.values()):,}", ""),
    ("Missing values", f"{sum(int(df.isna().sum().sum()) for df in frames.values()):,}", ""),
])

if frames:
    st.markdown("### Your datasets")
    sizes = {}
    for up in uploads:
        try:
            sizes[str(up.name).rsplit(".", 1)[0]] = format_bytes(len(up.getvalue()))
        except Exception:
            pass
    items = list(frames.items())
    for i in range(0, len(items), 3):
        row_cols = st.columns(3, gap="medium")
        for col, (name, frame) in zip(row_cols, items[i:i + 3]):
            col.markdown(dataset_card_html(name, frame, sizes.get(name, "")), unsafe_allow_html=True)

if frames:
    st.markdown("### Inventory")
    inventory = []
    for name, df in frames.items():
        inventory.append({
            "Dataset": name,
            "Rows": len(df),
            "Columns": len(df.columns),
            "Missing values": int(df.isna().sum().sum()),
            "Duplicate rows": int(df.duplicated().sum()),
        })
    st.dataframe(inventory, use_container_width=True, hide_index=True)

    selected = st.selectbox("Preview dataset", list(frames.keys()))
    df = frames[selected]
    st.markdown(f"### Preview: {selected}")
    st.caption(f"{len(df):,} rows × {len(df.columns):,} columns (first 100 rows shown)")
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)
    with st.expander("Schema: column names, types and quality"):
        details = []
        for col in df.columns:
            details.append({
                "Column": str(col),
                "Data type": str(df[col].dtype),
                "Missing values": int(df[col].isna().sum()),
                "Unique values": int(df[col].nunique(dropna=True)),
            })
        st.dataframe(details, use_container_width=True, hide_index=True)

left, right = st.columns([1, 2])
with left:
    if st.button("Remove all datasets", type="secondary", use_container_width=True):
        st.session_state["shared_uploaded_files"] = []
        st.rerun()
with right:
    st.caption("Uploads are held in Streamlit session memory; they are not permanently stored on disk by this page.")
