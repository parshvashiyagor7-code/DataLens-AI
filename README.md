# DataLens AI — Multipage Streamlit UI

This is a targeted UI/architecture update. It preserves the existing `test_lim.py` analysis engine invocation and the previous Ask Your Data result/chart renderer, including CSV and PNG downloads.

## Important: keep your existing engine files

This UI bundle deliberately does **not** include or overwrite your existing root `app.py` or `test_lim.py`. The current engine imports helper functions from `app.py`, and the Ask Your Data page invokes `test_lim.py` from the project root.

**Safest setup:** back up your existing project, then extract these files into the same project directory that already contains your working `app.py` and `test_lim.py`. Allow the new `streamlit_app.py`, `datalens_ui.py`, `.streamlit/`, `pages/`, and `requirements.txt` files to be added. Do not rename or overwrite the existing engine helper `app.py`. If you already have `.streamlit/config.toml`, back it up and merge the included `[theme]` values instead of blindly replacing your existing settings.

## Pages

- `streamlit_app.py` — Dashboard / landing page
- `pages/1_Ask_Your_Data.py` — existing analysis flow, result table, chart renderer, and downloads
- `pages/2_Data_Sources.py` — shared CSV/Excel uploads, preview, missing values, duplicate rows, column details
- `pages/3_Reports_History.py` — recent in-session questions, answer snapshots, and CSV export
- `pages/4_Settings.py` — engine file checks and optional Ollama CLI check

Uploads and history are session-only in this first version; they are not persisted to disk or a database.

## Run

From your existing project root:

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

## Validation note

All Python files were syntax-compiled in the build environment. Streamlit was not installed in that environment, so browser/runtime testing needs to be done locally with your existing engine and Ollama setup.
