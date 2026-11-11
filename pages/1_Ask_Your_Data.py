

import os

import sys

import json
import re

import tempfile

import subprocess



import streamlit as st

import pandas as pd

from datalens_ui import apply_brand_theme, record_analysis, page_header, stat_cards, apply_chart_style, chip





st.set_page_config(

    page_title="DataLens AI | Ask Your Data",

    page_icon="🔎",

    layout="wide"

)



apply_brand_theme()

page_header("Ask Your Data", "Ask questions in plain English and get an answer, result table, chart and downloads.", "🔎")



# --------------------------------------------------

# 1. MULTIPLE FILE UPLOAD

# --------------------------------------------------



new_uploads = st.file_uploader(

    "Add CSV or Excel datasets",

    type=["csv", "xlsx"],

    accept_multiple_files=True,

    key="ask_data_uploader"

)

# Keep uploaded files in session state so they remain available when the
# user navigates between multipage sections.
if new_uploads:
    st.session_state["shared_uploaded_files"] = new_uploads

uploaded_files = new_uploads or st.session_state.get("shared_uploaded_files", [])

if uploaded_files:
    st.caption(f"{len(uploaded_files)} dataset(s) available to the analyst.")



if uploaded_files:



    dataframes = {}

    load_errors = []



    # --------------------------------------------------

    # 2. LOAD ALL TABLES

    # --------------------------------------------------



    for uploaded_file in uploaded_files:

        try:

            table_name = os.path.splitext(uploaded_file.name)[0]



            if uploaded_file.name.lower().endswith(".csv"):

                df = pd.read_csv(uploaded_file)

            else:

                df = pd.read_excel(uploaded_file)



            if table_name in dataframes:

                load_errors.append(

                    f"Duplicate table name: {table_name}"

                )

            else:

                dataframes[table_name] = df



        except Exception as exc:

            load_errors.append(

                f"{uploaded_file.name}: {exc}"

            )



    for error in load_errors:

        st.error(error)



    if dataframes:



        st.markdown(chip(f"{len(dataframes)} dataset(s) loaded", "ok"), unsafe_allow_html=True)



        # --------------------------------------------------

        # 3. OVERALL METRICS

        # --------------------------------------------------



        total_rows = sum(len(df) for df in dataframes.values())

        total_columns = sum(

            len(df.columns) for df in dataframes.values()

        )

        total_missing = sum(

            int(df.isna().sum().sum())

            for df in dataframes.values()

        )



        stat_cards([
            ("Tables", f"{len(dataframes):,}", "Available to the analyst"),
            ("Total rows", f"{total_rows:,}", ""),
            ("Total columns", f"{total_columns:,}", ""),
            ("Missing values", f"{total_missing:,}", ""),
        ])



        # --------------------------------------------------

        # 4. TABLE PREVIEW

        # --------------------------------------------------



        st.subheader("Dataset Preview")



        selected_table = st.selectbox(

            "Select a table to preview",

            options=list(dataframes.keys())

        )



        selected_df = dataframes[selected_table]



        st.caption(

            f"{selected_table}: "

            f"{len(selected_df):,} rows × "

            f"{len(selected_df.columns)} columns"

        )



        st.dataframe(

            selected_df.head(50),

            use_container_width=True

        )



        with st.expander("View all loaded tables"):

            for table_name, table_df in dataframes.items():

                st.markdown(f"**{table_name}**")

                st.dataframe(

                    table_df.head(10),

                    use_container_width=True

                )



        # --------------------------------------------------

        # 5. ASK YOUR DATA

        # --------------------------------------------------



        st.subheader("Ask Your Data")



        st.caption(

            "You can ask questions involving one or multiple tables."

        )



        def _use_suggestion(text):
            st.session_state["analysis_question"] = text

        _first_table = next(iter(dataframes))
        _first_df = dataframes[_first_table]
        _num_cols = [
            c for c in _first_df.select_dtypes(include="number").columns
            if not str(c).lower().endswith("id") and str(c).lower() not in ("id", "index")
        ]
        _cat_cols = _first_df.select_dtypes(exclude="number").columns.tolist()
        _suggestions = [f"How many rows are in {_first_table}?"]
        if _cat_cols:
            _suggestions.append(f"How many records are there for each {_cat_cols[0]}?")
        if _cat_cols and _num_cols:
            _suggestions.append(f"What is the total {_num_cols[0]} by {_cat_cols[0]}?")
            _suggestions.append(f"Show the top 5 {_cat_cols[0]} by total {_num_cols[0]}")
        st.caption("Try a suggested question")
        _sug_cols = st.columns(len(_suggestions))
        for _i, (_col, _text) in enumerate(zip(_sug_cols, _suggestions)):
            _col.button(_text, key=f"suggestion_{_i}", on_click=_use_suggestion, args=(_text,), use_container_width=True)

        question = st.text_input(

            "Your question",

            placeholder=(

                "e.g. Which customer has the highest total order amount?"

            ),

            key="analysis_question"

        )



        analyze_clicked = st.button(

            "Analyze",

            type="primary"

        )



        if analyze_clicked:



            if not question.strip():

                st.warning("Please enter a question.")



            else:

                # This script now lives in /pages; the existing engine stays in the
                # project root beside app.py and test_lim.py.
                project_dir = os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )



                engine_path = os.path.join(

                    project_dir,

                    "test_lim.py"

                )



                if not os.path.isfile(engine_path):

                    st.error(

                        "test_lim.py was not found in the project folder."

                    )



                else:

                    # A temporary directory keeps all uploaded tables

                    # together for the existing multi-table engine.

                    try:

                        with tempfile.TemporaryDirectory() as temp_dir:



                            for uploaded_file in uploaded_files:

                                destination = os.path.join(

                                    temp_dir,

                                    os.path.basename(uploaded_file.name)

                                )



                                with open(destination, "wb") as file:

                                    file.write(

                                        uploaded_file.getvalue()

                                    )



                            env = os.environ.copy()
                            try:
                                gemini_api_key = st.secrets["GEMINI_API_KEY"]
                            except Exception:
                                gemini_api_key = os.environ.get("GEMINI_API_KEY")

                            if not gemini_api_key:
                                st.error("Gemini API key is not configured.")
                                st.stop()

                            env["GEMINI_API_KEY"] = gemini_api_key
                            env["MPLBACKEND"] = "Agg"



                            with st.spinner(

                                "Analyzing all uploaded tables..."

                            ):

                                process = subprocess.run(

                                    [sys.executable, engine_path],

                                    input=(

                                        f"{temp_dir}\n"

                                        f"{question.strip()}\n"

                                    ),

                                    capture_output=True,

                                    text=True,

                                    cwd=project_dir,

                                    env=env,

                                    timeout=300

                                )



                        if process.returncode != 0:

                            st.error(

                                "The analysis engine returned an error."

                            )



                            error_output = (

                                process.stderr

                                or process.stdout

                            )



                            st.code(error_output[-10000:])



                        else:

                            output = process.stdout

                            result_marker = "__STREAMLIT_RESULT__"



                            # --------------------------------------

                            # 6. STRUCTURED RESULT

                            # --------------------------------------



                            if result_marker in output:

                                json_text = output.rsplit(
                                    result_marker, 1
                                )[1].strip()

                                # Parse only the JSON line emitted by test_lim.py.
                                json_line = json_text.splitlines()[0].strip()
                                payload = json.loads(json_line)

                                # Some engine versions print CHART TYPE outside
                                # the JSON payload. Recover it as a fallback.
                                chart_match = re.search(
                                    r"CHART\s+TYPE\s*:\s*(BAR|LINE|PIE|HISTOGRAM|NONE)",
                                    output,
                                    flags=re.IGNORECASE
                                )
                                if not payload.get("chart_type") and chart_match:
                                    payload["chart_type"] = chart_match.group(1).upper()

                                # Capture completed analyses for the History page.
                                record_analysis(question.strip(), payload)



                                st.subheader("Answer")
                                from html import escape as _esc
                                _answer_html = _esc(str(payload.get("answer", "No answer returned."))).replace("\n", "<br>")
                                st.markdown(
                                    f'<div class="dl-card dl-answer"><p class="dl-answer-text">{_answer_html}</p></div>',
                                    unsafe_allow_html=True,
                                )



                                result_rows = payload.get("result", [])

                                result_columns = payload.get("columns", [])



                                if result_rows:

                                    result_df = pd.DataFrame(

                                        result_rows,

                                        columns=result_columns or None

                                    )



                                    st.subheader("Analysis Result")



                                    st.dataframe(

                                        result_df,

                                        use_container_width=True

                                    )



                                    st.download_button(

                                        "Download Result CSV",

                                        data=result_df.to_csv(

                                            index=False

                                        ).encode("utf-8"),

                                        file_name="analysis_result.csv",

                                        mime="text/csv"

                                    )

                                    # --------------------------------------
                                    # 7. PROFESSIONAL VISUALIZATION
                                    chart_type = str(payload.get("chart_type", "NONE")).upper()
                                    st.subheader("Visualization")

                                    if result_df.empty or chart_type == "NONE":
                                        st.info(
                                            "No chart is needed for this query, "
                                            "or the result contains no rows."
                                        )
                                    elif chart_type not in {"BAR", "LINE", "PIE", "HISTOGRAM"}:
                                        st.info(f"Chart type '{chart_type}' is not supported yet.")
                                    else:
                                        import io
                                        import matplotlib.pyplot as plt
                                        import numpy as np

                                        apply_chart_style()
                                        categorical_cols = result_df.select_dtypes(
                                            include=["object", "category", "string"]
                                        ).columns.tolist()
                                        numeric_cols = result_df.select_dtypes(
                                            include=["number"]
                                        ).columns.tolist()
                                        date_cols = [
                                            col for col in result_df.columns
                                            if any(word in str(col).lower() for word in
                                                   ("date", "time", "timestamp", "datetime"))
                                        ]

                                        fig = None
                                        chart_error = None
                                        question_title = question.strip().rstrip("?.! ")
                                        if question_title:
                                            chart_title = question_title[0].upper() + question_title[1:]
                                        else:
                                            chart_title = "Analysis Results"

                                        def format_value(value):
                                            try:
                                                number = float(value)
                                                if np.isfinite(number):
                                                    if number.is_integer():
                                                        return f"{number:,.0f}"
                                                    return f"{number:,.2f}"
                                            except (TypeError, ValueError):
                                                pass
                                            return str(value)

                                        try:
                                            if chart_type == "BAR":
                                                if not categorical_cols or not numeric_cols:
                                                    chart_error = "A bar chart needs a category and a numeric column."
                                                else:
                                                    category_col, value_col = categorical_cols[0], numeric_cols[0]
                                                    chart_df = result_df[[category_col, value_col]].copy()
                                                    chart_df[value_col] = pd.to_numeric(chart_df[value_col], errors="coerce")
                                                    chart_df = chart_df.dropna(subset=[value_col])
                                                    if chart_df.empty:
                                                        chart_error = "No numeric values are available for the bar chart."
                                                    else:
                                                        fig, ax = plt.subplots(figsize=(10, 6))
                                                        bars = ax.bar(chart_df[category_col].astype(str), chart_df[value_col])
                                                        ax.bar_label(
                                                            bars,
                                                            labels=[format_value(v) for v in chart_df[value_col]],
                                                            padding=3,
                                                            fontsize=9
                                                        )
                                                        ax.set_xlabel(category_col)
                                                        ax.set_ylabel(value_col)
                                                        ax.tick_params(axis="x", rotation=30)
                                                        ax.margins(y=0.15)

                                            elif chart_type == "LINE":
                                                if not numeric_cols:
                                                    chart_error = "A line chart needs a numeric result column."
                                                else:
                                                    chart_df = result_df.copy()
                                                    if date_cols:
                                                        x_col = date_cols[0]
                                                        chart_df[x_col] = pd.to_datetime(chart_df[x_col], errors="coerce")
                                                        chart_df = chart_df.dropna(subset=[x_col]).sort_values(x_col)
                                                        x_values = chart_df[x_col]
                                                        x_label = x_col
                                                    else:
                                                        x_values = np.arange(1, len(chart_df) + 1)
                                                        x_label = "Row"
                                                    if chart_df.empty:
                                                        chart_error = "No valid rows are available for the line chart."
                                                    else:
                                                        fig, ax = plt.subplots(figsize=(10, 6))
                                                        for value_col in numeric_cols:
                                                            y_values = pd.to_numeric(chart_df[value_col], errors="coerce")
                                                            ax.plot(x_values, y_values, marker="o", label=value_col)
                                                        ax.set_xlabel(x_label)
                                                        ax.set_ylabel("Value")
                                                        if len(numeric_cols) > 1:
                                                            ax.legend()

                                            elif chart_type == "PIE":
                                                if not categorical_cols or not numeric_cols:
                                                    chart_error = "A pie chart needs a category and a numeric column."
                                                else:
                                                    values = pd.to_numeric(result_df[numeric_cols[0]], errors="coerce").fillna(0)
                                                    labels = result_df[categorical_cols[0]].astype(str)
                                                    valid = values > 0
                                                    values, labels = values[valid], labels[valid]
                                                    if values.empty:
                                                        chart_error = "The pie chart has no positive values to display."
                                                    else:
                                                        fig, ax = plt.subplots(figsize=(8, 7))
                                                        wedges, _, _ = ax.pie(
                                                            values,
                                                            labels=None,
                                                            autopct="%1.1f%%",
                                                            startangle=90,
                                                            pctdistance=0.7
                                                        )
                                                        ax.legend(wedges, labels, title=categorical_cols[0],
                                                                  loc="center left", bbox_to_anchor=(1, 0.5))

                                            elif chart_type == "HISTOGRAM":
                                                if not numeric_cols:
                                                    chart_error = "A histogram needs a numeric column."
                                                else:
                                                    values = pd.to_numeric(result_df[numeric_cols[0]], errors="coerce").dropna()
                                                    if values.empty:
                                                        chart_error = "No valid numeric values are available for the histogram."
                                                    else:
                                                        fig, ax = plt.subplots(figsize=(10, 6))
                                                        ax.hist(values, bins=10, edgecolor="white")
                                                        ax.set_xlabel(numeric_cols[0])
                                                        ax.set_ylabel("Frequency")

                                            if chart_error:
                                                st.info(chart_error)
                                            elif fig is not None:
                                                fig.suptitle(chart_title, fontsize=14, fontweight="bold", y=0.98)
                                                fig.tight_layout(rect=[0, 0, 1, 0.94])
                                                st.pyplot(fig, use_container_width=True)
                                                png_buffer = io.BytesIO()
                                                fig.savefig(png_buffer, format="png", dpi=200, bbox_inches="tight")
                                                png_buffer.seek(0)
                                                safe_chart_name = re.sub(r"[^A-Za-z0-9_-]+", "_", chart_title).strip("_")[:60]
                                                st.download_button(
                                                    "Download Chart as PNG",
                                                    data=png_buffer.getvalue(),
                                                    file_name=f"{safe_chart_name or 'analysis_chart'}.png",
                                                    mime="image/png",
                                                    key="download_chart_png"
                                                )
                                        finally:
                                            if fig is not None:
                                                plt.close(fig)



                                else:

                                    st.info(

                                        "The analysis returned no result rows."

                                    )



                            else:

                                # Fallback for the old text-only output.

                                marker = (

                                    "================ FINAL ANSWER ================"

                                )



                                if marker in output:

                                    answer = output.rsplit(

                                        marker, 1

                                    )[1].strip()



                                    st.subheader("Answer")

                                    st.write(answer)

                                else:

                                    st.warning(

                                        "The final answer marker was not found."

                                    )



                            with st.expander("Technical output"):

                                st.text(output[-12000:])



                    except subprocess.TimeoutExpired:

                        st.error(

                            "Analysis exceeded 5 minutes and was stopped."

                        )



                    except json.JSONDecodeError as exc:

                        st.error(

                            f"Could not parse the structured result: {exc}"

                        )



                    except Exception as exc:

                        st.error(f"Analysis failed: {exc}")



else:

    st.info(

        "Upload one or more CSV/Excel files to begin analysis. Upload multiple files for multi-table questions."

    )
