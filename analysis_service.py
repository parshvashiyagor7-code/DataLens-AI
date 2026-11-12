"""Reusable helpers for DataLens AI analysis orchestration.

The legacy test_lim.py remains the CLI execution engine in this phase.
These helpers centralize structured-output parsing so the Streamlit page and
future direct runner can share one contract.
"""
from __future__ import annotations

import json
from typing import Any


RESULT_MARKER = "__STREAMLIT_RESULT__"


def parse_engine_output(stdout: str) -> dict[str, Any]:
    """Parse the final structured result emitted by test_lim.py."""
    if not isinstance(stdout, str) or RESULT_MARKER not in stdout:
        raise ValueError("Analysis engine did not emit a structured result.")

    raw = stdout.rsplit(RESULT_MARKER, 1)[1].strip()
    if not raw:
        raise ValueError("Analysis engine emitted an empty result.")

    line = raw.splitlines()[0].strip()
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError("Analysis engine returned invalid result JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError("Analysis result must be a JSON object.")

    payload.setdefault("chart_type", "NONE")
    payload.setdefault("answer", "")
    payload.setdefault("result", [])
    payload.setdefault("columns", [])
    return payload


def validate_analysis_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the engine-to-UI result contract.

    The function deliberately does not reinterpret values or fabricate rows.
    It only rejects malformed structures and supplies safe optional defaults.
    """
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dictionary")

    result = payload.get("result", [])
    columns = payload.get("columns", [])
    if result is None:
        result = []
    if not isinstance(result, list):
        raise ValueError("'result' must be a list of records")
    if not isinstance(columns, list):
        raise ValueError("'columns' must be a list")
    if any(not isinstance(column, (str, int, float)) for column in columns):
        raise ValueError("Every result column name must be a string or number")
    if any(not isinstance(row, dict) for row in result):
        raise ValueError("Every result row must be an object/dictionary")

    normalized_columns = [str(column) for column in columns]
    if len(normalized_columns) != len(set(normalized_columns)):
        raise ValueError("Result contains duplicate column names")

    # Preserve engine-provided column order and append any row keys omitted
    # from the columns metadata so the UI never silently hides returned data.
    for row in result:
        for key in row:
            key = str(key)
            if key not in normalized_columns:
                normalized_columns.append(key)

    payload["result"] = result
    payload["columns"] = normalized_columns
    payload["chart_type"] = str(payload.get("chart_type") or "NONE").upper()
    payload["answer"] = str(payload.get("answer") or "")
    payload.setdefault("status", "success")
    payload["row_count"] = len(result)
    return payload
