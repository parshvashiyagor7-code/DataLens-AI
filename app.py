import pandas as pd
import os
import re
from difflib import SequenceMatcher


def load_dataset(file_path):

    if file_path.lower().endswith(".csv"):
        df = pd.read_csv(file_path)

    elif file_path.lower().endswith(".xlsx"):
        df = pd.read_excel(file_path)

    else:
        raise ValueError(
            "Only CSV and Excel files are supported."
        )

    return df


# =========================================================
# GENERIC DATASET LOADER
# =========================================================

def load_user_dataset(path):

    # -----------------------------------------------------
    # CASE 1: SINGLE FILE
    # -----------------------------------------------------

    if os.path.isfile(path):

        df = load_dataset(path)

        return {
            os.path.splitext(
                os.path.basename(path)
            )[0]: df
        }

    # -----------------------------------------------------
    # CASE 2: FOLDER WITH MULTIPLE FILES
    # -----------------------------------------------------

    if os.path.isdir(path):

        tables = {}

        for file_name in os.listdir(path):

            # Ignore Excel/Office lock files and hidden temporary artifacts.
            if file_name.startswith(("~$", ".")):
                continue

            file_path = os.path.join(
                path,
                file_name
            )

            if file_name.lower().endswith(
                (".csv", ".xlsx")
            ):

                table_name = os.path.splitext(
                    file_name
                )[0]

                tables[table_name] = load_dataset(
                    file_path
                )

        if not tables:

            raise ValueError(
                "No CSV or Excel files found in folder."
            )

        return tables

    # -----------------------------------------------------
    # INVALID PATH
    # -----------------------------------------------------

    raise ValueError(
        "Invalid file or folder path."
    )


def validate_dataset_tables(tables):
    """Return actionable data-quality issues without modifying user data."""
    issues = []
    if not isinstance(tables, dict) or not tables:
        return ["No datasets are available. Upload at least one CSV or Excel file."]

    for table_name, df in tables.items():
        if not isinstance(df, pd.DataFrame):
            issues.append(f"{table_name}: dataset could not be represented as a table.")
            continue
        if df.empty:
            issues.append(f"{table_name}: table has no data rows.")
        if len(df.columns) == 0:
            issues.append(f"{table_name}: table has no columns.")
        duplicated = [str(col) for col in df.columns[df.columns.duplicated()].tolist()]
        if duplicated:
            issues.append(
                f"{table_name}: duplicate column names detected ({', '.join(duplicated)}). "
                "Rename them before analysis to avoid ambiguous column references."
            )
        blank_columns = [
            str(col) for col in df.columns
            if str(col).strip() == "" or str(col).lower().startswith("unnamed:")
        ]
        if blank_columns:
            issues.append(
                f"{table_name}: blank/unnamed columns detected ({', '.join(blank_columns)}). "
                "Consider removing or renaming them."
            )
        all_missing = [
            str(col) for position, col in enumerate(df.columns)
            if df.iloc[:, position].isna().all()
        ]
        if all_missing:
            issues.append(
                f"{table_name}: entirely empty columns ({', '.join(all_missing)})."
            )
    return issues


def build_schema_context(tables, sample_value_limit=3):
    """Create a compact, deterministic schema summary for UI/debugging."""
    context = {}
    for table_name, df in tables.items():
        columns = []
        for position, column in enumerate(df.columns):
            series = df.iloc[:, position]
            samples = []
            for value in series.dropna().head(500).drop_duplicates().head(sample_value_limit):
                samples.append(value.isoformat() if hasattr(value, "isoformat") else str(value))
            columns.append({
                "name": str(column),
                "dtype": str(series.dtype),
                "missing": int(series.isna().sum()),
                "unique": int(series.nunique(dropna=True)),
                "samples": samples,
            })
        context[str(table_name)] = {
            "rows": int(len(df)),
            "columns": columns,
        }
    return context


# =========================================================
# SCHEMA COLUMN RESOLUTION
# =========================================================

def normalize_column_name(value):
    """Normalize a column label for punctuation/case-insensitive matching."""
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def resolve_column_name(name, columns, fuzzy_threshold=0.88, ambiguity_margin=0.06):
    """Resolve a requested column conservatively; return None if ambiguous."""
    if name is None:
        return None

    requested = str(name).strip()
    if not requested:
        return None

    column_list = list(columns)
    # Exact and normalized matches always take precedence.
    for column in column_list:
        if str(column).casefold() == requested.casefold():
            return column

    normalized = normalize_column_name(requested)
    normalized_matches = [
        column for column in column_list
        if normalize_column_name(column) == normalized
    ]
    if len(normalized_matches) == 1:
        return normalized_matches[0]
    if len(normalized_matches) > 1:
        return None

    # Fuzzy matching is intentionally conservative to avoid selecting the
    # wrong metric when two column names are similar.
    scored = sorted(
        ((SequenceMatcher(None, normalized, normalize_column_name(col)).ratio(), col)
         for col in column_list),
        reverse=True,
        key=lambda item: item[0],
    )
    if not scored or scored[0][0] < fuzzy_threshold:
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < ambiguity_margin:
        return None
    return scored[0][1]


def rank_numeric_columns(question, tables):
    """Rank numeric columns using the question wording and observed schema.

    Returns (best_column, confidence). Ambiguous or weak matches return (None, 0).
    This deliberately avoids dataset-specific column-name fallbacks.
    """
    question_tokens = set(re.findall(r"[a-z0-9]+", str(question).casefold()))
    semantic_groups = {
        "revenue": {"revenue", "sales", "sale", "income", "turnover", "amount", "value"},
        "amount": {"amount", "value", "total", "cost", "price", "spend", "spent", "revenue", "sales"},
        "price": {"price", "unit", "cost", "amount", "value"},
        "score": {"score", "rating", "points", "performance"},
        "quantity": {"quantity", "qty", "units", "count", "volume"},
        "profit": {"profit", "margin", "earnings", "net"},
    }
    requested = set()
    for token in question_tokens:
        requested.update(semantic_groups.get(token, {token}))

    candidates = []
    seen = set()
    for table_name, df in tables.items():
        for column in df.columns:
            if column in seen or not pd.api.types.is_numeric_dtype(df[column]):
                continue
            seen.add(column)
            tokens = set(re.findall(r"[a-z0-9]+", str(column).casefold()))
            overlap = len(tokens & requested)
            if overlap:
                # Prefer meaningful overlap; penalize identifiers and row counters.
                penalty = 1 if any(t in tokens for t in {"id", "index", "row", "serial"}) else 0
                score = overlap / max(len(tokens), 1) - penalty
                candidates.append((score, str(column), column))

    candidates.sort(key=lambda item: item[0], reverse=True)
    if not candidates or candidates[0][0] <= 0:
        return None, 0.0
    if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.10:
        return None, candidates[0][0]
    return candidates[0][2], candidates[0][0]


# =========================================================
# DATA PROFILING
# =========================================================

def profile_data(df, sample_value_limit=5):

    profile = {

        "shape": {
            "rows": len(df),
            "columns": len(df.columns)
        },

        "columns": {
            column: str(df[column].dtype)
            for column in df.columns
        },

        "missing_values": {
            column: int(df[column].isnull().sum())
            for column in df.columns
            if df[column].isnull().sum() > 0
        },

        "unique_values": {
            column: int(df[column].nunique())
            for column in df.columns
        },

        # Small, representative examples help the semantic parser map natural
        # language values (e.g. "Gold") to the right categorical field.
        "sample_values": {
            column: [
                (value.isoformat() if hasattr(value, "isoformat") else str(value))
                for value in df[column].dropna().head(1000).drop_duplicates().head(sample_value_limit).tolist()
            ]
            for column in df.columns
        },

        "numeric_columns": list(
            df.select_dtypes(
                include="number"
            ).columns
        ),

        "categorical_columns": list(
            df.select_dtypes(
                include=["object", "string", "category"]
            ).columns
        )
    }

    return profile


# =========================================================
# FILTER TOOL
# =========================================================

def filter_data(
    df,
    column,
    operator,
    value
):

    if column not in df.columns:

        raise ValueError(
            f"Column '{column}' does not exist."
        )

    if operator == ">":

        result = df[df[column] > value]

    elif operator == "<":

        result = df[df[column] < value]

    elif operator == ">=":

        result = df[df[column] >= value]

    elif operator == "<=":

        result = df[df[column] <= value]

    elif operator == "==":

        result = df[df[column] == value]

    elif operator == "!=":

        result = df[df[column] != value]

    else:

        raise ValueError(
            "Unsupported operator."
        )

    return result


# =========================================================
# GROUP TOOL
# =========================================================

def group_data(
    df,
    group_column,
    value_column,
    aggregation
):

    if group_column not in df.columns:

        raise ValueError(
            f"Column '{group_column}' does not exist."
        )

    if value_column not in df.columns:

        raise ValueError(
            f"Column '{value_column}' does not exist."
        )

    # -----------------------------------------------------
    # SUM
    # -----------------------------------------------------

    if aggregation == "sum":

        result = (
            df.groupby(group_column)[value_column]
            .sum()
            .reset_index()
        )

    # -----------------------------------------------------
    # MEAN
    # -----------------------------------------------------

    elif aggregation == "mean":

        result = (
            df.groupby(group_column)[value_column]
            .mean()
            .reset_index()
        )

    # -----------------------------------------------------
    # MIN
    # -----------------------------------------------------

    elif aggregation == "min":

        result = (
            df.groupby(group_column)[value_column]
            .min()
            .reset_index()
        )

    # -----------------------------------------------------
    # MAX
    # -----------------------------------------------------

    elif aggregation == "max":

        result = (
            df.groupby(group_column)[value_column]
            .max()
            .reset_index()
        )

    # -----------------------------------------------------
    # COUNT
    # -----------------------------------------------------

    elif aggregation == "count":

        # Use size() instead of
        # groupby()[value_column].count()
        #
        # This also works when:
        # group_column == value_column

        result = (
            df.groupby(group_column)
            .size()
            .reset_index(name="Count")
        )

    # -----------------------------------------------------
    # INVALID AGGREGATION
    # -----------------------------------------------------

    else:

        raise ValueError(
            "Unsupported aggregation."
        )

    return result


# =========================================================
# JOIN TOOL
# =========================================================

def join_data(
    left_df,
    right_df,
    left_column,
    right_column,
    how="inner"
):

    result = left_df.merge(
        right_df,
        left_on=left_column,
        right_on=right_column,
        how=how
    )

    return result


# =========================================================
# TOOL REGISTRY
# =========================================================

TOOL_REGISTRY = {

    "FILTER": filter_data,

    "GROUP": group_data,

    "PROFILE": profile_data,

    "JOIN": join_data

}