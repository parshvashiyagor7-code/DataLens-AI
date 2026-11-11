import pandas as pd
import os


def load_dataset(file_path):

    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)

    elif file_path.endswith(".xlsx"):
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


# =========================================================
# DATA PROFILING
# =========================================================

def profile_data(df):

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

        "numeric_columns": list(
            df.select_dtypes(
                include="number"
            ).columns
        ),

        "categorical_columns": list(
            df.select_dtypes(
                include="object"
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