"""
Schema validation for the Adult dataset.

Provides basic validation without requiring pandera/pydantic.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# Expected schema for model-ready data
EXPECTED_COLUMNS = {
    "age": "int64",
    "workclass": "object",
    "fnlwgt": "int64",
    "education": "object",
    "education_num": "int64",
    "marital_status": "object",
    "occupation": "object",
    "relationship": "object",
    "race": "object",
    "sex": "object",
    "capital_gain": "int64",
    "capital_loss": "int64",
    "hours_per_week": "int64",
    "native_country": "object",
    "income": "object",
    "split": "object",
    "race_binary": "object",
}

REQUIRED_COLUMNS = list(EXPECTED_COLUMNS.keys())

# Valid values for categorical columns
VALID_SEX = {"Male", "Female"}
VALID_INCOME = {">50K", "<=50K"}
VALID_SPLIT = {"train", "test", "val"}
VALID_RACE_BINARY = {"White", "Non-White"}


class SchemaValidationError(Exception):
    """Raised when schema validation fails."""

    pass


def validate_columns(df: pd.DataFrame) -> list[str]:
    """
    Validate that dataframe has required columns.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to validate.

    Returns
    -------
    list[str]
        List of validation errors (empty if valid).
    """
    errors = []
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {missing}")

    return errors


def validate_no_nulls(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> list[str]:
    """
    Validate that specified columns have no null values.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to validate.
    columns : list[str], optional
        Columns to check. Defaults to all required columns.

    Returns
    -------
    list[str]
        List of validation errors.
    """
    if columns is None:
        columns = [c for c in REQUIRED_COLUMNS if c in df.columns]

    errors = []
    for col in columns:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                errors.append(f"Column '{col}' has {null_count} null values")

    return errors


def validate_categorical_values(df: pd.DataFrame) -> list[str]:
    """
    Validate that categorical columns have valid values.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to validate.

    Returns
    -------
    list[str]
        List of validation errors.
    """
    errors = []

    # Check sex values
    if "sex" in df.columns:
        invalid = set(df["sex"].dropna().unique()) - VALID_SEX
        if invalid:
            errors.append(f"Invalid sex values: {invalid}")

    # Check income values
    if "income" in df.columns:
        invalid = set(df["income"].dropna().unique()) - VALID_INCOME
        if invalid:
            errors.append(f"Invalid income values: {invalid}")

    # Check split values
    if "split" in df.columns:
        invalid = set(df["split"].dropna().unique()) - VALID_SPLIT
        if invalid:
            errors.append(f"Invalid split values: {invalid}")

    # Check race_binary values
    if "race_binary" in df.columns:
        invalid = set(df["race_binary"].dropna().unique()) - VALID_RACE_BINARY
        if invalid:
            errors.append(f"Invalid race_binary values: {invalid}")

    return errors


def validate_numeric_ranges(df: pd.DataFrame) -> list[str]:
    """
    Validate that numeric columns are within expected ranges.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to validate.

    Returns
    -------
    list[str]
        List of validation errors.
    """
    errors = []

    ranges = {
        "age": (0, 120),
        "education_num": (1, 20),
        "hours_per_week": (0, 168),
        "capital_gain": (0, None),
        "capital_loss": (0, None),
    }

    for col, (min_val, max_val) in ranges.items():
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()

            if min_val is not None and col_min < min_val:
                errors.append(f"Column '{col}' has values below {min_val}: {col_min}")
            if max_val is not None and col_max > max_val:
                errors.append(f"Column '{col}' has values above {max_val}: {col_max}")

    return errors


def validate_dataframe(
    df: pd.DataFrame,
    raise_on_error: bool = True,
) -> list[str]:
    """
    Perform full schema validation on a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to validate.
    raise_on_error : bool
        Raise SchemaValidationError if validation fails.

    Returns
    -------
    list[str]
        List of all validation errors.

    Raises
    ------
    SchemaValidationError
        If raise_on_error is True and validation fails.
    """
    all_errors = []

    all_errors.extend(validate_columns(df))
    all_errors.extend(validate_no_nulls(df))
    all_errors.extend(validate_categorical_values(df))
    all_errors.extend(validate_numeric_ranges(df))

    if raise_on_error and all_errors:
        error_msg = "Schema validation failed:\n" + "\n".join(f"  - {e}" for e in all_errors)
        raise SchemaValidationError(error_msg)

    return all_errors


def validate_inference_input(
    df: pd.DataFrame,
    required_columns: list[str] | None = None,
) -> list[str]:
    """
    Validate input data for inference (doesn't require target/split columns).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    required_columns : list[str], optional
        Required feature columns. Defaults to feature columns only.

    Returns
    -------
    list[str]
        List of validation errors.
    """
    if required_columns is None:
        # Feature columns (exclude target, split, sensitive)
        required_columns = [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education_num",
            "marital_status",
            "occupation",
            "relationship",
            "native_country",
            "capital_gain",
            "capital_loss",
            "hours_per_week",
        ]

    errors = []
    missing = set(required_columns) - set(df.columns)
    if missing:
        errors.append(f"Missing feature columns: {missing}")

    # Check for nulls in features
    for col in required_columns:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                errors.append(f"Feature '{col}' has {null_count} null values")

    return errors


def get_schema_summary(df: pd.DataFrame) -> dict[str, Any]:
    """
    Get a summary of the dataframe schema.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to summarize.

    Returns
    -------
    dict[str, Any]
        Schema summary with column info, counts, etc.
    """
    return {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
    }
