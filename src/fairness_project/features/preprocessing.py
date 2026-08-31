"""Feature preprocessing pipeline for the Adult dataset."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fairness_project.data.schema import (
    CATEGORICAL_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
)

if TYPE_CHECKING:
    pass


# Columns that should be excluded from features
EXCLUDE_COLUMNS = ["sex", "race", "race_binary", "income", "split", "fnlwgt"]


def categorical_oov_evidence(
    train_frame: pd.DataFrame,
    evaluated_splits: Mapping[str, pd.DataFrame],
    categorical_columns: Sequence[str] = CATEGORICAL_FEATURE_COLUMNS,
) -> dict[str, Any]:
    """Audit split-level categorical values against the fitted training vocabulary."""
    columns = list(categorical_columns)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("categorical_columns must be a nonempty unique sequence")
    missing_train = [column for column in columns if column not in train_frame.columns]
    if missing_train:
        raise ValueError(f"Training frame is missing categorical columns: {missing_train}")
    if train_frame.empty:
        raise ValueError("Training frame must not be empty")

    vocabularies = {
        column: {str(value) for value in train_frame[column].tolist()} for column in columns
    }
    split_evidence: dict[str, Any] = {}
    for split_name, frame in evaluated_splits.items():
        if not isinstance(split_name, str) or not split_name:
            raise ValueError("Evaluated split names must be nonempty strings")
        if frame.empty:
            raise ValueError(f"Evaluated split {split_name!r} must not be empty")
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Evaluated split {split_name!r} is missing columns: {missing}")

        any_oov = pd.Series(False, index=frame.index, dtype=bool)
        column_evidence: dict[str, Any] = {}
        for column in columns:
            normalized = frame[column].map(str)
            oov_mask = ~normalized.isin(vocabularies[column])
            any_oov |= oov_mask
            unknown_values = sorted(set(normalized[oov_mask].tolist()))
            affected_rows = int(oov_mask.sum())
            column_evidence[column] = {
                "training_vocabulary_size": len(vocabularies[column]),
                "unknown_distinct_values": len(unknown_values),
                "unknown_values": unknown_values,
                "affected_rows": affected_rows,
                "affected_share": affected_rows / len(frame),
            }
        rows_with_any_oov = int(any_oov.sum())
        split_evidence[split_name] = {
            "row_count": len(frame),
            "rows_with_any_oov": rows_with_any_oov,
            "rows_with_any_oov_share": rows_with_any_oov / len(frame),
            "columns": column_evidence,
        }

    return {
        "schema_version": "1.0",
        "reference_split": "train",
        "evaluation_behavior": "one-hot ignore",
        "serving_behavior": "reject before scoring",
        "splits": split_evidence,
    }


def build_preprocessing_pipeline(
    df: pd.DataFrame,
    exclude_cols: list[str] | None = None,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """
    Build a sklearn preprocessing pipeline for the Adult dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Training dataframe to infer column types from.
    exclude_cols : list[str], optional
        Columns to exclude from preprocessing. Defaults to sensitive/target columns.

    Returns
    -------
    Tuple[ColumnTransformer, List[str], List[str]]
        - Preprocessing transformer
        - List of numeric column names
        - List of categorical column names
    """
    if exclude_cols is None:
        requested_columns = FEATURE_COLUMNS
    else:
        requested_columns = [column for column in df.columns if column not in exclude_cols]
    missing = [column for column in requested_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Cannot build preprocessing contract; missing columns: {missing}")

    if exclude_cols is None:
        numeric_cols = list(NUMERIC_FEATURE_COLUMNS)
        categorical_cols = list(CATEGORICAL_FEATURE_COLUMNS)
        mismatched_numeric = [column for column in numeric_cols if not is_numeric_dtype(df[column])]
        mismatched_categorical = [
            column for column in categorical_cols if is_numeric_dtype(df[column])
        ]
        if mismatched_numeric or mismatched_categorical:
            raise ValueError(
                "Feature dtypes do not match the canonical preprocessing contract; "
                f"non_numeric={mismatched_numeric}, numeric_categorical={mismatched_categorical}"
            )
    else:
        numeric_cols = [column for column in requested_columns if is_numeric_dtype(df[column])]
        categorical_cols = [column for column in requested_columns if column not in numeric_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    return preprocess, numeric_cols, categorical_cols


def get_feature_columns(df: pd.DataFrame, exclude_cols: list[str] | None = None) -> list[str]:
    """
    Get list of feature columns (excluding sensitive/target columns).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to extract column names from.
    exclude_cols : list[str], optional
        Columns to exclude. Defaults to sensitive/target columns.

    Returns
    -------
    list[str]
        List of feature column names.
    """
    if exclude_cols is None:
        missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"Missing canonical feature columns: {missing}")
        return list(FEATURE_COLUMNS)

    return [col for col in df.columns if col not in exclude_cols]
