"""Feature preprocessing pipeline for the Adult dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if TYPE_CHECKING:
    from typing import Tuple, List


# Columns that should be excluded from features
EXCLUDE_COLUMNS = ["sex", "race", "race_binary", "income", "split"]


def build_preprocessing_pipeline(
    df: pd.DataFrame,
    exclude_cols: list[str] | None = None,
) -> Tuple[ColumnTransformer, List[str], List[str]]:
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
        exclude_cols = EXCLUDE_COLUMNS

    # Identify column types
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Remove excluded columns
    for col in exclude_cols:
        if col in numeric_cols:
            numeric_cols.remove(col)
        if col in categorical_cols:
            categorical_cols.remove(col)

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
        exclude_cols = EXCLUDE_COLUMNS

    return [col for col in df.columns if col not in exclude_cols]
