"""Deterministic train/validation/test split utilities."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_STRATIFY_COLUMNS = ("income", "sex", "race_binary")


def create_train_val_test_split(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    random_state: int = 42,
    stratify_columns: Sequence[str] = DEFAULT_STRATIFY_COLUMNS,
) -> pd.DataFrame:
    """Carve validation rows from the original training partition.

    The bundled Adult files already define the external test partition. This
    function leaves it untouched and stratifies the validation draw jointly by
    target and protected-group columns so small groups are not accidentally
    removed or heavily distorted by one random draw.
    """
    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")
    if "split" not in df.columns:
        raise ValueError("Dataframe must contain a 'split' column")
    if (df["split"] == "val").any():
        raise ValueError("Validation rows already exist; start from the original train/test data")

    missing = [column for column in stratify_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing stratification columns: {missing}")

    train_indices = df.index[df["split"] == "train"]
    if train_indices.empty:
        raise ValueError("No rows with split='train' were found")
    if not (df["split"] == "test").any():
        raise ValueError("No rows with split='test' were found")

    train_rows = df.loc[train_indices]
    strata = train_rows[list(stratify_columns)].astype(str).agg("|".join, axis=1)
    sparse_strata = strata.value_counts()[lambda counts: counts < 2]
    if not sparse_strata.empty:
        raise ValueError(
            "Every target/group stratum needs at least two training rows; "
            f"found sparse strata: {sparse_strata.to_dict()}"
        )

    _, val_indices = train_test_split(
        train_indices.to_numpy(),
        test_size=val_ratio,
        random_state=random_state,
        stratify=strata.to_numpy(),
    )

    result = df.copy()
    result.loc[val_indices, "split"] = "val"
    return result
