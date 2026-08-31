"""
Data preprocessing for the Adult dataset.

Handles loading raw data, cleaning, and creating the model-ready dataset.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

from .download import COLUMN_NAMES
from .quality import audit_processed_quality, audit_raw_attrition
from .schema import validate_dataframe


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_raw_data(
    train_path: str | Path,
    test_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw Adult dataset files.

    Parameters
    ----------
    train_path : str | Path
        Path to adult.data file.
    test_path : str | Path
        Path to adult.test file.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Training and test dataframes.
    """
    # Load training data
    df_train = pd.read_csv(
        train_path,
        names=COLUMN_NAMES,
        sep=r",\s*",
        engine="python",
        na_values="?",
    )

    # Load test data (skip first row which contains a comment)
    df_test = pd.read_csv(
        test_path,
        names=COLUMN_NAMES,
        sep=r",\s*",
        engine="python",
        na_values="?",
        skiprows=1,
    )

    # Clean income labels in test set (they have a trailing period)
    df_test["income"] = df_test["income"].str.rstrip(".")

    return df_train, df_test


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Adult dataset.

    - Remove rows with missing values
    - Standardize string values
    - Create binary target variable

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    df = df.copy()

    # Normalize string missing markers before complete-case deletion. This also
    # catches whitespace-padded question marks instead of silently retaining them.
    for col in df.columns:
        if is_object_dtype(df[col].dtype) or is_string_dtype(df[col].dtype):
            df[col] = df[col].str.strip().replace({"?": pd.NA, "": pd.NA})

    df = df.dropna()

    # Ensure income is standardized
    df["income"] = df["income"].replace({">50K.": ">50K", "<=50K.": "<=50K"})

    return df


def create_binary_race(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary race column (White vs Non-White).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'race' column.

    Returns
    -------
    pd.DataFrame
        Dataframe with added 'race_binary' column.
    """
    df = df.copy()
    df["race_binary"] = df["race"].apply(lambda x: "White" if x == "White" else "Non-White")
    return df


def prepare_model_ready_data(
    train_path: str | Path = "data/raw/adult/adult.data",
    test_path: str | Path = "data/raw/adult/adult.test",
    output_path: str | Path = "data/processed/adult/adult_model_ready.csv",
    quality_report_path: str | Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Prepare the model-ready dataset from raw Adult data.

    Parameters
    ----------
    train_path : str | Path
        Path to training data.
    test_path : str | Path
        Path to test data.
    output_path : str | Path
        Path to save processed data.
    verbose : bool
        Print progress messages.

    Returns
    -------
    pd.DataFrame
        Combined model-ready dataframe.
    """
    if verbose:
        print(f"Loading raw data from {train_path} and {test_path}...")

    df_train, df_test = load_raw_data(train_path, test_path)
    raw_quality = audit_raw_attrition(df_train, df_test)

    if verbose:
        print(f"Raw train shape: {df_train.shape}")
        print(f"Raw test shape: {df_test.shape}")

    # Clean data
    if verbose:
        print("Cleaning data...")

    df_train = clean_data(df_train)
    df_test = clean_data(df_test)

    if verbose:
        print(f"Clean train shape: {df_train.shape}")
        print(f"Clean test shape: {df_test.shape}")

    # Add split column
    df_train["split"] = "train"
    df_test["split"] = "test"

    # Combine
    df = pd.concat([df_train, df_test], ignore_index=True)

    # Create binary race
    df = create_binary_race(df)
    validate_dataframe(df)
    processed_quality = audit_processed_quality(df)

    # Save the data and its semantics report as deterministic, replace-only artifacts.
    output_path = Path(output_path)
    _write_csv_atomic(output_path, df)
    resolved_quality_path = (
        Path(quality_report_path)
        if quality_report_path is not None
        else output_path.with_suffix(".quality.json")
    )
    _write_json_atomic(
        resolved_quality_path,
        {
            "schema_version": "1.0",
            "audit_type": "adult_preprocessing_evidence",
            "raw": raw_quality,
            "processed": processed_quality,
            "model_ready": {
                "filename": output_path.name,
                "sha256": _sha256_file(output_path),
                "row_count": len(df),
            },
            "sources": {
                "train": {
                    "filename": Path(train_path).name,
                    "sha256": _sha256_file(Path(train_path)),
                },
                "test": {
                    "filename": Path(test_path).name,
                    "sha256": _sha256_file(Path(test_path)),
                },
            },
        },
    )

    if verbose:
        print(f"\nModel-ready data saved to: {output_path}")
        print(f"Data-semantics evidence saved to: {resolved_quality_path}")
        print(f"Total samples: {len(df)}")
        print(f"Train samples: {(df['split'] == 'train').sum()}")
        print(f"Test samples: {(df['split'] == 'test').sum()}")
        print("\nIncome distribution:")
        print(df["income"].value_counts(normalize=True))
        print("\nSex distribution:")
        print(df["sex"].value_counts(normalize=True))

    return df


def create_validation_split(
    df: pd.DataFrame,
    val_size: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Create a validation split from training data.

    This is needed for leakage-free EO threshold tuning.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'split' column.
    val_size : float
        Proportion of training data to use for validation.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Dataframe with updated 'split' column (train/val/test).
    """
    from fairness_project.data.split import create_train_val_test_split

    return create_train_val_test_split(
        df,
        val_ratio=val_size,
        random_state=random_state,
    )


def main() -> None:
    """CLI entry point for data preprocessing."""
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess Adult dataset")
    parser.add_argument(
        "--train-path",
        default="data/raw/adult/adult.data",
        help="Path to training data",
    )
    parser.add_argument(
        "--test-path",
        default="data/raw/adult/adult.test",
        help="Path to test data",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        default="data/processed/adult/adult_model_ready.csv",
        help="Output path for processed data",
    )
    parser.add_argument(
        "--quality-report-path",
        default=None,
        help="Output path for the data-semantics JSON sidecar",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress output",
    )
    args = parser.parse_args()

    prepare_model_ready_data(
        train_path=args.train_path,
        test_path=args.test_path,
        output_path=args.output_path,
        quality_report_path=args.quality_report_path,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
