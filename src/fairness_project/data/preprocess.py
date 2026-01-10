"""
Data preprocessing for the Adult dataset.

Handles loading raw data, cleaning, and creating the model-ready dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .download import COLUMN_NAMES


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

    # Drop rows with missing values
    df = df.dropna()

    # Standardize string columns (strip whitespace)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()

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

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    if verbose:
        print(f"\nModel-ready data saved to: {output_path}")
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
    df = df.copy()
    np.random.seed(random_state)

    # Only modify training data
    train_mask = df["split"] == "train"
    n_train = train_mask.sum()
    n_val = int(n_train * val_size)

    # Random selection for validation
    train_indices = df[train_mask].index.tolist()
    val_indices = np.random.choice(train_indices, size=n_val, replace=False)

    df.loc[val_indices, "split"] = "val"

    return df


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
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
