"""
Data download utilities for the Adult dataset.

The Adult (Census Income) dataset is from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/2/adult

License: CC BY 4.0 (Creative Commons Attribution 4.0 International)
Citation: Becker, B., & Kohavi, R. (1996). Adult [Dataset].
          UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

# UCI Adult dataset URLs
ADULT_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
ADULT_TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
ADULT_NAMES_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names"

# Column names for the Adult dataset
COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]


def download_file(url: str, dest_path: Path, verbose: bool = True) -> Path:
    """
    Download a file from URL to destination path.

    Parameters
    ----------
    url : str
        URL to download from.
    dest_path : Path
        Destination file path.
    verbose : bool
        Print progress messages.

    Returns
    -------
    Path
        Path to downloaded file.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Downloading {url}...")

    try:
        urllib.request.urlretrieve(url, dest_path)
        if verbose:
            print(f"Saved to {dest_path}")
        return dest_path
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}") from e


def download_adult_dataset(
    output_dir: str | Path = "data/raw/adult",
    verbose: bool = True,
) -> dict[str, Path]:
    """
    Download the Adult dataset from UCI repository.

    Parameters
    ----------
    output_dir : str | Path
        Directory to save downloaded files.
    verbose : bool
        Print progress messages.

    Returns
    -------
    dict[str, Path]
        Dictionary mapping file types to their paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {}

    # Download training data
    train_path = output_dir / "adult.data"
    if not train_path.exists():
        files["train"] = download_file(ADULT_DATA_URL, train_path, verbose)
    else:
        if verbose:
            print(f"Training data already exists: {train_path}")
        files["train"] = train_path

    # Download test data
    test_path = output_dir / "adult.test"
    if not test_path.exists():
        files["test"] = download_file(ADULT_TEST_URL, test_path, verbose)
    else:
        if verbose:
            print(f"Test data already exists: {test_path}")
        files["test"] = test_path

    # Download column names/documentation
    names_path = output_dir / "adult.names"
    if not names_path.exists():
        files["names"] = download_file(ADULT_NAMES_URL, names_path, verbose)
    else:
        if verbose:
            print(f"Column names already exist: {names_path}")
        files["names"] = names_path

    if verbose:
        print("\nAdult dataset download complete!")
        print(f"Files saved to: {output_dir}")

    return files


def main() -> None:
    """CLI entry point for data download."""
    import argparse

    parser = argparse.ArgumentParser(description="Download Adult dataset")
    parser.add_argument(
        "--output-dir",
        "-o",
        default="data/raw/adult",
        help="Output directory",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress output",
    )
    args = parser.parse_args()

    download_adult_dataset(args.output_dir, verbose=not args.quiet)


if __name__ == "__main__":
    main()
