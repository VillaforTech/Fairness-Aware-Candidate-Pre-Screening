"""Input/Output utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def save_predictions(
    model_name: str,
    y_true: pd.Series | list,
    y_pred: list | pd.Series,
    index: pd.Index | list,
    out_dir: str | Path = "data/predictions",
) -> Path:
    """
    Save predictions to CSV file.

    Parameters
    ----------
    model_name : str
        Name of the model (used in filename).
    y_true : pd.Series | list
        Ground truth labels.
    y_pred : list | pd.Series
        Predictions.
    index : pd.Index | list
        Row indices.
    out_dir : str | Path
        Output directory.

    Returns
    -------
    Path
        Path to saved file.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Handle pandas Series
    if hasattr(y_true, "values"):
        y_true_vals = y_true.values
    else:
        y_true_vals = y_true

    df_out = pd.DataFrame(
        {
            "index": index,
            "y_true": y_true_vals,
            "y_pred": y_pred,
        }
    )

    file_path = out / f"{model_name}_preds.csv"
    df_out.to_csv(file_path, index=False)

    print(f" Saved predictions to {file_path}")
    return file_path


def load_predictions(file_path: str | Path) -> pd.DataFrame:
    """
    Load predictions from CSV file.

    Parameters
    ----------
    file_path : str | Path
        Path to predictions CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: index, y_true, y_pred.
    """
    return pd.read_csv(file_path)


def save_metrics(
    metrics: list[dict[str, Any]],
    out_path: str | Path,
) -> Path:
    """
    Save metrics to CSV file.

    Parameters
    ----------
    metrics : list[dict[str, Any]]
        List of metric dictionaries.
    out_path : str | Path
        Output file path.

    Returns
    -------
    Path
        Path to saved file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(metrics)
    df.to_csv(out_path, index=False)

    print(f"Saved metrics to {out_path}")
    return out_path
