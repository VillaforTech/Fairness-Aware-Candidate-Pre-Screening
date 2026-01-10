"""Path utilities and constants."""

from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns
    -------
    Path
        Path to project root (directory containing pyproject.toml).
    """
    current = Path.cwd()

    # Walk up to find pyproject.toml
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent

    # Fallback to cwd
    return current


def get_data_dir() -> Path:
    """Get the data directory."""
    return get_project_root() / "data"


def get_raw_data_dir() -> Path:
    """Get the raw data directory."""
    return get_data_dir() / "raw"


def get_processed_data_dir() -> Path:
    """Get the processed data directory."""
    return get_data_dir() / "processed"


def get_predictions_dir() -> Path:
    """Get the predictions directory."""
    return get_data_dir() / "predictions"


def get_metrics_dir() -> Path:
    """Get the metrics directory."""
    return get_data_dir() / "metrics"


def get_plots_dir() -> Path:
    """Get the plots directory."""
    return get_data_dir() / "plots"


def get_runs_dir() -> Path:
    """Get the runs directory for experiment tracking."""
    return get_project_root() / "runs"


def get_models_dir() -> Path:
    """Get the models directory."""
    return get_project_root() / "models"


def get_configs_dir() -> Path:
    """Get the configs directory."""
    return get_project_root() / "configs"


# Default paths for Adult dataset
DEFAULT_MODEL_READY_PATH = "data/processed/adult/adult_model_ready.csv"


def ensure_dir(path: Path) -> Path:
    """
    Ensure directory exists.

    Parameters
    ----------
    path : Path
        Directory path.

    Returns
    -------
    Path
        The same path after ensuring it exists.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
