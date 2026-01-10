"""
Configuration system for the fairness project.

Supports loading from YAML files and environment variables.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Try pydantic, fall back to dataclasses
try:
    from pydantic import BaseModel, Field

    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = None
    Field = None


# ============================================================================
# DEFAULT PATHS
# ============================================================================

DEFAULT_DATA_PATH = "data/processed/adult/adult_model_ready.csv"
DEFAULT_RAW_DATA_DIR = "data/raw/adult"
DEFAULT_PROCESSED_DATA_DIR = "data/processed/adult"
DEFAULT_PREDICTIONS_DIR = "data/predictions"
DEFAULT_METRICS_DIR = "data/metrics"
DEFAULT_PLOTS_DIR = "data/plots"
DEFAULT_RUNS_DIR = "runs"
DEFAULT_MODELS_DIR = "models"
DEFAULT_CONFIG_DIR = "configs"


# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

@dataclass
class DataConfig:
    """Data configuration."""

    raw_data_dir: str = DEFAULT_RAW_DATA_DIR
    processed_data_dir: str = DEFAULT_PROCESSED_DATA_DIR
    model_ready_path: str = DEFAULT_DATA_PATH
    test_size: float = 0.2
    val_size: float = 0.15  # Validation split from training data


@dataclass
class ModelConfig:
    """Model training configuration."""

    # General
    model_type: str = "xgb"  # lr, rf, xgb
    random_state: int = 42

    # Logistic Regression
    lr_max_iter: int = 500

    # Random Forest
    rf_n_estimators: int = 300
    rf_max_depth: int | None = None

    # XGBoost
    xgb_n_estimators: int = 300
    xgb_max_depth: int = 4
    xgb_learning_rate: float = 0.1
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8


@dataclass
class FairnessConfig:
    """Fairness configuration."""

    # Sensitive attributes
    sensitive_attributes: list[str] = field(default_factory=lambda: ["sex", "race_binary"])
    privileged_groups: dict[str, str] = field(
        default_factory=lambda: {"sex": "Male", "race_binary": "White"}
    )

    # Equal Opportunity settings
    eo_base_threshold: float = 0.5
    eo_n_thresholds: int = 101
    eo_search_range: tuple[float, float] = (0.0, 0.5)


@dataclass
class OutputConfig:
    """Output paths configuration."""

    predictions_dir: str = DEFAULT_PREDICTIONS_DIR
    metrics_dir: str = DEFAULT_METRICS_DIR
    plots_dir: str = DEFAULT_PLOTS_DIR
    runs_dir: str = DEFAULT_RUNS_DIR
    models_dir: str = DEFAULT_MODELS_DIR


@dataclass
class Config:
    """Main configuration class."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    fairness: FairnessConfig = field(default_factory=FairnessConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Global settings
    seed: int = 42
    verbose: bool = True
    n_jobs: int = -1


# ============================================================================
# SEED MANAGEMENT
# ============================================================================

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Set sklearn random state through numpy
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Try to set torch seed if available
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_random_state(config: Config | None = None) -> int:
    """
    Get random state from config or environment.

    Parameters
    ----------
    config : Config, optional
        Configuration object.

    Returns
    -------
    int
        Random state value.
    """
    if config is not None:
        return config.seed

    # Check environment variable
    env_seed = os.environ.get("FAIRNESS_SEED")
    if env_seed is not None:
        return int(env_seed)

    return 42  # Default


# ============================================================================
# CONFIG LOADING
# ============================================================================

def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """
    Load configuration from YAML file.

    Parameters
    ----------
    path : str | Path
        Path to YAML configuration file.

    Returns
    -------
    dict[str, Any]
        Configuration dictionary.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for loading config files. Install with: pip install pyyaml")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        return yaml.safe_load(f)


def config_from_dict(data: dict[str, Any]) -> Config:
    """
    Create Config from dictionary.

    Parameters
    ----------
    data : dict[str, Any]
        Configuration dictionary.

    Returns
    -------
    Config
        Configuration object.
    """
    config = Config()

    # Update nested configs
    if "data" in data:
        for key, value in data["data"].items():
            if hasattr(config.data, key):
                setattr(config.data, key, value)

    if "model" in data:
        for key, value in data["model"].items():
            if hasattr(config.model, key):
                setattr(config.model, key, value)

    if "fairness" in data:
        for key, value in data["fairness"].items():
            if hasattr(config.fairness, key):
                setattr(config.fairness, key, value)

    if "output" in data:
        for key, value in data["output"].items():
            if hasattr(config.output, key):
                setattr(config.output, key, value)

    # Update top-level settings
    for key in ["seed", "verbose", "n_jobs"]:
        if key in data:
            setattr(config, key, data[key])

    return config


def load_config(path: str | Path | None = None) -> Config:
    """
    Load configuration from file or return defaults.

    Parameters
    ----------
    path : str | Path | None
        Path to configuration file. If None, returns default config.

    Returns
    -------
    Config
        Configuration object.
    """
    if path is None:
        return Config()

    data = load_yaml_config(path)
    return config_from_dict(data)


def save_config(config: Config, path: str | Path) -> None:
    """
    Save configuration to YAML file.

    Parameters
    ----------
    config : Config
        Configuration object.
    path : str | Path
        Path to save configuration file.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for saving config files. Install with: pip install pyyaml")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclasses to dict
    from dataclasses import asdict

    data = {
        "seed": config.seed,
        "verbose": config.verbose,
        "n_jobs": config.n_jobs,
        "data": asdict(config.data),
        "model": asdict(config.model),
        "fairness": asdict(config.fairness),
        "output": asdict(config.output),
    }

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# ============================================================================
# GLOBAL CONFIG INSTANCE
# ============================================================================

_global_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _global_config
    _global_config = config
    set_seed(config.seed)


def init_config(config_path: str | Path | None = None, seed: int | None = None) -> Config:
    """
    Initialize global configuration.

    Parameters
    ----------
    config_path : str | Path | None
        Path to configuration file.
    seed : int | None
        Override seed value.

    Returns
    -------
    Config
        Initialized configuration.
    """
    config = load_config(config_path)

    if seed is not None:
        config.seed = seed
        config.model.random_state = seed

    set_config(config)
    return config
