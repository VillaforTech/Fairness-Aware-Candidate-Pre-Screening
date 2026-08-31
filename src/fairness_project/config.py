"""
Configuration system for the fairness project.

Supports loading from YAML files and environment variables.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

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
    policy_attribute: str = "sex"
    privileged_value: str = "Male"
    unprivileged_value: str = "Female"

    # Equal Opportunity settings
    eo_base_threshold: float = 0.5
    eo_n_thresholds: int = 101
    eo_search_range: tuple[float, float] = (0.0, 0.5)
    frontier_max_abs_tpr_gap: float = 0.05
    frontier_max_accuracy_loss: float = 0.03
    review_max_automated_error: float = 0.10
    review_min_automated_samples: int = 250
    subgroup_min_support: int = 50
    subgroup_min_class_count: int = 20


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

    schema_version: str = "2.0"
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
        raise ImportError(
            "PyYAML is required for loading config files. Install with: pip install pyyaml"
        ) from None

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must contain a YAML object: {path}")
    return cast(dict[str, Any], payload)


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
    from dataclasses import fields

    config = Config()
    allowed_top_level = {
        "schema_version",
        "data",
        "model",
        "fairness",
        "output",
        "seed",
        "verbose",
        "n_jobs",
    }
    unknown_top_level = sorted(set(data) - allowed_top_level)
    if unknown_top_level:
        raise ValueError(f"Unknown top-level configuration keys: {unknown_top_level}")
    if data.get("schema_version", config.schema_version) != config.schema_version:
        raise ValueError(f"Unsupported configuration schema: {data.get('schema_version')!r}")

    for section_name in ("data", "model", "fairness", "output"):
        if section_name not in data:
            continue
        section_payload = data[section_name]
        if not isinstance(section_payload, dict):
            raise ValueError(f"Configuration section '{section_name}' must be an object")
        section = getattr(config, section_name)
        allowed = {item.name for item in fields(section)}
        unknown = sorted(set(section_payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown keys in configuration section '{section_name}': {unknown}")
        for key, value in section_payload.items():
            if section_name == "fairness" and key == "eo_search_range":
                value = tuple(value)
            setattr(section, key, value)

    for key in ("seed", "verbose", "n_jobs"):
        if key in data:
            setattr(config, key, data[key])
    _validate_config(config)
    return config


def _validate_config(config: Config) -> None:
    """Reject semantically invalid settings before an experiment starts."""
    if config.model.model_type not in {"lr", "rf", "xgb"}:
        raise ValueError("model.model_type must be one of: lr, rf, xgb")
    if not isinstance(config.seed, int) or isinstance(config.seed, bool) or config.seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not 0 < config.data.val_size < 1:
        raise ValueError("data.val_size must be between 0 and 1")
    if config.fairness.policy_attribute not in config.fairness.sensitive_attributes:
        raise ValueError("fairness.policy_attribute must be listed in sensitive_attributes")
    if config.fairness.privileged_value == config.fairness.unprivileged_value:
        raise ValueError("privileged_value and unprivileged_value must differ")
    search_range = config.fairness.eo_search_range
    if len(search_range) != 2 or not 0 <= search_range[0] <= search_range[1] <= 1:
        raise ValueError("fairness.eo_search_range must be an ordered pair within [0, 1]")
    probabilities = {
        "eo_base_threshold": config.fairness.eo_base_threshold,
        "frontier_max_abs_tpr_gap": config.fairness.frontier_max_abs_tpr_gap,
        "frontier_max_accuracy_loss": config.fairness.frontier_max_accuracy_loss,
        "review_max_automated_error": config.fairness.review_max_automated_error,
    }
    for name, value in probabilities.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
            raise ValueError(f"fairness.{name} must be within [0, 1]")
    if config.fairness.eo_n_thresholds < 2:
        raise ValueError("fairness.eo_n_thresholds must be at least 2")
    if config.fairness.subgroup_min_support < 1:
        raise ValueError("fairness.subgroup_min_support must be positive")
    if config.fairness.subgroup_min_class_count < 1:
        raise ValueError("fairness.subgroup_min_class_count must be positive")
    if config.fairness.review_min_automated_samples < 1:
        raise ValueError("fairness.review_min_automated_samples must be positive")


def resolved_config(config: Config) -> dict[str, Any]:
    """Return the canonical JSON-safe configuration recorded in every artifact."""
    from dataclasses import asdict

    return cast(dict[str, Any], _convert_tuples_to_lists(asdict(config)))


def config_sha256(config: Config) -> str:
    """Hash the fully resolved configuration, not only a user-supplied YAML file."""
    payload = json.dumps(resolved_config(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def _convert_tuples_to_lists(obj: Any) -> Any:
    """Recursively convert tuples to lists for YAML serialization."""
    if isinstance(obj, tuple):
        return [_convert_tuples_to_lists(item) for item in obj]
    elif isinstance(obj, list):
        return [_convert_tuples_to_lists(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: _convert_tuples_to_lists(value) for key, value in obj.items()}
    return obj


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
        raise ImportError(
            "PyYAML is required for saving config files. Install with: pip install pyyaml"
        ) from None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclasses to dict
    from dataclasses import asdict

    data = {
        "schema_version": config.schema_version,
        "seed": config.seed,
        "verbose": config.verbose,
        "n_jobs": config.n_jobs,
        "data": asdict(config.data),
        "model": asdict(config.model),
        "fairness": asdict(config.fairness),
        "output": asdict(config.output),
    }

    # Convert tuples to lists for safe YAML serialization
    data = _convert_tuples_to_lists(data)

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
