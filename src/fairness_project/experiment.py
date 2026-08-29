"""Canonical leakage-free experiment and artifact pipeline."""

from __future__ import annotations

import importlib.metadata
import platform
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from fairness_project.config import Config, set_seed
from fairness_project.data.schema import FEATURE_COLUMNS, validate_dataframe
from fairness_project.data.split import create_train_val_test_split
from fairness_project.evaluation.diagnostics import (
    group_diagnostics,
    paired_bootstrap_intervals,
)
from fairness_project.evaluation.evaluate import evaluate_predictions
from fairness_project.fairness.postprocess import apply_thresholds, tune_equal_opportunity
from fairness_project.governance.gate import (
    REPORT_SCHEMA_VERSION,
    GateResult,
    GateThresholds,
    check_gate,
)
from fairness_project.models.artifact import ARTIFACT_SCHEMA_VERSION, save_bundle, sha256_file
from fairness_project.models.train import ModelType, train_model
from fairness_project.provenance import git_state, source_sha256


@dataclass(frozen=True)
class ExperimentResult:
    """Paths and verdict returned by a completed experiment."""

    run_id: str
    run_dir: Path
    report: dict[str, Any]
    gate: GateResult


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _dependency_versions() -> dict[str, str]:
    packages = (
        "fairness-project",
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "joblib",
    )
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _model_kwargs(config: Config, model_type: ModelType) -> dict[str, Any]:
    if model_type == "lr":
        return {"max_iter": config.model.lr_max_iter}
    if model_type == "rf":
        return {
            "n_estimators": config.model.rf_n_estimators,
            "max_depth": config.model.rf_max_depth,
            "n_jobs": config.n_jobs,
        }
    return {
        "n_estimators": config.model.xgb_n_estimators,
        "max_depth": config.model.xgb_max_depth,
        "learning_rate": config.model.xgb_learning_rate,
        "subsample": config.model.xgb_subsample,
        "colsample_bytree": config.model.xgb_colsample_bytree,
        "n_jobs": config.n_jobs,
    }


def _split_cell_counts(df: pd.DataFrame) -> list[dict[str, Any]]:
    counts = (
        df.groupby(["split", "income", "sex", "race_binary"], observed=True)
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["split", "income", "sex", "race_binary"])
    )
    return cast(list[dict[str, Any]], counts.to_dict(orient="records"))


def _diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_rows: pd.DataFrame,
) -> dict[str, Any]:
    intersection = test_rows["sex"].astype(str) + " x " + test_rows["race_binary"].astype(str)
    return {
        "sex": group_diagnostics(y_true, y_pred, test_rows["sex"]),
        "race_binary": group_diagnostics(y_true, y_pred, test_rows["race_binary"]),
        "sex_x_race_binary": group_diagnostics(y_true, y_pred, intersection),
    }


def _validate_run_id(run_id: str) -> str:
    if run_id in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError("run_id may contain only letters, numbers, dots, underscores, and dashes")
    return run_id


def run_experiment(
    *,
    data_path: str | Path,
    output_dir: str | Path = "runs",
    model_type: ModelType = "xgb",
    seed: int = 42,
    val_ratio: float = 0.15,
    run_id: str | None = None,
    config: Config | None = None,
    bootstrap_samples: int = 500,
    gate_thresholds: GateThresholds | None = None,
) -> ExperimentResult:
    """Train, evaluate, gate, and persist one self-describing run."""
    resolved_config = config or Config()
    resolved_config.seed = seed
    resolved_config.model.random_state = seed
    set_seed(seed)

    source = Path(data_path)
    if not source.is_file():
        raise FileNotFoundError(f"Model-ready dataset not found: {source}")
    frame = pd.read_csv(source)
    if "race_binary" not in frame.columns and "race" in frame.columns:
        frame["race_binary"] = np.where(frame["race"] == "White", "White", "Non-White")
    validate_dataframe(frame)
    split_frame = create_train_val_test_split(
        frame,
        val_ratio=val_ratio,
        random_state=seed,
    )

    train_rows = split_frame[split_frame["split"] == "train"].copy()
    val_rows = split_frame[split_frame["split"] == "val"].copy()
    test_rows = split_frame[split_frame["split"] == "test"].copy()
    y_train = (train_rows["income"] == ">50K").astype(int).to_numpy()
    y_val = (val_rows["income"] == ">50K").astype(int).to_numpy()
    y_test = (test_rows["income"] == ">50K").astype(int).to_numpy()

    from fairness_project.features.preprocessing import build_preprocessing_pipeline

    preprocess, _, _ = build_preprocessing_pipeline(train_rows[FEATURE_COLUMNS])
    model = train_model(
        model_type,
        preprocess,
        train_rows[FEATURE_COLUMNS],
        y_train,
        random_state=seed,
        **_model_kwargs(resolved_config, model_type),
    )

    val_probabilities = model.predict_proba(val_rows[FEATURE_COLUMNS])[:, 1]
    test_probabilities = model.predict_proba(test_rows[FEATURE_COLUMNS])[:, 1]
    baseline_predictions = (
        test_probabilities >= resolved_config.fairness.eo_base_threshold
    ).astype(int)
    tuning = tune_equal_opportunity(
        y_val=y_val,
        y_proba_val=val_probabilities,
        sensitive_val=val_rows["sex"].to_numpy(),
        privileged_value="Male",
        unprivileged_value="Female",
        base_threshold=resolved_config.fairness.eo_base_threshold,
        n_thresholds=resolved_config.fairness.eo_n_thresholds,
        search_range=resolved_config.fairness.eo_search_range,
    )
    adjusted_predictions = apply_thresholds(
        y_pred_proba=test_probabilities,
        sensitive_attr=test_rows["sex"].to_numpy(),
        threshold_priv=tuning["threshold_priv"],
        threshold_unpriv=tuning["threshold_unpriv"],
        privileged_value="Male",
        unprivileged_value="Female",
    )

    baseline_metrics = evaluate_predictions(
        y_test,
        baseline_predictions,
        test_rows["sex"].to_numpy(),
        "Male",
        test_probabilities,
    )
    adjusted_metrics = evaluate_predictions(
        y_test,
        adjusted_predictions,
        test_rows["sex"].to_numpy(),
        "Male",
        test_probabilities,
    )

    revision = git_state()
    resolved_run_id = _validate_run_id(
        run_id
        or f"{model_type}-seed-{seed}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    timestamp = datetime.now(timezone.utc).isoformat()
    metadata = {
        "run_id": resolved_run_id,
        "timestamp": timestamp,
        "seed": seed,
        "model_type": model_type,
        "git_commit": revision.commit,
        "dirty_worktree": revision.dirty_worktree,
        "data_path": str(source),
        "data_sha256": sha256_file(source),
        "source_sha256": source_sha256(),
        "python_version": platform.python_version(),
        "dependencies": _dependency_versions(),
    }
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "metadata": metadata,
        "protocol": {
            "dataset": "UCI Adult (1994 Census income classification)",
            "scope": "benchmark evaluation only; not a hiring validity study",
            "official_test_partition_preserved": True,
            "validation_strategy": "joint stratification by income, sex, and race_binary",
            "validation_ratio": val_ratio,
            "threshold_tuning_split": "val",
            "final_evaluation_split": "test",
            "sensitive_attribute": "sex",
            "privileged_group": "Male",
            "unprivileged_group": "Female",
            "feature_columns": FEATURE_COLUMNS,
            "split_counts": split_frame["split"].value_counts().sort_index().to_dict(),
            "split_cell_counts": _split_cell_counts(split_frame),
        },
        "results": {
            "baseline_metrics": baseline_metrics,
            "metrics": adjusted_metrics,
            "thresholds": {
                "privileged": float(tuning["threshold_priv"]),
                "unprivileged": float(tuning["threshold_unpriv"]),
            },
            "validation_tuning": tuning,
            "subgroup_diagnostics": {
                "baseline": _diagnostics(y_test, baseline_predictions, test_rows),
                "adjusted": _diagnostics(y_test, adjusted_predictions, test_rows),
            },
        },
    }
    if bootstrap_samples:
        report["results"]["uncertainty"] = paired_bootstrap_intervals(
            y_true=y_test,
            baseline_pred=baseline_predictions,
            adjusted_pred=adjusted_predictions,
            sensitive=test_rows["sex"].to_numpy(),
            privileged_group="Male",
            samples=bootstrap_samples,
            random_state=seed,
        )

    report = cast(dict[str, Any], _json_safe(report))
    gate_result = check_gate(report, gate_thresholds)
    report["governance"] = gate_result.to_dict()
    policy = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_id": resolved_run_id,
        "serving": {
            "policy_id": "global-threshold-v1",
            "kind": "global_threshold",
            "threshold": resolved_config.fairness.eo_base_threshold,
            "fairness_adjustment_applied": False,
            "scope": "Adult-income classification demo only",
        },
        "offline_evaluation": {
            "policy_id": "one-sided-opportunity-uplift-v1",
            "kind": "group_thresholds",
            "sensitive_attribute": "sex",
            "privileged_value": "Male",
            "unprivileged_value": "Female",
            "thresholds": report["results"]["thresholds"],
            "tuned_on": "validation",
            "experimental_only": True,
            "served_by_api": False,
        },
    }
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": resolved_run_id,
        "model_type": model_type,
        "created_at": timestamp,
        "git_commit": revision.commit,
        "dirty_worktree": revision.dirty_worktree,
        "data_sha256": metadata["data_sha256"],
        "source_sha256": metadata["source_sha256"],
        "model_sha256": "pending",
        "report_sha256": "pending",
        "policy_sha256": "pending",
        "feature_columns": FEATURE_COLUMNS,
        "positive_class": 1,
        "base_threshold": resolved_config.fairness.eo_base_threshold,
        "python_version": metadata["python_version"],
        "dependencies": metadata["dependencies"],
        "split_seed": seed,
        "governance": gate_result.to_dict(),
        "experimental_only": True,
    }
    run_dir = save_bundle(
        run_dir=Path(output_dir) / resolved_run_id,
        model=model,
        manifest=manifest,
        report=report,
        policy=policy,
    )

    predictions = pd.DataFrame(
        {
            "source_index": test_rows.index,
            "y_true": y_test,
            "probability": test_probabilities,
            "baseline_prediction": baseline_predictions,
            "adjusted_prediction": adjusted_predictions,
            "sex": test_rows["sex"].to_numpy(),
            "race_binary": test_rows["race_binary"].to_numpy(),
        }
    )
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    return ExperimentResult(resolved_run_id, run_dir, report, gate_result)
