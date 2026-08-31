"""Canonical leakage-free experiment and artifact pipeline."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from fairness_project.config import (
    Config,
    config_from_dict,
    config_sha256,
    set_seed,
)
from fairness_project.config import (
    resolved_config as config_payload,
)
from fairness_project.data.quality import audit_processed_quality
from fairness_project.data.schema import (
    FEATURE_COLUMNS,
    FEATURE_CONTRACT_ID,
    REQUIRED_COLUMNS,
    SAMPLE_WEIGHT_COLUMN,
    validate_dataframe,
)
from fairness_project.data.split import create_train_val_test_split
from fairness_project.evaluation.diagnostics import (
    group_diagnostics,
    paired_bootstrap_intervals,
)
from fairness_project.evaluation.evaluate import evaluate_predictions
from fairness_project.evaluation.intersectional import intersectional_diagnostics
from fairness_project.evaluation.overlap import (
    exact_feature_overlap_mask,
    exact_feature_overlap_sensitivity,
)
from fairness_project.fairness.frontier import optimize_validation_policy_frontier
from fairness_project.fairness.postprocess import apply_thresholds, tune_equal_opportunity
from fairness_project.fairness.selective import evaluate_review_band, select_review_band
from fairness_project.governance.gate import (
    REPORT_SCHEMA_VERSION,
    GateResult,
    GateThresholds,
    check_gate,
)
from fairness_project.models.artifact import ARTIFACT_SCHEMA_VERSION, save_bundle, sha256_file
from fairness_project.models.train import ModelType, train_model
from fairness_project.monitoring import build_snapshot
from fairness_project.provenance import git_state, source_sha256
from fairness_project.reporting import render_audit_html


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


def _validation_dependence_evidence(
    *,
    train_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    validation_labels: np.ndarray,
    validation_probabilities: np.ndarray,
    config: Config,
) -> dict[str, Any]:
    """Retune after excluding validation rows repeated in the fit partition."""
    feature_overlap = exact_feature_overlap_mask(
        reference_rows=train_rows,
        compared_rows=validation_rows,
    )
    exact_record_columns = [column for column in REQUIRED_COLUMNS if column != "split"]
    train_records = set(train_rows[exact_record_columns].itertuples(index=False, name=None))
    record_overlap = np.fromiter(
        (
            tuple(row) in train_records
            for row in validation_rows[exact_record_columns].itertuples(
                index=False,
                name=None,
            )
        ),
        dtype=bool,
        count=len(validation_rows),
    )
    novel = ~feature_overlap
    feature_overlap_rows = int(feature_overlap.sum())
    record_overlap_rows = int(record_overlap.sum())
    counts = {
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "exact_feature_overlap_rows": feature_overlap_rows,
        "exact_feature_overlap_rate": feature_overlap_rows / len(validation_rows),
        "exact_full_record_overlap_rows": record_overlap_rows,
        "exact_full_record_overlap_rate": record_overlap_rows / len(validation_rows),
        "overlap_excluded_validation_rows": int(novel.sum()),
    }
    if not novel.any():
        return {
            "schema_version": "1.0",
            "counts": counts,
            "overlap_excluded_retuning": {
                "status": "not_estimable",
                "reason": "no_validation_rows_remain_after_exact_feature_overlap_exclusion",
            },
        }

    try:
        frontier = optimize_validation_policy_frontier(
            y_true=validation_labels[novel],
            probabilities=validation_probabilities[novel],
            groups=validation_rows.loc[novel, config.fairness.policy_attribute].to_numpy(),
            privileged_value=config.fairness.privileged_value,
            unprivileged_value=config.fairness.unprivileged_value,
            grid_size=config.fairness.eo_n_thresholds,
            global_threshold=config.fairness.eo_base_threshold,
            max_abs_tpr_gap=config.fairness.frontier_max_abs_tpr_gap,
            max_accuracy_loss=config.fairness.frontier_max_accuracy_loss,
        )
        review = select_review_band(
            y_true_validation=validation_labels[novel],
            y_proba_validation=validation_probabilities[novel],
            base_threshold=config.fairness.eo_base_threshold,
            max_automated_error_rate=config.fairness.review_max_automated_error,
            min_automated_samples=min(
                config.fairness.review_min_automated_samples,
                int(novel.sum()),
            ),
        )
    except ValueError as exc:
        return {
            "schema_version": "1.0",
            "counts": counts,
            "overlap_excluded_retuning": {
                "status": "not_estimable",
                "reason": str(exc),
            },
        }

    return {
        "schema_version": "1.0",
        "counts": counts,
        "overlap_excluded_retuning": {
            "status": "completed",
            "frontier_selection_status": frontier.status,
            "selected_frontier_policy": (
                asdict(frontier.selected) if frontier.selected is not None else None
            ),
            "review_policy": review.to_dict(include_candidates=False),
            "statement": (
                "The model and validation probabilities stay fixed; only policy selection "
                "is repeated after exact-feature overlap exclusion."
            ),
        },
    }


def _diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_rows: pd.DataFrame,
) -> dict[str, Any]:
    intersection = test_rows["sex"].astype(str) + " x " + test_rows["race_binary"].astype(str)
    detailed_intersection = test_rows["sex"].astype(str) + " x " + test_rows["race"].astype(str)
    return {
        "sex": group_diagnostics(y_true, y_pred, test_rows["sex"]),
        "race": group_diagnostics(y_true, y_pred, test_rows["race"]),
        "race_binary": group_diagnostics(y_true, y_pred, test_rows["race_binary"]),
        "sex_x_race_binary": group_diagnostics(y_true, y_pred, intersection),
        "sex_x_race": group_diagnostics(y_true, y_pred, detailed_intersection),
    }


def _validate_run_id(run_id: str) -> str:
    if run_id in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError("run_id may contain only letters, numbers, dots, underscores, and dashes")
    return run_id


def _data_semantics_evidence(
    source: Path,
    frame: pd.DataFrame,
) -> tuple[dict[str, Any], str | None]:
    """Bind an optional preprocessing sidecar to a fresh processed-data audit."""

    processed = audit_processed_quality(frame)
    sidecar_path = source.with_suffix(".quality.json")
    if not sidecar_path.is_file():
        return {
            "source": "computed_from_model_ready_csv",
            "sidecar_sha256": None,
            "raw": None,
            "processed": processed,
        }, None

    try:
        with sidecar_path.open(encoding="utf-8") as handle:
            sidecar = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read data-quality sidecar {sidecar_path}: {exc}") from exc
    if not isinstance(sidecar, dict) or sidecar.get("audit_type") != "adult_preprocessing_evidence":
        raise ValueError("Data-quality sidecar has an unsupported structure")
    model_ready = sidecar.get("model_ready")
    if not isinstance(model_ready, dict) or model_ready.get("sha256") != sha256_file(source):
        raise ValueError("Data-quality sidecar does not match the model-ready CSV digest")
    if sidecar.get("processed") != processed:
        raise ValueError("Data-quality sidecar processed audit does not match the loaded CSV")
    raw = sidecar.get("raw")
    if not isinstance(raw, dict):
        raise ValueError("Data-quality sidecar is missing its raw attrition audit")
    digest = sha256_file(sidecar_path)
    return {
        "source": "bound_preprocessing_sidecar",
        "sidecar_filename": sidecar_path.name,
        "sidecar_sha256": digest,
        "raw": raw,
        "processed": processed,
        "sources": sidecar.get("sources"),
    }, digest


def run_experiment(
    *,
    data_path: str | Path,
    output_dir: str | Path = "runs",
    model_type: ModelType | None = None,
    seed: int = 42,
    val_ratio: float | None = None,
    run_id: str | None = None,
    config: Config | None = None,
    bootstrap_samples: int = 500,
    gate_thresholds: GateThresholds | None = None,
) -> ExperimentResult:
    """Train, evaluate, gate, and persist one self-describing run."""
    resolved_config = config_from_dict(config_payload(config or Config()))
    model_type = cast(ModelType, model_type or resolved_config.model.model_type)
    effective_val_ratio = resolved_config.data.val_size if val_ratio is None else float(val_ratio)
    if not 0 < effective_val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    resolved_config.seed = seed
    resolved_config.model.random_state = seed
    resolved_config.model.model_type = model_type
    resolved_config.data.val_size = effective_val_ratio
    set_seed(seed)
    policy_attribute = resolved_config.fairness.policy_attribute
    privileged_value = resolved_config.fairness.privileged_value
    unprivileged_value = resolved_config.fairness.unprivileged_value

    source = Path(data_path)
    if not source.is_file():
        raise FileNotFoundError(f"Model-ready dataset not found: {source}")
    frame = pd.read_csv(source)
    validate_dataframe(frame)
    data_semantics, data_quality_digest = _data_semantics_evidence(source, frame)
    split_frame = create_train_val_test_split(
        frame,
        val_ratio=effective_val_ratio,
        random_state=seed,
    )
    observed_policy_groups = set(split_frame[policy_attribute].astype(str).unique())
    expected_policy_groups = {privileged_value, unprivileged_value}
    if observed_policy_groups != expected_policy_groups:
        raise ValueError(
            f"Policy attribute {policy_attribute!r} must contain exactly "
            f"{sorted(expected_policy_groups)}, got {sorted(observed_policy_groups)}"
        )

    train_rows = split_frame[split_frame["split"] == "train"].copy()
    val_rows = split_frame[split_frame["split"] == "val"].copy()
    test_rows = split_frame[split_frame["split"] == "test"].copy()
    y_train = (train_rows["income"] == ">50K").astype(int).to_numpy()
    y_val = (val_rows["income"] == ">50K").astype(int).to_numpy()
    y_test = (test_rows["income"] == ">50K").astype(int).to_numpy()

    from fairness_project.features.preprocessing import (
        build_preprocessing_pipeline,
        categorical_oov_evidence,
    )

    preprocess, numeric_features, categorical_features = build_preprocessing_pipeline(
        train_rows[FEATURE_COLUMNS]
    )
    oov_evidence = categorical_oov_evidence(
        train_rows,
        {"validation": val_rows, "test": test_rows},
        categorical_features,
    )
    model = train_model(
        model_type,
        preprocess,
        train_rows[FEATURE_COLUMNS],
        y_train,
        random_state=seed,
        **_model_kwargs(resolved_config, model_type),
    )
    transformed_feature_names = [
        str(value) for value in model.named_steps["preprocess"].get_feature_names_out()
    ]

    val_probabilities = model.predict_proba(val_rows[FEATURE_COLUMNS])[:, 1]
    test_probabilities = model.predict_proba(test_rows[FEATURE_COLUMNS])[:, 1]
    baseline_predictions = (
        test_probabilities >= resolved_config.fairness.eo_base_threshold
    ).astype(int)
    tuning = tune_equal_opportunity(
        y_val=y_val,
        y_proba_val=val_probabilities,
        sensitive_val=val_rows[policy_attribute].to_numpy(),
        privileged_value=privileged_value,
        unprivileged_value=unprivileged_value,
        base_threshold=resolved_config.fairness.eo_base_threshold,
        n_thresholds=resolved_config.fairness.eo_n_thresholds,
        search_range=resolved_config.fairness.eo_search_range,
    )
    frontier_result = optimize_validation_policy_frontier(
        y_true=y_val,
        probabilities=val_probabilities,
        groups=val_rows[policy_attribute].to_numpy(),
        privileged_value=privileged_value,
        unprivileged_value=unprivileged_value,
        grid_size=resolved_config.fairness.eo_n_thresholds,
        global_threshold=resolved_config.fairness.eo_base_threshold,
        max_abs_tpr_gap=resolved_config.fairness.frontier_max_abs_tpr_gap,
        max_accuracy_loss=resolved_config.fairness.frontier_max_accuracy_loss,
    )
    review_policy = select_review_band(
        y_true_validation=y_val,
        y_proba_validation=val_probabilities,
        base_threshold=resolved_config.fairness.eo_base_threshold,
        max_automated_error_rate=resolved_config.fairness.review_max_automated_error,
        min_automated_samples=min(
            resolved_config.fairness.review_min_automated_samples,
            len(y_val),
        ),
    )
    validation_dependence = _validation_dependence_evidence(
        train_rows=train_rows,
        validation_rows=val_rows,
        validation_labels=y_val,
        validation_probabilities=val_probabilities,
        config=resolved_config,
    )
    selected_frontier_policy = frontier_result.selected or frontier_result.baseline
    adjusted_predictions = apply_thresholds(
        y_pred_proba=test_probabilities,
        sensitive_attr=test_rows[policy_attribute].to_numpy(),
        threshold_priv=selected_frontier_policy.threshold_privileged,
        threshold_unpriv=selected_frontier_policy.threshold_unprivileged,
        privileged_value=privileged_value,
        unprivileged_value=unprivileged_value,
    )

    baseline_metrics = evaluate_predictions(
        y_test,
        baseline_predictions,
        test_rows[policy_attribute].to_numpy(),
        privileged_value,
        test_probabilities,
    )
    adjusted_metrics = evaluate_predictions(
        y_test,
        adjusted_predictions,
        test_rows[policy_attribute].to_numpy(),
        privileged_value,
        test_probabilities,
    )
    test_weights = test_rows[SAMPLE_WEIGHT_COLUMN].to_numpy(dtype=float)
    weighted_baseline_metrics = evaluate_predictions(
        y_test,
        baseline_predictions,
        test_rows[policy_attribute].to_numpy(),
        privileged_value,
        test_probabilities,
        sample_weight=test_weights,
    )
    weighted_adjusted_metrics = evaluate_predictions(
        y_test,
        adjusted_predictions,
        test_rows[policy_attribute].to_numpy(),
        privileged_value,
        test_probabilities,
        sample_weight=test_weights,
    )
    intersection_groups = {
        "sex": test_rows["sex"].to_numpy(),
        "race": test_rows["race"].to_numpy(),
    }
    intersectional_baseline = intersectional_diagnostics(
        y_test,
        baseline_predictions,
        test_probabilities,
        intersection_groups,
        min_support=resolved_config.fairness.subgroup_min_support,
        min_positive=resolved_config.fairness.subgroup_min_class_count,
        min_negative=resolved_config.fairness.subgroup_min_class_count,
    )
    intersectional_adjusted = intersectional_diagnostics(
        y_test,
        adjusted_predictions,
        test_probabilities,
        intersection_groups,
        min_support=resolved_config.fairness.subgroup_min_support,
        min_positive=resolved_config.fairness.subgroup_min_class_count,
        min_negative=resolved_config.fairness.subgroup_min_class_count,
    )
    weighted_intersectional_adjusted = intersectional_diagnostics(
        y_test,
        adjusted_predictions,
        test_probabilities,
        intersection_groups,
        sample_weight=test_weights,
        min_support=resolved_config.fairness.subgroup_min_support,
        min_positive=resolved_config.fairness.subgroup_min_class_count,
        min_negative=resolved_config.fairness.subgroup_min_class_count,
    )
    review_evaluation = evaluate_review_band(
        y_test,
        test_probabilities,
        review_policy,
        groups=intersection_groups,
        min_group_support=resolved_config.fairness.subgroup_min_support,
        min_automated_group_samples=resolved_config.fairness.subgroup_min_class_count,
    )
    overlap_sensitivity = exact_feature_overlap_sensitivity(
        reference_rows=pd.concat([train_rows, val_rows], axis=0),
        heldout_rows=test_rows,
        y_true=y_test,
        baseline_predictions=baseline_predictions,
        adjusted_predictions=adjusted_predictions,
        probabilities=test_probabilities,
        sensitive=test_rows[policy_attribute].to_numpy(),
        privileged_group=privileged_value,
    )

    revision = git_state()
    resolved_run_id = _validate_run_id(
        run_id
        or f"{model_type}-seed-{seed}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    timestamp = datetime.now(timezone.utc).isoformat()
    monitoring_frame = test_rows[
        [*FEATURE_COLUMNS, "sex", "race", "race_binary", SAMPLE_WEIGHT_COLUMN]
    ].copy()
    monitoring_frame["score"] = test_probabilities
    monitoring_frame["prediction"] = baseline_predictions
    monitoring_frame["label"] = y_test
    monitoring_snapshot = build_snapshot(
        monitoring_frame,
        feature_columns=FEATURE_COLUMNS,
        categorical_columns=categorical_features,
        score_column="score",
        prediction_column="prediction",
        protected_columns=["sex", "race", "race_binary"],
        label_column="label",
        sample_weight_column=SAMPLE_WEIGHT_COLUMN,
        timestamp=timestamp,
    )
    metadata = {
        "run_id": resolved_run_id,
        "timestamp": timestamp,
        "seed": seed,
        "model_type": model_type,
        "git_commit": revision.commit,
        "dirty_worktree": revision.dirty_worktree,
        "data_path": str(source),
        "data_sha256": sha256_file(source),
        "data_quality_sha256": data_quality_digest,
        "source_sha256": source_sha256(),
        "python_version": platform.python_version(),
        "dependencies": _dependency_versions(),
        "resolved_config": config_payload(resolved_config),
        "config_sha256": config_sha256(resolved_config),
        "model_parameters": _model_kwargs(resolved_config, model_type),
        "preprocessing": {
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "transformed_feature_count": len(transformed_feature_names),
            "transformed_feature_names": transformed_feature_names,
            "unknown_category_policy": (
                "evaluation encoder ignores OOV; serving rejects OOV; "
                "split-level OOV evidence recorded"
            ),
            "categorical_oov_evidence": oov_evidence,
        },
    }
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "metadata": metadata,
        "protocol": {
            "dataset": "UCI Adult (1994 Census income classification)",
            "scope": "benchmark evaluation only; not a hiring validity study",
            "official_test_partition_preserved": True,
            "validation_strategy": "joint stratification by income, sex, and race_binary",
            "validation_ratio": effective_val_ratio,
            "threshold_tuning_split": "val",
            "final_evaluation_split": "test",
            "sensitive_attribute": policy_attribute,
            "privileged_group": privileged_value,
            "unprivileged_group": unprivileged_value,
            "feature_contract_id": FEATURE_CONTRACT_ID,
            "feature_columns": FEATURE_COLUMNS,
            "sample_weight_column": SAMPLE_WEIGHT_COLUMN,
            "sample_weight_used_as_predictor": False,
            "split_counts": split_frame["split"].value_counts().sort_index().to_dict(),
            "split_cell_counts": _split_cell_counts(split_frame),
        },
        "results": {
            "baseline_metrics": baseline_metrics,
            "metrics": adjusted_metrics,
            "thresholds": {
                "privileged": selected_frontier_policy.threshold_privileged,
                "unprivileged": selected_frontier_policy.threshold_unprivileged,
            },
            "validation_tuning": {
                "policy_id": "pareto-opportunity-frontier-v2",
                "kind": "group_thresholds",
                "method": "constrained Pareto frontier",
                "selection": asdict(frontier_result),
                "one_sided_comparator": tuning,
            },
            "validation_dependence": validation_dependence,
            "subgroup_diagnostics": {
                "baseline": _diagnostics(y_test, baseline_predictions, test_rows),
                "adjusted": _diagnostics(y_test, adjusted_predictions, test_rows),
            },
            "sampling_weight_sensitivity": {
                "status": "sensitivity_only",
                "weight_column": SAMPLE_WEIGHT_COLUMN,
                "weight_used_as_predictor": False,
                "baseline_metrics": weighted_baseline_metrics,
                "adjusted_metrics": weighted_adjusted_metrics,
                "interpretation": (
                    "CPS final weights are used only to compare weighted and unweighted rates; "
                    "these are not Census design-based confidence intervals."
                ),
            },
            "intersectional_uncertainty": {
                "baseline": intersectional_baseline,
                "adjusted": intersectional_adjusted,
                "weighted_adjusted_sensitivity": weighted_intersectional_adjusted,
            },
            "selective_review": {
                "policy": {
                    **review_policy.to_dict(),
                    "policy_id": "global-review-band-v2",
                    "kind": "global_review_band",
                    "selected_on": "validation",
                    "review_decision": "manual_review_required",
                    "fairness_adjustment_applied": False,
                    "protected_attributes_used": False,
                },
                "held_out_evaluation": review_evaluation,
                "scope": (
                    "Global probability-only review policy; human review quality is not "
                    "measured by this benchmark."
                ),
            },
            "data_quality": data_semantics,
            "feature_overlap_sensitivity": overlap_sensitivity,
            "monitoring_reference": {
                "status": "included_in_bundle",
                "schema_version": monitoring_snapshot["schema_version"],
                "row_count": monitoring_snapshot["row_count"],
                "aggregate_only": monitoring_snapshot["evidence"][
                    "contains_only_aggregate_statistics"
                ],
                "delayed_labels_available": monitoring_snapshot["evidence"]["labels_available"],
            },
        },
    }
    if bootstrap_samples:
        report["results"]["uncertainty"] = paired_bootstrap_intervals(
            y_true=y_test,
            baseline_pred=baseline_predictions,
            adjusted_pred=adjusted_predictions,
            sensitive=test_rows[policy_attribute].to_numpy(),
            privileged_group=privileged_value,
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
            "policy_id": "global-review-band-v2",
            "kind": "global_review_band",
            "threshold": resolved_config.fairness.eo_base_threshold,
            "lower_threshold": review_policy.lower_threshold,
            "upper_threshold": review_policy.upper_threshold,
            "review_decision": "manual_review_required",
            "selected_on": "validation",
            "max_automated_error_rate": review_policy.max_automated_error_rate,
            "fairness_adjustment_applied": False,
            "protected_attributes_used": False,
            "scope": "Adult-income policy simulation only",
        },
        "offline_evaluation": {
            "policy_id": "pareto-opportunity-frontier-v2",
            "kind": "group_thresholds",
            "sensitive_attribute": policy_attribute,
            "privileged_value": privileged_value,
            "unprivileged_value": unprivileged_value,
            "thresholds": report["results"]["thresholds"],
            "tuned_on": "val",
            "selection_status": frontier_result.status,
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
        "data_quality_sha256": metadata["data_quality_sha256"],
        "source_sha256": metadata["source_sha256"],
        "config_sha256": metadata["config_sha256"],
        "model_sha256": "pending",
        "report_sha256": "pending",
        "policy_sha256": "pending",
        "monitoring_sha256": "pending",
        "feature_columns": FEATURE_COLUMNS,
        "feature_contract_id": FEATURE_CONTRACT_ID,
        "sample_weight_column": SAMPLE_WEIGHT_COLUMN,
        "sample_weight_used_as_feature": False,
        "positive_class": 1,
        "base_threshold": resolved_config.fairness.eo_base_threshold,
        "python_version": metadata["python_version"],
        "dependencies": metadata["dependencies"],
        "resolved_config": metadata["resolved_config"],
        "model_parameters": metadata["model_parameters"],
        "preprocessing": metadata["preprocessing"],
        "split_seed": seed,
        "governance": gate_result.to_dict(),
        "experimental_only": True,
    }
    predictions = pd.DataFrame(
        {
            "source_index": test_rows.index,
            "y_true": y_test,
            "probability": test_probabilities,
            "baseline_prediction": baseline_predictions,
            "adjusted_prediction": adjusted_predictions,
            "sex": test_rows["sex"].to_numpy(),
            "race": test_rows["race"].to_numpy(),
            "race_binary": test_rows["race_binary"].to_numpy(),
            SAMPLE_WEIGHT_COLUMN: test_rows[SAMPLE_WEIGHT_COLUMN].to_numpy(),
        }
    )
    run_dir = save_bundle(
        run_dir=Path(output_dir) / resolved_run_id,
        model=model,
        manifest=manifest,
        report=report,
        policy=policy,
        predictions_csv=predictions.to_csv(index=False).encode("utf-8"),
        audit_html=render_audit_html(report),
        monitoring=monitoring_snapshot,
    )
    return ExperimentResult(resolved_run_id, run_dir, report, gate_result)
