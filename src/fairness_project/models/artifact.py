"""Versioned, fail-closed model artifact bundles."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from fairness_project.data.schema import (
    CATEGORICAL_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    FEATURE_CONTRACT_ID,
    NUMERIC_FEATURE_COLUMNS,
    SAMPLE_WEIGHT_COLUMN,
)
from fairness_project.governance.gate import GateThresholds, check_gate
from fairness_project.monitoring import validate_snapshot
from fairness_project.provenance import UNAVAILABLE_GIT_COMMIT

ARTIFACT_SCHEMA_VERSION = "2.0"
MODEL_FILENAME = "model.joblib"
MANIFEST_FILENAME = "manifest.json"
REPORT_FILENAME = "report.json"
POLICY_FILENAME = "policy.json"
PREDICTIONS_FILENAME = "predictions.csv"
AUDIT_HTML_FILENAME = "audit.html"
MONITORING_FILENAME = "monitoring.json"
MODEL_RUNTIME_DEPENDENCIES = ("numpy", "pandas", "scikit-learn", "xgboost", "joblib")


class ArtifactValidationError(ValueError):
    """Raised when a run bundle is incomplete or internally inconsistent."""


@dataclass(frozen=True)
class ModelBundle:
    """Loaded model plus its validated provenance and policy documents."""

    model: Any
    manifest: dict[str, Any]
    report: dict[str, Any]
    policy: dict[str, Any]
    monitoring: dict[str, Any]
    run_dir: Path


def sha256_file(path: str | Path) -> str:
    """Compute a SHA-256 digest without loading the whole file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write strict JSON atomically; NaN and infinity are rejected."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def read_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object and reject non-object roots."""
    source = Path(path)
    try:
        with source.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"Cannot read {source.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactValidationError(f"{source.name} must contain a JSON object")
    return payload


def _require_string(payload: dict[str, Any], key: str, document: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ArtifactValidationError(f"{document}.{key} must be a non-empty string")
    return value


def _require_sha256(payload: dict[str, Any], key: str, document: str) -> str:
    value = _require_string(payload, key, document)
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ArtifactValidationError(f"{document}.{key} must be a lowercase SHA-256 digest")
    return value


def _require_probability(payload: dict[str, Any], key: str, document: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactValidationError(f"{document}.{key} must be numeric")
    probability = float(value)
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ArtifactValidationError(f"{document}.{key} must be finite and within [0, 1]")
    return probability


def _validate_governance(governance: Any, document: str) -> dict[str, Any]:
    if not isinstance(governance, dict):
        raise ArtifactValidationError(f"{document} must be an object")
    expected_fields = {
        "passed",
        "violations",
        "metrics_checked",
        "thresholds",
        "report_valid",
    }
    if set(governance) != expected_fields:
        raise ArtifactValidationError(
            f"{document} must contain the exact governance verdict fields"
        )
    if not isinstance(governance.get("passed"), bool):
        raise ArtifactValidationError(f"{document}.passed must be Boolean")
    if governance.get("report_valid") is not True:
        raise ArtifactValidationError(f"{document}.report_valid must be true")
    violations = governance.get("violations")
    if not isinstance(violations, list) or any(not isinstance(value, str) for value in violations):
        raise ArtifactValidationError(f"{document}.violations must be an array of strings")
    if not isinstance(governance.get("metrics_checked"), dict):
        raise ArtifactValidationError(f"{document}.metrics_checked must be an object")
    thresholds = governance.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ArtifactValidationError(f"{document}.thresholds must be an object")
    expected_thresholds = set(GateThresholds().to_dict())
    if set(thresholds) != expected_thresholds:
        raise ArtifactValidationError(
            f"{document}.thresholds must contain the exact gate policy fields"
        )
    try:
        GateThresholds(**thresholds)
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"{document}.thresholds is invalid: {exc}") from exc
    return governance


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactValidationError(
            f"Unsupported manifest schema: {manifest.get('schema_version')!r}"
        )
    for key in (
        "run_id",
        "model_type",
        "created_at",
        "git_commit",
    ):
        _require_string(manifest, key, "manifest")
    for key in (
        "data_sha256",
        "source_sha256",
        "config_sha256",
        "model_sha256",
        "report_sha256",
        "policy_sha256",
        "predictions_sha256",
        "audit_html_sha256",
        "monitoring_sha256",
    ):
        _require_sha256(manifest, key, "manifest")
    data_quality_sha256 = manifest.get("data_quality_sha256")
    if data_quality_sha256 is not None and (
        not isinstance(data_quality_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", data_quality_sha256) is None
    ):
        raise ArtifactValidationError(
            "manifest.data_quality_sha256 must be null or a lowercase SHA-256 digest"
        )

    git_commit = manifest["git_commit"]
    dirty_worktree = manifest.get("dirty_worktree")
    if git_commit == UNAVAILABLE_GIT_COMMIT:
        if dirty_worktree is not None:
            raise ArtifactValidationError(
                "manifest.dirty_worktree must be null when git_commit is unavailable"
            )
    elif re.fullmatch(r"[0-9a-f]{40}", git_commit) is None:
        raise ArtifactValidationError(
            "manifest.git_commit must be a full lowercase 40-hex commit or 'unavailable'"
        )
    elif not isinstance(dirty_worktree, bool):
        raise ArtifactValidationError("manifest.dirty_worktree must be Boolean for a Git checkout")

    if manifest.get("feature_columns") != FEATURE_COLUMNS:
        raise ArtifactValidationError(
            "Manifest feature_columns do not match the canonical Adult feature contract"
        )
    if manifest.get("feature_contract_id") != FEATURE_CONTRACT_ID:
        raise ArtifactValidationError("Manifest feature_contract_id is unsupported")
    if manifest.get("sample_weight_column") != SAMPLE_WEIGHT_COLUMN:
        raise ArtifactValidationError("Manifest sample_weight_column is unsupported")
    if manifest.get("sample_weight_used_as_feature") is not False:
        raise ArtifactValidationError("Census sample weight must not be used as a predictor")
    positive_class = manifest.get("positive_class")
    if isinstance(positive_class, bool) or positive_class != 1:
        raise ArtifactValidationError(
            "manifest.positive_class must be 1 for the fixed Adult income label contract"
        )
    _require_probability(manifest, "base_threshold", "manifest")
    if not isinstance(manifest.get("dependencies"), dict):
        raise ArtifactValidationError("manifest.dependencies must be an object")
    if not isinstance(manifest.get("resolved_config"), dict):
        raise ArtifactValidationError("manifest.resolved_config must be an object")
    if not isinstance(manifest.get("model_parameters"), dict):
        raise ArtifactValidationError("manifest.model_parameters must be an object")
    preprocessing = manifest.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise ArtifactValidationError("manifest.preprocessing must be an object")
    if preprocessing.get("numeric_features") != NUMERIC_FEATURE_COLUMNS:
        raise ArtifactValidationError(
            "manifest.preprocessing.numeric_features does not match the canonical contract"
        )
    if preprocessing.get("categorical_features") != CATEGORICAL_FEATURE_COLUMNS:
        raise ArtifactValidationError(
            "manifest.preprocessing.categorical_features does not match the canonical contract"
        )
    transformed_feature_names = preprocessing.get("transformed_feature_names")
    if (
        not isinstance(transformed_feature_names, list)
        or not transformed_feature_names
        or any(not isinstance(value, str) or not value for value in transformed_feature_names)
    ):
        raise ArtifactValidationError(
            "manifest.preprocessing.transformed_feature_names must be a nonempty string array"
        )
    if len(set(transformed_feature_names)) != len(transformed_feature_names):
        raise ArtifactValidationError(
            "manifest.preprocessing.transformed_feature_names must be unique"
        )
    if preprocessing.get("transformed_feature_count") != len(transformed_feature_names):
        raise ArtifactValidationError(
            "manifest.preprocessing.transformed_feature_count is inconsistent"
        )
    if preprocessing.get("unknown_category_policy") != (
        "evaluation encoder ignores OOV; serving rejects OOV; split-level OOV evidence recorded"
    ):
        raise ArtifactValidationError(
            "manifest.preprocessing.unknown_category_policy is unsupported"
        )
    oov_evidence = preprocessing.get("categorical_oov_evidence")
    if not isinstance(oov_evidence, dict) or oov_evidence.get("schema_version") != "1.0":
        raise ArtifactValidationError(
            "manifest.preprocessing.categorical_oov_evidence is missing or unsupported"
        )
    if oov_evidence.get("reference_split") != "train" or not isinstance(
        oov_evidence.get("splits"), dict
    ):
        raise ArtifactValidationError(
            "manifest.preprocessing.categorical_oov_evidence is malformed"
        )
    _require_string(manifest, "python_version", "manifest")
    _validate_governance(manifest.get("governance"), "manifest.governance")
    if manifest.get("experimental_only") is not True:
        raise ArtifactValidationError("manifest.experimental_only must be true")


def _validate_policy(policy: dict[str, Any], manifest: dict[str, Any]) -> None:
    if policy.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactValidationError("Unsupported policy schema")
    if set(policy) != {
        "schema_version",
        "artifact_id",
        "serving",
        "offline_evaluation",
    }:
        raise ArtifactValidationError("Policy must contain the exact versioned policy fields")
    if policy.get("artifact_id") != manifest["run_id"]:
        raise ArtifactValidationError("Policy artifact_id does not match the manifest")
    serving = policy.get("serving")
    if not isinstance(serving, dict):
        raise ArtifactValidationError("policy.serving must be an object")
    if set(serving) != {
        "policy_id",
        "kind",
        "threshold",
        "lower_threshold",
        "upper_threshold",
        "review_decision",
        "selected_on",
        "max_automated_error_rate",
        "fairness_adjustment_applied",
        "protected_attributes_used",
        "scope",
    }:
        raise ArtifactValidationError("policy.serving must contain the exact serving-policy fields")
    _require_string(serving, "policy_id", "policy.serving")
    if serving.get("kind") != "global_review_band":
        raise ArtifactValidationError("Only an explicit global review-band policy is supported")
    if serving.get("fairness_adjustment_applied") is not False:
        raise ArtifactValidationError(
            "Serving policy must state that no fairness adjustment is applied"
        )
    serving_threshold = _require_probability(serving, "threshold", "policy.serving")
    if serving_threshold != float(manifest["base_threshold"]):
        raise ArtifactValidationError("Policy threshold does not match the manifest")
    lower = _require_probability(serving, "lower_threshold", "policy.serving")
    upper = _require_probability(serving, "upper_threshold", "policy.serving")
    _require_probability(serving, "max_automated_error_rate", "policy.serving")
    if not lower <= serving_threshold <= upper:
        raise ArtifactValidationError("Review-band thresholds must contain the base threshold")
    if serving.get("review_decision") != "manual_review_required":
        raise ArtifactValidationError("Serving policy must define manual_review_required")
    if serving.get("selected_on") != "validation":
        raise ArtifactValidationError("Serving policy must be selected on validation")
    if serving.get("protected_attributes_used") is not False:
        raise ArtifactValidationError("Serving policy must not use protected attributes")
    if serving.get("scope") != "Adult-income policy simulation only":
        raise ArtifactValidationError("Serving policy scope is unsupported")
    offline = policy.get("offline_evaluation")
    if not isinstance(offline, dict) or offline.get("experimental_only") is not True:
        raise ArtifactValidationError("Offline threshold policy must be marked experimental_only")
    if set(offline) != {
        "policy_id",
        "kind",
        "sensitive_attribute",
        "privileged_value",
        "unprivileged_value",
        "thresholds",
        "tuned_on",
        "selection_status",
        "experimental_only",
        "served_by_api",
    }:
        raise ArtifactValidationError(
            "policy.offline_evaluation must contain the exact offline-policy fields"
        )
    _require_string(offline, "policy_id", "policy.offline_evaluation")
    if offline.get("kind") != "group_thresholds":
        raise ArtifactValidationError("Offline policy kind must be group_thresholds")
    if offline.get("served_by_api") is not False:
        raise ArtifactValidationError("Offline policy must state that it is not served by the API")
    if offline.get("tuned_on") != "val":
        raise ArtifactValidationError("Offline policy must be tuned on the val split")
    if offline.get("selection_status") not in {"feasible", "infeasible"}:
        raise ArtifactValidationError(
            "Offline policy selection_status must be feasible or infeasible"
        )
    for key in ("sensitive_attribute", "privileged_value", "unprivileged_value"):
        _require_string(offline, key, "policy.offline_evaluation")
    thresholds = offline.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ArtifactValidationError("policy.offline_evaluation.thresholds must be an object")
    if set(thresholds) != {"privileged", "unprivileged"}:
        raise ArtifactValidationError(
            "policy.offline_evaluation.thresholds must contain exact group thresholds"
        )
    _require_probability(thresholds, "privileged", "policy.offline_evaluation.thresholds")
    _require_probability(thresholds, "unprivileged", "policy.offline_evaluation.thresholds")


def _validate_report_binding(report: dict[str, Any], manifest: dict[str, Any]) -> None:
    metadata = report.get("metadata")
    if not isinstance(metadata, dict):
        raise ArtifactValidationError("report.metadata must be an object")
    bindings = {
        "run_id": "run_id",
        "model_type": "model_type",
        "git_commit": "git_commit",
        "data_sha256": "data_sha256",
        "data_quality_sha256": "data_quality_sha256",
        "source_sha256": "source_sha256",
        "config_sha256": "config_sha256",
        "python_version": "python_version",
        "dependencies": "dependencies",
        "dirty_worktree": "dirty_worktree",
        "seed": "split_seed",
    }
    for report_key, manifest_key in bindings.items():
        if metadata.get(report_key) != manifest.get(manifest_key):
            raise ArtifactValidationError(
                f"Report {report_key} does not match manifest {manifest_key}"
            )
    for key in ("resolved_config", "model_parameters", "preprocessing"):
        if metadata.get(key) != manifest.get(key):
            raise ArtifactValidationError(f"Report {key} does not match manifest {key}")
    report_governance = _validate_governance(report.get("governance"), "report.governance")
    computed_governance = check_gate(report).to_dict()
    if report_governance != computed_governance:
        raise ArtifactValidationError(
            "Report governance verdict does not match a fresh gate evaluation"
        )
    if report_governance != manifest.get("governance"):
        raise ArtifactValidationError("Report governance verdict does not match manifest")


def _validate_policy_report_binding(policy: dict[str, Any], report: dict[str, Any]) -> None:
    results = report.get("results")
    protocol = report.get("protocol")
    if not isinstance(results, dict) or not isinstance(protocol, dict):
        raise ArtifactValidationError("Report results and protocol must be objects")

    selective_review = results.get("selective_review")
    reported_serving = (
        selective_review.get("policy") if isinstance(selective_review, dict) else None
    )
    if not isinstance(reported_serving, dict):
        raise ArtifactValidationError("report.results.selective_review.policy must be an object")
    serving = policy["serving"]
    serving_bindings = {
        "policy_id": "policy_id",
        "kind": "kind",
        "threshold": "base_threshold",
        "lower_threshold": "lower_threshold",
        "upper_threshold": "upper_threshold",
        "max_automated_error_rate": "max_automated_error_rate",
        "selected_on": "selected_on",
        "review_decision": "review_decision",
        "fairness_adjustment_applied": "fairness_adjustment_applied",
        "protected_attributes_used": "protected_attributes_used",
    }
    for policy_key, report_key in serving_bindings.items():
        if serving.get(policy_key) != reported_serving.get(report_key):
            raise ArtifactValidationError(
                f"Serving policy {policy_key} does not match selective-review evidence"
            )

    offline = policy["offline_evaluation"]
    validation_tuning = results.get("validation_tuning")
    selection = validation_tuning.get("selection") if isinstance(validation_tuning, dict) else None
    if not isinstance(validation_tuning, dict) or not isinstance(selection, dict):
        raise ArtifactValidationError(
            "report.results.validation_tuning selection evidence must be an object"
        )
    if offline.get("policy_id") != validation_tuning.get("policy_id"):
        raise ArtifactValidationError("Offline policy ID does not match validation evidence")
    if offline.get("kind") != validation_tuning.get("kind"):
        raise ArtifactValidationError("Offline policy kind does not match validation evidence")
    if offline.get("selection_status") != selection.get("status"):
        raise ArtifactValidationError(
            "Offline policy selection status does not match validation evidence"
        )
    if offline.get("thresholds") != results.get("thresholds"):
        raise ArtifactValidationError("Offline thresholds do not match held-out report policy")
    protocol_bindings = {
        "sensitive_attribute": "sensitive_attribute",
        "privileged_value": "privileged_group",
        "unprivileged_value": "unprivileged_group",
        "tuned_on": "threshold_tuning_split",
    }
    for policy_key, report_key in protocol_bindings.items():
        if offline.get(policy_key) != protocol.get(report_key):
            raise ArtifactValidationError(
                f"Offline policy {policy_key} does not match report protocol"
            )


def _validate_model_contract(model: Any, manifest: dict[str, Any]) -> None:
    feature_names = [str(value) for value in getattr(model, "feature_names_in_", [])]
    if feature_names != FEATURE_COLUMNS:
        raise ArtifactValidationError(
            f"Model feature_names_in_ do not match the manifest contract: {feature_names}"
        )
    classes = list(getattr(model, "classes_", []))
    if classes != [0, 1]:
        raise ArtifactValidationError(f"Model classes must be [0, 1], got {classes}")

    try:
        preprocessor = model.named_steps["preprocess"]
        named_transformers = preprocessor.named_transformers_
        numeric_transformer = named_transformers["num"]
        categorical_transformer = named_transformers["cat"]
        transformer_columns = {
            str(name): list(columns)
            for name, _transformer, columns in preprocessor.transformers_
            if name in {"num", "cat"}
        }
        transformed_names = [str(value) for value in preprocessor.get_feature_names_out()]
        categories = categorical_transformer.categories_
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ArtifactValidationError(
            "Model does not expose the fitted canonical preprocessing contract"
        ) from exc

    if transformer_columns.get("num") != NUMERIC_FEATURE_COLUMNS:
        raise ArtifactValidationError(
            "Fitted numeric transformer columns do not match the canonical contract"
        )
    if transformer_columns.get("cat") != CATEGORICAL_FEATURE_COLUMNS:
        raise ArtifactValidationError(
            "Fitted categorical transformer columns do not match the canonical contract"
        )
    if numeric_transformer is None or categorical_transformer is None:
        raise ArtifactValidationError("Fitted preprocessing transformers are missing")
    if getattr(categorical_transformer, "handle_unknown", None) != "ignore":
        raise ArtifactValidationError("Fitted categorical OOV behavior is unsupported")
    if len(categories) != len(CATEGORICAL_FEATURE_COLUMNS):
        raise ArtifactValidationError(
            "Fitted categorical vocabulary does not match the canonical contract"
        )
    for feature, values in zip(CATEGORICAL_FEATURE_COLUMNS, categories, strict=True):
        normalized = [str(value) for value in values]
        if not normalized or len(set(normalized)) != len(normalized):
            raise ArtifactValidationError(
                f"Fitted vocabulary for {feature} must be nonempty and unique"
            )

    preprocessing = manifest["preprocessing"]
    if transformed_names != preprocessing["transformed_feature_names"]:
        raise ArtifactValidationError("Fitted transformed feature names do not match the manifest")


def _validate_runtime(manifest: dict[str, Any]) -> None:
    recorded_python = str(manifest["python_version"]).split(".")[:2]
    runtime_python = [str(sys.version_info.major), str(sys.version_info.minor)]
    if recorded_python != runtime_python:
        raise ArtifactValidationError(
            "Runtime Python does not match the artifact environment: "
            f"recorded={'.'.join(recorded_python)}, runtime={'.'.join(runtime_python)}"
        )

    recorded_dependencies = manifest["dependencies"]
    for package in MODEL_RUNTIME_DEPENDENCIES:
        expected = recorded_dependencies.get(package)
        if not isinstance(expected, str) or not expected:
            raise ArtifactValidationError(f"Manifest does not record dependency {package}")
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ArtifactValidationError(
                f"Required runtime dependency is missing: {package}"
            ) from exc
        if actual != expected:
            raise ArtifactValidationError(
                f"Runtime dependency mismatch for {package}: recorded={expected}, runtime={actual}"
            )


def save_bundle(
    *,
    run_dir: str | Path,
    model: Any,
    manifest: dict[str, Any],
    report: dict[str, Any],
    policy: dict[str, Any],
    predictions_csv: bytes,
    audit_html: str,
    monitoring: dict[str, Any],
) -> Path:
    """Persist and atomically publish a complete, integrity-bound model bundle."""
    destination = Path(run_dir)
    if destination.exists():
        raise FileExistsError(f"Run directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        model_path = temporary / MODEL_FILENAME
        joblib.dump(model, model_path)
        report_path = write_json(temporary / REPORT_FILENAME, report)
        policy_path = write_json(temporary / POLICY_FILENAME, policy)
        predictions_path = temporary / PREDICTIONS_FILENAME
        predictions_path.write_bytes(predictions_csv)
        audit_html_path = temporary / AUDIT_HTML_FILENAME
        audit_html_path.write_text(audit_html, encoding="utf-8")
        validate_snapshot(monitoring)
        monitoring_path = write_json(temporary / MONITORING_FILENAME, monitoring)

        completed_manifest = dict(manifest)
        completed_manifest["model_sha256"] = sha256_file(model_path)
        completed_manifest["report_sha256"] = sha256_file(report_path)
        completed_manifest["policy_sha256"] = sha256_file(policy_path)
        completed_manifest["predictions_sha256"] = sha256_file(predictions_path)
        completed_manifest["audit_html_sha256"] = sha256_file(audit_html_path)
        completed_manifest["monitoring_sha256"] = sha256_file(monitoring_path)
        _validate_manifest(completed_manifest)
        _validate_policy(policy, completed_manifest)
        _validate_report_binding(report, completed_manifest)
        _validate_policy_report_binding(policy, report)
        _validate_model_contract(model, completed_manifest)
        write_json(temporary / MANIFEST_FILENAME, completed_manifest)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def load_bundle(run_dir: str | Path) -> ModelBundle:
    """Load and verify every file required for local inference."""
    source = Path(run_dir)
    required = [
        MODEL_FILENAME,
        MANIFEST_FILENAME,
        REPORT_FILENAME,
        POLICY_FILENAME,
        PREDICTIONS_FILENAME,
        AUDIT_HTML_FILENAME,
        MONITORING_FILENAME,
    ]
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise ArtifactValidationError(f"Run bundle is missing files: {missing}")

    manifest = read_json(source / MANIFEST_FILENAME)
    report = read_json(source / REPORT_FILENAME)
    policy = read_json(source / POLICY_FILENAME)
    monitoring = read_json(source / MONITORING_FILENAME)
    _validate_manifest(manifest)
    _validate_policy(policy, manifest)

    if sha256_file(source / MODEL_FILENAME) != manifest["model_sha256"]:
        raise ArtifactValidationError("Model digest does not match manifest")
    if sha256_file(source / REPORT_FILENAME) != manifest["report_sha256"]:
        raise ArtifactValidationError("Report digest does not match manifest")
    if sha256_file(source / POLICY_FILENAME) != manifest["policy_sha256"]:
        raise ArtifactValidationError("Policy digest does not match manifest")
    if sha256_file(source / PREDICTIONS_FILENAME) != manifest["predictions_sha256"]:
        raise ArtifactValidationError("Predictions digest does not match manifest")
    if sha256_file(source / AUDIT_HTML_FILENAME) != manifest["audit_html_sha256"]:
        raise ArtifactValidationError("Audit HTML digest does not match manifest")
    if sha256_file(source / MONITORING_FILENAME) != manifest["monitoring_sha256"]:
        raise ArtifactValidationError("Monitoring snapshot digest does not match manifest")
    try:
        validate_snapshot(monitoring)
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"Monitoring snapshot is invalid: {exc}") from exc
    _validate_report_binding(report, manifest)
    _validate_policy_report_binding(policy, report)
    _validate_runtime(manifest)

    try:
        model = joblib.load(source / MODEL_FILENAME)
    except Exception as exc:
        raise ArtifactValidationError(f"Cannot load model: {exc}") from exc
    if not callable(getattr(model, "predict_proba", None)):
        raise ArtifactValidationError("Model must implement predict_proba")
    _validate_model_contract(model, manifest)

    return ModelBundle(
        model=model,
        manifest=manifest,
        report=report,
        policy=policy,
        monitoring=monitoring,
        run_dir=source,
    )
