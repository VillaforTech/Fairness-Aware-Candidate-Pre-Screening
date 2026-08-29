"""Versioned, fail-closed model artifact bundles."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from fairness_project.data.schema import FEATURE_COLUMNS
from fairness_project.provenance import UNAVAILABLE_GIT_COMMIT

ARTIFACT_SCHEMA_VERSION = "1.0"
MODEL_FILENAME = "model.joblib"
MANIFEST_FILENAME = "manifest.json"
REPORT_FILENAME = "report.json"
POLICY_FILENAME = "policy.json"
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
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, destination)
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
        "model_sha256",
        "report_sha256",
        "policy_sha256",
    ):
        _require_sha256(manifest, key, "manifest")

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
    positive_class = manifest.get("positive_class")
    if isinstance(positive_class, bool) or positive_class not in (0, 1):
        raise ArtifactValidationError("manifest.positive_class must be 0 or 1")
    _require_probability(manifest, "base_threshold", "manifest")
    if not isinstance(manifest.get("dependencies"), dict):
        raise ArtifactValidationError("manifest.dependencies must be an object")
    _require_string(manifest, "python_version", "manifest")
    if not isinstance(manifest.get("governance"), dict):
        raise ArtifactValidationError("manifest.governance must be an object")
    if manifest.get("experimental_only") is not True:
        raise ArtifactValidationError("manifest.experimental_only must be true")


def _validate_policy(policy: dict[str, Any], manifest: dict[str, Any]) -> None:
    if policy.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactValidationError("Unsupported policy schema")
    if policy.get("artifact_id") != manifest["run_id"]:
        raise ArtifactValidationError("Policy artifact_id does not match the manifest")
    serving = policy.get("serving")
    if not isinstance(serving, dict):
        raise ArtifactValidationError("policy.serving must be an object")
    _require_string(serving, "policy_id", "policy.serving")
    if serving.get("kind") != "global_threshold":
        raise ArtifactValidationError(
            "Only an explicit global-threshold serving policy is supported"
        )
    if serving.get("fairness_adjustment_applied") is not False:
        raise ArtifactValidationError(
            "Serving policy must state that no fairness adjustment is applied"
        )
    serving_threshold = _require_probability(serving, "threshold", "policy.serving")
    if serving_threshold != float(manifest["base_threshold"]):
        raise ArtifactValidationError("Policy threshold does not match the manifest")
    offline = policy.get("offline_evaluation")
    if not isinstance(offline, dict) or offline.get("experimental_only") is not True:
        raise ArtifactValidationError("Offline threshold policy must be marked experimental_only")
    _require_string(offline, "policy_id", "policy.offline_evaluation")
    if offline.get("kind") != "group_thresholds":
        raise ArtifactValidationError("Offline policy kind must be group_thresholds")
    if offline.get("served_by_api") is not False:
        raise ArtifactValidationError("Offline policy must state that it is not served by the API")
    thresholds = offline.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ArtifactValidationError("policy.offline_evaluation.thresholds must be an object")
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
        "source_sha256": "source_sha256",
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
    if report.get("governance") != manifest.get("governance"):
        raise ArtifactValidationError("Report governance verdict does not match manifest")


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
) -> Path:
    """Persist a complete model bundle and fill its content digests."""
    destination = Path(run_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    model_path = destination / MODEL_FILENAME
    joblib.dump(model, model_path)
    report_path = write_json(destination / REPORT_FILENAME, report)
    policy_path = write_json(destination / POLICY_FILENAME, policy)

    completed_manifest = dict(manifest)
    completed_manifest["model_sha256"] = sha256_file(model_path)
    completed_manifest["report_sha256"] = sha256_file(report_path)
    completed_manifest["policy_sha256"] = sha256_file(policy_path)
    _validate_manifest(completed_manifest)
    _validate_policy(policy, completed_manifest)
    _validate_report_binding(report, completed_manifest)
    write_json(destination / MANIFEST_FILENAME, completed_manifest)
    return destination


def load_bundle(run_dir: str | Path) -> ModelBundle:
    """Load and verify every file required for local inference."""
    source = Path(run_dir)
    required = [MODEL_FILENAME, MANIFEST_FILENAME, REPORT_FILENAME, POLICY_FILENAME]
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise ArtifactValidationError(f"Run bundle is missing files: {missing}")

    manifest = read_json(source / MANIFEST_FILENAME)
    report = read_json(source / REPORT_FILENAME)
    policy = read_json(source / POLICY_FILENAME)
    _validate_manifest(manifest)
    _validate_policy(policy, manifest)

    if sha256_file(source / MODEL_FILENAME) != manifest["model_sha256"]:
        raise ArtifactValidationError("Model digest does not match manifest")
    if sha256_file(source / REPORT_FILENAME) != manifest["report_sha256"]:
        raise ArtifactValidationError("Report digest does not match manifest")
    if sha256_file(source / POLICY_FILENAME) != manifest["policy_sha256"]:
        raise ArtifactValidationError("Policy digest does not match manifest")
    _validate_report_binding(report, manifest)
    _validate_runtime(manifest)

    try:
        model = joblib.load(source / MODEL_FILENAME)
    except Exception as exc:
        raise ArtifactValidationError(f"Cannot load model: {exc}") from exc
    if not callable(getattr(model, "predict_proba", None)):
        raise ArtifactValidationError("Model must implement predict_proba")
    classes = list(getattr(model, "classes_", []))
    if classes != [0, 1]:
        raise ArtifactValidationError(f"Model classes must be [0, 1], got {classes}")

    return ModelBundle(model=model, manifest=manifest, report=report, policy=policy, run_dir=source)
