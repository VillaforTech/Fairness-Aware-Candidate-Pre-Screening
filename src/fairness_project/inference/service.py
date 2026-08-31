"""Shared, policy-explicit inference for HTTP and batch callers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fairness_project.data.schema import CATEGORICAL_FEATURE_COLUMNS, FEATURE_COLUMNS
from fairness_project.models.artifact import ArtifactValidationError, ModelBundle, load_bundle

CONTRACT_PROBE = {
    "age": 35,
    "workclass": "Private",
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "native_country": "United-States",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
}

NUMERIC_FEATURE_RANGES: dict[str, tuple[int, int | None]] = {
    "age": (0, 120),
    "education_num": (1, 20),
    "capital_gain": (0, None),
    "capital_loss": (0, None),
    "hours_per_week": (0, 168),
}
CATEGORICAL_FEATURES = tuple(CATEGORICAL_FEATURE_COLUMNS)


class InferenceContractError(ValueError):
    """Raised when input or model output violates the artifact contract."""


@dataclass(frozen=True)
class PredictionBatch:
    """Predictions plus the explicit policy used to produce them."""

    predictions: np.ndarray
    decisions: np.ndarray
    probabilities: np.ndarray
    threshold: float
    lower_threshold: float
    upper_threshold: float
    policy_id: str
    artifact_id: str


class InferenceService:
    """Run review-aware simulations from one integrity-validated bundle."""

    def __init__(self, bundle: ModelBundle, *, allow_governance_rejected: bool = False):
        self.bundle = bundle
        if (
            self.bundle.manifest["governance"].get("passed") is not True
            and not allow_governance_rejected
        ):
            raise ArtifactValidationError(
                "Governance rejected this artifact; pass the explicit research override to "
                "run an evaluation-only simulation"
            )
        try:
            preprocessor = self.bundle.model.named_steps["preprocess"]
            encoder = preprocessor.named_transformers_["cat"]
            categories = encoder.categories_
            categorical_features = next(
                list(columns)
                for name, _transformer, columns in preprocessor.transformers_
                if name == "cat"
            )
        except (AttributeError, KeyError, StopIteration, TypeError) as exc:
            raise ArtifactValidationError(
                "Model does not expose the fitted categorical vocabulary"
            ) from exc
        if categorical_features != CATEGORICAL_FEATURE_COLUMNS or len(categorical_features) != len(
            categories
        ):
            raise ArtifactValidationError(
                "Fitted categorical features do not match the canonical vocabulary contract"
            )
        self._categorical_vocabulary = {
            str(feature): {str(value) for value in values}
            for feature, values in zip(categorical_features, categories, strict=True)
        }
        try:
            probabilities = np.asarray(
                self.bundle.model.predict_proba(pd.DataFrame([CONTRACT_PROBE])),
                dtype=float,
            )
        except Exception as exc:
            raise ArtifactValidationError(
                f"Model is incompatible with the canonical 11-feature contract: {exc}"
            ) from exc
        if probabilities.shape != (1, 2) or not np.isfinite(probabilities).all():
            raise ArtifactValidationError(
                "Model compatibility probe did not return one finite two-class probability row"
            )

    @classmethod
    def from_run(
        cls,
        run_dir: str | Path,
        *,
        allow_governance_rejected: bool = False,
    ) -> InferenceService:
        return cls(
            load_bundle(run_dir),
            allow_governance_rejected=allow_governance_rejected,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        serving = self.bundle.policy["serving"]
        return {
            "artifact_id": self.bundle.manifest["run_id"],
            "schema_version": self.bundle.manifest["schema_version"],
            "model_type": self.bundle.manifest["model_type"],
            "created_at": self.bundle.manifest["created_at"],
            "decision_policy": serving,
            "governance": self.bundle.manifest["governance"],
            "evaluation_only": True,
        }

    def predict(self, frame: pd.DataFrame) -> PredictionBatch:
        if frame.empty:
            raise InferenceContractError("At least one input row is required")

        missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
        extra = [column for column in frame.columns if column not in FEATURE_COLUMNS]
        if missing or extra:
            raise InferenceContractError(
                f"Input columns do not match the model contract; missing={missing}, extra={extra}"
            )
        if frame[FEATURE_COLUMNS].isnull().any().any():
            null_columns = (
                frame[FEATURE_COLUMNS].columns[frame[FEATURE_COLUMNS].isnull().any()].tolist()
            )
            raise InferenceContractError(f"Null values are not allowed in: {null_columns}")

        features = frame[FEATURE_COLUMNS].copy()
        for column, (minimum, maximum) in NUMERIC_FEATURE_RANGES.items():
            values = features[column]
            if values.map(lambda value: isinstance(value, (bool, np.bool_))).any():
                raise InferenceContractError(f"{column} must contain integers, not Booleans")
            numeric = pd.to_numeric(values, errors="coerce")
            if not np.isfinite(numeric.to_numpy(dtype=float)).all():
                raise InferenceContractError(f"{column} must contain finite integers")
            if not np.equal(numeric, np.floor(numeric)).all():
                raise InferenceContractError(f"{column} must contain integers")
            if (numeric < minimum).any() or (maximum is not None and (numeric > maximum).any()):
                upper = f" and at most {maximum}" if maximum is not None else ""
                raise InferenceContractError(f"{column} must be at least {minimum}{upper}")
            features[column] = numeric

        for column in CATEGORICAL_FEATURES:
            values = features[column]
            if not values.map(lambda value: isinstance(value, str)).all():
                raise InferenceContractError(f"{column} must contain strings")
            stripped = values.str.strip()
            if stripped.eq("").any():
                raise InferenceContractError(f"{column} cannot be blank")
            vocabulary = self._categorical_vocabulary.get(column)
            if vocabulary is None:
                raise InferenceContractError(f"{column} is missing from the fitted vocabulary")
            unknown = sorted(set(stripped) - vocabulary)
            if unknown:
                raise InferenceContractError(
                    f"{column} contains categories not seen during training: {unknown}"
                )
            features[column] = stripped

        try:
            raw_probabilities = np.asarray(
                self.bundle.model.predict_proba(features),
                dtype=float,
            )
        except Exception as exc:
            raise InferenceContractError(
                f"Model rejected the canonical feature schema: {exc}"
            ) from exc

        if raw_probabilities.shape != (len(frame), 2):
            raise InferenceContractError(
                "predict_proba must return one two-class probability row per input"
            )
        if not np.isfinite(raw_probabilities).all():
            raise InferenceContractError("Model returned non-finite probabilities")
        if ((raw_probabilities < 0) | (raw_probabilities > 1)).any():
            raise InferenceContractError("Model returned probabilities outside [0, 1]")

        positive_class = self.bundle.manifest["positive_class"]
        classes = list(self.bundle.model.classes_)
        positive_index = classes.index(positive_class)
        probabilities = raw_probabilities[:, positive_index]
        threshold = float(self.bundle.policy["serving"]["threshold"])
        lower_threshold = float(self.bundle.policy["serving"]["lower_threshold"])
        upper_threshold = float(self.bundle.policy["serving"]["upper_threshold"])
        decisions = np.where(probabilities >= threshold, "auto_positive", "auto_negative")
        if lower_threshold < upper_threshold:
            review_mask = (probabilities >= lower_threshold) & (probabilities <= upper_threshold)
            decisions = decisions.astype("<U24")
            decisions[review_mask] = "manual_review_required"
        predictions = (probabilities >= threshold).astype(int)
        predictions[decisions == "manual_review_required"] = -1
        return PredictionBatch(
            predictions=predictions,
            decisions=decisions,
            probabilities=probabilities,
            threshold=threshold,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
            policy_id=self.bundle.policy["serving"]["policy_id"],
            artifact_id=self.bundle.manifest["run_id"],
        )
