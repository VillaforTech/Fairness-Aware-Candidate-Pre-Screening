"""Exact predictive-feature overlap sensitivity for held-out evaluation.

This audit asks a narrow robustness question: do reported held-out results
change after removing rows whose complete canonical model-feature identity also
appears in the fit/reference rows?  It never changes predictions, thresholds,
or policy parameters.

Identity matching deliberately does not use a digest as the equality test.
Canonical tuples are sorted and matched with exact value equality, so a hash
collision cannot create an overlap observation.
"""

from __future__ import annotations

import warnings
from bisect import bisect_left
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from pandas.api.types import is_integer_dtype, is_object_dtype, is_string_dtype

from fairness_project.data.schema import (
    FEATURE_COLUMNS,
    FEATURE_CONTRACT_ID,
    INTEGER_COLUMNS,
)
from fairness_project.evaluation.evaluate import evaluate_predictions

OVERLAP_SENSITIVITY_SCHEMA_VERSION = "1.0"
_FAIRNESS_METRICS = ("SPD", "DI", "TPR_gap", "FPR_gap")
_METRIC_KEYS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "SPD",
    "DI",
    "FPR_gap",
    "TPR_priv",
    "TPR_unpriv",
    "TPR_gap",
)

FeatureValue = int | str
FeatureIdentity = tuple[FeatureValue, ...]
JsonScalar = str | int | float | bool

__all__ = [
    "OVERLAP_SENSITIVITY_SCHEMA_VERSION",
    "OverlapSensitivityValidationError",
    "exact_feature_overlap_mask",
    "exact_feature_overlap_sensitivity",
]


class OverlapSensitivityValidationError(ValueError):
    """Raised when overlap-sensitivity inputs violate the audit contract."""


def _validate_frame(frame: pd.DataFrame, name: str) -> list[FeatureIdentity]:
    if not isinstance(frame, pd.DataFrame):
        raise OverlapSensitivityValidationError(f"{name} must be a pandas DataFrame")
    if frame.empty:
        raise OverlapSensitivityValidationError(f"{name} must contain at least one row")

    columns = list(frame.columns)
    if any(not isinstance(column, str) or not column for column in columns):
        raise OverlapSensitivityValidationError(f"{name} must use non-empty string column names")
    if len(set(columns)) != len(columns):
        raise OverlapSensitivityValidationError(f"{name} must not contain duplicate column names")
    missing = sorted(set(FEATURE_COLUMNS) - set(columns))
    if missing:
        raise OverlapSensitivityValidationError(
            f"{name} is missing canonical feature columns: {missing}"
        )

    canonical = frame.loc[:, FEATURE_COLUMNS]
    if canonical.isna().any(axis=None):
        missing_columns = sorted(canonical.columns[canonical.isna().any()].tolist())
        raise OverlapSensitivityValidationError(
            f"{name} canonical features must not contain missing values: {missing_columns}"
        )

    integer_features = set(FEATURE_COLUMNS) & INTEGER_COLUMNS
    for column in FEATURE_COLUMNS:
        series = canonical[column]
        if column in integer_features:
            if not is_integer_dtype(series.dtype) or is_object_dtype(series.dtype):
                raise OverlapSensitivityValidationError(
                    f"{name}.{column} must use an integer dtype"
                )
            if any(isinstance(value, (bool, np.bool_)) for value in series.tolist()):
                raise OverlapSensitivityValidationError(
                    f"{name}.{column} must contain integers, not booleans"
                )
        else:
            if not (is_object_dtype(series.dtype) or is_string_dtype(series.dtype)):
                raise OverlapSensitivityValidationError(f"{name}.{column} must use a string dtype")
            invalid = [
                value
                for value in series.tolist()
                if not isinstance(value, str) or not value.strip()
            ]
            if invalid:
                raise OverlapSensitivityValidationError(
                    f"{name}.{column} must contain non-empty strings"
                )

    identities: list[FeatureIdentity] = []
    for row in canonical.itertuples(index=False, name=None):
        identity: list[FeatureValue] = []
        for column, value in zip(FEATURE_COLUMNS, row, strict=True):
            if column in integer_features:
                if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
                    raise OverlapSensitivityValidationError(
                        f"{name}.{column} must contain integer values"
                    )
                identity.append(int(value))
            else:
                # The dtype/value validation above makes this narrowing exact.
                identity.append(str(value))
        identities.append(tuple(identity))
    return identities


def _one_dimensional(values: ArrayLike, name: str, length: int) -> NDArray[Any]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise OverlapSensitivityValidationError(f"{name} must be one-dimensional")
    if len(array) != length:
        raise OverlapSensitivityValidationError(
            f"{name} must have the same length as heldout_rows ({length})"
        )
    return array


def _binary(values: ArrayLike, name: str, length: int) -> NDArray[np.int_]:
    array = _one_dimensional(values, name, length)
    try:
        numeric = array.astype(float)
    except (TypeError, ValueError) as exc:
        raise OverlapSensitivityValidationError(
            f"{name} must contain only binary values 0 and 1"
        ) from exc
    if not np.isfinite(numeric).all() or not np.isin(numeric, [0.0, 1.0]).all():
        raise OverlapSensitivityValidationError(f"{name} must contain only binary values 0 and 1")
    return numeric.astype(int)


def _probabilities(values: ArrayLike, length: int) -> NDArray[np.float64]:
    array = _one_dimensional(values, "probabilities", length)
    try:
        numeric = array.astype(float)
    except (TypeError, ValueError) as exc:
        raise OverlapSensitivityValidationError(
            "probabilities must contain numeric values"
        ) from exc
    if not np.isfinite(numeric).all():
        raise OverlapSensitivityValidationError("probabilities must contain only finite values")
    if ((numeric < 0) | (numeric > 1)).any():
        raise OverlapSensitivityValidationError("probabilities must contain values between 0 and 1")
    return numeric.astype(float)


def _json_scalar(value: Any, name: str) -> JsonScalar:
    if isinstance(value, np.generic):
        value = value.item()
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError) as exc:
        raise OverlapSensitivityValidationError(f"{name} must contain scalar values") from exc
    if not isinstance(missing, (bool, np.bool_)):
        raise OverlapSensitivityValidationError(f"{name} must contain scalar values")
    if bool(missing):
        raise OverlapSensitivityValidationError(f"{name} must not contain missing values")
    if isinstance(value, float) and not np.isfinite(value):
        raise OverlapSensitivityValidationError(f"{name} must not contain non-finite values")
    if not isinstance(value, (str, int, float, bool)):
        raise OverlapSensitivityValidationError(
            f"{name} values must be scalar strings, numbers, or booleans"
        )
    return value


def _sensitive_values(
    values: ArrayLike,
    privileged_group: Any,
    length: int,
) -> tuple[NDArray[Any], JsonScalar]:
    array = _one_dimensional(values, "sensitive", length)
    normalized = [_json_scalar(value, "sensitive") for value in array.tolist()]
    privileged = _json_scalar(privileged_group, "privileged_group")

    def kind(value: JsonScalar) -> str:
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, str):
            return "str"
        if isinstance(value, Real):
            return "number"
        raise AssertionError("unreachable")

    observed_kinds = {kind(value) for value in normalized}
    if len(observed_kinds) != 1 or kind(privileged) not in observed_kinds:
        raise OverlapSensitivityValidationError(
            "sensitive and privileged_group must use one consistent scalar type family"
        )
    return np.asarray(normalized), privileged


def _unique_exact_identities(identities: list[FeatureIdentity]) -> list[FeatureIdentity]:
    ordered = sorted(identities)
    unique: list[FeatureIdentity] = []
    for identity in ordered:
        if not unique or identity != unique[-1]:
            unique.append(identity)
    return unique


def _has_exact_identity(
    ordered_reference: list[FeatureIdentity],
    identity: FeatureIdentity,
) -> bool:
    position = bisect_left(ordered_reference, identity)
    return position < len(ordered_reference) and ordered_reference[position] == identity


def exact_feature_overlap_mask(
    *,
    reference_rows: pd.DataFrame,
    compared_rows: pd.DataFrame,
) -> NDArray[np.bool_]:
    """Return exact canonical-feature membership in ``reference_rows``."""
    reference_identities = _validate_frame(reference_rows, "reference_rows")
    compared_identities = _validate_frame(compared_rows, "compared_rows")
    unique_reference = _unique_exact_identities(reference_identities)
    return np.asarray(
        [_has_exact_identity(unique_reference, identity) for identity in compared_identities],
        dtype=bool,
    )


def _json_number(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Integral):
        return int(value)
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _performance_metrics(
    y_true: NDArray[np.int_],
    predictions: NDArray[np.int_],
    probabilities: NDArray[np.float64],
) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics: dict[str, Any] = {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, probabilities)
    except ValueError:
        metrics["roc_auc"] = None
    try:
        metrics["pr_auc"] = average_precision_score(y_true, probabilities)
    except ValueError:
        metrics["pr_auc"] = None
    metrics["brier_score"] = brier_score_loss(y_true, probabilities)
    for key in ("SPD", "DI", "FPR_gap", "TPR_priv", "TPR_unpriv", "TPR_gap"):
        metrics[key] = None
    return metrics


def _observed_groups(sensitive: NDArray[Any]) -> list[JsonScalar]:
    groups: list[JsonScalar] = []
    for value in sensitive.tolist():
        normalized = _json_scalar(value, "sensitive")
        if not any(normalized == observed for observed in groups):
            groups.append(normalized)
    return groups


def _fairness_evidence(
    y_true: NDArray[np.int_],
    predictions: NDArray[np.int_],
    sensitive: NDArray[Any],
    privileged_group: JsonScalar,
) -> dict[str, Any]:
    groups = _observed_groups(sensitive)
    structure_reasons: list[str] = []
    if privileged_group not in groups:
        structure_reasons.append("privileged_group_absent")
    unprivileged_groups = [group for group in groups if group != privileged_group]
    if not unprivileged_groups:
        structure_reasons.append("unprivileged_group_absent")
    if len(groups) != 2:
        structure_reasons.append("binary_fairness_requires_exactly_two_groups")

    def unavailable(reasons: list[str]) -> dict[str, Any]:
        return {
            "evidence_status": "not_estimable",
            "evidence_reasons": sorted(set(reasons)),
        }

    metric_evidence: dict[str, dict[str, Any]] = {}
    if structure_reasons:
        for metric in _FAIRNESS_METRICS:
            metric_evidence[metric] = unavailable(structure_reasons)
    else:
        privileged_mask = sensitive == privileged_group
        unprivileged_mask = ~privileged_mask
        metric_evidence["SPD"] = {
            "evidence_status": "sufficient",
            "evidence_reasons": [],
        }

        privileged_selection_rate = float(predictions[privileged_mask].mean())
        if privileged_selection_rate == 0:
            metric_evidence["DI"] = unavailable(["zero_privileged_selection_rate"])
        else:
            metric_evidence["DI"] = {
                "evidence_status": "sufficient",
                "evidence_reasons": [],
            }

        tpr_reasons: list[str] = []
        if int(y_true[privileged_mask].sum()) == 0:
            tpr_reasons.append("zero_positive_denominator_privileged_group")
        if int(y_true[unprivileged_mask].sum()) == 0:
            tpr_reasons.append("zero_positive_denominator_unprivileged_group")
        metric_evidence["TPR_gap"] = (
            unavailable(tpr_reasons)
            if tpr_reasons
            else {"evidence_status": "sufficient", "evidence_reasons": []}
        )

        fpr_reasons: list[str] = []
        if int((y_true[privileged_mask] == 0).sum()) == 0:
            fpr_reasons.append("zero_negative_denominator_privileged_group")
        if int((y_true[unprivileged_mask] == 0).sum()) == 0:
            fpr_reasons.append("zero_negative_denominator_unprivileged_group")
        metric_evidence["FPR_gap"] = (
            unavailable(fpr_reasons)
            if fpr_reasons
            else {"evidence_status": "sufficient", "evidence_reasons": []}
        )

    unavailable_metrics = [
        metric
        for metric, evidence in metric_evidence.items()
        if evidence["evidence_status"] != "sufficient"
    ]
    if not unavailable_metrics:
        status = "sufficient"
    elif len(unavailable_metrics) == len(_FAIRNESS_METRICS):
        status = "not_estimable"
    else:
        status = "partial"
    reasons = sorted(
        {reason for evidence in metric_evidence.values() for reason in evidence["evidence_reasons"]}
    )
    return {
        "evidence_status": status,
        "evidence_reasons": reasons,
        "metrics": metric_evidence,
    }


def _evaluated_condition(
    y_true: NDArray[np.int_],
    predictions: NDArray[np.int_],
    probabilities: NDArray[np.float64],
    sensitive: NDArray[Any],
    privileged_group: JsonScalar,
) -> dict[str, Any]:
    evidence = _fairness_evidence(y_true, predictions, sensitive, privileged_group)
    groups = _observed_groups(sensitive)
    safe_for_canonical_evaluation = len(groups) == 2 and privileged_group in groups
    if safe_for_canonical_evaluation:
        from sklearn.exceptions import UndefinedMetricWarning

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            raw_metrics = evaluate_predictions(
                y_true=y_true,
                y_pred=predictions,
                sensitive=sensitive,
                privileged_group=privileged_group,
                y_proba=probabilities,
            )
    else:
        raw_metrics = _performance_metrics(y_true, predictions, probabilities)

    metrics = {key: _json_number(raw_metrics.get(key)) for key in _METRIC_KEYS}
    # Evidence status, rather than a NaN, owns every undefined fairness metric.
    for metric in _FAIRNESS_METRICS:
        if evidence["metrics"][metric]["evidence_status"] != "sufficient":
            metrics[metric] = None
    if evidence["metrics"]["TPR_gap"]["evidence_status"] != "sufficient":
        metrics["TPR_priv"] = None
        metrics["TPR_unpriv"] = None

    return {"metrics": metrics, "fairness_evidence": evidence}


def _empty_condition() -> dict[str, Any]:
    reason = ["no_overlap_excluded_rows"]
    metric_evidence = {
        metric: {
            "evidence_status": "not_estimable",
            "evidence_reasons": reason,
        }
        for metric in _FAIRNESS_METRICS
    }
    return {
        "metrics": None,
        "fairness_evidence": {
            "evidence_status": "not_estimable",
            "evidence_reasons": reason,
            "metrics": metric_evidence,
        },
    }


def _evaluation_slice(
    mask: NDArray[np.bool_],
    y_true: NDArray[np.int_],
    baseline_predictions: NDArray[np.int_],
    adjusted_predictions: NDArray[np.int_],
    probabilities: NDArray[np.float64],
    sensitive: NDArray[Any],
    privileged_group: JsonScalar,
    *,
    empty_reason: str | None = None,
) -> dict[str, Any]:
    row_count = int(mask.sum())
    if row_count == 0:
        reason = empty_reason or "empty_evaluation_slice"
        empty = _empty_condition()
        if reason != "no_overlap_excluded_rows":
            empty["fairness_evidence"]["evidence_reasons"] = [reason]
            for metric in _FAIRNESS_METRICS:
                empty["fairness_evidence"]["metrics"][metric]["evidence_reasons"] = [reason]
        return {
            "row_count": 0,
            "evidence_status": "not_estimable",
            "evidence_reasons": [reason],
            "baseline": empty,
            "adjusted": _empty_condition(),
        }

    baseline = _evaluated_condition(
        y_true[mask],
        baseline_predictions[mask],
        probabilities[mask],
        sensitive[mask],
        privileged_group,
    )
    adjusted = _evaluated_condition(
        y_true[mask],
        adjusted_predictions[mask],
        probabilities[mask],
        sensitive[mask],
        privileged_group,
    )
    reasons = sorted(
        {
            *baseline["fairness_evidence"]["evidence_reasons"],
            *adjusted["fairness_evidence"]["evidence_reasons"],
        }
    )
    status = "sufficient" if not reasons else "limited"
    return {
        "row_count": row_count,
        "evidence_status": status,
        "evidence_reasons": reasons,
        "baseline": baseline,
        "adjusted": adjusted,
    }


def exact_feature_overlap_sensitivity(
    *,
    reference_rows: pd.DataFrame,
    heldout_rows: pd.DataFrame,
    y_true: ArrayLike,
    baseline_predictions: ArrayLike,
    adjusted_predictions: ArrayLike,
    probabilities: ArrayLike,
    sensitive: ArrayLike,
    privileged_group: Any,
) -> dict[str, Any]:
    """Evaluate fixed held-out predictions before and after exact-overlap removal.

    ``reference_rows`` and ``heldout_rows`` may contain audit metadata such as
    labels, protected attributes, weights, or split names.  Every canonical
    feature must be present exactly once and only ``FEATURE_COLUMNS`` determine
    identity.  Consequently, different labels for an otherwise identical row
    do not change overlap membership.
    """

    reference_identities = _validate_frame(reference_rows, "reference_rows")
    heldout_identities = _validate_frame(heldout_rows, "heldout_rows")
    heldout_count = len(heldout_identities)
    true = _binary(y_true, "y_true", heldout_count)
    baseline = _binary(baseline_predictions, "baseline_predictions", heldout_count)
    adjusted = _binary(adjusted_predictions, "adjusted_predictions", heldout_count)
    probability = _probabilities(probabilities, heldout_count)
    sensitive_values, privileged = _sensitive_values(
        sensitive,
        privileged_group,
        heldout_count,
    )

    unique_reference = _unique_exact_identities(reference_identities)
    overlap_mask = exact_feature_overlap_mask(
        reference_rows=reference_rows,
        compared_rows=heldout_rows,
    )
    novel_mask = ~overlap_mask
    overlap_positions = np.flatnonzero(overlap_mask).astype(int).tolist()
    overlap_count = int(overlap_mask.sum())
    novel_count = heldout_count - overlap_count

    result = {
        "schema_version": OVERLAP_SENSITIVITY_SCHEMA_VERSION,
        "audit_type": "exact_feature_overlap_sensitivity",
        "identity": {
            "feature_contract_id": FEATURE_CONTRACT_ID,
            "columns": list(FEATURE_COLUMNS),
            "comparison": "sorted_canonical_tuples_with_exact_value_equality",
            "hash_used_for_final_equality": False,
            "non_feature_columns_ignored": True,
        },
        "policy": {
            "retuned": False,
            "statement": (
                "Predictions, probabilities, thresholds, and policy outputs are fixed inputs; "
                "the audit only filters evaluation rows."
            ),
        },
        "counts": {
            "reference_rows": len(reference_identities),
            "reference_unique_feature_identities": len(unique_reference),
            "reference_duplicate_rows_beyond_first": (
                len(reference_identities) - len(unique_reference)
            ),
            "held_out_rows": heldout_count,
            "overlap_rows": overlap_count,
            "novel_rows": novel_count,
            "overlap_rate": float(overlap_count / heldout_count),
        },
        "overlap_positions": overlap_positions,
        "slices": {
            "all_held_out": _evaluation_slice(
                np.ones(heldout_count, dtype=bool),
                true,
                baseline,
                adjusted,
                probability,
                sensitive_values,
                privileged,
            ),
            "overlap_excluded": _evaluation_slice(
                novel_mask,
                true,
                baseline,
                adjusted,
                probability,
                sensitive_values,
                privileged,
                empty_reason="no_overlap_excluded_rows",
            ),
        },
    }
    return result
