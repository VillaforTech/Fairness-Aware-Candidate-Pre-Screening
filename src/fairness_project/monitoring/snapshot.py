"""Privacy-bounded snapshots for offline drift and fairness audits.

The snapshot contract intentionally stores aggregate statistics and compact
quantile sketches, never rows. Comparisons are descriptive audit evidence. They
are not a production monitor, a statistical-significance test, or approval to
use a model for employment decisions.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas.api.types import is_bool_dtype, is_numeric_dtype

SNAPSHOT_SCHEMA_VERSION = "1.0"
SNAPSHOT_KIND = "fairness_project.offline_monitoring_snapshot"
COMPARISON_KIND = "fairness_project.offline_monitoring_comparison"
AUDIT_SCOPE = "offline_audit_only_not_production_evidence"

_QUANTILE_PROBABILITIES = np.linspace(0.0, 1.0, 101)
_UNKNOWN_TOKENS = frozenset({"", "?", "unknown", "unk", "__unknown__", "<unknown>"})
_DERIVED_VALUE_RTOL = 1e-12
_DERIVED_VALUE_ATOL = 1e-12


@dataclass(frozen=True)
class DriftThresholds:
    """Fail-closed policy thresholds for one snapshot comparison."""

    min_rows: int = 100
    min_group_rows: int = 30
    max_numeric_psi: float = 0.25
    max_numeric_ks_distance: float = 0.20
    max_categorical_total_variation: float = 0.20
    max_oov_share: float = 0.05
    max_unknown_share_increase: float = 0.05
    max_score_psi: float = 0.25
    max_score_ks_distance: float = 0.20
    max_selection_rate_change: float = 0.10
    max_group_composition_total_variation: float = 0.10
    max_accuracy_drop: float = 0.05
    max_true_positive_rate_drop: float = 0.10
    max_false_positive_rate_increase: float = 0.10
    max_selection_rate_gap_increase: float = 0.10
    max_true_positive_rate_gap_increase: float = 0.10
    max_false_positive_rate_gap_increase: float = 0.10
    require_labels: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.min_rows, int)
            or isinstance(self.min_rows, bool)
            or self.min_rows < 1
        ):
            raise ValueError("min_rows must be a positive integer")
        if (
            not isinstance(self.min_group_rows, int)
            or isinstance(self.min_group_rows, bool)
            or self.min_group_rows < 1
        ):
            raise ValueError("min_group_rows must be a positive integer")

        bounded_fields = (
            "max_numeric_ks_distance",
            "max_categorical_total_variation",
            "max_oov_share",
            "max_unknown_share_increase",
            "max_score_ks_distance",
            "max_selection_rate_change",
            "max_group_composition_total_variation",
            "max_accuracy_drop",
            "max_true_positive_rate_drop",
            "max_false_positive_rate_increase",
            "max_selection_rate_gap_increase",
            "max_true_positive_rate_gap_increase",
            "max_false_positive_rate_gap_increase",
        )
        for name in bounded_fields:
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be numeric")
            if not np.isfinite(float(value)) or not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be finite and between 0 and 1")

        for name in ("max_numeric_psi", "max_score_psi"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be numeric")
            if not np.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not isinstance(self.require_labels, bool):
            raise ValueError("require_labels must be Boolean")


SnapshotThresholds = DriftThresholds


def validate_snapshot(snapshot: Mapping[str, Any]) -> None:
    """Validate a serialized snapshot against the exact aggregate-only contract."""

    _validate_snapshot(snapshot)


def build_snapshot(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    score_column: str,
    prediction_column: str,
    protected_columns: Sequence[str] = (),
    label_column: str | None = None,
    sample_weight_column: str | None = None,
    categorical_columns: Sequence[str] = (),
    timestamp: str | datetime | None = None,
) -> dict[str, Any]:
    """Build a JSON-ready, aggregate-only snapshot from an offline audit frame.

    ``sample_weight_column`` is accepted only as audit metadata and weighted
    sensitivity context. It is forbidden from the model feature contract and
    never affects the primary drift gate.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if frame.empty:
        raise ValueError("frame must contain at least one row")
    if frame.columns.has_duplicates:
        raise ValueError("frame columns must be unique")

    features = _validated_names("feature_columns", feature_columns, require_nonempty=True)
    protected = _validated_names("protected_columns", protected_columns)
    categoricals = _validated_names("categorical_columns", categorical_columns)
    score = _validated_name("score_column", score_column)
    prediction = _validated_name("prediction_column", prediction_column)
    label = _validated_optional_name("label_column", label_column)
    weight_column = _validated_optional_name("sample_weight_column", sample_weight_column)

    if not set(categoricals).issubset(features):
        raise ValueError("categorical_columns must be a subset of feature_columns")
    if set(features) & set(protected):
        raise ValueError("protected_columns cannot be model feature columns")
    if weight_column in features:
        raise ValueError("sample_weight_column is audit-only and cannot be a feature")

    role_columns = features + protected + [score, prediction]
    if label is not None:
        role_columns.append(label)
    if weight_column is not None:
        role_columns.append(weight_column)
    if len(role_columns) != len(set(role_columns)):
        raise ValueError("column roles must be distinct")

    actual_columns = frame.columns.tolist()
    if set(actual_columns) != set(role_columns):
        missing = sorted(set(role_columns) - set(actual_columns))
        unexpected = sorted(set(actual_columns) - set(role_columns))
        raise ValueError(
            f"frame must match the declared snapshot schema exactly; missing={missing}, "
            f"unexpected={unexpected}"
        )

    numeric_features = [name for name in features if name not in categoricals]
    for name in numeric_features + [score]:
        _validated_numeric_series(frame[name], name)
    for name in categoricals + protected:
        _validated_categorical_series(frame[name], name)

    predictions = _validated_binary_series(frame[prediction], prediction)
    scores = frame[score].to_numpy(dtype=float)
    if (scores < 0).any() or (scores > 1).any():
        raise ValueError(f"Column '{score}' must contain scores between 0 and 1")

    labels: np.ndarray | None = None
    if label is not None:
        labels = _validated_binary_series(frame[label], label)

    weights: np.ndarray | None = None
    if weight_column is not None:
        _validated_numeric_series(frame[weight_column], weight_column)
        weights = frame[weight_column].to_numpy(dtype=float)
        if (weights < 0).any():
            raise ValueError("sample weights must be non-negative")
        if float(weights.sum()) <= 0:
            raise ValueError("sample weights must have positive total weight")

    numeric_distributions = {
        name: _numeric_summary(frame[name].to_numpy(dtype=float)) for name in numeric_features
    }
    categorical_distributions = {name: _categorical_summary(frame[name]) for name in categoricals}
    protected_summaries = {name: _categorical_summary(frame[name]) for name in protected}

    delayed_labels: dict[str, Any]
    if labels is None:
        delayed_labels = {
            "status": "unavailable",
            "column": None,
            "class_counts": None,
            "performance": None,
            "protected_group_metrics": {},
        }
    else:
        delayed_labels = {
            "status": "available",
            "column": label,
            "class_counts": {
                "negative": int((labels == 0).sum()),
                "positive": int((labels == 1).sum()),
            },
            "performance": _binary_metrics(labels, predictions),
            "protected_group_metrics": {
                name: _protected_group_metrics(frame[name], labels, predictions)
                for name in protected
            },
        }

    weight_audit = _weight_audit(
        weights=weights,
        column=weight_column,
        scores=scores,
        predictions=predictions,
        labels=labels,
        protected={name: frame[name] for name in protected},
    )

    snapshot: dict[str, Any] = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "kind": SNAPSHOT_KIND,
        "audit_scope": AUDIT_SCOPE,
        "generated_at": _normalized_timestamp(timestamp),
        "row_count": int(len(frame)),
        "contract": {
            "columns": actual_columns,
            "dtypes": {name: str(frame[name].dtype) for name in actual_columns},
            "feature_columns": features,
            "categorical_columns": categoricals,
            "numeric_columns": numeric_features,
            "score_column": score,
            "prediction_column": prediction,
            "protected_columns": protected,
            "label_column": label,
            "sample_weight_column": weight_column,
        },
        "distributions": {
            "numeric": numeric_distributions,
            "categorical": categorical_distributions,
        },
        "outcomes": {
            "score": _numeric_summary(scores),
            "prediction": _prediction_summary(predictions),
        },
        "protected_audit": {
            "row_level_data_included": False,
            "columns": protected_summaries,
        },
        "delayed_labels": delayed_labels,
        "weight_audit": weight_audit,
        "evidence": {
            "snapshot_rows": int(len(frame)),
            "labels_available": labels is not None,
            "protected_attributes_configured": bool(protected),
            "contains_only_aggregate_statistics": True,
        },
    }
    _validate_snapshot(snapshot)
    return snapshot


def compare_snapshots(
    reference: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    thresholds: DriftThresholds | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare two aggregate snapshots with a fail-closed audit gate."""

    _validate_snapshot(reference)
    _validate_snapshot(current)
    policy = _coerce_thresholds(thresholds)
    _validate_comparable_contract(reference["contract"], current["contract"])

    violations: list[dict[str, Any]] = []
    evidence_gaps: list[dict[str, Any]] = []
    ref_rows = int(reference["row_count"])
    cur_rows = int(current["row_count"])
    for side, count in (("reference", ref_rows), ("current", cur_rows)):
        if count < policy.min_rows:
            evidence_gaps.append(
                {
                    "code": f"{side}_rows_below_minimum",
                    "observed": count,
                    "required": policy.min_rows,
                }
            )

    numeric_drift: dict[str, Any] = {}
    for name in reference["contract"]["numeric_columns"]:
        drift = _numeric_drift(
            reference["distributions"]["numeric"][name],
            current["distributions"]["numeric"][name],
        )
        numeric_drift[name] = drift
        _append_upper_violation(
            violations,
            code="numeric_psi_exceeded",
            scope=name,
            observed=drift["psi"],
            threshold=policy.max_numeric_psi,
        )
        _append_upper_violation(
            violations,
            code="numeric_ks_distance_exceeded",
            scope=name,
            observed=drift["ks_distance"],
            threshold=policy.max_numeric_ks_distance,
        )

    categorical_drift: dict[str, Any] = {}
    for name in reference["contract"]["categorical_columns"]:
        drift = _categorical_drift(
            reference["distributions"]["categorical"][name],
            current["distributions"]["categorical"][name],
        )
        categorical_drift[name] = drift
        _append_upper_violation(
            violations,
            code="categorical_total_variation_exceeded",
            scope=name,
            observed=drift["total_variation_distance"],
            threshold=policy.max_categorical_total_variation,
        )
        _append_upper_violation(
            violations,
            code="categorical_oov_share_exceeded",
            scope=name,
            observed=drift["oov_share"],
            threshold=policy.max_oov_share,
        )
        _append_upper_violation(
            violations,
            code="categorical_unknown_share_increase_exceeded",
            scope=name,
            observed=max(0.0, drift["unknown_share_change"]),
            threshold=policy.max_unknown_share_increase,
        )

    score_drift = _numeric_drift(reference["outcomes"]["score"], current["outcomes"]["score"])
    _append_upper_violation(
        violations,
        code="score_psi_exceeded",
        scope="score",
        observed=score_drift["psi"],
        threshold=policy.max_score_psi,
    )
    _append_upper_violation(
        violations,
        code="score_ks_distance_exceeded",
        scope="score",
        observed=score_drift["ks_distance"],
        threshold=policy.max_score_ks_distance,
    )

    ref_selection = float(reference["outcomes"]["prediction"]["selection_rate"])
    cur_selection = float(current["outcomes"]["prediction"]["selection_rate"])
    selection_drift = {
        "reference": ref_selection,
        "current": cur_selection,
        "change": cur_selection - ref_selection,
        "absolute_change": abs(cur_selection - ref_selection),
    }
    _append_upper_violation(
        violations,
        code="selection_rate_change_exceeded",
        scope="prediction",
        observed=selection_drift["absolute_change"],
        threshold=policy.max_selection_rate_change,
    )

    group_composition: dict[str, Any] = {}
    for name in reference["contract"]["protected_columns"]:
        ref_group = reference["protected_audit"]["columns"][name]
        cur_group = current["protected_audit"]["columns"][name]
        drift = _categorical_drift(ref_group, cur_group)
        group_composition[name] = drift
        _append_upper_violation(
            violations,
            code="group_composition_total_variation_exceeded",
            scope=name,
            observed=drift["total_variation_distance"],
            threshold=policy.max_group_composition_total_variation,
        )
        _append_group_evidence_gaps(evidence_gaps, name, ref_group, cur_group, policy)

    delayed_label_drift = _compare_delayed_labels(
        reference,
        current,
        policy,
        violations,
        evidence_gaps,
    )

    if violations:
        gate_status = "FAIL"
    elif evidence_gaps:
        gate_status = "INSUFFICIENT_EVIDENCE"
    else:
        gate_status = "PASS"

    result: dict[str, Any] = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "kind": COMPARISON_KIND,
        "audit_scope": AUDIT_SCOPE,
        "reference_generated_at": reference["generated_at"],
        "current_generated_at": current["generated_at"],
        "schema_match": True,
        "thresholds": asdict(policy),
        "evidence": {
            "status": "sufficient" if not evidence_gaps else "insufficient",
            "reference_rows": ref_rows,
            "current_rows": cur_rows,
            "evidence_gaps": evidence_gaps,
        },
        "feature_drift": {
            "numeric": numeric_drift,
            "categorical": categorical_drift,
        },
        "outcome_drift": {
            "score": score_drift,
            "selection": selection_drift,
        },
        "group_composition_drift": group_composition,
        "delayed_label_drift": delayed_label_drift,
        "weight_audit": {
            "primary_gate_uses_sample_weights": False,
            "reference_status": reference["weight_audit"]["status"],
            "current_status": current["weight_audit"]["status"],
        },
        "gate": {
            "status": gate_status,
            "passed": gate_status == "PASS",
            "fail_closed": True,
            "violations": violations,
            "evidence_gaps": evidence_gaps,
            "meaning": "Configured offline audit thresholds only; not deployment approval.",
        },
        "methodology": {
            "numeric": (
                "PSI and KS-like descriptive distance reconstructed from 101-point quantile "
                "sketches; no hypothesis-test p-values"
            ),
            "categorical": "total variation, reference-vocabulary OOV share, and unknown share",
            "protected_data": "aggregate group statistics only; no row-level protected data",
            "weights": "sample weights are audit-only and do not affect the primary gate",
        },
    }
    _ensure_json_ready(result, "comparison")
    return result


def _validated_names(
    name: str, values: Sequence[str], *, require_nonempty: bool = False
) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of column names")
    result = [_validated_name(name, value) for value in values]
    if require_nonempty and not result:
        raise ValueError(f"{name} must contain at least one column")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _validated_name(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must contain non-empty string column names")
    return value


def _validated_optional_name(name: str, value: object) -> str | None:
    if value is None:
        return None
    return _validated_name(name, value)


def _validated_numeric_series(series: pd.Series, name: str) -> None:
    if not is_numeric_dtype(series.dtype) or is_bool_dtype(series.dtype):
        raise ValueError(f"Column '{name}' must use a numeric dtype")
    values = series.to_numpy(dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError(f"Column '{name}' must contain finite numeric values")


def _validated_categorical_series(series: pd.Series, name: str) -> None:
    if series.isna().any():
        raise ValueError(f"Column '{name}' must not contain null values")
    if not series.map(lambda value: isinstance(value, str)).all():
        raise ValueError(f"Column '{name}' must contain string categories")


def _validated_binary_series(series: pd.Series, name: str) -> NDArray[np.int_]:
    if series.isna().any():
        raise ValueError(f"Column '{name}' must not contain null values")
    values = series.to_numpy()
    try:
        numeric = values.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Column '{name}' must contain binary values 0 and 1") from exc
    if not np.isfinite(numeric).all() or not np.isin(numeric, [0.0, 1.0]).all():
        raise ValueError(f"Column '{name}' must contain binary values 0 and 1")
    return np.asarray(numeric, dtype=np.int_)


def _numeric_summary(values: np.ndarray) -> dict[str, Any]:
    numeric = np.asarray(values, dtype=float)
    quantiles = np.quantile(numeric, _QUANTILE_PROBABILITIES)
    return {
        "count": int(len(numeric)),
        "mean": float(numeric.mean()),
        "standard_deviation": float(numeric.std(ddof=0)),
        "minimum": float(numeric.min()),
        "maximum": float(numeric.max()),
        "quantile_probabilities": [float(value) for value in _QUANTILE_PROBABILITIES],
        "quantile_values": [float(value) for value in quantiles],
    }


def _categorical_summary(series: pd.Series) -> dict[str, Any]:
    counts_series = series.value_counts(dropna=False).sort_index()
    counts = {str(name): int(value) for name, value in counts_series.items()}
    total = int(len(series))
    shares = {name: value / total for name, value in counts.items()}
    unknown_count = sum(
        count for name, count in counts.items() if name.strip().casefold() in _UNKNOWN_TOKENS
    )
    return {
        "count": total,
        "cardinality": len(counts),
        "counts": counts,
        "shares": shares,
        "unknown_count": int(unknown_count),
        "unknown_share": unknown_count / total,
    }


def _prediction_summary(predictions: np.ndarray) -> dict[str, Any]:
    positive = int((predictions == 1).sum())
    count = int(len(predictions))
    return {
        "count": count,
        "negative_count": count - positive,
        "positive_count": positive,
        "selection_rate": positive / count,
    }


def _binary_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    weight = np.ones(len(labels), dtype=float) if weights is None else weights.astype(float)
    tp = float(weight[(labels == 1) & (predictions == 1)].sum())
    fn = float(weight[(labels == 1) & (predictions == 0)].sum())
    fp = float(weight[(labels == 0) & (predictions == 1)].sum())
    tn = float(weight[(labels == 0) & (predictions == 0)].sum())
    total = tp + fn + fp + tn
    positive = tp + fn
    negative = fp + tn
    selected = tp + fp
    return {
        "count": int(len(labels)),
        "total_weight": total,
        "positive_weight": positive,
        "negative_weight": negative,
        "true_positive": tp,
        "false_negative": fn,
        "false_positive": fp,
        "true_negative": tn,
        "accuracy": (tp + tn) / total,
        "true_positive_rate": _safe_ratio(tp, positive),
        "false_positive_rate": _safe_ratio(fp, negative),
        "precision": _safe_ratio(tp, selected),
        "base_rate": positive / total,
        "selection_rate": selected / total,
    }


def _protected_group_metrics(
    series: pd.Series,
    labels: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    values = series.to_numpy(dtype=object)
    for group in sorted(str(value) for value in series.unique()):
        mask = values == group
        group_weights = None if weights is None else weights[mask]
        groups[group] = _binary_metrics(labels[mask], predictions[mask], group_weights)
    return {"groups": groups, "row_level_data_included": False}


def _weight_audit(
    *,
    weights: np.ndarray | None,
    column: str | None,
    scores: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray | None,
    protected: Mapping[str, pd.Series],
) -> dict[str, Any]:
    if weights is None:
        return {
            "status": "not_provided",
            "column": None,
            "role": "audit_only",
            "used_as_model_feature": False,
            "used_for_primary_metrics_or_gate": False,
            "summary": None,
            "weighted_sensitivity": None,
        }
    total = float(weights.sum())
    effective_n = total**2 / float(np.square(weights).sum())
    group_composition: dict[str, Any] = {}
    for name, series in protected.items():
        values = series.to_numpy(dtype=object)
        group_composition[name] = {
            str(group): float(weights[values == group].sum()) / total
            for group in sorted(series.unique())
        }
    return {
        "status": "provided",
        "column": column,
        "role": "audit_only",
        "used_as_model_feature": False,
        "used_for_primary_metrics_or_gate": False,
        "summary": {
            "count": int(len(weights)),
            "sum": total,
            "mean": float(weights.mean()),
            "minimum": float(weights.min()),
            "maximum": float(weights.max()),
            "kish_effective_sample_size": effective_n,
        },
        "weighted_sensitivity": {
            "score_mean": float(np.average(scores, weights=weights)),
            "selection_rate": float(np.average(predictions, weights=weights)),
            "performance": (
                None if labels is None else _binary_metrics(labels, predictions, weights)
            ),
            "protected_group_composition": group_composition,
        },
    }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator <= 0 else numerator / denominator


def _normalized_timestamp(timestamp: str | datetime | None) -> str:
    if timestamp is None:
        value = datetime.now(timezone.utc)
    elif isinstance(timestamp, str):
        text = timestamp.strip()
        if not text:
            raise ValueError("timestamp must be a non-empty ISO-8601 value")
        try:
            value = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("timestamp must be an ISO-8601 value") from exc
    elif isinstance(timestamp, datetime):
        value = timestamp
    else:
        raise TypeError("timestamp must be a string, datetime, or None")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _coerce_thresholds(
    thresholds: DriftThresholds | Mapping[str, Any] | None,
) -> DriftThresholds:
    if thresholds is None:
        return DriftThresholds()
    if isinstance(thresholds, DriftThresholds):
        return thresholds
    if not isinstance(thresholds, Mapping):
        raise TypeError("thresholds must be DriftThresholds, a mapping, or None")
    allowed = {field.name for field in fields(DriftThresholds)}
    unexpected = sorted(set(thresholds) - allowed)
    if unexpected:
        raise ValueError(f"Unknown threshold fields: {unexpected}")
    return DriftThresholds(**dict(thresholds))


def _numeric_drift(reference: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, float]:
    ref_values = np.asarray(reference["quantile_values"], dtype=float)
    cur_values = np.asarray(current["quantile_values"], dtype=float)
    points = np.unique(np.concatenate([ref_values, cur_values]))
    ref_cdf = np.searchsorted(ref_values, points, side="right") / len(ref_values)
    cur_cdf = np.searchsorted(cur_values, points, side="right") / len(cur_values)
    ks_distance = float(np.max(np.abs(ref_cdf - cur_cdf)))

    interior_edges = np.unique(np.quantile(ref_values, np.linspace(0.1, 0.9, 9)))
    if ref_values[0] == ref_values[-1]:
        ref_constant = float(ref_values[0])
        cur_median = float(np.median(cur_values))
        if cur_median == ref_constant and cur_values[0] == cur_values[-1]:
            interior_edges = np.array([], dtype=float)
        elif cur_median != ref_constant:
            interior_edges = np.array([(ref_constant + cur_median) / 2], dtype=float)
        else:
            distinct_current = cur_values[cur_values != ref_constant]
            nearest = float(distinct_current[np.argmin(np.abs(distinct_current - ref_constant))])
            interior_edges = np.array([(ref_constant + nearest) / 2], dtype=float)
    edges = np.concatenate(([-np.inf], interior_edges, [np.inf]))
    ref_counts = np.histogram(ref_values, bins=edges)[0].astype(float)
    cur_counts = np.histogram(cur_values, bins=edges)[0].astype(float)
    smoothing = 0.5
    ref_share = (ref_counts + smoothing) / (ref_counts.sum() + smoothing * len(ref_counts))
    cur_share = (cur_counts + smoothing) / (cur_counts.sum() + smoothing * len(cur_counts))
    psi = float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))
    return {
        "psi": psi,
        "ks_distance": ks_distance,
        "reference_mean": float(reference["mean"]),
        "current_mean": float(current["mean"]),
        "mean_change": float(current["mean"]) - float(reference["mean"]),
    }


def _categorical_drift(reference: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    ref_shares = {str(key): float(value) for key, value in reference["shares"].items()}
    cur_shares = {str(key): float(value) for key, value in current["shares"].items()}
    categories = sorted(set(ref_shares) | set(cur_shares))
    total_variation = 0.5 * sum(
        abs(cur_shares.get(category, 0.0) - ref_shares.get(category, 0.0))
        for category in categories
    )
    oov_categories = sorted(set(cur_shares) - set(ref_shares))
    oov_share = sum(cur_shares[category] for category in oov_categories)
    ref_unknown = float(reference["unknown_share"])
    cur_unknown = float(current["unknown_share"])
    return {
        "total_variation_distance": total_variation,
        "oov_categories": oov_categories,
        "oov_share": oov_share,
        "reference_unknown_share": ref_unknown,
        "current_unknown_share": cur_unknown,
        "unknown_share_change": cur_unknown - ref_unknown,
        "maximum_absolute_category_share_change": max(
            (
                abs(cur_shares.get(category, 0.0) - ref_shares.get(category, 0.0))
                for category in categories
            ),
            default=0.0,
        ),
    }


def _append_upper_violation(
    violations: list[dict[str, Any]],
    *,
    code: str,
    scope: str,
    observed: float,
    threshold: float,
) -> None:
    if observed > threshold + 1e-12:
        violations.append(
            {
                "code": code,
                "scope": scope,
                "observed": observed,
                "threshold": threshold,
                "comparison": "greater_than",
            }
        )


def _append_group_evidence_gaps(
    gaps: list[dict[str, Any]],
    name: str,
    reference: Mapping[str, Any],
    current: Mapping[str, Any],
    policy: DriftThresholds,
) -> None:
    ref_counts = {str(key): int(value) for key, value in reference["counts"].items()}
    cur_counts = {str(key): int(value) for key, value in current["counts"].items()}
    for group in sorted(set(ref_counts) | set(cur_counts)):
        if ref_counts.get(group, 0) < policy.min_group_rows:
            gaps.append(
                {
                    "code": "reference_group_rows_below_minimum",
                    "scope": f"{name}={group}",
                    "observed": ref_counts.get(group, 0),
                    "required": policy.min_group_rows,
                }
            )
        if cur_counts.get(group, 0) < policy.min_group_rows:
            gaps.append(
                {
                    "code": "current_group_rows_below_minimum",
                    "scope": f"{name}={group}",
                    "observed": cur_counts.get(group, 0),
                    "required": policy.min_group_rows,
                }
            )


def _compare_delayed_labels(
    reference: Mapping[str, Any],
    current: Mapping[str, Any],
    policy: DriftThresholds,
    violations: list[dict[str, Any]],
    evidence_gaps: list[dict[str, Any]],
) -> dict[str, Any]:
    ref_labels = reference["delayed_labels"]
    cur_labels = current["delayed_labels"]
    if ref_labels["status"] != "available" or cur_labels["status"] != "available":
        if policy.require_labels:
            evidence_gaps.append(
                {
                    "code": "required_delayed_labels_unavailable",
                    "reference_status": ref_labels["status"],
                    "current_status": cur_labels["status"],
                }
            )
        return {
            "status": "unavailable_in_one_or_both_snapshots",
            "performance": None,
            "fairness": {},
            "used_for_gate": False,
        }

    ref_performance = ref_labels["performance"]
    cur_performance = cur_labels["performance"]
    performance: dict[str, Any] = {}
    for metric in (
        "accuracy",
        "true_positive_rate",
        "false_positive_rate",
        "precision",
        "base_rate",
        "selection_rate",
    ):
        ref_value = ref_performance[metric]
        cur_value = cur_performance[metric]
        performance[metric] = {
            "reference": ref_value,
            "current": cur_value,
            "change": None if ref_value is None or cur_value is None else cur_value - ref_value,
        }
    for metric in ("true_positive_rate", "false_positive_rate"):
        if performance[metric]["change"] is None:
            evidence_gaps.append(
                {
                    "code": "overall_label_metric_not_estimable",
                    "scope": metric,
                    "reference": performance[metric]["reference"],
                    "current": performance[metric]["current"],
                }
            )

    _append_metric_drop(
        violations,
        performance,
        "accuracy",
        "accuracy_drop_exceeded",
        policy.max_accuracy_drop,
    )
    _append_metric_drop(
        violations,
        performance,
        "true_positive_rate",
        "true_positive_rate_drop_exceeded",
        policy.max_true_positive_rate_drop,
    )
    fpr_change = performance["false_positive_rate"]["change"]
    if fpr_change is not None:
        _append_upper_violation(
            violations,
            code="false_positive_rate_increase_exceeded",
            scope="overall",
            observed=max(0.0, fpr_change),
            threshold=policy.max_false_positive_rate_increase,
        )

    fairness: dict[str, Any] = {}
    for name in reference["contract"]["protected_columns"]:
        ref_groups = ref_labels["protected_group_metrics"][name]["groups"]
        cur_groups = cur_labels["protected_group_metrics"][name]["groups"]
        eligible = [
            group
            for group in sorted(set(ref_groups) & set(cur_groups))
            if int(ref_groups[group]["count"]) >= policy.min_group_rows
            and int(cur_groups[group]["count"]) >= policy.min_group_rows
        ]
        metric_drift: dict[str, Any] = {}
        for metric in ("selection_rate", "true_positive_rate", "false_positive_rate"):
            ref_gap = _group_span(ref_groups, eligible, metric)
            cur_gap = _group_span(cur_groups, eligible, metric)
            metric_drift[metric] = {
                "reference_absolute_gap": ref_gap,
                "current_absolute_gap": cur_gap,
                "gap_change": None if ref_gap is None or cur_gap is None else cur_gap - ref_gap,
            }
        fairness[name] = {
            "eligible_groups": eligible,
            "metrics": metric_drift,
            "row_level_data_included": False,
        }
        if len(eligible) < 2:
            evidence_gaps.append(
                {
                    "code": "insufficient_groups_for_fairness_drift",
                    "scope": name,
                    "observed": len(eligible),
                    "required": 2,
                }
            )
            continue
        for metric in ("true_positive_rate", "false_positive_rate"):
            if metric_drift[metric]["gap_change"] is None:
                evidence_gaps.append(
                    {
                        "code": "group_fairness_metric_not_estimable",
                        "scope": f"{name}:{metric}",
                    }
                )
        threshold_by_metric = {
            "selection_rate": policy.max_selection_rate_gap_increase,
            "true_positive_rate": policy.max_true_positive_rate_gap_increase,
            "false_positive_rate": policy.max_false_positive_rate_gap_increase,
        }
        for metric, threshold in threshold_by_metric.items():
            change = metric_drift[metric]["gap_change"]
            if change is not None:
                _append_upper_violation(
                    violations,
                    code=f"{metric}_gap_increase_exceeded",
                    scope=name,
                    observed=max(0.0, change),
                    threshold=threshold,
                )

    return {
        "status": "available",
        "performance": performance,
        "fairness": fairness,
        "used_for_gate": True,
    }


def _append_metric_drop(
    violations: list[dict[str, Any]],
    performance: Mapping[str, Any],
    metric: str,
    code: str,
    threshold: float,
) -> None:
    change = performance[metric]["change"]
    if change is not None:
        _append_upper_violation(
            violations,
            code=code,
            scope="overall",
            observed=max(0.0, -change),
            threshold=threshold,
        )


def _group_span(groups: Mapping[str, Any], eligible: Sequence[str], metric: str) -> float | None:
    values = [groups[group][metric] for group in eligible]
    finite_values = [float(value) for value in values if value is not None]
    if len(finite_values) < 2:
        return None
    return max(finite_values) - min(finite_values)


def _validate_comparable_contract(reference: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    required_roles = (
        "feature_columns",
        "categorical_columns",
        "numeric_columns",
        "score_column",
        "prediction_column",
        "protected_columns",
    )
    mismatched = [name for name in required_roles if reference[name] != current[name]]

    required_columns = (
        list(reference["feature_columns"])
        + list(reference["protected_columns"])
        + [reference["score_column"], reference["prediction_column"]]
    )
    for name in required_columns:
        if reference["dtypes"].get(name) != current["dtypes"].get(name):
            mismatched.append(f"dtype:{name}")

    for optional in ("label_column", "sample_weight_column"):
        ref_name = reference[optional]
        cur_name = current[optional]
        if ref_name is not None and cur_name is not None:
            if ref_name != cur_name:
                mismatched.append(optional)
            elif reference["dtypes"].get(ref_name) != current["dtypes"].get(cur_name):
                mismatched.append(f"dtype:{ref_name}")
    if mismatched:
        raise ValueError(f"Snapshots have incompatible exact schemas: {sorted(set(mismatched))}")


def _validate_snapshot(snapshot: Mapping[str, Any]) -> None:
    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")
    expected_top = {
        "schema_version",
        "kind",
        "audit_scope",
        "generated_at",
        "row_count",
        "contract",
        "distributions",
        "outcomes",
        "protected_audit",
        "delayed_labels",
        "weight_audit",
        "evidence",
    }
    if set(snapshot) != expected_top:
        raise ValueError("snapshot does not match the exact top-level schema")
    if snapshot["schema_version"] != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("snapshot schema_version is unsupported")
    if snapshot["kind"] != SNAPSHOT_KIND or snapshot["audit_scope"] != AUDIT_SCOPE:
        raise ValueError("snapshot kind or audit scope is invalid")
    _normalized_timestamp(snapshot["generated_at"])
    if isinstance(snapshot["row_count"], bool) or not isinstance(snapshot["row_count"], int):
        raise ValueError("snapshot row_count must be a positive integer")
    if snapshot["row_count"] < 1:
        raise ValueError("snapshot row_count must be a positive integer")

    contract = snapshot["contract"]
    expected_contract = {
        "columns",
        "dtypes",
        "feature_columns",
        "categorical_columns",
        "numeric_columns",
        "score_column",
        "prediction_column",
        "protected_columns",
        "label_column",
        "sample_weight_column",
    }
    if not isinstance(contract, Mapping) or set(contract) != expected_contract:
        raise ValueError("snapshot contract does not match the exact schema")
    columns = contract["columns"]
    if not isinstance(columns, list) or not all(isinstance(value, str) for value in columns):
        raise ValueError("snapshot contract columns must be a list of strings")
    if len(columns) != len(set(columns)):
        raise ValueError("snapshot contract columns must be unique")
    if not isinstance(contract["dtypes"], Mapping) or set(contract["dtypes"]) != set(columns):
        raise ValueError("snapshot contract dtypes must exactly cover columns")
    if not all(isinstance(value, str) and value for value in contract["dtypes"].values()):
        raise ValueError("snapshot contract dtypes must be non-empty strings")

    for list_name in (
        "feature_columns",
        "categorical_columns",
        "numeric_columns",
        "protected_columns",
    ):
        values = contract[list_name]
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError(f"snapshot contract {list_name} must be a list of strings")
        if len(values) != len(set(values)):
            raise ValueError(f"snapshot contract {list_name} must be unique")
    if not contract["feature_columns"]:
        raise ValueError("snapshot contract feature_columns must not be empty")
    for scalar_name in ("score_column", "prediction_column"):
        if not isinstance(contract[scalar_name], str) or not contract[scalar_name]:
            raise ValueError(f"snapshot contract {scalar_name} must be a non-empty string")
    for optional_name in ("label_column", "sample_weight_column"):
        optional_value = contract[optional_name]
        if optional_value is not None and (
            not isinstance(optional_value, str) or not optional_value
        ):
            raise ValueError(
                f"snapshot contract {optional_name} must be a non-empty string or null"
            )

    features = contract["feature_columns"]
    categoricals = contract["categorical_columns"]
    numeric = contract["numeric_columns"]
    protected = contract["protected_columns"]
    if set(categoricals) | set(numeric) != set(features) or set(categoricals) & set(numeric):
        raise ValueError("snapshot categorical and numeric columns must partition features")
    if set(features) & set(protected):
        raise ValueError("snapshot protected columns cannot be model features")
    role_columns = (
        list(features) + list(protected) + [contract["score_column"], contract["prediction_column"]]
    )
    if contract["label_column"] is not None:
        role_columns.append(contract["label_column"])
    if contract["sample_weight_column"] is not None:
        role_columns.append(contract["sample_weight_column"])
    if len(role_columns) != len(set(role_columns)) or set(role_columns) != set(columns):
        raise ValueError("snapshot column roles must be distinct and exactly cover columns")

    numeric_names = contract["numeric_columns"]
    categorical_names = contract["categorical_columns"]
    protected_names = contract["protected_columns"]
    distributions = snapshot["distributions"]
    if not isinstance(distributions, Mapping) or set(distributions) != {"numeric", "categorical"}:
        raise ValueError("snapshot distributions do not match the exact schema")
    if set(distributions["numeric"]) != set(numeric_names):
        raise ValueError("snapshot numeric distributions do not match the contract")
    if set(distributions["categorical"]) != set(categorical_names):
        raise ValueError("snapshot categorical distributions do not match the contract")
    for summary in distributions["numeric"].values():
        _validate_numeric_summary(summary, int(snapshot["row_count"]))
    for summary in distributions["categorical"].values():
        _validate_categorical_summary(summary, int(snapshot["row_count"]))

    outcomes = snapshot["outcomes"]
    if not isinstance(outcomes, Mapping) or set(outcomes) != {"score", "prediction"}:
        raise ValueError("snapshot outcomes do not match the exact schema")
    _validate_numeric_summary(outcomes["score"], int(snapshot["row_count"]))
    _validate_prediction_summary(outcomes["prediction"], int(snapshot["row_count"]))

    protected_audit = snapshot["protected_audit"]
    if (
        not isinstance(protected_audit, Mapping)
        or set(protected_audit) != {"row_level_data_included", "columns"}
        or protected_audit["row_level_data_included"] is not False
        or set(protected_audit["columns"]) != set(protected_names)
    ):
        raise ValueError("snapshot protected audit does not match the aggregate-only schema")
    for summary in protected_audit["columns"].values():
        _validate_categorical_summary(summary, int(snapshot["row_count"]))

    _validate_delayed_labels(
        snapshot["delayed_labels"],
        contract,
        int(snapshot["row_count"]),
        prediction_summary=outcomes["prediction"],
        protected_summaries=protected_audit["columns"],
    )
    _validate_weight_audit(snapshot["weight_audit"], contract, int(snapshot["row_count"]))

    evidence = snapshot["evidence"]
    expected_evidence = {
        "snapshot_rows",
        "labels_available",
        "protected_attributes_configured",
        "contains_only_aggregate_statistics",
    }
    if (
        not isinstance(evidence, Mapping)
        or set(evidence) != expected_evidence
        or evidence.get("contains_only_aggregate_statistics") is not True
        or evidence.get("snapshot_rows") != snapshot["row_count"]
        or evidence.get("labels_available")
        is not (snapshot["delayed_labels"]["status"] == "available")
        or evidence.get("protected_attributes_configured") is not bool(protected_names)
    ):
        raise ValueError("snapshot evidence metadata is invalid")
    _ensure_json_ready(snapshot, "snapshot")


def _validate_numeric_summary(summary: Mapping[str, Any], expected_count: int) -> None:
    expected = {
        "count",
        "mean",
        "standard_deviation",
        "minimum",
        "maximum",
        "quantile_probabilities",
        "quantile_values",
    }
    if not isinstance(summary, Mapping) or set(summary) != expected:
        raise ValueError("numeric summary does not match the exact schema")
    if summary["count"] != expected_count:
        raise ValueError("numeric summary count does not match snapshot row_count")
    probabilities = np.asarray(summary["quantile_probabilities"], dtype=float)
    values = np.asarray(summary["quantile_values"], dtype=float)
    if len(probabilities) != 101 or len(values) != 101:
        raise ValueError("numeric summary must contain a 101-point quantile sketch")
    if not np.isfinite(probabilities).all() or not np.isfinite(values).all():
        raise ValueError("numeric summary must contain only finite values")
    if not np.allclose(probabilities, _QUANTILE_PROBABILITIES):
        raise ValueError("numeric summary quantile probabilities are invalid")
    if np.any(np.diff(values) < 0):
        raise ValueError("numeric summary quantile values must be sorted")
    scalars = [
        summary["mean"],
        summary["standard_deviation"],
        summary["minimum"],
        summary["maximum"],
    ]
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(float(value))
        for value in scalars
    ):
        raise ValueError("numeric summary scalars must be finite")


def _validate_categorical_summary(summary: Mapping[str, Any], expected_count: int) -> None:
    expected = {"count", "cardinality", "counts", "shares", "unknown_count", "unknown_share"}
    if not isinstance(summary, Mapping) or set(summary) != expected:
        raise ValueError("categorical summary does not match the exact schema")
    if summary["count"] != expected_count:
        raise ValueError("categorical summary count does not match snapshot row_count")
    counts = summary["counts"]
    shares = summary["shares"]
    if not isinstance(counts, Mapping) or not isinstance(shares, Mapping):
        raise ValueError("categorical counts and shares must be mappings")
    cardinality = summary["cardinality"]
    if (
        set(counts) != set(shares)
        or not isinstance(cardinality, int)
        or isinstance(cardinality, bool)
        or cardinality != len(counts)
    ):
        raise ValueError("categorical counts and shares must have matching categories")
    if not all(isinstance(key, str) for key in counts):
        raise ValueError("categorical category names must be strings")
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in counts.values()
    ):
        raise ValueError("categorical counts must be non-negative integers")
    if sum(counts.values()) != expected_count:
        raise ValueError("categorical counts must sum to snapshot row_count")
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(float(value))
        and 0 <= float(value) <= 1
        for value in shares.values()
    ):
        raise ValueError("categorical shares must be finite and between zero and one")
    if not _derived_values_match(sum(float(value) for value in shares.values()), 1.0):
        raise ValueError("categorical shares must sum to one")
    for category, count in counts.items():
        if not _derived_values_match(float(shares[category]), count / expected_count):
            raise ValueError("categorical shares must match category counts")
    expected_unknown_count = sum(
        count
        for category, count in counts.items()
        if category.strip().casefold() in _UNKNOWN_TOKENS
    )
    if (
        not isinstance(summary["unknown_count"], int)
        or isinstance(summary["unknown_count"], bool)
        or summary["unknown_count"] != expected_unknown_count
    ):
        raise ValueError("categorical unknown_count must match unknown category counts")
    unknown_share = summary["unknown_share"]
    if (
        not isinstance(unknown_share, (int, float))
        or isinstance(unknown_share, bool)
        or not np.isfinite(float(unknown_share))
        or not 0 <= float(unknown_share) <= 1
    ):
        raise ValueError("categorical unknown_share must be between zero and one")
    if not _derived_values_match(
        float(unknown_share),
        expected_unknown_count / expected_count,
    ):
        raise ValueError("categorical unknown_share must match unknown category counts")


def _validate_prediction_summary(summary: Mapping[str, Any], expected_count: int) -> None:
    expected = {"count", "negative_count", "positive_count", "selection_rate"}
    if not isinstance(summary, Mapping) or set(summary) != expected:
        raise ValueError("prediction summary does not match the exact schema")
    for name in ("count", "negative_count", "positive_count"):
        value = summary[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("prediction summary counts must be non-negative integers")
    if summary["count"] != expected_count:
        raise ValueError("prediction count does not match snapshot row_count")
    if summary["negative_count"] + summary["positive_count"] != expected_count:
        raise ValueError("prediction class counts must sum to snapshot row_count")
    rate = summary["selection_rate"]
    if (
        not isinstance(rate, (int, float))
        or isinstance(rate, bool)
        or not np.isfinite(float(rate))
        or not 0 <= float(rate) <= 1
        or not _derived_values_match(float(rate), summary["positive_count"] / expected_count)
    ):
        raise ValueError("prediction selection_rate must match positive_count")


def _validate_delayed_labels(
    delayed: Mapping[str, Any],
    contract: Mapping[str, Any],
    expected_count: int,
    *,
    prediction_summary: Mapping[str, Any],
    protected_summaries: Mapping[str, Any],
) -> None:
    expected = {
        "status",
        "column",
        "class_counts",
        "performance",
        "protected_group_metrics",
    }
    if not isinstance(delayed, Mapping) or set(delayed) != expected:
        raise ValueError("delayed-label audit does not match the exact schema")
    status = delayed["status"]
    if status == "unavailable":
        if (
            contract["label_column"] is not None
            or delayed["column"] is not None
            or delayed["class_counts"] is not None
            or delayed["performance"] is not None
            or delayed["protected_group_metrics"] != {}
        ):
            raise ValueError("unavailable delayed-label audit contains label evidence")
        return
    if (
        status != "available"
        or contract["label_column"] is None
        or delayed["column"] != contract["label_column"]
    ):
        raise ValueError("available delayed-label audit must match the label contract")
    counts = delayed["class_counts"]
    if not isinstance(counts, Mapping) or set(counts) != {"negative", "positive"}:
        raise ValueError("delayed-label class_counts do not match the exact schema")
    if (
        not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in counts.values()
        )
        or sum(counts.values()) != expected_count
    ):
        raise ValueError("delayed-label class_counts must sum to snapshot row_count")
    performance = delayed["performance"]
    _validate_binary_metrics(performance, expected_count, require_unit_weights=True)
    if not _derived_values_match(
        counts["positive"], performance["positive_weight"]
    ) or not _derived_values_match(counts["negative"], performance["negative_weight"]):
        raise ValueError("delayed-label class_counts must match performance class totals")
    predicted_positive = float(performance["true_positive"]) + float(performance["false_positive"])
    predicted_negative = float(performance["true_negative"]) + float(performance["false_negative"])
    if not _derived_values_match(
        prediction_summary["positive_count"], predicted_positive
    ) or not _derived_values_match(prediction_summary["negative_count"], predicted_negative):
        raise ValueError("prediction counts must match delayed-label confusion totals")

    protected_metrics = delayed["protected_group_metrics"]
    if not isinstance(protected_metrics, Mapping) or set(protected_metrics) != set(
        contract["protected_columns"]
    ):
        raise ValueError("delayed-label group metrics do not match protected columns")
    aggregate_names = (
        "total_weight",
        "positive_weight",
        "negative_weight",
        "true_positive",
        "false_negative",
        "false_positive",
        "true_negative",
    )
    for protected_name, audit in protected_metrics.items():
        if (
            not isinstance(audit, Mapping)
            or set(audit) != {"groups", "row_level_data_included"}
            or audit["row_level_data_included"] is not False
            or not isinstance(audit["groups"], Mapping)
        ):
            raise ValueError("protected group metrics are not aggregate-only")
        protected_summary = protected_summaries[protected_name]
        if set(audit["groups"]) != set(protected_summary["counts"]):
            raise ValueError("protected group metrics must match protected category names")
        group_total = 0
        aggregate = dict.fromkeys(aggregate_names, 0.0)
        for group, metrics in audit["groups"].items():
            if not isinstance(group, str):
                raise ValueError("protected group names must be strings")
            if metrics.get("count") != protected_summary["counts"][group]:
                raise ValueError("protected group metric counts must match protected counts")
            _validate_binary_metrics(
                metrics,
                int(metrics.get("count", -1)),
                require_unit_weights=True,
            )
            group_total += int(metrics["count"])
            for name in aggregate_names:
                aggregate[name] += float(metrics[name])
        if group_total != expected_count:
            raise ValueError("protected group metric counts must sum to snapshot row_count")
        if any(
            not _derived_values_match(aggregate[name], performance[name])
            for name in aggregate_names
        ):
            raise ValueError("protected group confusion totals must match overall performance")


def _validate_binary_metrics(
    metrics: Mapping[str, Any],
    expected_count: int,
    *,
    require_unit_weights: bool = False,
) -> None:
    expected = {
        "count",
        "total_weight",
        "positive_weight",
        "negative_weight",
        "true_positive",
        "false_negative",
        "false_positive",
        "true_negative",
        "accuracy",
        "true_positive_rate",
        "false_positive_rate",
        "precision",
        "base_rate",
        "selection_rate",
    }
    if not isinstance(metrics, Mapping) or set(metrics) != expected:
        raise ValueError("binary metrics do not match the exact schema")
    if (
        not isinstance(metrics["count"], int)
        or isinstance(metrics["count"], bool)
        or metrics["count"] != expected_count
        or expected_count < 1
    ):
        raise ValueError("binary metric count is invalid")
    weight_names = (
        "total_weight",
        "positive_weight",
        "negative_weight",
        "true_positive",
        "false_negative",
        "false_positive",
        "true_negative",
    )
    for name in weight_names:
        value = metrics[name]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError("binary metric weights must be finite and non-negative")
    if float(metrics["total_weight"]) <= 0:
        raise ValueError("binary metric total_weight must be positive")
    confusion_total = sum(
        float(metrics[name])
        for name in ("true_positive", "false_negative", "false_positive", "true_negative")
    )
    total = float(metrics["total_weight"])
    positive = float(metrics["positive_weight"])
    negative = float(metrics["negative_weight"])
    true_positive = float(metrics["true_positive"])
    false_negative = float(metrics["false_negative"])
    false_positive = float(metrics["false_positive"])
    true_negative = float(metrics["true_negative"])
    if not _derived_values_match(confusion_total, total):
        raise ValueError("binary confusion weights must sum to total_weight")
    if not _derived_values_match(positive, true_positive + false_negative):
        raise ValueError("binary positive_weight must match positive confusion weights")
    if not _derived_values_match(negative, false_positive + true_negative):
        raise ValueError("binary negative_weight must match negative confusion weights")
    if not _derived_values_match(positive + negative, total):
        raise ValueError("binary positive and negative weights must sum to total_weight")
    if require_unit_weights:
        unweighted_values = (
            total,
            positive,
            negative,
            true_positive,
            false_negative,
            false_positive,
            true_negative,
        )
        if not _derived_values_match(total, expected_count) or any(
            not _derived_values_match(value, round(value)) for value in unweighted_values
        ):
            raise ValueError("unweighted binary confusion values must be integer row counts")

    expected_rates = {
        "accuracy": (true_positive + true_negative) / total,
        "true_positive_rate": _safe_ratio(true_positive, positive),
        "false_positive_rate": _safe_ratio(false_positive, negative),
        "precision": _safe_ratio(true_positive, true_positive + false_positive),
        "base_rate": positive / total,
        "selection_rate": (true_positive + false_positive) / total,
    }
    for name in (
        "accuracy",
        "true_positive_rate",
        "false_positive_rate",
        "precision",
        "base_rate",
        "selection_rate",
    ):
        value = metrics[name]
        if value is not None and (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(float(value))
            or not 0 <= float(value) <= 1
        ):
            raise ValueError("binary metric rates must be null or between zero and one")
        expected_rate = expected_rates[name]
        if expected_rate is None:
            if value is not None:
                raise ValueError(f"binary metric {name} must be null when its denominator is zero")
        elif value is None or not _derived_values_match(value, expected_rate):
            raise ValueError(f"binary metric {name} must match confusion weights")


def _validate_weight_audit(
    audit: Mapping[str, Any], contract: Mapping[str, Any], expected_count: int
) -> None:
    expected = {
        "status",
        "column",
        "role",
        "used_as_model_feature",
        "used_for_primary_metrics_or_gate",
        "summary",
        "weighted_sensitivity",
    }
    if not isinstance(audit, Mapping) or set(audit) != expected:
        raise ValueError("weight audit does not match the exact schema")
    if (
        audit["role"] != "audit_only"
        or audit["used_as_model_feature"] is not False
        or audit["used_for_primary_metrics_or_gate"] is not False
    ):
        raise ValueError("sample weights must remain audit-only")
    if audit["status"] == "not_provided":
        if (
            contract["sample_weight_column"] is not None
            or audit["column"] is not None
            or audit["summary"] is not None
            or audit["weighted_sensitivity"] is not None
        ):
            raise ValueError("missing weight audit contains weight evidence")
        return
    if audit["status"] != "provided" or audit["column"] != contract["sample_weight_column"]:
        raise ValueError("provided weight audit must match the weight contract")
    summary = audit["summary"]
    expected_summary = {
        "count",
        "sum",
        "mean",
        "minimum",
        "maximum",
        "kish_effective_sample_size",
    }
    if not isinstance(summary, Mapping) or set(summary) != expected_summary:
        raise ValueError("weight summary does not match the exact schema")
    if summary["count"] != expected_count:
        raise ValueError("weight count must match snapshot row_count")
    for name in expected_summary - {"count"}:
        value = summary[name]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError("weight summary values must be finite and non-negative")
    sensitivity = audit["weighted_sensitivity"]
    expected_sensitivity = {
        "score_mean",
        "selection_rate",
        "performance",
        "protected_group_composition",
    }
    if not isinstance(sensitivity, Mapping) or set(sensitivity) != expected_sensitivity:
        raise ValueError("weighted sensitivity does not match the exact schema")
    for name in ("score_mean", "selection_rate"):
        value = sensitivity[name]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(float(value))
            or not 0 <= float(value) <= 1
        ):
            raise ValueError("weighted outcome rates must be between zero and one")
    if sensitivity["performance"] is not None:
        _validate_binary_metrics(sensitivity["performance"], expected_count)
    composition = sensitivity["protected_group_composition"]
    if not isinstance(composition, Mapping) or set(composition) != set(
        contract["protected_columns"]
    ):
        raise ValueError("weighted group composition does not match protected columns")
    for shares in composition.values():
        if not isinstance(shares, Mapping) or not np.isclose(
            sum(float(value) for value in shares.values()), 1.0
        ):
            raise ValueError("weighted group composition shares must sum to one")


def _ensure_json_ready(value: Mapping[str, Any], name: str) -> None:
    try:
        json.dumps(value, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON-ready and contain no non-finite values") from exc


def _derived_values_match(left: int | float, right: int | float) -> bool:
    return bool(
        np.isclose(
            float(left),
            float(right),
            rtol=_DERIVED_VALUE_RTOL,
            atol=_DERIVED_VALUE_ATOL,
        )
    )
