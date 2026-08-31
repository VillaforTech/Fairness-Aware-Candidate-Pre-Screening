"""Aggregate comparable evaluation reports into a repeated-seed stability study.

The study deliberately treats repeated runs as a sensitivity analysis. Every run
reuses the same official test partition, so the resulting distributions are not
independent estimates of population uncertainty.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

from fairness_project.governance.gate import GateThresholds

STABILITY_SCHEMA_VERSION = "1.0"
SUPPORTED_METRICS = ("accuracy", "SPD", "DI", "TPR_gap", "FPR_gap")
_PARITY_TARGETS = {"SPD": 0.0, "DI": 1.0, "TPR_gap": 0.0, "FPR_gap": 0.0}
_SHA256_LENGTH = 64
_LIMITATION = (
    "All runs reuse the same official test partition. This study measures sensitivity "
    "to the training/validation split, model fitting, and policy selection across seeds; "
    "it does not estimate independent population uncertainty."
)


class StabilityStudyError(ValueError):
    """Raised when reports cannot form one comparable stability study."""


@dataclass(frozen=True)
class _ValidatedRun:
    seed: int
    schema_version: str
    model_type: str
    model_parameters_json: str
    resolved_config_json: str
    resolved_config: dict[str, Any]
    data_sha256: str
    source_sha256: str
    protocol_json: str
    protocol: dict[str, Any]
    feature_contract_id: str
    feature_columns: tuple[str, ...]
    gate_passed: bool
    gate_thresholds_json: str
    gate_thresholds: dict[str, float]
    metrics: dict[str, dict[str, float]]
    thresholds: dict[str, float]


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StabilityStudyError(f"{path} must be an object")
    return value


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StabilityStudyError(f"{path} must be a nonempty string")
    return value


def _sha256(value: Any, path: str) -> str:
    digest = _nonempty_string(value, path)
    if len(digest) != _SHA256_LENGTH or any(
        character not in "0123456789abcdefABCDEF" for character in digest
    ):
        raise StabilityStudyError(f"{path} must be a 64-hex SHA-256 digest")
    return digest.lower()


def _canonical_json(value: Any, path: str) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise StabilityStudyError(
            f"{path} must be JSON-serializable without non-finite values"
        ) from exc


def _json_object_copy(value: Mapping[str, Any], path: str) -> dict[str, Any]:
    serialized = _canonical_json(value, path)
    copy = json.loads(serialized)
    if not isinstance(copy, dict):  # pragma: no cover - guaranteed by Mapping input
        raise StabilityStudyError(f"{path} must be an object")
    return copy


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise StabilityStudyError(f"{path} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise StabilityStudyError(f"{path} must be a finite number")
    return number


def _metric(value: Any, name: str, path: str) -> float:
    number = _finite_number(value, path)
    if name == "accuracy" and not 0.0 <= number <= 1.0:
        raise StabilityStudyError(f"{path} must be between 0 and 1")
    if name in {"SPD", "TPR_gap", "FPR_gap"} and not -1.0 <= number <= 1.0:
        raise StabilityStudyError(f"{path} must be between -1 and 1")
    if name == "DI" and number < 0.0:
        raise StabilityStudyError(f"{path} must be nonnegative")
    return number


def _extract_metrics(value: Any, path: str) -> dict[str, float]:
    metrics = _object(value, path)
    selected = {
        name: _metric(metrics[name], name, f"{path}.{name}")
        for name in SUPPORTED_METRICS
        if name in metrics
    }
    if "accuracy" not in selected:
        raise StabilityStudyError(f"{path}.accuracy is required")
    if not any(name in selected for name in _PARITY_TARGETS):
        raise StabilityStudyError(f"{path} must contain at least one supported fairness metric")
    return selected


def _extract_feature_contract(
    protocol: Mapping[str, Any], report_index: int
) -> tuple[str, tuple[str, ...]]:
    prefix = f"reports[{report_index}].protocol"
    contract_id = _nonempty_string(
        protocol.get("feature_contract_id"), f"{prefix}.feature_contract_id"
    )
    raw_columns = protocol.get("feature_columns")
    if not isinstance(raw_columns, list) or not raw_columns:
        raise StabilityStudyError(f"{prefix}.feature_columns must be a nonempty array")
    if any(not isinstance(column, str) or not column for column in raw_columns):
        raise StabilityStudyError(f"{prefix}.feature_columns must contain nonempty strings")
    if len(set(raw_columns)) != len(raw_columns):
        raise StabilityStudyError(f"{prefix}.feature_columns must not contain duplicates")
    return contract_id, tuple(raw_columns)


def _comparable_resolved_config(metadata: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    """Remove only the per-run seed fields from a resolved configuration."""
    payload = _object(metadata.get("resolved_config"), f"{prefix}.metadata.resolved_config")
    comparable = _json_object_copy(payload, f"{prefix}.metadata.resolved_config")
    if "seed" not in comparable:
        raise StabilityStudyError(f"{prefix}.metadata.resolved_config.seed is required")
    comparable.pop("seed")
    model = comparable.get("model")
    if not isinstance(model, dict):
        raise StabilityStudyError(f"{prefix}.metadata.resolved_config.model must be an object")
    if "random_state" not in model:
        raise StabilityStudyError(
            f"{prefix}.metadata.resolved_config.model.random_state is required"
        )
    model.pop("random_state")
    return comparable


def _gate_threshold_policy(governance: Mapping[str, Any], prefix: str) -> dict[str, float]:
    payload = _object(governance.get("thresholds"), f"{prefix}.governance.thresholds")
    expected = set(GateThresholds().to_dict())
    observed = set(payload)
    if observed != expected:
        raise StabilityStudyError(
            f"{prefix}.governance.thresholds must contain the exact gate policy fields"
        )
    try:
        return GateThresholds(**dict(payload)).to_dict()
    except (TypeError, ValueError) as exc:
        raise StabilityStudyError(f"{prefix}.governance.thresholds is invalid: {exc}") from exc


def _validate_report(report: Any, report_index: int) -> _ValidatedRun:
    prefix = f"reports[{report_index}]"
    payload = _object(report, prefix)
    schema_version = _nonempty_string(payload.get("schema_version"), f"{prefix}.schema_version")

    metadata = _object(payload.get("metadata"), f"{prefix}.metadata")
    seed_value = metadata.get("seed")
    if isinstance(seed_value, bool) or not isinstance(seed_value, Integral) or seed_value < 0:
        raise StabilityStudyError(f"{prefix}.metadata.seed must be a nonnegative integer")
    seed = int(seed_value)
    model_type = _nonempty_string(metadata.get("model_type"), f"{prefix}.metadata.model_type")
    model_parameters = _object(
        metadata.get("model_parameters"), f"{prefix}.metadata.model_parameters"
    )
    model_parameters_json = _canonical_json(model_parameters, f"{prefix}.metadata.model_parameters")
    resolved_config = _comparable_resolved_config(metadata, prefix)
    resolved_config_json = _canonical_json(resolved_config, f"{prefix}.metadata.resolved_config")
    data_sha256 = _sha256(metadata.get("data_sha256"), f"{prefix}.metadata.data_sha256")
    source_sha256 = _sha256(metadata.get("source_sha256"), f"{prefix}.metadata.source_sha256")

    protocol_mapping = _object(payload.get("protocol"), f"{prefix}.protocol")
    contract_id, feature_columns = _extract_feature_contract(protocol_mapping, report_index)
    protocol = _json_object_copy(protocol_mapping, f"{prefix}.protocol")
    comparable_protocol = {
        key: value
        for key, value in protocol.items()
        if key not in {"split_counts", "split_cell_counts"}
    }
    protocol_json = _canonical_json(comparable_protocol, f"{prefix}.protocol")

    results = _object(payload.get("results"), f"{prefix}.results")
    metrics = {
        "baseline": _extract_metrics(
            results.get("baseline_metrics"), f"{prefix}.results.baseline_metrics"
        ),
        "adjusted": _extract_metrics(results.get("metrics"), f"{prefix}.results.metrics"),
    }
    threshold_payload = _object(results.get("thresholds"), f"{prefix}.results.thresholds")
    thresholds: dict[str, float] = {}
    for group in ("privileged", "unprivileged"):
        threshold = _finite_number(
            threshold_payload.get(group), f"{prefix}.results.thresholds.{group}"
        )
        if not 0.0 <= threshold <= 1.0:
            raise StabilityStudyError(
                f"{prefix}.results.thresholds.{group} must be between 0 and 1"
            )
        thresholds[group] = threshold

    governance = _object(payload.get("governance"), f"{prefix}.governance")
    gate_passed = governance.get("passed")
    if not isinstance(gate_passed, bool):
        raise StabilityStudyError(f"{prefix}.governance.passed must be Boolean")
    if governance.get("report_valid") is not True:
        raise StabilityStudyError(
            f"{prefix}.governance.report_valid must be true for a completed valid run"
        )
    gate_thresholds = _gate_threshold_policy(governance, prefix)
    gate_thresholds_json = _canonical_json(gate_thresholds, f"{prefix}.governance.thresholds")

    return _ValidatedRun(
        seed=seed,
        schema_version=schema_version,
        model_type=model_type,
        model_parameters_json=model_parameters_json,
        resolved_config_json=resolved_config_json,
        resolved_config=resolved_config,
        data_sha256=data_sha256,
        source_sha256=source_sha256,
        protocol_json=protocol_json,
        protocol=protocol,
        feature_contract_id=contract_id,
        feature_columns=feature_columns,
        gate_passed=gate_passed,
        gate_thresholds_json=gate_thresholds_json,
        gate_thresholds=gate_thresholds,
        metrics=metrics,
        thresholds=thresholds,
    )


def _require_comparable(runs: Sequence[_ValidatedRun]) -> None:
    reference = runs[0]
    comparisons = (
        ("report schema", "schema_version"),
        ("model type", "model_type"),
        ("model parameters", "model_parameters_json"),
        ("resolved policy configuration", "resolved_config_json"),
        ("data", "data_sha256"),
        ("source", "source_sha256"),
        ("feature contract", "feature_contract_id"),
        ("feature columns", "feature_columns"),
        ("protocol", "protocol_json"),
        ("gate threshold policy", "gate_thresholds_json"),
    )
    for run in runs[1:]:
        for label, attribute in comparisons:
            if getattr(run, attribute) != getattr(reference, attribute):
                raise StabilityStudyError(
                    f"seed {run.seed} is incomparable with seed {reference.seed}: {label} differs"
                )

    for condition in ("baseline", "adjusted"):
        expected = set(reference.metrics[condition])
        for run in runs[1:]:
            observed = set(run.metrics[condition])
            if observed != expected:
                raise StabilityStudyError(
                    f"seed {run.seed} is incomparable with seed {reference.seed}: "
                    f"{condition} metric coverage differs"
                )


def _quantile(values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated quantile, matching NumPy's default method."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


def _distribution(values: Sequence[float]) -> dict[str, float | int | list[float]]:
    if not values:  # pragma: no cover - callers always aggregate validated metrics
        raise StabilityStudyError("cannot summarize an empty distribution")
    return {
        "count": len(values),
        "values": [float(value) for value in values],
        "min": float(min(values)),
        "q25": _quantile(values, 0.25),
        "median": _quantile(values, 0.5),
        "q75": _quantile(values, 0.75),
        "max": float(max(values)),
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)),
    }


def _worst_gaps(runs: Sequence[_ValidatedRun], condition: str) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    available_metrics = runs[0].metrics[condition]
    for metric, parity_target in _PARITY_TARGETS.items():
        if metric not in available_metrics:
            continue
        candidates = [
            (
                abs(run.metrics[condition][metric] - parity_target),
                run.seed,
                run.metrics[condition][metric],
            )
            for run in runs
        ]
        distance, seed, value = max(candidates, key=lambda item: (item[0], -item[1]))
        output[metric] = {
            "seed": seed,
            "value": value,
            "parity_target": parity_target,
            "distance_from_parity": distance,
        }
    return output


def _max_parity_deviation(run: _ValidatedRun, condition: str) -> float:
    deviations = [
        abs(run.metrics[condition][metric] - target)
        for metric, target in _PARITY_TARGETS.items()
        if metric in run.metrics[condition]
    ]
    return max(deviations)


def _worst_seed_ranking(runs: Sequence[_ValidatedRun]) -> list[dict[str, float | int | bool]]:
    ranked = sorted(
        runs,
        key=lambda run: (
            -int(not run.gate_passed),
            -_max_parity_deviation(run, "adjusted"),
            run.metrics["adjusted"]["accuracy"],
            run.seed,
        ),
    )
    return [
        {
            "rank": rank,
            "seed": run.seed,
            "gate_passed": run.gate_passed,
            "adjusted_max_parity_deviation": _max_parity_deviation(run, "adjusted"),
            "adjusted_accuracy": run.metrics["adjusted"]["accuracy"],
        }
        for rank, run in enumerate(ranked, start=1)
    ]


def summarize_stability(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build a JSON-serializable repeated-seed stability summary.

    Reports must be completed, valid run reports with unique seeds and identical
    report schema, resolved policy configuration apart from seed, gate policy,
    model configuration, input-data digest, source digest, evaluation protocol,
    and feature contract. Any malformed or incomparable report raises
    :class:`StabilityStudyError` instead of producing a partial summary.
    """
    if isinstance(reports, (str, bytes)) or not isinstance(reports, Sequence):
        raise StabilityStudyError("reports must be a sequence of report objects")
    if len(reports) < 2:
        raise StabilityStudyError("stability study requires at least two reports")

    runs = [_validate_report(report, index) for index, report in enumerate(reports)]
    seeds = [run.seed for run in runs]
    if len(set(seeds)) != len(seeds):
        duplicates = sorted(seed for seed in set(seeds) if seeds.count(seed) > 1)
        raise StabilityStudyError(f"report seeds must be unique; duplicates: {duplicates}")

    runs = sorted(runs, key=lambda run: run.seed)
    _require_comparable(runs)
    reference = runs[0]

    conditions: dict[str, Any] = {}
    for condition in ("baseline", "adjusted"):
        conditions[condition] = {
            "metrics": {
                metric: _distribution([run.metrics[condition][metric] for run in runs])
                for metric in SUPPORTED_METRICS
                if metric in reference.metrics[condition]
            },
            "worst_absolute_gaps": _worst_gaps(runs, condition),
        }

    pass_count = sum(run.gate_passed for run in runs)
    ranking = _worst_seed_ranking(runs)
    model_parameters = json.loads(reference.model_parameters_json)
    return {
        "schema_version": STABILITY_SCHEMA_VERSION,
        "study_type": "repeated-seed-stability",
        "report_count": len(runs),
        "seeds": [run.seed for run in runs],
        "comparability": {
            "report_schema_version": reference.schema_version,
            "model_type": reference.model_type,
            "model_parameters": model_parameters,
            "resolved_config": reference.resolved_config,
            "data_sha256": reference.data_sha256,
            "source_sha256": reference.source_sha256,
            "protocol": reference.protocol,
            "feature_contract": {
                "id": reference.feature_contract_id,
                "columns": list(reference.feature_columns),
            },
            "gate_thresholds": reference.gate_thresholds,
        },
        "governance": {
            "pass_count": pass_count,
            "fail_count": len(runs) - pass_count,
            "pass_rate": pass_count / len(runs),
        },
        "conditions": conditions,
        "thresholds": {
            group: _distribution([run.thresholds[group] for run in runs])
            for group in ("privileged", "unprivileged")
        },
        "worst_seed": ranking[0]["seed"],
        "worst_seed_ranking": ranking,
        "ranking_method": (
            "Governance failures rank before passes, then larger adjusted parity deviation, "
            "lower adjusted accuracy, and finally lower seed."
        ),
        "distribution_semantics": (
            "Quartiles use linear interpolation and std is the population standard deviation "
            "across the supplied seeds."
        ),
        "limitations": [_LIMITATION],
    }
