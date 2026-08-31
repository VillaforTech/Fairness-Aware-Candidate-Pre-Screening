"""Tests for repeated-seed stability aggregation."""

from __future__ import annotations

import copy
import json
import math
from typing import Any

import pytest

from fairness_project.evaluation.stability import StabilityStudyError, summarize_stability
from fairness_project.governance.gate import GateThresholds


def _report(
    seed: int,
    *,
    passed: bool = True,
    baseline_accuracy: float = 0.86,
    adjusted_accuracy: float = 0.85,
    baseline_spd: float = 0.20,
    adjusted_spd: float = 0.08,
    baseline_di: float = 0.50,
    adjusted_di: float = 0.82,
    baseline_tpr_gap: float = 0.10,
    adjusted_tpr_gap: float = 0.03,
    baseline_fpr_gap: float = 0.12,
    adjusted_fpr_gap: float = 0.05,
    privileged_threshold: float = 0.5,
    unprivileged_threshold: float = 0.4,
) -> dict[str, Any]:
    digest = "a" * 64
    source_digest = "b" * 64
    return {
        "schema_version": "1.0",
        "metadata": {
            "seed": seed,
            "model_type": "xgb",
            "model_parameters": {"max_depth": 4, "n_estimators": 300},
            "resolved_config": {
                "schema_version": "2.0",
                "seed": seed,
                "model": {"random_state": seed, "max_depth": 4},
                "fairness": {"frontier_max_abs_tpr_gap": 0.05},
            },
            "data_sha256": digest,
            "source_sha256": source_digest,
        },
        "protocol": {
            "dataset": "UCI Adult",
            "official_test_partition_preserved": True,
            "validation_strategy": "joint stratification",
            "feature_contract_id": "adult-income-v2",
            "feature_columns": ["age", "education", "hours_per_week"],
        },
        "results": {
            "baseline_metrics": {
                "accuracy": baseline_accuracy,
                "SPD": baseline_spd,
                "DI": baseline_di,
                "TPR_gap": baseline_tpr_gap,
                "FPR_gap": baseline_fpr_gap,
            },
            "metrics": {
                "accuracy": adjusted_accuracy,
                "SPD": adjusted_spd,
                "DI": adjusted_di,
                "TPR_gap": adjusted_tpr_gap,
                "FPR_gap": adjusted_fpr_gap,
            },
            "thresholds": {
                "privileged": privileged_threshold,
                "unprivileged": unprivileged_threshold,
            },
        },
        "governance": {
            "passed": passed,
            "report_valid": True,
            "thresholds": GateThresholds().to_dict(),
        },
    }


def test_summarizes_runs_in_seed_order_and_is_json_serializable() -> None:
    reports = [
        _report(
            11,
            passed=True,
            adjusted_accuracy=0.84,
            adjusted_spd=0.05,
            unprivileged_threshold=0.38,
        ),
        _report(
            3,
            passed=False,
            adjusted_accuracy=0.82,
            adjusted_spd=0.15,
            unprivileged_threshold=0.42,
        ),
        _report(
            7,
            passed=True,
            adjusted_accuracy=0.86,
            adjusted_spd=0.10,
            unprivileged_threshold=0.40,
        ),
    ]

    summary = summarize_stability(reports)

    assert summary["seeds"] == [3, 7, 11]
    assert summary["report_count"] == 3
    assert summary["governance"] == {"pass_count": 2, "fail_count": 1, "pass_rate": 2 / 3}
    assert summary["comparability"]["feature_contract"] == {
        "id": "adult-income-v2",
        "columns": ["age", "education", "hours_per_week"],
    }
    assert summary["conditions"]["adjusted"]["metrics"]["accuracy"]["mean"] == pytest.approx(0.84)
    assert summary["thresholds"]["unprivileged"]["median"] == pytest.approx(0.40)
    assert json.loads(json.dumps(summary)) == summary


def test_distribution_uses_linear_quartiles_and_population_std() -> None:
    reports = [
        _report(seed, adjusted_accuracy=accuracy)
        for seed, accuracy in zip((1, 2, 3, 4), (0.70, 0.80, 0.90, 1.00), strict=True)
    ]

    distribution = summarize_stability(reports)["conditions"]["adjusted"]["metrics"]["accuracy"]

    assert distribution == pytest.approx(
        {
            "count": 4,
            "values": [0.70, 0.80, 0.90, 1.00],
            "min": 0.70,
            "q25": 0.775,
            "median": 0.85,
            "q75": 0.925,
            "max": 1.00,
            "mean": 0.85,
            "std": math.sqrt(0.0125),
        }
    )


def test_worst_gap_uses_parity_target_and_lower_seed_to_break_tie() -> None:
    reports = [
        _report(9, adjusted_spd=-0.20, adjusted_di=1.25),
        _report(4, adjusted_spd=0.20, adjusted_di=0.75),
    ]

    gaps = summarize_stability(reports)["conditions"]["adjusted"]["worst_absolute_gaps"]

    assert gaps["SPD"] == {
        "seed": 4,
        "value": 0.20,
        "parity_target": 0.0,
        "distance_from_parity": 0.20,
    }
    assert gaps["DI"] == {
        "seed": 4,
        "value": 0.75,
        "parity_target": 1.0,
        "distance_from_parity": 0.25,
    }


def test_worst_seed_ranking_is_deterministic_and_governance_first() -> None:
    reports = [
        _report(9, passed=True, adjusted_accuracy=0.70, adjusted_spd=0.40),
        _report(8, passed=False, adjusted_accuracy=0.90, adjusted_spd=0.10),
        _report(2, passed=False, adjusted_accuracy=0.90, adjusted_spd=-0.10),
    ]

    summary = summarize_stability(reports)

    assert summary["worst_seed"] == 2
    assert [entry["seed"] for entry in summary["worst_seed_ranking"]] == [2, 8, 9]
    assert [entry["rank"] for entry in summary["worst_seed_ranking"]] == [1, 2, 3]


def test_consistently_absent_optional_metric_is_omitted() -> None:
    reports = [_report(1), _report(2)]
    for report in reports:
        del report["results"]["baseline_metrics"]["FPR_gap"]
        del report["results"]["metrics"]["FPR_gap"]

    summary = summarize_stability(reports)

    assert "FPR_gap" not in summary["conditions"]["baseline"]["metrics"]
    assert "FPR_gap" not in summary["conditions"]["adjusted"]["worst_absolute_gaps"]


def test_seed_specific_split_evidence_does_not_break_comparability() -> None:
    first = _report(1)
    second = _report(2)
    first["protocol"]["split_counts"] = {"train": 80, "val": 20, "test": 40}
    second["protocol"]["split_counts"] = {"train": 79, "val": 21, "test": 40}
    first["protocol"]["split_cell_counts"] = [{"cell": "A", "n": 10}]
    second["protocol"]["split_cell_counts"] = [{"cell": "A", "n": 11}]

    summary = summarize_stability([first, second])

    assert summary["seeds"] == [1, 2]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda report: report.update(schema_version="2.0"), "report schema differs"),
        (lambda report: report["metadata"].update(model_type="rf"), "model type differs"),
        (
            lambda report: report["metadata"].update(model_parameters={"max_depth": 8}),
            "model parameters differs",
        ),
        (
            lambda report: report["metadata"]["resolved_config"]["fairness"].update(
                frontier_max_abs_tpr_gap=0.10
            ),
            "resolved policy configuration differs",
        ),
        (lambda report: report["metadata"].update(data_sha256="c" * 64), "data differs"),
        (lambda report: report["metadata"].update(source_sha256="d" * 64), "source differs"),
        (
            lambda report: report["protocol"].update(feature_contract_id="adult-income-v3"),
            "feature contract differs",
        ),
        (
            lambda report: report["protocol"].update(validation_strategy="random"),
            "protocol differs",
        ),
        (
            lambda report: report["governance"]["thresholds"].update(max_tpr_gap=0.10),
            "gate threshold policy differs",
        ),
    ],
)
def test_rejects_incomparable_reports(mutate, match: str) -> None:
    first = _report(1)
    second = _report(2)
    mutate(second)

    with pytest.raises(StabilityStudyError, match=match):
        summarize_stability([first, second])


def test_rejects_different_metric_coverage() -> None:
    first = _report(1)
    second = _report(2)
    del second["results"]["metrics"]["FPR_gap"]

    with pytest.raises(StabilityStudyError, match="adjusted metric coverage differs"):
        summarize_stability([first, second])


@pytest.mark.parametrize(
    ("reports", "match"),
    [
        ([], "at least two"),
        ([_report(1)], "at least two"),
        ([_report(1), _report(1)], "seeds must be unique"),
    ],
)
def test_rejects_invalid_collection(reports: list[dict[str, Any]], match: str) -> None:
    with pytest.raises(StabilityStudyError, match=match):
        summarize_stability(reports)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda report: report["metadata"].update(seed=True), "seed must be a nonnegative integer"),
        (lambda report: report["metadata"].update(data_sha256="bad"), "64-hex"),
        (
            lambda report: report["results"]["metrics"].update(accuracy=float("nan")),
            "accuracy must be a finite number",
        ),
        (
            lambda report: report["results"]["metrics"].update(SPD=2.0),
            "SPD must be between -1 and 1",
        ),
        (
            lambda report: report["results"]["thresholds"].update(privileged=1.1),
            "privileged must be between 0 and 1",
        ),
        (lambda report: report["governance"].update(passed="yes"), "passed must be Boolean"),
        (
            lambda report: report["governance"].update(report_valid=False),
            "report_valid must be true",
        ),
    ],
)
def test_rejects_malformed_reports(mutate, match: str) -> None:
    first = _report(1)
    second = copy.deepcopy(_report(2))
    mutate(second)

    with pytest.raises(StabilityStudyError, match=match):
        summarize_stability([first, second])


def test_requires_accuracy_and_a_fairness_metric() -> None:
    missing_accuracy = _report(2)
    del missing_accuracy["results"]["metrics"]["accuracy"]

    with pytest.raises(StabilityStudyError, match="accuracy is required"):
        summarize_stability([_report(1), missing_accuracy])

    missing_fairness = _report(2)
    missing_fairness["results"]["metrics"] = {"accuracy": 0.8}
    with pytest.raises(StabilityStudyError, match="at least one supported fairness metric"):
        summarize_stability([_report(1), missing_fairness])
