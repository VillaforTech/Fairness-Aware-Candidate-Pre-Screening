"""Tests for aggregate-only offline monitoring snapshots."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from fairness_project.monitoring.snapshot import (
    AUDIT_SCOPE,
    DriftThresholds,
    build_snapshot,
    compare_snapshots,
    validate_snapshot,
)


def _frame(*, labels: bool = True, weights: bool = False) -> pd.DataFrame:
    data = {
        "age": [22.0, 31.0, 45.0, 58.0, 27.0, 39.0, 51.0, 64.0],
        "workclass": ["Private", "Private", "Public", "Public"] * 2,
        "score": [0.1, 0.8, 0.7, 0.2, 0.3, 0.9, 0.6, 0.4],
        "prediction": [0, 1, 1, 0, 0, 1, 1, 0],
        "sex": ["Female"] * 4 + ["Male"] * 4,
    }
    if labels:
        data["label"] = [0, 1, 1, 0, 0, 1, 0, 1]
    if weights:
        data["fnlwgt"] = [1.0, 2.0, 1.0, 2.0, 4.0, 1.0, 3.0, 2.0]
    return pd.DataFrame(data)


def _snapshot(frame: pd.DataFrame, *, labels: bool = True, weights: bool = False):
    return build_snapshot(
        frame,
        feature_columns=["age", "workclass"],
        categorical_columns=["workclass"],
        score_column="score",
        prediction_column="prediction",
        protected_columns=["sex"],
        label_column="label" if labels else None,
        sample_weight_column="fnlwgt" if weights else None,
        timestamp="2026-08-30T12:00:00-05:00",
    )


def _permissive_thresholds(**overrides):
    values = {
        "min_rows": 1,
        "min_group_rows": 1,
        "max_numeric_psi": 10.0,
        "max_numeric_ks_distance": 1.0,
        "max_categorical_total_variation": 1.0,
        "max_oov_share": 1.0,
        "max_unknown_share_increase": 1.0,
        "max_score_psi": 10.0,
        "max_score_ks_distance": 1.0,
        "max_selection_rate_change": 1.0,
        "max_group_composition_total_variation": 1.0,
        "max_accuracy_drop": 1.0,
        "max_true_positive_rate_drop": 1.0,
        "max_false_positive_rate_increase": 1.0,
        "max_selection_rate_gap_increase": 1.0,
        "max_true_positive_rate_gap_increase": 1.0,
        "max_false_positive_rate_gap_increase": 1.0,
    }
    values.update(overrides)
    return DriftThresholds(**values)


def test_snapshot_is_json_ready_aggregate_only_and_does_not_mutate_input() -> None:
    frame = _frame(weights=True)
    original = frame.copy(deep=True)

    snapshot = _snapshot(frame, weights=True)

    pd.testing.assert_frame_equal(frame, original)
    json.dumps(snapshot, allow_nan=False)
    assert snapshot["audit_scope"] == AUDIT_SCOPE
    assert snapshot["generated_at"] == "2026-08-30T17:00:00Z"
    assert snapshot["protected_audit"]["row_level_data_included"] is False
    assert snapshot["protected_audit"]["columns"]["sex"]["counts"] == {
        "Female": 4,
        "Male": 4,
    }
    assert set(snapshot["protected_audit"]) == {"row_level_data_included", "columns"}
    assert snapshot["weight_audit"]["role"] == "audit_only"
    assert snapshot["weight_audit"]["used_as_model_feature"] is False
    assert snapshot["weight_audit"]["used_for_primary_metrics_or_gate"] is False


def test_identical_snapshots_pass_and_never_emit_p_values() -> None:
    reference = _snapshot(_frame())
    current = _snapshot(_frame())

    result = compare_snapshots(reference, current, thresholds=_permissive_thresholds())

    assert result["gate"]["status"] == "PASS"
    assert result["gate"]["passed"] is True
    assert result["feature_drift"]["numeric"]["age"]["psi"] == pytest.approx(0.0)
    assert result["outcome_drift"]["score"]["ks_distance"] == pytest.approx(0.0)
    assert "p_value" not in json.dumps(result).lower()
    assert "p-values" in result["methodology"]["numeric"]


def test_shift_replay_triggers_numeric_categorical_outcome_and_group_violations() -> None:
    reference_frame = _frame()
    current_frame = _frame()
    current_frame["age"] += 100
    current_frame["workclass"] = ["Novel"] * 8
    current_frame["score"] = [0.99] * 8
    current_frame["prediction"] = [1] * 8
    current_frame["sex"] = ["Female"] * 7 + ["Male"]
    current_frame["label"] = reference_frame["label"]

    result = compare_snapshots(
        _snapshot(reference_frame),
        _snapshot(current_frame),
        thresholds=_permissive_thresholds(
            max_numeric_psi=0.01,
            max_numeric_ks_distance=0.01,
            max_categorical_total_variation=0.01,
            max_oov_share=0.01,
            max_score_psi=0.01,
            max_score_ks_distance=0.01,
            max_selection_rate_change=0.01,
            max_group_composition_total_variation=0.01,
        ),
    )

    codes = {violation["code"] for violation in result["gate"]["violations"]}
    assert result["gate"]["status"] == "FAIL"
    assert "numeric_psi_exceeded" in codes
    assert "numeric_ks_distance_exceeded" in codes
    assert "categorical_total_variation_exceeded" in codes
    assert "categorical_oov_share_exceeded" in codes
    assert "score_psi_exceeded" in codes
    assert "selection_rate_change_exceeded" in codes
    assert "group_composition_total_variation_exceeded" in codes
    assert result["feature_drift"]["categorical"]["workclass"]["oov_categories"] == ["Novel"]


def test_unknown_share_is_explicit_and_gated() -> None:
    reference = _frame()
    current = _frame()
    current.loc[:3, "workclass"] = "?"

    result = compare_snapshots(
        _snapshot(reference),
        _snapshot(current),
        thresholds=_permissive_thresholds(max_unknown_share_increase=0.1),
    )

    drift = result["feature_drift"]["categorical"]["workclass"]
    assert drift["current_unknown_share"] == pytest.approx(0.5)
    assert any(
        item["code"] == "categorical_unknown_share_increase_exceeded"
        for item in result["gate"]["violations"]
    )


def test_delayed_labels_enable_performance_and_fairness_drift_checks() -> None:
    reference = _frame()
    current = _frame()
    current["prediction"] = 1 - current["label"]

    result = compare_snapshots(
        _snapshot(reference),
        _snapshot(current),
        thresholds=_permissive_thresholds(
            max_accuracy_drop=0.01,
            max_true_positive_rate_drop=0.01,
            max_false_positive_rate_increase=0.01,
        ),
    )

    assert result["delayed_label_drift"]["status"] == "available"
    assert result["delayed_label_drift"]["performance"]["accuracy"]["current"] == 0
    codes = {violation["code"] for violation in result["gate"]["violations"]}
    assert "accuracy_drop_exceeded" in codes
    assert "true_positive_rate_drop_exceeded" in codes
    assert "false_positive_rate_increase_exceeded" in codes
    assert result["delayed_label_drift"]["fairness"]["sex"]["row_level_data_included"] is False


def test_unavailable_delayed_labels_are_omitted_unless_policy_requires_them() -> None:
    unlabeled = _snapshot(_frame(labels=False), labels=False)
    reference = _snapshot(_frame())

    optional = compare_snapshots(
        reference,
        unlabeled,
        thresholds=_permissive_thresholds(require_labels=False),
    )
    required = compare_snapshots(
        reference,
        unlabeled,
        thresholds=_permissive_thresholds(require_labels=True),
    )

    assert optional["delayed_label_drift"] == {
        "status": "unavailable_in_one_or_both_snapshots",
        "performance": None,
        "fairness": {},
        "used_for_gate": False,
    }
    assert optional["gate"]["status"] == "PASS"
    assert required["gate"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert required["gate"]["passed"] is False


def test_small_snapshots_and_protected_groups_fail_closed_as_insufficient() -> None:
    result = compare_snapshots(
        _snapshot(_frame()),
        _snapshot(_frame()),
        thresholds=_permissive_thresholds(min_rows=10, min_group_rows=5),
    )

    assert result["gate"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["gate"]["fail_closed"] is True
    codes = {gap["code"] for gap in result["gate"]["evidence_gaps"]}
    assert "reference_rows_below_minimum" in codes
    assert "current_group_rows_below_minimum" in codes


def test_unestimable_label_rates_fail_closed_as_insufficient_evidence() -> None:
    frame = _frame()
    frame["label"] = 1

    result = compare_snapshots(
        _snapshot(frame),
        _snapshot(frame),
        thresholds=_permissive_thresholds(),
    )

    assert result["gate"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert any(
        gap["code"] == "overall_label_metric_not_estimable"
        for gap in result["gate"]["evidence_gaps"]
    )


def test_sample_weights_are_audit_only_and_cannot_change_primary_gate() -> None:
    reference_frame = _frame(weights=True)
    current_frame = _frame(weights=True)
    current_frame["fnlwgt"] = [1000.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    result = compare_snapshots(
        _snapshot(reference_frame, weights=True),
        _snapshot(current_frame, weights=True),
        thresholds=_permissive_thresholds(),
    )

    assert result["gate"]["status"] == "PASS"
    assert result["weight_audit"]["primary_gate_uses_sample_weights"] is False

    with pytest.raises(ValueError, match="audit-only"):
        build_snapshot(
            reference_frame,
            feature_columns=["age", "workclass", "fnlwgt"],
            categorical_columns=["workclass"],
            score_column="score",
            prediction_column="prediction",
            protected_columns=["sex"],
            label_column="label",
            sample_weight_column="fnlwgt",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.assign(age=np.inf), "finite"),
        (lambda frame: frame.assign(score=1.2), "between 0 and 1"),
        (lambda frame: frame.assign(prediction=2), "binary values"),
        (lambda frame: frame.assign(workclass=None), "null"),
        (lambda frame: frame.assign(extra=1), "exactly"),
    ],
)
def test_snapshot_rejects_invalid_or_nonfinite_data(mutation, message) -> None:
    with pytest.raises(ValueError, match=message):
        _snapshot(mutation(_frame()))


def test_compare_rejects_exact_schema_drift_and_corrupted_snapshots() -> None:
    reference = _snapshot(_frame())
    current_frame = _frame().rename(columns={"age": "years"})
    current = build_snapshot(
        current_frame,
        feature_columns=["years", "workclass"],
        categorical_columns=["workclass"],
        score_column="score",
        prediction_column="prediction",
        protected_columns=["sex"],
        label_column="label",
        timestamp="2026-08-30T17:00:00Z",
    )
    with pytest.raises(ValueError, match="incompatible exact schemas"):
        compare_snapshots(reference, current, thresholds=_permissive_thresholds())

    corrupted = copy.deepcopy(reference)
    corrupted["unexpected"] = True
    with pytest.raises(ValueError, match="exact top-level schema"):
        compare_snapshots(corrupted, reference, thresholds=_permissive_thresholds())

    nonfinite = copy.deepcopy(reference)
    nonfinite["outcomes"]["score"]["mean"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        compare_snapshots(nonfinite, reference, thresholds=_permissive_thresholds())


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ("delayed_labels", "performance", "accuracy"),
            0.5,
            "accuracy must match confusion weights",
        ),
        (
            ("delayed_labels", "performance", "positive_weight"),
            5.0,
            "positive_weight must match positive confusion weights",
        ),
        (
            ("delayed_labels", "performance", "negative_weight"),
            5.0,
            "negative_weight must match negative confusion weights",
        ),
        (
            (
                "delayed_labels",
                "protected_group_metrics",
                "sex",
                "groups",
                "Female",
                "true_positive_rate",
            ),
            0.5,
            "true_positive_rate must match confusion weights",
        ),
    ],
)
def test_snapshot_rejects_tampered_label_derived_values(path, value, message) -> None:
    snapshot = _snapshot(_frame())
    target = snapshot
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        validate_snapshot(snapshot)


def test_snapshot_rejects_class_counts_that_contradict_performance_totals() -> None:
    snapshot = _snapshot(_frame())
    snapshot["delayed_labels"]["class_counts"] = {"negative": 3, "positive": 5}

    with pytest.raises(ValueError, match="class_counts must match performance class totals"):
        validate_snapshot(snapshot)


def test_snapshot_rejects_prediction_counts_that_contradict_label_confusion() -> None:
    snapshot = _snapshot(_frame())
    snapshot["outcomes"]["prediction"].update(
        {"negative_count": 5, "positive_count": 3, "selection_rate": 3 / 8}
    )

    with pytest.raises(ValueError, match="prediction counts must match"):
        validate_snapshot(snapshot)


def test_snapshot_rejects_unknown_fields_that_contradict_category_counts() -> None:
    snapshot = _snapshot(_frame())
    categorical = snapshot["distributions"]["categorical"]["workclass"]
    categorical["unknown_count"] = 1
    categorical["unknown_share"] = 1 / 8

    with pytest.raises(ValueError, match="unknown_count must match unknown category counts"):
        validate_snapshot(snapshot)


def test_snapshot_rejects_group_metrics_that_contradict_protected_categories() -> None:
    snapshot = _snapshot(_frame())
    groups = snapshot["delayed_labels"]["protected_group_metrics"]["sex"]["groups"]
    groups["Other"] = groups.pop("Female")

    with pytest.raises(ValueError, match="must match protected category names"):
        validate_snapshot(snapshot)


def test_snapshot_rejects_group_confusion_totals_that_contradict_overall_metrics() -> None:
    snapshot = _snapshot(_frame())
    male = snapshot["delayed_labels"]["protected_group_metrics"]["sex"]["groups"]["Male"]
    male.update(
        {
            "true_positive": 2.0,
            "false_negative": 0.0,
            "false_positive": 0.0,
            "true_negative": 2.0,
            "accuracy": 1.0,
            "true_positive_rate": 1.0,
            "false_positive_rate": 0.0,
            "precision": 1.0,
        }
    )

    with pytest.raises(ValueError, match="must match overall performance"):
        validate_snapshot(snapshot)


def test_snapshot_rejects_tampered_weighted_performance_rate() -> None:
    snapshot = _snapshot(_frame(weights=True), weights=True)
    snapshot["weight_audit"]["weighted_sensitivity"]["performance"]["selection_rate"] = 0.0

    with pytest.raises(ValueError, match="selection_rate must match confusion weights"):
        validate_snapshot(snapshot)


def test_threshold_configuration_is_strict() -> None:
    with pytest.raises(ValueError, match="Unknown threshold"):
        compare_snapshots(
            _snapshot(_frame()),
            _snapshot(_frame()),
            thresholds={"max_magic": 0.1},
        )
    with pytest.raises(ValueError, match="between 0 and 1"):
        DriftThresholds(max_selection_rate_change=1.1)
