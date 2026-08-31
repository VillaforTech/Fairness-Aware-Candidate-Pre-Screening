"""Tests for uncertainty-aware intersectional diagnostics."""

import numpy as np
import pytest

from fairness_project.evaluation.intersectional import intersectional_diagnostics


def _row_for(result, **attributes):
    return next(row for row in result["groups"] if row["attributes"] == attributes)


def test_intersectional_diagnostics_reports_rates_intervals_and_worst_spans() -> None:
    result = intersectional_diagnostics(
        y_true=np.array([1, 1, 0, 0, 1, 1, 0, 0]),
        y_pred=np.array([1, 0, 1, 0, 1, 1, 0, 0]),
        y_proba=np.array([0.9, 0.4, 0.7, 0.2, 0.9, 0.8, 0.2, 0.1]),
        groups={
            "sex": np.array(["F"] * 4 + ["M"] * 4),
            "region": np.array(["A"] * 8),
        },
        min_support=1,
        min_positive=1,
        min_negative=1,
    )

    female = _row_for(result, sex="F", region="A")
    male = _row_for(result, sex="M", region="A")
    assert female["selection_rate"]["estimate"] == pytest.approx(0.5)
    assert female["tpr"]["estimate"] == pytest.approx(0.5)
    assert female["fpr"]["estimate"] == pytest.approx(0.5)
    assert female["tpr"]["interval"]["method"] == "wilson_score"
    assert female["tpr"]["interval"]["lower"] < 0.5
    assert female["tpr"]["interval"]["upper"] > 0.5
    assert male["tpr"]["estimate"] == pytest.approx(1.0)
    assert result["worst_group_spans"]["tpr"]["absolute_span"] == pytest.approx(0.5)
    assert result["worst_group_spans"]["fpr"]["absolute_span"] == pytest.approx(0.5)
    assert result["dimensions"] == ["sex", "region"]


def test_calibration_metrics_are_reported() -> None:
    result = intersectional_diagnostics(
        y_true=np.array([1, 0]),
        y_pred=np.array([1, 0]),
        y_proba=np.array([0.9, 0.1]),
        groups=np.array(["A", "A"]),
        ece_bins=2,
        min_support=1,
        min_positive=1,
        min_negative=1,
    )
    calibration = result["groups"][0]["calibration"]
    assert calibration["brier_score"] == pytest.approx(0.01)
    assert calibration["ece"] == pytest.approx(0.1)
    assert calibration["evidence_status"] == "sufficient"


def test_weighted_rates_use_kish_effective_sample_size_and_label_method() -> None:
    result = intersectional_diagnostics(
        y_true=np.array([1, 1, 0, 0]),
        y_pred=np.array([1, 0, 0, 0]),
        y_proba=np.array([0.9, 0.4, 0.2, 0.1]),
        groups=np.array(["A"] * 4),
        sample_weight=np.array([9.0, 1.0, 1.0, 1.0]),
        min_support=1,
        min_positive=1,
        min_negative=1,
    )
    row = result["groups"][0]
    assert row["tpr"]["estimate"] == pytest.approx(0.9)
    assert row["tpr"]["effective_n"] == pytest.approx(100 / 82)
    assert (
        row["tpr"]["interval"]["method"] == "weighted_wilson_kish_effective_sample_size_sensitivity"
    )
    assert result["weighted"] is True
    assert "sensitivity" in result["methodology"]["rate_interval"]


def test_small_cells_remain_visible_but_are_excluded_from_worst_group_spans() -> None:
    result = intersectional_diagnostics(
        y_true=np.array([1, 0, 1, 0]),
        y_pred=np.array([1, 0, 0, 1]),
        y_proba=np.array([0.8, 0.2, 0.4, 0.7]),
        groups=np.array(["A", "A", "B", "B"]),
        min_support=3,
        min_positive=2,
        min_negative=2,
    )
    assert len(result["groups"]) == 2
    assert all(row["evidence_status"] == "limited" for row in result["groups"])
    assert "support_below_minimum" in result["groups"][0]["evidence_reasons"]
    assert result["groups"][0]["selection_rate"]["estimate"] is not None
    assert result["worst_group_spans"]["selection_rate"]["evidence_status"] == "insufficient_groups"


def test_undefined_conditional_rate_is_explicit() -> None:
    result = intersectional_diagnostics(
        y_true=np.array([0, 0, 1, 0]),
        y_pred=np.array([0, 1, 1, 0]),
        y_proba=np.array([0.1, 0.8, 0.9, 0.2]),
        groups=np.array(["A", "A", "B", "B"]),
        min_support=1,
        min_positive=1,
        min_negative=1,
    )
    group_a = _row_for(result, group="A")
    assert group_a["tpr"]["estimate"] is None
    assert group_a["tpr"]["interval"] is None
    assert group_a["tpr"]["evidence_status"] == "not_estimable"
    assert "zero_metric_denominator" in group_a["tpr"]["evidence_reasons"]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"y_true": np.array([0, 2])}, "binary values"),
        ({"y_pred": np.array([0, -1])}, "binary values"),
        ({"y_proba": np.array([0.2, np.nan])}, "finite"),
        ({"y_proba": np.array([0.2, 1.1])}, "between 0 and 1"),
        ({"groups": np.array(["A"])}, "same length"),
        ({"sample_weight": np.array([1.0, -1.0])}, "nonnegative"),
        ({"sample_weight": np.array([0.0, 0.0])}, "positive total"),
    ],
)
def test_intersectional_diagnostics_rejects_invalid_arrays(overrides, message) -> None:
    arguments = {
        "y_true": np.array([1, 0]),
        "y_pred": np.array([1, 0]),
        "y_proba": np.array([0.8, 0.2]),
        "groups": np.array(["A", "B"]),
        "min_support": 1,
        "min_positive": 1,
        "min_negative": 1,
    }
    arguments.update(overrides)
    with pytest.raises(ValueError, match=message):
        intersectional_diagnostics(**arguments)


def test_intersectional_diagnostics_rejects_missing_group_and_bad_configuration() -> None:
    base = {
        "y_true": np.array([1, 0]),
        "y_pred": np.array([1, 0]),
        "y_proba": np.array([0.8, 0.2]),
    }
    with pytest.raises(ValueError, match="missing"):
        intersectional_diagnostics(**base, groups=np.array(["A", np.nan], dtype=object))
    with pytest.raises(ValueError, match="confidence"):
        intersectional_diagnostics(**base, groups=np.array(["A", "B"]), confidence=1.0)
    with pytest.raises(ValueError, match="ece_bins"):
        intersectional_diagnostics(**base, groups=np.array(["A", "B"]), ece_bins=1)
    with pytest.raises(ValueError, match="positive integers"):
        intersectional_diagnostics(**base, groups=np.array(["A", "B"]), min_support=0)
