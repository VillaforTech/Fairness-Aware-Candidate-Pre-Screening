"""Tests for the canonical validation-only threshold policy."""

import numpy as np
import pytest

from fairness_project.fairness.postprocess import (
    apply_thresholds,
    compute_tpr,
    equal_opportunity_postprocessing,
    find_optimal_threshold,
    tune_equal_opportunity,
)


def test_compute_tpr() -> None:
    y_true = np.array([1, 1, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 1])
    assert compute_tpr(y_true, y_pred) == pytest.approx(0.5)


def test_compute_tpr_is_undefined_without_positive_labels() -> None:
    assert np.isnan(compute_tpr(np.array([0, 0]), np.array([0, 1])))


def test_tuning_reduces_validation_tpr_gap(biased_predictions) -> None:
    info = tune_equal_opportunity(
        y_val=biased_predictions["y_true"],
        y_proba_val=biased_predictions["y_proba"],
        sensitive_val=biased_predictions["sensitive"],
    )
    assert abs(info["tpr_priv_val"] - info["tpr_unpriv_after_val"]) < 0.1


def test_frozen_threshold_application_does_not_accept_labels() -> None:
    probabilities = np.array([0.8, 0.4, 0.45, 0.2])
    sensitive = np.array(["Male", "Male", "Female", "Female"])
    predictions = apply_thresholds(probabilities, sensitive, 0.5, 0.4)
    assert predictions.tolist() == [1, 0, 1, 0]


def test_threshold_tie_prefers_least_policy_change() -> None:
    threshold, achieved = find_optimal_threshold(
        y_true=np.array([1, 1, 1]),
        y_proba=np.array([0.9, 0.8, 0.7]),
        target_tpr=1.0,
        search_range=(0.0, 0.5),
        reference_threshold=0.5,
    )
    assert threshold == pytest.approx(0.5)
    assert achieved == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("truth", "probability", "sensitive", "message"),
    [
        (np.array([1]), np.array([0.5, 0.6]), np.array(["Male"]), "equal lengths"),
        (
            np.array([1, 0]),
            np.array([0.5, np.nan]),
            np.array(["Male", "Female"]),
            "finite",
        ),
        (
            np.array([1, 0]),
            np.array([0.5, 1.2]),
            np.array(["Male", "Female"]),
            "between 0 and 1",
        ),
    ],
)
def test_tuning_rejects_invalid_inputs(truth, probability, sensitive, message) -> None:
    with pytest.raises(ValueError, match=message):
        equal_opportunity_postprocessing(truth, probability, sensitive)


def test_tuning_rejects_non_estimable_group() -> None:
    with pytest.raises(ValueError, match="Female.*no positive labels"):
        tune_equal_opportunity(
            y_val=np.array([1, 0, 0, 0]),
            y_proba_val=np.array([0.8, 0.2, 0.3, 0.1]),
            sensitive_val=np.array(["Male", "Male", "Female", "Female"]),
        )


def test_application_rejects_unknown_group() -> None:
    with pytest.raises(ValueError, match="Unexpected sensitive"):
        apply_thresholds(
            np.array([0.8, 0.6, 0.4]),
            np.array(["Male", "Female", "Unknown"]),
            0.5,
            0.4,
        )


def test_no_adjustment_when_unprivileged_tpr_is_higher() -> None:
    predictions, info = equal_opportunity_postprocessing(
        y_true=np.array([1, 1, 1, 1, 0, 0]),
        y_pred_proba=np.array([0.9, 0.4, 0.9, 0.8, 0.1, 0.2]),
        sensitive_attr=np.array(["Male", "Male", "Female", "Female", "Male", "Female"]),
    )
    assert info["threshold_unpriv"] == pytest.approx(0.5)
    assert set(np.unique(predictions)) <= {0, 1}
