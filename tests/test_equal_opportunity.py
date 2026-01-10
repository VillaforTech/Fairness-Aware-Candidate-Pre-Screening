"""Tests for Equal Opportunity post-processing."""

import numpy as np
import pytest


class TestComputeTPR:
    """Tests for TPR computation in EO module."""

    def test_basic_tpr(self):
        """Test basic TPR calculation."""
        from src.techniques.equal_opportunity import compute_tpr

        y_true = np.array([1, 1, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 0, 1])  # 2/4 positives correct

        tpr = compute_tpr(y_true, y_pred)

        assert abs(tpr - 0.5) < 1e-6

    def test_perfect_tpr(self):
        """Test perfect TPR."""
        from src.techniques.equal_opportunity import compute_tpr

        y_true = np.array([1, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0])

        tpr = compute_tpr(y_true, y_pred)

        assert abs(tpr - 1.0) < 1e-6


class TestEqualOpportunityPostprocessing:
    """Tests for EO post-processing."""

    def test_reduces_tpr_gap(self, biased_predictions):
        """Test that EO post-processing reduces TPR gap."""
        from src.techniques.equal_opportunity import (
            compute_tpr,
            equal_opportunity_postprocessing,
        )

        y_true = biased_predictions["y_true"]
        y_proba = biased_predictions["y_proba"]
        sensitive = biased_predictions["sensitive"]

        # Apply EO
        y_pred_eo, info = equal_opportunity_postprocessing(
            y_true=y_true,
            y_pred_proba=y_proba,
            sensitive_attr=sensitive,
        )

        # Compute TPR gap after
        male_mask = sensitive == "Male"
        female_mask = sensitive == "Female"

        tpr_male = compute_tpr(y_true[male_mask], y_pred_eo[male_mask])
        tpr_female = compute_tpr(y_true[female_mask], y_pred_eo[female_mask])
        gap_after = abs(tpr_male - tpr_female)

        # Gap should be small after EO
        assert gap_after < 0.1, f"TPR gap after EO is {gap_after}, expected < 0.1"

    def test_returns_info_dict(self, biased_predictions):
        """Test that info dict contains expected keys."""
        from src.techniques.equal_opportunity import equal_opportunity_postprocessing

        y_pred_eo, info = equal_opportunity_postprocessing(
            y_true=biased_predictions["y_true"],
            y_pred_proba=biased_predictions["y_proba"],
            sensitive_attr=biased_predictions["sensitive"],
        )

        assert "tpr_priv" in info
        assert "tpr_unpriv_before" in info
        assert "tpr_unpriv_after" in info
        assert "threshold_used" in info

    def test_no_adjustment_when_equal(self):
        """Test that no adjustment is made when TPRs are already equal."""
        from src.techniques.equal_opportunity import equal_opportunity_postprocessing

        np.random.seed(42)
        n = 100

        # Equal TPR for both groups
        y_true = np.concatenate([np.ones(50), np.ones(50)])
        y_proba = np.concatenate([
            np.random.uniform(0.6, 0.9, 50),
            np.random.uniform(0.6, 0.9, 50),
        ])
        sensitive = np.array(["Male"] * 50 + ["Female"] * 50)

        y_pred_eo, info = equal_opportunity_postprocessing(
            y_true=y_true.astype(int),
            y_pred_proba=y_proba,
            sensitive_attr=sensitive,
        )

        # Threshold should be 0.5 if no adjustment needed
        # or close to 0.5 if TPRs were already equal
        assert info["threshold_used"] >= 0.4

    def test_raises_on_missing_group(self):
        """Test that ValueError is raised when a group is missing."""
        from src.techniques.equal_opportunity import equal_opportunity_postprocessing

        y_true = np.array([1, 1, 0, 0])
        y_proba = np.array([0.8, 0.6, 0.3, 0.2])
        sensitive = np.array(["Male", "Male", "Male", "Male"])  # No females

        with pytest.raises(ValueError, match="unprivileged"):
            equal_opportunity_postprocessing(
                y_true=y_true,
                y_pred_proba=y_proba,
                sensitive_attr=sensitive,
            )

    def test_output_is_binary(self, biased_predictions):
        """Test that output predictions are binary."""
        from src.techniques.equal_opportunity import equal_opportunity_postprocessing

        y_pred_eo, _ = equal_opportunity_postprocessing(
            y_true=biased_predictions["y_true"],
            y_pred_proba=biased_predictions["y_proba"],
            sensitive_attr=biased_predictions["sensitive"],
        )

        unique_values = np.unique(y_pred_eo)
        assert all(v in [0, 1] for v in unique_values)


class TestEOLeakageGuard:
    """Tests to guard against data leakage in EO post-processing.

    Note: The current implementation has known leakage issues.
    These tests document the expected behavior for the fixed version.
    """

    def test_tuning_should_use_validation_not_test(self):
        """
        Document that EO threshold tuning should NOT use test labels.

        Current implementation violates this - threshold is tuned on
        the same data it's evaluated on. This test documents the
        expected behavior after PR7 fix.
        """
        # This is a documentation test - the fix is in PR7
        # After fix: threshold should be tuned on val, applied to test
        pass

    def test_apply_thresholds_does_not_use_labels(self):
        """Test that apply_thresholds function doesn't require labels."""
        from src.fairness_project.fairness.postprocess import apply_thresholds

        y_proba = np.array([0.8, 0.6, 0.3, 0.2, 0.7, 0.4])
        sensitive = np.array(["Male", "Male", "Male", "Female", "Female", "Female"])

        # Should work without y_true
        y_pred = apply_thresholds(
            y_pred_proba=y_proba,
            sensitive_attr=sensitive,
            threshold_priv=0.5,
            threshold_unpriv=0.3,
        )

        assert len(y_pred) == len(y_proba)
        assert all(v in [0, 1] for v in np.unique(y_pred))
