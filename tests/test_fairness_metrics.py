"""Tests for fairness metrics."""

import numpy as np


class TestDemographicParity:
    """Tests for demographic parity calculation."""

    def test_equal_rates(self):
        """Test with equal positive rates."""
        from src.metrics.fairness import demographic_parity

        y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        dp = demographic_parity(y_pred, sensitive)

        assert abs(dp["A"] - 0.5) < 1e-6
        assert abs(dp["B"] - 0.5) < 1e-6

    def test_unequal_rates(self):
        """Test with unequal positive rates."""
        from src.metrics.fairness import demographic_parity

        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        dp = demographic_parity(y_pred, sensitive)

        assert abs(dp["A"] - 1.0) < 1e-6
        assert abs(dp["B"] - 0.0) < 1e-6


class TestStatisticalParityDifference:
    """Tests for SPD calculation."""

    def test_fair_predictions(self):
        """Test with fair predictions (SPD = 0)."""
        from src.metrics.fairness import statistical_parity_difference

        y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        spd = statistical_parity_difference(y_pred, sensitive, "A")

        assert abs(spd) < 1e-6

    def test_biased_predictions(self):
        """Test with biased predictions."""
        from src.metrics.fairness import statistical_parity_difference

        # Group A: 100% positive, Group B: 0% positive
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        spd = statistical_parity_difference(y_pred, sensitive, "A")

        assert abs(spd - 1.0) < 1e-6  # 1.0 - 0.0 = 1.0

    def test_negative_spd(self):
        """Test SPD can be negative (unprivileged has higher rate)."""
        from src.metrics.fairness import statistical_parity_difference

        y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        spd = statistical_parity_difference(y_pred, sensitive, "A")

        assert abs(spd + 1.0) < 1e-6  # 0.0 - 1.0 = -1.0


class TestDisparateImpact:
    """Tests for disparate impact calculation."""

    def test_fair_predictions(self):
        """Test with fair predictions (DI = 1)."""
        from src.metrics.fairness import disparate_impact

        y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        di = disparate_impact(y_pred, sensitive, "A")

        assert abs(di - 1.0) < 1e-6

    def test_biased_predictions(self):
        """Test with biased predictions (DI < 1)."""
        from src.metrics.fairness import disparate_impact

        y_pred = np.array([1, 1, 1, 1, 1, 0, 0, 0])
        sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        di = disparate_impact(y_pred, sensitive, "A")

        # B rate = 0.25, A rate = 1.0, DI = 0.25
        assert abs(di - 0.25) < 1e-6

    def test_zero_privileged_rate(self):
        """Test with zero privileged rate returns NaN."""
        from src.metrics.fairness import disparate_impact

        y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        di = disparate_impact(y_pred, sensitive, "A")

        assert np.isnan(di)


class TestTruePositiveRate:
    """Tests for TPR calculation."""

    def test_perfect_tpr(self):
        """Test with perfect TPR = 1."""
        from src.metrics.fairness import true_positive_rate

        y_true = np.array([1, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 1])  # All positives correct

        tpr = true_positive_rate(y_true, y_pred)

        assert abs(tpr - 1.0) < 1e-6

    def test_zero_tpr(self):
        """Test with zero TPR."""
        from src.metrics.fairness import true_positive_rate

        y_true = np.array([1, 1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 0, 0])  # All negative predictions

        tpr = true_positive_rate(y_true, y_pred)

        assert abs(tpr) < 1e-6

    def test_partial_tpr(self):
        """Test with partial TPR."""
        from src.metrics.fairness import true_positive_rate

        y_true = np.array([1, 1, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 0, 0])  # 2 of 4 positives correct

        tpr = true_positive_rate(y_true, y_pred)

        assert abs(tpr - 0.5) < 1e-6

    def test_no_positives(self):
        """Test with no positive examples."""
        from src.metrics.fairness import true_positive_rate

        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 0, 1, 0])

        tpr = true_positive_rate(y_true, y_pred)

        assert abs(tpr) < 1e-6  # Returns 0 when no positives


class TestComputeFairnessMetrics:
    """Tests for the combined fairness metrics function."""

    def test_returns_all_metrics(self, sample_binary_data):
        """Test that all metrics are returned."""
        from src.metrics.fairness import compute_fairness_metrics

        result = compute_fairness_metrics(
            y_true=sample_binary_data["y_true"],
            y_pred=sample_binary_data["y_pred"],
            sensitive=sample_binary_data["sensitive"],
            privileged_group="Male",
        )

        assert "DP" in result
        assert "SPD" in result
        assert "DI" in result

    def test_with_fixture_data(self, biased_predictions):
        """Test with biased fixture data."""
        from src.metrics.fairness import compute_fairness_metrics

        result = compute_fairness_metrics(
            y_true=biased_predictions["y_true"],
            y_pred=biased_predictions["y_pred"],
            sensitive=biased_predictions["sensitive"],
            privileged_group="Male",
        )

        # Males should have higher positive rate
        assert result["SPD"] > 0
        assert result["DI"] < 1.0
