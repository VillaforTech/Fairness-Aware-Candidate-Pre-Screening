"""Tests for the governance gate module."""

from __future__ import annotations

import json

import pytest

from fairness_project.governance.gate import (
    GateThresholds,
    check_gate,
    load_report,
    main,
)


def _make_report(
    accuracy: float = 0.85,
    tpr_gap: float = 0.03,
    di: float = 0.90,
    spd: float = 0.05,
) -> dict:
    """Create a sample report dict."""
    return {
        "metadata": {"run_id": "test_run"},
        "results": {
            "metrics": {
                "accuracy": accuracy,
                "TPR_gap": tpr_gap,
                "DI": di,
                "SPD": spd,
            }
        },
    }


class TestCheckGate:
    def test_passing_report(self):
        report = _make_report(accuracy=0.85, tpr_gap=0.03, di=0.90, spd=0.05)
        result = check_gate(report)
        assert result.passed is True
        assert result.violations == []

    def test_failing_accuracy(self):
        report = _make_report(accuracy=0.50)
        result = check_gate(report)
        assert result.passed is False
        assert any("accuracy" in v for v in result.violations)

    def test_failing_tpr_gap(self):
        report = _make_report(tpr_gap=0.20)
        result = check_gate(report)
        assert result.passed is False
        assert any("TPR_gap" in v for v in result.violations)

    def test_failing_di(self):
        report = _make_report(di=0.50)
        result = check_gate(report)
        assert result.passed is False
        assert any("DI" in v for v in result.violations)

    def test_failing_spd(self):
        report = _make_report(spd=0.30)
        result = check_gate(report)
        assert result.passed is False
        assert any("SPD" in v for v in result.violations)

    def test_multiple_violations(self):
        report = _make_report(accuracy=0.50, tpr_gap=0.20, di=0.40, spd=0.30)
        result = check_gate(report)
        assert result.passed is False
        assert len(result.violations) == 4

    def test_custom_thresholds(self):
        report = _make_report(accuracy=0.70)
        # With default thresholds this would fail
        result_default = check_gate(report)
        assert result_default.passed is False

        # With relaxed thresholds it should pass
        thresholds = GateThresholds(min_accuracy=0.60)
        result_custom = check_gate(report, thresholds)
        assert result_custom.passed is True

    def test_missing_metrics_still_passes(self):
        report = {"metadata": {}, "results": {"metrics": {}}}
        result = check_gate(report)
        assert result.passed is True
        assert result.metrics_checked == {}

    def test_metrics_checked_populated(self):
        report = _make_report()
        result = check_gate(report)
        assert "accuracy" in result.metrics_checked
        assert "TPR_gap" in result.metrics_checked
        assert "DI" in result.metrics_checked
        assert "SPD" in result.metrics_checked


class TestLoadReport:
    def test_load_valid_report(self, tmp_path):
        report = _make_report()
        path = tmp_path / "report.json"
        path.write_text(json.dumps(report))

        loaded = load_report(path)
        assert loaded["metadata"]["run_id"] == "test_run"

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_report("/nonexistent/report.json")


class TestMainCLI:
    def test_main_pass(self, tmp_path):
        report = _make_report()
        path = tmp_path / "report.json"
        path.write_text(json.dumps(report))

        exit_code = main(["--report", str(path)])
        assert exit_code == 0

    def test_main_fail(self, tmp_path):
        report = _make_report(accuracy=0.50, tpr_gap=0.20)
        path = tmp_path / "report.json"
        path.write_text(json.dumps(report))

        exit_code = main(["--report", str(path)])
        assert exit_code == 1

    def test_main_custom_thresholds(self, tmp_path):
        report = _make_report(accuracy=0.70)
        path = tmp_path / "report.json"
        path.write_text(json.dumps(report))

        # Should fail with defaults
        assert main(["--report", str(path)]) == 1

        # Should pass with relaxed threshold
        assert main(["--report", str(path), "--min-accuracy", "0.60"]) == 0
