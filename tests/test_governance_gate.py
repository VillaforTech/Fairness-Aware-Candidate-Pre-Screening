"""Tests for strict evaluation-report governance checks."""

from __future__ import annotations

import json
import math

import pytest

from fairness_project.governance.gate import GateThresholds, check_gate, load_report, main


def _make_report(
    accuracy: float = 0.85,
    baseline_accuracy: float = 0.86,
    tpr_gap: float = 0.03,
    fpr_gap: float = 0.03,
    di: float = 0.90,
    spd: float = 0.05,
) -> dict:
    return {
        "schema_version": "2.0",
        "metadata": {
            "run_id": "test_run",
            "seed": 42,
            "model_type": "lr",
            "git_commit": "a" * 40,
            "dirty_worktree": False,
            "data_sha256": "b" * 64,
            "source_sha256": "c" * 64,
        },
        "results": {
            "baseline_metrics": {"accuracy": baseline_accuracy},
            "metrics": {
                "accuracy": accuracy,
                "TPR_gap": tpr_gap,
                "FPR_gap": fpr_gap,
                "DI": di,
                "SPD": spd,
            },
            "validation_tuning": {
                "selection": {
                    "status": "feasible",
                    "selected": {
                        "threshold_privileged": 0.5,
                        "threshold_unprivileged": 0.4,
                    },
                }
            },
        },
    }


def test_passing_report_checks_accuracy_tradeoff() -> None:
    result = check_gate(_make_report())
    assert result.passed is True
    assert result.report_valid is True
    assert result.exit_code == 0
    assert result.metrics_checked["accuracy_drop"] == pytest.approx(0.01)
    assert result.thresholds == GateThresholds().to_dict()


def test_infeasible_validation_policy_is_a_valid_rejection() -> None:
    report = _make_report()
    report["results"]["validation_tuning"]["selection"] = {
        "status": "infeasible",
        "selected": None,
    }

    result = check_gate(report)

    assert result.report_valid is True
    assert result.passed is False
    assert result.exit_code == 1
    assert result.metrics_checked["validation_policy_status"] == "infeasible"
    assert any("No offline policy candidate" in item for item in result.violations)


def test_undefined_disparate_impact_is_a_valid_policy_rejection() -> None:
    report = _make_report()
    report["results"]["metrics"]["DI"] = None

    result = check_gate(report)

    assert result.report_valid is True
    assert result.passed is False
    assert result.metrics_checked["DI"] is None
    assert any("DI is not estimable" in item for item in result.violations)


@pytest.mark.parametrize(
    "selection",
    [
        None,
        {"status": "unknown", "selected": None},
        {"status": "feasible", "selected": None},
        {"status": "infeasible", "selected": {}},
    ],
)
def test_malformed_validation_policy_evidence_fails_closed(selection) -> None:
    report = _make_report()
    report["results"]["validation_tuning"]["selection"] = selection

    result = check_gate(report)

    assert result.report_valid is False
    assert result.exit_code == 2


@pytest.mark.parametrize(
    ("overrides", "metric"),
    [
        ({"accuracy": 0.50}, "accuracy"),
        ({"baseline_accuracy": 0.90, "accuracy": 0.85}, "accuracy_drop"),
        ({"tpr_gap": 0.20}, "TPR_gap"),
        ({"fpr_gap": 0.20}, "FPR_gap"),
        ({"di": 0.50}, "DI"),
        ({"di": 1.50}, "DI"),
        ({"spd": 0.30}, "SPD"),
    ],
)
def test_policy_violations_fail(overrides, metric) -> None:
    result = check_gate(_make_report(**overrides))
    assert result.passed is False
    assert result.report_valid is True
    assert result.exit_code == 1
    assert any(metric in violation for violation in result.violations)


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf, True, "0.9"])
def test_invalid_metric_types_and_non_finite_values_fail(invalid) -> None:
    report = _make_report()
    report["results"]["metrics"]["TPR_gap"] = invalid
    result = check_gate(report)
    assert result.passed is False
    assert result.report_valid is False
    assert result.exit_code == 2
    assert any("TPR_gap" in violation for violation in result.violations)


def test_missing_metrics_and_metadata_fail_closed() -> None:
    result = check_gate({"schema_version": "2.0", "metadata": {}, "results": {}})
    assert result.passed is False
    assert result.report_valid is False
    assert any("results.metrics must be an object" in item for item in result.violations)
    assert any("metadata.run_id" in item for item in result.violations)


def test_wrong_schema_version_fails() -> None:
    report = _make_report()
    report["schema_version"] = "0.1"
    result = check_gate(report)
    assert result.passed is False
    assert result.report_valid is False
    assert result.exit_code == 2


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("run_id", "", "nonempty string"),
        ("run_id", "   ", "nonempty string"),
        ("run_id", 42, "nonempty string"),
        ("seed", True, "non-Boolean nonnegative integer"),
        ("seed", -1, "non-Boolean nonnegative integer"),
        ("seed", 42.0, "non-Boolean nonnegative integer"),
        ("model_type", "svm", "must be one of"),
        ("model_type", "LR", "must be one of"),
        ("git_commit", "a" * 39, "lowercase 40-hex"),
        ("git_commit", "A" * 40, "lowercase 40-hex"),
        ("git_commit", "g" * 40, "lowercase 40-hex"),
        ("data_sha256", "b" * 63, "64-hex SHA-256"),
        ("data_sha256", "z" * 64, "64-hex SHA-256"),
        ("source_sha256", 123, "64-hex SHA-256"),
    ],
)
def test_invalid_metadata_makes_report_structurally_invalid(field, value, expected) -> None:
    report = _make_report()
    report["metadata"][field] = value

    result = check_gate(report)

    assert result.passed is False
    assert result.report_valid is False
    assert result.exit_code == 2
    assert any(expected in violation for violation in result.violations)
    assert result.metrics_checked == {}


def test_installed_distribution_provenance_is_explicit_and_valid() -> None:
    report = _make_report()
    report["metadata"]["git_commit"] = "unavailable"
    report["metadata"]["dirty_worktree"] = None

    result = check_gate(report)

    assert result.report_valid is True
    assert result.exit_code == 0


@pytest.mark.parametrize(
    ("git_commit", "dirty_worktree", "expected"),
    [
        ("unavailable", False, "must be null"),
        ("a" * 40, None, "must be Boolean"),
    ],
)
def test_git_revision_and_worktree_state_must_agree(git_commit, dirty_worktree, expected) -> None:
    report = _make_report()
    report["metadata"]["git_commit"] = git_commit
    report["metadata"]["dirty_worktree"] = dirty_worktree

    result = check_gate(report)

    assert result.report_valid is False
    assert result.exit_code == 2
    assert any(expected in violation for violation in result.violations)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_accuracy": 2},
        {"max_accuracy_drop": -0.1},
        {"max_tpr_gap": math.nan},
        {"max_fpr_gap": -0.1},
        {"min_disparate_impact": 0},
        {"max_disparate_impact": 0.9},
        {"max_spd": True},
    ],
)
def test_invalid_policy_thresholds_are_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        GateThresholds(**kwargs)


def test_custom_thresholds() -> None:
    report = _make_report(accuracy=0.70, baseline_accuracy=0.71)
    assert check_gate(report).passed is False
    custom = GateThresholds(min_accuracy=0.60)
    result = check_gate(report, custom)
    assert result.passed is True
    assert result.thresholds == custom.to_dict()

    report["governance"] = result.to_dict()
    reproduced = check_gate(report)
    assert reproduced.passed is True
    assert reproduced.thresholds == custom.to_dict()


def test_malformed_persisted_threshold_policy_is_rejected() -> None:
    report = _make_report()
    report["governance"] = {"thresholds": {"min_accuracy": 0.5}}

    with pytest.raises(ValueError, match="exact gate policy fields"):
        check_gate(report)


def test_uncertainty_interval_can_fail_a_passing_point_estimate() -> None:
    report = _make_report(tpr_gap=0.01)
    report["results"]["uncertainty"] = {
        "intervals": {"adjusted": {"TPR_gap": {"lower": -0.08, "median": 0.01, "upper": 0.04}}}
    }

    result = check_gate(report)

    assert result.report_valid is True
    assert result.passed is False
    assert any("TPR_gap interval" in violation for violation in result.violations)


def test_load_report_and_cli_exit_codes(tmp_path) -> None:
    passing = tmp_path / "passing.json"
    failing = tmp_path / "failing.json"
    malformed = tmp_path / "malformed.json"
    passing.write_text(json.dumps(_make_report()))
    failing.write_text(json.dumps(_make_report(di=0.4)))
    malformed_report = _make_report()
    malformed_report["metadata"]["git_commit"] = "abc123"
    malformed.write_text(json.dumps(malformed_report))
    assert load_report(passing)["metadata"]["run_id"] == "test_run"
    assert main(["--report", str(passing)]) == 0
    assert main(["--report", str(failing)]) == 1
    assert main(["--report", str(malformed)]) == 2


def test_cli_returns_error_for_invalid_json(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not-json}")
    assert main(["--report", str(path)]) == 2


def test_fairness_gate_command_uses_the_same_exit_codes(tmp_path) -> None:
    from typer.testing import CliRunner

    from fairness_project.cli import app

    runner = CliRunner()
    passing = tmp_path / "passing.json"
    failing = tmp_path / "failing.json"
    malformed = tmp_path / "malformed.json"
    invalid_json = tmp_path / "invalid.json"
    passing.write_text(json.dumps(_make_report()))
    failing.write_text(json.dumps(_make_report(di=0.4)))
    malformed_report = _make_report()
    malformed_report["metadata"]["seed"] = True
    malformed.write_text(json.dumps(malformed_report))
    invalid_json.write_text("{not-json}")

    assert runner.invoke(app, ["gate", "--report", str(passing)]).exit_code == 0
    assert runner.invoke(app, ["gate", "--report", str(failing)]).exit_code == 1
    assert runner.invoke(app, ["gate", "--report", str(malformed)]).exit_code == 2
    assert runner.invoke(app, ["gate", "--report", str(invalid_json)]).exit_code == 2


def test_load_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_report("/nonexistent/report.json")
