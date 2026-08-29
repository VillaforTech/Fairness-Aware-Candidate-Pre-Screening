"""CLI contract tests for the canonical experiment path."""

from __future__ import annotations

from typer.testing import CliRunner

import fairness_project.cli as cli
from fairness_project.experiment import ExperimentResult
from fairness_project.governance.gate import GateResult

runner = CliRunner()


def test_help_exposes_supported_workflow_only() -> None:
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    assert "audit" in result.stdout
    assert "predict" in result.stdout
    assert "mitigate" not in result.stdout


def test_audit_options_reach_experiment(monkeypatch, tmp_path) -> None:
    data_path = tmp_path / "data.csv"
    data_path.write_text("placeholder")
    captured = {}

    def fake_run_experiment(**kwargs):
        captured.update(kwargs)
        return ExperimentResult(
            run_id="cli-run",
            run_dir=tmp_path / "runs" / "cli-run",
            report={},
            gate=GateResult(passed=False, violations=["expected failure"]),
        )

    monkeypatch.setattr(cli, "run_experiment", fake_run_experiment)
    result = runner.invoke(
        cli.app,
        [
            "audit",
            "--model",
            "rf",
            "--seed",
            "17",
            "--data-path",
            str(data_path),
            "--output-dir",
            str(tmp_path / "runs"),
            "--run-id",
            "cli-run",
            "--bootstrap-samples",
            "25",
        ],
    )

    assert result.exit_code == 0
    assert captured["model_type"] == "rf"
    assert captured["seed"] == 17
    assert captured["run_id"] == "cli-run"
    assert captured["bootstrap_samples"] == 25
    assert "Experimental policy gate: FAILED" in result.stdout
