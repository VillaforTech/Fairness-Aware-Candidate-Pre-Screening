"""
Experimental policy gate for evaluation reports.

Validates evaluation reports against configurable fairness and performance thresholds.

Usage:
    python -m fairness_project.governance.gate --report path/to/report.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import Any, cast

from fairness_project.provenance import UNAVAILABLE_GIT_COMMIT

REPORT_SCHEMA_VERSION = "1.0"
SUPPORTED_MODEL_TYPES = frozenset({"lr", "rf", "xgb"})
_GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")


@dataclass
class GateThresholds:
    """Configurable thresholds for governance gate checks."""

    min_accuracy: float = 0.80
    max_accuracy_drop: float = 0.02
    max_tpr_gap: float = 0.05
    min_disparate_impact: float = 0.80
    max_disparate_impact: float = 1.25
    max_spd: float = 0.10

    def __post_init__(self) -> None:
        unit_interval_fields = {
            "min_accuracy": self.min_accuracy,
            "max_accuracy_drop": self.max_accuracy_drop,
            "max_tpr_gap": self.max_tpr_gap,
            "max_spd": self.max_spd,
        }
        for name, value in unit_interval_fields.items():
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if (
            isinstance(self.min_disparate_impact, bool)
            or not isinstance(self.min_disparate_impact, Real)
            or not math.isfinite(self.min_disparate_impact)
        ):
            raise ValueError("min_disparate_impact must be a finite number")
        if not 0 < self.min_disparate_impact <= 1:
            raise ValueError("min_disparate_impact must be greater than 0 and at most 1")
        if (
            isinstance(self.max_disparate_impact, bool)
            or not isinstance(self.max_disparate_impact, Real)
            or not math.isfinite(self.max_disparate_impact)
        ):
            raise ValueError("max_disparate_impact must be a finite number")
        if self.max_disparate_impact < 1:
            raise ValueError("max_disparate_impact must be finite and at least 1")
        if self.min_disparate_impact > self.max_disparate_impact:
            raise ValueError("disparate-impact bounds are inconsistent")


@dataclass
class GateResult:
    """Result of a governance gate check."""

    passed: bool
    violations: list[str] = field(default_factory=list)
    metrics_checked: dict[str, Any] = field(default_factory=dict)
    report_valid: bool = True

    @property
    def exit_code(self) -> int:
        """Return the CLI exit code represented by this result."""
        if not self.report_valid:
            return 2
        return 0 if self.passed else 1

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "passed": self.passed,
            "violations": self.violations,
            "metrics_checked": self.metrics_checked,
            "report_valid": self.report_valid,
        }


def _validate_metadata(metadata: Any, errors: list[str]) -> None:
    """Validate metadata required to reproduce and identify an evaluation."""
    if not isinstance(metadata, dict):
        errors.append("metadata must be an object")
        return

    run_id = metadata.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        errors.append("metadata.run_id must be a nonempty string")

    seed = metadata.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        errors.append("metadata.seed must be a non-Boolean nonnegative integer")

    model_type = metadata.get("model_type")
    if model_type not in SUPPORTED_MODEL_TYPES:
        allowed = ", ".join(sorted(SUPPORTED_MODEL_TYPES))
        errors.append(f"metadata.model_type must be one of: {allowed}")

    git_commit = metadata.get("git_commit")
    dirty_worktree = metadata.get("dirty_worktree")
    if git_commit == UNAVAILABLE_GIT_COMMIT:
        if dirty_worktree is not None:
            errors.append("metadata.dirty_worktree must be null when git_commit is unavailable")
    elif not isinstance(git_commit, str) or _GIT_COMMIT_PATTERN.fullmatch(git_commit) is None:
        errors.append("metadata.git_commit must be a full lowercase 40-hex commit or 'unavailable'")
    elif not isinstance(dirty_worktree, bool):
        errors.append("metadata.dirty_worktree must be Boolean for a Git checkout")

    for name in ("data_sha256", "source_sha256"):
        value = metadata.get(name)
        if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
            errors.append(f"metadata.{name} must be a 64-hex SHA-256 digest")


def _read_metric(
    metrics: dict[str, Any],
    name: str,
    violations: list[str],
    domain: tuple[float, float] | None = None,
) -> float | None:
    if name not in metrics:
        violations.append(f"missing required metric: {name}")
        return None

    value = metrics[name]
    if isinstance(value, bool) or not isinstance(value, Real):
        violations.append(f"{name} must be a number, got {type(value).__name__}")
        return None
    value = float(value)
    if not math.isfinite(value):
        violations.append(f"{name} must be finite")
        return None
    if domain is not None and not domain[0] <= value <= domain[1]:
        violations.append(f"{name}={value:.4f} is outside [{domain[0]}, {domain[1]}]")
        return None
    return value


def load_report(path: str | Path) -> dict[str, Any]:
    """Load a JSON evaluation report.

    Parameters
    ----------
    path : str | Path
        Path to the JSON report file.

    Returns
    -------
    dict[str, Any]
        Parsed report dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Report not found: {path}")

    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Evaluation report must contain a JSON object")
    return cast(dict[str, Any], payload)


def check_gate(
    report: dict[str, Any],
    thresholds: GateThresholds | None = None,
) -> GateResult:
    """Validate a report against governance thresholds.

    Parameters
    ----------
    report : dict[str, Any]
        Evaluation report (as produced by generate_json_report).
    thresholds : GateThresholds, optional
        Thresholds to check against. Uses defaults if not provided.

    Returns
    -------
    GateResult
        Pass/fail result with any violations.
    """
    if thresholds is None:
        thresholds = GateThresholds()
    else:
        # Dataclasses are mutable, so validate again in case callers changed a
        # threshold after construction.
        thresholds.__post_init__()

    structural_errors: list[str] = []

    if not isinstance(report, dict):
        return GateResult(
            passed=False,
            violations=["report must be a JSON object"],
            report_valid=False,
        )
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        structural_errors.append(
            f"schema_version must be '{REPORT_SCHEMA_VERSION}', "
            f"got {report.get('schema_version')!r}"
        )

    _validate_metadata(report.get("metadata"), structural_errors)

    results = report.get("results")
    if not isinstance(results, dict):
        structural_errors.append("results must be an object")
        metrics: Any = None
        baseline_metrics: Any = None
    else:
        metrics = results.get("metrics")
        baseline_metrics = results.get("baseline_metrics")

    if isinstance(metrics, dict):
        accuracy = _read_metric(metrics, "accuracy", structural_errors, (0.0, 1.0))
        tpr_gap = _read_metric(metrics, "TPR_gap", structural_errors, (-1.0, 1.0))
        di = _read_metric(metrics, "DI", structural_errors, (0.0, float("inf")))
        spd = _read_metric(metrics, "SPD", structural_errors, (-1.0, 1.0))
    else:
        structural_errors.append("results.metrics must be an object")
        accuracy = tpr_gap = di = spd = None

    if isinstance(baseline_metrics, dict):
        baseline_accuracy = _read_metric(
            baseline_metrics,
            "accuracy",
            structural_errors,
            (0.0, 1.0),
        )
    else:
        structural_errors.append("results.baseline_metrics must be an object")
        baseline_accuracy = None

    if structural_errors:
        return GateResult(
            passed=False,
            violations=structural_errors,
            report_valid=False,
        )

    # The early return above proves that all required metrics are populated.
    assert accuracy is not None
    assert tpr_gap is not None
    assert di is not None
    assert spd is not None
    assert baseline_accuracy is not None

    violations: list[str] = []
    metrics_checked: dict[str, Any] = {}

    # Check accuracy
    if accuracy is not None:
        metrics_checked["accuracy"] = accuracy
        if accuracy < thresholds.min_accuracy:
            violations.append(f"accuracy={accuracy:.4f} < min_accuracy={thresholds.min_accuracy}")

    # Check TPR gap
    if tpr_gap is not None:
        metrics_checked["TPR_gap"] = tpr_gap
        if abs(tpr_gap) > thresholds.max_tpr_gap:
            violations.append(
                f"|TPR_gap|={abs(tpr_gap):.4f} > max_tpr_gap={thresholds.max_tpr_gap}"
            )

    # Check disparate impact
    if di is not None:
        metrics_checked["DI"] = di
        if di < thresholds.min_disparate_impact:
            violations.append(
                f"DI={di:.4f} < min_disparate_impact={thresholds.min_disparate_impact}"
            )
        if di > thresholds.max_disparate_impact:
            violations.append(
                f"DI={di:.4f} > max_disparate_impact={thresholds.max_disparate_impact}"
            )

    # Check SPD
    if spd is not None:
        metrics_checked["SPD"] = spd
        if abs(spd) > thresholds.max_spd:
            violations.append(f"|SPD|={abs(spd):.4f} > max_spd={thresholds.max_spd}")

    if accuracy is not None and baseline_accuracy is not None:
        accuracy_drop = baseline_accuracy - accuracy
        metrics_checked["baseline_accuracy"] = baseline_accuracy
        metrics_checked["accuracy_drop"] = accuracy_drop
        if accuracy_drop > thresholds.max_accuracy_drop:
            violations.append(
                f"accuracy_drop={accuracy_drop:.4f} > "
                f"max_accuracy_drop={thresholds.max_accuracy_drop}"
            )

    return GateResult(
        passed=len(violations) == 0,
        violations=violations,
        metrics_checked=metrics_checked,
        report_valid=True,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for governance gate.

    Returns
    -------
    int
        Exit code: 0 for pass, 1 for a policy rejection, 2 for malformed input.
    """
    parser = argparse.ArgumentParser(
        description="Run governance gate checks on an evaluation report",
    )
    parser.add_argument(
        "--report",
        required=True,
        help="Path to JSON evaluation report",
    )
    parser.add_argument("--min-accuracy", type=float, default=None)
    parser.add_argument("--max-accuracy-drop", type=float, default=None)
    parser.add_argument("--max-tpr-gap", type=float, default=None)
    parser.add_argument("--min-di", type=float, default=None)
    parser.add_argument("--max-di", type=float, default=None)
    parser.add_argument("--max-spd", type=float, default=None)

    args = parser.parse_args(argv)

    try:
        defaults = GateThresholds()
        thresholds = GateThresholds(
            min_accuracy=args.min_accuracy
            if args.min_accuracy is not None
            else defaults.min_accuracy,
            max_accuracy_drop=args.max_accuracy_drop
            if args.max_accuracy_drop is not None
            else defaults.max_accuracy_drop,
            max_tpr_gap=args.max_tpr_gap if args.max_tpr_gap is not None else defaults.max_tpr_gap,
            min_disparate_impact=args.min_di
            if args.min_di is not None
            else defaults.min_disparate_impact,
            max_disparate_impact=args.max_di
            if args.max_di is not None
            else defaults.max_disparate_impact,
            max_spd=args.max_spd if args.max_spd is not None else defaults.max_spd,
        )
        report = load_report(args.report)
        result = check_gate(report, thresholds)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Governance Gate: ERROR\n  - {exc}")
        return 2

    if result.report_valid:
        status = "PASSED" if result.passed else "FAILED"
        issue_label = "Violations"
    else:
        status = "ERROR"
        issue_label = "Errors"
    print(f"Governance Gate: {status}")
    print(f"Metrics checked: {json.dumps(result.metrics_checked, indent=2)}")

    if result.violations:
        print(f"{issue_label}:")
        for v in result.violations:
            print(f"  - {v}")

    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
