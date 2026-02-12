"""
Governance gate for automated deployment decisions.

Validates evaluation reports against configurable fairness and performance thresholds.

Usage:
    python -m fairness_project.governance.gate --report path/to/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GateThresholds:
    """Configurable thresholds for governance gate checks."""

    min_accuracy: float = 0.80
    max_tpr_gap: float = 0.05
    min_disparate_impact: float = 0.80
    max_spd: float = 0.10


@dataclass
class GateResult:
    """Result of a governance gate check."""

    passed: bool
    violations: list[str] = field(default_factory=list)
    metrics_checked: dict[str, Any] = field(default_factory=dict)


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
        return json.load(f)


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

    violations: list[str] = []
    metrics_checked: dict[str, Any] = {}

    # Extract metrics from report structure
    results = report.get("results", {})
    metrics = results.get("metrics", {})

    # Check accuracy
    accuracy = metrics.get("accuracy")
    if accuracy is not None:
        metrics_checked["accuracy"] = accuracy
        if accuracy < thresholds.min_accuracy:
            violations.append(f"accuracy={accuracy:.4f} < min_accuracy={thresholds.min_accuracy}")

    # Check TPR gap
    tpr_gap = metrics.get("TPR_gap")
    if tpr_gap is not None:
        metrics_checked["TPR_gap"] = tpr_gap
        if abs(tpr_gap) > thresholds.max_tpr_gap:
            violations.append(
                f"|TPR_gap|={abs(tpr_gap):.4f} > max_tpr_gap={thresholds.max_tpr_gap}"
            )

    # Check disparate impact
    di = metrics.get("DI")
    if di is not None:
        metrics_checked["DI"] = di
        if di < thresholds.min_disparate_impact:
            violations.append(
                f"DI={di:.4f} < min_disparate_impact={thresholds.min_disparate_impact}"
            )

    # Check SPD
    spd = metrics.get("SPD")
    if spd is not None:
        metrics_checked["SPD"] = spd
        if abs(spd) > thresholds.max_spd:
            violations.append(f"|SPD|={abs(spd):.4f} > max_spd={thresholds.max_spd}")

    return GateResult(
        passed=len(violations) == 0,
        violations=violations,
        metrics_checked=metrics_checked,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for governance gate.

    Returns
    -------
    int
        Exit code: 0 for pass, 1 for fail.
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
    parser.add_argument("--max-tpr-gap", type=float, default=None)
    parser.add_argument("--min-di", type=float, default=None)
    parser.add_argument("--max-spd", type=float, default=None)

    args = parser.parse_args(argv)

    # Build thresholds with any overrides
    thresholds = GateThresholds()
    if args.min_accuracy is not None:
        thresholds.min_accuracy = args.min_accuracy
    if args.max_tpr_gap is not None:
        thresholds.max_tpr_gap = args.max_tpr_gap
    if args.min_di is not None:
        thresholds.min_disparate_impact = args.min_di
    if args.max_spd is not None:
        thresholds.max_spd = args.max_spd

    report = load_report(args.report)
    result = check_gate(report, thresholds)

    print(f"Governance Gate: {'PASSED' if result.passed else 'FAILED'}")
    print(f"Metrics checked: {json.dumps(result.metrics_checked, indent=2)}")

    if result.violations:
        print("Violations:")
        for v in result.violations:
            print(f"  - {v}")

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
