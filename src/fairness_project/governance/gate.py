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
from dataclasses import asdict, dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import Any, cast

from fairness_project.provenance import UNAVAILABLE_GIT_COMMIT

REPORT_SCHEMA_VERSION = "2.0"
SUPPORTED_MODEL_TYPES = frozenset({"lr", "rf", "xgb"})
_GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")


@dataclass
class GateThresholds:
    """Configurable thresholds for governance gate checks."""

    min_accuracy: float = 0.80
    max_accuracy_drop: float = 0.02
    max_tpr_gap: float = 0.05
    max_fpr_gap: float = 0.05
    max_intersectional_tpr_span: float = 0.10
    max_intersectional_fpr_span: float = 0.10
    min_disparate_impact: float = 0.80
    max_disparate_impact: float = 1.25
    max_spd: float = 0.10

    def __post_init__(self) -> None:
        unit_interval_fields = {
            "min_accuracy": self.min_accuracy,
            "max_accuracy_drop": self.max_accuracy_drop,
            "max_tpr_gap": self.max_tpr_gap,
            "max_fpr_gap": self.max_fpr_gap,
            "max_intersectional_tpr_span": self.max_intersectional_tpr_span,
            "max_intersectional_fpr_span": self.max_intersectional_fpr_span,
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

    def to_dict(self) -> dict[str, float]:
        """Return the exact numeric policy evaluated by the gate."""
        return {name: float(value) for name, value in asdict(self).items()}


@dataclass
class GateResult:
    """Result of a governance gate check."""

    passed: bool
    violations: list[str] = field(default_factory=list)
    metrics_checked: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
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
            "thresholds": self.thresholds,
            "report_valid": self.report_valid,
        }


def _stored_thresholds(report: dict[str, Any]) -> GateThresholds:
    """Load a persisted gate policy when rechecking a completed report."""
    governance = report.get("governance")
    if not isinstance(governance, dict) or "thresholds" not in governance:
        return GateThresholds()
    payload = governance.get("thresholds")
    if not isinstance(payload, dict):
        raise ValueError("governance.thresholds must be an object")
    expected = set(GateThresholds().to_dict())
    observed = set(payload)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            "governance.thresholds must contain the exact gate policy fields; "
            f"missing={missing}, extra={extra}"
        )
    try:
        return GateThresholds(**payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid governance.thresholds: {exc}") from exc


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
        thresholds = _stored_thresholds(report) if isinstance(report, dict) else GateThresholds()
    else:
        # Dataclasses are mutable, so validate again in case callers changed a
        # threshold after construction.
        thresholds.__post_init__()

    structural_errors: list[str] = []

    if not isinstance(report, dict):
        return GateResult(
            passed=False,
            violations=["report must be a JSON object"],
            thresholds=thresholds.to_dict(),
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
        selection_status: Any = None
    else:
        metrics = results.get("metrics")
        baseline_metrics = results.get("baseline_metrics")
        validation_tuning = results.get("validation_tuning")
        selection = (
            validation_tuning.get("selection") if isinstance(validation_tuning, dict) else None
        )
        if not isinstance(validation_tuning, dict):
            structural_errors.append("results.validation_tuning must be an object")
        if not isinstance(selection, dict):
            structural_errors.append("results.validation_tuning.selection must be an object")
            selection_status = None
        else:
            selection_status = selection.get("status")
            if selection_status not in {"feasible", "infeasible"}:
                structural_errors.append(
                    "results.validation_tuning.selection.status must be feasible or infeasible"
                )
            selected_policy = selection.get("selected")
            if selection_status == "feasible" and not isinstance(selected_policy, dict):
                structural_errors.append(
                    "results.validation_tuning.selection.selected must be an object "
                    "when status is feasible"
                )
            if selection_status == "infeasible" and selected_policy is not None:
                structural_errors.append(
                    "results.validation_tuning.selection.selected must be null "
                    "when status is infeasible"
                )

    if isinstance(metrics, dict):
        accuracy = _read_metric(metrics, "accuracy", structural_errors, (0.0, 1.0))
        tpr_gap = _read_metric(metrics, "TPR_gap", structural_errors, (-1.0, 1.0))
        fpr_gap = _read_metric(metrics, "FPR_gap", structural_errors, (-1.0, 1.0))
        if "DI" not in metrics:
            structural_errors.append("missing required metric: DI")
            di = None
        elif metrics["DI"] is None:
            # Disparate impact is mathematically undefined when the privileged
            # group has zero selections. That is valid evidence, but it cannot
            # satisfy the policy gate.
            di = None
        else:
            di = _read_metric(metrics, "DI", structural_errors, (0.0, float("inf")))
        spd = _read_metric(metrics, "SPD", structural_errors, (-1.0, 1.0))
    else:
        structural_errors.append("results.metrics must be an object")
        accuracy = tpr_gap = fpr_gap = di = spd = None

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
            thresholds=thresholds.to_dict(),
            report_valid=False,
        )

    # The early return above proves that all required metrics are populated.
    assert accuracy is not None
    assert tpr_gap is not None
    assert fpr_gap is not None
    assert spd is not None
    assert baseline_accuracy is not None
    assert isinstance(results, dict)

    violations: list[str] = []
    metrics_checked: dict[str, Any] = {}

    metrics_checked["validation_policy_status"] = selection_status
    if selection_status != "feasible":
        violations.append("No offline policy candidate satisfied the validation constraints")

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

    metrics_checked["FPR_gap"] = fpr_gap
    if abs(fpr_gap) > thresholds.max_fpr_gap:
        violations.append(f"|FPR_gap|={abs(fpr_gap):.4f} > max_fpr_gap={thresholds.max_fpr_gap}")

    # Check disparate impact
    if di is None:
        metrics_checked["DI"] = None
        violations.append("DI is not estimable because the privileged selection rate is zero")
    else:
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

    uncertainty = results.get("uncertainty")
    if isinstance(uncertainty, dict):
        intervals = uncertainty.get("intervals")
        adjusted_intervals = intervals.get("adjusted") if isinstance(intervals, dict) else None
        if isinstance(adjusted_intervals, dict):
            for metric_name, bound in (
                ("TPR_gap", thresholds.max_tpr_gap),
                ("FPR_gap", thresholds.max_fpr_gap),
                ("SPD", thresholds.max_spd),
            ):
                interval = adjusted_intervals.get(metric_name)
                if not isinstance(interval, dict):
                    continue
                lower = interval.get("lower")
                upper = interval.get("upper")
                if not isinstance(lower, Real) or not isinstance(upper, Real):
                    continue
                lower_value = float(lower)
                upper_value = float(upper)
                metrics_checked[f"{metric_name}_interval"] = [lower_value, upper_value]
                worst_bound = max(abs(lower_value), abs(upper_value))
                if worst_bound > bound:
                    violations.append(
                        f"{metric_name} interval reaches |{worst_bound:.4f}| > limit={bound}"
                    )
            di_interval = adjusted_intervals.get("DI")
            if isinstance(di_interval, dict):
                lower = di_interval.get("lower")
                upper = di_interval.get("upper")
                if isinstance(lower, Real) and isinstance(upper, Real):
                    lower_value = float(lower)
                    upper_value = float(upper)
                    metrics_checked["DI_interval"] = [lower_value, upper_value]
                    if lower_value < thresholds.min_disparate_impact:
                        violations.append(
                            f"DI interval lower={lower_value:.4f} < "
                            f"min_disparate_impact={thresholds.min_disparate_impact}"
                        )
                    if upper_value > thresholds.max_disparate_impact:
                        violations.append(
                            f"DI interval upper={upper_value:.4f} > "
                            f"max_disparate_impact={thresholds.max_disparate_impact}"
                        )

    review = results.get("selective_review")
    if isinstance(review, dict):
        held_out = review.get("held_out_evaluation")
        overall = held_out.get("overall") if isinstance(held_out, dict) else None
        if isinstance(overall, dict):
            automated_error = overall.get("automated_error_rate")
            constraint_met = overall.get("constraint_met_on_held_out")
            if isinstance(automated_error, Real):
                metrics_checked["review_band_automated_error_rate"] = float(automated_error)
            if constraint_met is False:
                violations.append(
                    "Global review band missed its validation-selected automated-error limit "
                    "on held-out data"
                )

    intersectional = results.get("intersectional_uncertainty")
    if isinstance(intersectional, dict):
        adjusted = intersectional.get("adjusted")
        spans = adjusted.get("worst_group_spans") if isinstance(adjusted, dict) else None
        if isinstance(spans, dict):
            for metric_name, limit in (
                ("tpr", thresholds.max_intersectional_tpr_span),
                ("fpr", thresholds.max_intersectional_fpr_span),
            ):
                span_payload = spans.get(metric_name)
                span = span_payload.get("absolute_span") if isinstance(span_payload, dict) else None
                if isinstance(span, Real):
                    span_value = float(span)
                    metrics_checked[f"intersectional_{metric_name}_span"] = span_value
                    if span_value > limit:
                        violations.append(
                            f"intersectional {metric_name.upper()} span={span_value:.4f} "
                            f"> limit={limit}"
                        )

    return GateResult(
        passed=len(violations) == 0,
        violations=violations,
        metrics_checked=metrics_checked,
        thresholds=thresholds.to_dict(),
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
    parser.add_argument("--max-fpr-gap", type=float, default=None)
    parser.add_argument("--max-intersectional-tpr-span", type=float, default=None)
    parser.add_argument("--max-intersectional-fpr-span", type=float, default=None)
    parser.add_argument("--min-di", type=float, default=None)
    parser.add_argument("--max-di", type=float, default=None)
    parser.add_argument("--max-spd", type=float, default=None)

    args = parser.parse_args(argv)

    try:
        report = load_report(args.report)
        defaults = _stored_thresholds(report)
        overrides_requested = any(
            value is not None
            for value in (
                args.min_accuracy,
                args.max_accuracy_drop,
                args.max_tpr_gap,
                args.max_fpr_gap,
                args.max_intersectional_tpr_span,
                args.max_intersectional_fpr_span,
                args.min_di,
                args.max_di,
                args.max_spd,
            )
        )
        thresholds = GateThresholds(
            min_accuracy=args.min_accuracy
            if args.min_accuracy is not None
            else defaults.min_accuracy,
            max_accuracy_drop=args.max_accuracy_drop
            if args.max_accuracy_drop is not None
            else defaults.max_accuracy_drop,
            max_tpr_gap=args.max_tpr_gap if args.max_tpr_gap is not None else defaults.max_tpr_gap,
            max_fpr_gap=args.max_fpr_gap if args.max_fpr_gap is not None else defaults.max_fpr_gap,
            max_intersectional_tpr_span=(
                args.max_intersectional_tpr_span
                if args.max_intersectional_tpr_span is not None
                else defaults.max_intersectional_tpr_span
            ),
            max_intersectional_fpr_span=(
                args.max_intersectional_fpr_span
                if args.max_intersectional_fpr_span is not None
                else defaults.max_intersectional_fpr_span
            ),
            min_disparate_impact=args.min_di
            if args.min_di is not None
            else defaults.min_disparate_impact,
            max_disparate_impact=args.max_di
            if args.max_di is not None
            else defaults.max_disparate_impact,
            max_spd=args.max_spd if args.max_spd is not None else defaults.max_spd,
        )
        result = check_gate(report, thresholds if overrides_requested else None)
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
