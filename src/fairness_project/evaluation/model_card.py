"""Render a model card from the same report consumed by the policy gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from fairness_project.models.artifact import load_bundle


def load_run_data(
    run_id: str,
    runs_dir: str | Path = "runs",
) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle = load_bundle(Path(runs_dir) / run_id)
    return bundle.manifest, bundle.report


def generate_model_card(manifest: dict[str, Any], report: dict[str, Any]) -> str:
    """Generate a deterministic card from the integrity-bound report and manifest."""

    def metric(value: Any) -> str:
        return "not estimable" if value is None else f"{float(value):.4f}"

    metadata = report["metadata"]
    protocol = report["protocol"]
    results = report["results"]
    baseline = results["baseline_metrics"]
    adjusted = results["metrics"]
    thresholds = results["thresholds"]
    split_counts = protocol["split_counts"]
    uncertainty = results.get("uncertainty", {})
    adjusted_intervals = uncertainty.get("intervals", {}).get("adjusted", {})
    governance = report["governance"]
    dirty_state = manifest["dirty_worktree"]
    dirty_label = "unavailable (installed distribution)" if dirty_state is None else dirty_state
    sensitive_attribute = protocol["sensitive_attribute"]
    privileged_group = protocol["privileged_group"]
    unprivileged_group = protocol["unprivileged_group"]
    review_policy = results["selective_review"]["policy"]
    review_evaluation = results["selective_review"]["held_out_evaluation"]["overall"]
    overlap_sensitivity = results.get("feature_overlap_sensitivity")
    lines = [
        "# System Card: Auditable Fair-ML Policy Lab",
        "",
        f"> Generated from validated run `{manifest['run_id']}`.",
        "> This is an income-classification policy-audit benchmark, not a hiring model.",
        "",
        "## Provenance",
        "",
        f"- Model: `{manifest['model_type']}`",
        f"- Created: {manifest['created_at']}",
        f"- Git commit: `{manifest['git_commit']}`",
        f"- Data SHA-256: `{manifest['data_sha256']}`",
        f"- Source SHA-256: `{manifest['source_sha256']}`",
        f"- Resolved config SHA-256: `{manifest['config_sha256']}`",
        f"- Feature contract: `{manifest['feature_contract_id']}`",
        f"- Seed: {metadata['seed']}",
        f"- Dirty worktree recorded: {dirty_label}",
        "",
        "## Evaluation protocol",
        "",
        f"- Dataset: {protocol['dataset']}",
        f"- Validation: {protocol['validation_strategy']}",
        (
            "- Rows: "
            f"{split_counts['train']:,} fit / {split_counts['val']:,} validation / "
            f"{split_counts['test']:,} test"
        ),
        "- The offline policy frontier was selected on validation labels only.",
        "- Final metrics were computed once on the preserved official test partition.",
        "- Protected attributes and the CPS final weight were excluded from model features.",
        (
            "- Frozen offline thresholds: "
            f"{privileged_group} `{float(thresholds['privileged']):.3f}`, "
            f"{unprivileged_group} `{float(thresholds['unprivileged']):.3f}` "
            f"for `{sensitive_attribute}`."
        ),
        "- Offline group thresholds are never served by the simulation API.",
        "",
        "## Results",
        "",
        "| Metric | Baseline | Offline adjusted | Change |",
        "|---|---:|---:|---:|",
    ]
    for metric_name in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "SPD",
        "DI",
        "TPR_gap",
        "FPR_gap",
    ):
        before_value = baseline.get(metric_name)
        after_value = adjusted.get(metric_name)
        change = (
            "not estimable"
            if before_value is None or after_value is None
            else f"{float(after_value) - float(before_value):+.4f}"
        )
        lines.append(
            f"| {metric_name} | {metric(before_value)} | {metric(after_value)} | {change} |"
        )

    if adjusted_intervals:
        confidence = float(uncertainty["confidence"]) * 100
        lines.extend(
            [
                "",
                f"### Adjusted {confidence:.0f}% paired-bootstrap intervals",
                "",
                "| Metric | Lower | Upper |",
                "|---|---:|---:|",
            ]
        )
        for metric_name in ("accuracy", "SPD", "DI", "TPR_gap", "FPR_gap"):
            interval = adjusted_intervals.get(metric_name)
            if not isinstance(interval, dict):
                continue
            lines.append(
                f"| {metric_name} | {metric(interval['lower'])} | {metric(interval['upper'])} |"
            )

    lines.extend(
        [
            "",
            "## Global review-band simulation",
            "",
            (
                f"The probability-only policy sends scores from "
                f"`{float(review_policy['lower_threshold']):.3f}` to "
                f"`{float(review_policy['upper_threshold']):.3f}` to review. "
                f"Held-out automation coverage was "
                f"`{float(review_evaluation['automation_coverage']):.1%}` with automated "
                f"accuracy `{metric(review_evaluation['automated_accuracy'])}`."
            ),
            "",
            "Human review is modeled as workload, not assumed to be accurate or unbiased.",
        ]
    )

    if isinstance(overlap_sensitivity, dict):
        counts = overlap_sensitivity.get("counts")
        slices = overlap_sensitivity.get("slices")
        if isinstance(counts, dict) and isinstance(slices, dict):
            all_rows = slices.get("all_held_out")
            novel_rows = slices.get("overlap_excluded")
            if isinstance(all_rows, dict) and isinstance(novel_rows, dict):
                all_adjusted = all_rows.get("adjusted")
                novel_adjusted = novel_rows.get("adjusted")
                all_metrics = (
                    all_adjusted.get("metrics") if isinstance(all_adjusted, dict) else None
                )
                novel_metrics = (
                    novel_adjusted.get("metrics") if isinstance(novel_adjusted, dict) else None
                )
                lines.extend(
                    [
                        "",
                        "## Exact-feature overlap sensitivity",
                        "",
                        (
                            f"The held-out set contained `{int(counts.get('overlap_rows', 0)):,}` "
                            f"exact canonical-feature overlaps and "
                            f"`{int(counts.get('novel_rows', 0)):,}` novel rows. "
                            "The audit removed overlaps without retraining or retuning."
                        ),
                    ]
                )
                if isinstance(all_metrics, dict) and isinstance(novel_metrics, dict):
                    lines.append(
                        "Adjusted accuracy was "
                        f"`{metric(all_metrics.get('accuracy'))}` on all held-out rows and "
                        f"`{metric(novel_metrics.get('accuracy'))}` on the novel-only slice."
                    )
                lines.extend(
                    [
                        "",
                        "This sensitivity does not make the remaining slice an independent "
                        "or externally representative dataset.",
                    ]
                )

    verdict = "passed" if governance["passed"] else "rejected"
    lines.extend(
        [
            "",
            "## Experimental policy gate",
            "",
            f"### Verdict: {verdict}",
            "",
        ]
    )
    if governance["violations"]:
        lines.extend(f"- {violation}" for violation in governance["violations"])
    else:
        lines.append("- No configured threshold violations.")

    lines.extend(
        [
            "",
            "## Serving boundary",
            "",
            "The evaluation-only API serves one global review band based only on model",
            "probability. It rejects protected attributes, Census weights, unknown fields,",
            "and unseen categories. A governance-rejected artifact requires an explicit",
            "research override before the service will start.",
            "",
            "## Limitations",
            "",
            "- Adult is a 1994 census-income dataset, not applicant or job-performance data.",
            "- Adult contains no applicants, qualifications, hiring decisions, or job outcomes.",
            "- Group and intersectional estimates can be unstable for small cells.",
            "- Row bootstrap intervals are not Census design-based confidence intervals.",
            "- Human review quality, appeals, accommodations, and downstream outcomes are absent.",
            "- A gate pass would not establish job validity, safety, legality, or compliance.",
            "",
            "## Authors",
            "",
            "Roberto Villafuerte and Charles Santhakumar, University of Helsinki collaboration.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a model card from a run bundle")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output", default="docs/model_card.md")
    args = parser.parse_args(argv)
    try:
        manifest, report = load_run_data(args.run_id, args.runs_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(generate_model_card(manifest, report), encoding="utf-8")
    print(f"Model card generated at: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
