"""Render a self-contained HTML view of one fairness audit artifact.

The renderer has no browser-side dependencies and intentionally emits no JavaScript.
Every value originating in the report is escaped before it reaches the document.
"""

from __future__ import annotations

import html
import json
import math
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from statistics import median
from typing import Any

_CORE_METRICS = (
    ("accuracy", "Accuracy", "higher is better", False),
    ("SPD", "Selection-rate gap", "zero is the parity reference", True),
    ("DI", "Selection-rate ratio", "one is the parity reference", False),
    ("TPR_gap", "True-positive-rate gap", "zero is the parity reference", True),
)

_CSS = r"""
:root {
  color-scheme: light;
  --paper: #f2efe7;
  --sheet: #fffdf8;
  --ink: #17201f;
  --muted: #60706b;
  --line: #cbd2cb;
  --teal: #006c67;
  --teal-soft: #cde5df;
  --rust: #aa4f2f;
  --rust-soft: #f0d7ca;
  --gold: #c99628;
  --shadow: 0 22px 70px rgba(26, 42, 39, .11);
  --radius: 22px;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  color: var(--ink);
  background:
    radial-gradient(circle at 12% 2%, rgba(0, 108, 103, .12), transparent 27rem),
    linear-gradient(90deg, rgba(23, 32, 31, .025) 1px, transparent 1px),
    var(--paper);
  background-size: auto, 24px 24px, auto;
  line-height: 1.55;
}
a { color: inherit; }
.shell { width: min(1180px, calc(100% - 32px)); margin: 24px auto 72px; }
.masthead {
  position: relative;
  overflow: hidden;
  display: grid;
  grid-template-columns: minmax(0, 1.8fr) minmax(230px, .7fr);
  gap: 48px;
  min-height: 430px;
  padding: clamp(32px, 6vw, 76px);
  color: #f9f6ed;
  background: #142321;
  border-radius: calc(var(--radius) + 8px);
  box-shadow: var(--shadow);
}
.masthead::after {
  content: "";
  position: absolute;
  width: 330px;
  height: 330px;
  right: -70px;
  top: -90px;
  border: 1px solid rgba(255, 255, 255, .23);
  border-radius: 50%;
  box-shadow: 0 0 0 54px rgba(255,255,255,.035), 0 0 0 108px rgba(255,255,255,.025);
}
.eyebrow, .kicker {
  margin: 0 0 13px;
  color: #73d1c4;
  font: 700 .72rem/1.2 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  letter-spacing: .16em;
  text-transform: uppercase;
}
h1, h2, h3, p { margin-top: 0; }
h1 {
  max-width: 760px;
  margin-bottom: 24px;
  font-family: Georgia, "Times New Roman", serif;
  font-size: clamp(3rem, 7vw, 6.7rem);
  font-weight: 500;
  line-height: .92;
  letter-spacing: -.055em;
}
.dek { max-width: 700px; margin-bottom: 0; color: #cbd9d5; font-size: clamp(1rem, 2vw, 1.22rem); }
.run-stamp {
  align-self: end;
  position: relative;
  z-index: 1;
  padding-left: 20px;
  border-left: 2px solid #73d1c4;
}
.run-stamp dt { color: #91aaa3; font: 650 .68rem/1.2 ui-monospace, monospace; letter-spacing: .12em; text-transform: uppercase; }
.run-stamp dd { margin: 6px 0 20px; overflow-wrap: anywhere; font-size: .95rem; }
.section { margin-top: 24px; padding: clamp(26px, 5vw, 58px); background: var(--sheet); border: 1px solid rgba(23,32,31,.08); border-radius: var(--radius); }
.section-heading { display: grid; grid-template-columns: minmax(0, 1fr) minmax(220px, .7fr); gap: 32px; align-items: end; margin-bottom: 34px; }
.section-heading h2 { margin-bottom: 0; font-family: Georgia, "Times New Roman", serif; font-size: clamp(2rem, 4vw, 3.4rem); font-weight: 500; letter-spacing: -.035em; line-height: 1; }
.section-heading p { margin: 0; color: var(--muted); }
.metric-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }
.metric-card { padding: 21px; background: #f8f6ef; border: 1px solid var(--line); border-radius: 16px; }
.metric-card h3 { min-height: 2.4em; margin-bottom: 4px; font-size: .95rem; }
.metric-card .hint { min-height: 2.5em; margin-bottom: 18px; color: var(--muted); font-size: .76rem; }
.metric-card svg { display: block; width: 100%; height: auto; }
.metric-delta { margin: 10px 0 0; color: var(--muted); font: 650 .75rem/1.4 ui-monospace, monospace; }
.evidence-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
.evidence-item { padding: 18px; border: 1px solid var(--line); border-radius: 14px; }
.evidence-item strong { display: block; margin-bottom: 3px; font-size: .9rem; }
.evidence-item span { color: var(--muted); font-size: .78rem; }
.dot { display: inline-block; width: 8px; height: 8px; margin-right: 8px; border-radius: 50%; background: var(--rust); }
.dot.present { background: var(--teal); }
.datum-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
.datum-grid .datum { display: flex; flex-direction: column; gap: 7px; }
.datum-grid .datum span { color: var(--muted); font: 700 .65rem/1.2 ui-monospace, monospace; letter-spacing: .07em; text-transform: uppercase; }
.datum-grid .datum strong { font: 500 1.45rem/1.1 Georgia, serif; }
.frontier-layout { display: grid; grid-template-columns: minmax(0, 1.65fr) minmax(220px, .55fr); gap: 32px; align-items: center; }
.chart-frame { padding: 14px; background: #f8f6ef; border: 1px solid var(--line); border-radius: 16px; }
.chart-frame svg { display: block; width: 100%; height: auto; }
.chart-note { color: var(--muted); font-size: .82rem; }
.chart-note strong { color: var(--ink); }
.table-scroll { overflow-x: auto; border: 1px solid var(--line); border-radius: 14px; }
table { width: 100%; border-collapse: collapse; font-size: .83rem; }
caption { padding: 17px 18px 11px; text-align: left; color: var(--muted); font-size: .78rem; }
th, td { padding: 11px 14px; border-bottom: 1px solid #dde1dc; text-align: right; vertical-align: top; white-space: nowrap; }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
thead th { color: #45534f; background: #f0eee7; font: 700 .68rem/1.2 ui-monospace, monospace; letter-spacing: .055em; text-transform: uppercase; }
tbody tr:last-child td { border-bottom: 0; }
.status { display: inline-flex; align-items: center; gap: 7px; padding: 4px 8px; color: #235b50; background: var(--teal-soft); border-radius: 999px; font-size: .7rem; }
.status.limited { color: #834126; background: var(--rust-soft); }
.empty { padding: 24px; color: var(--muted); background: #f8f6ef; border: 1px dashed #aeb8b1; border-radius: 14px; }
.sensitivity-grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(220px, .45fr); gap: 28px; align-items: start; }
.annotation { padding: 20px; color: #384944; background: #e5eee9; border-left: 3px solid var(--teal); border-radius: 4px 12px 12px 4px; font-size: .84rem; }
.governance { display: grid; grid-template-columns: minmax(220px, .55fr) minmax(0, 1fr); gap: 28px; }
.gate-card { padding: 26px; color: #fffaf2; background: #173531; border-radius: 16px; }
.gate-card.rejected { background: #5b3024; }
.gate-card .gate-label { margin-bottom: 8px; color: #abd9d1; font: 700 .7rem/1.2 ui-monospace, monospace; letter-spacing: .12em; text-transform: uppercase; }
.gate-card.rejected .gate-label { color: #f1c4b3; }
.gate-card h3 { margin-bottom: 8px; font: 500 2rem/1.1 Georgia, serif; }
.gate-card p { margin-bottom: 0; color: rgba(255,255,255,.76); font-size: .8rem; }
.violation-list { margin: 0; padding-left: 20px; }
.violation-list li + li { margin-top: 8px; }
.distribution-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
.distribution-card { padding: 20px; border: 1px solid var(--line); border-radius: 14px; }
.distribution-card h3 { margin-bottom: 2px; font-size: .9rem; overflow-wrap: anywhere; }
.distribution-card .summary { color: var(--muted); font: 650 .72rem/1.4 ui-monospace, monospace; }
.distribution-card svg { display: block; width: 100%; height: auto; margin-top: 12px; }
.provenance-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
.datum { padding: 15px 16px; background: #f8f6ef; border-top: 2px solid var(--teal); }
.datum dt { color: var(--muted); font: 700 .65rem/1.2 ui-monospace, monospace; letter-spacing: .07em; text-transform: uppercase; }
.datum dd { margin: 8px 0 0; overflow-wrap: anywhere; font-size: .8rem; }
details { margin-top: 16px; }
summary { cursor: pointer; color: var(--teal); font-weight: 700; }
pre { overflow: auto; padding: 18px; color: #dfeae6; background: #13201e; border-radius: 12px; font: .72rem/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; }
.limits { color: #f8f4e9; background: #182522; }
.limits .section-heading p { color: #b8c7c2; }
.limits ul { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px 42px; margin: 0; padding-left: 21px; }
.limits li { color: #d6e0dc; }
.footer { display: flex; justify-content: space-between; gap: 20px; padding: 24px 8px; color: var(--muted); font-size: .75rem; }
.reveal { animation: reveal .55s cubic-bezier(.2,.8,.2,1) both; }
.section:nth-of-type(2) { animation-delay: .04s; }
.section:nth-of-type(3) { animation-delay: .08s; }
@keyframes reveal { from { opacity: 0; transform: translateY(9px); } to { opacity: 1; transform: none; } }
@media (max-width: 900px) {
  .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .evidence-grid, .datum-grid, .provenance-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .frontier-layout, .sensitivity-grid, .governance { grid-template-columns: 1fr; }
}
@media (max-width: 640px) {
  .shell { width: min(100% - 18px, 1180px); margin-top: 9px; }
  .masthead { grid-template-columns: 1fr; min-height: 0; gap: 36px; padding: 28px 22px; border-radius: 18px; }
  h1 { font-size: clamp(2.8rem, 16vw, 4.6rem); }
  .section { padding: 27px 20px; border-radius: 16px; }
  .section-heading { grid-template-columns: 1fr; gap: 12px; }
  .metric-grid, .evidence-grid, .datum-grid, .distribution-grid, .provenance-grid, .limits ul { grid-template-columns: 1fr; }
  .footer { flex-direction: column; }
}
@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto; }
  *, *::before, *::after { animation-duration: .01ms !important; animation-iteration-count: 1 !important; transition-duration: .01ms !important; }
}
@media print {
  @page { size: A4; margin: 12mm; }
  :root { --paper: #fff; --sheet: #fff; --ink: #000; --muted: #444; }
  body { background: #fff; font-size: 9pt; }
  .shell { width: 100%; margin: 0; }
  .masthead { min-height: 0; padding: 18mm 14mm; color: #fff; background: #142321 !important; border-radius: 0; box-shadow: none; print-color-adjust: exact; -webkit-print-color-adjust: exact; }
  .section { margin-top: 8mm; padding: 8mm; break-inside: avoid; border: 1px solid #bbb; }
  .metric-card, .distribution-card, .evidence-item, .datum, .chart-frame, tr { break-inside: avoid; }
  .reveal { animation: none; }
  details { display: none; }
  .footer { padding-inline: 0; }
}
"""


def _escape(value: Any) -> str:
    """Escape a value for HTML text and attribute contexts."""

    if value is None:
        return "Not recorded"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return html.escape(str(value), quote=True)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _metric_value(metrics: Mapping[str, Any], key: str) -> float:
    value = _number(metrics.get(key))
    if value is None:
        raise ValueError(f"results metrics must contain a finite numeric {key!r}")
    return value


def _optional_metric_value(metrics: Mapping[str, Any], key: str) -> float | None:
    """Read a fairness metric that can be undefined for an empty rate denominator."""

    value = metrics.get(key)
    if value is None:
        return None
    number = _number(value)
    if number is None:
        raise ValueError(f"results metrics must contain a finite numeric {key!r} or null")
    return number


def _validate_report(report: Any) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    if not isinstance(report, Mapping):
        raise ValueError("report must be an object")
    schema_version = report.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version.strip():
        raise ValueError("report.schema_version must be a nonempty string")

    metadata = report.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("report.metadata must be an object")
    run_id = metadata.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("report.metadata.run_id must be a nonempty string")

    protocol = report.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("report.protocol must be an object")

    results = report.get("results")
    if not isinstance(results, Mapping):
        raise ValueError("report.results must be an object")
    baseline = results.get("baseline_metrics")
    adjusted = results.get("metrics")
    if not isinstance(baseline, Mapping):
        raise ValueError("report.results.baseline_metrics must be an object")
    if not isinstance(adjusted, Mapping):
        raise ValueError("report.results.metrics must be an object")

    for metrics in (baseline, adjusted):
        accuracy = _metric_value(metrics, "accuracy")
        spd = _optional_metric_value(metrics, "SPD")
        di = _optional_metric_value(metrics, "DI")
        tpr_gap = _optional_metric_value(metrics, "TPR_gap")
        if not 0 <= accuracy <= 1:
            raise ValueError("accuracy must be between 0 and 1")
        if spd is not None and not -1 <= spd <= 1:
            raise ValueError("SPD must be between -1 and 1")
        if di is not None and di < 0:
            raise ValueError("DI must be nonnegative")
        if tpr_gap is not None and not -1 <= tpr_gap <= 1:
            raise ValueError("TPR_gap must be between -1 and 1")
    return metadata, protocol, results


def _fmt(value: Any, digits: int = 4) -> str:
    number = _number(value)
    if number is None:
        return "n/a"
    return f"{number:.{digits}f}"


def _metric_card(
    key: str,
    label: str,
    hint: str,
    absolute: bool,
    baseline: Mapping[str, Any],
    adjusted: Mapping[str, Any],
) -> str:
    before_raw = _number(baseline.get(key))
    after_raw = _number(adjusted.get(key))
    before_value = before_raw if before_raw is not None else 0.0
    after_value = after_raw if after_raw is not None else 0.0
    before = abs(before_value) if absolute else before_value
    after = abs(after_value) if absolute else after_value
    scale = max(1.0, before, after)
    if key == "DI":
        scale = max(1.25, before, after) * 1.04
    bar_width = 190.0
    before_width = max(0.0, min(bar_width, before / scale * bar_width))
    after_width = max(0.0, min(bar_width, after / scale * bar_width))
    marker = 24.0 + (1.0 / scale * bar_width) if key == "DI" else None
    marker_svg = ""
    if marker is not None:
        marker_svg = (
            f'<line x1="{marker:.2f}" y1="17" x2="{marker:.2f}" y2="64" '
            'stroke="#c99628" stroke-width="1.5" stroke-dasharray="3 3" />'
        )
    delta = "n/a" if before_raw is None or after_raw is None else f"{after_raw - before_raw:+.4f}"
    safe_key = key.lower().replace("_", "-")
    return f"""
<article class="metric-card">
  <h3>{_escape(label)}</h3>
  <p class="hint">{_escape(hint)}</p>
  <svg viewBox="0 0 238 84" role="img" aria-labelledby="metric-{safe_key}-title metric-{safe_key}-desc">
    <title id="metric-{safe_key}-title">{_escape(label)} comparison</title>
    <desc id="metric-{safe_key}-desc">Baseline {_fmt(before_raw)}; adjusted {_fmt(after_raw)}.</desc>
    {marker_svg}
    <text x="0" y="29" fill="#60706b" font-size="9">BASE</text>
    <rect x="24" y="19" width="190" height="13" rx="6.5" fill="#dfe3de" />
    <rect x="24" y="19" width="{before_width:.2f}" height="13" rx="6.5" fill="#aa4f2f" />
    <text x="237" y="29" text-anchor="end" fill="#17201f" font-size="10">{_fmt(before_raw)}</text>
    <text x="0" y="59" fill="#60706b" font-size="9">ADJ</text>
    <rect x="24" y="49" width="190" height="13" rx="6.5" fill="#dfe3de" />
    <rect x="24" y="49" width="{after_width:.2f}" height="13" rx="6.5" fill="#006c67" />
    <text x="237" y="59" text-anchor="end" fill="#17201f" font-size="10">{_fmt(after_raw)}</text>
  </svg>
  <p class="metric-delta">delta {delta}</p>
</article>"""


def _evidence_register(
    results: Mapping[str, Any], report: Mapping[str, Any], stability: Any
) -> str:
    tuning = results.get("validation_tuning")
    entries = (
        ("Offline policy frontier", isinstance(tuning, Mapping), "validation-only policy search"),
        (
            "Subgroup cells",
            isinstance(results.get("subgroup_diagnostics"), Mapping),
            "descriptive counts and rates",
        ),
        (
            "Paired uncertainty",
            isinstance(results.get("uncertainty"), Mapping),
            "conditional test-row intervals",
        ),
        (
            "Sampling-weight sensitivity",
            isinstance(results.get("sampling_weight_sensitivity"), Mapping),
            "weighted and unweighted comparison",
        ),
        (
            "Data semantics",
            isinstance(results.get("data_quality"), Mapping),
            "attrition, duplicates, overlap, and sample-weight role",
        ),
        (
            "Novel-only sensitivity",
            isinstance(results.get("feature_overlap_sensitivity"), Mapping),
            "fixed-policy metrics after exact-overlap removal",
        ),
        (
            "Validation dependence",
            isinstance(results.get("validation_dependence"), Mapping),
            "overlap-excluded policy retuning",
        ),
        (
            "Monitoring reference",
            isinstance(results.get("monitoring_reference"), Mapping),
            "aggregate-only drift baseline",
        ),
        (
            "Governance gate",
            isinstance(report.get("governance"), Mapping),
            "explicit policy checks",
        ),
        ("Across-run stability", isinstance(stability, Mapping), "supplied repeat-run evidence"),
    )
    cards = []
    for label, present, detail in entries:
        word = "present" if present else "not recorded"
        cards.append(
            '<div class="evidence-item">'
            f'<strong><i class="dot{" present" if present else ""}"></i>{_escape(label)}</strong>'
            f"<span>{_escape(word)}: {_escape(detail)}</span></div>"
        )
    return "".join(cards)


def _data_quality(results: Mapping[str, Any]) -> str:
    evidence = results.get("data_quality")
    if not isinstance(evidence, Mapping):
        return '<div class="empty">No data-semantics audit is recorded.</div>'
    raw = evidence.get("raw")
    processed = evidence.get("processed")
    selected = raw if isinstance(raw, Mapping) else processed
    if not isinstance(selected, Mapping):
        return '<div class="empty">The data-semantics audit is incomplete.</div>'
    attrition = selected.get("attrition")
    overall = attrition.get("overall") if isinstance(attrition, Mapping) else None
    duplicates = selected.get("duplicates")
    exact = duplicates.get("exact_rows") if isinstance(duplicates, Mapping) else None
    conflicts = duplicates.get("conflicting_labels") if isinstance(duplicates, Mapping) else None
    overlap = (
        duplicates.get("cross_split_predictive_features")
        if isinstance(duplicates, Mapping)
        else None
    )
    weight = selected.get("fnlwgt")
    cards = (
        ("Input rows", overall.get("input_rows") if isinstance(overall, Mapping) else None),
        (
            "Rows removed",
            overall.get("deleted_rows") if isinstance(overall, Mapping) else None,
        ),
        (
            "Deletion rate",
            _fmt(overall.get("deletion_rate"), 3) if isinstance(overall, Mapping) else "n/a",
        ),
        (
            "Exact duplicate groups",
            exact.get("duplicate_groups") if isinstance(exact, Mapping) else None,
        ),
        (
            "Conflicting feature groups",
            conflicts.get("conflicting_feature_groups") if isinstance(conflicts, Mapping) else None,
        ),
        (
            "Cross-split overlaps",
            overlap.get("pairwise_overlap_feature_groups")
            if isinstance(overlap, Mapping)
            else None,
        ),
    )
    rendered = "".join(
        f'<div class="datum"><span>{_escape(label)}</span><strong>{_escape(value)}</strong></div>'
        for label, value in cards
    )
    weight_role = weight.get("contract_role") if isinstance(weight, Mapping) else "not recorded"
    raw_status = "Raw attrition sidecar bound" if isinstance(raw, Mapping) else "Processed CSV only"
    return (
        f'<div class="datum-grid">{rendered}</div>'
        f'<p class="chart-note"><strong>{_escape(raw_status)}.</strong> '
        f"Census final weight role: {_escape(weight_role)}. Counts expose benchmark semantics; "
        "they do not resolve duplicate identities or validate the target for employment use.</p>"
    )


def _overlap_sensitivity(results: Mapping[str, Any]) -> str:
    sensitivity = results.get("feature_overlap_sensitivity")
    if not isinstance(sensitivity, Mapping):
        return '<div class="empty">No exact-feature overlap sensitivity is recorded.</div>'
    counts = sensitivity.get("counts")
    slices = sensitivity.get("slices")
    if not isinstance(counts, Mapping) or not isinstance(slices, Mapping):
        return '<div class="empty">The exact-feature overlap sensitivity is incomplete.</div>'
    all_rows = slices.get("all_held_out")
    novel_rows = slices.get("overlap_excluded")
    if not isinstance(all_rows, Mapping) or not isinstance(novel_rows, Mapping):
        return '<div class="empty">The exact-feature overlap slices are incomplete.</div>'

    cards = (
        ("Reference rows", counts.get("reference_rows")),
        ("Held-out rows", counts.get("held_out_rows")),
        ("Exact overlaps", counts.get("overlap_rows")),
        ("Novel rows", counts.get("novel_rows")),
        ("Overlap rate", _fmt(counts.get("overlap_rate"), 3)),
        ("Novel evidence", novel_rows.get("evidence_status", "not recorded")),
    )
    rendered_cards = "".join(
        f'<div class="datum"><span>{_escape(label)}</span><strong>{_escape(value)}</strong></div>'
        for label, value in cards
    )

    def condition_metrics(container: Mapping[str, Any], policy: str) -> Mapping[str, Any]:
        condition = container.get(policy)
        if not isinstance(condition, Mapping):
            return {}
        metrics = condition.get("metrics")
        return metrics if isinstance(metrics, Mapping) else {}

    all_baseline = condition_metrics(all_rows, "baseline")
    all_adjusted = condition_metrics(all_rows, "adjusted")
    novel_baseline = condition_metrics(novel_rows, "baseline")
    novel_adjusted = condition_metrics(novel_rows, "adjusted")
    table_rows: list[str] = []
    for key, label, _, absolute in (
        *_CORE_METRICS,
        ("FPR_gap", "False-positive-rate gap", "zero is the parity reference", True),
    ):
        values = [
            _number(source.get(key))
            for source in (all_baseline, all_adjusted, novel_baseline, novel_adjusted)
        ]
        displayed = [abs(value) if absolute and value is not None else value for value in values]
        table_rows.append(
            f"<tr><td>{_escape(label)}</td>"
            + "".join(f"<td>{_fmt(value)}</td>" for value in displayed)
            + "</tr>"
        )

    reasons = novel_rows.get("evidence_reasons")
    if isinstance(reasons, Sequence) and not isinstance(reasons, (str, bytes)) and reasons:
        evidence_note = ", ".join(map(str, reasons))
    else:
        evidence_note = "All configured binary-group rate denominators were available."
    return f"""
<div class="datum-grid">{rendered_cards}</div>
<div class="sensitivity-grid" style="margin-top: 22px">
  <div class="table-scroll">
    <table>
      <caption>The same frozen predictions and thresholds are evaluated before and after removing exact canonical feature identities seen in train or validation.</caption>
      <thead><tr><th>Metric</th><th>Base all</th><th>Adj. all</th><th>Base novel</th><th>Adj. novel</th></tr></thead>
      <tbody>{"".join(table_rows)}</tbody>
    </table>
  </div>
  <aside class="annotation">The policy is not retuned on the novel-only slice. Evidence note: {_escape(evidence_note)} Exact-feature removal tests sensitivity to repeated identities; it does not create an independent external dataset.</aside>
</div>"""


def _validation_dependence(results: Mapping[str, Any]) -> str:
    evidence = results.get("validation_dependence")
    if not isinstance(evidence, Mapping):
        return '<div class="empty">No validation-overlap dependence audit is recorded.</div>'
    counts = evidence.get("counts")
    alternate = evidence.get("overlap_excluded_retuning")
    tuning = results.get("validation_tuning")
    primary_selection = tuning.get("selection") if isinstance(tuning, Mapping) else None
    if not isinstance(counts, Mapping) or not isinstance(alternate, Mapping):
        return '<div class="empty">The validation-overlap dependence audit is incomplete.</div>'

    cards = (
        ("Fit rows", counts.get("train_rows")),
        ("Validation rows", counts.get("validation_rows")),
        ("Feature-overlap rows", counts.get("exact_feature_overlap_rows")),
        ("Feature-overlap rate", _fmt(counts.get("exact_feature_overlap_rate"), 3)),
        ("Exact full-record rows", counts.get("exact_full_record_overlap_rows")),
        ("Novel validation rows", counts.get("overlap_excluded_validation_rows")),
    )
    rendered_cards = "".join(
        f'<div class="datum"><span>{_escape(label)}</span><strong>{_escape(value)}</strong></div>'
        for label, value in cards
    )
    primary_selected = (
        primary_selection.get("selected") if isinstance(primary_selection, Mapping) else None
    )
    alternate_selected = alternate.get("selected_frontier_policy")
    primary_status = (
        primary_selection.get("status", "not recorded")
        if isinstance(primary_selection, Mapping)
        else "not recorded"
    )
    alternate_status = alternate.get("frontier_selection_status", alternate.get("status"))

    def threshold(policy: Any, key: str) -> Any:
        return policy.get(key) if isinstance(policy, Mapping) else None

    primary_review = results.get("selective_review")
    primary_review_policy = (
        primary_review.get("policy") if isinstance(primary_review, Mapping) else None
    )
    alternate_review = alternate.get("review_policy")
    reason = alternate.get("reason")
    reason_note = (
        f'<p class="chart-note"><strong>Alternate retuning not estimable:</strong> '
        f"{_escape(reason)}.</p>"
        if reason
        else ""
    )
    return f"""
<div class="datum-grid">{rendered_cards}</div>
<div class="table-scroll" style="margin-top: 22px">
  <table>
    <caption>The fitted model and probabilities stay fixed. Only validation policy selection is repeated after exact-feature overlap exclusion.</caption>
    <thead><tr><th>Policy evidence</th><th>Primary validation</th><th>Overlap-excluded validation</th></tr></thead>
    <tbody>
      <tr><td>Frontier status</td><td>{_escape(primary_status)}</td><td>{_escape(alternate_status)}</td></tr>
      <tr><td>Privileged threshold</td><td>{_fmt(threshold(primary_selected, "threshold_privileged"))}</td><td>{_fmt(threshold(alternate_selected, "threshold_privileged"))}</td></tr>
      <tr><td>Unprivileged threshold</td><td>{_fmt(threshold(primary_selected, "threshold_unprivileged"))}</td><td>{_fmt(threshold(alternate_selected, "threshold_unprivileged"))}</td></tr>
      <tr><td>Review lower bound</td><td>{_fmt(threshold(primary_review_policy, "lower_threshold"))}</td><td>{_fmt(threshold(alternate_review, "lower_threshold"))}</td></tr>
      <tr><td>Review upper bound</td><td>{_fmt(threshold(primary_review_policy, "upper_threshold"))}</td><td>{_fmt(threshold(alternate_review, "upper_threshold"))}</td></tr>
    </tbody>
  </table>
</div>
{reason_note}
<p class="chart-note">These alternate policies diagnose dependence in the validation selector. They never replace the frozen policies evaluated on the reserved test partition.</p>"""


def _frontier_points(
    results: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any] | None]:
    tuning = results.get("validation_tuning")
    if not isinstance(tuning, Mapping):
        return [], None
    selection = tuning.get("selection")
    if not isinstance(selection, Mapping):
        return [], None
    raw_frontier = selection.get("frontier")
    if not isinstance(raw_frontier, Sequence) or isinstance(raw_frontier, (str, bytes)):
        return [], None
    points: list[Mapping[str, Any]] = []
    for item in raw_frontier:
        if not isinstance(item, Mapping):
            continue
        if _number(item.get("accuracy")) is None or _number(item.get("tpr_gap")) is None:
            continue
        points.append(item)
    selected = selection.get("selected")
    return points, selected if isinstance(selected, Mapping) else None


def _same_policy(left: Mapping[str, Any], right: Mapping[str, Any] | None) -> bool:
    if right is None:
        return False
    for key in ("threshold_privileged", "threshold_unprivileged"):
        left_value = _number(left.get(key))
        right_value = _number(right.get(key))
        if left_value is None or right_value is None or not math.isclose(left_value, right_value):
            return False
    return True


def _frontier_chart(results: Mapping[str, Any]) -> str:
    points, selected = _frontier_points(results)
    if not points:
        return '<div class="empty">No validation frontier is recorded in this artifact.</div>'
    ordered = sorted(
        points,
        key=lambda item: (abs(_metric_value(item, "tpr_gap")), _metric_value(item, "accuracy")),
    )
    gaps = [abs(_metric_value(item, "tpr_gap")) for item in ordered]
    accuracies = [_metric_value(item, "accuracy") for item in ordered]
    max_gap = max(max(gaps), 0.01)
    min_accuracy = min(accuracies)
    max_accuracy = max(accuracies)
    accuracy_span = max(max_accuracy - min_accuracy, 0.01)

    def x(value: float) -> float:
        return 54 + value / max_gap * 468

    def y(value: float) -> float:
        return 190 - (value - min_accuracy) / accuracy_span * 142

    coords = [(x(gap), y(accuracy)) for gap, accuracy in zip(gaps, accuracies, strict=True)]
    polyline = " ".join(f"{cx:.2f},{cy:.2f}" for cx, cy in coords)
    circles = []
    for item, (cx, cy) in zip(ordered, coords, strict=True):
        chosen = _same_policy(item, selected)
        circles.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{6 if chosen else 4}" '
            f'fill="{"#c99628" if chosen else "#006c67"}" stroke="#fffdf8" stroke-width="2">'
            f"<title>Accuracy {_fmt(item.get('accuracy'))}, absolute TPR gap "
            f"{_fmt(abs(_metric_value(item, 'tpr_gap')))}</title></circle>"
        )
    selected_note = "No candidate satisfied the configured constraints."
    if selected is not None:
        selected_note = (
            f"Selected validation point: accuracy {_fmt(selected.get('accuracy'))}; "
            f"absolute TPR gap {_fmt(abs(_metric_value(selected, 'tpr_gap')))}; "
            f"thresholds {_fmt(selected.get('threshold_privileged'), 3)} / "
            f"{_fmt(selected.get('threshold_unprivileged'), 3)}."
        )
    return f"""
<div class="frontier-layout">
  <div class="chart-frame">
    <svg viewBox="0 0 560 230" role="img" aria-labelledby="frontier-title frontier-desc">
      <title id="frontier-title">Validation policy frontier</title>
      <desc id="frontier-desc">Accuracy plotted against absolute true-positive-rate gap. Gold marks the selected validation point.</desc>
      <line x1="54" y1="190" x2="522" y2="190" stroke="#9aa59f" />
      <line x1="54" y1="48" x2="54" y2="190" stroke="#9aa59f" />
      <line x1="54" y1="48" x2="522" y2="48" stroke="#d9ddd8" stroke-dasharray="3 4" />
      <polyline points="{polyline}" fill="none" stroke="#006c67" stroke-width="2" />
      {"".join(circles)}
      <text x="288" y="220" text-anchor="middle" fill="#60706b" font-size="11">ABSOLUTE TPR GAP  |  LOWER IS BETTER</text>
      <text transform="translate(14 119) rotate(-90)" text-anchor="middle" fill="#60706b" font-size="11">ACCURACY  |  HIGHER IS BETTER</text>
      <text x="54" y="205" fill="#60706b" font-size="9">0</text>
      <text x="522" y="205" text-anchor="end" fill="#60706b" font-size="9">{max_gap:.3f}</text>
      <text x="47" y="192" text-anchor="end" fill="#60706b" font-size="9">{min_accuracy:.3f}</text>
      <text x="47" y="51" text-anchor="end" fill="#60706b" font-size="9">{max_accuracy:.3f}</text>
    </svg>
  </div>
  <p class="chart-note"><strong>{len(points)} nondominated operating points.</strong><br>{_escape(selected_note)}<br><br>This chart describes validation trade-offs. It is not evidence of job relatedness, external validity, or legal acceptability.</p>
</div>"""


def _group_cell_status(cell: Mapping[str, Any]) -> tuple[str, str]:
    n = _number(cell.get("n"))
    positives = _number(cell.get("positive_labels"))
    negatives = _number(cell.get("negative_labels"))
    if n is None:
        return "unreported", "limited"
    if n < 30:
        return "limited cell", "limited"
    if positives == 0:
        return "TPR unavailable", "limited"
    if negatives == 0:
        return "FPR unavailable", "limited"
    return "descriptive", ""


def _subgroup_tables(results: Mapping[str, Any]) -> str:
    diagnostics = results.get("subgroup_diagnostics")
    if not isinstance(diagnostics, Mapping):
        return (
            '<div class="empty">No subgroup diagnostic cells are recorded in this artifact.</div>'
        )
    baseline = diagnostics.get("baseline")
    adjusted = diagnostics.get("adjusted")
    if not isinstance(baseline, Mapping) or not isinstance(adjusted, Mapping):
        return '<div class="empty">Subgroup diagnostics are incomplete.</div>'
    blocks: list[str] = []
    dimensions = sorted(set(map(str, baseline)) | set(map(str, adjusted)))
    for dimension in dimensions:
        before_groups = baseline.get(dimension, {})
        after_groups = adjusted.get(dimension, {})
        if not isinstance(before_groups, Mapping) or not isinstance(after_groups, Mapping):
            continue
        rows: list[str] = []
        groups = sorted(set(map(str, before_groups)) | set(map(str, after_groups)))
        for group in groups:
            before = before_groups.get(group, {})
            after = after_groups.get(group, {})
            if not isinstance(before, Mapping):
                before = {}
            if not isinstance(after, Mapping):
                after = {}
            evidence_source = after if after else before
            status, status_class = _group_cell_status(evidence_source)
            rows.append(
                "<tr>"
                f"<td>{_escape(dimension)}</td><td>{_escape(group)}</td>"
                f"<td>{_fmt(evidence_source.get('n'), 0)}</td>"
                f"<td>{_fmt(before.get('predicted_positive_rate'))}</td>"
                f"<td>{_fmt(after.get('predicted_positive_rate'))}</td>"
                f"<td>{_fmt(before.get('tpr'))}</td><td>{_fmt(after.get('tpr'))}</td>"
                f"<td>{_fmt(before.get('fpr'))}</td><td>{_fmt(after.get('fpr'))}</td>"
                f'<td><span class="status {status_class}">{_escape(status)}</span></td>'
                "</tr>"
            )
        if rows:
            blocks.append(
                '<div class="table-scroll">'
                f"<table><caption>{_escape(dimension)}: descriptive test-set cells; no multiple-comparison or causal adjustment.</caption>"
                "<thead><tr><th>Dimension</th><th>Group</th><th>n</th>"
                "<th>Selection base</th><th>Selection adj.</th><th>TPR base</th>"
                "<th>TPR adj.</th><th>FPR base</th><th>FPR adj.</th><th>Evidence</th>"
                f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
            )
    return "".join(blocks) or '<div class="empty">No readable subgroup cells were found.</div>'


def _sensitivity_table(results: Mapping[str, Any]) -> str:
    sensitivity = results.get("sampling_weight_sensitivity")
    if not isinstance(sensitivity, Mapping):
        return (
            '<div class="empty">No sampling-weight sensitivity is recorded in this artifact.</div>'
        )
    weighted = sensitivity.get("adjusted_metrics")
    unweighted = results.get("metrics")
    if not isinstance(weighted, Mapping) or not isinstance(unweighted, Mapping):
        return '<div class="empty">Sampling-weight sensitivity is incomplete.</div>'
    rows = []
    for key, label, _, absolute in _CORE_METRICS:
        raw_unweighted = _number(unweighted.get(key))
        raw_weighted = _number(weighted.get(key))
        if raw_unweighted is None or raw_weighted is None:
            continue
        shown_unweighted = abs(raw_unweighted) if absolute else raw_unweighted
        shown_weighted = abs(raw_weighted) if absolute else raw_weighted
        rows.append(
            f"<tr><td>{_escape(label)}</td><td>{shown_unweighted:.4f}</td>"
            f"<td>{shown_weighted:.4f}</td><td>{shown_weighted - shown_unweighted:+.4f}</td></tr>"
        )
    interpretation = sensitivity.get(
        "interpretation",
        "Weights are used here only as a sensitivity comparison.",
    )
    return f"""
<div class="sensitivity-grid">
  <div class="table-scroll">
    <table>
      <caption>Adjusted-policy metrics. Gap columns use absolute magnitude.</caption>
      <thead><tr><th>Metric</th><th>Unweighted</th><th>Weighted</th><th>Delta</th></tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
  </div>
  <aside class="annotation">{_escape(interpretation)} Weight sensitivity does not repair construct validity or turn the benchmark into an employment study.</aside>
</div>"""


def _governance(report: Mapping[str, Any]) -> str:
    governance = report.get("governance")
    if not isinstance(governance, Mapping):
        return '<div class="empty">No governance-gate verdict is recorded in this artifact.</div>'
    passed = governance.get("passed") is True
    valid = governance.get("report_valid") is True
    if not valid:
        heading = "Artifact rejected as malformed"
    elif passed:
        heading = "Configured gate accepted"
    else:
        heading = "Configured gate rejected"
    violations = governance.get("violations")
    items: list[str] = []
    if isinstance(violations, Sequence) and not isinstance(violations, (str, bytes)):
        items = [f"<li>{_escape(item)}</li>" for item in violations]
    if not items:
        items = ["<li>No gate violations were recorded.</li>"]
    card_class = "gate-card" if passed and valid else "gate-card rejected"
    return f"""
<div class="governance">
  <div class="{card_class}">
    <p class="gate-label">Policy gate</p>
    <h3>{_escape(heading)}</h3>
    <p>A gate verdict only evaluates configured numeric rules. It is not a fairness, compliance, or deployment approval.</p>
  </div>
  <div><h3>Recorded violations</h3><ul class="violation-list">{"".join(items)}</ul></div>
</div>"""


def _numeric_sequence(value: Any) -> list[float]:
    if isinstance(value, Mapping):
        for key in ("values", "observations", "samples"):
            if key in value:
                return _numeric_sequence(value[key])
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    output: list[float] = []
    for item in value:
        number = _number(item)
        if number is not None:
            output.append(number)
    return output


def _record_metric(record: Mapping[str, Any], key: str) -> float | None:
    candidates: list[Any] = [record]
    for container_key in ("metrics", "adjusted_metrics", "thresholds"):
        candidates.append(record.get(container_key))
    results = record.get("results")
    if isinstance(results, Mapping):
        candidates.extend([results, results.get("metrics"), results.get("thresholds")])
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            value = _number(candidate.get(key))
            if value is not None:
                return value
    return None


def _stability_series(stability: Mapping[str, Any]) -> dict[str, list[float]]:
    series: dict[str, list[float]] = {}
    conditions = stability.get("conditions")
    if isinstance(conditions, Mapping):
        for condition in ("baseline", "adjusted"):
            condition_payload = conditions.get(condition)
            metrics = (
                condition_payload.get("metrics") if isinstance(condition_payload, Mapping) else None
            )
            if not isinstance(metrics, Mapping):
                continue
            for metric in sorted(metrics, key=str):
                values = _numeric_sequence(metrics[metric])
                if values:
                    series[f"{condition} {metric}"] = values
    thresholds = stability.get("thresholds")
    if isinstance(thresholds, Mapping):
        for group in ("privileged", "unprivileged"):
            values = _numeric_sequence(thresholds.get(group))
            if values:
                series[f"{group} threshold"] = values
    for container_key in ("distributions", "metrics"):
        container = stability.get(container_key)
        if isinstance(container, Mapping):
            for key in sorted(container, key=str):
                values = _numeric_sequence(container[key])
                if values:
                    series[str(key)] = values
    records = stability.get("runs", stability.get("records"))
    if isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
        record_maps = [item for item in records if isinstance(item, Mapping)]
        for key in (
            "accuracy",
            "SPD",
            "DI",
            "TPR_gap",
            "threshold_privileged",
            "threshold_unprivileged",
        ):
            values = [
                value for item in record_maps if (value := _record_metric(item, key)) is not None
            ]
            if values:
                series.setdefault(key, values)
    return series


def _distribution_card(label: str, values: list[float], index: int) -> str:
    low = min(values)
    high = max(values)
    center = median(values)
    span = high - low
    positions = [20 + ((value - low) / span * 280 if span else 140) for value in values]
    dots = "".join(
        f'<circle cx="{position:.2f}" cy="27" r="3.5" fill="#006c67" opacity=".72" />'
        for position in positions
    )
    median_x = 20 + ((center - low) / span * 280 if span else 140)
    return f"""
<article class="distribution-card">
  <h3>{_escape(label)}</h3>
  <p class="summary">n={len(values)}  min={low:.4f}  median={center:.4f}  max={high:.4f}</p>
  <svg viewBox="0 0 320 54" role="img" aria-labelledby="dist-{index}-title dist-{index}-desc">
    <title id="dist-{index}-title">{_escape(label)} across supplied runs</title>
    <desc id="dist-{index}-desc">{len(values)} observations from {low:.4f} to {high:.4f}, median {center:.4f}.</desc>
    <line x1="20" y1="27" x2="300" y2="27" stroke="#cbd2cb" stroke-width="2" />
    {dots}
    <line x1="{median_x:.2f}" y1="11" x2="{median_x:.2f}" y2="43" stroke="#c99628" stroke-width="2" />
  </svg>
</article>"""


def _stability(stability: Mapping[str, Any] | None) -> str:
    if stability is None:
        return (
            '<div class="empty">No repeat-run stability artifact was supplied to this report.</div>'
        )
    if not isinstance(stability, Mapping):
        raise ValueError("stability must be an object when provided")
    series = _stability_series(stability)
    if not series:
        return '<div class="empty">The supplied stability artifact contains no numeric distributions.</div>'
    cards = [
        _distribution_card(label, values, index)
        for index, (label, values) in enumerate(sorted(series.items()), start=1)
    ]
    return (
        '<div class="distribution-grid">'
        + "".join(cards)
        + '</div><p class="chart-note">These are conditional repeat-run sensitivities, not independent samples from a target employment population.</p>'
    )


def _provenance(metadata: Mapping[str, Any], protocol: Mapping[str, Any]) -> str:
    items = (
        ("Run ID", metadata.get("run_id")),
        ("Created", metadata.get("timestamp")),
        ("Model", metadata.get("model_type")),
        ("Seed", metadata.get("seed")),
        ("Git commit", metadata.get("git_commit")),
        ("Dirty worktree", metadata.get("dirty_worktree")),
        ("Data SHA-256", metadata.get("data_sha256")),
        ("Source SHA-256", metadata.get("source_sha256")),
        ("Config SHA-256", metadata.get("config_sha256")),
        ("Feature contract", protocol.get("feature_contract_id")),
        ("Evaluation split", protocol.get("final_evaluation_split")),
        ("Threshold-tuning split", protocol.get("threshold_tuning_split")),
    )
    cards = "".join(
        f'<div class="datum"><dt>{_escape(label)}</dt><dd>{_escape(value)}</dd></div>'
        for label, value in items
    )
    snapshot = json.dumps(
        {"metadata": dict(metadata), "protocol": dict(protocol)},
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    )
    return (
        f'<dl class="provenance-grid">{cards}</dl>'
        "<details><summary>Machine-readable provenance snapshot</summary>"
        f"<pre>{_escape(snapshot)}</pre></details>"
    )


def render_audit_html(report: dict[str, Any], stability: dict[str, Any] | None = None) -> str:
    """Return one deterministic, portable HTML audit document.

    Parameters
    ----------
    report:
        A canonical experiment report. Core metadata, protocol, baseline metrics,
        and adjusted metrics are validated before rendering.
    stability:
        Optional repeat-run artifact. Numeric ``distributions`` or ``runs`` are
        visualized when present.
    """

    metadata, protocol, results = _validate_report(report)
    if stability is not None and not isinstance(stability, Mapping):
        raise ValueError("stability must be an object when provided")
    baseline = results["baseline_metrics"]
    adjusted = results["metrics"]
    assert isinstance(baseline, Mapping)
    assert isinstance(adjusted, Mapping)
    scope = protocol.get("scope", "Benchmark evaluation only")
    dataset = protocol.get("dataset", "Dataset not recorded")
    metric_cards = "".join(
        _metric_card(key, label, hint, absolute, baseline, adjusted)
        for key, label, hint, absolute in _CORE_METRICS
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>Auditable Fair-ML Policy Lab | {_escape(metadata.get("run_id"))}</title>
  <style>{_CSS}</style>
</head>
<body>
<main class="shell">
  <header class="masthead reveal">
    <div>
      <p class="eyebrow">Evidence artifact / {_escape(report.get("schema_version"))}</p>
      <h1>Auditable Fair-ML Policy Lab</h1>
      <p class="dek">A portable record of model utility, group-metric trade-offs, policy constraints, uncertainty, and provenance. It is designed to expose evidence, including rejection, without converting metrics into a claim of fairness.</p>
    </div>
    <dl class="run-stamp">
      <dt>Run</dt><dd>{_escape(metadata.get("run_id"))}</dd>
      <dt>Dataset</dt><dd>{_escape(dataset)}</dd>
      <dt>Scope</dt><dd>{_escape(scope)}</dd>
    </dl>
  </header>

  <section class="section reveal" aria-labelledby="metrics-heading">
    <div class="section-heading"><div><p class="kicker">01 / operating point</p><h2 id="metrics-heading">What changed</h2></div><p>Baseline and adjusted test metrics are shown side by side. A smaller gap is not, by itself, proof that a decision system is fair or valid.</p></div>
    <div class="metric-grid">{metric_cards}</div>
  </section>

  <section class="section reveal" aria-labelledby="evidence-heading">
    <div class="section-heading"><div><p class="kicker">02 / evidence map</p><h2 id="evidence-heading">What is in the bundle</h2></div><p>Presence means the artifact contains that evidence class. It does not rate its quality or establish deployment readiness.</p></div>
    <div class="evidence-grid">{_evidence_register(results, report, stability)}</div>
  </section>

  <section class="section reveal" aria-labelledby="data-heading">
    <div class="section-heading"><div><p class="kicker">03 / data semantics</p><h2 id="data-heading">What preprocessing changes</h2></div><p>Attrition, duplicates, label conflicts, and cross-split overlaps are visible before interpreting model metrics.</p></div>
    {_data_quality(results)}
  </section>

  <section class="section reveal" aria-labelledby="overlap-heading">
    <div class="section-heading"><div><p class="kicker">04 / identity sensitivity</p><h2 id="overlap-heading">What remains without repeats</h2></div><p>Exact canonical feature identities seen in train or validation are removed from a second, fixed-policy view of the held-out set.</p></div>
    <h3>Before policy selection</h3>
    {_validation_dependence(results)}
    <h3 style="margin-top: 34px">After policy selection</h3>
    {_overlap_sensitivity(results)}
  </section>

  <section class="section reveal" aria-labelledby="frontier-heading">
    <div class="section-heading"><div><p class="kicker">05 / policy frontier</p><h2 id="frontier-heading">Utility versus |TPR gap|</h2></div><p>The frontier is fitted on validation data only. Gold identifies the recorded selection when one exists.</p></div>
    {_frontier_chart(results)}
  </section>

  <section class="section reveal" aria-labelledby="subgroups-heading">
    <div class="section-heading"><div><p class="kicker">06 / subgroup ledger</p><h2 id="subgroups-heading">Cell-level evidence</h2></div><p>Counts sit beside rates so sparse and inestimable cells remain visible. These estimates are descriptive.</p></div>
    {_subgroup_tables(results)}
  </section>

  <section class="section reveal" aria-labelledby="sensitivity-heading">
    <div class="section-heading"><div><p class="kicker">07 / weighting</p><h2 id="sensitivity-heading">Sampling-weight sensitivity</h2></div><p>A change under weighting is an analysis signal. It does not represent a design-based population estimate unless the study explicitly supports that claim.</p></div>
    {_sensitivity_table(results)}
  </section>

  <section class="section reveal" aria-labelledby="gate-heading">
    <div class="section-heading"><div><p class="kicker">08 / governance</p><h2 id="gate-heading">Gate verdict</h2></div><p>The report preserves failed checks as first-class evidence instead of hiding them behind a success narrative.</p></div>
    {_governance(report)}
  </section>

  <section class="section reveal" aria-labelledby="stability-heading">
    <div class="section-heading"><div><p class="kicker">09 / repeat runs</p><h2 id="stability-heading">Stability distributions</h2></div><p>Repeated splits or seeds reveal policy sensitivity that a single point estimate cannot show.</p></div>
    {_stability(stability)}
  </section>

  <section class="section reveal" aria-labelledby="provenance-heading">
    <div class="section-heading"><div><p class="kicker">10 / lineage</p><h2 id="provenance-heading">Provenance</h2></div><p>Identifiers, hashes, split roles, and worktree state make the evidence traceable to an exact run context.</p></div>
    {_provenance(metadata, protocol)}
  </section>

  <section class="section limits reveal" aria-labelledby="limits-heading">
    <div class="section-heading"><div><p class="kicker">11 / scope boundary</p><h2 id="limits-heading">How to read this artifact</h2></div><p>{_escape(scope)}</p></div>
    <ul>
      <li>UCI Adult is a historical Census income-classification benchmark, not applicant or job-performance evidence.</li>
      <li>Parity metrics and threshold trade-offs do not establish job relatedness, individual or causal fairness, legal compliance, or deployment safety.</li>
      <li>Group-specific thresholds shown here are offline analytical policies and are not evidence that they should be served.</li>
      <li>Results remain conditional on the data, labels, group definitions, model, split, policy constraints, and uncertainty method recorded in the bundle.</li>
    </ul>
  </section>

  <footer class="footer"><span>Auditable Fair-ML Policy Lab</span><span>Self-contained HTML / no remote dependencies / no client-side scripts</span></footer>
</main>
</body>
</html>
"""


def write_audit_html(
    report: dict[str, Any],
    output_path: str | Path,
    stability: dict[str, Any] | None = None,
) -> Path:
    """Render an audit and write it as UTF-8 HTML, returning the output path."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_audit_html(report, stability), encoding="utf-8")
    return destination
