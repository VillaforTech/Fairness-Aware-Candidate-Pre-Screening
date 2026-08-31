# Contributing

Contributions are welcome when they preserve the project's evidence boundaries.
The shortest acceptable change is one that keeps data meaning, validation-only
selection, held-out evaluation, and artifact provenance aligned.

## Set up

```bash
git clone https://github.com/VillaforTech/Fairness-Aware-Candidate-Pre-Screening.git
cd Fairness-Aware-Candidate-Pre-Screening
uv sync --locked --extra dev --extra api
```

Use a focused branch and keep generated runs, processed data, and local model
artifacts out of the commit unless a reference-evidence update is part of the
reviewed change.

## Required checks

```bash
uv run pytest tests -q -ra
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy --config-file pyproject.toml
uv build
```

Run a real audit when changing data preparation, preprocessing, models, policy
selection, evaluation, governance, artifacts, reporting, or inference.

## Invariants

Every change must preserve these rules unless the proposal explicitly replaces
one with a stronger reviewed design:

1. The official UCI test partition is never used for model or policy selection.
2. `fnlwgt`, protected attributes, target, and split markers stay outside the
   model feature contract.
3. The API does not accept or apply protected attributes.
4. Offline group thresholds are never served.
5. Small or undefined groups remain visible with evidence status.
6. Governance distinguishes a valid rejection from malformed evidence.
7. A run bundle is atomically published with exactly seven required files, and
   every material evidence file is hash-bound.
8. A rejected artifact cannot become the default simulation artifact.
9. Adult is described as Census-income data, not applicant or job-performance
   data.
10. No metric is presented as proof of legal compliance or employment validity.
11. Canonical preprocessing preserves a digest-bound `.quality.json` sidecar;
    experiments verify and embed its data-semantics evidence.
12. Monitoring snapshots remain aggregate-only, role-explicit, schema-validated,
    and separate from the model-selection and governance gates.
13. Exact-overlap sensitivity uses all canonical feature values for identity,
    ignores labels and audit metadata, and never retunes a policy on held-out
    rows.

## Adding a metric

Include:

- a precise definition and sign convention;
- behavior for zero denominators and non-finite values;
- unit tests for valid, invalid, and degenerate inputs;
- report serialization and HTML rendering tests;
- governance behavior if the metric becomes gate-relevant; and
- documentation of what the metric does and does not identify.

## Adding a policy selector

Selection must receive only fit or validation evidence. Provide:

- a deterministic objective and tie-break rule;
- explicit feasibility and infeasibility results;
- a frozen policy representation;
- held-out evaluation that never retunes the policy;
- tests for group absence, class absence, unknown groups, invalid probabilities,
  ties, and boundary thresholds; and
- a clear serving decision. Protected-attribute policies remain offline.

## Changing artifact schemas

Bump the schema version and update manifest, report, policy, writer, loader,
tests, docs, and reference evidence together. Do not add an unbound side file to
a run directory. The required bundle is `manifest.json`, `model.joblib`,
`policy.json`, `report.json`, `predictions.csv`, `audit.html`, and
`monitoring.json`.

## Changing data-semantics evidence

Keep source digests, model-ready digest, row counts, missingness, duplicate and
feature-collision diagnostics, label conflicts, cross-split overlap, protected
group composition, and the `fnlwgt` audit-only role aligned. A present sidecar
must be verified against a fresh processed-data audit before an experiment
starts. Never infer a missing protected value or reinterpret feature-vector
equality as proof of person identity.

Add tests for deterministic output, atomic publication, digest mismatch,
processed-audit mismatch, missing sidecars, and malformed sidecars when this
contract changes.

## Changing overlap sensitivity

Preserve exact tuple equality over every ordered model feature. Do not use a
hash as the final equality test, include labels or protected attributes in
identity, or silently drop a non-estimable novel slice. The complete and
overlap-excluded views must receive the same frozen probabilities, predictions,
and thresholds. Test duplicates, conflicts, no overlap, complete overlap,
missing group denominators, determinism, and input immutability.

## Changing offline monitoring

Keep snapshots aggregate-only and preserve strict feature, category, score,
prediction, protected-column, optional-label, and audit-weight roles. Update the
snapshot schema, manifest binding, comparator, CLI, default thresholds, tests,
and [`docs/monitoring.md`](docs/monitoring.md) together.

Comparison tests must cover `PASS`, `FAIL`, and `INSUFFICIENT_EVIDENCE`, schema
or role mismatch, insufficient total and group counts, category drift, numeric
drift, score and selection drift, and optional delayed-label checks. A monitoring
result must not silently change the trained policy, retrospective governance
verdict, or simulation behavior.

## Reporting results

Generate results from a clean source commit. Record the run ID, seed, model,
data digest, source digest, resolved-config digest, gate verdict, and relevant
uncertainty. Keep failed gates and negative findings. Never rewrite a rejection
as a success because one metric improved.

## Pull request notes

Describe:

- the problem and the evidence boundary it affects;
- files and contracts changed;
- tests and real runs executed;
- any schema or reference-evidence migration; and
- limitations or unresolved questions.

Do not include raw personal data, credentials, private tokens, or unreviewed
employment claims.
