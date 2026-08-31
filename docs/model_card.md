# System Card: Auditable Fair-ML Policy Lab

This document describes the stable system contract. Each audit bundle also
contains run-specific evidence in `report.json`, `manifest.json`, `audit.html`,
and `monitoring.json`. A generated card can be produced from the same
integrity-validated bundle, so metrics never need to be copied by hand.

## System purpose

The lab evaluates an Adult-income classifier, an offline group-threshold policy,
and a separate global review-band policy. It is designed to make policy search,
uncertainty, subgroup evidence, stability, governance, and artifact provenance
inspectable in one workflow.

The target is the UCI Adult `>50K` label. The system does not measure candidate
quality, qualification, suitability, work performance, or hiring success.

## Data

- Source: UCI Adult, derived from 1994 US Census data
- Original task: binary income classification
- Preserved evaluation partition: UCI's official test file
- Validation draw: original training rows, jointly stratified by income, sex,
  and `race_binary`
- Protected audit fields: sex, original UCI race, and derived `race_binary`
- Sampling weight: `fnlwgt`, excluded from predictors and used only for
  weighted sensitivity
- Complete-case attrition: 3,620 of 48,842 raw rows, or 7.41%
- Data-semantics evidence: digest-bound raw and processed attrition,
  missingness, composition, duplicate, conflict, and split-overlap audits

See [`data.md`](data.md) for lineage and limitations.

## Model contract

Contract ID: `adult-income-v2-no-census-weight`

The ordered 11 features are:

```text
age, workclass, education, education_num, marital_status, occupation,
relationship, native_country, capital_gain, capital_loss, hours_per_week
```

Numeric fields are standardized and categorical fields are one-hot encoded in
one fitted pipeline. Protected attributes, target, split marker, and `fnlwgt`
are excluded. The artifact loader checks the contract against the fitted model.

The five numeric fields, in transformer order, are `age`, `education_num`,
`capital_gain`, `capital_loss`, and `hours_per_week`. The six categorical fields,
in transformer order, are `workclass`, `education`, `marital_status`,
`occupation`, `relationship`, and `native_country`.

The fitted one-hot encoder uses `handle_unknown="ignore"`, so validation and test
OOV categories produce zeros for that field's category block. Each run records
split-level and per-field OOV values, affected rows, and shares against the
training vocabulary. The local simulator uses the fitted vocabulary as an input
allowlist and rejects OOV categories before scoring.

The fitted vocabulary is serialized in the digest-bound model, its transformed
feature names are repeated in report and manifest preprocessing evidence, and
the simulator reads the same encoder category arrays. There is no separate
serving vocabulary to drift independently.

Supported estimators are Logistic Regression, Random Forest, and XGBoost.
Model parameters, transformed feature names, source digest, data digest,
runtime, dependencies, and resolved configuration are recorded per run. Run
creation clones the input configuration, synchronizes the effective model type,
validation ratio, seed, and model `random_state`, then hashes the effective
resolved configuration. The caller's configuration object is not mutated.

## Evaluated policies

### Baseline

One global probability threshold produces the paired reference predictions.

### Offline group-threshold policy

Validation labels and one declared binary group attribute are used to enumerate
a two-dimensional threshold grid. The optimizer returns the nondominated
accuracy versus absolute TPR-gap frontier and selects the highest-accuracy point
that meets the configured TPR-gap and accuracy-loss constraints.

This policy is an offline evaluation object. It requires protected-group values
and is never used by the local API.

### Global review band

A second validation-only search chooses a symmetric probability band around the
global threshold. Scores inside the band produce
`manual_review_required`; scores outside it produce an automatic simulation
decision. Selection maximizes automation coverage under an automated-error
constraint and minimum sample count.

Held-out evaluation reports coverage, automated error, review rate, and observed
group spans. No reviewer is implemented, and reviewer accuracy or bias is not
assumed.

### Validation-overlap dependence

The run counts exact 11-feature and exact full-record overlap from fit rows into
validation. It repeats both policy searches on validation rows whose feature
identity does not occur in fit, without changing the fitted model or validation
probabilities. The alternate policies are sensitivity evidence only. They do
not replace the canonical policies evaluated on held-out test rows, and an
unestimable retune remains explicit.

## Measurements

Each run records:

- accuracy, precision, recall, F1, ROC AUC, PR AUC, and Brier score;
- statistical parity difference and disparate impact;
- privileged and unprivileged TPR and FPR with signed gaps;
- paired baseline and adjusted metrics on identical held-out rows;
- sex, original-race, binary-race, and observed sex by original-race cells;
- support, class counts, confusion counts, calibration, and Wilson intervals;
- `sufficient`, `limited`, or `not_estimable` evidence states;
- weighted versus unweighted sensitivity using `fnlwgt`;
- paired label-and-group-stratified bootstrap intervals;
- fixed-policy metrics on the complete and exact-overlap-excluded held-out
  slices, with explicit estimability evidence;
- bound raw and processed data-semantics evidence;
- an aggregate-only held-out monitoring reference;
- validation selection evidence for both policy mechanisms; and
- train-to-validation overlap counts and overlap-excluded policy-retuning
  evidence.

Rates with zero denominators remain not estimable. Small cells stay visible and
are excluded from worst-group spans when their evidence state is insufficient.

## Robustness study

`fairness study` repeats model fitting and both policy selections across unique
seeds. Reports are aggregated only when schema, data, source, model type and
parameters, resolved configuration apart from per-run seed fields, protocol,
feature contract, gate threshold policy, and metric coverage match.

The summary includes distributions for performance and disparity metrics,
threshold distributions, gate pass rate, worst gap by metric, and a deterministic
worst-seed ranking. Every run reuses the same official test rows, so the study
measures training and policy-selection sensitivity rather than independent
population uncertainty.

## Offline monitoring evidence

Each bundle includes a strict aggregate reference built from held-out features,
baseline scores and predictions, observed audit groups, delayed labels, and
audit-only sample weights. It stores distributions and quantile sketches, never
source rows.

`fairness monitor compare` validates a current snapshot against the same role
schema and reports descriptive feature, score, prediction, group-composition,
and optional delayed-label drift. Its status is `PASS`, `FAIL`, or
`INSUFFICIENT_EVIDENCE`. This gate is separate from the model-policy gate and
does not approve an operating context. See [`monitoring.md`](monitoring.md).

Snapshot validation also checks relational consistency among derived
aggregates: category counts, shares, and unknown summaries; prediction counts
and selection rate; class and confusion totals; protected-group rollups; and
binary rates derived from confusion denominators. Contradictory aggregates are
rejected before publication, load, or comparison.

## Governance

The strict schema and configured gate check point metrics, bootstrap interval
bounds, adjusted intersectional TPR and FPR spans, and the held-out review-band
constraint. It distinguishes a pass, a valid rejection, and malformed evidence.

The exact gate thresholds are persisted with the verdict. Bundle save and load
re-evaluate report evidence under those thresholds and require the fresh result
to match report and manifest. The serving review-band and offline group policy
are separately cross-bound to their report selection, threshold, and protocol
evidence.

A pass would mean only that the encoded criteria held for one benchmark
configuration. See [`governance.md`](governance.md).

## Simulation boundary

The local v2 API:

- accepts exactly the 11 model features;
- rejects protected attributes, `fnlwgt`, extra fields, nulls, invalid ranges,
  and unseen categories;
- serves only the global review band;
- returns policy and artifact identity with every result; and
- refuses governance-rejected artifacts unless a research override is explicit.

The interface is for local evidence inspection. It has no operational
employment workflow, security perimeter, privacy program, appeal path, or
accountable decision owner.

## Artifact integrity

Runs are published atomically from a hidden sibling temporary directory with a
fresh UUID suffix. The manifest binds:

- `model.joblib`
- `report.json`
- `policy.json`
- `predictions.csv`
- `audit.html`
- `monitoring.json`

The loader checks each digest, document binding, Python minor, dependency
version, class mapping, model feature names, policy contract, and monitoring
schema. It also binds the preprocessing quality sidecar digest when available.
The manifest is not signed and cannot protect against an attacker who rewrites
the entire bundle.

Other writers described as atomic use the same collision-free sibling temporary
path rule for preprocessing CSV and JSON, standalone JSON and monitoring output,
batch CSV, and the incomplete-study marker. Direct HTML export is not an atomic
writer.

## Material limitations

- Adult is a historical income benchmark, not a hiring-validity dataset.
- Income is not a defensible proxy for job performance or qualification.
- Group metrics do not prove individual fairness or causal discrimination.
- Post-processing cannot repair a missing construct, biased label, or harmful
  data-collection process.
- Binary and coarse demographic categories omit identities and within-group
  variation.
- Row bootstrap intervals do not cover dataset shift, model selection, or the
  Census sample design.
- Weighted sensitivity does not establish population-representative estimates.
- Feature-vector collisions do not identify people, but conflicts and
  cross-split overlap weaken simple independence assumptions.
- Removing repeated feature identities does not make the remaining slice an
  independent or externally representative dataset.
- Aggregate monitoring can expose descriptive shift, not prove root cause,
  representativeness, or operational safety.
- Review-band evaluation measures workload and automated error, not the quality
  of human decisions.
- No result in this repository establishes legal compliance or safe use in an
  employment process.

## Authors

Roberto Villafuerte and Charles Santhakumar, University of Helsinki
collaboration.
