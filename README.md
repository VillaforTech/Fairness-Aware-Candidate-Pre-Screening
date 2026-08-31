# Auditable Fair-ML Policy Lab

An evidence-first workbench for studying how a classification policy changes
utility, group disparities, uncertainty, and review workload.

The repository name is historical. The data is UCI Adult, a 1994 Census-income
benchmark, not a record of applicants or job performance. The lab is built to
make that boundary hard to miss: protected attributes stay in the audit plane,
the scoring contract has 11 non-protected features, and the local API is named
`simulate` rather than `predict`.

> **Evaluation boundary:** do not use this system for employment decisions. Its
> results cannot establish job relatedness, legal compliance, external validity,
> or the quality of a human review process.

[Read the project case study](https://www.villafortech.com/projects/fairness-aware-candidate-pre-screening/)

## The engineering question

Many fairness demos stop after printing one gap. This project asks a harder
question:

> Can one run produce enough evidence to inspect the policy choice, its
> uncertainty, its subgroup behavior, its stability, and the exact artifact
> that generated the result?

The answer is a versioned audit bundle, not a fairness badge.

## System at a glance

```text
UCI Adult raw files
        |
        v
deterministic cleaning + strict schema
        |\
        | `--> digest-bound data-semantics sidecar
        |
        +---------------------------+
        |                           |
        v                           v
original train                  official test
        |                           held once
        +--> fit rows
        |
        +--> joint-stratified validation
                 |
                 +--> train-vocabulary OOV audit
                 |
                 +--> train-overlap measurement
                 |        |
                 |        `--> overlap-excluded policy retuning evidence
                 |
                 +--> 2D group-threshold search
                 |        |
                 |        +--> accuracy / |TPR gap| Pareto frontier
                 |
                 +--> global probability review band
                          |
                          v
                  frozen policy choices
                          |
                          v
               paired held-out evaluation
                          |
        +-----------------+------------------+
        |                 |                  |
        v                 v                  v
 bootstrap intervals   intersectional     weighted sensitivity
                       evidence states    using fnlwgt
        |                 |                  |
        +-----------------+------------------+
                          |
                          v
          exact-overlap removal + novel-only metrics
                          |
                          v
                 fail-closed policy gate
                          |
                          v
              integrity-bound seven-file bundle
                          |
                          v
          aggregate monitoring reference + offline comparison
```

## Technical depth

| Layer | What is implemented |
|---|---|
| Data contract | Exact columns, dtype families, value domains, official test preservation, and joint stratification by label, sex, and binary race |
| Data semantics | Digest-bound raw and processed quality evidence covering 7.41% complete-case attrition, missingness by group, duplicate feature vectors, conflicting labels, cross-split overlap, and `fnlwgt` semantics |
| Feature contract | `adult-income-v2-no-census-weight`, with five ordered numeric fields, six ordered categorical fields, fitted training vocabularies, and `fnlwgt` excluded from predictors |
| Policy search | Exhaustive two-dimensional threshold search on validation data, a nondominated accuracy versus absolute TPR-gap frontier, explicit feasibility constraints, and deterministic tie-breaking |
| Abstention | One global probability-only review band selected on validation error and coverage, then frozen for held-out evaluation |
| Diagnostics | Performance, SPD, DI, TPR gap, FPR gap, sex by original-race cells, calibration, support counts, Wilson intervals, and explicit evidence states |
| Uncertainty | Paired label-and-group-stratified bootstrap intervals for baseline, adjusted, and change metrics |
| Robustness | Policy retuning after excluding validation rows repeated in train, fixed-policy held-out metrics after excluding rows repeated in train or validation, and repeated-seed studies with strict comparability checks |
| Governance | Strict report schema, point and interval checks, intersectional span limits, review-policy checks, and distinct pass, reject, and malformed-report exit codes |
| Artifact integrity | Collision-free sibling temporary paths for atomic writers, SHA-256 binding for model, report, policy, predictions, HTML, and monitoring, plus fresh gate, policy-evidence, runtime, and feature-contract validation on load |
| Offline monitoring | Strict aggregate-only snapshots whose derived counts, shares, confusion totals, and rates are internally validated before descriptive drift comparison |
| Interface | A v2 local simulation API that rejects protected attributes, Census weights, extra fields, nulls, and unseen categories |

## Reproduce one audit

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
uv sync --locked --extra dev --extra api

uv run fairness data preprocess \
  --input-dir data/raw/adult \
  --output-path data/processed/adult/adult_model_ready.csv

uv run fairness audit \
  --model xgb \
  --seed 42 \
  --data-path data/processed/adult/adult_model_ready.csv \
  --output-dir runs \
  --run-id xgb-seed-42 \
  --bootstrap-samples 500
```

Preprocessing writes
`data/processed/adult/adult_model_ready.quality.json` beside the CSV. The
sidecar binds both raw source files and the model-ready CSV by SHA-256. The audit
command verifies that binding, recomputes the processed-data evidence, and
embeds the data-semantics record in `report.json`.

The audit command does not hide a policy rejection. Add `--require-gate-pass`
when a rejected result must fail the calling job.

Command overrides are applied to a cloned configuration. The effective model
type and validation ratio replace their resolved-config fields, while the run
seed is synchronized across the global seed, split seed, estimator
`random_state`, and `model.random_state`. The caller's configuration object is
not mutated. The effective resolved configuration and its SHA-256 are recorded
in both report and manifest evidence.

Every completed run is one self-contained directory:

```text
runs/xgb-seed-42/
|-- manifest.json      artifact identity, contracts, hashes, runtime, verdict
|-- model.joblib       fitted preprocessing and classifier pipeline
|-- policy.json        global review band plus offline group-threshold policy
|-- predictions.csv    held-out probabilities and paired policy outcomes
|-- report.json        full protocol, metrics, uncertainty, and diagnostics
|-- audit.html         portable, responsive, print-ready evidence report
`-- monitoring.json    aggregate-only held-out drift reference
```

Recheck the gate or rebuild the HTML without retraining:

```bash
uv run fairness gate --report runs/xgb-seed-42/report.json

uv run fairness render-report \
  --run-dir runs/xgb-seed-42 \
  --output /tmp/xgb-seed-42-audit.html
```

The loader validates the complete bundle before rendering. It re-evaluates the
report with the persisted gate thresholds, requires that verdict to match both
report and manifest, and cross-checks the serving and offline policies against
their report selection evidence. Editing one bound file invalidates the
artifact.

## Run a stability study

A single seed is not a robustness result. The study command repeats the full
fit, validation policy search, and held-out evaluation for each seed:

```bash
uv run fairness study \
  --model xgb \
  --seeds 0,1,2,3,4 \
  --data-path data/processed/adult/adult_model_ready.csv \
  --output-dir studies \
  --study-id xgb-five-seed \
  --bootstrap-samples 100
```

`stability.json` records distributions and worst cases only after confirming
that data, source, model type and parameters, resolved configuration apart from
per-run seed fields, protocol, feature contract, gate threshold policy, and
metric coverage match across runs. The seeds share one official test partition,
so this is a sensitivity study, not five independent population samples.

## Compare offline monitoring snapshots

Every run includes `monitoring.json`, an aggregate reference built from the
official held-out rows, baseline scores and predictions, delayed labels, audit
groups, and `fnlwgt` sensitivity. It contains summaries and 101-point quantile
sketches, never source rows.

Create a current snapshot from an offline audit CSV with the same declared
feature and outcome roles:

```bash
uv run fairness monitor snapshot \
  --input-csv offline/current.csv \
  --output-json offline/current.json \
  --feature-columns age,workclass,education,education_num,marital_status,occupation,relationship,native_country,capital_gain,capital_loss,hours_per_week \
  --categorical-columns workclass,education,marital_status,occupation,relationship,native_country \
  --score-column score \
  --prediction-column prediction \
  --protected-columns sex,race,race_binary \
  --label-column label \
  --sample-weight-column fnlwgt

uv run fairness monitor compare \
  --reference-json runs/xgb-seed-42/monitoring.json \
  --current-json offline/current.json \
  --output-json offline/comparison.json \
  --require-pass
```

The comparison checks numeric PSI and a quantile-based KS-like distance,
categorical total variation and OOV share, score and selection drift, protected
group composition, and delayed-label performance and group gaps when labels are
available. Snapshot validation also rejects derived aggregates that contradict
their source aggregates, including category shares, unknown counts, prediction
counts, class and confusion totals, group totals, or derived rates. `PASS` means
every configured offline threshold had enough evidence and held. `FAIL` means
at least one threshold was violated.
`INSUFFICIENT_EVIDENCE` means no violation was detected but row, group, label,
or estimability requirements were not met. See
[`docs/monitoring.md`](docs/monitoring.md).

## Policy design

### Offline Pareto frontier

The policy optimizer enumerates every privileged and unprivileged threshold
pair on the configured grid. It keeps the nondominated accuracy versus absolute
TPR-gap frontier, then chooses the highest-accuracy feasible point under the
validation constraints. Ties prefer smaller TPR, FPR, and selection-rate gaps,
then less movement from the global threshold.

This policy needs protected-group values. It is evaluated offline and is never
served by the API.

### Global review band

A separate policy places a symmetric band around the global probability
threshold. Scores inside the band become `manual_review_required`; scores
outside it receive an automatic positive or negative simulation decision. The
band is selected on validation labels to maximize automation coverage under a
configured automated-error limit and minimum automated sample count.

The held-out report measures coverage, error, and review burden by observed
group. It does not assume that review is correct, unbiased, available, or
legally appropriate.

### Validation-overlap dependence

The report measures how many validation rows exactly repeat the 11-feature
identity of a fit row, and separately counts exact full-record overlap. It then
repeats both policy searches on validation rows that do not overlap train. The
model and validation probabilities remain fixed. This retuning is sensitivity
evidence only and does not replace the policies used for held-out evaluation.
If no rows remain or a selector cannot be estimated, the report records
`not_estimable` with a reason.

### Exact-feature overlap sensitivity

The report identifies held-out rows whose complete 11-feature identity also
appears in train or validation. It then recomputes baseline and adjusted
metrics on the overlap-excluded slice using the same frozen probabilities,
predictions, and thresholds. Exact tuple equality is the membership test, and
labels, protected attributes, split markers, and weights do not define
identity. No policy is retuned in this held-out analysis. This exposes
dependence on repeated benchmark records without
pretending that the remaining slice is an external dataset.

## Experimental policy gate

| Check | Default |
|---|---:|
| Adjusted accuracy | at least 0.80 |
| Accuracy drop from baseline | at most 0.02 |
| Absolute TPR gap | at most 0.05 |
| Absolute FPR gap | at most 0.05 |
| Disparate impact | 0.80 to 1.25 |
| Absolute SPD | at most 0.10 |
| Intersectional TPR span | at most 0.10 |
| Intersectional FPR span | at most 0.10 |
| Held-out automated error | preserve the validation-selected limit |

When bootstrap intervals exist, the gate also checks their worst bounds for
TPR gap, FPR gap, SPD, and DI. Small intersectional cells are marked `limited`
or `not_estimable`; they are not converted to zero or silently removed.

These defaults are engineering criteria for this repository. They are not
universal fairness rules or legal thresholds. Exit codes are `0` for pass, `1`
for a valid report that is rejected, and `2` for malformed input or invalid
configuration.

## Local simulation API

The service refuses a governance-rejected bundle by default. For deliberate
local inspection of such a research artifact, use the explicit override:

```bash
ALLOW_REJECTED_RESEARCH_BUNDLE=1 \
RUN_DIR=runs/xgb-seed-42 \
  uv run uvicorn fairness_project.inference.api:app \
  --host 127.0.0.1 \
  --port 8000
```

Routes:

- `GET /health`
- `GET /ready`
- `GET /v2/metadata`
- `POST /v2/simulate`
- `POST /v2/simulate-batch`

The request contains exactly these 11 features:

```text
age, workclass, education, education_num, marital_status, occupation,
relationship, native_country, capital_gain, capital_loss, hours_per_week
```

`sex`, `race`, `race_binary`, and `fnlwgt` are not accepted. The API applies the
global review band only, never the offline group thresholds. See
[`docs/api_spec.md`](docs/api_spec.md) for the full contract.

The fitted one-hot encoder intentionally ignores a validation or test category
that was not observed in fit rows, representing that feature's category block
with zeros so evaluation can continue. Each run records validation and test OOV
counts, values, affected rows, and shares against the training vocabulary. The
simulation service uses that same fitted vocabulary but rejects OOV input before
calling the model.

Batch simulation uses the same service:

```bash
uv run fairness simulate \
  --run-dir runs/xgb-seed-42 \
  --input-csv examples/input.csv \
  --output-csv /tmp/adult-simulation.csv \
  --allow-rejected-research-bundle
```

## Where results live

Run-specific numbers belong to the generated `report.json`, `audit.html`,
`monitoring.json`, and system card. They are tied to data and data-quality
digests, a source digest, resolved config, model parameters, seed, and runtime.
This avoids presenting an old README table as if it described new code.

Every writer described as atomic uses a hidden temporary path beside its final
destination with a fresh UUID suffix, followed by `os.replace`. This applies to
preprocessed CSV and JSON evidence, JSON and monitoring outputs, batch results,
complete bundle directories, and the incomplete-study marker. A direct HTML
export is not described as atomic.

The manifest is an integrity check, not a signature. An attacker who can rewrite
the entire bundle can also rewrite its hashes.

## Development checks

```bash
uv sync --locked --extra dev --extra api
uv run pytest tests -q -ra
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy --config-file pyproject.toml
uv build
```

## Repository map

```text
src/fairness_project/
|-- cli.py                 supported command surface
|-- experiment.py          canonical single-run orchestration
|-- study.py               repeated-seed orchestration
|-- data/                  preparation, semantics audit, schema, split contract
|-- features/              11-feature preprocessing pipeline
|-- models/                training and immutable artifact bundles
|-- fairness/              frontier search and selective review
|-- evaluation/            metrics, uncertainty, intersections, overlap, stability
|-- governance/            strict report policy gate
|-- inference/             shared service, v2 API, CSV simulation
|-- monitoring/            aggregate snapshot and offline drift comparison
`-- reporting/             self-contained HTML evidence report
```

Design details are in [`docs/architecture.md`](docs/architecture.md), the
evaluation protocol is in [`docs/methodology.md`](docs/methodology.md), and the
interpretation contract is in
[`docs/responsible_ai.md`](docs/responsible_ai.md). The offline monitoring
contract is in [`docs/monitoring.md`](docs/monitoring.md).

## Primary sources

- [UCI Adult dataset and citation](https://archive.ics.uci.edu/dataset/2/adult)
- [Equal Opportunity in Supervised Learning](https://proceedings.neurips.cc/paper_files/paper/2016/hash/9d2682367c3935defcb1f9e247a97c0-Abstract.html)
- [Statistical Inference for Fairness Auditing](https://jmlr.org/papers/v25/23-0739.html)
- [NIST AI Risk Management Framework 1.0](https://doi.org/10.6028/NIST.AI.100-1)
- [EEOC: Employment Tests and Selection Procedures](https://www.eeoc.gov/laws/guidance/employment-tests-and-selection-procedures)

The full annotated reading list is in
[`docs/references.md`](docs/references.md).

## Authors and attribution

Developed by **Roberto Villafuerte** and **Charles Santhakumar** during
Trustworthy Machine Learning studies at the University of Helsinki.

Dataset citation:

> Becker, B., & Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning
> Repository. <https://doi.org/10.24432/C5XW20>

Machine-readable citation metadata is in [`CITATION.cff`](CITATION.cff).

## License

No software license has been granted. Both contributors must agree before
redistribution terms are added. The bundled UCI Adult data has separate terms;
see [`docs/data.md`](docs/data.md).
