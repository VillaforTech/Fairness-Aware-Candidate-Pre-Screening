# Fairness-Aware Candidate Pre-Screening

A reproducible audit of fairness interventions on the UCI Adult dataset.

The repository keeps its original name, but its claim is deliberately precise:
this is a reproducible study of model evaluation,
group-threshold post-processing, and governance checks. Adult contains 1994
census-income records. It contains neither applicants nor job-performance
labels, so it cannot validate a hiring system.

> **Not for employment decisions.** A lower fairness gap, or even a passing
> configurable gate, would not establish job relatedness, legal compliance,
> external validity, or safety.

[Read the project note on Roberto's website](https://www.villafortech.com/projects/fairness-aware-candidate-pre-screening/).

## What changed in v0.2

- One installed command now owns training, validation-only threshold tuning,
  test evaluation, reporting, model persistence, and the policy verdict.
- Validation is jointly stratified by target, sex, and binary race while the
  official UCI test partition remains untouched.
- Every run records data and package-source hashes, dependency versions, split
  cell counts, subgroup denominators, and policy thresholds. Source checkouts
  also record their Git commit and dirty state; installed distributions record
  that Git state is unavailable instead of inspecting the caller's repository.
- The gate rejects missing, malformed, non-finite, Boolean, or out-of-domain
  metrics and checks accuracy loss as well as absolute accuracy.
- API and batch inference load the same validated bundle and state the exact
  decision policy in every output.
- The API serves a global 0.5 threshold only. The sex-specific threshold
  experiment remains offline and is never described as deployed behavior.
- A locked environment, package-integrity check, real-artifact API test, and
  generated-report CI job replace mock-only confidence.

## Reproduce the reference audit

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

The command writes a self-contained local bundle:

```text
runs/xgb-seed-42/
├── manifest.json      # hashes, feature contract, versions, gate verdict
├── model.joblib       # fitted 12-feature pipeline
├── policy.json        # served baseline policy + offline EO policy
├── predictions.csv    # held-out rows and paired predictions
└── report.json        # metrics, cells, intervals, protocol, provenance
```

Run the policy checker again without retraining:

```bash
uv run fairness gate --report runs/xgb-seed-42/report.json
```

Exit `0` means the configured checks passed. Exit `1` means the report was
valid but one or more policy criteria failed. Exit `2` means the report or gate
configuration was malformed.

## Reference result

The checked report at
[`reports/reference/xgb-seed-42-v1/report.json`](reports/reference/xgb-seed-42-v1/report.json)
was generated on 2026-08-28 with seed 42 and the locked v0.2 environment.
Protected attributes were excluded from model features.

Split sizes:

- fit: 25,637
- validation: 4,525
- preserved test: 15,060

| Metric | Baseline | Offline adjusted | Adjusted 95% interval |
|---|---:|---:|---:|
| Accuracy | 0.8694 | 0.8678 | [0.8630, 0.8728] |
| SPD | 0.1754 | 0.1563 | [0.1468, 0.1670] |
| DI | 0.3400 | 0.4120 | [0.3817, 0.4370] |
| TPR gap (Male − Female) | 0.0504 | -0.0124 | [-0.0560, 0.0268] |

The validation-selected thresholds were `0.500` for Male and `0.405` for
Female. Accuracy decreased by `0.0016`. The paired bootstrap estimates
test-sample uncertainty while preserving label/group cell sizes; it does not
measure sensitivity to a different validation split or a different population.

### Verdict: rejected by the default policy

The adjusted run passes the minimum accuracy, maximum accuracy loss, and TPR-gap
checks. It fails:

- `DI=0.4120`, required range `[0.80, 1.25]`
- `|SPD|=0.1563`, required maximum `0.10`

Reducing one disparity measure does not make the benchmark acceptable. The
failure is persisted in both `report.json` and the local artifact manifest.

## Evaluation design

```text
Bundled raw UCI files
        │
        ▼
clean + hash data
        │
        ├── official test partition ───────────────────────────────┐
        │                                                          │
        └── original train                                         │
              ├── fit rows ──► fit preprocessing + classifier      │
              └── joint-stratified validation                      │
                            │                                      │
                            └── tune one-sided opportunity policy  │
                                                                   ▼
                                     frozen thresholds ──► paired test evaluation
                                                                   │
                                      report + intervals + gate ◄──┘
```

Threshold selection uses labels from validation only. Test labels are used once
for final evaluation. A tie between thresholds with the same TPR error resolves
to the threshold closest to the base policy, avoiding the old `0.0` first-match
failure. Undefined subgroup TPRs, unknown groups, invalid probabilities, and
misaligned arrays fail closed.

## Experimental policy gate

| Check | Default |
|---|---:|
| Adjusted accuracy | at least 0.80 |
| Accuracy drop from baseline | at most 0.02 |
| Absolute TPR gap | at most 0.05 |
| Disparate impact | 0.80 to 1.25 |
| Absolute SPD | at most 0.10 |

These values are configurable engineering checks, not universal ethical or
legal rules. The report schema is versioned and requires run ID, seed, model
type, source hash, and data hash before the gate can pass. It also requires a
checkout-anchored Git commit and dirty flag, or the explicit unavailable/null
pair used by installed distributions.

## Local inference boundary

First create a run bundle. Then start the local demo:

```bash
RUN_DIR=runs/xgb-seed-42 \
  uv run uvicorn fairness_project.inference.api:app \
  --host 127.0.0.1 \
  --port 8000
```

Useful routes:

- `GET /health`: process liveness
- `GET /ready`: validated bundle readiness
- `GET /v1/metadata`: artifact, gate, and decision-policy provenance
- `POST /v1/predict`: one baseline Adult-income prediction
- `POST /v1/predict-batch`: 1 to 1,000 baseline predictions

Prediction responses include `artifact_id`, `decision_policy`, and
`decision_threshold`. The input contract forbids extra fields, including sex
and race. The API therefore cannot silently apply the offline group thresholds.

Batch inference uses the same service:

```bash
uv run fairness predict \
  --run-dir runs/xgb-seed-42 \
  --input-csv path/to/adult-features.csv \
  --output-csv /tmp/adult-predictions.csv
```

Invalid or incomplete input produces no output file.

## Development checks

```bash
uv sync --locked --extra dev --extra api
uv run pytest tests -q -ra
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy --config-file pyproject.toml
uv build
```

CI runs all tests on Python 3.10, 3.11, and 3.12 with API dependencies present.
It also installs the built wheel outside the checkout, executes a real audit to
verify package provenance, and generates an LR report whose expected DI/SPD
rejection must be observed. Serialized artifacts remain bound to the Python
minor and exact model-library versions recorded in their manifest.

## Repository map

```text
src/fairness_project/
├── cli.py                  # supported commands
├── experiment.py           # canonical run orchestration
├── provenance.py           # package hash + checkout-anchored revision state
├── data/                   # raw-data preparation, schema, stratified split
├── features/               # 12-feature preprocessing contract
├── models/                 # training, artifact validation, run discovery
├── fairness/               # offline threshold tuning and application
├── evaluation/             # metrics, subgroup cells, intervals, model card
├── governance/             # strict report policy gate
├── inference/              # shared service, HTTP API, batch path
└── monitoring/             # experimental drift utilities
```

The older flat modules under `src/models`, `src/metrics`, `src/techniques`, and
`src/main*.py` remain only as historical prototype code. The installed CLI, current
tests, API, and reference report do not import them.

## Limits that remain

- One seed and one validation draw do not establish threshold stability.
- The reference uncertainty intervals do not cover dataset shift or model
  selection uncertainty.
- `race_binary` collapses heterogeneous identities; intersectional cells are
  diagnostic, not a complete fairness analysis.
- `fnlwgt` remains a predictor in this benchmark. A weighted sensitivity study
  and a version excluding it are still needed.
- No causal, construct-validity, user, legal, or deployment study exists.
- The local API has no authentication, authorization, or production security
  review.

## Authors and attribution

This work was developed by **Roberto Villafuerte** and **Charles Santhakumar**
during Trustworthy Machine Learning studies at the University of Helsinki.

Dataset citation:

> Becker, B., & Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning
> Repository. <https://doi.org/10.24432/C5XW20>

Machine-readable software citation metadata is in [`CITATION.cff`](CITATION.cff).

## License

No software license has been granted. The two contributors must agree before
redistribution terms are added. The bundled UCI Adult data is separately
available under CC BY 4.0; see [`docs/data.md`](docs/data.md).
