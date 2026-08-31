# Repository guidance

## Project contract

This repository is the Auditable Fair-ML Policy Lab. It studies policy trade-offs on
the UCI Adult Census-income benchmark. Adult is not applicant or job-performance
data, and no result here validates an employment decision system.

Only `src/fairness_project/` is supported. Historical notebooks, flat scripts, and
alternate training paths were removed in v0.3 and remain recoverable from the
`coursework-v0.2` tag.

## Required commands

```bash
uv sync --locked --extra dev --extra api
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy --config-file pyproject.toml
uv run pytest tests -q -ra
uv build
```

Real workflow:

```bash
uv run fairness data preprocess \
  --input-dir data/raw/adult \
  --output-path data/processed/adult/adult_model_ready.csv

uv run fairness audit \
  --model xgb \
  --seed 42 \
  --data-path data/processed/adult/adult_model_ready.csv \
  --output-dir runs \
  --run-id xgb-seed-42

uv run fairness study \
  --model xgb \
  --seeds 0,1,2,3,4 \
  --data-path data/processed/adult/adult_model_ready.csv \
  --output-dir studies \
  --study-id xgb-five-seed
```

## Architecture invariants

- Preserve the official UCI test partition. Fit the model on fit rows and select
  policies on validation rows only.
- The model contract has exactly 11 features. Sex, race, `race_binary`, `fnlwgt`,
  target, and split markers never enter the scoring plane.
- `fnlwgt` is the Census final sampling weight. Use it only for clearly labelled
  weighted sensitivity, never as a predictor or primary gate input.
- The two-dimensional group-threshold Pareto policy is offline-only. The simulator
  accepts no protected attributes and serves only the global review band.
- Keep undefined and small-cell metrics explicit. Never coerce a missing denominator
  to zero or hide a limited evidence state.
- Exact-overlap sensitivity compares canonical feature tuples, excludes labels and
  audit fields from identity, and never retrains or retunes on held-out rows.
- The governance gate distinguishes pass, valid rejection, and malformed evidence.
  Preserve failed gates as results.
- Every run publishes atomically and binds seven files: `model.joblib`,
  `manifest.json`, `policy.json`, `predictions.csv`, `report.json`, `audit.html`, and
  aggregate-only `monitoring.json`.
- A rejected run requires an explicit research override for local simulation.
- Do not describe benchmark parity as legal compliance, job relatedness, external
  validity, causal fairness, or deployment safety.

## Module map

| Module | Responsibility |
|---|---|
| `data/` | UCI loading, raw attrition evidence, strict schema, and split contract |
| `features/` | Exact 11-feature preprocessing |
| `models/` | Classifier training and integrity-bound artifacts |
| `fairness/` | Offline Pareto search and global selective review |
| `evaluation/` | Metrics, bootstrap, intersectional evidence, exact overlap, and stability |
| `governance/` | Strict report validation and fail-closed criteria |
| `monitoring/` | Aggregate snapshots and tri-state offline drift comparison |
| `inference/` | Shared simulation service, v2 HTTP API, and CSV path |
| `reporting/` | Self-contained HTML evidence report |

## Evidence updates

Generate reference evidence from a clean source commit so `git_commit`,
`dirty_worktree`, and `source_sha256` remain meaningful. Commit source first, run the
study outside the checkout, then commit the curated evidence in a second commit. A
manifest hash is an integrity check, not a cryptographic signature.
