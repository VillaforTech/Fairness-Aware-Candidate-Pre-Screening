# CLAUDE.md

This file provides repository guidance for coding agents.

## Project Overview

Reproducible fairness benchmark audit using the Adult (Census Income) dataset. It evaluates an offline Equal Opportunity post-processing policy with a leakage-free protocol. Protected attributes are sex and binary race. Adult is not applicant or job-performance data, so this repository does not validate a hiring system and must not be used for employment decisions.

## Common Commands

```bash
# Install the locked development environment
uv sync --locked --extra dev --extra api

# Run tests (PYTHONPATH is configured in pyproject.toml)
uv run pytest tests -q -ra

# Run a single test file
uv run pytest tests/test_fairness_metrics.py -v

# Run a single test
uv run pytest tests/test_fairness_metrics.py::test_function_name -v

# Test with coverage
uv run pytest tests --cov=src/fairness_project --cov-report=html

# Lint and format
uv run ruff check src tests
uv run ruff format --check src tests

# Type check
uv run mypy --config-file pyproject.toml

# CLI
uv run fairness --help
uv run fairness data preprocess --input-dir data/raw/adult --output-path data/processed/adult/adult_model_ready.csv
uv run fairness audit --model xgb --seed 42 --data-path data/processed/adult/adult_model_ready.csv --output-dir runs --run-id xgb-seed-42
uv run fairness gate --report runs/xgb-seed-42/report.json

# API server
RUN_DIR=runs/xgb-seed-42 uv run uvicorn fairness_project.inference.api:app --host 127.0.0.1 --port 8000

# Docker
docker compose up --build
```

## Architecture

The project has two code paths under `src/`: historical flat prototype modules and the supported `src/fairness_project/` package. Only the supported package is built, tested, and used by the CLI.

### Modern Package (`src/fairness_project/`)

| Module | Purpose |
|--------|---------|
| `config.py` | Dataclass + Pydantic config system, loads from `configs/default.yaml`, global singleton |
| `cli.py` | Typer-based CLI entry point (registered as `fairness` console script) |
| `data/` | UCI Adult dataset download and preprocessing |
| `features/` | Sklearn preprocessing pipeline (StandardScaler + OneHotEncoder) |
| `models/` | Train LR/RF/XGBoost; validated artifact persistence |
| `metrics/fairness.py` | SPD, Disparate Impact, TPR/FPR gap calculations |
| `metrics/performance.py` | Accuracy, precision, recall, F1 |
| `fairness/postprocess.py` | Equal Opportunity threshold tuning per group |
| `evaluation/` | Diagnostics, uncertainty intervals, and report generation |
| `inference/` | FastAPI REST API + batch prediction with Pydantic schemas |
| `monitoring/drift.py` | PSI, KS statistic, fairness drift detection |

### Data Flow

```
Raw data -> preprocess -> split (fit/validation/preserved test)
-> fit preprocessing and classifier -> validation-only policy tuning
-> frozen policy -> paired test evaluation -> report and gate verdict
-> validated bundle -> baseline-only API and batch inference
```

### Critical: Leakage-Free Protocol

EO thresholds are tuned **only on the validation set**. Final metrics are computed on the held-out test set. This prevents optimistic bias in fairness metric reporting. See `evaluation/leakage_free.py`.

## Configuration

YAML-based config at `configs/default.yaml`. Nested dataclasses cover data, model, fairness, and output settings. Seed 42 is the default and is propagated through the supported pipeline.

## CI

GitHub Actions runs lint, formatting, MyPy, tests on Python 3.10/3.11/3.12, package-integrity checks, an installed-wheel audit, a real report with expected policy rejection, and a container readiness probe.

## Key Constraints

- **Offline post-processing only**: mitigation is threshold adjustment, not in-processing
- **Models**: Logistic Regression, Random Forest, XGBoost
- **Fairness metrics**: SPD (ideal: 0), Disparate Impact (ideal: 1.0), TPR Gap (ideal: 0)
- **Serving boundary**: the API uses one global 0.5 threshold and never applies group-specific thresholds
- **Evidence boundary**: lower benchmark gaps do not establish job relatedness, legal compliance, external validity, or safety
- **Ruff config**: line length 100, Python 3.10+ target, rules E/W/F/I/B/C4/UP
