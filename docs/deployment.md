# Local demo operations

This project supports a reproducible local demo, not a production deployment.
The API loads one fail-closed run bundle and serves the baseline global
threshold. It never serves the offline sex-specific threshold intervention.

## Build a complete run bundle

```bash
uv sync --locked --python 3.12 --extra dev --extra api

uv run fairness data preprocess \
  --input-dir data/raw/adult \
  --output-path data/processed/adult/adult_model_ready.csv

uv run fairness audit \
  --model xgb \
  --seed 42 \
  --data-path data/processed/adult/adult_model_ready.csv \
  --output-dir runs \
  --run-id local-xgb \
  --bootstrap-samples 500
```

A complete `runs/local-xgb/` directory contains:

```text
manifest.json   artifact identity, hashes, feature contract, and gate result
model.joblib    fitted preprocessing and classifier pipeline
policy.json     served global policy and offline experimental policy
report.json     protocol, metrics, subgroup cells, intervals, and verdict
predictions.csv held-out rows and both prediction variants
```

The loader checks required files, schema versions, hashes, run IDs, the exact
12-feature contract, Python minor version, model-library versions, class
mapping, and probability output before readiness.

The application code is tested on Python 3.10–3.12, but serialized bundles are
runtime-specific. The default recipe deliberately creates and serves the bundle
with Python 3.12. A bundle produced with another supported Python minor must be
served by an image built with that same minor.

## Run and smoke-test locally

```bash
RUN_DIR=runs/local-xgb \
  uv run uvicorn fairness_project.inference.api:app \
  --host 127.0.0.1 \
  --port 8000
```

In another terminal:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/ready
curl --fail http://127.0.0.1:8000/v1/metadata
```

`/health` is liveness. `/ready` is the artifact-integrity check and should be
used for traffic readiness.

## Docker demo

Build the image, then mount a complete bundle read-only:

```bash
docker build \
  --build-arg PYTHON_VERSION=3.12 \
  -t fairness-audit-api:0.2.0 .

docker run --rm \
  --publish 127.0.0.1:8000:8000 \
  --volume "$PWD/runs/local-xgb:/app/run:ro" \
  fairness-audit-api:0.2.0
```

Compose uses `FAIRNESS_RUN_DIR` for the same read-only mount:

```bash
FAIRNESS_PYTHON_VERSION=3.12 \
FAIRNESS_RUN_DIR="$PWD/runs/local-xgb" \
  docker compose up --build
```

For example, serving a bundle generated with Python 3.11 requires
`--build-arg PYTHON_VERSION=3.11`, or `FAIRNESS_PYTHON_VERSION=3.11` with
Compose. A mismatched image is expected to remain unready rather than attempt an
unsafe cross-runtime load.

If the bundle is absent, incomplete, incompatible, corrupted, or does not match
the digests recorded in its manifest, liveness still responds but readiness and
prediction endpoints return `503`. The manifest is not signed, so these checks
do not protect against an attacker who can rewrite the entire bundle.

## Batch path

HTTP and CSV inference share the same service and policy:

```bash
uv run fairness predict \
  --run-dir runs/local-xgb \
  --input-csv examples/input.csv \
  --output-csv predictions/local-xgb.csv
```

Batch output is written atomically. Invalid input produces no partial result.

## Deliberate non-production boundary

Do not expose this service to real users or use its output for employment
decisions. It lacks a valid hiring target, prospective validation,
authentication, authorization, TLS termination, rate limiting, secrets
management, monitoring, incident response, privacy controls, appeal paths, and
an accountable operating process. A passing experiment gate would not close
those gaps.
