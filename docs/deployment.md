# Local simulation operations

This repository provides a reproducible local service for inspecting one audit
bundle. It does not provide a production deployment path.

## Build a run bundle

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

Preprocessing also writes a digest-bound
`adult_model_ready.quality.json` sidecar. The experiment verifies it against the
CSV and embeds its data-semantics evidence in the report.

Run creation clones the supplied configuration before applying effective
command values. Model type and validation ratio are synchronized with the
resolved model and data sections. The run seed is synchronized with the global
seed, split seed, estimator `random_state`, and resolved model `random_state`.
The effective configuration is SHA-256 bound without mutating the caller's
configuration object.

The run is published atomically only after all seven files are written and their
cross-document contracts validate:

```text
manifest.json   identity, hashes, contracts, runtime, and gate verdict
model.joblib    fitted preprocessing and classifier pipeline
policy.json     global review band and offline evaluation policy
report.json     protocol, metrics, uncertainty, diagnostics, and verdict
predictions.csv held-out probabilities and paired policy outcomes
audit.html      portable visual evidence report
monitoring.json aggregate-only held-out drift reference
```

The bundle uses a hidden sibling temporary directory with a fresh UUID suffix,
then one `os.replace` publishes the destination. Other atomic outputs use the
same collision-free sibling-path rule for preprocessing CSV and JSON, standalone
JSON and monitoring files, batch CSV, and incomplete-study markers. Direct HTML
export is a separate convenience write and is not described as atomic.

The loader checks:

- schema versions and run IDs;
- SHA-256 digests for all bound files;
- data, data-quality, source, and resolved-config fingerprints;
- report and manifest agreement on effective model type, seed, the synchronized
  resolved configuration, and its digest;
- the exact ordered 11-feature contract and canonical numeric/categorical
  transformer assignments;
- fitted training vocabularies, `handle_unknown="ignore"`, transformed feature
  names, and the binary class mapping;
- policy kind, thresholds, review behavior, and protected-attribute flags;
- serving-policy agreement with selective-review evidence and offline-policy
  agreement with validation selection, held-out thresholds, and protocol;
- a fresh gate evaluation using the persisted threshold set, with report and
  manifest verdict agreement;
- report, data-semantics, and manifest agreement;
- monitoring snapshot schema and digest;
- Python minor version; and
- exact model-runtime dependency versions.

Serialized model bundles are runtime-specific. Build and load the bundle with
the same supported Python minor and dependency versions.

## Run locally

The service refuses a governance-rejected artifact by default. This is the
normal safe behavior. To inspect a rejected research run deliberately, set the
explicit override:

```bash
ALLOW_REJECTED_RESEARCH_BUNDLE=1 \
RUN_DIR=runs/local-xgb \
  uv run uvicorn fairness_project.inference.api:app \
  --host 127.0.0.1 \
  --port 8000
```

In another terminal:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/ready
curl --fail http://127.0.0.1:8000/v2/metadata
```

`/health` checks only the process. `/ready` requires a complete and permitted
artifact.

## Docker demonstration

Build an image, then mount one complete bundle read-only:

```bash
docker build \
  --build-arg PYTHON_VERSION=3.12 \
  -t fairness-audit-api:0.3.0 .

docker run --rm \
  --publish 127.0.0.1:8000:8000 \
  --env ALLOW_REJECTED_RESEARCH_BUNDLE=1 \
  --volume "$PWD/runs/local-xgb:/app/run:ro" \
  fairness-audit-api:0.3.0
```

The override is appropriate only for a deliberate evaluation demonstration.
Do not add it to a general-purpose environment configuration.

If the bundle is absent, incomplete, changed, incompatible, or not allowed,
liveness can still respond while readiness and simulation endpoints return
`503`.

## Batch simulation

HTTP and CSV use the same contract and `InferenceService`:

```bash
uv run fairness simulate \
  --run-dir runs/local-xgb \
  --input-csv examples/input.csv \
  --output-csv /tmp/local-xgb-simulation.csv \
  --allow-rejected-research-bundle
```

Batch output is written through a fresh UUID-suffixed sibling temporary file and
an atomic replace. Invalid input creates no partial result, and concurrent
writers do not share a temporary filename.

## Offline monitoring comparison

`monitoring.json` is a held-out reference, not a service process. Build a
current aggregate snapshot and compare it offline:

```bash
uv run fairness monitor snapshot \
  --input-csv offline/current.csv \
  --output-json offline/current.json \
  --feature-columns age,workclass,education,education_num,marital_status,occupation,relationship,native_country,capital_gain,capital_loss,hours_per_week \
  --categorical-columns workclass,education,marital_status,occupation,relationship,native_country \
  --score-column score \
  --prediction-column prediction \
  --protected-columns sex,race,race_binary

uv run fairness monitor compare \
  --reference-json runs/local-xgb/monitoring.json \
  --current-json offline/current.json \
  --output-json offline/comparison.json \
  --require-pass
```

The comparison reports `PASS`, `FAIL`, or `INSUFFICIENT_EVIDENCE`. It does not
run on a schedule or send alerts. See [`monitoring.md`](monitoring.md) for label,
weight, threshold, and evidence semantics.

Snapshot validation rejects internally contradictory aggregates before
publication, load, or comparison. This includes category shares and unknown
summaries that disagree with counts, prediction counts that disagree with
selection rate or confusion totals, protected-group totals that do not roll up,
and binary rates that cannot be derived from their stored confusion values.

## Operating boundary

Keep the service on localhost. It has no:

- valid employment target or job-specific validation;
- authentication, authorization, or transport-security termination;
- rate limiting, secrets management, retention policy, or incident response;
- privacy impact assessment or data-subject process;
- reviewer staffing, training, quality measurement, or conflict controls;
- accommodations, appeal, contestability, or adverse-action workflow; or
- drift response owner and accountable decision authority.

The API is useful because it makes artifact loading, request validation,
abstention, and decision provenance testable. Those engineering properties do
not turn the benchmark into an operational hiring system.
