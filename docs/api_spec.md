# Local simulation API

The v2 HTTP interface runs one validated Adult-income audit bundle. It is a
policy simulator for inspecting artifact and decision behavior. It is not an
employment or applicant-ranking API.

The API applies one global probability review band. The offline group-threshold
policy is never loaded into the request path, and protected attributes are not
part of the request schema.

## Start the service

Create a complete run bundle first. See
[`deployment.md`](deployment.md#build-a-run-bundle).

The service rejects a governance-rejected bundle unless the caller makes the
research override explicit:

```bash
ALLOW_REJECTED_RESEARCH_BUNDLE=1 \
RUN_DIR=runs/local-xgb \
  uv run uvicorn fairness_project.inference.api:app \
  --host 127.0.0.1 \
  --port 8000
```

Swagger UI is available at <http://127.0.0.1:8000/docs> and ReDoc at
<http://127.0.0.1:8000/redoc>.

## Readiness and provenance

### `GET /health`

Process liveness only:

```json
{"status": "ok"}
```

A `200` response does not mean that an artifact is loaded.

### `GET /ready`

Returns `200` after the complete bundle passes schema, digest, runtime,
model-class, canonical numeric/categorical transformer, fitted-vocabulary,
policy-to-report, fresh persisted-threshold gate, and aggregate
monitoring-snapshot validation.

```json
{
  "ready": true,
  "artifact_id": "local-xgb"
}
```

Missing, incompatible, corrupted, or governance-rejected bundles return `503`.
The explicit research override changes only the last condition.

`monitoring.json` is required and hash-bound even though the API does not expose
a monitoring endpoint. Snapshot creation and comparison remain explicit offline
CLI operations.

### `GET /v2/metadata`

The response names the artifact and the global policy actually used by the
service.

| Field | Meaning |
|---|---|
| `artifact_id` | Run ID bound across manifest, policy, and report |
| `schema_version` | Artifact schema version |
| `model_type` | `lr`, `rf`, or `xgb` |
| `created_at` | Run creation time |
| `decision_policy` | Global review-band ID, base threshold, lower and upper bounds, review label, and selection scope |
| `governance` | Persisted gate verdict and violations |
| `evaluation_only` | Always `true` |
| `api_version` | `v2` |

The reported model type is the effective run value. Run creation clones the
input configuration, synchronizes that model type, validation ratio, seed, and
model `random_state`, and hashes the resulting resolved configuration. Bundle
loading requires report and manifest copies of that provenance to agree.

## Exact 11-feature request

```json
{
  "age": 35,
  "workclass": "Private",
  "education": "Bachelors",
  "education_num": 13,
  "marital_status": "Married-civ-spouse",
  "occupation": "Exec-managerial",
  "relationship": "Husband",
  "native_country": "United-States",
  "capital_gain": 5000,
  "capital_loss": 0,
  "hours_per_week": 40
}
```

| Field | Type | Constraint |
|---|---|---|
| `age` | integer | 0 to 120 |
| `workclass` | string | nonblank and observed during training |
| `education` | string | nonblank and observed during training |
| `education_num` | integer | 1 to 20 |
| `marital_status` | string | nonblank and observed during training |
| `occupation` | string | nonblank and observed during training |
| `relationship` | string | nonblank and observed during training |
| `native_country` | string | nonblank and observed during training |
| `capital_gain` | integer | at least 0 |
| `capital_loss` | integer | at least 0 |
| `hours_per_week` | integer | 0 to 168 |

Missing fields, extra fields, nulls, blank categories, Boolean numerics,
non-integer numerics, out-of-range values, and unseen categories are rejected.

The canonical numeric fields are `age`, `education_num`, `capital_gain`,
`capital_loss`, and `hours_per_week`. The canonical categorical fields are
`workclass`, `education`, `marital_status`, `occupation`, `relationship`, and
`native_country`. Their transformer assignments and order are validated when
the bundle loads.

`sex`, `race`, `race_binary`, and `fnlwgt` are deliberately absent. This keeps
protected attributes and the Census final weight outside the simulation
contract.

Evaluation and serving intentionally differ for OOV categories. The fitted
one-hot encoder uses `handle_unknown="ignore"`, so validation or test OOV values
produce an all-zero block for that categorical field. The run records OOV
values, affected rows, and shares separately for validation and test. The API
and CSV simulator derive their accepted vocabularies from that same fitted
encoder and reject OOV strings before `predict_proba` runs.

## `POST /v2/simulate`

The body is one exact 11-feature request. The response fields are:

| Field | Meaning |
|---|---|
| `prediction` | `0` or `1` for an automatic decision, otherwise `null` |
| `decision` | `auto_negative`, `auto_positive`, or `manual_review_required` |
| `probability` | Model probability for the Adult `>50K` label |
| `label` | `<=50K`, `>50K`, or `manual_review_required` |
| `decision_threshold` | Global base threshold |
| `review_lower_threshold` | Lower edge of the frozen review band |
| `review_upper_threshold` | Upper edge of the frozen review band |
| `decision_policy` | Global policy ID from the bundle |
| `artifact_id` | Loaded run ID |

The probability estimates the Adult income label under this benchmark. It is
not a probability of qualification, suitability, or job success.

## `POST /v2/simulate-batch`

The body has one `instances` array containing 1 to 1,000 exact request objects:

```json
{
  "instances": [
    {
      "age": 35,
      "workclass": "Private",
      "education": "Bachelors",
      "education_num": 13,
      "marital_status": "Married-civ-spouse",
      "occupation": "Exec-managerial",
      "relationship": "Husband",
      "native_country": "United-States",
      "capital_gain": 5000,
      "capital_loss": 0,
      "hours_per_week": 40
    }
  ]
}
```

All rows are validated before inference. The response contains the same output
objects in a `predictions` array.

## Error contract

| Status | Meaning |
|---:|---|
| `200` | Request completed |
| `422` | Request or model-output contract failed |
| `503` | No complete, compatible, permitted bundle is ready |

An unexpected `500` is a defect, not a model outcome.

A `503` can also reflect a stale governance verdict, a persisted gate-threshold
policy that no longer reproduces the verdict, or a policy file that disagrees
with its report selection evidence. An explicit rejected-run override permits a
valid policy rejection only; it does not bypass integrity or contract checks.

## Logging and service boundary

Request logs record endpoint, artifact ID, policy ID, and decision class. They
do not record input values, probabilities, or deterministic input hashes.

The service has no authentication, authorization, TLS termination, rate
limiting, retention controls, privacy program, reviewer workflow, or incident
response process. Bind it to localhost and treat it as a local evidence tool.
