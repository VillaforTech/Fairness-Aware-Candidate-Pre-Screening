# Local API reference

The HTTP API is a local reference interface over one validated experiment
bundle. It is covered by an end-to-end test that trains, saves, reloads, and
serves a real model artifact.

It is not a hiring API. The service exposes the baseline global-threshold model
only; the sex-specific threshold experiment remains offline evaluation data.

## Start the service

Set `RUN_DIR` to a complete bundle containing `model.joblib`, `manifest.json`,
`policy.json`, and `report.json`. First generate `runs/local-xgb` using the
[local demo procedure](deployment.md#build-a-complete-run-bundle); the committed
reference directory intentionally contains only its report:

```bash
RUN_DIR=runs/local-xgb \
  uv run uvicorn fairness_project.inference.api:app \
  --host 127.0.0.1 \
  --port 8000
```

Swagger UI is available at <http://127.0.0.1:8000/docs> and ReDoc at
<http://127.0.0.1:8000/redoc>.

## Readiness and provenance

### `GET /health`

Liveness only. A `200` response means the process can answer requests; it does
not mean a usable artifact was loaded.

```json
{"status": "ok"}
```

### `GET /ready`

Returns `200` only after the complete bundle has passed schema, digest, runtime
dependency, model, and feature-contract validation.

```json
{
  "ready": true,
  "artifact_id": "local-xgb"
}
```

Missing or incompatible bundles return `503`.

### `GET /v1/metadata`

Names the loaded artifact and the policy that the API actually serves.

```json
{
  "artifact_id": "local-xgb",
  "schema_version": "1.0",
  "model_type": "xgb",
  "created_at": "2026-08-28T21:52:49.369300+00:00",
  "decision_policy": {
    "policy_id": "global-threshold-v1",
    "kind": "global_threshold",
    "threshold": 0.5,
    "fairness_adjustment_applied": false
  },
  "governance": {
    "passed": false,
    "violations": [
      "DI=0.4120 < min_disparate_impact=0.8",
      "|SPD|=0.1563 > max_spd=0.1"
    ]
  },
  "evaluation_only": true,
  "api_version": "v1"
}
```

## Prediction contract

The request has exactly the same 12 feature columns used by training:

```json
{
  "age": 35,
  "workclass": "Private",
  "fnlwgt": 200000,
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
| `workclass` | string | required |
| `fnlwgt` | integer | at least 0 |
| `education` | string | required |
| `education_num` | integer | 1 to 20 |
| `marital_status` | string | required |
| `occupation` | string | required |
| `relationship` | string | required |
| `native_country` | string | required |
| `capital_gain` | integer | at least 0 |
| `capital_loss` | integer | at least 0 |
| `hours_per_week` | integer | 0 to 168 |

Missing, null, blank categorical, non-integer numeric, Boolean numeric, and
extra fields are rejected. `sex` and `race` are deliberately not accepted as
model inputs, so the API cannot silently apply the offline group-threshold
policy. CSV batch inference enforces the same value contract.

### `POST /v1/predict`

```json
{
  "prediction": 1,
  "probability": 0.73,
  "label": ">50K",
  "decision_threshold": 0.5,
  "decision_policy": "global-threshold-v1",
  "artifact_id": "local-xgb"
}
```

The probability is for the `>50K` Adult-dataset label. It is not a probability
of job suitability.

### `POST /v1/predict-batch`

The body has one `instances` array containing 1 to 1,000 prediction records.
Every row is validated before inference.

```json
{
  "instances": [
    {
      "age": 35,
      "workclass": "Private",
      "fnlwgt": 200000,
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

The response wraps the same prediction objects in a `predictions` array.

## Errors

| Status | Meaning |
|---:|---|
| `200` | Request succeeded |
| `422` | Request or inference contract was invalid |
| `503` | No complete, compatible bundle is ready |

Unexpected failures remain `500`; they should be treated as defects, not as
valid model outcomes.

## Logging and security boundary

Prediction logs contain the endpoint, artifact ID, policy ID, and predicted
class. They do not contain input values, probabilities, or deterministic hashes
of the input. The service has no authentication, authorization, rate limiting,
transport security, retention controls, or production security review.
