# API Reference

## Overview

The Fairness Project API provides a versioned REST interface for fairness-aware candidate pre-screening predictions.

- **Base URL**: `http://localhost:8000`
- **API Prefix**: `/v1/`
- **Content Type**: `application/json`
- **Interactive Docs**: Swagger UI at [`/docs`](http://localhost:8000/docs), ReDoc at [`/redoc`](http://localhost:8000/redoc)

## Endpoints

### `GET /health`

Unversioned health check endpoint for liveness probes.

**Response** (`HealthResponse`):

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Always `"healthy"` if the server is up |
| `model_loaded` | bool | Whether a model is currently loaded |

**Status Codes**:

| Code | Description |
|------|-------------|
| 200 | Server is running |

---

### `GET /v1/metadata`

Returns model version, training info, and fairness metrics for governance and audit purposes.

**Response** (`MetadataResponse`):

```json
{
  "model_version": "20240115_143022",
  "trained_at": "2024-01-15T14:30:22",
  "model_type": "xgb",
  "fairness_metrics": {
    "accuracy": 0.862,
    "SPD": 0.158,
    "DI": 0.455,
    "TPR_gap": -0.015
  },
  "api_version": "v1"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `model_version` | string \| null | Run ID of the loaded model |
| `trained_at` | string \| null | ISO timestamp of training |
| `model_type` | string \| null | Model type (lr, rf, xgb) |
| `fairness_metrics` | object | Fairness and performance metrics from training |
| `api_version` | string | API version (always `"v1"`) |

**Status Codes**:

| Code | Description |
|------|-------------|
| 200 | Metadata returned (fields may be null if no model loaded) |

---

### `POST /v1/predict`

Make a single prediction for one candidate.

**Request Body** (`PredictionInput`):

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

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `age` | int | 0-120 | Age of individual |
| `workclass` | string | required | Type of employment |
| `fnlwgt` | int | >= 0 | Final sampling weight |
| `education` | string | required | Highest education level |
| `education_num` | int | 1-20 | Education level as number |
| `marital_status` | string | required | Marital status |
| `occupation` | string | required | Type of occupation |
| `relationship` | string | required | Relationship status |
| `native_country` | string | required | Country of origin |
| `capital_gain` | int | >= 0 | Capital gains |
| `capital_loss` | int | >= 0 | Capital losses |
| `hours_per_week` | int | 0-168 | Hours worked per week |

**Response** (`PredictionOutput`):

```json
{
  "prediction": 1,
  "probability": 0.73,
  "label": ">50K"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | int | Predicted class (0=<=50K, 1=>50K) |
| `probability` | float | Probability of >50K |
| `label` | string | Human-readable prediction |

**Status Codes**:

| Code | Description |
|------|-------------|
| 200 | Prediction returned |
| 422 | Validation error (invalid input) |
| 503 | Model not loaded |

---

### `POST /v1/predict-batch`

Make predictions for multiple candidates in a single request.

**Request Body** (`BatchPredictionInput`):

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
    },
    {
      "age": 28,
      "workclass": "State-gov",
      "fnlwgt": 150000,
      "education": "Masters",
      "education_num": 14,
      "marital_status": "Never-married",
      "occupation": "Prof-specialty",
      "relationship": "Not-in-family",
      "native_country": "United-States",
      "capital_gain": 0,
      "capital_loss": 0,
      "hours_per_week": 50
    }
  ]
}
```

**Response** (`BatchPredictionOutput`):

```json
{
  "predictions": [
    {"prediction": 1, "probability": 0.73, "label": ">50K"},
    {"prediction": 0, "probability": 0.35, "label": "<=50K"}
  ]
}
```

**Status Codes**:

| Code | Description |
|------|-------------|
| 200 | Predictions returned |
| 422 | Validation error in one or more instances |
| 503 | Model not loaded |

---

## Versioning Strategy

All prediction and metadata endpoints are prefixed with `/v1/`. The health check endpoint is unversioned.

**What constitutes a breaking change** (triggers a new version):
- Removing or renaming request/response fields
- Changing field types or constraints
- Removing an endpoint
- Changing the meaning of existing fields

**Non-breaking changes** (same version):
- Adding optional request fields
- Adding response fields
- Adding new endpoints
- Improving validation messages

**Deprecation policy**: When a new API version is introduced, the previous version will remain available for at least 3 months with a `Deprecation` header in responses.

## Error Codes

| Code | Meaning | Example |
|------|---------|---------|
| 200 | Success | Valid prediction returned |
| 422 | Validation Error | `{"detail": [{"loc": ["body", "age"], "msg": "ensure this value is less than or equal to 120", "type": "value_error"}]}` |
| 503 | Service Unavailable | `{"detail": "Model not loaded. Set MODEL_PATH env var."}` |
| 500 | Internal Server Error | Unexpected error during prediction |

## Audit Logging

Every prediction is logged as a structured JSON line:

```json
{
  "timestamp": "2024-01-15T14:30:22.123456Z",
  "level": "INFO",
  "logger": "fairness_project.api",
  "message": "prediction",
  "endpoint": "/v1/predict",
  "input_hash": "a1b2c3d4e5f6",
  "prediction": 1,
  "probability": 0.73
}
```

Input data is **never logged directly**. Instead, a SHA-256 hash of the input is recorded for traceability without exposing PII.

## Data Schema Reference

The API accepts 12 input fields derived from the Adult dataset. Protected attributes (`sex`, `race`, `income`) are intentionally excluded from the prediction input.

For full dataset documentation, see [Data Documentation](data.md).
