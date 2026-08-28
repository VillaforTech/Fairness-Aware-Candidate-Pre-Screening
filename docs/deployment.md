# API Scaffold and Operations Notes

> **Current status:** The routes are tested with mock models, but the bundled
> XGBoost artifact is incompatible with the API's 12-field input schema.
> End-to-end prediction serving and the Docker path are not currently verified.

## Verified Scope

The API test suite verifies request validation, health and metadata responses,
single predictions, and batch predictions by injecting mock models:

```bash
python -m pip install -e ".[dev,api]"
pytest tests/test_api.py -q
```

These tests establish route behavior only. They do not establish compatibility
with a trained artifact or a runnable deployment.

## Known Blockers

- The tracked XGBoost pipeline expects 14 raw features, while the API accepts 12.
  An actual `/v1/predict` request therefore fails with the bundled artifact.
- The tracked joblib artifact emits scikit-learn and XGBoost compatibility
  warnings when loaded with the current unconstrained dependency versions.
- There is no end-to-end test that trains, saves, loads, and serves one artifact.
- The Dockerfile and Compose configuration have not passed an end-to-end smoke
  test with a compatible model.
- The service has no authentication, authorization, rate limiting, or deployment
  security review.

## Interface Contract

The intended request and response shapes are documented in
[API Specification](api_spec.md). They should be treated as a development target
until the blockers above are resolved.

## Exit Criteria for a Verified Local Demo

1. Define one canonical feature schema shared by training and inference.
2. Train and serialize an artifact in a pinned environment.
3. Add an integration test that loads the artifact and receives HTTP 200 from
   `/v1/predict` for a documented fixture.
4. Build the Docker image and repeat the same prediction smoke test.
5. Record the exact dependency lock, artifact provenance, and expected output.
