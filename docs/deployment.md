# Deployment & Operations Manual

## Local Development

```bash
# Install with API dependencies
pip install -e ".[api]"

# Start the development server with auto-reload
MODEL_PATH=models/model.joblib uvicorn fairness_project.inference.api:app \
  --host 0.0.0.0 --port 8000 --reload
```

The `--reload` flag watches for file changes and restarts the server automatically.

## Docker

### Build and Run

```bash
# Build the image
docker build -t fairness-api .

# Run with model volume mount
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -e MODEL_PATH=/app/models/model.joblib \
  fairness-api
```

### Docker Compose

```bash
# Start all services
docker-compose up

# Start in background
docker-compose up -d

# Override config file
docker-compose run -e CONFIG_PATH=/app/configs/custom.yaml fairness-api
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | *(none)* | Path to the joblib model file. Required for predictions. |
| `CONFIG_PATH` | *(none)* | Path to YAML config file. Optional; uses defaults if not set. |
| `WORKERS` | `1` | Number of uvicorn worker processes. |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |

## Health Checks

| Probe | Endpoint | Purpose |
|-------|----------|---------|
| Liveness | `GET /health` | Server is running |
| Readiness | `GET /v1/metadata` | Model is loaded and serving |

Example health check in Docker Compose:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 5s
  retries: 3
```

## Monitoring and Logging

All API logs are emitted as structured JSON lines:

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

Pipe logs to your aggregation system (ELK, CloudWatch, Datadog) for monitoring prediction distributions and latency.

## Scaling

### Horizontal Scaling

Increase the number of container replicas behind a load balancer:

```yaml
# docker-compose.yml
services:
  api:
    deploy:
      replicas: 3
```

### Vertical Scaling

Increase workers per container:

```bash
docker run -e WORKERS=4 -p 8000:8000 fairness-api
```

## Model Updates

Zero-downtime model update procedure:

1. **Train** a new model: `fairness train --model xgb`
2. **Evaluate**: `fairness evaluate`
3. **Gate check**: `python -m fairness_project.governance.gate --report <report_path>`
4. **Copy** the new model file to the model volume
5. **Restart** the API container (graceful): `docker-compose restart api`
6. **Verify**: `curl http://localhost:8000/v1/metadata` to confirm the new model version

## Security Notes

- **No built-in authentication**: The API does not include auth. For production, place it behind a reverse proxy (nginx, Traefik) with TLS and authentication.
- **PII hashing**: Input data is hashed in audit logs. Raw PII is never persisted in logs.
- **Protected attributes excluded**: The prediction input schema intentionally excludes `sex`, `race`, and `income` fields.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| 503 "Model not loaded" | `MODEL_PATH` not set or file missing | Set `MODEL_PATH` to a valid `.joblib` file |
| Connection refused on :8000 | Server not running | Start with `uvicorn fairness_project.inference.api:app` |
| High latency on batch requests | Large batch size or single worker | Increase `WORKERS` or reduce batch size |
| Model predictions differ from training | Different preprocessing pipeline | Ensure the model file includes the full sklearn pipeline |

## Production Checklist

- [ ] Governance gate passed for the deployed model
- [ ] `MODEL_PATH` points to the gated model file
- [ ] `LOG_LEVEL` set to `INFO` or `WARNING`
- [ ] Health check probes configured
- [ ] Structured logs piped to aggregation
- [ ] TLS termination and authentication configured at reverse proxy
- [ ] Rate limiting configured
- [ ] Monitoring alerts for 5xx errors and latency spikes
