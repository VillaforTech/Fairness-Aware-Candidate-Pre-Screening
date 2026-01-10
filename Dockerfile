# Fairness Project API Dockerfile
# ================================
# Build and run the fairness API in a container

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY pyproject.toml ./
COPY src/ ./src/

# Install the package with API dependencies
RUN pip install --no-cache-dir -e ".[api]"

# Copy model if provided during build
ARG MODEL_PATH=""
COPY ${MODEL_PATH:-pyproject.toml} /app/model.joblib*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/model.joblib

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API
CMD ["uvicorn", "fairness_project.inference.api:app", "--host", "0.0.0.0", "--port", "8000"]
