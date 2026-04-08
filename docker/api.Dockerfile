# Stage 1: Builder
# Use Python 3.11 slim image to match requires-python = ">=3.11"
FROM python:3.11-slim as builder

# Install uv via the official python package manager
RUN pip install uv

# Set the working directory
WORKDIR /app

# Create the virtual environment before installing dependencies
ENV VIRTUAL_ENV=/app/.venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy pyproject.toml and uv.lock for layer caching
COPY pyproject.toml uv.lock ./

# Install ONLY dependencies to ensure Docker layer caching is preserved
RUN uv sync --extra api --no-install-project

# Copy application source code so that the Python package can actually build
COPY src/ ./src/
COPY agents/ ./agents/
COPY app/ ./app/

# Intentionally install the project as editable to correctly symlink /app/src into site-packages.
RUN uv pip install --no-deps -e .

# Stage 2: Runtime
# Use identical Python version to ensure dynamic library compatibility
FROM python:3.11-slim

# Install system runtime dependencies required by ML models (e.g. LightGBM requires libgomp1)
# Security Hardening: Remove apt lists to avoid caching unnecessary packages
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Optimize Python execution in containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app"

# Security Hardening: Create and use a non-root system user and group
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -d /app -s /bin/bash appuser

# Set working directory and adjust ownership
WORKDIR /app

# Copy the fully resolved virtual environment from the builder stage
COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv

# Copy pyproject.toml so app logic can resolve the project root
COPY --chown=appuser:appgroup pyproject.toml ./

# Copy application source code logic (frequently changed, so copied after dependencies)
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup agents/ ./agents/
COPY --chown=appuser:appgroup app/api/ ./app/api/
COPY --chown=appuser:appgroup configs/inference/ ./configs/inference/
COPY --chown=appuser:appgroup configs/agents/ ./configs/agents/

# Model Artifact Handling
# Baking artifacts into the image for the POC (total size ~95 MB)
COPY --chown=appuser:appgroup artifacts/models/ ./artifacts/models/
COPY --chown=appuser:appgroup artifacts/explore/ ./artifacts/explore/

# Switch to the non-root 'appuser' before executing the final CMD
USER appuser

# Expose API traffic port
EXPOSE 8000

# Healthcheck to probe the FastAPI /health endpoint
# A generous 60s start-period gives time for ML models and LangGraph components to compile
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]

# Command to begin hosting the backend application
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
