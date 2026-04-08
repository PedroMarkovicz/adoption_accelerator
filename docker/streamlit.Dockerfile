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

# Install dependencies safely without needing the full project source
RUN uv sync --extra streamlit --no-install-project

# Stage 2: Runtime
FROM python:3.11-slim

# Security Hardening: Remove apt lists to avoid caching unnecessary packages
RUN rm -rf /var/lib/apt/lists/*

# Set up Streamlit environment variables and prevent browser from opening
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

# Security Hardening: Create and use a non-root system user and group
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -d /app -s /bin/bash appuser

# Set working directory and adjust ownership
WORKDIR /app

# Copy the fully resolved virtual environment from the builder stage
COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv

# Copy only the Streamlit application source code logic
# (Ensures no ML artifacts or heavy models are loaded in this container)
COPY --chown=appuser:appgroup app/streamlit/ ./app/streamlit/

# Switch to the non-root 'appuser' before executing the final CMD
USER appuser

# Expose Streamlit traffic port
EXPOSE 8501

# Healthcheck to probe the Streamlit health endpoint
# 15s start-period is sufficient as there is no model-loading overhead
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"]

# Command to host the Streamlit frontend
CMD ["streamlit", "run", "app/streamlit/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
