# ─── Zen Agent — Dockerfile ─────────────────────────────────────
FROM python:3.12-slim AS base

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements-armv7.txt

# Copy app
COPY . .

# Create data directory
RUN mkdir -p /app/data

# Expose port
EXPOSE 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:9090/api/health || exit 1

# Default: run server
CMD ["python3", "-m", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "9090", "--log-level", "info"]
