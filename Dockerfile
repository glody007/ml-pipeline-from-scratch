# =============================================================================
# Multi-stage build for production API
# =============================================================================

# Stage 1: Build dependencies
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements-api.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements-api.txt

# Stage 2: Production image (minimal)
FROM python:3.11-slim AS production
WORKDIR /app

# Install pre-built wheels (faster, smaller)
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copy only serving code (not training code)
COPY src/api.py src/features.py /app/src/
COPY configs/ /app/configs/

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

# =============================================================================
# Development target (use with: docker build --target development .)
# =============================================================================
FROM python:3.11-slim AS development
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
# Source mounted via volumes in docker-compose
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
