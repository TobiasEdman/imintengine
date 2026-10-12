# ── Multi-stage build optimised for ARM64 (M1/M2 Mac) ────────────────────
# Stage 1: Build native wheels for rasterio/GDAL/shapely
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    g++ git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /wheels
COPY pyproject.toml README.md LICENSE THIRD_PARTY_LICENSES.md ./
COPY imint/ imint/
RUN pip wheel --no-cache-dir --no-deps --wheel-dir=/wheels \
    "git+https://github.com/TobiasEdman/des-contracts.git@v0.1.0"
RUN pip wheel --no-cache-dir --wheel-dir=/wheels --find-links=/wheels ".[api]"

# Stage 2: Runtime — slim image with pre-built wheels
FROM python:3.11-slim

# Runtime GDAL libs only (no compiler)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal36 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install from pre-built wheels (fast, no compilation)
COPY --from=builder /wheels /wheels
COPY pyproject.toml README.md LICENSE THIRD_PARTY_LICENSES.md ./
COPY imint/ imint/
RUN pip install --no-cache-dir --no-index --find-links=/wheels \
    "imint-engine[api]==0.2.0" \
    && rm -rf /wheels

# Copy application code
COPY imint/ imint/
COPY executors/ executors/
COPY config/ config/

# CDSE credentials (if available — gitignored, optional)
# Use a shell trick: copy if exists, skip if not
RUN true
COPY .cdse_credential[s] /app/

# No ENTRYPOINT — ColonyOS Docker executor wraps cmd in sh -c
# and sets it as the container command
CMD ["python", "executors/colonyos.py"]
