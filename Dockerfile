FROM python:3.11-slim

# System deps for rasterio/GDAL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY imint/ imint/
COPY executors/ executors/
COPY config/ config/

# ColonyOS entry point
ENTRYPOINT ["python", "executors/colonyos.py"]
