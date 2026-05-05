# Cloud-detection model runner (Python 3.11)
#
# Hosts deep-learning cloud detectors that need newer Python than the host
# (omnicloudmask + clouds2mask both rely on PEP 604 union syntax / torchgeo
# that demands Python ≥ 3.10).
#
# The container is single-shot: it reads .npz tiles from /work/in/, writes
# probability rasters to /work/out/, then exits. Driven by run_cloud_models.py
# in the same directory, which is bind-mounted at /opt/runner/.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps. libgdal-dev is needed for the rasterio sdist build on
# linux/arm64 (no manylinux wheel for 1.4.3 + py3.11 yet).
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
        ca-certificates curl libexpat1 libgomp1 \
        libgdal-dev gdal-bin gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first (smaller wheels, no CUDA download)
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1 torchvision==0.20.1

# Cloud detectors + their deps. s2cloudless is also installed here so a
# single container handles all open-source cloud models in one run.
RUN pip install \
        s2cloudless==1.7.3 \
        omnicloudmask \
        rasterio \
        numpy==1.26.4 \
        scipy

WORKDIR /opt/runner
ENTRYPOINT ["python", "/opt/runner/run_cloud_models.py"]
