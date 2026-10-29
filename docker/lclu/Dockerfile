FROM python@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f
WORKDIR /opt/imintengine
COPY docker/lclu/system_runtime.py docker/lclu/system_runtime.py
RUN python docker/lclu/system_runtime.py /opt/lclu-system
ENV LD_LIBRARY_PATH=/opt/lclu-system/lib/x86_64-linux-gnu:/opt/lclu-system/usr/lib/x86_64-linux-gnu
COPY docker/lclu/requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements.lock
COPY imint/ imint/
COPY scripts/ scripts/
COPY config/ config/
COPY docs/ docs/
COPY docker/lclu/run.py docker/lclu/run.py
COPY docker/lclu/smoke.py docker/lclu/smoke.py
COPY SOURCE_MANIFEST.json SOURCE_MANIFEST.json
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/opt/imintengine \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 GDAL_CACHEMAX=256 VPP_SOURCE=wekeo \
    IMINT_API_MAX_PIXELS=262144 IMINT_API_MAX_JOBS=4 \
    IMINT_LCLU_MANIFEST=/work/stockholm/serving.json IMINT_API_OUTPUT_DIR=/work/results
EXPOSE 8000
RUN python docker/lclu/smoke.py
CMD ["python", "-m", "uvicorn", "imint.api.v1:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--lifespan", "on"]
