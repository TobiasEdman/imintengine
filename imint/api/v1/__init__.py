"""ImintEngine HTTP API v1.

Importable entry point for ASGI runners:

    uvicorn imint.api.v1:app
    gunicorn imint.api.v1:app -k uvicorn.workers.UvicornWorker

Routes:
    GET  /v1/health                 — liveness probe
    GET  /v1/analyzers              — list available analyzers + registry metadata
    POST /v1/analyze                — submit an analysis job, returns job_id
    GET  /v1/jobs/{id}              — poll status / retrieve result

See `imint/api/v1/server.py` for the implementation.
"""

from imint.api.v1.server import app  # noqa: F401

__all__ = ["app"]
