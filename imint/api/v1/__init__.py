"""ImintEngine HTTP API v1.

Importable entry point for ASGI runners:

    uvicorn imint.api.v1:app --host 127.0.0.1 --workers 1 --lifespan on

One worker and one replica only; no reload. Jobs are lost on restart.
See README.md for the local ownership guard and deployment limitations.

Routes:
    GET  /v1/health                 — liveness probe
    GET  /v1/analyzers              — list available analyzers + registry metadata
    POST /v1/analyze                — submit an analysis job, returns job_id
    GET  /v1/jobs/{id}              — poll status / retrieve result

See `imint/api/v1/server.py` for the implementation.
"""

from imint.api.v1.server import app  # noqa: F401

__all__ = ["app"]
