"""FastAPI app for ImintEngine v1.

Per agentic_workflow rollout plan W4.1.

Decouples ImintEngine from any specific executor by exposing the
existing `run_job(IMINTJob) → IMINTResult` entry point as HTTP. Same
analysis pipeline; new boundary.

Design notes:
 - Jobs run in a background task pool, not inline. POST /v1/analyze
   returns immediately with a job_id; clients poll GET /v1/jobs/{id}.
 - The job store is in-memory by default (suitable for single-process
   uvicorn). Production deploys swap in a Redis-backed store; the
   `JobStore` interface is small.
 - Authentication is intentionally NOT wired yet (rollout W4 is
   "infra-only"). The first customer pilot will require OAuth2 or an
   API-key middleware; placeholder hook is in place.
 - SLA telemetry (W4.5) is wired via the `instrumentation` module if
   present (fail-open like des-agent / des-chatbot).

Run locally:
    uvicorn imint.api.v1:app --reload

Production:
    gunicorn imint.api.v1:app -k uvicorn.workers.UvicornWorker -w 4
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from imint.api.v1.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    AnalyzerInfo,
    AnalyzerSummary,
    AnalyzersResponse,
    HealthResponse,
    JobStatus,
    JobStatusResponse,
)
from imint.api.v1.telemetry import (
    emit_access_log,
    metrics_payload,
    new_request_id,
    record_request,
)

# Schema + embedding identity surfaced via /health and /analyzers — pulled
# from des-contracts (v0.1.0+) so the API advertises which contract version
# its results conform to.
try:
    from des_contracts import EMBEDDING_CONFIG, __version__ as DES_CONTRACTS_VERSION
    from des_contracts.schema import SCHEMA_VERSION as UNIFIED_SCHEMA_VERSION
except ImportError:  # pragma: no cover
    DES_CONTRACTS_VERSION = "unknown"
    UNIFIED_SCHEMA_VERSION = "unknown"
    EMBEDDING_CONFIG = None  # type: ignore[assignment]

logger = logging.getLogger("imint.api.v1")


# ── Job store ───────────────────────────────────────────────────────────────
#
# Minimal in-memory job store. Each job is a dict; production will swap this
# for a Redis-backed implementation behind the same interface (get/put/list).
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = asyncio.Lock()


async def _put_job(job_id: str, payload: dict[str, Any]) -> None:
    async with _jobs_lock:
        _jobs[job_id] = payload


async def _get_job(job_id: str) -> dict[str, Any] | None:
    async with _jobs_lock:
        return _jobs.get(job_id)


# ── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ImintEngine API",
    description=(
        "HTTP boundary for ImintEngine. Submit AOI + date, get IMINT analysis "
        "(LULC + harvest-maturity from the dual-head model + auxiliary analyzers). "
        "Per agentic_workflow rollout plan W4.1."
    ),
    version="0.1.0",
    openapi_url="/v1/openapi.json",
    docs_url="/v1/docs",
    redoc_url="/v1/redoc",
)


# ── Telemetry middleware ───────────────────────────────────────────────────
#
# W4.5 SLA telemetry. Per-request: method, path, status, duration_ms.
# Fail-open like des-agent / des-chatbot — no OTel installed → no-op.

try:
    from instrumentation import record_chat_ms, span  # type: ignore[import-not-found]

    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False

    def span(*_, **__):  # type: ignore[misc]
        from contextlib import nullcontext
        return nullcontext()

    def record_chat_ms(*_, **__):  # type: ignore[misc]
        return None


@app.middleware("http")
async def telemetry_middleware(request: Request, call_next):
    """Per-request telemetry pipeline (W4.5).

    Three signals per request:
      1. Structured JSON access log to stderr (always on).
      2. Prometheus histogram + counter (if prometheus_client installed).
      3. OTel span (if opentelemetry installed).

    Plus an X-Request-Id response header for distributed-trace stitching
    and an X-Response-Time-MS header for client-side SLA monitoring.
    """
    start = time.monotonic()
    request_id = new_request_id()
    # `route.path` is the parameterised pattern (e.g. /v1/jobs/{job_id});
    # falls back to URL path if FastAPI hasn't matched yet.
    route_template = getattr(request.scope.get("route"), "path", request.url.path)

    with span(
        f"http.{request.method.lower()}",
        path=request.url.path,
        route=route_template,
        request_id=request_id,
    ):
        try:
            response = await call_next(request)
            duration_ms = (time.monotonic() - start) * 1000
            response.headers["X-Response-Time-MS"] = f"{duration_ms:.1f}"
            response.headers["X-Request-Id"] = request_id
            emit_access_log(
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=duration_ms,
                request_id=request_id,
            )
            record_request(
                method=request.method,
                route=route_template,
                status=response.status_code,
                duration_ms=duration_ms,
            )
            return response
        except Exception:
            duration_ms = (time.monotonic() - start) * 1000
            emit_access_log(
                method=request.method,
                path=request.url.path,
                status=500,
                duration_ms=duration_ms,
                request_id=request_id,
                extra={"error": "unhandled"},
            )
            record_request(
                method=request.method,
                route=route_template,
                status=500,
                duration_ms=duration_ms,
            )
            logger.exception(
                "%s %s -> EXCEPTION (%.1f ms) [request_id=%s]",
                request.method,
                request.url.path,
                duration_ms,
                request_id,
            )
            raise


@app.get("/v1/metrics", include_in_schema=False)
async def metrics() -> "Response":  # type: ignore[name-defined]
    """Prometheus scrape endpoint (W4.5).

    Returns plain text in Prometheus exposition format if
    prometheus_client is installed; an empty placeholder otherwise so
    scrape configs don't 404. Excluded from OpenAPI to keep /v1/docs
    focused on the analyze workflow.
    """
    from fastapi import Response  # local import to avoid top-level circulars

    body, content_type = metrics_payload()
    return Response(content=body, media_type=content_type)


# ── Endpoints ──────────────────────────────────────────────────────────────


@app.get("/v1/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    """Liveness probe. Includes the active contract versions so monitoring
    can detect a deploy that drifted off the canonical contracts."""
    return HealthResponse(
        status="ok",
        api_version="v1",
        schema_version=UNIFIED_SCHEMA_VERSION,
        embedding_identity=(
            EMBEDDING_CONFIG.identity if EMBEDDING_CONFIG is not None else "unknown"
        ),
    )


@app.get("/v1/analyzers", response_model=AnalyzersResponse, tags=["meta"])
async def list_analyzers() -> AnalyzersResponse:
    """List analyzers known to this ImintEngine instance.

    Reads `imint.engine.ANALYZER_REGISTRY` so downstream clients can
    discover capabilities without reading the YAML config.
    """
    try:
        from imint.engine import ANALYZER_REGISTRY
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ImintEngine analyzer registry unavailable",
        )

    analyzers = []
    for name, cls in ANALYZER_REGISTRY.items():
        doc = (cls.__doc__ or "").strip().split("\n", 1)[0] or None
        analyzers.append(
            AnalyzerInfo(
                name=name,
                enabled=True,
                description=doc,
                version=getattr(cls, "version", None),
            )
        )

    return AnalyzersResponse(
        schema_version=UNIFIED_SCHEMA_VERSION,
        analyzers=analyzers,
    )


@app.post(
    "/v1/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["analyze"],
)
async def submit_analysis(req: AnalyzeRequest) -> AnalyzeResponse:
    """Submit a new analysis job. Runs in the background; poll via /v1/jobs/{id}.

    Returns 202 Accepted immediately. The job_id in the response can be
    used to poll `GET /v1/jobs/{id}` for status and final result.
    """
    resp = AnalyzeResponse(
        poll_url="",  # filled in below once we have job_id
    )
    resp.poll_url = f"/v1/jobs/{resp.job_id}"

    # Stash request immediately so polling works even before the worker starts
    await _put_job(
        resp.job_id,
        {
            "job_id": resp.job_id,
            "date": req.date,
            "bbox": req.bbox.model_dump(),
            "status": JobStatus.QUEUED,
            "submitted_at": resp.submitted_at,
            "started_at": None,
            "completed_at": None,
            "duration_ms": None,
            "success": None,
            "analyzers": [],
            "summary_path": None,
            "error": None,
            "_request": req.model_dump(),
            "_callback_url": req.callback_url,
        },
    )

    # Fire-and-forget the worker. asyncio.create_task is fine for in-process
    # backgrounding; production swaps to Celery / RQ behind the same interface.
    asyncio.create_task(_run_job_async(resp.job_id))

    return resp


@app.get(
    "/v1/jobs/{job_id}",
    response_model=JobStatusResponse,
    tags=["analyze"],
)
async def get_job(job_id: str) -> JobStatusResponse:
    """Retrieve job status / final result."""
    job = await _get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"job {job_id} not found",
        )
    return JobStatusResponse(
        job_id=job["job_id"],
        date=job["date"],
        bbox=job["bbox"],
        status=job["status"],
        submitted_at=job["submitted_at"],
        started_at=job["started_at"],
        completed_at=job["completed_at"],
        duration_ms=job["duration_ms"],
        success=job["success"],
        analyzers=job["analyzers"],
        summary_path=job["summary_path"],
        error=job["error"],
    )


# ── Worker ─────────────────────────────────────────────────────────────────


async def _run_job_async(job_id: str) -> None:
    """Background worker — runs run_job() in a thread to avoid blocking the
    event loop on numpy/torch work."""
    job = await _get_job(job_id)
    if job is None:
        return

    job["status"] = JobStatus.RUNNING
    job["started_at"] = datetime.now(timezone.utc)
    await _put_job(job_id, job)
    start = time.monotonic()

    try:
        # Lazy import — keeps API import time low when worker isn't called.
        from imint.engine import run_job
        from imint.job import IMINTJob

        req = job["_request"]
        bbox = req["bbox"]

        # Construct the IMINTJob. Image data is fetched by run_job's
        # downstream via the existing fetch pipeline; the API layer
        # passes coords + date and lets the engine resolve the rest.
        imint_job = IMINTJob(
            date=req["date"],
            coords={
                "west": bbox["west"],
                "south": bbox["south"],
                "east": bbox["east"],
                "north": bbox["north"],
            },
            job_id=job_id,
        )

        # Run in a thread — run_job is blocking numpy/torch.
        result = await asyncio.to_thread(run_job, imint_job)

        job["completed_at"] = datetime.now(timezone.utc)
        job["duration_ms"] = (time.monotonic() - start) * 1000
        job["success"] = result.success
        job["error"] = result.error
        job["summary_path"] = result.summary_path
        job["analyzers"] = [
            AnalyzerSummary(
                name=getattr(r, "name", "unknown"),
                status=getattr(r, "status", "success"),
                duration_ms=getattr(r, "duration_ms", None),
                artifacts=list(getattr(r, "artifacts", []) or []),
                error=getattr(r, "error", None),
            ).model_dump()
            for r in (result.analyzer_results or [])
        ]
        job["status"] = JobStatus.SUCCESS if result.success else JobStatus.FAILED

    except Exception as e:  # pragma: no cover
        job["status"] = JobStatus.FAILED
        job["completed_at"] = datetime.now(timezone.utc)
        job["duration_ms"] = (time.monotonic() - start) * 1000
        job["success"] = False
        job["error"] = f"{type(e).__name__}: {e}"
        logger.exception("job %s crashed", job_id)

    await _put_job(job_id, job)
    record_chat_ms(  # repurposed metric — host instrumentation can map this
        job["duration_ms"] or 0.0,
        status=job["status"].value if isinstance(job["status"], JobStatus) else str(job["status"]),
    )

    # Optional webhook notification — best-effort, never blocks the worker.
    callback = job.get("_callback_url")
    if callback:
        asyncio.create_task(_notify_callback(job_id, callback))


async def _notify_callback(job_id: str, url: str) -> None:
    """POST the final job state to the configured webhook. Best-effort."""
    try:
        import httpx  # type: ignore[import-not-found]

        job = await _get_job(job_id)
        if job is None:
            return
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(url, json={
                "job_id": job_id,
                "status": (
                    job["status"].value
                    if isinstance(job["status"], JobStatus)
                    else str(job["status"])
                ),
                "success": job["success"],
                "duration_ms": job["duration_ms"],
            })
    except Exception:
        logger.exception("callback POST to %s failed for job %s", url, job_id)


# ── Exception handlers ─────────────────────────────────────────────────────


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all so API errors are JSON, not HTML."""
    logger.exception("unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": type(exc).__name__,
            "detail": str(exc),
            "path": str(request.url.path),
        },
    )
