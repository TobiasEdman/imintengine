"""FastAPI app for ImintEngine v1.

Per agentic_workflow rollout plan W4.1.

Decouples ImintEngine from any specific executor by exposing the
existing `run_job(IMINTJob) → IMINTResult` entry point as HTTP. Same
analysis pipeline; new boundary.

Deployment contract:
 - One ASGI worker, one replica, one host, lifespan enabled. Job state is
   memory-only and disappears on restart; artifacts remain on disk.
 - A POSIX file lock in IMINT_API_OUTPUT_DIR rejects a second local worker
   using that directory. It is NOT a distributed lock or shared job store.
 - This unauthenticated API is for trusted access behind an operator-managed
   access boundary. It is not a public multi-tenant service.
 - Callbacks and per-request analyzer controls are rejected (422).
 - Exact-day production DES fetch, cloud threshold 0.1, server YAML config.

Run from the repository root (no reload; no multi-worker gunicorn):
    uvicorn imint.api.v1:app --host 127.0.0.1 --workers 1 --lifespan on

See imint/api/v1/README.md for limits and shutdown behavior.
"""
from __future__ import annotations

import asyncio
import fcntl
from contextlib import asynccontextmanager
from contextvars import copy_context
import logging
import os
from pathlib import Path
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
# Single-process, ephemeral job state. Never deploy multiple replicas.
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = asyncio.Lock()


async def _put_job(job_id: str, payload: dict[str, Any]) -> None:
    async with _jobs_lock:
        _jobs[job_id] = payload


async def _get_job(job_id: str) -> dict[str, Any] | None:
    async with _jobs_lock:
        return _jobs.get(job_id)


_pending_jobs: set[asyncio.Task] = set()
_accepting_jobs = False
_analysis_slot = asyncio.Semaphore(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Hold local ownership until all accepted blocking jobs finish."""
    global _accepting_jobs
    root = _output_root()
    root.mkdir(parents=True, exist_ok=True)
    # Do not unlink this file: replacing its inode would bypass flock.
    with (root / ".api-worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "ImintEngine API requires one worker using IMINT_API_OUTPUT_DIR"
            ) from exc
        _accepting_jobs = True
        try:
            yield
        finally:
            _accepting_jobs = False
            # Do not cancel to_thread work: cancelling its await does not
            # stop the thread. Keep the ownership lock while draining.
            cancelled = False
            if _pending_jobs:
                drain = asyncio.gather(*tuple(_pending_jobs), return_exceptions=True)
                while not drain.done():
                    try:
                        await asyncio.shield(drain)
                    except asyncio.CancelledError:
                        # Lifespan cancellation must not release ownership
                        # while a blocking worker thread still writes outputs.
                        cancelled = True
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            if cancelled:
                raise asyncio.CancelledError


# ── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    lifespan=lifespan,
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
# Fail-open like omni-rag / des-chatbot — no OTel installed → no-op.

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
    route_template = "unmatched"

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
                route=getattr(request.scope.get("route"), "path", "unmatched"),
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
                route=getattr(request.scope.get("route"), "path", "unmatched"),
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
    if not _accepting_jobs:
        raise HTTPException(status_code=503, detail="API worker not started or shutting down")
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
        },
    )

    task = asyncio.create_task(_run_job_serial(resp.job_id))
    _pending_jobs.add(task)
    task.add_done_callback(_pending_jobs.discard)

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


def _output_root() -> Path:
    """Trusted operator setting, never a request-supplied path."""
    return Path(os.environ.get("IMINT_API_OUTPUT_DIR", "outputs/api")).resolve()


def _execute_analysis(req: dict[str, Any], job_id: str):
    """Fetch real, exact-day DES imagery before calling the engine.

    Uses the production fetcher, which rejects excess clouds and fetch
    failures. There is deliberately no development/synthetic fallback.
    """
    from imint.fetch import fetch_des_data
    from imint.engine import run_job
    from imint.job import IMINTJob

    config_path = Path("config/analyzers.yaml").resolve()
    if not config_path.is_file():
        raise RuntimeError("API requires config/analyzers.yaml in its working directory")
    bbox = req["bbox"]
    coords = {key: bbox[key] for key in ("west", "south", "east", "north")}
    fetched = fetch_des_data(
        date=req["date"], coords=coords, cloud_threshold=0.1,
        include_scl=True, date_window=0,
    )
    job = IMINTJob(
        date=req["date"], coords=coords, job_id=job_id,
        rgb=fetched.rgb, bands=fetched.bands, scl=fetched.scl, geo=fetched.geo,
        output_dir=str(_output_root() / job_id), config_path=str(config_path),
    )
    return run_job(job)


async def _run_job_serial(job_id: str) -> None:
    # Single production fetch/analysis at a time; waiting jobs stay queued.
    async with _analysis_slot:
        await _run_job_async(job_id)


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

    cancelled = False
    try:
        # Keep a Future, not a to_thread Task: event-loop teardown cancels
        # every Task, but must not invalidate our thread-completion signal.
        # Preserve to_thread's context propagation for instrumentation.
        work = asyncio.get_running_loop().run_in_executor(
            None, copy_context().run, _execute_analysis, job["_request"], job_id,
        )
        while not work.done():
            try:
                await asyncio.shield(work)
            except asyncio.CancelledError:
                # A cancelled waiter cannot stop blocking work. Keep the
                # analysis slot and lifespan ownership until it completes.
                cancelled = True
        result = work.result()

        job["completed_at"] = datetime.now(timezone.utc)
        job["duration_ms"] = (time.monotonic() - start) * 1000
        job["success"] = result.success
        job["error"] = result.error
        job["summary_path"] = result.summary_path
        job["analyzers"] = [
            AnalyzerSummary(
                name=r.analyzer,
                status="skipped" if r.skipped else ("success" if r.success else "failed"),
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
    if cancelled:
        # Persist the actual worker outcome before propagating cancellation.
        raise asyncio.CancelledError

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
