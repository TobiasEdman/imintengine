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
 - IMINT_API_MAX_PIXELS must specify an operator-validated positive pixel budget.
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
import math
import os
from pathlib import Path
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError

from imint.api.v1.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    AnalyzerInfo,
    AnalyzerSummary,
    AnalyzersResponse,
    HealthResponse,
    JobStatus,
    JobStatusResponse,
    LcluRequest,
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
_jobs_lock: asyncio.Lock | None = None


async def _put_job(job_id: str, payload: dict[str, Any]) -> None:
    if _jobs_lock is None:
        raise RuntimeError("API lifespan is not active")
    async with _jobs_lock:
        _jobs[job_id] = payload


async def _get_job(job_id: str) -> dict[str, Any] | None:
    if _jobs_lock is None:
        return _jobs.get(job_id)
    async with _jobs_lock:
        return _jobs.get(job_id)


_pending_jobs: set[asyncio.Task] = set()
_accepting_jobs = False
_analysis_slot: asyncio.Semaphore | None = None
_max_pixels: int | None = None
_max_jobs = 4
_lclu_service = None
_lclu_requests: dict[str, str] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Hold local ownership until all accepted blocking jobs finish."""
    global _accepting_jobs, _jobs_lock, _analysis_slot, _max_pixels, _max_jobs, _lclu_service
    max_pixels = _configured_max_pixels()
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
        # asyncio primitives bind on contention; never reuse across lifespans.
        _jobs_lock = asyncio.Lock()
        _analysis_slot = asyncio.Semaphore(1)
        _jobs.clear()
        _max_pixels = max_pixels
        _max_jobs = int(os.environ.get("IMINT_API_MAX_JOBS", "4"))
        if not 1 <= _max_jobs <= 64:
            raise RuntimeError("IMINT_API_MAX_JOBS must be between 1 and 64")
        _lclu_requests.clear()
        _lclu_service = None
        if manifest := os.environ.get("IMINT_LCLU_MANIFEST"):
            from imint.inference.lclu import LcluService
            _lclu_service = await asyncio.to_thread(LcluService.from_manifest, Path(manifest), root)
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
            _lclu_service = None
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
    # Routing happens inside call_next. Attach the bounded template afterward.
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
        finally:
            if _HAS_OTEL:
                # The shared instrumentation context manager yields None;
                # use OTel's current span while still inside that context.
                try:
                    from opentelemetry import trace
                    trace.get_current_span().set_attribute(
                        "route", getattr(request.scope.get("route"), "path", "unmatched"),
                    )
                except Exception:
                    pass  # Optional telemetry must not break the HTTP boundary.


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
    _validate_bbox_budget(req.bbox.model_dump())
    _check_capacity()
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
    task.add_done_callback(lambda done: _finish_job_task(resp.job_id, done))

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
        kind=job.get("kind", "analysis"),
        year=job.get("year"),
        phase=job.get("phase"),
        result=job.get("result"),
    )


# ── Worker ─────────────────────────────────────────────────────────────────


def _configured_max_pixels() -> int:
    """Require an operator-calibrated admission budget; no universal engine cap."""
    value = os.environ.get("IMINT_API_MAX_PIXELS", "").strip()
    if value.isascii() and value.isdecimal() and int(value) > 0:
        return int(value)
    raise RuntimeError("IMINT_API_MAX_PIXELS must be configured as a positive integer")


def _validate_bbox_budget(coords: dict) -> None:
    # Same projection and outward snapping as the production DES fetch path.
    # This helper performs no acquisition, credential lookup or model import.
    from imint.fetch import _to_nmd_grid, NMD_GRID_SIZE
    from rasterio.errors import RasterioError

    if _max_pixels is None:
        raise HTTPException(status_code=503, detail="API pixel budget is not configured")
    try:
        projected = _to_nmd_grid(coords)
        bounds = [projected[key] for key in ("west", "south", "east", "north")]
        if not all(math.isfinite(value) for value in bounds):
            raise ValueError("nonfinite projected bounds")
        width = int((projected["east"] - projected["west"]) // NMD_GRID_SIZE)
        height = int((projected["north"] - projected["south"]) // NMD_GRID_SIZE)
        valid = width > 0 and height > 0 and width * height <= _max_pixels
    except (ValueError, OverflowError, RasterioError):
        valid = False
    if not valid:
        raise HTTPException(
            status_code=422,
            detail=f"bbox must have a finite, nonempty projected raster of at most "
                   f"{_max_pixels} pixels on the engine's {NMD_GRID_SIZE} m grid",
        )


def _finish_job_task(job_id: str, task: asyncio.Task) -> None:
    """Finalize even tasks cancelled before their coroutine first executes."""
    _pending_jobs.discard(task)
    error = "Job cancelled before analysis completed" if task.cancelled() else task.exception()
    job = _jobs.get(job_id)
    if error is None or job is None or job["status"] not in (JobStatus.QUEUED, JobStatus.RUNNING):
        return
    # Done callbacks run on the same loop. No await/lock gap can interrupt
    # this transition; running threads have already been joined by the worker.
    completed = datetime.now(timezone.utc)
    job.update(status=JobStatus.FAILED, success=False, completed_at=completed,
               duration_ms=((completed - job["started_at"]).total_seconds() * 1000
                            if job["started_at"] else 0.0), error=str(error))


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
    if _analysis_slot is None:
        raise RuntimeError("API lifespan is not active")
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
        loop = asyncio.get_running_loop()
        if job.get("kind") == "lclu":
            def phase(value):
                loop.call_soon_threadsafe(job.update, {"phase": value})
            execute = lambda: _lclu_service.run(job["_grid"], job["year"], phase)
        else:
            execute = lambda: _execute_analysis(job["_request"], job_id)
        work = loop.run_in_executor(None, copy_context().run, execute)
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
        if job.get("kind") == "lclu":
            job.update(success=True, status=JobStatus.SUCCESS, phase="complete", result=result)
            result["artifacts"] = {name: f"/v1/lclu/jobs/{job_id}/files/{name}" for name in result["files"]}
        else:
            job["success"] = result.success
            job["error"] = result.error
            job["summary_path"] = result.summary_path
            job["analyzers"] = [
                AnalyzerSummary(
                    name=r.analyzer,
                    status="skipped" if r.skipped else ("success" if r.success else "failed"),
                    output_names=sorted(r.outputs),
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
        job["phase"] = "failed"
        # Source errors may contain private mount paths or signed remote URLs.
        job["error"] = ("Årskartan kunde inte skapas. Operatören kan kontrollera serverloggen."
                        if job.get("kind") == "lclu" else f"{type(e).__name__}: {e}")
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


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Invalid JSON numbers (NaN/Infinity) cannot be echoed by JSONResponse.
    # Do not reflect raw input or exception objects from Pydantic's context.
    return JSONResponse(status_code=422, content={"detail": [
        {"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]}
        for error in exc.errors()
    ]})


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


# Annual maps share this service's worker, queue and shutdown ownership.
def _check_capacity() -> None:
    if sum(j['status'] in (JobStatus.QUEUED, JobStatus.RUNNING) for j in _jobs.values()) >= _max_jobs:
        raise HTTPException(status_code=429, detail='Job queue is full; retry later',
                            headers={'Retry-After': '5'})


@app.get('/v1/lclu', tags=['lclu'])
async def lclu_info():
    from imint.inference.lclu import HISTORICAL_NOTE, legend, YEARS
    return {'ready': _accepting_jobs and _lclu_service is not None,
            'model': 'TESSERA R4', 'candidate_years': YEARS, 'legend': legend(),
            'historical_note': HISTORICAL_NOTE,
            'locations': _lclu_service.sources.locations() if _lclu_service else [],
            'areas': _lclu_service.sources.areas() if _lclu_service else [],
            'message': ('Välj en plats för att kontrollera årstillgång.' if _lclu_service else
                        'Tjänsten saknar verifierat R4-checkpoint och indatakatalog. Ingen klassificering körs.')}


@app.get('/v1/lclu/availability', tags=['lclu'])
async def lclu_availability(lon: float, lat: float):
    from imint.inference.lclu import Grid
    try:
        grid = Grid.from_lonlat(lon, lat)
    except ValueError:
        raise HTTPException(status_code=422, detail='Välj en plats inom Sveriges datatäckning.') from None
    years = _lclu_service.sources.years(grid) if _lclu_service else []
    return {'geometry': grid.describe(), 'years': years,
            'message': '' if years else 'Verifierade årsdata saknas för denna plats.'}


@app.post('/v1/lclu', response_model=AnalyzeResponse, status_code=202, tags=['lclu'])
async def submit_lclu(req: LcluRequest):
    from imint.inference.lclu import Grid, InputUnavailable
    from rasterio.warp import transform_bounds
    if not _accepting_jobs or _lclu_service is None:
        raise HTTPException(status_code=503, detail='Annual LCLU inputs/model are not configured')
    if _max_pixels is None or _max_pixels < 512 * 512:
        raise HTTPException(status_code=422, detail='Annual preparation requires a 512×512 pixel budget')
    grid = Grid.from_lonlat(req.lon, req.lat)
    try:
        key = _lclu_service.key(grid, req.year)
    except InputUnavailable as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    # No await between deduplication, admission and insertion: atomic on this
    # event loop even for bursts of simultaneous identical submissions.
    previous = _jobs.get(_lclu_requests.get(key, ''))
    if previous and previous['status'] != JobStatus.FAILED:
        return AnalyzeResponse(job_id=previous['job_id'], status=previous['status'],
                               submitted_at=previous['submitted_at'],
                               poll_url=f"/v1/jobs/{previous['job_id']}")
    _check_capacity()
    response = AnalyzeResponse(poll_url='')
    response.poll_url = f'/v1/jobs/{response.job_id}'
    b = grid.bounds(504)
    bbox = dict(zip(('west', 'south', 'east', 'north'), transform_bounds(
        'EPSG:3006', 'EPSG:4326', b['west'], b['south'], b['east'], b['north'])))
    _jobs[response.job_id] = {
        'job_id': response.job_id, 'date': None, 'bbox': bbox, 'kind': 'lclu',
        'year': req.year, 'status': JobStatus.QUEUED, 'phase': 'queued',
        'submitted_at': response.submitted_at, 'started_at': None, 'completed_at': None,
        'duration_ms': None, 'success': None, 'analyzers': [], 'summary_path': None,
        'error': None, 'result': None, '_request': req.model_dump(), '_grid': grid,
    }
    _lclu_requests[key] = response.job_id
    task = asyncio.create_task(_run_job_serial(response.job_id))
    _pending_jobs.add(task)
    task.add_done_callback(lambda done: _finish_job_task(response.job_id, done))
    return response


async def _lclu_artifacts(job_id: str) -> tuple[dict, Path]:
    job = await _get_job(job_id)
    service = _lclu_service
    if (not job or job.get('kind') != 'lclu' or job['status'] != JobStatus.SUCCESS
            or service is None):
        raise HTTPException(status_code=404, detail='Completed artifact not found')
    try:
        result = await asyncio.to_thread(service.cached, job['result']['cache_key'])
    except RuntimeError:
        logger.exception('Annual artifact integrity check failed for job %s', job_id)
        raise HTTPException(status_code=503, detail='Kartfilen kunde inte verifieras.') from None
    if result is None:
        raise HTTPException(status_code=404, detail='Artifact is no longer available')
    return result, service.output_root / result['cache_key']


@app.get('/v1/lclu/jobs/{job_id}/files/{name}', tags=['lclu'])
async def lclu_file(job_id: str, name: str):
    allowed = {'classification.tif': 'image/tiff', 'overlay.png': 'image/png',
               'legend.json': 'application/json', 'provenance.json': 'application/json'}
    if name not in allowed:
        raise HTTPException(status_code=404, detail='Completed artifact not found')
    _, directory = await _lclu_artifacts(job_id)
    return FileResponse(directory / name, media_type=allowed[name], filename=name)


@app.get('/v1/lclu/jobs/{job_id}/pixel', tags=['lclu'])
async def lclu_pixel(job_id: str, lon: float, lat: float):
    """Read the clicked native 10 m class; never rerun inference or sample RGB."""
    if not (math.isfinite(lon) and math.isfinite(lat) and -180 <= lon <= 180 and -90 <= lat <= 90):
        raise HTTPException(status_code=422, detail='Ogiltig koordinat.')
    result, directory = await _lclu_artifacts(job_id)
    ring = result['geometry']['footprint']['coordinates'][0]
    if not (min(p[0] for p in ring) <= lon <= max(p[0] for p in ring)
            and min(p[1] for p in ring) <= lat <= max(p[1] for p in ring)):
        return {'inside': False, 'year': result['year']}

    def sample():
        import rasterio
        from rasterio.warp import transform
        from rasterio.windows import Window
        import json
        from imint.inference.lclu import NODATA

        with rasterio.open(directory / 'classification.tif') as src:
            east, north = transform('EPSG:4326', src.crs, [lon], [lat])
            row, col = src.index(east[0], north[0])
            if not (0 <= row < src.height and 0 <= col < src.width):
                return {'inside': False, 'year': result['year']}
            value = int(src.read(1, window=Window(col, row, 1, 1))[0, 0])
        items = json.loads((directory / 'legend.json').read_text())
        item = next(item for item in items if item['id'] == value)
        return {'inside': True, 'year': result['year'], 'row': row, 'col': col,
                'class_id': value, 'name': item['name'], 'color': item['color'],
                'nodata': value == NODATA, 'note': item.get('note', '')}

    return await asyncio.to_thread(sample)


_WEB_ROOT = Path(__file__).resolve().parents[3] / 'docs'


@app.get('/lclu', include_in_schema=False)
async def lclu_page():
    return FileResponse(_WEB_ROOT / 'lclu.html')


@app.get('/lclu-assets/{name}', include_in_schema=False)
async def lclu_asset(name: str):
    files = {'lclu.js': 'js/lclu.js', 'lclu.css': 'css/lclu.css', 'styles.css': 'css/styles.css',
             'leaflet.js': 'js/vendor/leaflet.min.js', 'leaflet.css': 'css/leaflet.css'}
    if name not in files:
        raise HTTPException(status_code=404)
    return FileResponse(_WEB_ROOT / files[name])
