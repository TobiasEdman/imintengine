"""SLA telemetry for the v1 API (W4.5).

Three layers, fail-open at every step:

  1. Structured access log (always on) — one JSON line per request to
     stderr. Parseable by any log-shipper (Loki, Vector, fluent-bit).
     Fields: ts, method, path, status, duration_ms, user, request_id,
     data_currency_hours.

  2. Prometheus metrics (only if opentelemetry-exporter-prometheus or
     prometheus-client installed) — request_duration_seconds histogram
     with method/path/status/user labels. Exposed at /v1/metrics.

  3. OTel spans (only if opentelemetry installed) — already wired by
     the existing telemetry_middleware in server.py.

Per agentic_workflow rollout plan W4.5. The "user" label is left as
"anonymous" until auth lands; the field is in place so adding OAuth2
/ API-key middleware doesn't require a telemetry rewrite.
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any

# Per-request context set by middleware so log lines can pick up the user
# without threading it through every call site.
current_user: ContextVar[str] = ContextVar("current_user", default="anonymous")
current_request_id: ContextVar[str] = ContextVar("current_request_id", default="")


# ── Prometheus (optional) ──────────────────────────────────────────────────
_HAS_PROM = False
_request_histogram = None
_request_counter = None
_data_currency_gauge = None

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _request_histogram = Histogram(
        "imint_api_request_duration_seconds",
        "Latency of /v1/* requests, by route + method + status + user",
        ["method", "route", "status", "user"],
        # buckets tuned for analysis-job submission + polling: most calls
        # are <1s (status, /analyzers, /health). /analyze itself is async.
        buckets=(0.005, 0.025, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    )
    _request_counter = Counter(
        "imint_api_requests_total",
        "Total /v1/* requests",
        ["method", "route", "status", "user"],
    )
    _data_currency_gauge = Gauge(
        "imint_api_data_currency_hours",
        "Age of the data feeding the most-recent /v1/analyze request, "
        "in hours since last successful fetch (W4.2 cron).",
    )
    _HAS_PROM = True
except ImportError:
    pass


def is_enabled() -> bool:
    """True when prometheus_client is importable."""
    return _HAS_PROM


def emit_access_log(
    *,
    method: str,
    path: str,
    status: int,
    duration_ms: float,
    user: str | None = None,
    request_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Emit one structured JSON access-log line to stderr.

    Always on. Stdlib only — no external sink. Production captures via
    the container runtime's stderr stream.
    """
    record: dict[str, Any] = {
        "ts": time.time(),
        "method": method,
        "path": path,
        "status": status,
        "duration_ms": round(duration_ms, 2),
        "user": user or current_user.get(),
        "request_id": request_id or current_request_id.get() or "",
    }
    if extra:
        record.update(extra)
    try:
        sys.stderr.write(json.dumps(record, ensure_ascii=False) + "\n")
        sys.stderr.flush()
    except Exception:
        # Never let telemetry break the request handler.
        pass


def record_request(
    *,
    method: str,
    route: str,
    status: int,
    duration_ms: float,
    user: str | None = None,
) -> None:
    """Record one request in the Prometheus histogram + counter.

    `route` should be the parameterised route (e.g. "/v1/jobs/{job_id}"),
    not the rendered URL — keeps cardinality bounded.
    """
    if not _HAS_PROM:
        return
    user = user or current_user.get()
    labels = {
        "method": method,
        "route": route,
        "status": str(status),
        "user": user,
    }
    try:
        _request_histogram.labels(**labels).observe(duration_ms / 1000.0)
        _request_counter.labels(**labels).inc()
    except Exception:
        pass


def record_data_currency_hours(hours: float) -> None:
    """Update the data-currency gauge with the freshest known fetch age."""
    if not _HAS_PROM or _data_currency_gauge is None:
        return
    try:
        _data_currency_gauge.set(hours)
    except Exception:
        pass


def metrics_payload() -> tuple[bytes, str]:
    """Return (body, content_type) for the /v1/metrics endpoint.

    Empty body + text/plain when prometheus_client isn't installed —
    keeps the endpoint reachable for liveness, just emits nothing.
    """
    if not _HAS_PROM:
        return b"# prometheus_client not installed\n", "text/plain; charset=utf-8"
    try:
        return generate_latest(), CONTENT_TYPE_LATEST
    except Exception:
        return b"# generate_latest failed\n", "text/plain; charset=utf-8"


def new_request_id() -> str:
    """Generate a new request id; set both as the contextvar and return."""
    rid = uuid.uuid4().hex[:16]
    current_request_id.set(rid)
    return rid
