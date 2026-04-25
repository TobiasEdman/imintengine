"""Pydantic schemas for the v1 API.

Lives outside server.py so consumers can import the request/response
shapes (`from imint.api.v1.schemas import AnalyzeRequest`) without
spinning up FastAPI.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class BBox(BaseModel):
    """Geographic bounding box. WGS84 unless `crs` says otherwise."""

    west: float = Field(..., ge=-180.0, le=180.0)
    south: float = Field(..., ge=-90.0, le=90.0)
    east: float = Field(..., ge=-180.0, le=180.0)
    north: float = Field(..., ge=-90.0, le=90.0)
    crs: str = "EPSG:4326"


class AnalyzeRequest(BaseModel):
    """POST /v1/analyze body.

    A request asks ImintEngine to run analysis for one date + one bbox.
    The set of analyzers run is configured server-side via
    `config/analyzers.yaml`; clients can opt-out specific ones via
    `disable_analyzers` if needed.
    """

    date: str = Field(
        ...,
        description="ISO-8601 date, e.g. 2026-04-25. Must match an existing or fetchable scene.",
        json_schema_extra={"examples": ["2026-04-25"]},
    )
    bbox: BBox
    analyzers: list[str] | None = Field(
        default=None,
        description="Override server config. None = run all enabled analyzers.",
    )
    disable_analyzers: list[str] | None = Field(
        default=None,
        description="Names to skip even if enabled in server config.",
    )
    config_overrides: dict[str, dict[str, Any]] | None = Field(
        default=None,
        description="Per-analyzer config overrides, keyed by analyzer name.",
    )
    callback_url: str | None = Field(
        default=None,
        description="Optional webhook for completion notification (POST with JobStatusResponse).",
    )


class AnalyzeResponse(BaseModel):
    """POST /v1/analyze response."""

    job_id: str = Field(default_factory=lambda: str(uuid4()))
    status: JobStatus = JobStatus.QUEUED
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    poll_url: str = Field(
        ...,
        description="GET this URL to retrieve the result.",
    )


class AnalyzerSummary(BaseModel):
    """Per-analyzer result summary returned in the job response."""

    name: str
    status: str
    duration_ms: float | None = None
    artifacts: list[str] = Field(
        default_factory=list,
        description="Output paths produced by this analyzer (GeoTIFF / GeoJSON / PNG).",
    )
    error: str | None = None


class JobStatusResponse(BaseModel):
    """GET /v1/jobs/{id} response."""

    job_id: str
    date: str
    bbox: BBox
    status: JobStatus
    submitted_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: float | None = None
    success: bool | None = None
    analyzers: list[AnalyzerSummary] = Field(default_factory=list)
    summary_path: str | None = None
    error: str | None = None


class AnalyzerInfo(BaseModel):
    """One entry in GET /v1/analyzers."""

    name: str
    enabled: bool
    description: str | None = None
    version: str | None = None


class AnalyzersResponse(BaseModel):
    """GET /v1/analyzers response."""

    schema_version: str = Field(
        ...,
        description="Currently active des-contracts schema version.",
    )
    analyzers: list[AnalyzerInfo]


class HealthResponse(BaseModel):
    """GET /v1/health response."""

    status: str = "ok"
    api_version: str = "v1"
    schema_version: str
    embedding_identity: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
