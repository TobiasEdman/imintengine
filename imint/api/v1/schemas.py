"""Pydantic schemas for the v1 API.

Lives outside server.py so consumers can import the request/response
shapes (`from imint.api.v1.schemas import AnalyzeRequest`) without
spinning up FastAPI.
"""
from __future__ import annotations

from datetime import date as calendar_date, datetime, timezone
from typing import Literal
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class BBox(BaseModel):
    """Ordered, nonempty WGS84 bounding box; antimeridian spans unsupported."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    west: float = Field(..., ge=-180.0, le=180.0)
    south: float = Field(..., ge=-90.0, le=90.0)
    east: float = Field(..., ge=-180.0, le=180.0)
    north: float = Field(..., ge=-90.0, le=90.0)
    crs: Literal["EPSG:4326"] = "EPSG:4326"

    @model_validator(mode="after")
    def ordered_bounds(self) -> "BBox":
        if self.west >= self.east or self.south >= self.north:
            raise ValueError("bbox must satisfy west < east and south < north")
        return self


class AnalyzeRequest(BaseModel):
    """POST /v1/analyze body.

    A request asks ImintEngine to run analysis for one date + one bbox.
    The set of analyzers run is configured server-side via
    `config/analyzers.yaml`. Per-request analyzer controls and callbacks
    are unsupported and rejected as extra fields (HTTP 422).
    """

    model_config = ConfigDict(extra="forbid")

    date: str = Field(
        ...,
        description="ISO-8601 date, e.g. 2026-04-25. Must match an existing or fetchable scene.",
        json_schema_extra={"examples": ["2026-04-25"]},
    )
    bbox: BBox

    @field_validator("date")
    @classmethod
    def exact_date(cls, value: str) -> str:
        # The engine uses this value in output filenames: accept only a date.
        parsed = calendar_date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise ValueError("date must use YYYY-MM-DD")
        return value



class AnalyzeResponse(BaseModel):
    """POST /v1/analyze response."""

    job_id: str = Field(default_factory=lambda: str(uuid4()))
    status: JobStatus = JobStatus.QUEUED
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    poll_url: str = Field(
        ...,
        description="GET this URL to retrieve the result.",
    )


class LcluRequest(BaseModel):
    """One annual native-grid map; no dates, paths or provider overrides."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    lon: float = Field(ge=10, le=25)
    lat: float = Field(ge=55, le=70)
    year: int = Field(ge=2017, le=2025)


class AnalyzerSummary(BaseModel):
    """Per-analyzer result summary returned in the job response."""

    name: str
    status: str
    duration_ms: float | None = Field(
        default=None, description="Unknown: the engine does not record per-analyzer timing.",
    )
    artifacts: list[str] = Field(
        default_factory=list,
        description="Reserved output paths; empty until the engine supplies an artifact manifest.",
    )
    output_names: list[str] = Field(
        default_factory=list,
        description="Keys in AnalysisResult.outputs; raster values are not embedded in this summary.",
    )
    error: str | None = None


class JobStatusResponse(BaseModel):
    """GET /v1/jobs/{id} response."""

    job_id: str
    date: str | None = None
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
    kind: Literal["analysis", "lclu"] = "analysis"
    year: int | None = None
    phase: str | None = None
    result: dict | None = None


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
