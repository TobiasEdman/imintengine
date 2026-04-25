"""Tests for the v1 HTTP API.

Schema-only tests run without ImintEngine itself loaded (no torch / numpy
in the test process). Endpoint tests use FastAPI's TestClient against a
fully-imported app — they exercise the boundary, not the full pipeline.
"""
from __future__ import annotations

import pytest

# All API tests need fastapi + httpx; skip cleanly if not installed.
fastapi = pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient").TestClient


def test_schemas_validate_minimal_request():
    from imint.api.v1.schemas import AnalyzeRequest, BBox

    req = AnalyzeRequest(
        date="2026-04-25",
        bbox=BBox(west=11.5, south=58.0, east=12.0, north=58.5),
    )
    assert req.date == "2026-04-25"
    assert req.bbox.crs == "EPSG:4326"


def test_bbox_rejects_out_of_range():
    from imint.api.v1.schemas import BBox

    with pytest.raises(Exception):
        BBox(west=-200, south=0, east=0, north=0)


def test_health_endpoint():
    from imint.api.v1 import app

    client = TestClient(app)
    r = client.get("/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["api_version"] == "v1"
    assert "schema_version" in body
    assert "embedding_identity" in body


def test_analyzers_endpoint_returns_list():
    from imint.api.v1 import app

    client = TestClient(app)
    r = client.get("/v1/analyzers")
    # If ANALYZER_REGISTRY can't be loaded (e.g. torch missing in test env),
    # we expect 503 — that's acceptable. Otherwise expect 200 with the list.
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        body = r.json()
        assert "analyzers" in body
        assert "schema_version" in body


def test_unknown_job_returns_404():
    from imint.api.v1 import app

    client = TestClient(app)
    r = client.get("/v1/jobs/does-not-exist")
    assert r.status_code == 404
    body = r.json()
    assert "does-not-exist" in body["detail"]


def test_analyze_returns_202_with_poll_url():
    """Submit a job; we don't wait for it to finish (would need image fetch).
    Just confirm the boundary returns 202 + a poll_url."""
    from imint.api.v1 import app

    client = TestClient(app)
    r = client.post(
        "/v1/analyze",
        json={
            "date": "2026-04-25",
            "bbox": {"west": 11.5, "south": 58.0, "east": 12.0, "north": 58.5},
        },
    )
    assert r.status_code == 202
    body = r.json()
    assert "job_id" in body
    assert body["status"] == "queued"
    assert body["poll_url"] == f"/v1/jobs/{body['job_id']}"

    # Polling immediately should return queued or running, not 404.
    r = client.get(body["poll_url"])
    assert r.status_code == 200
    assert r.json()["status"] in ("queued", "running", "success", "failed")


def test_openapi_schema_published():
    from imint.api.v1 import app

    client = TestClient(app)
    r = client.get("/v1/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    assert spec["info"]["title"] == "ImintEngine API"
    paths = spec.get("paths", {})
    assert "/v1/health" in paths
    assert "/v1/analyzers" in paths
    assert "/v1/analyze" in paths
    assert "/v1/jobs/{job_id}" in paths
    # /v1/metrics is excluded from OpenAPI on purpose (W4.5).
    assert "/v1/metrics" not in paths


def test_response_includes_request_id_header():
    """W4.5: every response carries an X-Request-Id for trace stitching."""
    from imint.api.v1 import app

    client = TestClient(app)
    r = client.get("/v1/health")
    assert r.status_code == 200
    rid = r.headers.get("X-Request-Id")
    assert rid is not None and len(rid) >= 8


def test_response_includes_response_time_header():
    """W4.5: X-Response-Time-MS aids client-side SLA tracking."""
    from imint.api.v1 import app

    client = TestClient(app)
    r = client.get("/v1/health")
    assert r.status_code == 200
    rt = r.headers.get("X-Response-Time-MS")
    assert rt is not None
    assert float(rt) >= 0.0


def test_metrics_endpoint_reachable():
    """W4.5: /v1/metrics returns text/plain even without prometheus_client."""
    from imint.api.v1 import app

    client = TestClient(app)
    r = client.get("/v1/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
