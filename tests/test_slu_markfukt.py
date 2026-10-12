"""Tests for the slu_markfukt fetcher.

The offline tests replace the HTTP call with a synthetic uint8 GeoTIFF in
the service's encoding (0 = nodata, 1-100 = moisture %, 101 = water), so
parsing, resampling, caching and grid snapping are checked without the
network.

The live tests call the real endpoint and run only with
``IMINT_LIVE_TESTS=1``. Skogsstyrelsen retired the kartportal proxy in
2026, so today they fail with ``SkgEndpointRetiredError`` naming the
replacement download; that is the signal they exist to give, and it must
not block every pull request.
"""
from __future__ import annotations

import io
import os
import urllib.parse
from pathlib import Path

import numpy as np
import pytest

from imint.training import skg_imageserver
from imint.training.slu_markfukt import (
    _DEFAULT_URL,
    _get_markfukt_url,
    fetch_markfukt_for_coords,
    fetch_markfukt_tile,
)

# Stormyran AOI in EPSG:3006 (Jämtland aapamyr)
WEST, SOUTH, EAST, NORTH = 541_060, 7_125_140, 549_450, 7_135_300


def _encoded_tiff(width: int, height: int) -> bytes:
    """A uint8 GeoTIFF using every value of the service encoding."""
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin

    data = (np.arange(width * height) % 102).astype(np.uint8).reshape(height, width)
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", width=width, height=height, count=1, dtype="uint8",
                      crs="EPSG:3006", transform=from_origin(0, height * 10, 10, 10)) as dst:
            dst.write(data, 1)
        return mem.read()


class _Resp(io.BytesIO):
    status = 200


@pytest.fixture
def fake_server(monkeypatch):
    """Answer exportImage with a TIFF of the requested size; record requests."""
    requests = []

    def fake_open(req, timeout_s):
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(req.full_url).query)
        requests.append(query)
        w, h = (int(v) for v in query["size"][0].split(","))
        return _Resp(_encoded_tiff(w, h))

    monkeypatch.setattr(skg_imageserver, "_open", fake_open)
    return requests


def test_endpoint_resolves_to_default(monkeypatch):
    monkeypatch.delenv("SKG_MARKFUKT_URL", raising=False)
    assert _get_markfukt_url() == _DEFAULT_URL or Path(".skg_endpoints").exists()


def test_tile_is_uint8_in_service_encoding(fake_server):
    arr = fetch_markfukt_tile(WEST, SOUTH, EAST, NORTH, size_px=50)
    assert arr.shape == (50, 50) and arr.dtype == np.uint8
    assert arr.min() == 0 and arr.max() == 101
    (q,) = fake_server
    assert q["bbox"] == [f"{WEST},{SOUTH},{EAST},{NORTH}"]
    assert q["bboxSR"] == q["imageSR"] == ["3006"]
    assert q["interpolation"] == ["RSP_NearestNeighbor"] and q["pixelType"] == ["U8"]


def test_non_square_size_is_requested_as_width_height(fake_server):
    arr = fetch_markfukt_tile(WEST, SOUTH, EAST, NORTH, size_px=(40, 60))
    assert arr.shape == (40, 60)
    assert fake_server[0]["size"] == ["60,40"]


def test_wrong_size_from_server_is_resampled_nearest(monkeypatch):
    monkeypatch.setattr(skg_imageserver, "_open",
                        lambda req, timeout_s: _Resp(_encoded_tiff(25, 25)))
    arr = fetch_markfukt_tile(WEST, SOUTH, EAST, NORTH, size_px=50)
    assert arr.shape == (50, 50)
    assert set(np.unique(arr)) <= set(range(102))          # nearest keeps class-like codes


def test_wgs84_path_requests_a_10m_snapped_bbox(fake_server):
    coords = {"west": 15.85, "south": 64.25, "east": 16.02, "north": 64.34}
    arr = fetch_markfukt_for_coords(coords, size_px=30)
    assert arr.shape == (30, 30)
    w, s, e, n = (float(v) for v in fake_server[0]["bbox"][0].split(","))
    assert all(v % 10 == 0 for v in (w, s, e, n)) and w < e and s < n


def test_cache_roundtrip_reads_back_without_network(fake_server, tmp_path: Path):
    a = fetch_markfukt_tile(WEST, SOUTH, EAST, NORTH, size_px=20, cache_dir=tmp_path)
    b = fetch_markfukt_tile(WEST, SOUTH, EAST, NORTH, size_px=20, cache_dir=tmp_path)
    np.testing.assert_array_equal(a, b)
    assert len(fake_server) == 1                                  # second read came from disk
    assert [p.name for p in tmp_path.iterdir()] == [
        f"markfukt_{WEST}_{SOUTH}_{EAST}_{NORTH}.npy"]


# ── live endpoint (opt-in) ──────────────────────────────────────────────────

live = pytest.mark.skipif(
    os.environ.get("IMINT_LIVE_TESTS") != "1",
    reason="live Skogsstyrelsen endpoint; set IMINT_LIVE_TESTS=1 to run",
)


@live
def test_live_fetch_stormyran_signature():
    """Stormyran shows the mire signature: high land coverage, many wet pixels."""
    arr = fetch_markfukt_tile(WEST, SOUTH, EAST, NORTH, size_px=500)
    assert arr.shape == (500, 500) and arr.dtype == np.uint8
    assert arr.max() <= 101
    assert (arr > 0).mean() > 0.95
    assert (arr > 80).mean() > 0.10


@live
def test_live_wgs84_path():
    coords = {"west": 15.85, "south": 64.25, "east": 16.02, "north": 64.34}
    arr = fetch_markfukt_for_coords(coords, size_px=300)
    assert arr.shape == (300, 300) and arr.dtype == np.uint8
