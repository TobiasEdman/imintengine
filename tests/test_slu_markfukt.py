"""Tests for slu_markfukt fetcher.

These are integration tests against the live Skogsstyrelsen ImageServer
proxy. They are skipped automatically when offline.
"""
from __future__ import annotations

import socket
from pathlib import Path

import numpy as np
import pytest

from imint.training.slu_markfukt import (
    _DEFAULT_URL,
    _get_markfukt_url,
    fetch_markfukt_for_coords,
    fetch_markfukt_tile,
)


def _have_network() -> bool:
    try:
        socket.create_connection(("kartor.skogsstyrelsen.se", 443), timeout=3)
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _have_network(),
    reason="requires network access to kartor.skogsstyrelsen.se",
)


# Stormyran AOI in EPSG:3006 (Jämtland aapamyr)
WEST, SOUTH, EAST, NORTH = 541_060, 7_125_140, 549_450, 7_135_300


def test_endpoint_resolves_to_default():
    """Without env var or .skg_endpoints, falls back to public default."""
    assert _get_markfukt_url() in (
        _DEFAULT_URL,
        # Or a configured override, both are valid
    )


def test_fetch_stormyran_signature():
    """Stormyran AOI shows myr signature: high non-zero coverage + bimodal."""
    arr = fetch_markfukt_tile(WEST, SOUTH, EAST, NORTH, size_px=500)
    assert arr.shape == (500, 500)
    assert arr.dtype == np.uint8
    assert arr.max() <= 101, f"unexpected max {arr.max()}"
    assert arr.min() == 0

    # Land coverage check (Stormyran is inland, ≥95% should be non-zero)
    land_pct = (arr > 0).mean()
    assert land_pct > 0.95, f"land coverage {land_pct:.1%}, expected >95%"

    # Wetland signature: substantial high-moisture pixels (myr)
    high = (arr > 80).mean()
    assert high > 0.10, (
        f"high-moisture pixels {high:.1%}, expected >10% in aapamyr area"
    )


def test_wgs84_path():
    """WGS84 coords are reprojected and snapped to NMD grid."""
    coords = {"west": 15.85, "south": 64.25, "east": 16.02, "north": 64.34}
    arr = fetch_markfukt_for_coords(coords, size_px=300)
    assert arr.shape == (300, 300)
    assert arr.dtype == np.uint8


def test_cache_roundtrip(tmp_path: Path):
    """Cached read is bit-identical to fresh read."""
    args = (WEST, SOUTH, EAST, NORTH)
    a = fetch_markfukt_tile(*args, size_px=200, cache_dir=tmp_path)
    b = fetch_markfukt_tile(*args, size_px=200, cache_dir=tmp_path)
    np.testing.assert_array_equal(a, b)
    assert list(tmp_path.glob("markfukt_*.npy"))


def test_non_square_size():
    """size_px tuple supports non-square output."""
    arr = fetch_markfukt_tile(WEST, SOUTH, EAST, NORTH, size_px=(400, 600))
    assert arr.shape == (400, 600)
