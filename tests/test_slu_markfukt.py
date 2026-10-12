"""Tests for the slu_markfukt fetcher on the open-data mirror.

A synthetic 2 m county raster in the service encoding (0 nodata, 1–100
index, 101 water, 255 fill, 65535 raster nodata) stands in for the mirror,
so the 10 m sampling, encoding, caching and grid snapping are checked
without the PVC.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from imint.training import skg_open
from imint.training.slu_markfukt import fetch_markfukt_for_coords, fetch_markfukt_tile
from tests.skg_mirror_fixture import write_raster

# Stormyran AOI in EPSG:3006 (Jämtland aapamyr), on the 10 m lattice.
WEST, SOUTH, EAST, NORTH = 541_060, 7_125_140, 549_450, 7_135_300
H_PX, W_PX = (NORTH - SOUTH) // 10, (EAST - WEST) // 10


@pytest.fixture
def mirror(monkeypatch, tmp_path: Path):
    """A 2 m markfukt raster covering the AOI with a 40 m margin."""
    monkeypatch.setenv(skg_open.ENV_MIRROR_DIR, str(tmp_path))
    w, s, e, n = WEST - 40, SOUTH - 40, EAST + 40, NORTH + 40
    rows, cols = (n - s) // 2, (e - w) // 2
    data = (np.arange(rows * cols, dtype=np.int64) % 102).reshape(rows, cols).astype(np.uint16)
    data[:50, :] = 65535            # raster nodata strip
    data[-50:, :] = 255             # source fill strip
    write_raster(tmp_path / "markfukt" / "markfukt.vrt", data, w, n, 2.0, nodata=65535)
    return data, (w, n)


def test_tile_is_uint8_in_service_encoding(mirror):
    arr = fetch_markfukt_tile(WEST, SOUTH, EAST, NORTH, size_px=(H_PX, W_PX))
    assert arr.shape == (H_PX, W_PX) and arr.dtype == np.uint8
    assert set(np.unique(arr)) <= set(range(102)) | {255}


def test_each_cell_takes_the_2m_sample_at_its_centre(mirror):
    data, (w0, n0) = mirror
    arr = fetch_markfukt_tile(WEST, SOUTH, EAST, NORTH, size_px=(H_PX, W_PX))
    # 10 m cell (i, j) spans 5x5 source cells; its centre sample is at +2, +2.
    r0, c0 = (n0 - NORTH) // 2, (WEST - w0) // 2
    expected = data[r0 + 2::5, c0 + 2::5][:H_PX, :W_PX]
    expected = np.where(expected == 65535, 0, expected).astype(np.uint8)
    np.testing.assert_array_equal(arr, expected)


def test_raster_nodata_becomes_0_and_fill_stays_255(mirror):
    arr = fetch_markfukt_tile(WEST, SOUTH, EAST, NORTH, size_px=(H_PX, W_PX))
    # 50 source rows (100 m) minus the 40 m margin = 60 m: the first 6 cells
    assert (arr[:6] == 0).all() and (arr[6] > 0).any()          # 65535 -> 0
    assert (arr[-6:] == 255).all() and (arr[-7] != 255).any()   # 255 kept for the caller


def test_outside_the_raster_is_nodata_not_an_error(mirror):
    arr = fetch_markfukt_tile(WEST - 50_000, SOUTH, WEST - 50_000 + W_PX * 10, NORTH,
                              size_px=(H_PX, W_PX))
    assert arr.shape == (H_PX, W_PX) and arr.max() == 0


def test_extent_must_match_size_px(mirror):
    with pytest.raises(ValueError, match="bbox/size_px mismatch"):
        fetch_markfukt_tile(WEST, SOUTH, EAST, NORTH, size_px=50)


def test_wgs84_path_snaps_to_the_10m_lattice(mirror):
    coords = {"west": 15.85, "south": 64.25, "east": 16.02, "north": 64.34}
    from imint.training.nmd_grid import to_nmd_grid_bounds
    b = to_nmd_grid_bounds(coords)
    assert all(v % 10 == 0 for v in b.values()) and b["west"] < b["east"]
    size = ((b["north"] - b["south"]) // 10, (b["east"] - b["west"]) // 10)
    arr = fetch_markfukt_for_coords(coords, size_px=size)
    assert arr.shape == size


def test_cache_roundtrip_reads_back_without_the_mirror(mirror, tmp_path: Path):
    cache = tmp_path / "cache"
    a = fetch_markfukt_tile(WEST, SOUTH, EAST, NORTH, size_px=(H_PX, W_PX), cache_dir=cache)
    assert [p.name for p in cache.iterdir()] == [f"markfukt_{WEST}_{SOUTH}_{EAST}_{NORTH}.npy"]
    (tmp_path / "markfukt" / "markfukt.vrt").unlink()
    b = fetch_markfukt_tile(WEST, SOUTH, EAST, NORTH, size_px=(H_PX, W_PX), cache_dir=cache)
    np.testing.assert_array_equal(a, b)


def test_missing_mirror_names_the_staging_job(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(skg_open.ENV_MIRROR_DIR, str(tmp_path))
    with pytest.raises(skg_open.SkgMirrorMissingError, match="stage_skg_open"):
        fetch_markfukt_tile(WEST, SOUTH, EAST, NORTH, size_px=(H_PX, W_PX))
