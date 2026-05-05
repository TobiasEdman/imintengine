"""Tests for nvv_objektdata (object height/cover) fetcher."""
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from imint.training.nvv_objektdata import (
    _filename,
    _resolve_source,
    fetch_bush_cover_tile,
    fetch_bush_height_tile,
    fetch_objektdata_for_coords,
    fetch_objektdata_tile,
    fetch_tree_cover_tile,
    fetch_tree_height_tile_nvv,
)


WEST, SOUTH, EAST, NORTH = 540_000, 7_120_000, 560_000, 7_140_000


def _make_mock_tif(path: Path, value: int) -> None:
    """Build a small EPSG:3006 GeoTIFF filled with a constant value."""
    size_y, size_x = 2000, 2000
    transform = from_bounds(WEST, SOUTH, EAST, NORTH, size_x, size_y)
    arr = np.full((size_y, size_x), value, dtype=np.uint8)
    arr[:50, :] = 0  # nodata border
    with rasterio.open(
        path, "w",
        driver="GTiff", height=size_y, width=size_x,
        count=1, dtype="uint8", crs="EPSG:3006",
        transform=transform, nodata=0,
    ) as dst:
        dst.write(arr, 1)


@pytest.fixture
def mock_objektdata_root(tmp_path: Path) -> Path:
    """Build all four mock NVV objektdata GeoTIFFs."""
    obj_dir = tmp_path / "objektdata"
    obj_dir.mkdir()
    _make_mock_tif(obj_dir / _filename("hojd", "0_5_5"), value=30)      # 3.0 m bushes
    _make_mock_tif(obj_dir / _filename("hojd", "5_45"), value=180)      # 18 m trees
    _make_mock_tif(obj_dir / _filename("tackning", "0_5_5"), value=40)  # 40% bush cover
    _make_mock_tif(obj_dir / _filename("tackning", "5_45"), value=70)   # 70% tree cover
    return tmp_path


@pytest.mark.parametrize("kind,height_range,expected_filename", [
    ("hojd", "0_5_5", "Objekt_hojd_intervall_0_5_till_5_v1_3.tif"),
    ("hojd", "5_45", "Objekt_hojd_intervall_5_till_45_v1_3.tif"),
    ("tackning", "0_5_5", "Objekt_tackning_hojdintervall_0_5_till_5_v1_3.tif"),
    ("tackning", "5_45", "Objekt_tackning_hojdintervall_5_till_45_v1_3.tif"),
])
def test_filename_canonical(kind, height_range, expected_filename):
    assert _filename(kind, height_range) == expected_filename


def test_filename_invalid_kind():
    with pytest.raises(ValueError, match="unknown kind"):
        _filename("not-a-kind", "0_5_5")  # type: ignore


def test_resolve_source_missing(tmp_path: Path):
    assert _resolve_source("hojd", "0_5_5", tmp_path) is None


def test_fetch_raises_when_no_data():
    with pytest.raises(FileNotFoundError, match="NVV objektdata"):
        fetch_objektdata_tile(0, 0, 1000, 1000,
                              kind="hojd", height_range="0_5_5",
                              size_px=10, data_root="/tmp/xyz_does_not_exist")


def test_bush_height_signature(mock_objektdata_root: Path):
    arr = fetch_bush_height_tile(545_000, 7_125_000, 555_000, 7_135_000,
                                  size_px=500, data_root=mock_objektdata_root)
    assert arr.shape == (500, 500)
    assert arr.dtype == np.uint8
    assert arr.max() == 30
    assert arr.max() <= 50, "bush heights must be ≤5.0 m (50 dm)"


def test_tree_height_signature(mock_objektdata_root: Path):
    arr = fetch_tree_height_tile_nvv(545_000, 7_125_000, 555_000, 7_135_000,
                                      size_px=500, data_root=mock_objektdata_root)
    assert arr.shape == (500, 500)
    assert arr.max() == 180
    assert arr.max() >= 50, "tree heights must be ≥5.0 m (50 dm)"


def test_cover_layers_in_percent_range(mock_objektdata_root: Path):
    bush_cov = fetch_bush_cover_tile(545_000, 7_125_000, 555_000, 7_135_000,
                                      size_px=200, data_root=mock_objektdata_root)
    tree_cov = fetch_tree_cover_tile(545_000, 7_125_000, 555_000, 7_135_000,
                                      size_px=200, data_root=mock_objektdata_root)
    assert bush_cov.max() == 40
    assert tree_cov.max() == 70
    assert bush_cov.max() <= 100 and tree_cov.max() <= 100


def test_wgs84_path(mock_objektdata_root: Path):
    coords = {"west": 15.95, "south": 64.27, "east": 16.05, "north": 64.33}
    arr = fetch_objektdata_for_coords(
        coords, kind="hojd", height_range="0_5_5",
        size_px=300, data_root=mock_objektdata_root,
    )
    assert arr.shape == (300, 300)


def test_cache_roundtrip(mock_objektdata_root: Path, tmp_path: Path):
    cache = tmp_path / "cache"
    args = (545_000, 7_125_000, 555_000, 7_135_000)
    a = fetch_bush_height_tile(*args, size_px=200,
                                data_root=mock_objektdata_root, cache_dir=cache)
    b = fetch_bush_height_tile(*args, size_px=200,
                                data_root=mock_objektdata_root, cache_dir=cache)
    np.testing.assert_array_equal(a, b)


def test_cache_keys_distinct_per_layer(mock_objektdata_root: Path, tmp_path: Path):
    """Cache keys must differ between bush-height and tree-height to avoid clashes."""
    cache = tmp_path / "cache"
    args = (545_000, 7_125_000, 555_000, 7_135_000)
    fetch_bush_height_tile(*args, size_px=100,
                           data_root=mock_objektdata_root, cache_dir=cache)
    fetch_tree_height_tile_nvv(*args, size_px=100,
                                data_root=mock_objektdata_root, cache_dir=cache)
    files = sorted(p.name for p in cache.glob("*.npy"))
    assert len(files) == 2, f"expected 2 distinct cache files, got {files}"
    assert any("hojd_0_5_5_" in f for f in files)
    assert any("hojd_5_45_" in f for f in files)
