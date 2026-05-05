"""Tests for nvv_smi (Markfuktighetsindex) fetcher."""
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from imint.training.nvv_smi import (
    fetch_smi_for_coords,
    fetch_smi_tile,
    _resolve_root,
    _resolve_sources,
)


# AOI in EPSG:3006 (synthetic, ~Stormyran extent: 20×20 km)
WEST, SOUTH, EAST, NORTH = 540_000, 7_120_000, 560_000, 7_140_000


@pytest.fixture
def mock_smi_root(tmp_path: Path) -> Path:
    """Build a mock Markfuktighetsindex GeoTIFF at the expected location."""
    smi_dir = tmp_path / "markfuktighet"
    smi_dir.mkdir()

    size_y, size_x = 2000, 2000
    y, x = np.ogrid[:size_y, :size_x]
    cy, cx = size_y / 2, size_x / 2
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    arr = np.clip(80 - dist * 0.05, 0, 100).astype(np.uint8)
    arr[:50, :] = 255  # nodata border
    arr[-50:, :] = 255

    transform = from_bounds(WEST, SOUTH, EAST, NORTH, size_x, size_y)
    tif_path = smi_dir / "Markfuktighetsindex_NMD_Sverige.tif"
    with rasterio.open(
        tif_path, "w",
        driver="GTiff", height=size_y, width=size_x,
        count=1, dtype="uint8", crs="EPSG:3006",
        transform=transform, nodata=255,
    ) as dst:
        dst.write(arr, 1)

    return tmp_path


def test_resolve_root_default():
    """Default root resolves to ~/imint_data/nvv_aux."""
    root = _resolve_root(None)
    assert str(root).endswith("imint_data/nvv_aux")


def test_resolve_sources_missing_dir():
    """Missing directory returns empty list (not crash)."""
    assert _resolve_sources("/tmp/does_not_exist_xyz_12345") == []


def test_fetch_raises_when_no_data():
    """FileNotFoundError with helpful message when no GeoTIFF present."""
    with pytest.raises(FileNotFoundError, match="Markfuktighetsindex"):
        fetch_smi_tile(0, 0, 1000, 1000, size_px=10,
                       data_root="/tmp/does_not_exist_xyz_12345")


def test_fetch_3006_bbox(mock_smi_root: Path):
    """EPSG:3006 path returns correct shape, dtype, value range."""
    arr = fetch_smi_tile(545_000, 7_125_000, 555_000, 7_135_000,
                         size_px=1000, data_root=mock_smi_root)
    assert arr.shape == (1000, 1000)
    assert arr.dtype == np.uint8
    assert arr.max() <= 100, "nodata 255 should be mapped to 0"
    assert arr.mean() > 30, f"expected non-trivial moisture, got {arr.mean():.1f}"


def test_fetch_wgs84_coords(mock_smi_root: Path):
    """WGS84 path snaps to EPSG:3006 grid and reads window."""
    coords = {"west": 15.95, "south": 64.27, "east": 16.05, "north": 64.33}
    arr = fetch_smi_for_coords(coords, size_px=500, data_root=mock_smi_root)
    assert arr.shape == (500, 500)
    assert arr.dtype == np.uint8


def test_cache_roundtrip(mock_smi_root: Path, tmp_path: Path):
    """Cached read is bit-identical to fresh read."""
    cache_dir = tmp_path / "cache"
    args = (545_000, 7_125_000, 555_000, 7_135_000)
    a = fetch_smi_tile(*args, size_px=200, data_root=mock_smi_root, cache_dir=cache_dir)
    b = fetch_smi_tile(*args, size_px=200, data_root=mock_smi_root, cache_dir=cache_dir)
    np.testing.assert_array_equal(a, b)


def test_non_overlapping_aoi_returns_zeros(mock_smi_root: Path):
    """AOI fully outside source extent returns zeros, no crash."""
    arr = fetch_smi_tile(100_000, 7_000_000, 110_000, 7_010_000,
                         size_px=100, data_root=mock_smi_root)
    assert (arr == 0).all()


def test_non_square_size(mock_smi_root: Path):
    """size_px tuple supports non-square output."""
    arr = fetch_smi_tile(545_000, 7_125_000, 555_000, 7_135_000,
                         size_px=(800, 1000), data_root=mock_smi_root)
    assert arr.shape == (800, 1000)
