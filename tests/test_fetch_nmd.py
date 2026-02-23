"""
Tests for NMD data fetching, caching, and resampling.

All tests are offline — no real DES connection needed.
Cache tests pre-populate caches using the snapped EPSG:3006 key
(matching the internal _to_nmd_grid() + _nmd_cache_key() pipeline).
"""
from __future__ import annotations

import os
import numpy as np
import pytest

from imint.fetch import (
    _nmd_cache_key,
    _resample_nearest,
    _to_nmd_grid,
    fetch_nmd_data,
    NMDFetchResult,
    FetchError,
    NMD_GRID_SIZE,
    TARGET_CRS,
)


# ── Grid snapping tests ─────────────────────────────────────────────────────

class TestToNMDGrid:
    """Test _to_nmd_grid() WGS84 → EPSG:3006 + 10m snapping."""

    def test_returns_epsg3006(self):
        """Output should include crs key set to EPSG:3006."""
        coords = {"west": 13.0, "south": 55.5, "east": 13.1, "north": 55.6}
        result = _to_nmd_grid(coords)
        assert result["crs"] == TARGET_CRS
        assert "west" in result
        assert "south" in result
        assert "east" in result
        assert "north" in result

    def test_snapped_to_10m(self):
        """All bounds should be multiples of 10m."""
        coords = {"west": 13.0, "south": 55.5, "east": 13.1, "north": 55.6}
        result = _to_nmd_grid(coords)
        grid = NMD_GRID_SIZE
        assert result["west"] % grid == 0
        assert result["south"] % grid == 0
        assert result["east"] % grid == 0
        assert result["north"] % grid == 0

    def test_bbox_expands_not_shrinks(self):
        """Snapping should expand the bbox (floor for west/south, ceil for east/north)."""
        coords = {"west": 13.0, "south": 55.5, "east": 13.1, "north": 55.6}
        result = _to_nmd_grid(coords)
        # The projected bbox should be wider/taller than an unsnapped projection
        from rasterio.crs import CRS
        from rasterio.warp import transform_bounds
        w, s, e, n = transform_bounds(
            CRS.from_epsg(4326), CRS.from_epsg(3006),
            coords["west"], coords["south"], coords["east"], coords["north"],
        )
        assert result["west"] <= w
        assert result["south"] <= s
        assert result["east"] >= e
        assert result["north"] >= n

    def test_deterministic(self):
        """Same input should always produce same output."""
        coords = {"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0}
        r1 = _to_nmd_grid(coords)
        r2 = _to_nmd_grid(coords)
        assert r1 == r2

    def test_different_coords_different_output(self):
        """Different inputs should produce different outputs."""
        c1 = {"west": 13.0, "south": 55.5, "east": 13.1, "north": 55.6}
        c2 = {"west": 18.0, "south": 59.0, "east": 18.1, "north": 59.1}
        assert _to_nmd_grid(c1) != _to_nmd_grid(c2)

    def test_expansion_at_most_grid_size(self):
        """Snapping should expand at most NMD_GRID_SIZE per side."""
        coords = {"west": 13.0, "south": 55.5, "east": 13.1, "north": 55.6}
        result = _to_nmd_grid(coords)
        from rasterio.crs import CRS
        from rasterio.warp import transform_bounds
        w, s, e, n = transform_bounds(
            CRS.from_epsg(4326), CRS.from_epsg(3006),
            coords["west"], coords["south"], coords["east"], coords["north"],
        )
        grid = NMD_GRID_SIZE
        assert w - result["west"] < grid
        assert s - result["south"] < grid
        assert result["east"] - e < grid
        assert result["north"] - n < grid


# ── Cache key tests ──────────────────────────────────────────────────────────

class TestNMDCacheKey:
    """Test _nmd_cache_key() determinism and uniqueness."""

    def test_deterministic(self):
        """Same coords should always produce the same key."""
        coords = {"west": 13.0, "south": 55.5, "east": 13.1, "north": 55.6}
        assert _nmd_cache_key(coords) == _nmd_cache_key(coords)

    def test_unique_per_bbox(self):
        """Different coords should produce different keys."""
        c1 = {"west": 13.0, "south": 55.5, "east": 13.1, "north": 55.6}
        c2 = {"west": 14.0, "south": 56.5, "east": 14.1, "north": 56.6}
        assert _nmd_cache_key(c1) != _nmd_cache_key(c2)

    def test_length(self):
        """Cache key should be exactly 16 hex characters."""
        coords = {"west": 13.0, "south": 55.5, "east": 13.1, "north": 55.6}
        key = _nmd_cache_key(coords)
        assert len(key) == 16
        assert all(c in "0123456789abcdef" for c in key)

    def test_precision_matters(self):
        """Small coordinate changes should produce different keys."""
        c1 = {"west": 13.000000, "south": 55.500000, "east": 13.100000, "north": 55.600000}
        c2 = {"west": 13.000001, "south": 55.500000, "east": 13.100000, "north": 55.600000}
        assert _nmd_cache_key(c1) != _nmd_cache_key(c2)

    def test_snapped_coords_produce_integer_based_key(self):
        """Snapped EPSG:3006 coords (integers) should produce stable cache keys."""
        coords = {"west": 13.0, "south": 55.5, "east": 13.1, "north": 55.6}
        snapped = _to_nmd_grid(coords)
        # Snapped coords have integer values — cache key should be stable
        key = _nmd_cache_key(snapped)
        assert len(key) == 16
        # Same WGS84 input → same snapped → same key
        snapped2 = _to_nmd_grid(coords)
        assert _nmd_cache_key(snapped2) == key


# ── Resample tests ───────────────────────────────────────────────────────────

class TestResampleNearest:
    """Test _resample_nearest() for categorical data."""

    def test_same_shape_noop(self):
        """Same shape should return the original array."""
        arr = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        result = _resample_nearest(arr, (2, 2))
        np.testing.assert_array_equal(result, arr)

    def test_upscale(self):
        """Upscaling should preserve category values (no interpolation)."""
        arr = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        result = _resample_nearest(arr, (4, 4))
        assert result.shape == (4, 4)
        assert result.dtype == np.uint8
        # All values should be from the original set
        assert set(np.unique(result)).issubset({1, 2, 3, 4})

    def test_downscale(self):
        """Downscaling should preserve category values."""
        arr = np.ones((10, 10), dtype=np.uint8) * 42
        result = _resample_nearest(arr, (5, 5))
        assert result.shape == (5, 5)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, 42)

    def test_preserves_dtype(self):
        """Output dtype should match input dtype."""
        arr = np.array([[111, 3], [51, 61]], dtype=np.uint8)
        result = _resample_nearest(arr, (20, 20))
        assert result.dtype == np.uint8

    def test_no_fractional_values(self):
        """Nearest-neighbor should never produce fractional/interpolated values."""
        arr = np.array([[0, 255], [128, 64]], dtype=np.uint8)
        result = _resample_nearest(arr, (100, 100))
        # All values must be from the original set
        assert set(np.unique(result)).issubset({0, 64, 128, 255})


# ── fetch_nmd_data tests (mocked) ────────────────────────────────────────────

class TestFetchNMDData:
    """Test fetch_nmd_data() with mocked DES connection.

    Cache tests pre-populate using the snapped EPSG:3006 key
    (matching the internal _to_nmd_grid() + _nmd_cache_key() pipeline).
    """

    def _snapped_cache_key(self, coords: dict) -> str:
        """Get the cache key that fetch_nmd_data will actually use."""
        snapped = _to_nmd_grid(coords)
        return _nmd_cache_key(snapped)

    def test_cache_hit(self, tmp_path):
        """Cached NMD data should be returned without DES connection."""
        coords = {"west": 13.0, "south": 55.5, "east": 13.1, "north": 55.6}
        cache_dir = str(tmp_path / "nmd_cache")
        os.makedirs(cache_dir, exist_ok=True)

        # Pre-populate cache with the snapped key
        cache_key = self._snapped_cache_key(coords)
        nmd_data = np.full((50, 50), 111, dtype=np.uint8)
        np.save(os.path.join(cache_dir, f"nmd_{cache_key}.npy"), nmd_data)

        result = fetch_nmd_data(coords=coords, cache_dir=cache_dir)
        assert result.from_cache is True
        assert result.nmd_raster.shape == (50, 50)
        np.testing.assert_array_equal(result.nmd_raster, 111)

    def test_cache_hit_with_resample(self, tmp_path):
        """Cached data should be resampled to target_shape."""
        coords = {"west": 13.0, "south": 55.5, "east": 13.1, "north": 55.6}
        cache_dir = str(tmp_path / "nmd_cache")
        os.makedirs(cache_dir, exist_ok=True)

        cache_key = self._snapped_cache_key(coords)
        nmd_data = np.full((50, 50), 3, dtype=np.uint8)  # Cropland
        np.save(os.path.join(cache_dir, f"nmd_{cache_key}.npy"), nmd_data)

        result = fetch_nmd_data(coords=coords, target_shape=(100, 100), cache_dir=cache_dir)
        assert result.from_cache is True
        assert result.nmd_raster.shape == (100, 100)
        assert result.nmd_raster.dtype == np.uint8
        np.testing.assert_array_equal(result.nmd_raster, 3)

    def test_cache_miss_no_connection(self, tmp_path, monkeypatch):
        """Without cache and no DES connection, should raise FetchError."""
        import imint.fetch as fetch_module

        def _mock_connect(**kwargs):
            raise FetchError("No DES connection (mock)")

        monkeypatch.setattr(fetch_module, "_connect", _mock_connect)

        # Use valid Swedish coords (within EPSG:3006 bounds)
        coords = {"west": 15.0, "south": 57.0, "east": 15.1, "north": 57.1}
        cache_dir = str(tmp_path / "empty_cache")

        with pytest.raises(FetchError, match="No DES connection"):
            fetch_nmd_data(coords=coords, cache_dir=cache_dir)

    def test_result_dataclass(self):
        """NMDFetchResult should have correct fields."""
        raster = np.zeros((10, 10), dtype=np.uint8)
        result = NMDFetchResult(nmd_raster=raster, from_cache=True)
        assert result.nmd_raster is raster
        assert result.from_cache is True
        assert result.crs is None
        assert result.transform is None


# ── Export tests ──────────────────────────────────────────────────────────────

class TestNMDExports:
    """Test NMD export functions."""

    def test_save_nmd_overlay(self, tmp_path):
        """NMD overlay should create a valid RGB PNG with L2 palette."""
        from imint.exporters.export import save_nmd_overlay
        from PIL import Image

        l2_raster = np.zeros((20, 20), dtype=np.uint8)
        l2_raster[:10, :] = 1    # forest_pine (dark green)
        l2_raster[10:, :] = 18   # water_lakes (blue)

        path = str(tmp_path / "nmd_overlay.png")
        save_nmd_overlay(l2_raster, path)

        assert os.path.exists(path)
        img = Image.open(path)
        assert img.size == (20, 20)
        assert img.mode == "RGB"

        # Check colors
        arr = np.array(img)
        # forest_pine pixels should be dark green (0, 100, 0)
        np.testing.assert_array_equal(arr[0, 0], [0, 100, 0])
        # water_lakes pixels should be blue (0, 0, 255)
        np.testing.assert_array_equal(arr[15, 0], [0, 0, 255])

    def test_save_nmd_stats(self, tmp_path):
        """NMD stats JSON should be valid and contain expected keys."""
        import json
        from imint.exporters.export import save_nmd_stats

        stats = {
            "level1": {
                "forest": {"pixel_count": 100, "fraction": 0.5},
                "water": {"pixel_count": 100, "fraction": 0.5},
            },
            "level2": {
                "forest_pine": {"pixel_count": 100, "fraction": 0.5},
                "water_lakes": {"pixel_count": 100, "fraction": 0.5},
            },
        }
        cross_ref = {
            "spectral": {"forest": {"mean_ndvi": 0.7}},
        }

        path = str(tmp_path / "nmd_stats.json")
        save_nmd_stats(stats, cross_ref, path)

        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)

        assert "class_stats" in data
        assert "cross_reference" in data
        assert data["class_stats"]["level1"]["forest"]["fraction"] == 0.5
        assert data["cross_reference"]["spectral"]["forest"]["mean_ndvi"] == 0.7

    def test_save_nmd_stats_no_cross_ref(self, tmp_path):
        """NMD stats should work without cross-reference data."""
        import json
        from imint.exporters.export import save_nmd_stats

        stats = {"level1": {}, "level2": {}}
        path = str(tmp_path / "nmd_stats.json")
        save_nmd_stats(stats, None, path)

        with open(path) as f:
            data = json.load(f)
        assert data["cross_reference"] == {}


# ── GeoContext export tests ──────────────────────────────────────────────────

class TestGeoContextExports:
    """Test export functions with GeoContext parameter."""

    def _make_geo(self):
        """Create a realistic GeoContext for testing."""
        from rasterio.transform import from_bounds
        from imint.job import GeoContext

        # 100x100 pixel raster at 10m resolution in EPSG:3006
        west, south = 470000, 6240000
        east, north = 471000, 6241000  # 1km x 1km
        h, w = 100, 100
        transform = from_bounds(west, south, east, north, w, h)

        return GeoContext(
            crs="EPSG:3006",
            transform=transform,
            bounds_projected={"west": west, "south": south, "east": east, "north": north},
            bounds_wgs84={"west": 14.5, "south": 56.3, "east": 14.515, "north": 56.309},
            shape=(h, w),
        )

    def test_save_geotiff_with_geo(self, tmp_path):
        """GeoTIFF with GeoContext should use EPSG:3006 CRS."""
        import rasterio
        from imint.exporters.export import save_geotiff

        geo = self._make_geo()
        array = np.zeros((100, 100), dtype=np.uint8)
        path = str(tmp_path / "test.tif")

        save_geotiff(array, path, geo=geo)

        assert os.path.exists(path)
        with rasterio.open(path) as src:
            assert str(src.crs) == "EPSG:3006"
            assert src.transform == geo.transform
            assert src.width == 100
            assert src.height == 100

    def test_save_geotiff_without_geo_fallback(self, tmp_path):
        """GeoTIFF without GeoContext should fall back to WGS84."""
        import rasterio
        from imint.exporters.export import save_geotiff

        array = np.zeros((50, 50), dtype=np.uint8)
        coords = {"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0}
        path = str(tmp_path / "test_fallback.tif")

        save_geotiff(array, path, coords=coords)

        assert os.path.exists(path)
        with rasterio.open(path) as src:
            assert str(src.crs) == "EPSG:4326"

    def test_save_regions_geojson_with_geo(self, tmp_path):
        """GeoJSON with GeoContext should produce WGS84 coordinates."""
        import json
        from imint.exporters.export import save_regions_geojson

        geo = self._make_geo()
        regions = [
            {"bbox": {"x_min": 10, "y_min": 10, "x_max": 20, "y_max": 20}, "label": "test"},
        ]
        path = str(tmp_path / "test.geojson")

        save_regions_geojson(regions, path, geo=geo)

        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)

        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) == 1

        # Coordinates should be in WGS84 range (Sweden: lon ~10-25, lat ~55-70)
        coords = data["features"][0]["geometry"]["coordinates"][0]
        for lon, lat in coords:
            assert 10.0 < lon < 25.0, f"lon {lon} out of Sweden range"
            assert 55.0 < lat < 70.0, f"lat {lat} out of Sweden range"

    def test_save_regions_geojson_legacy_fallback(self, tmp_path):
        """GeoJSON without GeoContext should use legacy WGS84 interpolation."""
        import json
        from imint.exporters.export import save_regions_geojson

        regions = [
            {"bbox": {"x_min": 0, "y_min": 0, "x_max": 50, "y_max": 50}, "label": "test"},
        ]
        coords = {"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0}
        path = str(tmp_path / "test_legacy.geojson")

        save_regions_geojson(regions, path, coords=coords, image_shape=(100, 100, 3))

        with open(path) as f:
            data = json.load(f)

        polygon = data["features"][0]["geometry"]["coordinates"][0]
        # x_min=0, x_max=50 out of 100 → lon should be 14.5 to 15.0
        assert abs(polygon[0][0] - 14.5) < 0.01
        assert abs(polygon[1][0] - 15.0) < 0.01

    def test_save_regions_geojson_pixel_fallback(self, tmp_path):
        """GeoJSON without coords should use pixel coordinates."""
        import json
        from imint.exporters.export import save_regions_geojson

        regions = [
            {"bbox": {"x_min": 5, "y_min": 10, "x_max": 15, "y_max": 20}, "label": "test"},
        ]
        path = str(tmp_path / "test_pixel.geojson")

        save_regions_geojson(regions, path)

        with open(path) as f:
            data = json.load(f)

        polygon = data["features"][0]["geometry"]["coordinates"][0]
        # Should use raw pixel values
        assert polygon[0] == [5, 10]
        assert polygon[2] == [15, 20]
