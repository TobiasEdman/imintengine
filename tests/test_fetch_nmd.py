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
    read_nmd2023_window,
    NMDFetchResult,
    NMD2023RasterMissingError,
    NMD2023_DEFAULT_PATH,
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

        result = fetch_nmd_data(coords=coords, cache_dir=cache_dir, product=2018)
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

        result = fetch_nmd_data(coords=coords, target_shape=(100, 100), cache_dir=cache_dir,
                                product=2018)
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
            fetch_nmd_data(coords=coords, cache_dir=cache_dir, product=2018)

    def test_result_dataclass(self):
        """NMDFetchResult should have correct fields."""
        raster = np.zeros((10, 10), dtype=np.uint8)
        result = NMDFetchResult(nmd_raster=raster, from_cache=True)
        assert result.nmd_raster is raster
        assert result.from_cache is True
        assert result.crs is None
        assert result.transform is None

    def test_product_is_required_and_checked(self):
        coords = {"west": 15.0, "south": 57.0, "east": 15.1, "north": 57.1}
        with pytest.raises(TypeError, match="product"):
            fetch_nmd_data(coords=coords)
        for product in ("2023", 2020, None):
            with pytest.raises(ValueError, match="NMD product must be one of"):
                fetch_nmd_data(coords=coords, product=product)

    def test_2018_refuses_a_raster_path(self):
        coords = {"west": 15.0, "south": 57.0, "east": 15.1, "north": 57.1}
        with pytest.raises(ValueError, match="raster_path applies to NMD 2023 only"):
            fetch_nmd_data(coords=coords, product=2018, raster_path="nmd.tif")


# ── NMD 2023 (local national raster) ─────────────────────────────────────────

# The fire showcase AOI in WGS84; the raster is built around its snapped bbox.
_FIRE_WGS84 = {"west": 15.42, "south": 61.92, "east": 15.47, "north": 61.97}


def _write_nmd2023(path, west, north, data, *, dtype="uint16", crs="EPSG:3006"):
    """Write a synthetic NMD 2023-like GeoTIFF on the 10 m lattice."""
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(
        path, "w", driver="GTiff", width=data.shape[1], height=data.shape[0], count=1,
        dtype=dtype, crs=crs, transform=from_origin(west, north, 10, 10),
    ) as dst:
        dst.write(data.astype(dtype), 1)
    return str(path)


class TestFetchNMD2023:
    """NMD 2023 is a window of the national GeoTIFF, never DES or the 2018 cache."""

    MARGIN = 5  # px of raster beyond the bbox on every side

    @pytest.fixture
    def bbox(self):
        return _to_nmd_grid(_FIRE_WGS84)

    @pytest.fixture
    def raster(self, tmp_path, bbox):
        """A raster around the bbox: 4211 everywhere, 62 in the bbox's first row,
        and 4233 in its last column, so values above 255 and placement are both checked."""
        m = self.MARGIN
        h = (bbox["north"] - bbox["south"]) // 10
        w = (bbox["east"] - bbox["west"]) // 10
        data = np.full((h + 2 * m, w + 2 * m), 4211, dtype=np.uint16)
        data[m, m:m + w] = 62
        data[m:m + h, m + w - 1] = 4233
        path = _write_nmd2023(
            tmp_path / "NMD2023bas_test.tif", bbox["west"] - 10 * m, bbox["north"] + 10 * m, data)
        return path, data[m:m + h, m:m + w]

    @pytest.fixture(autouse=True)
    def _no_des(self, monkeypatch):
        import imint.fetch as fetch_module

        def _connect(**kwargs):
            raise AssertionError("NMD 2023 must not contact DES")

        monkeypatch.setattr(fetch_module, "_connect", _connect)
        monkeypatch.delenv("NMD2023_PATH", raising=False)

    def test_codes_above_255_survive_as_uint16(self, raster, bbox):
        path, expected = raster
        result = fetch_nmd_data(coords=_FIRE_WGS84, product=2023, raster_path=path)
        assert result.nmd_raster.dtype == np.uint16
        np.testing.assert_array_equal(result.nmd_raster, expected)
        assert result.nmd_raster[5, 5] == 4211   # would be 115 through uint8
        assert result.nmd_raster[5, -1] == 4233
        assert result.from_cache is False
        assert result.crs.to_epsg() == 3006
        assert (result.transform.c, result.transform.f) == (bbox["west"], bbox["north"])
        assert (result.transform.a, result.transform.e) == (10, -10)

    def test_never_served_from_the_2018_cache(self, raster, tmp_path, monkeypatch):
        """A 2018 entry for the same bbox exists; 2023 must ignore it and write none."""
        import imint.fetch as fetch_module

        path, expected = raster
        cache_dir = tmp_path / "nmd_cache"
        cache_dir.mkdir()
        key = _nmd_cache_key(_to_nmd_grid(_FIRE_WGS84))
        np.save(cache_dir / f"nmd_{key}.npy", np.full(expected.shape, 111, dtype=np.uint8))
        monkeypatch.setattr(fetch_module, "NMD_CACHE_DIR", str(cache_dir))

        result = fetch_nmd_data(coords=_FIRE_WGS84, product=2023, raster_path=path)
        np.testing.assert_array_equal(result.nmd_raster, expected)
        assert sorted(p.name for p in cache_dir.iterdir()) == [f"nmd_{key}.npy"]
        cached_2018 = fetch_nmd_data(coords=_FIRE_WGS84, product=2018)
        assert cached_2018.from_cache is True
        np.testing.assert_array_equal(cached_2018.nmd_raster, 111)
        with pytest.raises(ValueError, match="2018 \\(DES\\) only"):
            fetch_nmd_data(coords=_FIRE_WGS84, product=2023, raster_path=path,
                           cache_dir=str(cache_dir))

    def test_path_comes_from_the_environment_and_the_argument_wins(self, raster, tmp_path,
                                                                    monkeypatch):
        path, expected = raster
        monkeypatch.setenv("NMD2023_PATH", path)
        np.testing.assert_array_equal(
            fetch_nmd_data(coords=_FIRE_WGS84, product=2023).nmd_raster, expected)
        with pytest.raises(NMD2023RasterMissingError, match="elsewhere.tif"):
            fetch_nmd_data(coords=_FIRE_WGS84, product=2023,
                           raster_path=str(tmp_path / "elsewhere.tif"))

    def test_missing_raster_is_loud_and_tries_one_path(self, raster, tmp_path, monkeypatch):
        """A missing file raises; no second location is tried, DES is not contacted."""
        import imint.fetch as fetch_module

        path, expected = raster
        monkeypatch.setenv("NMD2023_PATH", str(tmp_path / "absent.tif"))
        monkeypatch.setattr(fetch_module, "NMD2023_DEFAULT_PATH", path)  # exists, must not be used
        with pytest.raises(NMD2023RasterMissingError, match="absent.tif"):
            fetch_nmd_data(coords=_FIRE_WGS84, product=2023)
        # Unset: the documented default (the ICE path) is used, and only it.
        monkeypatch.delenv("NMD2023_PATH")
        np.testing.assert_array_equal(
            fetch_nmd_data(coords=_FIRE_WGS84, product=2023).nmd_raster, expected)
        monkeypatch.setattr(fetch_module, "NMD2023_DEFAULT_PATH", str(tmp_path / "ice.tif"))
        with pytest.raises(NMD2023RasterMissingError, match="ice.tif"):
            fetch_nmd_data(coords=_FIRE_WGS84, product=2023)
        assert NMD2023_DEFAULT_PATH == "/data/nmd2023/NMD2023bas_v2_1.tif"

    def test_beyond_the_raster_reads_as_outside(self, tmp_path, bbox):
        """Only the bbox's western half lies on the raster; the rest is 0, not an error."""
        h = (bbox["north"] - bbox["south"]) // 10
        w = (bbox["east"] - bbox["west"]) // 10
        half = w // 2
        path = _write_nmd2023(tmp_path / "west.tif", bbox["west"], bbox["north"],
                              np.full((h, half), 4232, dtype=np.uint16))
        raster, transform, _ = read_nmd2023_window(bbox, path)
        assert raster.shape == (h, w) and raster.dtype == np.uint16
        assert (raster[:, :half] == 4232).all() and (raster[:, half:] == 0).all()
        assert transform.c == bbox["west"]

    def test_off_lattice_bbox_raises_instead_of_resampling(self, raster, bbox):
        path, _ = raster
        shifted = {**bbox, "west": bbox["west"] + 5, "east": bbox["east"] + 5}
        with pytest.raises(ValueError, match="not on the"):
            read_nmd2023_window(shifted, path)

    def test_a_2018_raster_is_refused(self, tmp_path, bbox):
        """Pointing NMD2023_PATH at the uint8 2018 file must not read it as 2023."""
        path = _write_nmd2023(tmp_path / "nmd2018.tif", bbox["west"], bbox["north"],
                              np.full((4, 4), 111), dtype="uint8")
        with pytest.raises(ValueError, match="not an NMD 2023 raster"):
            read_nmd2023_window(bbox, path)

    def test_target_shape_resamples_nearest_and_scales_the_transform(self, raster, bbox):
        path, expected = raster
        h, w = expected.shape
        result = fetch_nmd_data(coords=_FIRE_WGS84, product=2023, raster_path=path,
                                target_shape=(h // 2, w // 2))
        assert result.nmd_raster.shape == (h // 2, w // 2)
        assert result.nmd_raster.dtype == np.uint16
        assert set(np.unique(result.nmd_raster)) <= {62, 4211, 4233}
        assert result.transform.a == pytest.approx(10 * w / (w // 2))
        assert result.transform.c == bbox["west"]


class TestTrainingCallersStayOnNMD2018:
    """Training/label and showcase callers name NMD 2018; the r1 rung compares it with 2023."""

    CALLERS = (
        "imint/training/prepare_data.py",
        "imint/training/tile_fetch.py",
        "executors/seasonal_fetch.py",
        "executors/s2_seasonal_fetch.py",
        "scripts/submit_s2_jobs.py",
        "scripts/enrich_tiles_nmd.py",
        "scripts/generate_grazing_showcase.py",
    )

    def test_every_call_passes_product_2018(self):
        import ast
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        seen = {}
        for rel in self.CALLERS:
            calls = [
                node for node in ast.walk(ast.parse((root / rel).read_text()))
                if isinstance(node, ast.Call)
                and getattr(node.func, "id", getattr(node.func, "attr", None)) == "fetch_nmd_data"
            ]
            seen[rel] = [
                next((ast.literal_eval(k.value) for k in c.keywords if k.arg == "product"), None)
                for c in calls
            ]
        assert all(seen.values()), seen  # every listed file still fetches NMD
        assert {p for products in seen.values() for p in products} == {2018}, seen

    def test_remote_label_fallback_requests_2018(self, monkeypatch):
        import imint.fetch as fetch_module
        from imint.training.tile_config import TileConfig
        from imint.training.tile_fetch import _fetch_nmd_label_remote

        calls = []

        def _fake(coords, **kwargs):
            calls.append(kwargs)
            return NMDFetchResult(nmd_raster=np.full((4, 4), 111, dtype=np.uint8))

        monkeypatch.setattr(fetch_module, "fetch_nmd_data", _fake)
        tile = TileConfig(size_px=4)
        bbox = {"west": 500000, "south": 6500000, "east": 500040, "north": 6500040}
        out = _fetch_nmd_label_remote(bbox, tile)
        assert calls == [{"target_shape": (4, 4), "product": 2018}]
        np.testing.assert_array_equal(out, 111)


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
        save_nmd_overlay(l2_raster, path, palette="nmd2018_l2")

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

    def test_overlay_raw_palette_is_chosen_by_the_caller(self, tmp_path):
        """Raw codes render through the raw palette whatever their values."""
        from imint.exporters.export import save_nmd_overlay
        from PIL import Image

        # 3 and 2 would pass for level-2 indices; the palette decides, not the values.
        raw = np.array([[3, 2], [111, 0]], dtype=np.uint8)
        path = str(tmp_path / "raw.png")
        save_nmd_overlay(raw, path, palette="nmd2018_raw")
        arr = np.array(Image.open(path))
        np.testing.assert_array_equal(arr[0, 0], [255, 215, 0])   # 3 Åkermark
        np.testing.assert_array_equal(arr[0, 1], [139, 90, 43])   # 2 Öppen våtmark
        np.testing.assert_array_equal(arr[1, 0], [0, 100, 0])     # 111 Tallskog
        np.testing.assert_array_equal(arr[1, 1], [128, 128, 128])  # no colour → grey

    def test_overlay_refuses_unknown_palette_and_foreign_index(self, tmp_path):
        from imint.exporters.export import save_nmd_overlay

        path = str(tmp_path / "x.png")
        with pytest.raises(TypeError):
            save_nmd_overlay(np.zeros((2, 2), dtype=np.uint8), path)
        with pytest.raises(ValueError, match="unknown NMD overlay palette"):
            save_nmd_overlay(np.zeros((2, 2), dtype=np.uint8), path, palette="auto")
        # Raw 2018 codes handed to the level-2 palette used to be clipped or
        # auto-detected; now they are refused.
        with pytest.raises(ValueError, match="raster holds index 111"):
            save_nmd_overlay(np.full((2, 2), 111, dtype=np.uint8), path, palette="nmd2018_l2")
        assert not os.path.exists(path)

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
