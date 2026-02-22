"""Tests for imint/fetch.py — cloud detection and DES data fetching."""
from __future__ import annotations

import os
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

from imint.fetch import (
    check_cloud_fraction,
    FetchResult,
    FetchError,
    SCL_CLOUD_CLASSES,
    BANDS_10M,
    BANDS_20M_SPECTRAL,
    BANDS_20M_CATEGORICAL,
    NMD_GRID_SIZE,
    TARGET_CRS,
    _connect,
    _to_nmd_grid,
    fetch_des_data,
)


# ── check_cloud_fraction ─────────────────────────────────────────────────────

class TestCheckCloudFraction:
    """Verify SCL-based cloud fraction computation."""

    def test_all_clear(self):
        """No cloud pixels → fraction = 0.0."""
        scl = np.full((100, 100), 4, dtype=np.uint8)  # 4 = vegetation
        assert check_cloud_fraction(scl) == 0.0

    def test_all_cloudy(self):
        """All pixels cloud → fraction = 1.0."""
        scl = np.full((100, 100), 9, dtype=np.uint8)  # 9 = cloud_high
        assert check_cloud_fraction(scl) == 1.0

    def test_half_cloudy(self):
        """50% cloud pixels."""
        scl = np.zeros((10, 10), dtype=np.uint8)
        scl[:5, :] = 9  # top half = cloud
        assert abs(check_cloud_fraction(scl) - 0.5) < 1e-6

    def test_cirrus_counts_as_cloud(self):
        """SCL class 10 (thin_cirrus) should count as cloud."""
        scl = np.full((100, 100), 10, dtype=np.uint8)
        assert check_cloud_fraction(scl) == 1.0

    def test_medium_cloud(self):
        """SCL class 8 (cloud_medium_probability) should count as cloud."""
        scl = np.full((100, 100), 8, dtype=np.uint8)
        assert check_cloud_fraction(scl) == 1.0

    def test_non_cloud_classes(self):
        """SCL classes 0-7 and 11 should NOT count as cloud."""
        for cls in [0, 1, 2, 3, 4, 5, 6, 7, 11]:
            scl = np.full((10, 10), cls, dtype=np.uint8)
            assert check_cloud_fraction(scl) == 0.0, f"SCL class {cls} incorrectly counted as cloud"

    def test_mixed_classes(self):
        """Mixed SCL classes — only 8, 9, 10 should count."""
        scl = np.array([
            [4, 4, 8, 9],    # 2 cloud of 4
            [10, 5, 6, 7],   # 1 cloud of 4
            [11, 0, 1, 2],   # 0 cloud
        ], dtype=np.uint8)
        # 3 cloud pixels out of 12
        assert abs(check_cloud_fraction(scl) - 3.0/12.0) < 1e-6

    def test_cloud_classes_constant(self):
        """SCL_CLOUD_CLASSES should be {8, 9, 10}."""
        assert SCL_CLOUD_CLASSES == frozenset({8, 9, 10})


# ── FetchResult ──────────────────────────────────────────────────────────────

class TestFetchResult:
    """Verify FetchResult dataclass."""

    def test_basic_creation(self):
        """Should hold bands, scl, cloud_fraction, rgb."""
        h, w = 64, 64
        bands = {"B02": np.zeros((h, w)), "B03": np.zeros((h, w)), "B04": np.zeros((h, w))}
        rgb = np.zeros((h, w, 3))
        scl = np.zeros((h, w), dtype=np.uint8)

        result = FetchResult(
            bands=bands, scl=scl, cloud_fraction=0.1, rgb=rgb,
        )
        assert len(result.bands) == 3
        assert result.cloud_fraction == 0.1
        assert result.rgb.shape == (h, w, 3)
        assert result.crs is None
        assert result.transform is None

    def test_optional_crs_transform(self):
        """crs and transform should be optional."""
        result = FetchResult(
            bands={}, scl=None, cloud_fraction=0.0,
            rgb=np.zeros((1, 1, 3)), crs="EPSG:3006", transform="mock",
        )
        assert result.crs == "EPSG:3006"
        assert result.transform == "mock"

    def test_geo_field(self):
        """FetchResult should accept a geo field."""
        from imint.job import GeoContext
        geo = GeoContext(
            crs="EPSG:3006",
            transform="mock-transform",
            bounds_projected={"west": 370000, "south": 6150000, "east": 380000, "north": 6160000},
            bounds_wgs84={"west": 13.0, "south": 55.5, "east": 13.1, "north": 55.6},
            shape=(100, 100),
        )
        result = FetchResult(
            bands={}, scl=None, cloud_fraction=0.0,
            rgb=np.zeros((1, 1, 3)), geo=geo,
        )
        assert result.geo is not None
        assert result.geo.crs == "EPSG:3006"


# ── _connect authentication priority ─────────────────────────────────────────

class TestConnect:
    """Verify authentication priority in _connect."""

    @patch("imint.fetch.openeo", create=True)
    def test_explicit_token_has_priority(self, mock_openeo_module):
        """Explicit token argument should be used first."""
        import imint.fetch as fetch_mod

        mock_conn = MagicMock()
        # Patch openeo inside the module
        with patch.dict("sys.modules", {"openeo": mock_openeo_module}):
            mock_openeo_module.connect.return_value = mock_conn

            result = _connect(token="my-explicit-token")

            mock_conn.authenticate_oidc_access_token.assert_called_once_with(
                access_token="my-explicit-token", provider_id="egi"
            )
            assert result is mock_conn

    @patch.dict("os.environ", {"DES_TOKEN": "env-token"}, clear=False)
    @patch("imint.fetch.openeo", create=True)
    def test_env_token_fallback(self, mock_openeo_module):
        """DES_TOKEN env var should be used if no explicit token."""
        mock_conn = MagicMock()
        with patch.dict("sys.modules", {"openeo": mock_openeo_module}):
            mock_openeo_module.connect.return_value = mock_conn

            result = _connect(token=None)

            mock_conn.authenticate_oidc_access_token.assert_called_once_with(
                access_token="env-token", provider_id="egi"
            )

    @patch.dict("os.environ", {}, clear=False)
    @patch("imint.fetch.openeo", create=True)
    def test_file_token_fallback(self, mock_openeo_module, tmp_path):
        """Token file should be used if no explicit token, env var, or refresh token."""
        mock_conn = MagicMock()
        # Make refresh token fail so it falls through to file token
        mock_conn.authenticate_oidc_refresh_token.side_effect = Exception("no refresh token")
        with patch.dict("sys.modules", {"openeo": mock_openeo_module}):
            mock_openeo_module.connect.return_value = mock_conn

            # Remove DES_TOKEN from env if present
            with patch.dict("os.environ", {k: v for k, v in os.environ.items() if k != "DES_TOKEN"}):
                token_file = tmp_path / ".des_token"
                token_file.write_text("file-token\n")

                result = _connect(token=None, token_path=str(token_file))

                mock_conn.authenticate_oidc_access_token.assert_called_once_with(
                    access_token="file-token", provider_id="egi"
                )


# ── fetch_des_data (mocked integration) ─────────────────────────────────────

class TestFetchDesData:
    """Verify fetch_des_data with mocked openEO backend."""

    def _make_mock_geotiff(self, n_bands: int = 6, h: int = 64, w: int = 64):
        """Create a fake GeoTIFF bytes object for mocking."""
        import io
        import struct

        # We'll mock rasterio.open instead of creating real GeoTIFF bytes
        return b"fake-geotiff-bytes"

    def _mock_two_stage_fetch(self, mock_connect, mock_rasterio_open,
                               h=64, w=64, scl_value=4):
        """Set up mocks for the two-stage SCL-first fetch strategy.

        Stage 1: SCL fetch (load_collection x2: b02 ref + scl)
        Stage 2: Spectral fetch (load_collection x2: 10m + 20m)
        """
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        # All cubes share the same mock interface
        mock_cube = MagicMock()
        mock_conn.load_collection.return_value = mock_cube
        mock_cube.resample_cube_spatial.return_value = mock_cube
        mock_cube.merge_cubes.return_value = mock_cube
        mock_cube.reduce_dimension.return_value = mock_cube
        mock_cube.download.return_value = b"geotiff-data"

        # SCL GeoTIFF (1 band) and spectral GeoTIFF (7 bands)
        scl_raw = np.full((1, h, w), scl_value, dtype=np.uint16)
        spectral_raw = np.zeros((7, h, w), dtype=np.uint16)
        spectral_raw[0] = 1500  # b02
        spectral_raw[1] = 1600  # b03
        spectral_raw[2] = 1960  # b04
        spectral_raw[3] = 3000  # b08
        spectral_raw[4] = 2800  # b8a
        spectral_raw[5] = 2000  # b11
        spectral_raw[6] = 1800  # b12

        call_count = {"n": 0}
        def make_mock_src(data):
            """Create a rasterio mock src for given raw data."""
            src = MagicMock()
            src.read.return_value = data
            src.crs = "EPSG:3006"
            src.transform = "mock-transform"
            src.__enter__ = MagicMock(return_value=src)
            src.__exit__ = MagicMock(return_value=False)
            return src

        # rasterio.open is called twice: first for SCL, then for spectral
        mock_rasterio_open.side_effect = [
            make_mock_src(scl_raw),
            make_mock_src(spectral_raw),
        ]

        return mock_conn

    @patch("rasterio.open")
    @patch("imint.fetch._connect")
    def test_returns_fetch_result(self, mock_connect, mock_rasterio_open):
        """fetch_des_data should return a FetchResult with correct fields."""
        h, w = 64, 64
        self._mock_two_stage_fetch(mock_connect, mock_rasterio_open, h, w,
                                    scl_value=4)  # vegetation, no clouds

        result = fetch_des_data(
            date="2022-06-15",
            coords={"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0},
        )

        assert isinstance(result, FetchResult)
        assert "B02" in result.bands
        assert "B04" in result.bands
        assert "B8A" in result.bands
        assert "B11" in result.bands
        assert "B12" in result.bands
        assert result.scl is not None
        assert result.cloud_fraction == 0.0  # all vegetation
        assert result.rgb.shape == (h, w, 3)
        assert result.crs == "EPSG:3006"

        # Verify reflectance conversion: B04 DN=1960 → (1960-1000)/10000 = 0.096
        assert abs(result.bands["B04"].mean() - 0.096) < 1e-4

    @patch("rasterio.open")
    @patch("imint.fetch._connect")
    def test_rejects_cloudy_scene(self, mock_connect, mock_rasterio_open):
        """Should raise FetchError when cloud fraction exceeds threshold."""
        h, w = 64, 64
        self._mock_two_stage_fetch(mock_connect, mock_rasterio_open, h, w,
                                    scl_value=9)  # cloud_high_probability

        with pytest.raises(FetchError, match="too cloudy"):
            fetch_des_data(
                date="2022-06-15",
                coords={"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0},
                cloud_threshold=0.1,
            )

    @patch("rasterio.open")
    @patch("imint.fetch._connect")
    def test_skips_scl_check_when_disabled(self, mock_connect, mock_rasterio_open):
        """With include_scl=False, should skip cloud check and fetch everything."""
        h, w = 64, 64
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        mock_cube = MagicMock()
        mock_conn.load_collection.return_value = mock_cube
        mock_cube.resample_cube_spatial.return_value = mock_cube
        mock_cube.merge_cubes.return_value = mock_cube
        mock_cube.download.return_value = b"geotiff-data"

        spectral_raw = np.zeros((7, h, w), dtype=np.uint16)
        spectral_raw[0] = 1500
        mock_src = MagicMock()
        mock_src.read.return_value = spectral_raw
        mock_src.crs = "EPSG:3006"
        mock_src.transform = "mock-transform"
        mock_src.__enter__ = MagicMock(return_value=mock_src)
        mock_src.__exit__ = MagicMock(return_value=False)
        mock_rasterio_open.return_value = mock_src

        result = fetch_des_data(
            date="2022-06-15",
            coords={"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0},
            include_scl=False,
        )

        assert isinstance(result, FetchResult)
        assert result.scl is None
        assert result.cloud_fraction == 0.0

    @patch("imint.fetch._connect")
    def test_fetch_error_on_empty_data(self, mock_connect):
        """Should raise FetchError when DES returns empty spectral data."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        mock_cube = MagicMock()
        mock_conn.load_collection.return_value = mock_cube
        mock_cube.resample_cube_spatial.return_value = mock_cube
        mock_cube.merge_cubes.return_value = mock_cube
        mock_cube.download.return_value = b""  # empty

        with pytest.raises(FetchError, match="empty"):
            fetch_des_data(
                date="2022-06-15",
                coords={"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0},
                include_scl=False,
            )

    @patch("imint.fetch._connect")
    def test_fetch_error_on_connection_failure(self, mock_connect):
        """Should raise FetchError when connection fails."""
        mock_connect.side_effect = FetchError("connection failed")

        with pytest.raises(FetchError, match="connection failed"):
            fetch_des_data(
                date="2022-06-15",
                coords={"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0},
            )


# ── Band constants ───────────────────────────────────────────────────────────

class TestBandConstants:
    """Verify band grouping constants."""

    def test_10m_bands(self):
        assert BANDS_10M == ["b02", "b03", "b04", "b08"]

    def test_20m_spectral_bands(self):
        assert BANDS_20M_SPECTRAL == ["b8a", "b11", "b12"]

    def test_20m_categorical_bands(self):
        assert BANDS_20M_CATEGORICAL == ["scl"]

    def test_all_lowercase(self):
        """DES uses lowercase band names."""
        for band in BANDS_10M + BANDS_20M_SPECTRAL + BANDS_20M_CATEGORICAL:
            assert band == band.lower()


# ── Grid snapping constants ──────────────────────────────────────────────────

class TestGridConstants:
    """Verify NMD grid constants."""

    def test_grid_size(self):
        assert NMD_GRID_SIZE == 10

    def test_target_crs(self):
        assert TARGET_CRS == "EPSG:3006"


# ── _to_nmd_grid ────────────────────────────────────────────────────────────

class TestToNMDGrid:
    """Verify WGS84 → EPSG:3006 + 10m grid snapping."""

    def test_output_keys(self):
        """Should return west, south, east, north, crs."""
        coords = {"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0}
        result = _to_nmd_grid(coords)
        assert set(result.keys()) == {"west", "south", "east", "north", "crs"}

    def test_crs_is_epsg3006(self):
        coords = {"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0}
        result = _to_nmd_grid(coords)
        assert result["crs"] == "EPSG:3006"

    def test_all_bounds_divisible_by_grid(self):
        """All bounds must be exact multiples of NMD_GRID_SIZE (10m)."""
        coords = {"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0}
        result = _to_nmd_grid(coords)
        for key in ["west", "south", "east", "north"]:
            assert result[key] % NMD_GRID_SIZE == 0, f"{key}={result[key]} not divisible by {NMD_GRID_SIZE}"

    def test_bbox_only_expands(self):
        """Snapping must never shrink the bbox."""
        import math
        from rasterio.crs import CRS
        from rasterio.warp import transform_bounds

        coords = {"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0}
        w, s, e, n = transform_bounds(
            CRS.from_epsg(4326), CRS.from_epsg(3006),
            coords["west"], coords["south"], coords["east"], coords["north"],
        )
        result = _to_nmd_grid(coords)
        assert result["west"] <= w
        assert result["south"] <= s
        assert result["east"] >= e
        assert result["north"] >= n

    def test_deterministic(self):
        """Same input must always produce the same output."""
        coords = {"west": 13.5, "south": 55.5, "east": 14.0, "north": 56.0}
        assert _to_nmd_grid(coords) == _to_nmd_grid(coords)

    def test_projected_coords_in_reasonable_range(self):
        """SWEREF99 TM coords for Sweden should be in expected ranges."""
        coords = {"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0}
        result = _to_nmd_grid(coords)
        # EPSG:3006 easting ~300_000 to ~900_000 for Sweden
        assert 300_000 < result["west"] < 900_000
        assert 300_000 < result["east"] < 900_000
        # EPSG:3006 northing ~6_100_000 to ~7_700_000 for Sweden
        assert 6_100_000 < result["south"] < 7_700_000
        assert 6_100_000 < result["north"] < 7_700_000

    def test_fetch_des_data_sends_projected_coords(self):
        """fetch_des_data should pass projected coords to load_collection."""
        from unittest.mock import MagicMock, patch

        mock_conn = MagicMock()
        mock_cube = MagicMock()
        mock_conn.load_collection.return_value = mock_cube
        mock_cube.resample_cube_spatial.return_value = mock_cube
        mock_cube.merge_cubes.return_value = mock_cube
        mock_cube.download.return_value = b"geotiff-data"

        # Mock rasterio — called twice (SCL stage + spectral stage)
        def make_src(raw):
            src = MagicMock()
            src.read.return_value = raw
            src.crs = "EPSG:3006"
            src.transform = "mock-transform"
            src.__enter__ = MagicMock(return_value=src)
            src.__exit__ = MagicMock(return_value=False)
            return src

        scl_raw = np.full((1, 64, 64), 4, dtype=np.uint16)  # vegetation
        spectral_raw = np.zeros((7, 64, 64), dtype=np.uint16)

        coords = {"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0}

        with patch("imint.fetch._connect", return_value=mock_conn), \
             patch("rasterio.open", side_effect=[make_src(scl_raw), make_src(spectral_raw)]):
            fetch_des_data(date="2022-06-15", coords=coords)

        # Verify all load_collection calls used projected coords with crs key
        for call_args in mock_conn.load_collection.call_args_list:
            spatial_extent = call_args[1].get("spatial_extent") or call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("spatial_extent")
            if spatial_extent:
                assert "crs" in spatial_extent, "spatial_extent should include crs"
                assert spatial_extent["crs"] == "EPSG:3006"
