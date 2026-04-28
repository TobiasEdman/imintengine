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
    CDSE_BANDS_10M,
    CDSE_BANDS_20M_SPECTRAL,
    CDSE_BANDS_20M_CATEGORICAL,
    CDSE_COLLECTION,
    CDSE_OPENEO_URL,
    NMD_GRID_SIZE,
    TARGET_CRS,
    _connect,
    _connect_cdse,
    _to_nmd_grid,
    fetch_des_data,
    fetch_copernicus_data,
    fetch_sentinel2_data,
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

    @patch.dict("os.environ", {"DES_USER": "testuser", "DES_PASSWORD": "testpass"}, clear=False)
    @patch("imint.fetch.openeo", create=True)
    def test_basic_auth_fallback(self, mock_openeo_module):
        """DES_USER + DES_PASSWORD env vars should trigger Basic Auth."""
        mock_conn = MagicMock()
        with patch.dict("sys.modules", {"openeo": mock_openeo_module}):
            mock_openeo_module.connect.return_value = mock_conn

            # Remove DES_TOKEN so it falls through to Basic Auth
            with patch.dict("os.environ", {k: v for k, v in os.environ.items()
                                           if k != "DES_TOKEN"}, clear=True):
                # Re-set the basic auth env vars since clear=True removed them
                os.environ["DES_USER"] = "testuser"
                os.environ["DES_PASSWORD"] = "testpass"

                result = _connect(token=None)

                mock_conn.authenticate_basic.assert_called_once_with(
                    username="testuser", password="testpass"
                )
                assert result is mock_conn

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
            with patch.dict("os.environ", {k: v for k, v in os.environ.items()
                                           if k not in ("DES_TOKEN", "DES_USER", "DES_PASSWORD")}):
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

    def _mock_single_stage_fetch(self, mock_connect, mock_rasterio_open,
                                  h=64, w=64, scl_value=4, include_scl=True):
        """Set up mocks for the single-stage fetch (all bands in one request).

        Band order in the merged GeoTIFF:
          10m: b02=0, b03=1, b04=2, b08=3
          20m spectral: b05=4, b06=5, b07=6, b8a=7, b11=8, b12=9
          60m: b09=10
          SCL: 11 (if included)
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

        # Build single merged GeoTIFF: 11 spectral + 1 SCL (or 11 without SCL)
        n_bands = 12 if include_scl else 11
        raw = np.zeros((n_bands, h, w), dtype=np.uint16)
        raw[0] = 1500   # b02
        raw[1] = 1600   # b03
        raw[2] = 1960   # b04
        raw[3] = 3000   # b08
        raw[4] = 2200   # b05
        raw[5] = 2400   # b06
        raw[6] = 2600   # b07
        raw[7] = 2800   # b8a
        raw[8] = 2000   # b11
        raw[9] = 1800   # b12
        raw[10] = 1000  # b09
        if include_scl:
            raw[11] = scl_value  # SCL

        def make_mock_src(data):
            """Create a rasterio mock src for given raw data."""
            src = MagicMock()
            src.read.return_value = data
            src.crs = "EPSG:3006"
            src.transform = "mock-transform"
            src.__enter__ = MagicMock(return_value=src)
            src.__exit__ = MagicMock(return_value=False)
            return src

        # Single rasterio.open call for the merged GeoTIFF
        mock_rasterio_open.return_value = make_mock_src(raw)

        return mock_conn

    @patch("rasterio.open")
    @patch("imint.fetch._connect")
    def test_returns_fetch_result(self, mock_connect, mock_rasterio_open):
        """fetch_des_data should return a FetchResult with correct fields."""
        h, w = 64, 64
        self._mock_single_stage_fetch(mock_connect, mock_rasterio_open, h, w,
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
        self._mock_single_stage_fetch(mock_connect, mock_rasterio_open, h, w,
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

        spectral_raw = np.zeros((11, h, w), dtype=np.uint16)
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
        assert BANDS_20M_SPECTRAL == ["b05", "b06", "b07", "b8a", "b11", "b12"]

    def test_60m_bands(self):
        from imint.fetch import BANDS_60M
        assert BANDS_60M == ["b09"]

    def test_20m_categorical_bands(self):
        assert BANDS_20M_CATEGORICAL == ["scl"]

    def test_all_lowercase(self):
        """DES uses lowercase band names."""
        from imint.fetch import BANDS_60M
        for band in BANDS_10M + BANDS_20M_SPECTRAL + BANDS_60M + BANDS_20M_CATEGORICAL:
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

        # Mock rasterio — single call for merged GeoTIFF (11 spectral + 1 SCL)
        def make_src(raw):
            src = MagicMock()
            src.read.return_value = raw
            src.crs = "EPSG:3006"
            src.transform = "mock-transform"
            src.__enter__ = MagicMock(return_value=src)
            src.__exit__ = MagicMock(return_value=False)
            return src

        merged_raw = np.zeros((12, 64, 64), dtype=np.uint16)
        merged_raw[11] = 4  # SCL = vegetation

        coords = {"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0}

        with patch("imint.fetch._connect", return_value=mock_conn), \
             patch("rasterio.open", return_value=make_src(merged_raw)):
            fetch_des_data(date="2022-06-15", coords=coords)

        # Verify all load_collection calls used projected coords with crs key
        for call_args in mock_conn.load_collection.call_args_list:
            spatial_extent = call_args[1].get("spatial_extent") or call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("spatial_extent")
            if spatial_extent:
                assert "crs" in spatial_extent, "spatial_extent should include crs"
                assert spatial_extent["crs"] == "EPSG:3006"


# ── STAC-guided fetch_des_data ──────────────────────────────────────────────

class TestFetchDesDataSTAC:
    """Verify that fetch_des_data uses STAC when date_window > 0."""

    @patch("imint.fetch._stac_best_date")
    @patch("rasterio.open")
    @patch("imint.fetch._connect")
    def test_stac_called_with_date_window(self, mock_connect, mock_rasterio_open, mock_stac):
        """fetch_des_data should call _stac_best_date when date_window > 0."""
        mock_stac.return_value = "2022-06-14"  # STAC selects a nearby date

        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_cube = MagicMock()
        mock_conn.load_collection.return_value = mock_cube
        mock_cube.resample_cube_spatial.return_value = mock_cube
        mock_cube.merge_cubes.return_value = mock_cube
        mock_cube.download.return_value = b"geotiff-data"

        raw = np.zeros((12, 64, 64), dtype=np.uint16)
        raw[11] = 4  # SCL = vegetation
        src = MagicMock()
        src.read.return_value = raw
        src.crs = "EPSG:3006"
        src.transform = "mock-transform"
        src.__enter__ = MagicMock(return_value=src)
        src.__exit__ = MagicMock(return_value=False)
        mock_rasterio_open.return_value = src

        result = fetch_des_data(
            date="2022-06-15",
            coords={"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0},
            date_window=5,
        )

        assert isinstance(result, FetchResult)
        mock_stac.assert_called_once()
        # Should have been called with the original date and window
        args = mock_stac.call_args[0]
        assert args[1] == "2022-06-15"
        assert args[2] == 5

    @patch("rasterio.open")
    @patch("imint.fetch._connect")
    def test_stac_not_called_without_date_window(self, mock_connect, mock_rasterio_open):
        """fetch_des_data should NOT call STAC when date_window == 0."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_cube = MagicMock()
        mock_conn.load_collection.return_value = mock_cube
        mock_cube.resample_cube_spatial.return_value = mock_cube
        mock_cube.merge_cubes.return_value = mock_cube
        mock_cube.download.return_value = b"geotiff-data"

        raw = np.zeros((12, 64, 64), dtype=np.uint16)
        raw[11] = 4
        src = MagicMock()
        src.read.return_value = raw
        src.crs = "EPSG:3006"
        src.transform = "mock-transform"
        src.__enter__ = MagicMock(return_value=src)
        src.__exit__ = MagicMock(return_value=False)
        mock_rasterio_open.return_value = src

        with patch("imint.fetch._stac_best_date") as mock_stac:
            fetch_des_data(
                date="2022-06-15",
                coords={"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0},
                date_window=0,
            )
            mock_stac.assert_not_called()


# ── CDSE band constants ──────────────────────────────────────────────────────

class TestCDSEBandConstants:
    """Verify CDSE band grouping constants."""

    def test_cdse_10m_bands(self):
        assert CDSE_BANDS_10M == ["B02", "B03", "B04", "B08"]

    def test_cdse_20m_spectral_bands(self):
        assert CDSE_BANDS_20M_SPECTRAL == ["B05", "B06", "B07", "B8A", "B11", "B12"]

    def test_cdse_20m_categorical_bands(self):
        assert CDSE_BANDS_20M_CATEGORICAL == ["SCL"]

    def test_cdse_all_uppercase(self):
        """CDSE uses uppercase band names."""
        from imint.fetch import CDSE_BANDS_60M
        for band in CDSE_BANDS_10M + CDSE_BANDS_20M_SPECTRAL + CDSE_BANDS_60M + CDSE_BANDS_20M_CATEGORICAL:
            assert band == band.upper()

    def test_cdse_collection_name(self):
        assert CDSE_COLLECTION == "SENTINEL2_L2A"

    def test_cdse_openeo_url(self):
        assert "dataspace.copernicus.eu" in CDSE_OPENEO_URL


# ── _connect_cdse ────────────────────────────────────────────────────────────

class TestConnectCDSE:
    """Verify CDSE authentication priority."""

    @patch.dict("os.environ", {"CDSE_CLIENT_ID": "test-id", "CDSE_CLIENT_SECRET": "test-secret"}, clear=False)
    @patch("imint.fetch.openeo", create=True)
    def test_client_credentials_auth(self, mock_openeo_module):
        """Should use client_credentials when env vars are set."""
        mock_conn = MagicMock()
        with patch.dict("sys.modules", {"openeo": mock_openeo_module}):
            mock_openeo_module.connect.return_value = mock_conn

            result = _connect_cdse()

            mock_openeo_module.connect.assert_called_once_with(CDSE_OPENEO_URL)
            mock_conn.authenticate_oidc_client_credentials.assert_called_once_with(
                client_id="test-id",
                client_secret="test-secret",
            )
            assert result is mock_conn

    @patch.dict("os.environ", {}, clear=False)
    @patch("imint.fetch.openeo", create=True)
    def test_oidc_fallback(self, mock_openeo_module):
        """Should fall back to interactive OIDC when no client credentials or cred file."""
        mock_conn = MagicMock()
        with patch.dict("sys.modules", {"openeo": mock_openeo_module}):
            mock_openeo_module.connect.return_value = mock_conn

            # Remove CDSE env vars and mock away credentials file
            env = {k: v for k, v in os.environ.items()
                   if k not in ("CDSE_CLIENT_ID", "CDSE_CLIENT_SECRET")}
            with patch.dict("os.environ", env, clear=True), \
                 patch("os.path.isfile", return_value=False):
                result = _connect_cdse()

                mock_conn.authenticate_oidc.assert_called_once()
                assert result is mock_conn

    @patch.dict("os.environ", {"CDSE_CLIENT_ID": "bad-id", "CDSE_CLIENT_SECRET": "bad-secret"}, clear=False)
    @patch("imint.fetch.openeo", create=True)
    def test_client_credentials_failure(self, mock_openeo_module):
        """Should raise FetchError on auth failure."""
        mock_conn = MagicMock()
        mock_conn.authenticate_oidc_client_credentials.side_effect = Exception("auth failed")
        with patch.dict("sys.modules", {"openeo": mock_openeo_module}):
            mock_openeo_module.connect.return_value = mock_conn

            with pytest.raises(FetchError, match="client_credentials auth failed"):
                _connect_cdse()


# ── fetch_copernicus_data (mocked) ───────────────────────────────────────────

class TestFetchCopernicusData:
    """Verify fetch_copernicus_data with mocked CDSE backend."""

    def _mock_cdse_fetch(self, mock_connect, mock_rasterio_open,
                          h=64, w=64, scl_value=4, include_scl=True):
        """Set up mocks for CDSE fetch (all bands in one request).

        Band order in the merged GeoTIFF (UPPERCASE):
          10m: B02=0, B03=1, B04=2, B08=3
          20m spectral: B05=4, B06=5, B07=6, B8A=7, B11=8, B12=9
          60m: B09=10
          SCL: 11 (if included)
        """
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        mock_cube = MagicMock()
        mock_conn.load_collection.return_value = mock_cube
        mock_cube.resample_cube_spatial.return_value = mock_cube
        mock_cube.merge_cubes.return_value = mock_cube
        mock_cube.reduce_dimension.return_value = mock_cube
        mock_cube.download.return_value = b"geotiff-data"

        # Build single merged GeoTIFF: 11 spectral + 1 SCL
        n_bands = 12 if include_scl else 11
        # CDSE openEO DN values: backend applies RADIO_ADD_OFFSET,
        # so DN = reflectance * 10000 (offset=0)
        raw = np.zeros((n_bands, h, w), dtype=np.int16)
        raw[0] = 500    # B02 → refl = 500/10000 = 0.05
        raw[1] = 1000   # B03 → refl = 1000/10000 = 0.10
        raw[2] = 960    # B04 → refl = 960/10000 = 0.096
        raw[3] = 3000   # B08 → refl = 3000/10000 = 0.30
        raw[4] = 1500   # B05
        raw[5] = 1700   # B06
        raw[6] = 1900   # B07
        raw[7] = 2500   # B8A
        raw[8] = 1300   # B11
        raw[9] = 1100   # B12
        raw[10] = 800   # B09
        if include_scl:
            raw[11] = scl_value

        def make_mock_src(data):
            src = MagicMock()
            src.read.return_value = data
            src.crs = "EPSG:3006"
            src.transform = "mock-transform"
            src.__enter__ = MagicMock(return_value=src)
            src.__exit__ = MagicMock(return_value=False)
            return src

        mock_rasterio_open.return_value = make_mock_src(raw)
        return mock_conn

    @patch("imint.fetch._snap_to_target_grid")
    @patch("rasterio.open")
    @patch("imint.fetch._connect_cdse")
    def test_returns_fetch_result(self, mock_connect, mock_rasterio_open, mock_snap):
        """fetch_copernicus_data should return FetchResult with correct fields."""
        h, w = 64, 64
        self._mock_cdse_fetch(mock_connect, mock_rasterio_open, h, w, scl_value=4)
        # _snap_to_target_grid passes through raw unchanged
        mock_snap.side_effect = lambda raw, *a, **kw: (raw, "mock-transform")

        result = fetch_copernicus_data(
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

    @patch("imint.fetch._snap_to_target_grid")
    @patch("rasterio.open")
    @patch("imint.fetch._connect_cdse")
    def test_copernicus_reflectance_conversion(self, mock_connect, mock_rasterio_open, mock_snap):
        """Verify CDSE DN→reflectance uses copernicus offset.

        B04 DN=-40 → refl = (-40+1000)/10000 = 0.096
        """
        h, w = 64, 64
        self._mock_cdse_fetch(mock_connect, mock_rasterio_open, h, w, scl_value=4)
        mock_snap.side_effect = lambda raw, *a, **kw: (raw, "mock-transform")

        result = fetch_copernicus_data(
            date="2022-06-15",
            coords={"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0},
        )

        # B04 DN=960 → 960/10000 = 0.096 (openEO applies offset)
        assert abs(result.bands["B04"].mean() - 0.096) < 1e-4

    @patch("imint.fetch._snap_to_target_grid")
    @patch("rasterio.open")
    @patch("imint.fetch._connect_cdse")
    def test_uses_cdse_collection(self, mock_connect, mock_rasterio_open, mock_snap):
        """Should use SENTINEL2_L2A collection on CDSE."""
        h, w = 64, 64
        mock_conn = self._mock_cdse_fetch(mock_connect, mock_rasterio_open, h, w)
        mock_snap.side_effect = lambda raw, *a, **kw: (raw, "mock-transform")

        fetch_copernicus_data(
            date="2022-06-15",
            coords={"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0},
        )

        # Verify all load_collection calls used CDSE collection
        for call_args in mock_conn.load_collection.call_args_list:
            collection = call_args[1].get("collection_id") or call_args[0][0]
            assert collection == "SENTINEL2_L2A"

    @patch("imint.fetch._snap_to_target_grid")
    @patch("rasterio.open")
    @patch("imint.fetch._connect_cdse")
    def test_rejects_cloudy_scene(self, mock_connect, mock_rasterio_open, mock_snap):
        """Should raise FetchError when cloud fraction exceeds threshold."""
        h, w = 64, 64
        self._mock_cdse_fetch(mock_connect, mock_rasterio_open, h, w, scl_value=9)
        mock_snap.side_effect = lambda raw, *a, **kw: (raw, "mock-transform")

        with pytest.raises(FetchError, match="too cloudy"):
            fetch_copernicus_data(
                date="2022-06-15",
                coords={"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0},
                cloud_threshold=0.1,
            )

    @patch("imint.fetch._connect_cdse")
    def test_fetch_error_on_empty_data(self, mock_connect):
        """Should raise FetchError when CDSE returns empty data."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        mock_cube = MagicMock()
        mock_conn.load_collection.return_value = mock_cube
        mock_cube.resample_cube_spatial.return_value = mock_cube
        mock_cube.merge_cubes.return_value = mock_cube
        mock_cube.download.return_value = b""

        with pytest.raises(FetchError, match="empty"):
            fetch_copernicus_data(
                date="2022-06-15",
                coords={"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0},
                include_scl=False,
            )

    @patch("imint.fetch._snap_to_target_grid")
    @patch("imint.fetch._stac_best_date")
    @patch("rasterio.open")
    @patch("imint.fetch._connect_cdse")
    def test_uses_des_stac_for_date_selection(self, mock_connect, mock_rasterio_open, mock_stac, mock_snap):
        """fetch_copernicus_data should use DES STAC for date discovery."""
        mock_stac.return_value = "2022-06-14"
        mock_snap.side_effect = lambda raw, *a, **kw: (raw, "mock-transform")
        h, w = 64, 64
        self._mock_cdse_fetch(mock_connect, mock_rasterio_open, h, w, scl_value=4)

        result = fetch_copernicus_data(
            date="2022-06-15",
            coords={"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0},
            date_window=5,
        )

        assert isinstance(result, FetchResult)
        mock_stac.assert_called_once()


# ── fetch_sentinel2_data dispatcher ──────────────────────────────────────────

class TestFetchSentinel2Data:
    """Verify fetch_sentinel2_data dispatcher."""

    @patch("imint.fetch.fetch_des_data")
    def test_default_routes_to_des(self, mock_des):
        """Default source should route to fetch_des_data."""
        mock_des.return_value = MagicMock(spec=FetchResult)

        fetch_sentinel2_data(
            date="2022-06-15",
            coords={"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0},
        )

        mock_des.assert_called_once()

    @patch("imint.fetch.fetch_copernicus_data")
    def test_copernicus_routes_to_cdse(self, mock_cdse):
        """source='copernicus' should route to fetch_copernicus_data."""
        mock_cdse.return_value = MagicMock(spec=FetchResult)

        fetch_sentinel2_data(
            source="copernicus",
            date="2022-06-15",
            coords={"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0},
        )

        mock_cdse.assert_called_once()

    def test_invalid_source_raises(self):
        """Unknown source should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown source"):
            fetch_sentinel2_data(source="invalid")

    @patch("imint.fetch.fetch_des_data")
    def test_passes_kwargs_through(self, mock_des):
        """All kwargs should be forwarded to the underlying function."""
        mock_des.return_value = MagicMock(spec=FetchResult)

        fetch_sentinel2_data(
            source="des",
            date="2022-06-15",
            coords={"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0},
            cloud_threshold=0.5,
            date_window=3,
        )

        mock_des.assert_called_once_with(
            date="2022-06-15",
            coords={"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0},
            cloud_threshold=0.5,
            date_window=3,
        )


# ── _seasonal_window_to_date_range ───────────────────────────────────────────

class TestSeasonalWindowToDateRange:
    """Verify the seasonal-window → ISO date range helper used by both
    fetch_seasonal_dates (month mode) and fetch_seasonal_dates_doy (doy mode).
    """

    def test_month_31_day(self):
        """A 31-day month must end on day 31."""
        from imint.fetch import _seasonal_window_to_date_range
        assert _seasonal_window_to_date_range(2024, (1, 1), mode="month") == \
            ("2024-01-01", "2024-01-31")

    def test_month_30_day(self):
        """A 30-day month must end on day 30 (not day 31)."""
        from imint.fetch import _seasonal_window_to_date_range
        assert _seasonal_window_to_date_range(2024, (4, 4), mode="month") == \
            ("2024-04-01", "2024-04-30")

    def test_month_february_leap_year(self):
        """February in a leap year must end on day 29 (calendar.monthrange)."""
        from imint.fetch import _seasonal_window_to_date_range
        assert _seasonal_window_to_date_range(2024, (2, 2), mode="month") == \
            ("2024-02-01", "2024-02-29")

    def test_month_february_non_leap_year(self):
        """February in a non-leap year must end on day 28."""
        from imint.fetch import _seasonal_window_to_date_range
        assert _seasonal_window_to_date_range(2023, (2, 2), mode="month") == \
            ("2023-02-01", "2023-02-28")

    def test_month_window_spanning_two_months(self):
        """Multi-month window: start = first of m_start, end = last of m_end."""
        from imint.fetch import _seasonal_window_to_date_range
        assert _seasonal_window_to_date_range(2024, (4, 5), mode="month") == \
            ("2024-04-01", "2024-05-31")

    def test_month_window_february_to_april(self):
        """Feb–Apr 2024 (leap year) — end date is April 30, not Feb 29."""
        from imint.fetch import _seasonal_window_to_date_range
        assert _seasonal_window_to_date_range(2024, (2, 4), mode="month") == \
            ("2024-02-01", "2024-04-30")

    def test_doy_basic(self):
        """DOY mode converts day-of-year offsets to ISO dates."""
        from imint.fetch import _seasonal_window_to_date_range
        # 2024 is leap; doy 100 = 2024-04-09, doy 140 = 2024-05-19
        assert _seasonal_window_to_date_range(2024, (100, 140), mode="doy") == \
            ("2024-04-09", "2024-05-19")

    def test_doy_first_day(self):
        """doy=1 must map to Jan 1 (off-by-one safety)."""
        from imint.fetch import _seasonal_window_to_date_range
        assert _seasonal_window_to_date_range(2023, (1, 1), mode="doy") == \
            ("2023-01-01", "2023-01-01")

    def test_invalid_mode_raises(self):
        """Unknown mode must raise ValueError, not silently misbehave."""
        import pytest
        from imint.fetch import _seasonal_window_to_date_range
        with pytest.raises(ValueError, match="Unknown mode"):
            _seasonal_window_to_date_range(2024, (1, 1), mode="bogus")


# ── _unpack_openeo_gtiff_bytes ────────────────────────────────────────────
class TestUnpackOpeneoGTiffBytes:
    """Helper that unwraps gzip+tar-wrapped openEO GeoTIFF responses.

    Added 2026-04-28 after DES openEO began returning .tar.gz envelopes
    (one TIF per scene timestamp) for single-date single-band-group
    requests, breaking the bare ``rasterio.open(io.BytesIO(...))`` parse.
    """

    @staticmethod
    def _make_tif_bytes(value: int = 100, h: int = 8, w: int = 8) -> bytes:
        """Synthesise a tiny single-band uint8 GeoTIFF in memory."""
        import io
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        arr = np.full((1, h, w), value, dtype=np.uint8)
        with rasterio.io.MemoryFile() as mf:
            with mf.open(
                driver="GTiff", height=h, width=w, count=1, dtype="uint8",
                crs="EPSG:3006", transform=from_origin(0, 0, 10, 10),
            ) as dst:
                dst.write(arr)
            return mf.read()

    def test_bare_geotiff_passes_through(self):
        """A plain GeoTIFF must be returned unchanged."""
        from imint.fetch import _unpack_openeo_gtiff_bytes
        tif = self._make_tif_bytes(value=42)
        out = _unpack_openeo_gtiff_bytes(tif)
        assert out == tif

    def test_gzip_wrapped_geotiff_is_decompressed(self):
        """gzip-wrapped GeoTIFF must come back as the original TIF bytes."""
        import gzip
        from imint.fetch import _unpack_openeo_gtiff_bytes
        tif = self._make_tif_bytes(value=99)
        out = _unpack_openeo_gtiff_bytes(gzip.compress(tif))
        # Round-trip via rasterio (size differs across gzip implementations)
        import io
        import rasterio
        with rasterio.open(io.BytesIO(out)) as src:
            assert src.read()[0, 0, 0] == 99

    def test_single_member_targz_extracts_inner_tif(self):
        """A tar.gz with one .tif member returns the member's bytes."""
        import gzip
        import io
        import tarfile
        import rasterio
        from imint.fetch import _unpack_openeo_gtiff_bytes
        tif = self._make_tif_bytes(value=7)
        tar_buf = io.BytesIO()
        with tarfile.open(fileobj=tar_buf, mode="w") as tf:
            info = tarfile.TarInfo(name="scene_0.tif")
            info.size = len(tif)
            tf.addfile(info, io.BytesIO(tif))
        out = _unpack_openeo_gtiff_bytes(gzip.compress(tar_buf.getvalue()))
        with rasterio.open(io.BytesIO(out)) as src:
            assert src.read()[0, 0, 0] == 7

    def test_multi_member_targz_picks_earliest_by_name(self):
        """Multi-scene tarballs return the alphabetically-first member.

        DES names members ``out_2026_04_08T10_30_19.tif`` so sorted-by-name
        is sorted-by-acquisition-timestamp. Single-pick policy preserves
        single-sensor consistency for downstream analyzers — mosaicking is
        explicitly NOT the default. Callers that genuinely need multi-scene
        fusion should use ``return_bytes=True`` and parse the tar themselves.
        """
        import gzip
        import io
        import tarfile
        import rasterio
        from imint.fetch import _unpack_openeo_gtiff_bytes
        tif_early = self._make_tif_bytes(value=11, h=8, w=8)
        tif_late = self._make_tif_bytes(value=22, h=8, w=8)
        tar_buf = io.BytesIO()
        with tarfile.open(fileobj=tar_buf, mode="w") as tf:
            # Add in REVERSE name order to verify the helper sorts members
            for name, blob in (
                ("out_2026_04_08T10_40_41.tif", tif_late),
                ("out_2026_04_08T10_30_19.tif", tif_early),
            ):
                info = tarfile.TarInfo(name=name)
                info.size = len(blob)
                tf.addfile(info, io.BytesIO(blob))
        out = _unpack_openeo_gtiff_bytes(gzip.compress(tar_buf.getvalue()))
        with rasterio.open(io.BytesIO(out)) as src:
            arr = src.read()
            assert arr.shape == (1, 8, 8)
            assert arr[0, 0, 0] == 11, "must pick earliest-timestamp member"

    def test_tar_with_no_tifs_raises(self):
        """A tar containing no .tif members must raise FetchError."""
        import io
        import tarfile
        import pytest
        from imint.fetch import _unpack_openeo_gtiff_bytes, FetchError
        tar_buf = io.BytesIO()
        with tarfile.open(fileobj=tar_buf, mode="w") as tf:
            info = tarfile.TarInfo(name="readme.txt")
            payload = b"not a tif"
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
        with pytest.raises(FetchError, match="no GeoTIFF"):
            _unpack_openeo_gtiff_bytes(tar_buf.getvalue())


# ── L1C SAFE fetch from Google Cloud public bucket ─────────────────────────
class TestParseSafeMgrs:
    """Parser that derives ``(utm_zone, lat_band, square)`` from a SAFE name.

    The MGRS token drives the GCS bucket layout
    ``tiles/{utm}/{lat_band}/{square}/{SAFE}/`` so a wrong parse breaks
    every download URL.
    """

    def test_canonical_safe_name(self):
        from imint.fetch import _parse_safe_mgrs
        name = "S2A_MSIL1C_20260408T104041_N0512_R008_T33VUE_20260408T155548.SAFE"
        assert _parse_safe_mgrs(name) == ("33", "V", "UE")

    def test_southern_hemisphere_tile(self):
        from imint.fetch import _parse_safe_mgrs
        name = "S2B_MSIL1C_20240115T140749_N0509_R039_T18HUF_20240115T173255.SAFE"
        assert _parse_safe_mgrs(name) == ("18", "H", "UF")

    def test_invalid_name_raises(self):
        import pytest
        from imint.fetch import _parse_safe_mgrs
        with pytest.raises(ValueError, match="cannot parse MGRS"):
            _parse_safe_mgrs("not_a_safe.zip")


class TestStacBestL1cScene:
    """STAC-search wrapper that picks the cleanest matching L1C scene.

    The function is the only piece of the GCP fallback that talks to a
    network service. We mock requests.post so the test stays offline.
    """

    @staticmethod
    def _fake_stac(features):
        class _Resp:
            def __init__(self, payload):
                self.payload = payload
            def raise_for_status(self):
                pass
            def json(self):
                return self.payload
        return _Resp({"features": features})

    def test_picks_lowest_cloud_cover(self, monkeypatch):
        from unittest.mock import MagicMock
        import imint.fetch as f
        feats = [
            {
                "id": "noisy",
                "properties": {"eo:cloud_cover": 92.0,
                                "start_datetime": "2026-04-08T09:00:00Z",
                                "proj:code": "EPSG:32633",
                                "cubedash:region_code": "33VUE"},
                "assets": {
                    "b04": {"href": "s3://x/Y/SAFE_OLD.SAFE/GRANULE/.../B04.jp2"}
                },
            },
            {
                "id": "clean",
                "properties": {"eo:cloud_cover": 0.07,
                                "start_datetime": "2026-04-08T10:40:41Z",
                                "proj:code": "EPSG:32633",
                                "cubedash:region_code": "33VUE"},
                "assets": {
                    "b04": {"href": "s3://x/Y/S2A_MSIL1C_20260408T104041_N0512_R008_T33VUE_20260408T155548.SAFE/GRANULE/.../B04.jp2"}
                },
            },
        ]
        post = MagicMock(return_value=self._fake_stac(feats))
        import requests as _requests_mod
        monkeypatch.setattr(_requests_mod, "post", post)
        scene = f._stac_best_l1c_scene(
            {"west": 11.55, "south": 58.10, "east": 11.75, "north": 58.20},
            "2026-04-08",
        )
        assert scene["id"] == "clean"
        assert scene["cloud_cover"] == 0.07
        assert scene["safe_name"].endswith("_T33VUE_20260408T155548.SAFE")
        assert scene["tile_id"] == "33VUE"

    def test_no_matches_raises_fetcherror(self, monkeypatch):
        from unittest.mock import MagicMock
        import pytest
        import imint.fetch as f
        post = MagicMock(return_value=self._fake_stac([]))
        import requests as _requests_mod
        monkeypatch.setattr(_requests_mod, "post", post)
        with pytest.raises(f.FetchError, match="no L1C scenes"):
            f._stac_best_l1c_scene(
                {"west": 0, "south": 0, "east": 1, "north": 1},
                "2026-04-08",
            )

    def test_cloud_cap_filters(self, monkeypatch):
        from unittest.mock import MagicMock
        import pytest
        import imint.fetch as f
        feats = [
            {
                "id": "cloudy",
                "properties": {"eo:cloud_cover": 50.0,
                                "start_datetime": "2026-04-08T10:00Z"},
                "assets": {"b04": {"href": "s3://x/SAFE_OK.SAFE/GRANULE/B04.jp2"}},
            },
        ]
        post = MagicMock(return_value=self._fake_stac(feats))
        import requests as _requests_mod
        monkeypatch.setattr(_requests_mod, "post", post)
        with pytest.raises(f.FetchError):
            f._stac_best_l1c_scene(
                {"west": 0, "south": 0, "east": 1, "north": 1},
                "2026-04-08",
                cloud_max=10.0,
            )
