"""Tests for imint/fetch.py — cloud detection and DES data fetching."""
from __future__ import annotations

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
    _connect,
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
        """Token file should be used if no explicit token or env var."""
        mock_conn = MagicMock()
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

    @patch("rasterio.open")
    @patch("imint.fetch._connect")
    def test_returns_fetch_result(self, mock_connect, mock_rasterio_open):
        """fetch_des_data should return a FetchResult with correct fields."""
        h, w = 64, 64
        n_bands = 6  # 4 x 10m + 1 x 20m + 1 x SCL

        # Mock openEO connection and cube operations
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        mock_cube_10m = MagicMock()
        mock_cube_20m = MagicMock()
        mock_cube_scl = MagicMock()
        mock_merged = MagicMock()

        mock_conn.load_collection.side_effect = [
            mock_cube_10m, mock_cube_20m, mock_cube_scl
        ]
        mock_cube_20m.resample_cube_spatial.return_value = mock_cube_20m
        mock_cube_scl.resample_cube_spatial.return_value = mock_cube_scl
        mock_cube_10m.merge_cubes.return_value = mock_merged
        mock_merged.merge_cubes.return_value = mock_merged
        mock_merged.download.return_value = b"geotiff-data"

        # Mock rasterio to return fake band data
        # Band order: b02, b03, b04, b08, b11, scl
        raw = np.zeros((n_bands, h, w), dtype=np.uint16)
        raw[0] = 1500  # b02
        raw[1] = 1600  # b03
        raw[2] = 1960  # b04
        raw[3] = 3000  # b08
        raw[4] = 2000  # b11
        raw[5] = 4     # scl = vegetation (no clouds)

        mock_src = MagicMock()
        mock_src.read.return_value = raw
        mock_src.crs = "EPSG:3006"
        mock_src.transform = "mock-transform"
        mock_src.__enter__ = MagicMock(return_value=mock_src)
        mock_src.__exit__ = MagicMock(return_value=False)
        mock_rasterio_open.return_value = mock_src

        result = fetch_des_data(
            date="2022-06-15",
            coords={"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0},
        )

        assert isinstance(result, FetchResult)
        assert "B02" in result.bands
        assert "B04" in result.bands
        assert "B11" in result.bands
        assert result.scl is not None
        assert result.cloud_fraction == 0.0  # all vegetation
        assert result.rgb.shape == (h, w, 3)
        assert result.crs == "EPSG:3006"

        # Verify reflectance conversion: B04 DN=1960 → (1960-1000)/10000 = 0.096
        assert abs(result.bands["B04"].mean() - 0.096) < 1e-4

    @patch("imint.fetch._connect")
    def test_fetch_error_on_empty_data(self, mock_connect):
        """Should raise FetchError when DES returns empty data."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        mock_cube = MagicMock()
        mock_conn.load_collection.return_value = mock_cube
        mock_cube.resample_cube_spatial.return_value = mock_cube
        mock_cube.merge_cubes.return_value = mock_cube
        mock_cube.download.return_value = b""  # empty

        with pytest.raises(FetchError, match="empty data"):
            fetch_des_data(
                date="2022-06-15",
                coords={"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0},
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
        assert BANDS_20M_SPECTRAL == ["b11"]

    def test_20m_categorical_bands(self):
        assert BANDS_20M_CATEGORICAL == ["scl"]

    def test_all_lowercase(self):
        """DES uses lowercase band names."""
        for band in BANDS_10M + BANDS_20M_SPECTRAL + BANDS_20M_CATEGORICAL:
            assert band == band.lower()


import os
