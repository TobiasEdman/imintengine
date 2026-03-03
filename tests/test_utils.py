"""Tests for imint/utils.py — DN conversion and band mapping."""
from __future__ import annotations

import numpy as np
from imint.utils import dn_to_reflectance, des_to_imint_bands, bands_to_rgb


class TestDNToReflectance:
    """Verify Sentinel-2 L2A DN → reflectance conversion."""

    def test_typical_vegetation_pixel(self):
        """DN=1960 → reflectance = (1960 - 1000) / 10000 = 0.096."""
        dn = np.array([1960], dtype=np.float32)
        refl = dn_to_reflectance(dn)
        assert abs(refl[0] - 0.096) < 1e-4

    def test_offset_value(self):
        """DN=1000 → reflectance = (1000 - 1000) / 10000 = 0.0."""
        dn = np.array([1000], dtype=np.float32)
        refl = dn_to_reflectance(dn)
        assert abs(refl[0] - 0.0) < 1e-4

    def test_nodata_zero(self):
        """DN=0 (nodata) → reflectance = (0 - 1000) / 10000 = -0.1, clipped to 0.0."""
        dn = np.array([0], dtype=np.float32)
        refl = dn_to_reflectance(dn, clip=True)
        assert refl[0] == 0.0

    def test_high_reflectance(self):
        """DN=11000 → reflectance = (11000 - 1000) / 10000 = 1.0."""
        dn = np.array([11000], dtype=np.float32)
        refl = dn_to_reflectance(dn)
        assert abs(refl[0] - 1.0) < 1e-4

    def test_clipping_above_one(self):
        """DN=12000 → raw = 1.1, clipped to 1.0."""
        dn = np.array([12000], dtype=np.float32)
        refl = dn_to_reflectance(dn, clip=True)
        assert refl[0] == 1.0

    def test_no_clip(self):
        """Without clipping, values above 1.0 are preserved."""
        dn = np.array([12000], dtype=np.float32)
        refl = dn_to_reflectance(dn, clip=False)
        assert refl[0] > 1.0

    def test_array_shape_preserved(self):
        """Output shape should match input shape."""
        dn = np.random.randint(0, 9000, size=(100, 100)).astype(np.float32)
        refl = dn_to_reflectance(dn)
        assert refl.shape == dn.shape
        assert refl.dtype == np.float32


class TestDNSourceProfiles:
    """Verify different data source offset/scale profiles."""

    def test_des_source(self):
        """DES: (DN - 1000) / 10000."""
        dn = np.array([1960], dtype=np.float32)
        refl = dn_to_reflectance(dn, source="des")
        assert abs(refl[0] - 0.096) < 1e-4

    def test_copernicus_source(self):
        """Copernicus/CDSE via openEO: DN / 10000 (offset already applied by backend)."""
        dn = np.array([960], dtype=np.float32)
        refl = dn_to_reflectance(dn, source="copernicus")
        assert abs(refl[0] - 0.096) < 1e-4

    def test_copernicus_raw_source(self):
        """Copernicus/CDSE raw files: (DN + 1000) / 10000."""
        dn = np.array([960], dtype=np.float32)
        refl = dn_to_reflectance(dn, source="copernicus_raw")
        assert abs(refl[0] - 0.196) < 1e-4

    def test_legacy_source(self):
        """Legacy pre-PB04.00: DN / 10000."""
        dn = np.array([960], dtype=np.float32)
        refl = dn_to_reflectance(dn, source="legacy")
        assert abs(refl[0] - 0.096) < 1e-4

    def test_default_is_des(self):
        """Default source should be DES."""
        dn = np.array([2000], dtype=np.float32)
        refl_default = dn_to_reflectance(dn)
        refl_des = dn_to_reflectance(dn, source="des")
        np.testing.assert_array_equal(refl_default, refl_des)


class TestCopernicusReflectanceOffset:
    """Validate CDSE DN→reflectance offset for openEO and raw data.

    CDSE via openEO: the backend applies RADIO_ADD_OFFSET internally,
    so output DN is already corrected. reflectance = DN / 10000.

    CDSE raw files (S3/OData): RADIO_ADD_OFFSET=-1000 is NOT applied.
    reflectance = (DN + 1000) / 10000.
    """

    def test_copernicus_openeo_offset_range(self):
        """Verify CDSE openEO DN (offset=0) produces correct reflectance.

        openEO applies RADIO_ADD_OFFSET, so DN=960 → refl = 960/10000 = 0.096.
        """
        dn = np.array([0, 500, 960, 5000], dtype=np.float32)
        refl = dn_to_reflectance(dn, source="copernicus")
        expected = np.array([0.00, 0.05, 0.096, 0.50])
        np.testing.assert_allclose(refl, expected, atol=1e-6)

    def test_des_vs_copernicus_same_reflectance(self):
        """DES DN=1960 and CDSE openEO DN=960 should both give reflectance≈0.096.

        DES:   (1960 - 1000) / 10000 = 0.096
        CDSE:  960 / 10000 = 0.096  (openEO already applied offset)
        """
        des_refl = dn_to_reflectance(np.array([1960.0]), source="des")
        cdse_refl = dn_to_reflectance(np.array([960.0]), source="copernicus")
        np.testing.assert_allclose(des_refl, cdse_refl, atol=1e-6)

    def test_copernicus_nodata_clips_to_zero(self):
        """CDSE openEO DN=0 → refl = 0/10000 = 0.0."""
        dn = np.array([0], dtype=np.float32)
        refl = dn_to_reflectance(dn, source="copernicus", clip=True)
        assert refl[0] == 0.0

    def test_copernicus_high_reflectance(self):
        """CDSE openEO DN=10000 → refl = 10000/10000 = 1.0."""
        dn = np.array([10000], dtype=np.float32)
        refl = dn_to_reflectance(dn, source="copernicus")
        assert abs(refl[0] - 1.0) < 1e-6

    def test_copernicus_raw_offset(self):
        """CDSE raw files: DN=-500 → refl = (-500+1000)/10000 = 0.05."""
        dn = np.array([-500], dtype=np.float32)
        refl = dn_to_reflectance(dn, source="copernicus_raw", clip=False)
        np.testing.assert_allclose(refl, [0.05], atol=1e-6)

    def test_copernicus_raw_vs_openeo_equivalence(self):
        """Same reflectance from raw and openEO should agree.

        Raw DN=-40 → (−40+1000)/10000 = 0.096
        openEO DN=960 → 960/10000 = 0.096
        """
        raw_refl = dn_to_reflectance(np.array([-40.0]), source="copernicus_raw")
        openeo_refl = dn_to_reflectance(np.array([960.0]), source="copernicus")
        np.testing.assert_allclose(raw_refl, openeo_refl, atol=1e-6)


class TestDesToImintBands:
    """Verify DES lowercase → IMINT uppercase band mapping."""

    def test_basic_mapping(self):
        """Lowercase DES names should become uppercase IMINT names."""
        des = {
            "b02": np.array([1]),
            "b04": np.array([2]),
            "b08": np.array([3]),
            "b11": np.array([4]),
        }
        imint = des_to_imint_bands(des)
        assert set(imint.keys()) == {"B02", "B04", "B08", "B11"}

    def test_b8a_mapping(self):
        """b8a should map to B8A."""
        des = {"b8a": np.array([1])}
        imint = des_to_imint_bands(des)
        assert "B8A" in imint

    def test_values_preserved(self):
        """Array values should not be modified during mapping."""
        arr = np.array([1217, 2000, 3000])
        des = {"b04": arr}
        imint = des_to_imint_bands(des)
        np.testing.assert_array_equal(imint["B04"], arr)


class TestBandsToRGB:
    """Verify band dict → RGB conversion."""

    def test_basic_rgb(self):
        """Should produce (H, W, 3) array from bands."""
        h, w = 64, 64
        bands = {
            "B02": np.full((h, w), 0.1, dtype=np.float32),
            "B03": np.full((h, w), 0.2, dtype=np.float32),
            "B04": np.full((h, w), 0.3, dtype=np.float32),
        }
        rgb = bands_to_rgb(bands, percentile_stretch=False)
        assert rgb.shape == (h, w, 3)
        # R=B04, G=B03, B=B02
        np.testing.assert_allclose(rgb[:, :, 0], 0.3, atol=1e-6)
        np.testing.assert_allclose(rgb[:, :, 1], 0.2, atol=1e-6)
        np.testing.assert_allclose(rgb[:, :, 2], 0.1, atol=1e-6)

    def test_with_percentile_stretch(self):
        """With stretch, output should be in [0, 1]."""
        h, w = 64, 64
        bands = {
            "B02": np.random.rand(h, w).astype(np.float32) * 0.3,
            "B03": np.random.rand(h, w).astype(np.float32) * 0.3,
            "B04": np.random.rand(h, w).astype(np.float32) * 0.3,
        }
        rgb = bands_to_rgb(bands, percentile_stretch=True)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0
