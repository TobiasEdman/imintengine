"""Tests for the spectral analyzer."""
from __future__ import annotations

import numpy as np
from imint.analyzers.spectral import SpectralAnalyzer, LC_WATER, LC_VEGETATION, LC_BARE_SOIL


class TestSpectralNDVI:
    """Verify NDVI computation with known band values."""

    def test_high_ndvi_vegetation(self, bands_known):
        """Top half has B08=0.8, B04=0.1 → NDVI ≈ 0.78 (vegetation)."""
        analyzer = SpectralAnalyzer(config={})
        rgb = np.full((64, 64, 3), 0.5, dtype=np.float32)
        result = analyzer.run(rgb, bands=bands_known, date="2022-06-15")

        assert result.success
        ndvi = result.outputs["indices"]["NDVI"]

        # Top half: (0.8 - 0.1) / (0.8 + 0.1) ≈ 0.778
        top_mean = float(ndvi[:32, :].mean())
        assert top_mean > 0.7, f"Expected high NDVI in top half, got {top_mean}"

    def test_low_ndvi_bare_soil(self, bands_known):
        """Bottom half has B08=0.1, B04=0.8 → NDVI ≈ -0.78 (bare soil)."""
        analyzer = SpectralAnalyzer(config={})
        rgb = np.full((64, 64, 3), 0.5, dtype=np.float32)
        result = analyzer.run(rgb, bands=bands_known, date="2022-06-15")

        ndvi = result.outputs["indices"]["NDVI"]
        bottom_mean = float(ndvi[32:, :].mean())
        assert bottom_mean < -0.5, f"Expected low NDVI in bottom half, got {bottom_mean}"


class TestSpectralLandCover:
    """Verify pixel classification."""

    def test_vegetation_classified(self, bands_known):
        """Top half (high NDVI) should be classified as vegetation."""
        analyzer = SpectralAnalyzer(config={"ndvi_threshold": 0.3})
        rgb = np.full((64, 64, 3), 0.5, dtype=np.float32)
        result = analyzer.run(rgb, bands=bands_known, date="2022-06-15")

        lc = result.outputs["land_cover"]
        top_veg = (lc[:32, :] == LC_VEGETATION).mean()
        assert top_veg > 0.9, f"Expected top half mostly vegetation, got {top_veg:.0%}"

    def test_not_vegetation_in_bottom(self, bands_known):
        """Bottom half (negative NDVI) should NOT be classified as vegetation."""
        analyzer = SpectralAnalyzer(config={"ndvi_threshold": 0.3})
        rgb = np.full((64, 64, 3), 0.5, dtype=np.float32)
        result = analyzer.run(rgb, bands=bands_known, date="2022-06-15")

        lc = result.outputs["land_cover"]
        bottom_veg = (lc[32:, :] == LC_VEGETATION).mean()
        assert bottom_veg == 0.0, f"Expected no vegetation in bottom half, got {bottom_veg:.0%}"

    def test_stats_sum_to_one(self, bands_known):
        """Land cover fractions should sum to 1.0."""
        analyzer = SpectralAnalyzer(config={})
        rgb = np.full((64, 64, 3), 0.5, dtype=np.float32)
        result = analyzer.run(rgb, bands=bands_known, date="2022-06-15")

        stats = result.outputs["stats"]
        total = sum(stats.values())
        assert abs(total - 1.0) < 1e-6, f"Stats sum to {total}, expected 1.0"


class TestSpectralFallback:
    """Verify RGB fallback when bands are missing."""

    def test_runs_without_bands(self, rgb_random):
        """Should succeed using RGB approximation."""
        analyzer = SpectralAnalyzer(config={})
        result = analyzer.run(rgb_random, bands=None, date="2022-06-15")

        assert result.success
        assert result.metadata["fallback_rgb"] is True
        assert "NDVI" in result.outputs["indices"]

    def test_runs_with_empty_bands(self, rgb_random):
        """Should fall back when bands dict has no B08."""
        analyzer = SpectralAnalyzer(config={})
        result = analyzer.run(rgb_random, bands={}, date="2022-06-15")

        assert result.success
        assert result.metadata["fallback_rgb"] is True
