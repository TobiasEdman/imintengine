"""Tests for the spectral analyzer."""
from __future__ import annotations

import numpy as np
from imint.analyzers.spectral import SpectralAnalyzer


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


class TestSpectralNBR:
    """Verify NBR (Normalized Burn Ratio) computation."""

    def test_nbr_in_output(self, bands_known):
        """NBR should be included in spectral indices."""
        analyzer = SpectralAnalyzer(config={})
        rgb = np.full((64, 64, 3), 0.5, dtype=np.float32)
        result = analyzer.run(rgb, bands=bands_known, date="2022-06-15")

        assert result.success
        assert "NBR" in result.outputs["indices"]

    def test_high_nbr_vegetation(self, bands_known):
        """Top half has B08=0.8, B12=0.12 → NBR ≈ 0.74 (healthy vegetation)."""
        analyzer = SpectralAnalyzer(config={})
        rgb = np.full((64, 64, 3), 0.5, dtype=np.float32)
        result = analyzer.run(rgb, bands=bands_known, date="2022-06-15")

        nbr = result.outputs["indices"]["NBR"]
        # Top half: (0.8 - 0.12) / (0.8 + 0.12) ≈ 0.739
        top_mean = float(nbr[:32, :].mean())
        assert top_mean > 0.7, f"Expected high NBR in top half, got {top_mean}"

    def test_low_nbr_burned(self, bands_known):
        """Bottom half has B08=0.1, B12=0.12 → NBR ≈ -0.09 (potential burn)."""
        analyzer = SpectralAnalyzer(config={})
        rgb = np.full((64, 64, 3), 0.5, dtype=np.float32)
        result = analyzer.run(rgb, bands=bands_known, date="2022-06-15")

        nbr = result.outputs["indices"]["NBR"]
        # Bottom half: (0.1 - 0.12) / (0.1 + 0.12) ≈ -0.091
        bottom_mean = float(nbr[32:, :].mean())
        assert bottom_mean < 0, f"Expected negative NBR in bottom half, got {bottom_mean}"

    def test_nbr_fallback(self, rgb_random):
        """NBR should be present even in RGB fallback mode."""
        analyzer = SpectralAnalyzer(config={})
        result = analyzer.run(rgb_random, bands=None, date="2022-06-15")

        assert result.success
        assert "NBR" in result.outputs["indices"]


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
