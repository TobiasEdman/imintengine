"""
Tests for the NMD (Nationellt Marktäckedata) analyzer.

Tests class code mapping, class statistics computation, cross-reference
logic, and graceful degradation when NMD data is unavailable.
"""
from __future__ import annotations

import numpy as np
import pytest

from imint.analyzers.base import AnalysisResult
from imint.analyzers.nmd import (
    NMDAnalyzer,
    NMD_LEVEL1,
    NMD_LEVEL2,
    nmd_code_to_l1,
    nmd_code_to_l2,
    nmd_raster_to_l1,
    _compute_class_stats,
    _cross_reference,
    _spectral_cross_ref,
    _change_cross_ref,
    _anomaly_cross_ref,
)


# ── Code mapping tests ──────────────────────────────────────────────────────

class TestNMDCodeMapping:
    """Test NMD class code → category name mapping."""

    def test_forest_codes(self):
        """All forest codes (111-118, 121-128) should map to 'forest'."""
        for code in range(111, 119):
            assert nmd_code_to_l1(code) == "forest", f"Code {code} should be forest"
        for code in range(121, 129):
            assert nmd_code_to_l1(code) == "forest", f"Code {code} should be forest"

    def test_wetland(self):
        assert nmd_code_to_l1(2) == "wetland"

    def test_cropland(self):
        assert nmd_code_to_l1(3) == "cropland"

    def test_open_land(self):
        assert nmd_code_to_l1(41) == "open_land"
        assert nmd_code_to_l1(42) == "open_land"

    def test_developed(self):
        assert nmd_code_to_l1(51) == "developed"
        assert nmd_code_to_l1(52) == "developed"
        assert nmd_code_to_l1(53) == "developed"

    def test_water(self):
        assert nmd_code_to_l1(61) == "water"
        assert nmd_code_to_l1(62) == "water"

    def test_unknown_code(self):
        """Unknown codes should return 'unclassified'."""
        assert nmd_code_to_l1(0) == "unclassified"
        assert nmd_code_to_l1(255) == "unclassified"
        assert nmd_code_to_l1(99) == "unclassified"

    def test_l2_forest_pine(self):
        assert nmd_code_to_l2(111) == "forest_pine"

    def test_l2_forest_spruce(self):
        assert nmd_code_to_l2(112) == "forest_spruce"

    def test_l2_developed_roads(self):
        assert nmd_code_to_l2(53) == "developed_roads"

    def test_l2_water_sea(self):
        assert nmd_code_to_l2(62) == "water_sea"

    def test_l2_unknown(self):
        assert nmd_code_to_l2(0) == "unclassified"
        assert nmd_code_to_l2(255) == "unclassified"


class TestNMDRasterToL1:
    """Test raster-level Level 1 conversion."""

    def test_basic_conversion(self):
        """Known codes should map to correct L1 integers."""
        raster = np.array([[111, 3], [51, 61]], dtype=np.uint8)
        l1 = nmd_raster_to_l1(raster)
        assert l1[0, 0] == 1  # forest
        assert l1[0, 1] == 3  # cropland
        assert l1[1, 0] == 5  # developed
        assert l1[1, 1] == 6  # water

    def test_unknown_stays_zero(self):
        """Unknown codes should be 0 (unclassified)."""
        raster = np.array([[0, 255]], dtype=np.uint8)
        l1 = nmd_raster_to_l1(raster)
        assert l1[0, 0] == 0
        assert l1[0, 1] == 0

    def test_dtype_preserved(self):
        raster = np.array([[111]], dtype=np.uint8)
        l1 = nmd_raster_to_l1(raster)
        assert l1.dtype == np.uint8


# ── Class statistics tests ───────────────────────────────────────────────────

class TestNMDClassStats:
    """Test _compute_class_stats()."""

    def test_single_class(self):
        """Raster with a single class should return 100% for that class."""
        raster = np.full((10, 10), 111, dtype=np.uint8)  # All forest pine
        stats = _compute_class_stats(raster)

        assert "forest" in stats["level1"]
        assert stats["level1"]["forest"]["fraction"] == 1.0
        assert stats["level1"]["forest"]["pixel_count"] == 100

        assert "forest_pine" in stats["level2"]
        assert stats["level2"]["forest_pine"]["fraction"] == 1.0

    def test_mixed_classes(self):
        """Raster with mixed classes should report correct fractions."""
        raster = np.zeros((10, 10), dtype=np.uint8)
        raster[:5, :] = 111    # 50 pixels forest
        raster[5:, :5] = 3     # 25 pixels cropland
        raster[5:, 5:] = 61    # 25 pixels water

        stats = _compute_class_stats(raster)
        assert stats["level1"]["forest"]["fraction"] == 0.5
        assert stats["level1"]["cropland"]["fraction"] == 0.25
        assert stats["level1"]["water"]["fraction"] == 0.25

    def test_unclassified(self):
        """Unknown codes should be counted as unclassified."""
        raster = np.full((10, 10), 255, dtype=np.uint8)
        stats = _compute_class_stats(raster)
        assert "unclassified" in stats["level1"]
        assert stats["level1"]["unclassified"]["fraction"] == 1.0

    def test_empty_classes_not_in_stats(self):
        """Classes with 0 pixels should not appear in stats."""
        raster = np.full((10, 10), 111, dtype=np.uint8)
        stats = _compute_class_stats(raster)
        assert "water" not in stats["level1"]
        assert "cropland" not in stats["level1"]


# ── Cross-reference tests ────────────────────────────────────────────────────

class TestNMDCrossReference:
    """Test cross-referencing NMD with other analyzer results."""

    def _make_nmd_raster(self):
        """Create a 10x10 raster: top=forest(111), bottom=water(61)."""
        raster = np.zeros((10, 10), dtype=np.uint8)
        raster[:5, :] = 111  # forest
        raster[5:, :] = 61   # water
        return raster

    def test_spectral_cross_ref_ndvi(self):
        """Mean NDVI should be computed per LULC class."""
        nmd = self._make_nmd_raster()
        ndvi = np.zeros((10, 10), dtype=np.float32)
        ndvi[:5, :] = 0.8   # High NDVI in forest
        ndvi[5:, :] = -0.2  # Low NDVI in water

        spectral = AnalysisResult(
            analyzer="spectral", success=True,
            outputs={"indices": {"NDVI": ndvi, "NDWI": ndvi * -1}, "land_cover": None},
        )

        result = _spectral_cross_ref(nmd, spectral)
        assert abs(result["forest"]["mean_ndvi"] - 0.8) < 0.01
        assert abs(result["water"]["mean_ndvi"] - (-0.2)) < 0.01

    def test_spectral_cross_ref_land_cover(self):
        """Vegetation/water/built-up fractions per LULC class."""
        nmd = self._make_nmd_raster()
        lc = np.full((10, 10), 2, dtype=np.uint8)  # All vegetation
        lc[5:, :] = 1  # Water in bottom half

        spectral = AnalysisResult(
            analyzer="spectral", success=True,
            outputs={"indices": {}, "land_cover": lc},
        )

        result = _spectral_cross_ref(nmd, spectral)
        assert result["forest"]["vegetation_fraction"] == 1.0
        assert result["water"]["water_fraction"] == 1.0

    def test_change_cross_ref(self):
        """Change fraction should be broken down per LULC class."""
        nmd = self._make_nmd_raster()
        change_mask = np.zeros((10, 10), dtype=bool)
        change_mask[:5, :5] = True  # 25 changed pixels in forest

        change = AnalysisResult(
            analyzer="change_detection", success=True,
            outputs={"change_mask": change_mask},
        )

        result = _change_cross_ref(nmd, change)
        assert result["forest"]["change_fraction"] == 0.5  # 25/50
        assert result["forest"]["changed_pixels"] == 25
        assert result["water"]["change_fraction"] == 0.0

    def test_change_cross_ref_shape_mismatch(self):
        """Shape mismatch should return empty dict."""
        nmd = self._make_nmd_raster()
        change_mask = np.zeros((20, 20), dtype=bool)

        change = AnalysisResult(
            analyzer="change_detection", success=True,
            outputs={"change_mask": change_mask},
        )

        result = _change_cross_ref(nmd, change)
        assert result == {}

    def test_anomaly_cross_ref(self):
        """Anomaly detections should be counted per LULC class."""
        nmd = self._make_nmd_raster()

        regions = [
            {"bbox": {"y_min": 1, "y_max": 3, "x_min": 1, "x_max": 3}, "score": 0.9, "label": "anomaly"},
            {"bbox": {"y_min": 6, "y_max": 8, "x_min": 1, "x_max": 3}, "score": 0.7, "label": "anomaly"},
        ]
        objdet = AnalysisResult(
            analyzer="object_detection", success=True,
            outputs={"regions": regions},
        )

        result = _anomaly_cross_ref(nmd, objdet)
        assert result["forest"]["count"] == 1
        assert result["water"]["count"] == 1

    def test_anomaly_cross_ref_empty_regions(self):
        """Empty regions list should return empty dict."""
        nmd = self._make_nmd_raster()
        objdet = AnalysisResult(
            analyzer="object_detection", success=True,
            outputs={"regions": []},
        )

        result = _anomaly_cross_ref(nmd, objdet)
        assert result == {}

    def test_full_cross_reference(self):
        """_cross_reference should include all available analyzers."""
        nmd = self._make_nmd_raster()
        results = [
            AnalysisResult(
                analyzer="change_detection", success=True,
                outputs={"change_mask": np.zeros((10, 10), dtype=bool)},
            ),
            AnalysisResult(
                analyzer="spectral", success=True,
                outputs={"indices": {"NDVI": np.zeros((10, 10))}, "land_cover": None},
            ),
            AnalysisResult(
                analyzer="object_detection", success=True,
                outputs={"regions": []},
            ),
        ]

        cross_ref = _cross_reference(nmd, results)
        assert "change_detection" in cross_ref
        assert "spectral" in cross_ref
        # object_detection returns {} for empty regions, but key should not be present
        # since _anomaly_cross_ref returns {} which is falsy — wait it's always added
        assert "object_detection" in cross_ref

    def test_cross_reference_skips_failed(self):
        """Failed analyzer results should be skipped."""
        nmd = self._make_nmd_raster()
        results = [
            AnalysisResult(
                analyzer="spectral", success=False,
                error="some error",
            ),
        ]

        cross_ref = _cross_reference(nmd, results)
        assert "spectral" not in cross_ref


# ── Graceful degradation ─────────────────────────────────────────────────────

class TestNMDGracefulDegradation:
    """Test NMDAnalyzer graceful degradation."""

    def test_no_coords(self):
        """Missing coords should return nmd_available=False, success=True."""
        analyzer = NMDAnalyzer()
        rgb = np.random.rand(10, 10, 3).astype(np.float32)
        result = analyzer.analyze(rgb, coords=None)
        assert result.success is True
        assert result.outputs["nmd_available"] is False
        assert result.metadata["reason"] == "no_coords"

    def test_fetch_error(self, monkeypatch):
        """FetchError should not crash — returns nmd_available=False."""
        from imint.analyzers import nmd as nmd_module

        def _mock_fetch(*args, **kwargs):
            from imint.fetch import FetchError
            raise FetchError("mock error")

        monkeypatch.setattr(nmd_module, "fetch_nmd_data", _mock_fetch)

        analyzer = NMDAnalyzer()
        rgb = np.random.rand(10, 10, 3).astype(np.float32)
        coords = {"west": 13.0, "south": 55.5, "east": 13.1, "north": 55.6}
        result = analyzer.analyze(rgb, coords=coords)
        assert result.success is True
        assert result.outputs["nmd_available"] is False
        assert "fetch_failed" in result.metadata["reason"]

    def test_with_mock_nmd(self, monkeypatch):
        """With mock NMD data, analyzer should produce full results."""
        from imint.fetch import NMDFetchResult
        from imint.analyzers import nmd as nmd_module

        nmd_raster = np.full((10, 10), 111, dtype=np.uint8)  # All forest
        mock_result = NMDFetchResult(nmd_raster=nmd_raster, from_cache=True)

        monkeypatch.setattr(nmd_module, "fetch_nmd_data", lambda **kwargs: mock_result)

        analyzer = NMDAnalyzer()
        rgb = np.random.rand(10, 10, 3).astype(np.float32)
        coords = {"west": 13.0, "south": 55.5, "east": 13.1, "north": 55.6}

        # Without previous results
        result = analyzer.analyze(rgb, coords=coords)
        assert result.success is True
        assert result.outputs["nmd_available"] is True
        assert "forest" in result.outputs["class_stats"]["level1"]
        assert result.outputs["cross_reference"] == {}

    def test_with_mock_nmd_and_previous_results(self, monkeypatch):
        """With mock NMD + previous results, cross-reference should work."""
        from imint.fetch import NMDFetchResult
        from imint.analyzers import nmd as nmd_module

        nmd_raster = np.full((10, 10), 111, dtype=np.uint8)  # All forest
        mock_result = NMDFetchResult(nmd_raster=nmd_raster, from_cache=True)

        monkeypatch.setattr(nmd_module, "fetch_nmd_data", lambda **kwargs: mock_result)

        analyzer = NMDAnalyzer()
        rgb = np.random.rand(10, 10, 3).astype(np.float32)
        coords = {"west": 13.0, "south": 55.5, "east": 13.1, "north": 55.6}

        previous = [
            AnalysisResult(
                analyzer="spectral", success=True,
                outputs={
                    "indices": {"NDVI": np.full((10, 10), 0.7, dtype=np.float32)},
                    "land_cover": np.full((10, 10), 2, dtype=np.uint8),
                },
            ),
            AnalysisResult(
                analyzer="change_detection", success=True,
                outputs={"change_mask": np.ones((10, 10), dtype=bool)},
            ),
        ]

        result = analyzer.analyze(rgb, coords=coords, previous_results=previous)
        assert result.success is True
        assert result.outputs["nmd_available"] is True
        assert "spectral" in result.outputs["cross_reference"]
        assert "change_detection" in result.outputs["cross_reference"]

        # Check spectral cross-ref values
        sr = result.outputs["cross_reference"]["spectral"]
        assert abs(sr["forest"]["mean_ndvi"] - 0.7) < 0.01
        assert sr["forest"]["vegetation_fraction"] == 1.0

        # Check change cross-ref values
        cr = result.outputs["cross_reference"]["change_detection"]
        assert cr["forest"]["change_fraction"] == 1.0
