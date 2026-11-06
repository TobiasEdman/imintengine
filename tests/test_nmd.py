"""
Tests for the NMD (Nationellt Marktäckedata) analyzer.

Tests class code mapping, class statistics computation, cross-reference
logic, and graceful degradation when NMD data is unavailable.
"""
from __future__ import annotations

import numpy as np
import pytest

from imint.analyzers.base import AnalysisResult
from imint.schema.nmd2023 import NMD2023_CLASSES
from imint.analyzers.nmd import (
    NMDAnalyzer,
    NMD2018_LEGEND,
    NMD_LEVEL1,
    NMD_LEVEL2,
    L2_CLASSES,
    nmd_code_to_l1,
    nmd_code_to_l2,
    nmd_raster_to_l1,
    nmd_raster_to_l2,
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


class TestNMDRasterToL2:
    """Test raster-level Level 2 conversion."""

    def test_basic_conversion(self):
        """Known codes should map to correct L2 integers."""
        raster = np.array([[111, 112], [53, 62]], dtype=np.uint8)
        l2 = nmd_raster_to_l2(raster)
        # forest_pine=1, forest_spruce=2, developed_roads=17, water_sea=19
        assert l2[0, 0] == L2_CLASSES.index("forest_pine") + 1
        assert l2[0, 1] == L2_CLASSES.index("forest_spruce") + 1
        assert l2[1, 0] == L2_CLASSES.index("developed_roads") + 1
        assert l2[1, 1] == L2_CLASSES.index("water_sea") + 1

    def test_unknown_stays_zero(self):
        """Unknown codes should be 0 (unclassified)."""
        raster = np.array([[0, 255]], dtype=np.uint8)
        l2 = nmd_raster_to_l2(raster)
        assert l2[0, 0] == 0
        assert l2[0, 1] == 0

    def test_all_l2_classes_mapped(self):
        """Every L2 class should produce a unique non-zero code."""
        codes_seen = set()
        for name, nmd_codes in NMD_LEVEL2.items():
            code = list(nmd_codes)[0]
            raster = np.array([[code]], dtype=np.uint8)
            l2 = nmd_raster_to_l2(raster)
            val = int(l2[0, 0])
            assert val > 0, f"{name} should map to non-zero"
            codes_seen.add(val)
        assert len(codes_seen) == len(NMD_LEVEL2), "All L2 classes should have unique codes"

    def test_dtype_preserved(self):
        raster = np.array([[111]], dtype=np.uint8)
        l2 = nmd_raster_to_l2(raster)
        assert l2.dtype == np.uint8


class TestNMD2018LegendMatchesTheTables:
    """The 2018 legend reproduces the per-code loops it replaced, for every uint8 code."""

    @staticmethod
    def _loop_index(raster, groups):
        out = np.zeros_like(raster, dtype=np.uint8)
        for idx, codes in enumerate(groups.values(), start=1):
            for code in codes:
                out[raster == code] = idx
        return out

    def test_index_rasters_equal_the_loops(self):
        every_code = np.arange(256, dtype=np.uint8).reshape(16, 16)
        np.testing.assert_array_equal(
            nmd_raster_to_l1(every_code), self._loop_index(every_code, NMD_LEVEL1))
        np.testing.assert_array_equal(
            nmd_raster_to_l2(every_code), self._loop_index(every_code, NMD_LEVEL2))

    def test_stats_equal_the_isin_counts(self):
        rng = np.random.default_rng(0)
        codes = sorted(set().union(*NMD_LEVEL2.values()) | {0, 2, 99, 255})
        raster = rng.choice(np.array(codes, dtype=np.uint8), size=(40, 30))
        stats = _compute_class_stats(raster, NMD2018_LEGEND)
        known_l1 = set().union(*NMD_LEVEL1.values())
        expected_l1 = {name: int(np.isin(raster, list(c)).sum()) for name, c in NMD_LEVEL1.items()}
        expected_l1["unclassified"] = int((~np.isin(raster, list(known_l1))).sum())
        expected_l2 = {name: int(np.isin(raster, list(c)).sum()) for name, c in NMD_LEVEL2.items()}
        assert {k: v["pixel_count"] for k, v in stats["level1"].items()} == {
            k: v for k, v in expected_l1.items() if v}
        assert {k: v["pixel_count"] for k, v in stats["level2"].items()} == {
            k: v for k, v in expected_l2.items() if v}
        assert list(stats["level1"])[-1] == "unclassified"

    def test_2018_tolerates_unknown_codes(self):
        NMD2018_LEGEND.check(np.array([[0, 99, 255]], dtype=np.uint8))


# ── Class statistics tests ───────────────────────────────────────────────────

class TestNMDClassStats:
    """Test _compute_class_stats()."""

    def test_single_class(self):
        """Raster with a single class should return 100% for that class."""
        raster = np.full((10, 10), 111, dtype=np.uint8)  # All forest pine
        stats = _compute_class_stats(raster, NMD2018_LEGEND)

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

        stats = _compute_class_stats(raster, NMD2018_LEGEND)
        assert stats["level1"]["forest"]["fraction"] == 0.5
        assert stats["level1"]["cropland"]["fraction"] == 0.25
        assert stats["level1"]["water"]["fraction"] == 0.25

    def test_unclassified(self):
        """Unknown codes should be counted as unclassified."""
        raster = np.full((10, 10), 255, dtype=np.uint8)
        stats = _compute_class_stats(raster, NMD2018_LEGEND)
        assert "unclassified" in stats["level1"]
        assert stats["level1"]["unclassified"]["fraction"] == 1.0

    def test_empty_classes_not_in_stats(self):
        """Classes with 0 pixels should not appear in stats."""
        raster = np.full((10, 10), 111, dtype=np.uint8)
        stats = _compute_class_stats(raster, NMD2018_LEGEND)
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
        """Mean NDVI should be computed per L2 LULC class."""
        nmd = self._make_nmd_raster()
        ndvi = np.zeros((10, 10), dtype=np.float32)
        ndvi[:5, :] = 0.8   # High NDVI in forest_pine (111)
        ndvi[5:, :] = -0.2  # Low NDVI in water_lakes (61)

        spectral = AnalysisResult(
            analyzer="spectral", success=True,
            outputs={"indices": {"NDVI": ndvi, "NDWI": ndvi * -1}},
        )

        result = _spectral_cross_ref(nmd, spectral, NMD2018_LEGEND)
        assert abs(result["forest_pine"]["mean_ndvi"] - 0.8) < 0.01
        assert abs(result["water_lakes"]["mean_ndvi"] - (-0.2)) < 0.01

    def test_change_cross_ref(self):
        """Change fraction should be broken down per L2 LULC class."""
        nmd = self._make_nmd_raster()
        change_mask = np.zeros((10, 10), dtype=bool)
        change_mask[:5, :5] = True  # 25 changed pixels in forest_pine (111)

        change = AnalysisResult(
            analyzer="change_detection", success=True,
            outputs={"change_mask": change_mask},
        )

        result = _change_cross_ref(nmd, change, NMD2018_LEGEND)
        assert result["forest_pine"]["change_fraction"] == 0.5  # 25/50
        assert result["forest_pine"]["changed_pixels"] == 25
        assert result["water_lakes"]["change_fraction"] == 0.0

    def test_change_cross_ref_shape_mismatch(self):
        """Shape mismatch should return empty dict."""
        nmd = self._make_nmd_raster()
        change_mask = np.zeros((20, 20), dtype=bool)

        change = AnalysisResult(
            analyzer="change_detection", success=True,
            outputs={"change_mask": change_mask},
        )

        result = _change_cross_ref(nmd, change, NMD2018_LEGEND)
        assert result == {}

    def test_anomaly_cross_ref(self):
        """Anomaly detections should be counted per L2 LULC class."""
        nmd = self._make_nmd_raster()

        regions = [
            {"bbox": {"y_min": 1, "y_max": 3, "x_min": 1, "x_max": 3}, "score": 0.9, "label": "anomaly"},
            {"bbox": {"y_min": 6, "y_max": 8, "x_min": 1, "x_max": 3}, "score": 0.7, "label": "anomaly"},
        ]
        objdet = AnalysisResult(
            analyzer="object_detection", success=True,
            outputs={"regions": regions},
        )

        result = _anomaly_cross_ref(nmd, objdet, NMD2018_LEGEND)
        assert result["forest_pine"]["count"] == 1
        assert result["water_lakes"]["count"] == 1

    def test_anomaly_cross_ref_empty_regions(self):
        """Empty regions list should return empty dict."""
        nmd = self._make_nmd_raster()
        objdet = AnalysisResult(
            analyzer="object_detection", success=True,
            outputs={"regions": []},
        )

        result = _anomaly_cross_ref(nmd, objdet, NMD2018_LEGEND)
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
                outputs={"indices": {"NDVI": np.zeros((10, 10))}},
            ),
            AnalysisResult(
                analyzer="object_detection", success=True,
                outputs={"regions": []},
            ),
        ]

        cross_ref = _cross_reference(nmd, results, NMD2018_LEGEND)
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

        cross_ref = _cross_reference(nmd, results, NMD2018_LEGEND)
        assert "spectral" not in cross_ref


# ── Graceful degradation ─────────────────────────────────────────────────────

class TestNMDGracefulDegradation:
    """Test NMDAnalyzer graceful degradation."""

    def test_no_coords(self):
        """Missing coords should return nmd_available=False, success=True."""
        analyzer = NMDAnalyzer(config={"product": 2018})
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

        analyzer = NMDAnalyzer(config={"product": 2018})
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

        analyzer = NMDAnalyzer(config={"product": 2018})
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

        analyzer = NMDAnalyzer(config={"product": 2018})
        rgb = np.random.rand(10, 10, 3).astype(np.float32)
        coords = {"west": 13.0, "south": 55.5, "east": 13.1, "north": 55.6}

        previous = [
            AnalysisResult(
                analyzer="spectral", success=True,
                outputs={
                    "indices": {"NDVI": np.full((10, 10), 0.7, dtype=np.float32)},
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

        # Check spectral cross-ref values (now L2 keys)
        sr = result.outputs["cross_reference"]["spectral"]
        assert abs(sr["forest_pine"]["mean_ndvi"] - 0.7) < 0.01

        # Check change cross-ref values (now L2 keys)
        cr = result.outputs["cross_reference"]["change_detection"]
        assert cr["forest_pine"]["change_fraction"] == 1.0


# ── NMD 2023 in the analyzer ─────────────────────────────────────────────────

class TestNMD2023Analyzer:
    """The engine's NMD analyzer on NMD 2023: own classes, explicit product."""

    COORDS = {"west": 15.42, "south": 61.92, "east": 15.47, "north": 61.97}

    @staticmethod
    def _raster():
        """Rows: 111 forest, 4211 open land (>255), 62 sea, 23 låg fjällskog
        på våtmark, 54 torvtäkt, 0 outside coverage."""
        return np.repeat(np.array([111, 4211, 62, 23, 54, 0], dtype=np.uint16), 2)[:, None].repeat(5, 1)

    def _analyze(self, monkeypatch, raster, config=None, previous=None):
        from imint.fetch import NMDFetchResult
        from imint.analyzers import nmd as nmd_module

        calls = []

        def _fetch(**kwargs):
            calls.append(kwargs)
            return NMDFetchResult(nmd_raster=raster)

        monkeypatch.setattr(nmd_module, "fetch_nmd_data", _fetch)
        analyzer = NMDAnalyzer(config=config or {"product": 2023})
        rgb = np.zeros((*raster.shape, 3), dtype=np.float32)
        return analyzer.run(rgb, coords=self.COORDS, previous_results=previous), calls

    def test_product_is_required(self):
        analyzer = NMDAnalyzer(config={"enabled": True})
        rgb = np.zeros((4, 4, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="set 'product'"):
            analyzer.analyze(rgb, coords=None)
        result = analyzer.run(rgb, coords=self.COORDS)
        assert result.success is False and "product" in result.error
        for bad in ("2023", 2020):
            with pytest.raises(ValueError, match="NMD product must be one of"):
                NMDAnalyzer(config={"product": bad}).analyze(rgb, coords=None)

    def test_requests_2023_and_passes_the_raster_path(self, monkeypatch):
        result, calls = self._analyze(
            monkeypatch, self._raster(), {"product": 2023, "raster_path": "/x/NMD2023.tif"})
        assert result.success, result.error
        assert calls == [{"coords": self.COORDS, "target_shape": (12, 5),
                          "product": 2023, "raster_path": "/x/NMD2023.tif"}]
        assert result.outputs["product"] == 2023

    def test_own_classes_and_klassgrupp_statistics(self, monkeypatch):
        result, _ = self._analyze(monkeypatch, self._raster())
        stats = result.outputs["class_stats"]
        sixth = {"pixel_count": 10, "fraction": round(1 / 6, 4)}
        assert stats["level2"] == {"111": sixth, "4211": sixth, "62": sixth, "23": sixth, "54": sixth}
        assert stats["level1"] == {"forest": sixth, "wetland": sixth, "open_land": sixth,
                                   "developed": sixth, "water": sixth, "outside": sixth}
        codes = list(NMD2023_CLASSES)
        l2 = result.outputs["l2_raster"]
        assert l2.dtype == np.uint8
        assert [int(l2[r, 0]) for r in (0, 2, 4, 6, 8, 10)] == [
            codes.index(111) + 1, codes.index(4211) + 1, codes.index(62) + 1,
            codes.index(23) + 1, codes.index(54) + 1, 0]
        l1 = result.outputs["l1_raster"]
        assert [int(l1[r, 0]) for r in (0, 2, 4, 6, 8, 10)] == [1, 4, 6, 2, 5, 0]

    def test_a_code_outside_the_official_legend_fails_the_analyzer(self, monkeypatch):
        raster = self._raster()
        raster[0, 0] = 41  # NMD 2018 "övrig öppen mark utan vegetation", gone in 2023
        result, _ = self._analyze(monkeypatch, raster)
        assert result.success is False
        assert "outside its legend: [41]" in result.error

    def test_cross_reference_is_keyed_by_2023_code(self, monkeypatch):
        raster = self._raster()
        ndvi = np.where(raster == 4211, 0.3, 0.8).astype(np.float32)
        previous = [AnalysisResult(analyzer="spectral", success=True,
                                   outputs={"indices": {"NDVI": ndvi}})]
        result, _ = self._analyze(monkeypatch, raster, previous=previous)
        spectral = result.outputs["cross_reference"]["spectral"]
        assert set(spectral) == {"111", "4211", "62", "23", "54"}
        assert spectral["4211"]["mean_ndvi"] == pytest.approx(0.3)
        assert spectral["111"]["mean_ndvi"] == pytest.approx(0.8)

    def test_the_legends_cover_exactly_the_fetch_products(self):
        from imint.analyzers.nmd import NMD_LEGENDS
        from imint.fetch import NMD_PRODUCTS

        assert tuple(NMD_LEGENDS) == NMD_PRODUCTS


def test_engine_configs_choose_nmd_2023_explicitly():
    """The engine default is NMD 2023, stated in every engine config."""
    import yaml
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for rel in ("config/analyzers.yaml", "imint/config/analyzers.yaml",
                "config/analyzers_full.yaml"):
        assert yaml.safe_load((root / rel).read_text())["nmd"]["product"] == 2023, rel
    assert '"nmd": {"enabled": True, "product": 2023}' in (root / "run_full_analysis.py").read_text()
