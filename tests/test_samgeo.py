"""Tests for imint/analyzers/samgeo.py — SAMGeo zero-shot segmentation analyzer."""
from __future__ import annotations

import os
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

from imint.analyzers.samgeo import SAMGeoAnalyzer


# ── Dependency checks ────────────────────────────────────────────────────────

class TestSAMGeoDependencyChecks:
    """Verify graceful failure when dependencies are missing."""

    def test_fails_without_torch(self):
        """Should return success=False with install instruction when torch is missing."""
        with patch("imint.analyzers.samgeo._check_torch_available", return_value=False):
            analyzer = SAMGeoAnalyzer(config={})
            rgb = np.zeros((32, 32, 3), dtype=np.float32)
            result = analyzer.run(rgb)

            assert not result.success
            assert "torch" in result.error.lower()
            assert "pip install" in result.error

    def test_fails_without_samgeo(self):
        """Should return success=False with install instruction when samgeo is missing."""
        with patch("imint.analyzers.samgeo._check_torch_available", return_value=True), \
             patch("imint.analyzers.samgeo._check_samgeo_available", return_value=False):
            analyzer = SAMGeoAnalyzer(config={})
            rgb = np.zeros((32, 32, 3), dtype=np.float32)
            result = analyzer.run(rgb)

            assert not result.success
            assert "segment-geospatial" in result.error
            assert "pip install" in result.error

    def test_analyzer_name(self):
        """Analyzer should be named 'samgeo'."""
        analyzer = SAMGeoAnalyzer(config={})
        assert analyzer.name == "samgeo"


# ── Mode validation ──────────────────────────────────────────────────────────

class TestSAMGeoModeValidation:
    """Verify mode handling."""

    def test_unknown_mode_returns_error(self):
        """Should fail with clear error for unknown mode."""
        with patch("imint.analyzers.samgeo._check_torch_available", return_value=True), \
             patch("imint.analyzers.samgeo._check_samgeo_available", return_value=True):
            analyzer = SAMGeoAnalyzer(config={"mode": "invalid"})
            rgb = np.zeros((32, 32, 3), dtype=np.float32)
            result = analyzer.run(rgb)

            assert not result.success
            assert "invalid" in result.error
            assert "automatic" in result.error

    def test_points_mode_requires_points(self):
        """Points mode should fail without 'points' config."""
        with patch("imint.analyzers.samgeo._check_torch_available", return_value=True), \
             patch("imint.analyzers.samgeo._check_samgeo_available", return_value=True):
            analyzer = SAMGeoAnalyzer(config={"mode": "points"})
            rgb = np.zeros((32, 32, 3), dtype=np.float32)
            result = analyzer.run(rgb)

            assert not result.success
            assert "points" in result.error.lower()

    def test_text_mode_requires_text_prompt(self):
        """Text mode should fail without 'text_prompt' config."""
        with patch("imint.analyzers.samgeo._check_torch_available", return_value=True), \
             patch("imint.analyzers.samgeo._check_samgeo_available", return_value=True):
            analyzer = SAMGeoAnalyzer(config={"mode": "text"})
            rgb = np.zeros((32, 32, 3), dtype=np.float32)
            result = analyzer.run(rgb)

            assert not result.success
            assert "text_prompt" in result.error


# ── Automatic mode (mocked) ─────────────────────────────────────────────────

class TestSAMGeoAutomatic:
    """Test automatic segmentation with mocked SAMGeo model."""

    def _make_mock_rasterio(self, seg_data):
        """Create a mock rasterio context manager that returns seg_data."""
        mock_src = MagicMock()
        mock_src.read.return_value = seg_data
        mock_src.__enter__ = MagicMock(return_value=mock_src)
        mock_src.__exit__ = MagicMock(return_value=False)
        return MagicMock(return_value=mock_src)

    def _patch_samgeo_imports(self, mock_sam):
        """Create a fake samgeo module so patch() can resolve it."""
        import sys
        import types
        samgeo_mod = types.ModuleType("samgeo")
        samgeo_samgeo_mod = types.ModuleType("samgeo.samgeo")
        samgeo_samgeo_mod.SamGeo = MagicMock(return_value=mock_sam)
        samgeo_samgeo_mod.SamGeo2 = MagicMock(return_value=mock_sam)
        samgeo_mod.samgeo = samgeo_samgeo_mod
        samgeo_mod.SamGeo = samgeo_samgeo_mod.SamGeo
        sys.modules["samgeo"] = samgeo_mod
        sys.modules["samgeo.samgeo"] = samgeo_samgeo_mod
        return samgeo_mod

    def _cleanup_samgeo_imports(self):
        import sys
        sys.modules.pop("samgeo", None)
        sys.modules.pop("samgeo.samgeo", None)

    def test_automatic_output_structure(self, tmp_output_dir):
        """Automatic mode should return seg_mask and segment_stats."""
        h, w = 64, 64
        mock_seg = np.zeros((h, w), dtype=np.int32)
        mock_seg[10:30, 10:30] = 1
        mock_seg[35:55, 35:55] = 2

        mock_sam = MagicMock()
        self._patch_samgeo_imports(mock_sam)

        try:
            with patch("imint.analyzers.samgeo._check_torch_available", return_value=True), \
                 patch("imint.analyzers.samgeo._check_samgeo_available", return_value=True), \
                 patch("imint.analyzers.samgeo._get_device", return_value="cpu"), \
                 patch("rasterio.open", self._make_mock_rasterio(mock_seg)):
                analyzer = SAMGeoAnalyzer(config={
                    "mode": "automatic",
                    "min_mask_area": 10,
                    "sam_version": 2,
                })
                rgb = np.random.rand(h, w, 3).astype(np.float32)
                result = analyzer.run(rgb, output_dir=tmp_output_dir)
        finally:
            self._cleanup_samgeo_imports()

        assert result.success
        assert result.analyzer == "samgeo"
        assert "seg_mask" in result.outputs
        assert "segment_stats" in result.outputs

        stats = result.outputs["segment_stats"]
        assert "n_segments" in stats
        assert stats["n_segments"] == 2
        assert "segmented_fraction" in stats
        assert stats["segmented_fraction"] > 0

        assert result.metadata["mode"] == "automatic"
        assert result.metadata["sam_version"] == 2

    def test_min_mask_area_filtering(self, tmp_output_dir):
        """Segments below min_mask_area should be filtered out."""
        h, w = 64, 64
        mock_seg = np.zeros((h, w), dtype=np.int32)
        mock_seg[0:2, 0:2] = 1   # 4 pixels — below threshold
        mock_seg[10:30, 10:30] = 2  # 400 pixels — above threshold

        mock_sam = MagicMock()
        self._patch_samgeo_imports(mock_sam)

        try:
            with patch("imint.analyzers.samgeo._check_torch_available", return_value=True), \
                 patch("imint.analyzers.samgeo._check_samgeo_available", return_value=True), \
                 patch("imint.analyzers.samgeo._get_device", return_value="cpu"), \
                 patch("rasterio.open", self._make_mock_rasterio(mock_seg)):
                analyzer = SAMGeoAnalyzer(config={
                    "mode": "automatic",
                    "min_mask_area": 50,
                })
                rgb = np.random.rand(h, w, 3).astype(np.float32)
                result = analyzer.run(rgb, output_dir=tmp_output_dir)
        finally:
            self._cleanup_samgeo_imports()

        assert result.success
        seg_mask = result.outputs["seg_mask"]
        # Segment 1 (4 pixels) should have been filtered
        assert np.sum(seg_mask == 1) == 0
        # Segment 2 (400 pixels) should remain
        assert np.sum(seg_mask == 2) == 400
        assert result.outputs["segment_stats"]["n_segments"] == 1


# ── Engine integration ───────────────────────────────────────────────────────

class TestSAMGeoEngineRegistration:
    """Verify SAMGeo is registered in the engine."""

    def test_in_analyzer_registry(self):
        """SAMGeoAnalyzer should be in the engine's registry."""
        from imint.engine import ANALYZER_REGISTRY
        assert "samgeo" in ANALYZER_REGISTRY
        assert ANALYZER_REGISTRY["samgeo"] is SAMGeoAnalyzer

    def test_disabled_by_default_in_config(self):
        """SAMGeo should be disabled by default (requires pip install)."""
        import yaml
        with open("config/analyzers.yaml") as f:
            config = yaml.safe_load(f)
        assert "samgeo" in config
        assert config["samgeo"]["enabled"] is False

    def test_config_has_mode(self):
        """Config should specify default mode."""
        import yaml
        with open("config/analyzers.yaml") as f:
            config = yaml.safe_load(f)
        assert config["samgeo"]["mode"] == "automatic"


# ── Export integration ───────────────────────────────────────────────────────

class TestSAMGeoExport:
    """Test that _export handles samgeo results correctly."""

    def test_export_creates_overlay(self, tmp_output_dir):
        """_export should create a clean PNG overlay for samgeo results."""
        from imint.engine import _export
        from imint.analyzers.base import AnalysisResult
        from imint.job import IMINTJob

        seg_mask = np.zeros((32, 32), dtype=np.int32)
        seg_mask[10:20, 10:20] = 1
        seg_mask[20:30, 20:30] = 2

        result = AnalysisResult(
            analyzer="samgeo",
            success=True,
            outputs={
                "seg_mask": seg_mask,
                "segment_stats": {"n_segments": 2},
            },
            metadata={"mode": "automatic"},
        )

        rgb = np.random.rand(32, 32, 3).astype(np.float32)
        job = IMINTJob(
            date="2024-06-15",
            coords={"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0},
            rgb=rgb,
            output_dir=tmp_output_dir,
        )

        _export(result, job)

        # Should have created the overlay PNG
        expected_path = os.path.join(tmp_output_dir, "2024-06-15_samgeo_clean.png")
        assert os.path.exists(expected_path)

        from PIL import Image
        img = Image.open(expected_path)
        assert img.format == "PNG"
        assert img.size == (32, 32)


# ── HTML report integration ──────────────────────────────────────────────────

class TestSAMGeoHTMLReport:
    """Test that samgeo is included in the HTML report path candidates."""

    def test_samgeo_in_path_candidates(self):
        """The engine should look for samgeo_clean.png for the HTML report."""
        from imint.engine import _generate_html_report
        import inspect
        source = inspect.getsource(_generate_html_report)
        assert "samgeo" in source
