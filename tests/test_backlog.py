"""Tests for backlog items: InSAR analyzer, Clay FM, TerraMind."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch, MagicMock


# ══════════════════════════════════════════════════════════════════════════
# InSAR Analyzer
# ══════════════════════════════════════════════════════════════════════════

from imint.analyzers.insar import InSARAnalyzer, S1_WAVELENGTH_M


class TestInSARDependencyChecks:
    """Verify graceful failure when MintPy is missing."""

    def test_analyzer_name(self):
        analyzer = InSARAnalyzer(config={})
        assert analyzer.name == "insar"

    def test_velocity_mode_fails_without_mintpy(self):
        with patch("imint.analyzers.insar._check_mintpy_available", return_value=False):
            analyzer = InSARAnalyzer(config={"mode": "velocity"})
            rgb = np.zeros((32, 32, 3), dtype=np.float32)
            result = analyzer.run(rgb)
            assert not result.success
            assert "mintpy" in result.error.lower()

    def test_unknown_mode_returns_error(self):
        analyzer = InSARAnalyzer(config={"mode": "invalid"})
        rgb = np.zeros((32, 32, 3), dtype=np.float32)
        result = analyzer.run(rgb)
        assert not result.success
        assert "invalid" in result.error

    def test_differential_requires_phase_bands(self):
        analyzer = InSARAnalyzer(config={"mode": "differential"})
        rgb = np.zeros((32, 32, 3), dtype=np.float32)
        result = analyzer.run(rgb, bands=None)
        assert not result.success
        assert "phase_1" in result.error


class TestInSARDifferential:
    """Test differential InSAR (no MintPy needed)."""

    def test_basic_differential(self):
        h, w = 64, 64
        phase_1 = np.zeros((h, w), dtype=np.float32)
        phase_2 = np.ones((h, w), dtype=np.float32) * 0.5  # 0.5 radians shift

        analyzer = InSARAnalyzer(config={"mode": "differential"})
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        result = analyzer.run(rgb, bands={"phase_1": phase_1, "phase_2": phase_2})

        assert result.success
        assert "displacement_mm" in result.outputs
        assert "coherence" in result.outputs
        assert "stats" in result.outputs
        assert result.outputs["displacement_mm"].shape == (h, w)
        assert result.metadata["mode"] == "differential"

    def test_displacement_magnitude(self):
        """Known phase shift should produce known displacement."""
        h, w = 32, 32
        phase_1 = np.zeros((h, w), dtype=np.float32)
        # π radians phase shift → half wavelength / 4π → known displacement
        phase_2 = np.full((h, w), np.pi, dtype=np.float32)

        analyzer = InSARAnalyzer(config={
            "mode": "differential",
            "wavelength": S1_WAVELENGTH_M,
        })
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        result = analyzer.run(rgb, bands={"phase_1": phase_1, "phase_2": phase_2})

        assert result.success
        displacement = result.outputs["displacement_mm"]
        # π wrapped to [-π, π] → -π (np.angle wraps) → negative displacement
        expected_mm = (S1_WAVELENGTH_M / (4 * np.pi)) * (-np.pi) * 1000.0
        np.testing.assert_allclose(
            displacement[0, 0], expected_mm, rtol=0.01,
        )

    def test_flagging(self):
        h, w = 32, 32
        phase_1 = np.zeros((h, w), dtype=np.float32)
        phase_2 = np.full((h, w), 2.0, dtype=np.float32)  # Large displacement

        analyzer = InSARAnalyzer(config={
            "mode": "differential",
            "threshold_mm_yr": 1.0,  # Low threshold → everything flagged
        })
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        result = analyzer.run(rgb, bands={"phase_1": phase_1, "phase_2": phase_2})

        assert result.success
        stats = result.outputs["stats"]
        assert stats["flagged_fraction"] > 0

    def test_zero_displacement(self):
        h, w = 32, 32
        phase = np.zeros((h, w), dtype=np.float32)

        analyzer = InSARAnalyzer(config={"mode": "differential"})
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        result = analyzer.run(rgb, bands={"phase_1": phase, "phase_2": phase})

        assert result.success
        np.testing.assert_allclose(
            result.outputs["displacement_mm"], 0.0, atol=1e-6,
        )


class TestInSAREngineRegistration:
    """Verify InSAR is in the engine registry."""

    def test_in_analyzer_registry(self):
        from imint.engine import ANALYZER_REGISTRY
        assert "insar" in ANALYZER_REGISTRY
        assert ANALYZER_REGISTRY["insar"] is InSARAnalyzer

    def test_disabled_by_default_in_config(self):
        import yaml
        with open("config/analyzers.yaml") as f:
            config = yaml.safe_load(f)
        assert "insar" in config
        assert config["insar"]["enabled"] is False


# ══════════════════════════════════════════════════════════════════════════
# Clay Foundation Model
# ══════════════════════════════════════════════════════════════════════════

class TestClayFM:
    """Test Clay FM integration module."""

    def test_constants(self):
        from imint.fm.clay import CLAY_HF_REPO, CLAY_BANDS_S2, CLAY_BANDS_S1
        assert "clay" in CLAY_HF_REPO.lower()
        assert "B02" in CLAY_BANDS_S2
        assert "VV" in CLAY_BANDS_S1

    def test_check_available(self):
        from imint.fm.clay import check_clay_available
        # Should return True if torch is installed, False otherwise
        result = check_clay_available()
        assert isinstance(result, bool)

    def test_normalisation_shapes(self):
        from imint.fm.clay import CLAY_S2_MEAN, CLAY_S2_STD
        assert CLAY_S2_MEAN.shape == (7,)
        assert CLAY_S2_STD.shape == (7,)

    def test_prithvi_to_clay_conversion(self):
        from imint.fm.clay import prithvi_bands_to_clay
        bands = {
            "B02": np.ones((32, 32), dtype=np.float32) * 0.1,
            "B03": np.ones((32, 32), dtype=np.float32) * 0.2,
            "B04": np.ones((32, 32), dtype=np.float32) * 0.3,
            "B8A": np.ones((32, 32), dtype=np.float32) * 0.4,
            "B11": np.ones((32, 32), dtype=np.float32) * 0.5,
            "B12": np.ones((32, 32), dtype=np.float32) * 0.6,
        }
        clay_stack = prithvi_bands_to_clay(bands)
        assert clay_stack.shape == (7, 32, 32)
        # B08 approximated from B8A
        np.testing.assert_allclose(clay_stack[3], 0.4)  # B08 ≈ B8A


# ══════════════════════════════════════════════════════════════════════════
# TerraMind
# ══════════════════════════════════════════════════════════════════════════

class TestTerraMind:
    """Test TerraMind integration module."""

    def test_modalities(self):
        from imint.fm.terramind import MODALITIES
        assert "optical" in MODALITIES
        assert "sar" in MODALITIES
        assert "dem" in MODALITIES
        assert "lulc" in MODALITIES

    def test_optical_shape(self):
        from imint.fm.terramind import MODALITIES
        assert MODALITIES["optical"]["input_shape"] == (6, 224, 224)

    def test_sar_shape(self):
        from imint.fm.terramind import MODALITIES
        assert MODALITIES["sar"]["input_shape"] == (2, 224, 224)

    def test_check_available(self):
        from imint.fm.terramind import check_terramind_available
        result = check_terramind_available()
        assert isinstance(result, bool)

    def test_load_fails_without_deps(self):
        from imint.fm.terramind import load_terramind_model
        with patch("imint.fm.terramind.check_terramind_available", return_value=False):
            with pytest.raises(ImportError, match="terratorch"):
                load_terramind_model()

    def test_invalid_modality(self):
        from imint.fm.terramind import load_terramind_model
        with patch("imint.fm.terramind.check_terramind_available", return_value=True):
            with pytest.raises(ValueError, match="Unknown input modality"):
                load_terramind_model(modality_in="radar")
