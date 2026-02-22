"""Tests for imint/fm/terratorch_loader.py — shared FM infrastructure."""
from __future__ import annotations

from unittest.mock import patch, MagicMock
import numpy as np
import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from imint.fm.terratorch_loader import (
    check_terratorch_available,
    check_torch_available,
    get_device,
    load_backbone,
    bands_to_tensor,
)


# ── check_terratorch_available ───────────────────────────────────────────────

class TestCheckTerraTorchAvailable:
    """Test lazy import detection."""

    def test_returns_false_when_not_installed(self):
        """Should return False when terratorch is not importable."""
        with patch.dict("sys.modules", {"terratorch": None}):
            # Force ImportError by making the module None
            assert check_terratorch_available() is False

    def test_returns_false_when_torch_missing(self):
        """Should return False when torch is not importable."""
        with patch.dict("sys.modules", {"torch": None}):
            assert check_terratorch_available() is False


# ── check_torch_available ────────────────────────────────────────────────────

class TestCheckTorchAvailable:
    """Test torch-only import detection."""

    def test_returns_false_when_not_installed(self):
        """Should return False when torch is not importable."""
        with patch.dict("sys.modules", {"torch": None}):
            assert check_torch_available() is False

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
    def test_returns_true_when_installed(self):
        """Should return True when torch is importable."""
        assert check_torch_available() is True


# ── get_device ───────────────────────────────────────────────────────────────

class TestGetDevice:
    """Test device detection."""

    @patch("imint.fm.terratorch_loader.get_device.__module__", "imint.fm.terratorch_loader")
    def test_returns_string(self):
        """get_device should always return a string."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = get_device()
            assert isinstance(result, str)
            assert result in {"cpu", "cuda", "mps"}

    def test_prefers_cuda_when_available(self):
        """Should return 'cuda' when CUDA is available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = get_device()
            assert result == "cuda"

    def test_falls_back_to_cpu(self):
        """Should return 'cpu' when no GPU is available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = get_device()
            assert result == "cpu"

    def test_explicit_cpu_preference(self):
        """Explicit 'cpu' should always work."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = get_device(preferred="cpu")
            assert result == "cpu"


# ── load_backbone ────────────────────────────────────────────────────────────

class TestLoadBackbone:
    """Test backbone loading via TerraTorch registry."""

    def test_raises_import_error_without_terratorch(self):
        """Should raise ImportError with install instructions for non-Prithvi models."""
        with patch.dict("sys.modules", {"terratorch": None, "terratorch.registry": None}):
            with pytest.raises(ImportError, match="terratorch"):
                load_backbone("thor_eo_v1_base")

    def test_loads_from_registry(self):
        """Should call BACKBONE_REGISTRY.build() with correct args."""
        mock_registry = MagicMock()
        mock_model = MagicMock()
        mock_registry.BACKBONE_REGISTRY.build.return_value = mock_model

        mock_terratorch = MagicMock()
        mock_terratorch.registry = mock_registry

        with patch.dict("sys.modules", {
            "terratorch": mock_terratorch,
            "terratorch.registry": mock_registry,
        }):
            result = load_backbone("prithvi_eo_v2_300m_tl", pretrained=True)

        mock_registry.BACKBONE_REGISTRY.build.assert_called_once_with(
            "prithvi_eo_v2_300m_tl", pretrained=True,
        )
        mock_model.eval.assert_called_once()
        assert result is mock_model


# ── bands_to_tensor ──────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestBandsToTensor:
    """Test band dict → tensor conversion."""

    def _make_bands(self, h=32, w=32):
        """Create a test band dict with 7 bands."""
        return {
            "B02": np.full((h, w), 0.1, dtype=np.float32),
            "B03": np.full((h, w), 0.2, dtype=np.float32),
            "B04": np.full((h, w), 0.3, dtype=np.float32),
            "B08": np.full((h, w), 0.5, dtype=np.float32),
            "B8A": np.full((h, w), 0.45, dtype=np.float32),
            "B11": np.full((h, w), 0.15, dtype=np.float32),
            "B12": np.full((h, w), 0.12, dtype=np.float32),
        }

    def test_correct_shape(self):
        """Output tensor should be (1, C, H, W)."""
        import torch

        bands = self._make_bands(32, 32)
        band_order = ["B02", "B03", "B04", "B8A", "B11", "B12"]

        tensor = bands_to_tensor(bands, band_order, device="cpu")

        assert tensor.shape == (1, 6, 32, 32)
        assert tensor.dtype == torch.float32

    def test_band_order_preserved(self):
        """Bands should be stacked in the specified order."""
        bands = self._make_bands(4, 4)
        band_order = ["B04", "B02"]  # Reversed order

        tensor = bands_to_tensor(bands, band_order, device="cpu")

        # B04 = 0.3, B02 = 0.1
        assert abs(tensor[0, 0].mean().item() - 0.3) < 1e-5  # First channel = B04
        assert abs(tensor[0, 1].mean().item() - 0.1) < 1e-5  # Second channel = B02

    def test_missing_band_raises_key_error(self):
        """Should raise KeyError with helpful message when band is missing."""
        bands = {"B02": np.zeros((4, 4)), "B03": np.zeros((4, 4))}
        band_order = ["B02", "B03", "B04"]

        with pytest.raises(KeyError, match="B04"):
            bands_to_tensor(bands, band_order, device="cpu")

    def test_single_band(self):
        """Should work with a single band."""
        bands = {"B02": np.full((8, 8), 0.5, dtype=np.float32)}

        tensor = bands_to_tensor(bands, ["B02"], device="cpu")

        assert tensor.shape == (1, 1, 8, 8)
        assert abs(tensor.mean().item() - 0.5) < 1e-5
