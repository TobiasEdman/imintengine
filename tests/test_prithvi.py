"""Tests for imint/analyzers/prithvi.py — Prithvi-EO-2.0 foundation model analyzer."""
from __future__ import annotations

import os
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from imint.analyzers.prithvi import PrithviAnalyzer, PRITHVI_BANDS, PRITHVI_BACKBONE


# ── Band validation ──────────────────────────────────────────────────────────

class TestPrithviBandValidation:
    """Verify that Prithvi correctly validates required bands."""

    def _make_prithvi_bands(self, h=32, w=32):
        """Create a complete set of Prithvi-required bands."""
        return {
            "B02": np.full((h, w), 0.1, dtype=np.float32),
            "B03": np.full((h, w), 0.2, dtype=np.float32),
            "B04": np.full((h, w), 0.3, dtype=np.float32),
            "B8A": np.full((h, w), 0.45, dtype=np.float32),
            "B11": np.full((h, w), 0.15, dtype=np.float32),
            "B12": np.full((h, w), 0.12, dtype=np.float32),
        }

    def test_required_bands(self):
        """Prithvi requires exactly these 6 bands."""
        assert PRITHVI_BANDS == ["B02", "B03", "B04", "B8A", "B11", "B12"]

    def test_b8a_not_b08(self):
        """Prithvi requires B8A (865nm, 20m narrow NIR), NOT B08 (842nm, 10m)."""
        assert "B8A" in PRITHVI_BANDS
        assert "B08" not in PRITHVI_BANDS

    def test_backbone_name(self):
        """Default backbone should be prithvi_eo_v2_300m_tl."""
        assert PRITHVI_BACKBONE == "prithvi_eo_v2_300m_tl"

    @patch("imint.fm.terratorch_loader.check_torch_available", return_value=True)
    @patch("imint.fm.terratorch_loader.check_terratorch_available", return_value=True)
    def test_missing_bands_returns_error(self, mock_tt, mock_torch):
        """Should fail gracefully when required bands are missing."""
        analyzer = PrithviAnalyzer(config={})
        # Only provide B02 and B03 — missing B04, B8A, B11, B12
        bands = {
            "B02": np.zeros((32, 32), dtype=np.float32),
            "B03": np.zeros((32, 32), dtype=np.float32),
        }
        rgb = np.zeros((32, 32, 3), dtype=np.float32)
        result = analyzer.run(rgb, bands=bands)

        assert not result.success
        assert "Missing required bands" in result.error
        assert "B04" in result.error
        assert "B8A" in result.error

    @patch("imint.fm.terratorch_loader.check_torch_available", return_value=True)
    @patch("imint.fm.terratorch_loader.check_terratorch_available", return_value=True)
    def test_b08_is_not_b8a(self, mock_tt, mock_torch):
        """Having B08 should NOT satisfy the B8A requirement."""
        analyzer = PrithviAnalyzer(config={})
        bands = {
            "B02": np.zeros((32, 32), dtype=np.float32),
            "B03": np.zeros((32, 32), dtype=np.float32),
            "B04": np.zeros((32, 32), dtype=np.float32),
            "B08": np.zeros((32, 32), dtype=np.float32),  # NOT B8A!
            "B11": np.zeros((32, 32), dtype=np.float32),
            "B12": np.zeros((32, 32), dtype=np.float32),
        }
        rgb = np.zeros((32, 32, 3), dtype=np.float32)
        result = analyzer.run(rgb, bands=bands)

        assert not result.success
        assert "B8A" in result.error
        assert "B08" in result.error  # Should mention the difference

    @patch("imint.fm.terratorch_loader.check_torch_available", return_value=True)
    @patch("imint.fm.terratorch_loader.check_terratorch_available", return_value=True)
    def test_no_bands_returns_error(self, mock_tt, mock_torch):
        """Should fail when no bands are provided."""
        analyzer = PrithviAnalyzer(config={})
        rgb = np.zeros((32, 32, 3), dtype=np.float32)
        result = analyzer.run(rgb, bands=None)

        assert not result.success
        assert "Missing required bands" in result.error


# ── Without TerraTorch or Torch ─────────────────────────────────────────────

class TestPrithviWithoutDependencies:
    """Verify graceful failure when neither terratorch nor torch is installed."""

    def test_fails_with_install_instruction(self):
        """Should return success=False with install instructions."""
        with patch("imint.fm.terratorch_loader.check_terratorch_available", return_value=False), \
             patch("imint.fm.terratorch_loader.check_torch_available", return_value=False):
            analyzer = PrithviAnalyzer(config={})
            rgb = np.zeros((32, 32, 3), dtype=np.float32)
            bands = {b: np.zeros((32, 32), dtype=np.float32) for b in PRITHVI_BANDS}
            result = analyzer.run(rgb, bands=bands)

            assert not result.success
            assert "terratorch" in result.error.lower() or "torch" in result.error.lower()
            assert "pip install" in result.error

    def test_works_with_torch_only(self):
        """Should pass availability check with just torch (no terratorch)."""
        with patch("imint.fm.terratorch_loader.check_terratorch_available", return_value=False), \
             patch("imint.fm.terratorch_loader.check_torch_available", return_value=True):
            analyzer = PrithviAnalyzer(config={})
            rgb = np.zeros((32, 32, 3), dtype=np.float32)
            # Missing bands — should fail on band check, not availability
            result = analyzer.run(rgb, bands=None)

            assert not result.success
            assert "Missing required bands" in result.error

    def test_analyzer_name(self):
        """Analyzer should be named 'prithvi'."""
        analyzer = PrithviAnalyzer(config={})
        assert analyzer.name == "prithvi"


# ── Embeddings mode (mocked) ────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestPrithviEmbeddings:
    """Test embedding extraction with mocked model."""

    def _make_prithvi_bands(self, h=32, w=32):
        return {b: np.random.rand(h, w).astype(np.float32) for b in PRITHVI_BANDS}

    @patch("imint.fm.terratorch_loader.check_torch_available", return_value=True)
    @patch("imint.fm.terratorch_loader.check_terratorch_available", return_value=True)
    def test_embeddings_output_structure(self, mock_tt, mock_torch):
        """Embeddings mode should return embedding array and stats."""
        import torch

        # Mock the extract_encoder_features to return a known feature map
        fake_features = np.random.randn(1, 1024, 7, 7).astype(np.float32)

        # Mock the model backbone
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        # Mock get_prithvi_config
        mock_config = {
            "mean": [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0],
            "std": [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0],
        }

        with patch("imint.fm.terratorch_loader.load_backbone", return_value=mock_model), \
             patch("imint.fm.terratorch_loader.get_device", return_value="cpu"), \
             patch("imint.fm.terratorch_loader.get_prithvi_config", return_value=mock_config), \
             patch("imint.fm.terratorch_loader.extract_encoder_features", return_value=fake_features):
            analyzer = PrithviAnalyzer(config={"mode": "embeddings"})
            rgb = np.zeros((32, 32, 3), dtype=np.float32)
            bands = self._make_prithvi_bands()
            result = analyzer.run(rgb, bands=bands)

        assert result.success
        assert result.analyzer == "prithvi"
        assert "embedding" in result.outputs
        assert "stats" in result.outputs

        # Check embedding shape
        embedding = result.outputs["embedding"]
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1, 1024, 7, 7)

        # Check stats
        stats = result.outputs["stats"]
        assert "embedding_shape" in stats
        assert "mean" in stats
        assert "std" in stats
        assert "backbone" in stats
        assert "n_patches" in stats

        # Check metadata
        assert result.metadata["mode"] == "embeddings"
        assert result.metadata["bands_used"] == PRITHVI_BANDS


# ── Segmentation mode ───────────────────────────────────────────────────────

class TestPrithviSegmentationConfig:
    """Test segmentation config validation (no torch required)."""

    @patch("imint.fm.terratorch_loader.check_torch_available", return_value=True)
    @patch("imint.fm.terratorch_loader.check_terratorch_available", return_value=True)
    def test_requires_task_head_or_model_path(self, mock_tt, mock_torch):
        """Segmentation mode should require task_head or model_path in config."""
        analyzer = PrithviAnalyzer(config={"mode": "segmentation"})
        rgb = np.zeros((32, 32, 3), dtype=np.float32)
        bands = {b: np.zeros((32, 32), dtype=np.float32) for b in PRITHVI_BANDS}
        result = analyzer.run(rgb, bands=bands)

        assert not result.success
        assert "task_head" in result.error or "model_path" in result.error

    @patch("imint.fm.terratorch_loader.check_torch_available", return_value=True)
    @patch("imint.fm.terratorch_loader.check_terratorch_available", return_value=True)
    def test_error_lists_available_task_heads(self, mock_tt, mock_torch):
        """Error message should list available task heads."""
        analyzer = PrithviAnalyzer(config={"mode": "segmentation"})
        rgb = np.zeros((32, 32, 3), dtype=np.float32)
        bands = {b: np.zeros((32, 32), dtype=np.float32) for b in PRITHVI_BANDS}
        result = analyzer.run(rgb, bands=bands)

        assert "sen1floods11" in result.error
        assert "burn_scars" in result.error


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestPrithviSegmentation:
    """Test segmentation inference with mocked model."""

    @patch("imint.fm.terratorch_loader.check_torch_available", return_value=True)
    @patch("imint.fm.terratorch_loader.check_terratorch_available", return_value=True)
    def test_segmentation_with_model_path(self, mock_tt, mock_torch_check):
        """Segmentation via model_path should return seg_mask and class_stats."""
        import torch

        # Mock segmentation model — must return logits matching patch_size (224)
        def mock_forward(tensor):
            b, c, h, w = tensor.shape
            return torch.randn(b, 5, h, w)

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.side_effect = mock_forward
        mock_model.return_value = None  # Overridden by side_effect

        mock_config = {
            "mean": [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0],
            "std": [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0],
        }

        with patch("imint.fm.terratorch_loader.get_device", return_value="cpu"), \
             patch("imint.fm.terratorch_loader.get_prithvi_config", return_value=mock_config), \
             patch("torch.load", return_value=mock_model):
            analyzer = PrithviAnalyzer(config={
                "mode": "segmentation",
                "model_path": "/fake/model.pt",
            })
            rgb = np.zeros((32, 32, 3), dtype=np.float32)
            bands = {b: np.random.rand(32, 32).astype(np.float32) for b in PRITHVI_BANDS}
            result = analyzer.run(rgb, bands=bands)

        assert result.success
        assert "seg_mask" in result.outputs
        assert "class_stats" in result.outputs
        assert "n_classes" in result.outputs

        seg_mask = result.outputs["seg_mask"]
        assert isinstance(seg_mask, np.ndarray)
        assert seg_mask.shape == (32, 32)
        assert seg_mask.dtype == np.uint8

        # Class stats should sum to 1.0
        total_fraction = sum(v["fraction"] for v in result.outputs["class_stats"].values())
        assert abs(total_fraction - 1.0) < 0.01

        assert result.metadata["mode"] == "segmentation"

    @patch("imint.fm.terratorch_loader.check_torch_available", return_value=True)
    @patch("imint.fm.terratorch_loader.check_terratorch_available", return_value=True)
    def test_segmentation_with_task_head(self, mock_tt, mock_torch_check):
        """Segmentation via task_head should return seg_mask with class names."""
        import torch

        # Mock segmentation model — dynamic output matching input size
        def mock_forward(tensor):
            b, c, h, w = tensor.shape
            return torch.randn(b, 2, h, w)

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.side_effect = mock_forward

        mock_config = {
            "mean": [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0],
            "std": [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0],
        }

        task_config = {
            "num_classes": 2,
            "class_names": {0: "no_water", 1: "water/flood"},
            "description": "Flood segmentation (Sen1Floods11 dataset)",
        }

        with patch("imint.fm.terratorch_loader.get_device", return_value="cpu"), \
             patch("imint.fm.terratorch_loader.get_prithvi_config", return_value=mock_config), \
             patch("imint.fm.terratorch_loader.load_segmentation_model",
                   return_value=(mock_model, task_config)):
            analyzer = PrithviAnalyzer(config={
                "mode": "segmentation",
                "task_head": "sen1floods11",
            })
            rgb = np.zeros((32, 32, 3), dtype=np.float32)
            bands = {b: np.random.rand(32, 32).astype(np.float32) for b in PRITHVI_BANDS}
            result = analyzer.run(rgb, bands=bands)

        assert result.success
        assert result.metadata["task_head"] == "sen1floods11"
        assert result.metadata["description"] == "Flood segmentation (Sen1Floods11 dataset)"

        # Class stats should contain class names
        for cls_info in result.outputs["class_stats"].values():
            assert "name" in cls_info


# ── Task head registry ───────────────────────────────────────────────────────

class TestTaskHeadRegistry:
    """Test the task head registry and model loading infrastructure."""

    def test_registry_contains_sen1floods11(self):
        """Sen1Floods11 should be in the task head registry."""
        from imint.fm.terratorch_loader import TASK_HEAD_REGISTRY
        assert "sen1floods11" in TASK_HEAD_REGISTRY
        entry = TASK_HEAD_REGISTRY["sen1floods11"]
        assert entry["num_classes"] == 2
        assert entry["class_names"] == {0: "no_water", 1: "water/flood"}

    def test_registry_contains_burn_scars(self):
        """BurnScars should be in the task head registry."""
        from imint.fm.terratorch_loader import TASK_HEAD_REGISTRY
        assert "burn_scars" in TASK_HEAD_REGISTRY
        entry = TASK_HEAD_REGISTRY["burn_scars"]
        assert entry["num_classes"] == 2
        assert entry["class_names"] == {0: "no_burn", 1: "burned"}

    def test_list_task_heads(self):
        """list_task_heads should return a copy of the registry."""
        from imint.fm.terratorch_loader import list_task_heads, TASK_HEAD_REGISTRY
        heads = list_task_heads()
        assert heads == TASK_HEAD_REGISTRY
        # Should be a copy, not a reference
        heads["test"] = {}
        assert "test" not in TASK_HEAD_REGISTRY

    def test_all_entries_have_required_fields(self):
        """Every registry entry must have all required fields."""
        from imint.fm.terratorch_loader import TASK_HEAD_REGISTRY
        required_fields = [
            "repo_id", "filename", "num_classes", "feature_indices",
            "dropout", "class_names", "description",
        ]
        for name, entry in TASK_HEAD_REGISTRY.items():
            for field in required_fields:
                assert field in entry, f"Task head '{name}' missing field '{field}'"
            # Must have either decoder_channels (UPerNet) or decoder_type (UNet)
            has_decoder_config = "decoder_channels" in entry or "decoder_type" in entry
            assert has_decoder_config, f"Task head '{name}' needs decoder_channels or decoder_type"

    def test_unknown_task_head_raises(self):
        """load_segmentation_model should raise ValueError for unknown heads."""
        from imint.fm.terratorch_loader import load_segmentation_model
        with pytest.raises(ValueError, match="Unknown task head"):
            load_segmentation_model("nonexistent_model")


# ── UperNet architecture ─────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestUperNetArchitecture:
    """Test the UperNet decoder module."""

    def test_conv_bn_relu(self):
        """ConvBnRelu should produce correct output shape."""
        import torch
        from imint.fm.upernet import ConvBnRelu

        block = ConvBnRelu(64, 32, kernel=3, padding=1)
        x = torch.randn(1, 64, 16, 16)
        out = block(x)
        assert out.shape == (1, 32, 16, 16)

    def test_segmentation_head(self):
        """SegmentationHead should output (B, num_classes, H, W)."""
        import torch
        from imint.fm.upernet import SegmentationHead

        head = SegmentationHead(in_channels=256, num_classes=5, dropout=0.1)
        x = torch.randn(1, 256, 14, 14)
        head.eval()
        out = head(x)
        assert out.shape == (1, 5, 14, 14)

    def test_segmentation_model_forward(self):
        """PrithviSegmentationModel should produce correct output shape."""
        import torch
        from imint.fm.upernet import PrithviSegmentationModel
        from imint.fm.terratorch_loader import _load_prithvi_from_hf

        backbone = _load_prithvi_from_hf(pretrained=False)
        model = PrithviSegmentationModel(
            encoder=backbone,
            feature_indices=[5, 11, 17, 23],
            decoder_channels=256,
            num_classes=3,
        )
        model.eval()

        x = torch.randn(1, 6, 224, 224)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (1, 3, 224, 224)

    def test_checkpoint_key_mapping(self):
        """_map_checkpoint_keys should strip 'model.' prefix correctly."""
        from imint.fm.terratorch_loader import _map_checkpoint_keys
        import torch
        from imint.fm.upernet import PrithviSegmentationModel
        from imint.fm.terratorch_loader import _load_prithvi_from_hf

        backbone = _load_prithvi_from_hf(pretrained=False)
        model = PrithviSegmentationModel(
            encoder=backbone,
            feature_indices=[5, 11, 17, 23],
            decoder_channels=256,
            num_classes=2,
        )

        # Simulate a Lightning checkpoint with 'model.' prefix
        fake_sd = {}
        for k, v in model.state_dict().items():
            fake_sd[f"model.{k}"] = v

        mapped = _map_checkpoint_keys(fake_sd, model)

        # All model keys should be mapped
        model_keys = set(model.state_dict().keys())
        mapped_keys = set(mapped.keys())
        assert model_keys == mapped_keys


# ── Engine integration ───────────────────────────────────────────────────────

class TestPrithviEngineRegistration:
    """Verify Prithvi is registered in the engine."""

    def test_in_analyzer_registry(self):
        """PrithviAnalyzer should be in the engine's registry."""
        from imint.engine import ANALYZER_REGISTRY
        assert "prithvi" in ANALYZER_REGISTRY
        assert ANALYZER_REGISTRY["prithvi"] is PrithviAnalyzer

    def test_registry_order(self):
        """Prithvi should come before NMD in the registry."""
        from imint.engine import ANALYZER_REGISTRY
        keys = list(ANALYZER_REGISTRY.keys())
        assert keys.index("prithvi") < keys.index("nmd")

    def test_disabled_by_default(self):
        """Prithvi should be disabled in default config."""
        import yaml
        with open("config/analyzers.yaml") as f:
            config = yaml.safe_load(f)
        assert config["prithvi"]["enabled"] is False


# ── Embedding visualization ──────────────────────────────────────────────────

class TestPrithviEmbeddingVisualization:
    """Test embedding visualization export."""

    def test_save_creates_png(self, tmp_output_dir):
        """Should create a valid PNG file with the 3-panel visualization."""
        from imint.exporters.export import save_prithvi_embedding_viz

        embedding = np.random.randn(1, 1024, 7, 7).astype(np.float32)
        rgb = np.random.rand(224, 224, 3).astype(np.float32)
        path = os.path.join(tmp_output_dir, "test_embedding.png")

        result = save_prithvi_embedding_viz(embedding, rgb, path)

        assert result == path
        assert os.path.exists(path)
        # Verify it is a valid image
        from PIL import Image
        img = Image.open(path)
        assert img.format == "PNG"
        # Multi-panel figure should be wider than tall
        assert img.width > img.height

    def test_uniform_embedding(self, tmp_output_dir):
        """Should handle uniform (zero-variance) embedding gracefully."""
        from imint.exporters.export import save_prithvi_embedding_viz

        embedding = np.ones((1, 1024, 7, 7), dtype=np.float32)
        rgb = np.full((64, 64, 3), 0.5, dtype=np.float32)
        path = os.path.join(tmp_output_dir, "test_uniform.png")

        result = save_prithvi_embedding_viz(embedding, rgb, path)

        assert result == path
        assert os.path.exists(path)

    def test_nonsquare_image(self, tmp_output_dir):
        """Should handle non-square and various image sizes."""
        from imint.exporters.export import save_prithvi_embedding_viz

        embedding = np.random.randn(1, 1024, 7, 7).astype(np.float32)
        rgb = np.random.rand(128, 256, 3).astype(np.float32)
        path = os.path.join(tmp_output_dir, "test_nonsquare.png")

        result = save_prithvi_embedding_viz(embedding, rgb, path)

        assert result == path
        assert os.path.exists(path)

    def test_pca_normalization(self):
        """PCA channel values should be in [0, 1] after normalization."""
        from imint.exporters.export import _pca_feature_map

        features = np.random.randn(49, 1024).astype(np.float32)
        pca_rgb = _pca_feature_map(features, 7, 7)

        assert pca_rgb.shape == (7, 7, 3)
        assert pca_rgb.min() >= 0.0
        assert pca_rgb.max() <= 1.0

    def test_activation_magnitude(self):
        """L2-norm should produce correct spatial grid with non-negative values."""
        from imint.exporters.export import _activation_magnitude

        emb_3d = np.random.randn(1024, 7, 7).astype(np.float32)
        mag = _activation_magnitude(emb_3d)

        assert mag.shape == (7, 7)
        assert (mag >= 0).all()


# ── Segmentation overlay visualization ────────────────────────────────────

class TestPrithviSegmentationOverlay:
    """Test the enhanced save_prithvi_overlay function."""

    def test_basic_overlay_backward_compatible(self, tmp_output_dir):
        """Original 2-arg call should still work (backward compatibility)."""
        from imint.exporters.export import save_prithvi_overlay

        seg_mask = np.zeros((32, 32), dtype=np.uint8)
        seg_mask[:16, :] = 0
        seg_mask[16:, :] = 1
        path = os.path.join(tmp_output_dir, "test_seg_basic.png")

        result = save_prithvi_overlay(seg_mask, path)

        assert result == path
        assert os.path.exists(path)
        from PIL import Image
        img = Image.open(path)
        assert img.format == "PNG"

    def test_overlay_with_class_names(self, tmp_output_dir):
        """Overlay with class_names should create a legend instead of colorbar."""
        from imint.exporters.export import save_prithvi_overlay

        seg_mask = np.zeros((32, 32), dtype=np.uint8)
        seg_mask[16:, :] = 1
        class_names = {0: "no_water", 1: "water/flood"}
        path = os.path.join(tmp_output_dir, "test_seg_names.png")

        result = save_prithvi_overlay(seg_mask, path, class_names=class_names)

        assert result == path
        assert os.path.exists(path)

    def test_overlay_with_rgb_side_by_side(self, tmp_output_dir):
        """Providing rgb should create a wider 2-panel figure."""
        from imint.exporters.export import save_prithvi_overlay

        seg_mask = np.zeros((32, 32), dtype=np.uint8)
        seg_mask[16:, :] = 1
        rgb = np.random.rand(32, 32, 3).astype(np.float32)
        path = os.path.join(tmp_output_dir, "test_seg_sidebyside.png")

        result = save_prithvi_overlay(seg_mask, path, rgb=rgb)

        assert result == path
        assert os.path.exists(path)
        from PIL import Image
        img = Image.open(path)
        # 2-panel figure should be wider than single panel
        assert img.width > img.height

    def test_overlay_with_rgb_and_class_names(self, tmp_output_dir):
        """Full call with both rgb and class_names."""
        from imint.exporters.export import save_prithvi_overlay

        seg_mask = np.zeros((64, 64), dtype=np.uint8)
        seg_mask[:32, :] = 0
        seg_mask[32:, :] = 1
        rgb = np.random.rand(64, 64, 3).astype(np.float32)
        class_names = {0: "no_burn", 1: "burned"}
        path = os.path.join(tmp_output_dir, "test_seg_full.png")

        result = save_prithvi_overlay(
            seg_mask, path, rgb=rgb, class_names=class_names,
        )

        assert result == path
        assert os.path.exists(path)
        from PIL import Image
        img = Image.open(path)
        assert img.width > img.height

    def test_overlay_single_class(self, tmp_output_dir):
        """Should handle edge case of all-zeros mask (single class)."""
        from imint.exporters.export import save_prithvi_overlay

        seg_mask = np.zeros((16, 16), dtype=np.uint8)
        class_names = {0: "background"}
        path = os.path.join(tmp_output_dir, "test_seg_single.png")

        result = save_prithvi_overlay(seg_mask, path, class_names=class_names)

        assert result == path
        assert os.path.exists(path)


# ── Engine segmentation export ────────────────────────────────────────────

class TestEngineSegmentationExport:
    """Test that _export correctly threads class_names to save_prithvi_overlay."""

    def test_export_passes_class_names(self, tmp_output_dir):
        """_export should extract class_names from class_stats and pass to overlay."""
        from imint.engine import _export
        from imint.analyzers.base import AnalysisResult
        from imint.job import IMINTJob, GeoContext
        from rasterio.transform import from_bounds

        seg_mask = np.zeros((32, 32), dtype=np.uint8)
        seg_mask[16:, :] = 1

        result = AnalysisResult(
            analyzer="prithvi",
            success=True,
            outputs={
                "seg_mask": seg_mask,
                "class_stats": {
                    0: {"pixel_count": 512, "fraction": 0.5, "name": "no_water"},
                    1: {"pixel_count": 512, "fraction": 0.5, "name": "water/flood"},
                },
            },
            metadata={"mode": "segmentation"},
        )

        rgb = np.random.rand(32, 32, 3).astype(np.float32)
        geo = GeoContext(
            crs="EPSG:3006",
            transform=from_bounds(470000, 6240000, 470320, 6240320, 32, 32),
            bounds_projected={"west": 470000, "south": 6240000,
                              "east": 470320, "north": 6240320},
            bounds_wgs84={"west": 14.5, "south": 56.0,
                          "east": 15.5, "north": 57.0},
            shape=(32, 32),
        )

        job = IMINTJob(
            date="2022-06-15",
            coords={"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0},
            rgb=rgb,
            geo=geo,
            output_dir=tmp_output_dir,
        )

        with patch("imint.engine.save_prithvi_overlay") as mock_overlay, \
             patch("imint.engine.save_geotiff"):
            _export(result, job)

            mock_overlay.assert_called_once()
            call_kwargs = mock_overlay.call_args
            # Verify class_names were passed
            assert call_kwargs.kwargs.get("class_names") == {
                0: "no_water", 1: "water/flood",
            }
            # Verify rgb was passed
            assert call_kwargs.kwargs.get("rgb") is rgb
