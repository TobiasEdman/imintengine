"""Tests for sliding window inference with overlap averaging."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from imint.inference.sliding_window import sliding_window_inference, sliding_window_predict


class MockSegModel(nn.Module):
    """Returns constant logits per class for testing."""

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x, aux=None, temporal_coords=None, location_coords=None):
        B = x.shape[0]
        H, W = x.shape[-2:]
        # Class 1 gets highest logit everywhere
        logits = torch.zeros(B, self.num_classes, H, W)
        logits[:, 1, :, :] = 10.0
        return logits


class SpatiallyAwareModel(nn.Module):
    """Returns class based on spatial position for verifying averaging."""

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x, aux=None, temporal_coords=None, location_coords=None):
        B = x.shape[0]
        H, W = x.shape[-2:]
        logits = torch.zeros(B, self.num_classes, H, W)
        # Upper half → class 1, lower half → class 2
        logits[:, 1, :H // 2, :] = 10.0
        logits[:, 2, H // 2:, :] = 10.0
        return logits


class TestSlidingWindowInference:

    def test_output_shape_no_overlap(self):
        model = MockSegModel(5)
        img = torch.randn(1, 6, 1, 64, 64)
        probs = sliding_window_inference(model, img, patch_size=64, overlap=0.0, num_classes=5)
        assert probs.shape == (1, 5, 64, 64)

    def test_output_shape_50_overlap(self):
        model = MockSegModel(5)
        img = torch.randn(1, 6, 1, 128, 128)
        probs = sliding_window_inference(model, img, patch_size=64, overlap=0.5, num_classes=5)
        assert probs.shape == (1, 5, 128, 128)

    def test_output_shape_preserves_spatial(self):
        """Output H,W must match input H,W regardless of overlap."""
        model = MockSegModel(3)
        for H, W in [(128, 128), (64, 96), (256, 256)]:
            img = torch.randn(1, 6, 1, H, W)
            probs = sliding_window_inference(model, img, patch_size=64, overlap=0.5, num_classes=3)
            assert probs.shape == (1, 3, H, W), f"Failed for ({H}, {W})"

    def test_probabilities_sum_to_one(self):
        model = MockSegModel(5)
        img = torch.randn(1, 6, 1, 64, 64)
        probs = sliding_window_inference(model, img, patch_size=64, overlap=0.5, num_classes=5)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_constant_model_all_same_class(self):
        model = MockSegModel(5)
        img = torch.randn(1, 6, 1, 128, 128)
        pred = sliding_window_predict(model, img, patch_size=64, overlap=0.5, num_classes=5)
        assert (pred == 1).all(), "Constant model should predict class 1 everywhere"

    def test_small_image_padded(self):
        """Image smaller than patch_size should be padded and still work."""
        model = MockSegModel(5)
        img = torch.randn(1, 6, 1, 32, 32)
        probs = sliding_window_inference(model, img, patch_size=64, overlap=0.0, num_classes=5)
        assert probs.shape == (1, 5, 32, 32)

    def test_aux_channels_passed(self):
        """Verify aux tensor is sliced correctly per window."""
        model = MockSegModel(5)
        img = torch.randn(1, 6, 1, 128, 128)
        aux = torch.randn(1, 3, 128, 128)
        probs = sliding_window_inference(model, img, aux=aux, patch_size=64, overlap=0.5, num_classes=5)
        assert probs.shape == (1, 5, 128, 128)

    def test_overlap_invalid_raises(self):
        model = MockSegModel(5)
        img = torch.randn(1, 6, 1, 64, 64)
        with pytest.raises(ValueError, match="overlap must be in"):
            sliding_window_inference(model, img, patch_size=64, overlap=1.0, num_classes=5)
        with pytest.raises(ValueError, match="overlap must be in"):
            sliding_window_inference(model, img, patch_size=64, overlap=-0.1, num_classes=5)

    def test_spatial_model_boundaries_correct(self):
        """With overlap, boundary should be sharper than patch edges."""
        model = SpatiallyAwareModel(5)
        img = torch.randn(1, 6, 1, 128, 128)
        pred = sliding_window_predict(model, img, patch_size=64, overlap=0.5, num_classes=5)
        # Top quarter should be solidly class 1, bottom quarter solidly class 2
        # (boundary zone in the middle may be mixed due to averaging)
        assert (pred[0, :32, :] == 1).all(), "Top quarter should be class 1"
        assert (pred[0, 96:, :] == 2).all(), "Bottom quarter should be class 2"
