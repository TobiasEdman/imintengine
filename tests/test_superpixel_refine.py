"""Tests for superpixel refinement and morphological cleanup."""
import numpy as np
import pytest

from imint.inference.superpixel_refine import (
    superpixel_refine,
    morphological_cleanup,
)


def _make_two_class_tile(h=64, w=64):
    """Create a tile with two spectrally distinct halves.

    Left half: bright (class 1), right half: dark (class 2).
    Returns spectral (6, H, W) and ground truth (H, W).
    """
    spectral = np.zeros((6, h, w), dtype=np.float32)
    spectral[:, :, :w // 2] = 0.8  # bright left
    spectral[:, :, w // 2:] = 0.2  # dark right
    gt = np.zeros((h, w), dtype=np.uint8)
    gt[:, :w // 2] = 1
    gt[:, w // 2:] = 2
    return spectral, gt


def _make_softmax(gt, n_classes=5, confidence=0.8, noise=0.05):
    """Create noisy softmax probs from ground truth."""
    H, W = gt.shape
    probs = np.full((n_classes, H, W), noise / (n_classes - 1), dtype=np.float32)
    for c in range(n_classes):
        probs[c][gt == c] = confidence
    # Normalize
    probs /= probs.sum(axis=0, keepdims=True)
    return probs


class TestSuperpixelRefine:

    def test_output_shape(self):
        spectral, gt = _make_two_class_tile()
        probs = _make_softmax(gt)
        result = superpixel_refine(probs, spectral, n_segments=50)
        assert result.shape == (64, 64)
        assert result.dtype == np.uint8

    def test_preserves_clear_boundary(self):
        """Superpixel refinement should preserve a clear spectral boundary."""
        spectral, gt = _make_two_class_tile()
        probs = _make_softmax(gt, confidence=0.9)
        result = superpixel_refine(probs, spectral, n_segments=50)
        # Left half should be class 1, right half class 2
        assert (result[:, :28] == 1).mean() > 0.9
        assert (result[:, 36:] == 2).mean() > 0.9

    def test_cleans_noisy_predictions(self):
        """Noisy per-pixel predictions should be smoothed within superpixels."""
        spectral, gt = _make_two_class_tile(128, 128)
        probs = _make_softmax(gt, confidence=0.6, noise=0.15)
        # Add random pixel noise
        noise_mask = np.random.rand(128, 128) < 0.2
        pixel_pred = probs.argmax(axis=0)
        noisy_pred = pixel_pred.copy()
        noisy_pred[noise_mask] = np.random.randint(0, 5, noise_mask.sum())

        # Superpixel refinement should recover cleaner result
        result = superpixel_refine(probs, spectral, n_segments=200)
        # Should be closer to GT than noisy prediction
        acc_noisy = (noisy_pred == gt).mean()
        acc_refined = (result == gt).mean()
        assert acc_refined > acc_noisy, (
            f"Refined ({acc_refined:.3f}) should be better than noisy ({acc_noisy:.3f})"
        )

    @pytest.mark.parametrize("method", ["slic", "felzenszwalb", "watershed"])
    def test_all_methods_work(self, method):
        spectral, gt = _make_two_class_tile()
        probs = _make_softmax(gt)
        result = superpixel_refine(probs, spectral, method=method, n_segments=50)
        assert result.shape == (64, 64)

    @pytest.mark.parametrize("agg", ["mean_prob", "majority_vote", "weighted_mean"])
    def test_all_aggregations_work(self, agg):
        spectral, gt = _make_two_class_tile()
        probs = _make_softmax(gt)
        result = superpixel_refine(probs, spectral, aggregation=agg, n_segments=50)
        assert result.shape == (64, 64)

    def test_unknown_method_raises(self):
        probs = np.zeros((5, 32, 32), dtype=np.float32)
        spec = np.zeros((6, 32, 32), dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown method"):
            superpixel_refine(probs, spec, method="bogus")

    def test_unknown_aggregation_raises(self):
        probs = np.zeros((5, 32, 32), dtype=np.float32)
        spec = np.zeros((6, 32, 32), dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown aggregation"):
            superpixel_refine(probs, spec, aggregation="bogus")

    def test_six_band_spectral(self):
        """Should work with 6-band Sentinel-2 input, not just RGB."""
        spectral = np.random.rand(6, 64, 64).astype(np.float32)
        probs = np.random.dirichlet(np.ones(5), size=(64 * 64)).T.reshape(5, 64, 64).astype(np.float32)
        result = superpixel_refine(probs, spectral, n_segments=100)
        assert result.shape == (64, 64)

    def test_with_aux_channels(self):
        """Aux channels (DEM, height, VPP) should be used for superpixel generation."""
        spectral, gt = _make_two_class_tile()
        # Aux channels that reinforce the spectral boundary
        aux = np.zeros((3, 64, 64), dtype=np.float32)
        aux[0, :, :32] = 100.0  # DEM high on left
        aux[1, :, :32] = 15.0   # tree height on left
        probs = _make_softmax(gt, confidence=0.7)
        result = superpixel_refine(probs, spectral, aux=aux, n_segments=50)
        assert result.shape == (64, 64)
        assert (result[:, :28] == 1).mean() > 0.85


class TestMorphologicalCleanup:

    def test_removes_small_components(self):
        """Small isolated patches should be replaced by neighbor class."""
        pred = np.ones((64, 64), dtype=np.uint8)  # all class 1
        pred[30:33, 30:33] = 2  # 9 pixels of class 2 (< 25 min_pixels)
        cleaned = morphological_cleanup(pred, min_pixels=25)
        assert (cleaned == 1).all(), "Small component should be removed"

    def test_keeps_large_components(self):
        """Components above MMU should be preserved."""
        pred = np.ones((64, 64), dtype=np.uint8)
        pred[10:20, 10:20] = 2  # 100 pixels (> 25 min_pixels)
        cleaned = morphological_cleanup(pred, min_pixels=25)
        assert (cleaned[10:20, 10:20] == 2).all(), "Large component should stay"

    def test_skips_background(self):
        """Background (class 0) should not be cleaned."""
        pred = np.zeros((32, 32), dtype=np.uint8)
        pred[5:8, 5:8] = 0  # small background patch
        cleaned = morphological_cleanup(pred, min_pixels=25)
        assert (cleaned == pred).all()

    def test_output_shape(self):
        pred = np.random.randint(0, 5, (64, 64), dtype=np.uint8)
        cleaned = morphological_cleanup(pred)
        assert cleaned.shape == (64, 64)
        assert cleaned.dtype == np.uint8
