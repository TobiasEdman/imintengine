"""Tests for Dense CRF post-processing."""
import numpy as np
import pytest

from imint.inference.crf_postprocess import is_crf_available


class TestCRFAvailability:

    def test_is_crf_available_returns_bool(self):
        result = is_crf_available()
        assert isinstance(result, bool)


@pytest.mark.skipif(not is_crf_available(), reason="pydensecrf not installed")
class TestDenseCRF:

    def test_output_shape(self):
        from imint.inference.crf_postprocess import apply_dense_crf
        C, H, W = 5, 64, 64
        probs = np.random.dirichlet(np.ones(C), size=(H * W)).T.reshape(C, H, W).astype(np.float32)
        rgb = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
        result = apply_dense_crf(probs, rgb, n_iters=2)
        assert result.shape == (H, W)
        assert result.dtype == np.int32

    def test_confident_predictions_unchanged(self):
        """Near-one-hot softmax should not change after CRF."""
        from imint.inference.crf_postprocess import apply_dense_crf
        C, H, W = 5, 32, 32
        # All pixels confident class 2
        probs = np.full((C, H, W), 0.01, dtype=np.float32)
        probs[2, :, :] = 0.96
        rgb = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
        result = apply_dense_crf(probs, rgb, n_iters=5)
        # Vast majority should still be class 2
        assert (result == 2).mean() > 0.9

    def test_class_indices_in_range(self):
        from imint.inference.crf_postprocess import apply_dense_crf
        C, H, W = 23, 64, 64
        probs = np.random.dirichlet(np.ones(C), size=(H * W)).T.reshape(C, H, W).astype(np.float32)
        rgb = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
        result = apply_dense_crf(probs, rgb, n_iters=3)
        assert result.min() >= 0
        assert result.max() < C


class TestCRFGracefulFallback:

    def test_import_error_message(self):
        """If pydensecrf not installed, should give helpful error message."""
        if is_crf_available():
            pytest.skip("pydensecrf is installed")
        from imint.inference.crf_postprocess import apply_dense_crf
        probs = np.zeros((5, 32, 32), dtype=np.float32)
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        with pytest.raises(ImportError, match="pydensecrf"):
            apply_dense_crf(probs, rgb)
