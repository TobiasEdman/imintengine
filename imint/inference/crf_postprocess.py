"""
imint/inference/crf_postprocess.py — Dense CRF post-processing

Refines segmentation predictions by enforcing spatial consistency using
both unary potentials (model softmax) and pairwise potentials (spatial
and spectral proximity via bilateral filtering).

Requires: pip install pydensecrf
Falls back gracefully if not installed.
"""
from __future__ import annotations

import numpy as np


def apply_dense_crf(
    softmax_probs: np.ndarray,
    image_rgb: np.ndarray,
    n_iters: int = 5,
    sxy_gaussian: int = 3,
    compat_gaussian: float = 3.0,
    sxy_bilateral: int = 60,
    srgb_bilateral: int = 10,
    compat_bilateral: float = 10.0,
) -> np.ndarray:
    """Refine segmentation with fully-connected CRF.

    The bilateral kernel uses the RGB reference image to snap predictions
    to spectral edges — pixels with similar colour that are spatially
    close tend to share the same class.

    Args:
        softmax_probs: (C, H, W) float32 class probabilities from model.
        image_rgb: (H, W, 3) uint8 reference image (e.g. B04/B03/B02).
        n_iters: Number of mean-field iterations.
        sxy_gaussian: Spatial sigma for the smoothness kernel.
        compat_gaussian: Compatibility weight for smoothness kernel.
        sxy_bilateral: Spatial sigma for the bilateral kernel.
        srgb_bilateral: Colour sigma for the bilateral kernel.
        compat_bilateral: Compatibility weight for bilateral kernel.

    Returns:
        (H, W) int32 refined class indices.

    Raises:
        ImportError: If pydensecrf is not installed.
    """
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except ImportError:
        raise ImportError(
            "pydensecrf is required for CRF post-processing. "
            "Install with: pip install pydensecrf"
        )

    C, H, W = softmax_probs.shape
    assert image_rgb.shape == (H, W, 3), (
        f"RGB shape {image_rgb.shape} must match probs spatial dims ({H}, {W})"
    )

    # Ensure contiguous float32
    probs = np.ascontiguousarray(softmax_probs, dtype=np.float32)

    # Build unary potentials from softmax (negative log-likelihood)
    unary = unary_from_softmax(probs)

    # Create CRF
    d = dcrf.DenseCRF2D(W, H, C)
    d.setUnaryEnergy(unary)

    # Smoothness kernel (spatial proximity only)
    d.addPairwiseGaussian(
        sxy=sxy_gaussian,
        compat=compat_gaussian,
    )

    # Bilateral kernel (spatial + colour proximity)
    d.addPairwiseBilateral(
        sxy=sxy_bilateral,
        srgb=srgb_bilateral,
        rgbim=np.ascontiguousarray(image_rgb),
        compat=compat_bilateral,
    )

    # Run inference
    Q = d.inference(n_iters)
    refined = np.argmax(Q, axis=0).reshape(H, W).astype(np.int32)

    return refined


def is_crf_available() -> bool:
    """Check if pydensecrf is installed."""
    try:
        import pydensecrf  # noqa: F401
        return True
    except ImportError:
        return False
