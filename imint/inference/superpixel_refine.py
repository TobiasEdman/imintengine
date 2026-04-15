"""
imint/inference/superpixel_refine.py — Superpixel-based prediction refinement

Replicates NMD's object-based approach: generate superpixels from spectral
data, then assign each superpixel the dominant class from the model's softmax
predictions.  Boundaries are guaranteed to follow spectral edges.

Works with all 6 Sentinel-2 bands (B02, B03, B04, B8A, B11, B12), not just
RGB — captures SWIR-distinct boundaries invisible in visible bands.
"""
from __future__ import annotations

import numpy as np


def superpixel_refine(
    softmax_probs: np.ndarray,
    spectral: np.ndarray,
    aux: np.ndarray | None = None,
    method: str = "slic",
    aggregation: str = "mean_prob",
    n_segments: int = 500,
    compactness: float = 10.0,
    min_size: int = 10,
    **kwargs,
) -> np.ndarray:
    """Refine predictions using superpixel boundaries from spectral + aux data.

    Superpixels are computed from the concatenation of spectral bands and
    auxiliary channels (DEM, tree height, VPP phenology, etc.).  This means
    boundaries follow not just spectral edges but also terrain breaks,
    forest height transitions, and phenological gradients.

    Args:
        softmax_probs: (C, H, W) float32 per-class probabilities.
        spectral: (B, H, W) float32 spectral bands (raw or normalized).
        aux: Optional (N, H, W) float32 auxiliary channels.  Concatenated
            with spectral for superpixel generation.  Should be z-score
            normalized so all channels have comparable scale.
        method: ``"slic"``, ``"felzenszwalb"``, or ``"watershed"``.
        aggregation: ``"mean_prob"`` (average softmax per superpixel),
            ``"majority_vote"`` (argmax per pixel, count per superpixel),
            or ``"weighted_mean"`` (confidence-weighted softmax).
        n_segments: Target number of superpixels (SLIC/watershed).
        compactness: SLIC spatial vs spectral balance (lower = more
            spectral adherence, better for field boundaries).
        min_size: Minimum superpixel size in pixels.
        **kwargs: Passed to the segmentation function.

    Returns:
        (H, W) uint8 refined class predictions.
    """
    from skimage.segmentation import slic, felzenszwalb
    from skimage.filters import sobel

    C, H, W = softmax_probs.shape

    # Concatenate spectral + aux for richer superpixel boundaries
    if aux is not None and aux.shape[1:] == (H, W):
        combined = np.concatenate([spectral, aux], axis=0)
    else:
        combined = spectral

    # Prepare for segmentation: (B+N, H, W) → (H, W, B+N)
    spec_hwc = np.moveaxis(combined, 0, -1).astype(np.float64)

    if method == "slic":
        segments = slic(
            spec_hwc,
            n_segments=n_segments,
            compactness=compactness,
            convert2lab=False,
            channel_axis=-1,
            start_label=0,
            min_size_factor=0.3,
            **kwargs,
        )
    elif method == "felzenszwalb":
        scale = kwargs.pop("scale", 150)
        sigma = kwargs.pop("sigma", 0.5)
        segments = felzenszwalb(
            spec_hwc, scale=scale, sigma=sigma,
            min_size=min_size, channel_axis=-1, **kwargs,
        )
    elif method == "watershed":
        from skimage.segmentation import watershed as _ws
        # NDVI gradient as landscape function
        if spectral.shape[0] >= 4:
            nir, red = spectral[3], spectral[2]
            ndvi = (nir - red) / np.clip(nir + red, 1e-8, None)
        else:
            ndvi = spec_hwc.mean(axis=-1)
        grad = sobel(ndvi)
        markers = slic(
            spec_hwc, n_segments=n_segments, compactness=50,
            convert2lab=False, channel_axis=-1, start_label=1,
        )
        segments = _ws(grad, markers)
        segments -= segments.min()  # ensure 0-indexed
    else:
        raise ValueError(f"Unknown method: {method!r}")

    return _aggregate(softmax_probs, segments, C, aggregation)


def _aggregate(
    softmax_probs: np.ndarray,
    segments: np.ndarray,
    n_classes: int,
    mode: str,
) -> np.ndarray:
    """Assign class per superpixel by aggregating softmax predictions."""
    n_seg = segments.max() + 1
    flat_seg = segments.ravel()

    if mode == "mean_prob":
        seg_probs = np.zeros((n_seg, n_classes), dtype=np.float64)
        seg_count = np.zeros(n_seg, dtype=np.int64)
        for c in range(n_classes):
            np.add.at(seg_probs[:, c], flat_seg, softmax_probs[c].ravel())
        np.add.at(seg_count, flat_seg, 1)
        seg_probs /= np.maximum(seg_count[:, None], 1)
        seg_pred = seg_probs.argmax(axis=1).astype(np.uint8)

    elif mode == "majority_vote":
        pixel_pred = softmax_probs.argmax(axis=0).ravel()
        seg_votes = np.zeros((n_seg, n_classes), dtype=np.int64)
        np.add.at(seg_votes, (flat_seg, pixel_pred), 1)
        seg_pred = seg_votes.argmax(axis=1).astype(np.uint8)

    elif mode == "weighted_mean":
        confidence = softmax_probs.max(axis=0)  # (H, W)
        seg_probs = np.zeros((n_seg, n_classes), dtype=np.float64)
        for c in range(n_classes):
            np.add.at(seg_probs[:, c], flat_seg,
                       (softmax_probs[c] * confidence).ravel())
        seg_pred = seg_probs.argmax(axis=1).astype(np.uint8)

    else:
        raise ValueError(f"Unknown aggregation: {mode!r}")

    return seg_pred[segments]


def morphological_cleanup(
    prediction: np.ndarray,
    min_pixels: int = 25,
) -> np.ndarray:
    """Remove small connected components below minimum mapping unit.

    Args:
        prediction: (H, W) uint8 class predictions.
        min_pixels: Minimum component size (25 px = 0.25 ha at 10m).

    Returns:
        (H, W) uint8 cleaned predictions.
    """
    from scipy import ndimage

    cleaned = prediction.copy()
    for cls in np.unique(prediction):
        if cls == 0:
            continue  # skip background
        mask = prediction == cls
        labeled, n_components = ndimage.label(mask)
        for comp_id in range(1, n_components + 1):
            comp_mask = labeled == comp_id
            if comp_mask.sum() < min_pixels:
                # Replace with most common neighbor class
                dilated = ndimage.binary_dilation(comp_mask, iterations=1)
                border = dilated & ~comp_mask
                if border.any():
                    neighbor_classes = prediction[border]
                    neighbor_classes = neighbor_classes[neighbor_classes != cls]
                    if len(neighbor_classes) > 0:
                        replacement = np.bincount(neighbor_classes).argmax()
                        cleaned[comp_mask] = replacement

    return cleaned
