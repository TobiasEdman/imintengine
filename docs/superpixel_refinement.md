# Superpixel Refinement — Implementation Guide

## Concept

Replicate NMD's object-based approach as post-processing:
1. Model produces per-pixel softmax probabilities (coarse but semantically rich)
2. SLIC generates superpixels from spectral data (pixel-sharp boundaries)
3. Aggregate softmax per superpixel, assign majority class
4. Result: model's semantic accuracy + spectral-edge-aligned boundaries

## Why This Works

- SLIC operates on all 6 Sentinel-2 bands (not just RGB), capturing SWIR-distinct boundaries
- Superpixel boundaries follow spectral edges — exactly where class transitions occur
- The model's softmax provides semantic context (prevents spectrally plausible but wrong classifications)
- NMD2018 used the same approach: pixel classification + segmentation + majority vote

## Parameters

### SLIC (recommended default)
- n_segments=500 (for 256x256 tiles) or 1000 (for 512x512)
- compactness=10 (balance between spatial regularity and spectral adherence)
- convert2lab=False (required for 6-band input)

### Felzenszwalb (alternative for agricultural areas)
- scale=150, sigma=0.5, min_size=20
- Produces more natural boundaries for field parcels
- Less controllable segment count

### Watershed (alternative for mixed landscapes)
- Uses NDVI gradient as landscape function
- SLIC centroids as markers to prevent over-segmentation
- Best for agricultural fields where NDVI gradients are strong

## Performance

| Size | Method | n_segments | Time |
|------|--------|-----------|------|
| 256x256x6 | SLIC | 500 | 87 ms |
| 256x256x6 | Felzenszwalb | ~440 | 88 ms |
| 256x256x6 | Watershed | 500 | 65 ms |
| 512x512x6 | SLIC | 1000 | 500 ms |

## Aggregation Methods

### mean_prob (recommended)
Average softmax probabilities per superpixel, then argmax.
Best when model confidence is calibrated.

### majority_vote
Argmax per pixel, then count votes per superpixel.
More robust to poorly calibrated models.

### weighted_mean
Weight each pixel's softmax by its max confidence.
Gives more influence to high-confidence pixels within each superpixel.

## Integration

File: imint/inference/superpixel_refine.py (to be created)

Plugs in after sliding window inference (which produces full-tile softmax)
and before CRF (which operates on the refined predictions).

Pipeline: model → sliding window → superpixel refinement → CRF → morphological cleanup

## Dependencies

- scikit-image (already installed): SLIC, Felzenszwalb, watershed
- scipy (already installed): ndimage for label aggregation
- Optional: cuda-slic for GPU acceleration (pip install cuda-slic)
