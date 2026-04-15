# Resolution and Boundary Crispness — Technical Analysis

## The Problem

v5c/v6a predictions have smooth, blurry boundaries compared to NMD's pixel-sharp edges.
This document analyzes the bottlenecks and solutions.

## Bottleneck 1: Sentinel-2 Spectral Resolution (10m)

Each pixel covers 100m2. Sub-pixel features (field boundaries, narrow roads, hedgerows)
are mixed into neighboring pixels. NMD compensates with:

- **LiDAR point clouds** (sub-meter accuracy) for forest delineation
- **Vector overlays**: Fastighetskartan (buildings), LPIS (agricultural parcels),
  Hydrografi (water), SCB roads/infrastructure — these are rasterized as hard constraints
- **Aerial orthophotos** (0.5m) for validation and transitional water interpretation

We have LPIS + SKS vectors but not LiDAR or aerial photos. Our theoretical resolution
ceiling is determined by the 10m spectral data + available vector constraints.

## Bottleneck 2: Prithvi ViT Patch Embedding (16x16)

The ViT patch embedding is a Conv2d(6, 1024, kernel_size=16, stride=16) — a hard 16x
downsampling. Each ViT token represents a 160m x 160m ground area at 10m resolution.

| Input size | Feature map | Ground per token | Tokens (T=4) |
|-----------|-------------|-----------------|-------------|
| 224px | 14x14 | 160m x 160m | 784 |
| 256px | 16x16 | 160m x 160m | 1024 |
| 448px | 28x28 | 160m x 160m | 3136 |

Even at 448px, the feature map is 28x28. The UPerNet decoder upsamples back to input
resolution via bilinear interpolation, creating smooth probability gradients instead
of hard class transitions.

Self-attention cost is O(n2) in tokens: 448px is 16x more expensive than 224px.

## Bottleneck 3: Loss Function Behavior

- **Cross-entropy + Focal**: optimizes per-pixel accuracy, can produce noisy boundaries
- **Dice loss**: optimizes overlap, reasonably sharp but not edge-aware
- **Lovasz-softmax**: optimizes IoU directly, tends toward spatially smooth predictions
  (fewer fragments = better IoU, even if boundaries are less precise)

## How NMD Achieves Crisp Boundaries

Source: NMD2018 Product Description v2.2 (Naturvardsverket)

NMD2018 production pipeline:
1. **Pixel-based classification** (Maximum Likelihood on Sentinel-2 + LiDAR features)
2. **Segmentation into objects** (spectrally similar pixel groups)
3. **Majority vote per segment** (dominant class assigned to all pixels in segment)
4. **Vector overlay** (buildings, roads, water, parcels override satellite classification)
5. **Contextual cleanup** (remove isolated pixels near boundaries)

Key: NMD uses OBIA (Object-Based Image Analysis) — classify per pixel, then
snap to spectral boundaries via segmentation. We can replicate steps 2-3-5 as
post-processing.

NMD2023 shifted to purely pixel-based (no segmentation step) with AI-based tools.

## Solution Architecture

### Tier 1: Post-Processing (no retraining, immediate)

**1a. Superpixel refinement** — HIGHEST IMPACT
- Generate SLIC superpixels from 6-band spectral data (not just RGB)
- Aggregate model softmax probabilities per superpixel (mean probability)
- Assign majority class to all pixels in superpixel
- Boundaries guaranteed to follow spectral edges
- Cost: ~90ms per 256px tile, ~500ms per 512px tile
- Parameters: n_segments=500, compactness=10

**1b. Dense CRF**
- Bilateral kernel snaps predictions to spectral edges
- ~300ms per tile, good for forest/field boundaries
- Less effective for spectrally similar class transitions

**1c. Morphological cleanup**
- Remove connected components < MMU (0.25 ha = 25 pixels at 10m)
- Opening/closing to smooth jagged edges
- Cost: negligible

**1d. Multi-scale inference**
- Run at 0.75x, 1.0x, 1.5x scales, average softmax
- Higher scales capture finer detail (1.5x: each token covers 107m vs 160m)
- Cost: ~4x inference time

### Tier 2: Hybrid Pixel Classifier (light training, 2-3 days)

**2a. Per-pixel RF/XGBoost on spectral + ViT features**
- Feature vector per pixel: [6 spectral bands, NDVI, NDWI, 23 ViT softmax probs] = ~31 features
- Train RF/XGBoost on ground truth labels
- ViT provides semantic context, RF provides pixel-level spatial precision
- Expected: near-NMD crispness for spectrally distinct boundaries
- Training: minutes (65k pixels per tile, millions total)
- Inference: milliseconds per tile

### Tier 3: Architectural (retraining required, 3-7 days)

**3a. Boundary loss head**
- Auxiliary binary boundary prediction from same features
- Hausdorff distance or boundary F1 loss
- Forces network to allocate capacity to edges

**3b. Learned upsampling (PixelShuffle/CARAFE)**
- Replace bilinear upsampling in UPerNet decoder
- Learns to distribute sub-patch information across spatial dimensions
- Moderate improvement (1-3% mIoU at boundaries)

**3c. Smaller patch size (8x8)**
- 4x more tokens, 16x more self-attention cost
- Requires FlexiViT kernel resize + extensive fine-tuning
- Not practical within one week

## Implementation Priority

| Phase | Approach | Effort | Impact | Retraining? |
|-------|----------|--------|--------|-------------|
| 1 | Superpixel refinement | 1 day | HIGH | No |
| 1 | CRF post-processing | done | Medium | No |
| 1 | Morphological cleanup | 0.5 day | Low-Med | No |
| 2 | Multi-scale inference | 1 day | Medium | No |
| 2 | Pixel RF refinement | 2 days | HIGH | Light |
| 3 | Boundary loss head | 1 day | Medium | Yes |
| 3 | PixelShuffle decoder | 2 days | Low-Med | Yes |

## Fundamental Limit

The remaining gap between our model and NMD after all post-processing comes from
NMD's use of higher-resolution source data (LiDAR, 0.5m aerial photos) and
authoritative vector databases (Fastighetskartan buildings, Hydrografi water network).
These provide sub-pixel positional accuracy that 10m Sentinel-2 cannot match.

For truly NMD-matching boundaries at the parcel level, we would need to incorporate
cadastral/parcel vector data from Lantmateriet as hard constraints — similar to how
we already overlay LPIS parcels for crop classes.

## References

- NMD2018 Product Description v2.2 (Naturvardsverket)
- NMD2023 Product Description v0.1
- Berman et al. 2018 — Lovasz-softmax loss
- Krahenbuhl & Koltun 2011 — Dense CRF
- Achanta et al. 2012 — SLIC superpixels
- Beyer et al. 2023 — FlexiViT patch resize
