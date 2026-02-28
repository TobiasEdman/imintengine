"""
imint/analyzers/change_detection.py — Change detection analyzer

Compares the current image against a stored baseline per area.
First run saves the image as baseline and reports zero changes.
Subsequent runs flag changed pixels above a configurable threshold.

Multispectral: When spectral bands are available (B02–B12), uses a
6-band stack (B02, B03, B04, B08/NIR, B11/SWIR1, B12/SWIR2) for
change detection instead of just RGB. This captures vegetation stress,
moisture changes, and other phenomena invisible in the visible spectrum.

Cloud masking: When an SCL (Scene Classification Layer) array is provided,
pixels classified as cloud (SCL 8/9/10) in either the current image or
the baseline are excluded from comparison.

Grid alignment: Uses the shared co-registration module
(imint.coregistration) for both integer-pixel and sub-pixel alignment.
Sentinel-2 data fetched on different dates may land on slightly different
pixel grids after reprojection to EPSG:3006. When a baseline _geo.json
exists, the analyzer aligns both images before comparison.
"""

import os
import json
import numpy as np
from scipy import ndimage

from .base import BaseAnalyzer, AnalysisResult
from ..coregistration import (
    compute_grid_offset,
    align_arrays,
    estimate_subpixel_offset,
    subpixel_shift,
    coregister_to_reference,
)

# SCL classes treated as cloud (matches fetch.py SCL_CLOUD_CLASSES)
_SCL_CLOUD = frozenset({8, 9, 10})

# Bands used for multispectral change detection
# B02 (Blue), B03 (Green), B04 (Red), B08 (NIR), B11 (SWIR1), B12 (SWIR2)
CHANGE_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]


def _area_key(coords: dict) -> str:
    """Stable string key from bounding box coordinates."""
    return f"{coords['west']}_{coords['south']}_{coords['east']}_{coords['north']}"


def _build_stack(rgb, bands, band_list=CHANGE_BANDS):
    """Build (H, W, N) stack from bands dict. Falls back to RGB.

    Args:
        rgb: (H, W, 3) RGB array.
        bands: Dict of band name → (H, W) array, or None.
        band_list: List of band names to stack.

    Returns:
        Tuple of (stack, is_multispectral) where stack is (H, W, N).
    """
    if bands and all(b in bands for b in band_list):
        return np.stack([bands[b] for b in band_list], axis=-1).astype(np.float32), True
    return rgb.astype(np.float32), False


def _ndvi(b04, b08):
    """Compute NDVI from red (B04) and NIR (B08)."""
    return (b08 - b04) / (b08 + b04 + 1e-10)


def _ndwi(b03, b08):
    """Compute NDWI from green (B03) and NIR (B08)."""
    return (b03 - b08) / (b03 + b08 + 1e-10)


def _nbr(b08, b12):
    """Compute NBR from NIR (B08) and SWIR2 (B12)."""
    return (b08 - b12) / (b08 + b12 + 1e-10)



# Legacy aliases — these now live in imint.coregistration but are
# kept here for backward compatibility with any external callers.
_compute_grid_offset = compute_grid_offset
_subpixel_shift = subpixel_shift
_estimate_subpixel_offset = estimate_subpixel_offset
_align_arrays = align_arrays


class ChangeDetectionAnalyzer(BaseAnalyzer):
    name = "change_detection"

    def analyze(self, rgb, bands=None, date=None, coords=None,
                output_dir="outputs", scl=None, geo=None):
        threshold = self.config.get("threshold", 0.15)
        min_region_pixels = self.config.get("min_region_pixels", 50)

        # Determine baseline paths (keyed by area only, not season)
        area = _area_key(coords) if coords else "default"
        baseline_dir = os.path.join(output_dir, "..", "baselines")
        os.makedirs(baseline_dir, exist_ok=True)
        baseline_rgb_path = os.path.join(baseline_dir, f"{area}.npy")
        baseline_bands_path = os.path.join(baseline_dir, f"{area}_bands.npy")
        scl_path = os.path.join(baseline_dir, f"{area}_scl.npy")

        # Allow explicit baseline override via config
        cfg_baseline = self.config.get("baseline_path")
        if cfg_baseline:
            if cfg_baseline.endswith("_bands.npy"):
                baseline_bands_path = cfg_baseline
                baseline_rgb_path = cfg_baseline.replace("_bands.npy", ".npy")
            else:
                baseline_rgb_path = cfg_baseline
                baseline_bands_path = cfg_baseline.replace(".npy", "_bands.npy")
            scl_path = baseline_bands_path.replace("_bands.npy", "_scl.npy")

        # Build current multispectral stack
        current_stack, is_multispectral = _build_stack(rgb, bands)

        # First run (no baseline exists and none specified): save and return zero
        if not os.path.exists(baseline_rgb_path) and not os.path.exists(baseline_bands_path):
            np.save(baseline_rgb_path, rgb)
            if is_multispectral:
                np.save(baseline_bands_path, current_stack)
            if scl is not None:
                np.save(scl_path, scl)
            return AnalysisResult(
                analyzer=self.name,
                success=True,
                outputs={
                    "change_fraction": 0.0,
                    "n_regions": 0,
                    "regions": [],
                    "change_mask": np.zeros(rgb.shape[:2], dtype=bool),
                },
                metadata={
                    "baseline_saved": baseline_rgb_path,
                    "threshold": threshold,
                    "multispectral": is_multispectral,
                    "n_bands": current_stack.shape[-1],
                },
            )

        # Load baseline — prefer multispectral, fall back to RGB
        baseline_is_multispectral = False
        if os.path.exists(baseline_bands_path) and is_multispectral:
            baseline_stack = np.load(baseline_bands_path)
            if baseline_stack.shape == current_stack.shape:
                baseline_is_multispectral = True
            else:
                # Shape mismatch — re-save and return zero
                np.save(baseline_rgb_path, rgb)
                np.save(baseline_bands_path, current_stack)
                if scl is not None:
                    np.save(scl_path, scl)
                return AnalysisResult(
                    analyzer=self.name,
                    success=True,
                    outputs={
                        "change_fraction": 0.0,
                        "n_regions": 0,
                        "regions": [],
                        "change_mask": np.zeros(rgb.shape[:2], dtype=bool),
                    },
                    metadata={"baseline_resaved": True, "reason": "shape_mismatch"},
                )

        if not baseline_is_multispectral:
            # Fall back to RGB baseline
            if not os.path.exists(baseline_rgb_path):
                # Only bands baseline exists but current has no bands — save new
                np.save(baseline_rgb_path, rgb)
                if is_multispectral:
                    np.save(baseline_bands_path, current_stack)
                if scl is not None:
                    np.save(scl_path, scl)
                return AnalysisResult(
                    analyzer=self.name,
                    success=True,
                    outputs={
                        "change_fraction": 0.0,
                        "n_regions": 0,
                        "regions": [],
                        "change_mask": np.zeros(rgb.shape[:2], dtype=bool),
                    },
                    metadata={"baseline_saved": baseline_rgb_path, "threshold": threshold},
                )

            baseline_rgb = np.load(baseline_rgb_path)
            if baseline_rgb.shape != rgb.shape:
                np.save(baseline_rgb_path, rgb)
                if is_multispectral:
                    np.save(baseline_bands_path, current_stack)
                if scl is not None:
                    np.save(scl_path, scl)
                return AnalysisResult(
                    analyzer=self.name,
                    success=True,
                    outputs={
                        "change_fraction": 0.0,
                        "n_regions": 0,
                        "regions": [],
                        "change_mask": np.zeros(rgb.shape[:2], dtype=bool),
                    },
                    metadata={"baseline_resaved": True, "reason": "shape_mismatch"},
                )

            # Use RGB for comparison (legacy baseline)
            baseline_stack = baseline_rgb.astype(np.float32)
            current_stack = rgb.astype(np.float32)

        # ── Co-registration (integer + sub-pixel) ─────────────────────
        # Primary alignment is now in fetch.py (_snap_to_target_grid),
        # which ensures all fetches produce data on the same NMD 10m grid.
        # This fallback handles baselines created before that fix, or
        # cases where the baseline _geo.json reveals a residual offset.
        # Uses the shared coregistration module for both levels.
        full_h, full_w = rgb.shape[:2]
        alignment_offset = (0, 0)
        subpixel_offset = (0.0, 0.0)
        align_row0, align_col0 = 0, 0

        baseline_geo_path = baseline_bands_path.replace("_bands.npy", "_geo.json")
        if not os.path.exists(baseline_geo_path):
            baseline_geo_path = baseline_rgb_path.replace(".npy", "_geo.json")

        cur_transform = None
        bl_transform = None
        if geo and hasattr(geo, "transform") and geo.transform:
            cur_transform = list(geo.transform)[:6]
        if os.path.exists(baseline_geo_path):
            with open(baseline_geo_path) as _gf:
                bl_geo = json.load(_gf)
            bl_transform = bl_geo.get("transform")

        # Use the shared co-registration pipeline
        band_idx = 2 if current_stack.shape[-1] > 2 else 0
        current_stack, baseline_stack, coreg_meta = coregister_to_reference(
            target=current_stack,
            reference=baseline_stack,
            target_transform=cur_transform,
            reference_transform=bl_transform,
            subpixel=True,
            reference_band=band_idx,
        )
        alignment_offset = coreg_meta["integer_offset"]
        subpixel_offset = coreg_meta["subpixel_offset"]
        align_row0, align_col0 = coreg_meta["crop_origin"]

        # Build combined cloud mask from current + baseline SCL
        aligned_h, aligned_w = current_stack.shape[:2]
        cloud_mask = np.zeros((aligned_h, aligned_w), dtype=bool)
        current_cloud_frac = 0.0
        baseline_cloud_frac = 0.0

        if scl is not None and scl.shape == rgb.shape[:2]:
            # Crop SCL to the aligned region
            scl_crop = scl[align_row0:align_row0 + aligned_h,
                           align_col0:align_col0 + aligned_w]
            current_cloud = np.isin(scl_crop, list(_SCL_CLOUD))
            cloud_mask |= current_cloud
            current_cloud_frac = float(current_cloud.sum()) / max(scl_crop.size, 1)

        if os.path.exists(scl_path):
            baseline_scl = np.load(scl_path)
            # If baseline SCL also needs alignment
            if alignment_offset != (0, 0) and baseline_scl.shape[:2] != (aligned_h, aligned_w):
                drow, dcol = alignment_offset
                bl_r0 = max(0, -drow)
                bl_c0 = max(0, -dcol)
                baseline_scl = baseline_scl[bl_r0:bl_r0 + aligned_h,
                                            bl_c0:bl_c0 + aligned_w]
            if baseline_scl.shape == (aligned_h, aligned_w):
                baseline_cloud = np.isin(baseline_scl, list(_SCL_CLOUD))
                cloud_mask |= baseline_cloud
                baseline_cloud_frac = float(baseline_cloud.sum()) / max(baseline_scl.size, 1)

        # Compute pixel-wise difference, mask out clouds
        diff = np.linalg.norm(current_stack - baseline_stack, axis=-1)
        change_mask = (diff > threshold) & ~cloud_mask

        # Morphological cleaning
        struct = ndimage.generate_binary_structure(2, 1)
        change_mask = ndimage.binary_opening(change_mask, structure=struct, iterations=1)
        change_mask = ndimage.binary_closing(change_mask, structure=struct, iterations=1)

        # Change fraction relative to valid (non-cloud) pixels
        cloud_masked_pixels = int(cloud_mask.sum())
        valid_pixels = aligned_h * aligned_w - cloud_masked_pixels
        change_fraction = float(change_mask.sum()) / max(valid_pixels, 1)

        metadata = {
            "threshold": threshold,
            "baseline": baseline_bands_path if baseline_is_multispectral else baseline_rgb_path,
            "multispectral": baseline_is_multispectral,
            "n_bands": int(current_stack.shape[-1]),
            "bands_used": CHANGE_BANDS if baseline_is_multispectral else ["R", "G", "B"],
            "cloud_masked_pixels": cloud_masked_pixels,
            "valid_pixels": valid_pixels,
            "cloud_fraction_current": round(current_cloud_frac, 4),
            "cloud_fraction_baseline": round(baseline_cloud_frac, 4),
        }

        if alignment_offset != (0, 0) or subpixel_offset != (0.0, 0.0):
            metadata["grid_alignment"] = {
                "offset_row": alignment_offset[0],
                "offset_col": alignment_offset[1],
                "aligned_shape": [aligned_h, aligned_w],
                "subpixel_dy": round(subpixel_offset[0], 4),
                "subpixel_dx": round(subpixel_offset[1], 4),
            }

        # Index-based change metadata (only when multispectral)
        dnbr_aligned = None
        if baseline_is_multispectral and is_multispectral:
            valid = ~cloud_mask
            # Current and baseline band arrays (from stacks)
            # Stack order: B02=0, B03=1, B04=2, B08=3, B11=4, B12=5
            cur_ndvi = _ndvi(current_stack[..., 2], current_stack[..., 3])
            bas_ndvi = _ndvi(baseline_stack[..., 2], baseline_stack[..., 3])
            cur_ndwi = _ndwi(current_stack[..., 1], current_stack[..., 3])
            bas_ndwi = _ndwi(baseline_stack[..., 1], baseline_stack[..., 3])

            # dNBR = NBR_pre - NBR_post (positive = burn severity)
            nbr_pre = _nbr(baseline_stack[..., 3], baseline_stack[..., 5])
            nbr_post = _nbr(current_stack[..., 3], current_stack[..., 5])
            dnbr_aligned = nbr_pre - nbr_post

            if valid.any():
                dv = dnbr_aligned[valid]
                metadata["ndvi_diff_mean"] = round(float((cur_ndvi - bas_ndvi)[valid].mean()), 4)
                metadata["ndwi_diff_mean"] = round(float((cur_ndwi - bas_ndwi)[valid].mean()), 4)
                metadata["dnbr_mean"] = round(float(dv.mean()), 4)
                metadata["dnbr_max"] = round(float(dv.max()), 4)
                n_valid = int(valid.sum())
                metadata["dnbr_severity"] = {
                    "high_severity": round(float((dv >= 0.66).sum() / n_valid), 4),
                    "moderate_high": round(float(((dv >= 0.44) & (dv < 0.66)).sum() / n_valid), 4),
                    "moderate_low": round(float(((dv >= 0.27) & (dv < 0.44)).sum() / n_valid), 4),
                    "low_severity": round(float(((dv >= 0.1) & (dv < 0.27)).sum() / n_valid), 4),
                    "unburned": round(float(((dv >= -0.1) & (dv < 0.1)).sum() / n_valid), 4),
                }
            else:
                metadata["ndvi_diff_mean"] = 0.0
                metadata["ndwi_diff_mean"] = 0.0
                metadata["dnbr_mean"] = 0.0
                metadata["dnbr_max"] = 0.0
                metadata["dnbr_severity"] = {}

        # Place aligned results back into full-size arrays so outputs
        # match the original image dimensions (important for overlays).
        if alignment_offset != (0, 0):
            full_change_mask = np.zeros((full_h, full_w), dtype=bool)
            full_change_mask[align_row0:align_row0 + aligned_h,
                             align_col0:align_col0 + aligned_w] = change_mask
            change_mask = full_change_mask

            full_diff = np.zeros((full_h, full_w), dtype=diff.dtype)
            full_diff[align_row0:align_row0 + aligned_h,
                      align_col0:align_col0 + aligned_w] = diff
            diff = full_diff

            if dnbr_aligned is not None:
                full_dnbr = np.zeros((full_h, full_w), dtype=dnbr_aligned.dtype)
                full_dnbr[align_row0:align_row0 + aligned_h,
                          align_col0:align_col0 + aligned_w] = dnbr_aligned
                dnbr_aligned = full_dnbr

        # Connected components
        labeled, n_features = ndimage.label(change_mask)
        regions = []
        for i in range(1, n_features + 1):
            component = labeled == i
            pixel_count = int(component.sum())
            if pixel_count < min_region_pixels:
                continue
            ys, xs = np.where(component)
            bbox = {
                "y_min": int(ys.min()),
                "y_max": int(ys.max()),
                "x_min": int(xs.min()),
                "x_max": int(xs.max()),
            }
            regions.append({"bbox": bbox, "pixel_count": pixel_count})

        outputs = {
            "change_fraction": change_fraction,
            "n_regions": len(regions),
            "regions": regions,
            "change_mask": change_mask,
            "change_diff": diff,
        }
        if dnbr_aligned is not None:
            outputs["dnbr"] = dnbr_aligned

        return AnalysisResult(
            analyzer=self.name,
            success=True,
            outputs=outputs,
            metadata=metadata,
        )
