"""
imint/analyzers/change_detection.py — Change detection analyzer

Compares the current image against a stored baseline per season/area.
First run saves the image as baseline and reports zero changes.
Subsequent runs flag changed pixels above a configurable threshold.

Multispectral: When spectral bands are available (B02–B12), uses a
6-band stack (B02, B03, B04, B08/NIR, B11/SWIR1, B12/SWIR2) for
change detection instead of just RGB. This captures vegetation stress,
moisture changes, and other phenomena invisible in the visible spectrum.

Cloud masking: When an SCL (Scene Classification Layer) array is provided,
pixels classified as cloud (SCL 8/9/10) in either the current image or
the baseline are excluded from comparison.
"""

import os
import numpy as np
from scipy import ndimage

from .base import BaseAnalyzer, AnalysisResult

# SCL classes treated as cloud (matches fetch.py SCL_CLOUD_CLASSES)
_SCL_CLOUD = frozenset({8, 9, 10})

# Bands used for multispectral change detection
# B02 (Blue), B03 (Green), B04 (Red), B08 (NIR), B11 (SWIR1), B12 (SWIR2)
CHANGE_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]


def _season(date: str) -> str:
    """Map ISO date to season name."""
    month = int(date.split("-")[1])
    if month in (3, 4, 5):
        return "spring"
    elif month in (6, 7, 8):
        return "summer"
    elif month in (9, 10, 11):
        return "autumn"
    return "winter"


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


class ChangeDetectionAnalyzer(BaseAnalyzer):
    name = "change_detection"

    def analyze(self, rgb, bands=None, date=None, coords=None,
                output_dir="outputs", scl=None):
        threshold = self.config.get("threshold", 0.15)
        min_region_pixels = self.config.get("min_region_pixels", 50)

        # Determine baseline paths
        season = _season(date) if date else "unknown"
        area = _area_key(coords) if coords else "default"
        baseline_dir = os.path.join(output_dir, "..", "baselines")
        os.makedirs(baseline_dir, exist_ok=True)
        baseline_rgb_path = os.path.join(baseline_dir, f"{season}_{area}.npy")
        baseline_bands_path = os.path.join(baseline_dir, f"{season}_{area}_bands.npy")
        scl_path = os.path.join(baseline_dir, f"{season}_{area}_scl.npy")

        # Build current multispectral stack
        current_stack, is_multispectral = _build_stack(rgb, bands)

        # First run: save baseline, return zero changes
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

        # Build combined cloud mask from current + baseline SCL
        cloud_mask = np.zeros(rgb.shape[:2], dtype=bool)
        current_cloud_frac = 0.0
        baseline_cloud_frac = 0.0

        if scl is not None and scl.shape == rgb.shape[:2]:
            current_cloud = np.isin(scl, list(_SCL_CLOUD))
            cloud_mask |= current_cloud
            current_cloud_frac = float(current_cloud.sum()) / max(scl.size, 1)

        if os.path.exists(scl_path):
            baseline_scl = np.load(scl_path)
            if baseline_scl.shape == rgb.shape[:2]:
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

        # Change fraction relative to valid (non-cloud) pixels
        cloud_masked_pixels = int(cloud_mask.sum())
        valid_pixels = rgb.shape[0] * rgb.shape[1] - cloud_masked_pixels
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

        # Index-based change metadata (only when multispectral)
        if baseline_is_multispectral and is_multispectral:
            valid = ~cloud_mask
            # Current and baseline band arrays (from stacks)
            # Stack order: B02=0, B03=1, B04=2, B08=3, B11=4, B12=5
            cur_ndvi = _ndvi(current_stack[..., 2], current_stack[..., 3])
            bas_ndvi = _ndvi(baseline_stack[..., 2], baseline_stack[..., 3])
            cur_ndwi = _ndwi(current_stack[..., 1], current_stack[..., 3])
            bas_ndwi = _ndwi(baseline_stack[..., 1], baseline_stack[..., 3])

            if valid.any():
                metadata["ndvi_diff_mean"] = round(float((cur_ndvi - bas_ndvi)[valid].mean()), 4)
                metadata["ndwi_diff_mean"] = round(float((cur_ndwi - bas_ndwi)[valid].mean()), 4)
            else:
                metadata["ndvi_diff_mean"] = 0.0
                metadata["ndwi_diff_mean"] = 0.0

        return AnalysisResult(
            analyzer=self.name,
            success=True,
            outputs={
                "change_fraction": change_fraction,
                "n_regions": len(regions),
                "regions": regions,
                "change_mask": change_mask,
            },
            metadata=metadata,
        )
