"""
imint/analyzers/spectral.py — Spectral index analyzer

Computes NDVI, NDWI, NDBI, and EVI from Sentinel-2 bands.
Classifies each pixel as water, vegetation, built-up, or bare soil.
Falls back to RGB approximations if spectral bands are missing.
"""
from __future__ import annotations

import numpy as np
from .base import BaseAnalyzer, AnalysisResult

# Land cover class encoding
LC_WATER = 1
LC_VEGETATION = 2
LC_BUILT_UP = 3
LC_BARE_SOIL = 4


def _safe_ratio(a, b):
    """Normalized difference ratio with epsilon to avoid division by zero."""
    return (a - b) / (a + b + 1e-10)


class SpectralAnalyzer(BaseAnalyzer):
    name = "spectral"

    def analyze(self, rgb, bands=None, date=None, coords=None, output_dir="outputs"):
        # Extract bands — fall back to RGB approximations
        if bands and "B08" in bands:
            b02 = bands.get("B02", rgb[:, :, 2] if rgb is not None else np.zeros((256, 256)))
            b03 = bands.get("B03", rgb[:, :, 1] if rgb is not None else np.zeros((256, 256)))
            b04 = bands.get("B04", rgb[:, :, 0] if rgb is not None else np.zeros((256, 256)))
            b08 = bands["B08"]
            b11 = bands.get("B11", np.full_like(b08, 0.1))
            fallback = False
        else:
            b04 = rgb[:, :, 0].astype(np.float32)  # Red
            b03 = rgb[:, :, 1].astype(np.float32)  # Green
            b02 = rgb[:, :, 2].astype(np.float32)  # Blue
            b08 = np.mean(rgb, axis=-1).astype(np.float32)  # Rough NIR proxy
            b11 = np.full_like(b08, 0.1)
            fallback = True

        # Compute indices
        ndvi = _safe_ratio(b08, b04)
        ndwi = _safe_ratio(b03, b08)
        ndbi = _safe_ratio(b11, b08)
        evi = 2.5 * (b08 - b04) / (b08 + 6.0 * b04 - 7.5 * b02 + 1.0 + 1e-10)

        # Land cover classification
        h, w = ndvi.shape
        land_cover = np.full((h, w), LC_BARE_SOIL, dtype=np.uint8)

        ndwi_thresh = self.config.get("ndwi_threshold", 0.3)
        ndvi_thresh = self.config.get("ndvi_threshold", 0.3)
        ndbi_thresh = self.config.get("ndbi_threshold", 0.0)

        land_cover[ndwi > ndwi_thresh] = LC_WATER
        land_cover[(ndvi > ndvi_thresh) & (land_cover == LC_BARE_SOIL)] = LC_VEGETATION
        land_cover[(ndbi > ndbi_thresh) & (land_cover == LC_BARE_SOIL)] = LC_BUILT_UP

        total = h * w
        stats = {
            "water_fraction": float((land_cover == LC_WATER).sum()) / total,
            "vegetation_fraction": float((land_cover == LC_VEGETATION).sum()) / total,
            "built_up_fraction": float((land_cover == LC_BUILT_UP).sum()) / total,
            "bare_soil_fraction": float((land_cover == LC_BARE_SOIL).sum()) / total,
        }

        return AnalysisResult(
            analyzer=self.name,
            success=True,
            outputs={
                "indices": {"NDVI": ndvi, "NDWI": ndwi, "NDBI": ndbi, "EVI": evi},
                "land_cover": land_cover,
                "stats": stats,
            },
            metadata={"fallback_rgb": fallback, "date": date},
        )
