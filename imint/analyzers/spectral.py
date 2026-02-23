"""
imint/analyzers/spectral.py — Spectral index analyzer

Computes NDVI, NDWI, NDBI, EVI, and NBR from Sentinel-2 bands.
NBR (Normalized Burn Ratio) = (B08 - B12) / (B08 + B12) measures burn severity.
Falls back to RGB approximations if spectral bands are missing.
"""
from __future__ import annotations

import numpy as np
from .base import BaseAnalyzer, AnalysisResult

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
            b12 = bands.get("B12", np.full_like(b08, 0.1))
            fallback = False
        else:
            b04 = rgb[:, :, 0].astype(np.float32)  # Red
            b03 = rgb[:, :, 1].astype(np.float32)  # Green
            b02 = rgb[:, :, 2].astype(np.float32)  # Blue
            b08 = np.mean(rgb, axis=-1).astype(np.float32)  # Rough NIR proxy
            b11 = np.full_like(b08, 0.1)
            b12 = np.full_like(b08, 0.1)
            fallback = True

        # Compute indices
        ndvi = _safe_ratio(b08, b04)
        ndwi = _safe_ratio(b03, b08)
        ndbi = _safe_ratio(b11, b08)
        evi = 2.5 * (b08 - b04) / (b08 + 6.0 * b04 - 7.5 * b02 + 1.0 + 1e-10)
        nbr = _safe_ratio(b08, b12)

        return AnalysisResult(
            analyzer=self.name,
            success=True,
            outputs={
                "indices": {"NDVI": ndvi, "NDWI": ndwi, "NDBI": ndbi, "EVI": evi, "NBR": nbr},
            },
            metadata={"fallback_rgb": fallback, "date": date},
        )
