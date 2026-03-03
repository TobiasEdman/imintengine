"""
imint/analyzers/shoreline.py — Shoreline detection via spectral indices

Detects water–land boundaries using NDWI (Normalized Difference Water
Index) and MNDWI (Modified NDWI) with Otsu thresholding — the same
proven approach used by CoastSat (Vos et al., 2019).

Spectral indices:
    NDWI  = (Green − NIR)  / (Green + NIR)    → B03, B08
    MNDWI = (Green − SWIR) / (Green + SWIR)   → B03, B11

The analyzer produces:
  - 4-class segmentation map: water / shallow water / sediment / land
  - Binary water mask
  - Shoreline edge contours (vectorised)
  - Per-class statistics

Reference: Vos et al. (2019), "CoastSat: A Google Earth Engine-enabled
Python toolkit to extract shorelines from publicly available satellite
imagery", Environmental Modelling & Software, 122, 104528.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .base import AnalysisResult, BaseAnalyzer

CLASS_NAMES = ["water", "shallow_water", "sediment", "land"]
CLASS_LABELS_SV = ["Vatten", "Grunt vatten", "Sediment", "Land"]


class ShorelineAnalyzer(BaseAnalyzer):
    """Spectral-index shoreline detector (NDWI/MNDWI + Otsu).

    Works on multi-band Sentinel-2 data.  Requires bands B03 (Green),
    B08 (NIR), and optionally B11 (SWIR-1) for MNDWI.
    """

    name = "shoreline"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self._ndwi_threshold: float | None = None
        self._mndwi_threshold: float | None = None

    # ------------------------------------------------------------------ #
    #  Spectral indices
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """NDWI = (Green − NIR) / (Green + NIR).  Water > 0."""
        denom = green + nir + 1e-10
        return (green - nir) / denom

    @staticmethod
    def compute_mndwi(green: np.ndarray, swir: np.ndarray) -> np.ndarray:
        """MNDWI = (Green − SWIR) / (Green + SWIR).  Water > 0."""
        denom = green + swir + 1e-10
        return (green - swir) / denom

    # ------------------------------------------------------------------ #
    #  Classification
    # ------------------------------------------------------------------ #

    @staticmethod
    def _otsu(index: np.ndarray) -> float:
        """Compute Otsu threshold on a normalised index image."""
        # Scale to 0–255 uint8 for cv2.threshold
        valid = index[np.isfinite(index)]
        if len(valid) == 0:
            return 0.0
        lo, hi = float(np.nanpercentile(valid, 1)), float(np.nanpercentile(valid, 99))
        scaled = np.clip((index - lo) / (hi - lo + 1e-10) * 255, 0, 255).astype(np.uint8)
        thresh_val, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Convert back to index scale
        return lo + (thresh_val / 255.0) * (hi - lo)

    def classify(
        self,
        bands: dict[str, np.ndarray],
        rgb: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Produce a 4-class segmentation from spectral bands.

        Classes:
            0 = deep water    (NDWI > otsu  AND  MNDWI > otsu)
            1 = shallow water (NDWI > otsu  OR   high-NDWI fringe)
            2 = sediment      (NDWI near threshold + low NDVI)
            3 = land          (everything else)

        Returns:
            seg_map: (H, W) uint8, values 0–3.
            info: dict with thresholds and fractions.
        """
        green = bands.get("B03")
        nir = bands.get("B08")
        swir = bands.get("B11")

        if green is None or nir is None:
            raise ValueError("Bands B03 (Green) and B08 (NIR) required")

        ndwi = self.compute_ndwi(green, nir)
        ndwi_thresh = self._otsu(ndwi)
        self._ndwi_threshold = ndwi_thresh

        # Water mask from NDWI
        water_ndwi = ndwi > ndwi_thresh

        # MNDWI if SWIR available (better at distinguishing built-up)
        if swir is not None:
            mndwi = self.compute_mndwi(green, swir)
            mndwi_thresh = self._otsu(mndwi)
            self._mndwi_threshold = mndwi_thresh
            water_mndwi = mndwi > mndwi_thresh
            # Combine: require both NDWI and MNDWI for "deep water"
            deep_water = water_ndwi & water_mndwi
            shallow_water = water_ndwi & (~water_mndwi)
        else:
            mndwi = None
            deep_water = water_ndwi
            shallow_water = np.zeros_like(water_ndwi, dtype=bool)

        # Sediment detection: near water edge + low vegetation
        # NDVI for vegetation
        red = bands.get("B04")
        if red is not None and nir is not None:
            ndvi = (nir - red) / (nir + red + 1e-10)
            # Sediment: NOT water, low NDVI (< 0.2), NDWI close to threshold
            is_low_veg = ndvi < 0.2
            is_near_water = (ndwi > ndwi_thresh - 0.15) & (ndwi <= ndwi_thresh)
            sediment = is_low_veg & is_near_water
        else:
            sediment = np.zeros_like(deep_water, dtype=bool)

        # Build segmentation map
        H, W = green.shape
        seg_map = np.full((H, W), 3, dtype=np.uint8)  # default: land
        seg_map[sediment] = 2
        seg_map[shallow_water] = 1
        seg_map[deep_water] = 0

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        for cls in [0, 1]:
            mask = (seg_map == cls).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            seg_map[mask == 1] = cls

        total = H * W
        info = {
            "ndwi_threshold": round(float(ndwi_thresh), 4),
            "mndwi_threshold": round(float(self._mndwi_threshold), 4) if self._mndwi_threshold else None,
            "water_fraction": float((seg_map <= 1).sum()) / total,
            "deep_water_fraction": float((seg_map == 0).sum()) / total,
            "shallow_water_fraction": float((seg_map == 1).sum()) / total,
            "sediment_fraction": float((seg_map == 2).sum()) / total,
            "land_fraction": float((seg_map == 3).sum()) / total,
        }

        return seg_map, info

    # ------------------------------------------------------------------ #
    #  Shoreline extraction
    # ------------------------------------------------------------------ #

    @staticmethod
    def extract_shoreline(seg_map: np.ndarray) -> np.ndarray:
        """Extract shoreline as binary edge mask (water <-> non-water).

        Args:
            seg_map: (H, W) uint8 — class indices (0, 1 = water classes).

        Returns:
            (H, W) uint8 — binary shoreline mask (255 = shoreline pixel).
        """
        water = (seg_map <= 1).astype(np.uint8)

        # Morphological cleanup
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        water = cv2.morphologyEx(water, cv2.MORPH_CLOSE, kernel_clean)
        water = cv2.morphologyEx(water, cv2.MORPH_OPEN, kernel_clean)

        # Edge via morphological gradient
        kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(water, cv2.MORPH_GRADIENT, kernel_edge)

        return (edges * 255).astype(np.uint8)

    @staticmethod
    def extract_contours(
        shoreline_mask: np.ndarray, min_length: int = 15
    ) -> list[np.ndarray]:
        """Vectorize shoreline mask into contour polylines.

        Returns list of (N, 2) arrays in pixel coordinates [x, y].
        """
        contours, _ = cv2.findContours(
            shoreline_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        results = []
        for c in contours:
            if len(c) >= min_length:
                coords = c.reshape(-1, 2)
                results.append(coords)
        return results

    # ------------------------------------------------------------------ #
    #  Multi-year batch processing
    # ------------------------------------------------------------------ #

    def predict_multi_year(
        self,
        yearly_bands: dict[int, dict[str, np.ndarray]],
        yearly_rgbs: dict[int, np.ndarray] | None = None,
    ) -> dict[int, dict]:
        """Run classification for multiple years.

        Returns:
            {year: {"seg_map": ..., "shoreline": ..., "contours": [...], "info": {...}}}
        """
        results = {}
        for year in sorted(yearly_bands.keys()):
            print(f"    {year}...", end=" ", flush=True)
            rgb = yearly_rgbs[year] if yearly_rgbs else None
            seg_map, info = self.classify(yearly_bands[year], rgb)
            shoreline = self.extract_shoreline(seg_map)
            contours = self.extract_contours(shoreline)
            results[year] = {
                "seg_map": seg_map,
                "shoreline": shoreline,
                "contours": contours,
                "info": info,
            }
            print(f"water={info['water_fraction']:.1%}, "
                  f"{len(contours)} contours, "
                  f"NDWI thresh={info['ndwi_threshold']:.3f}")
        return results

    # ------------------------------------------------------------------ #
    #  BaseAnalyzer interface
    # ------------------------------------------------------------------ #

    def analyze(
        self,
        rgb: np.ndarray,
        bands: dict[str, np.ndarray] | None = None,
        date: str | None = None,
        coords: dict | None = None,
        output_dir: str = "outputs",
    ) -> AnalysisResult:
        """Run shoreline detection as part of the IMINT pipeline."""
        if bands is None:
            return AnalysisResult(
                analyzer=self.name, success=False,
                error="Multi-band input required (B03, B08)",
            )

        seg_map, info = self.classify(bands, rgb)
        shoreline_mask = self.extract_shoreline(seg_map)

        class_fractions = {}
        for i, name in enumerate(CLASS_NAMES):
            class_fractions[name] = float((seg_map == i).sum()) / (seg_map.shape[0] * seg_map.shape[1])

        return AnalysisResult(
            analyzer=self.name,
            success=True,
            outputs={
                "segmentation_map": seg_map,
                "shoreline_mask": shoreline_mask,
                "water_fraction": info["water_fraction"],
                "class_fractions": class_fractions,
                "ndwi": self.compute_ndwi(bands["B03"], bands["B08"]),
            },
            metadata={
                "date": date,
                "method": "NDWI/MNDWI + Otsu (CoastSat approach)",
                "ndwi_threshold": info["ndwi_threshold"],
                "mndwi_threshold": info.get("mndwi_threshold"),
                "reference": "Vos et al. (2019) CoastSat",
            },
        )
