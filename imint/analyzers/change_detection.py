"""
imint/analyzers/change_detection.py — Change detection analyzer

Compares the current image against a stored baseline per season/area.
First run saves the image as baseline and reports zero changes.
Subsequent runs flag changed pixels above a configurable threshold.
"""

import os
import numpy as np
from scipy import ndimage
from pathlib import Path

from .base import BaseAnalyzer, AnalysisResult


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


class ChangeDetectionAnalyzer(BaseAnalyzer):
    name = "change_detection"

    def analyze(self, rgb, bands=None, date=None, coords=None, output_dir="outputs"):
        threshold = self.config.get("threshold", 0.15)
        min_region_pixels = self.config.get("min_region_pixels", 50)

        # Determine baseline path
        season = _season(date) if date else "unknown"
        area = _area_key(coords) if coords else "default"
        baseline_dir = os.path.join(output_dir, "..", "baselines")
        os.makedirs(baseline_dir, exist_ok=True)
        baseline_path = os.path.join(baseline_dir, f"{season}_{area}.npy")

        # First run: save baseline, return zero changes
        if not os.path.exists(baseline_path):
            np.save(baseline_path, rgb)
            return AnalysisResult(
                analyzer=self.name,
                success=True,
                outputs={
                    "change_fraction": 0.0,
                    "n_regions": 0,
                    "regions": [],
                    "change_mask": np.zeros(rgb.shape[:2], dtype=bool),
                },
                metadata={"baseline_saved": baseline_path, "threshold": threshold},
            )

        # Load baseline and compute difference
        baseline = np.load(baseline_path)
        if baseline.shape != rgb.shape:
            np.save(baseline_path, rgb)
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

        diff = np.linalg.norm(rgb.astype(np.float32) - baseline.astype(np.float32), axis=-1)
        change_mask = diff > threshold

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

        change_fraction = float(change_mask.sum()) / change_mask.size

        return AnalysisResult(
            analyzer=self.name,
            success=True,
            outputs={
                "change_fraction": change_fraction,
                "n_regions": len(regions),
                "regions": regions,
                "change_mask": change_mask,
            },
            metadata={"threshold": threshold, "baseline": baseline_path},
        )
