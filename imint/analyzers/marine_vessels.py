"""
imint/analyzers/marine_vessels.py — Marine vessel detection from Sentinel-2

Detects marine vessels (including small pleasure crafts) using a YOLO11s
model fine-tuned on Sentinel-2 L1C-TCI imagery.  Uses SAHI sliding-window
inference to handle large images and merges overlapping predictions.

Based on:
    Mäyrä et al. (2025) "Mapping recreational marine traffic from
    Sentinel-2 imagery using YOLO object detection models",
    Remote Sensing of Environment.

Model weights: https://huggingface.co/mayrajeo/marine-vessel-yolo
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .base import BaseAnalyzer, AnalysisResult

# SCL class 6 = water
_SCL_WATER = 6

# Default HuggingFace model location
_HF_REPO = "mayrajeo/marine-vessel-yolo"
_HF_FILENAME = "yolo11s_tci.pt"
_LOCAL_CACHE = Path(__file__).resolve().parent.parent / "fm" / "marine_vessels"


def _resolve_model_path(config_path: str | None) -> str:
    """Resolve model .pt path: local file → cached download → HF download."""
    if config_path and os.path.isfile(config_path):
        return config_path

    cached = _LOCAL_CACHE / _HF_FILENAME
    if cached.exists():
        return str(cached)

    # Download from HuggingFace
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to auto-download the marine vessel "
            "model.  Install with: pip install huggingface_hub"
        )

    _LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=_HF_REPO,
        filename=_HF_FILENAME,
        local_dir=str(_LOCAL_CACHE),
    )
    return path


class MarineVesselAnalyzer(BaseAnalyzer):
    """Detect marine vessels in Sentinel-2 imagery using YOLO + SAHI."""

    name = "marine_vessels"

    def analyze(
        self,
        rgb: np.ndarray,
        bands=None,
        date=None,
        coords=None,
        output_dir="outputs",
        previous_results=None,
        scl: np.ndarray | None = None,
    ) -> AnalysisResult:
        confidence = self.config.get("confidence", 0.286)
        chip_size = self.config.get("chip_size", 320)
        overlap = self.config.get("overlap_ratio", 0.2)
        max_bbox_m = self.config.get("max_bbox_m", 750)
        water_filter = self.config.get("water_filter", True)
        model_path_cfg = self.config.get("model_path")

        # ── Import dependencies ───────────────────────────────────────
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction
        except ImportError:
            return AnalysisResult(
                analyzer=self.name, success=False,
                error="sahi not installed. Install with: pip install sahi",
            )

        try:
            model_path = _resolve_model_path(model_path_cfg)
        except Exception as e:
            return AnalysisResult(
                analyzer=self.name, success=False,
                error=f"Failed to load model: {e}",
            )

        # ── Prepare image (TCI formula) ────────────────────────────────
        # The YOLO model was trained on L1C-TCI imagery which uses the ESA
        # TCI formula: pixel = clip(reflectance × 2.5 × 255, 0, 255).
        # Using simple reflectance × 255 produces images that are too dark,
        # causing near-zero detections.  The × 2.5 factor is critical.
        if rgb.dtype != np.uint8:
            img = (rgb * 2.5 * 255).clip(0, 255).astype(np.uint8)
        else:
            img = rgb

        # ── Load model via SAHI ───────────────────────────────────────
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",  # SAHI uses yolov8 type for all ultralytics models
            model_path=model_path,
            confidence_threshold=confidence,
            device="cpu",
        )

        # ── Sliding-window inference ──────────────────────────────────
        result = get_sliced_prediction(
            img,
            detection_model,
            slice_height=chip_size,
            slice_width=chip_size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            verbose=0,
        )

        # ── Parse predictions ─────────────────────────────────────────
        regions = []
        for pred in result.object_prediction_list:
            bbox = pred.bbox
            x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
            regions.append({
                "bbox": {
                    "y_min": int(y1), "y_max": int(y2),
                    "x_min": int(x1), "x_max": int(x2),
                },
                "pixel_count": int((x2 - x1) * (y2 - y1)),
                "score": float(pred.score.value),
                "label": "vessel",
            })

        n_before = len(regions)

        # ── Post-processing: water mask (SCL) ─────────────────────────
        n_land_removed = 0
        if water_filter and scl is not None and regions:
            filtered = []
            for r in regions:
                bb = r["bbox"]
                cy = (bb["y_min"] + bb["y_max"]) // 2
                cx = (bb["x_min"] + bb["x_max"]) // 2
                # Check SCL at centroid — keep only water pixels
                if 0 <= cy < scl.shape[0] and 0 <= cx < scl.shape[1]:
                    if scl[cy, cx] == _SCL_WATER:
                        filtered.append(r)
                    else:
                        n_land_removed += 1
                else:
                    filtered.append(r)  # keep if out of SCL bounds
            regions = filtered

        # ── Post-processing: max bbox size ────────────────────────────
        n_size_removed = 0
        pixel_size_m = 10.0  # Sentinel-2 10m bands
        max_bbox_px = max_bbox_m / pixel_size_m
        if regions:
            filtered = []
            for r in regions:
                bb = r["bbox"]
                w = bb["x_max"] - bb["x_min"]
                h = bb["y_max"] - bb["y_min"]
                if w <= max_bbox_px and h <= max_bbox_px:
                    filtered.append(r)
                else:
                    n_size_removed += 1
            regions = filtered

        # ── Compute area stats ────────────────────────────────────────
        h_img, w_img = rgb.shape[:2]
        area_km2 = (h_img * pixel_size_m * w_img * pixel_size_m) / 1e6
        density = len(regions) / max(area_km2, 0.01)

        return AnalysisResult(
            analyzer=self.name,
            success=True,
            outputs={
                "regions": regions,
                "vessel_count": len(regions),
                "vessel_density_per_km2": round(density, 3),
            },
            metadata={
                "model": model_path,
                "confidence": confidence,
                "chip_size": chip_size,
                "overlap_ratio": overlap,
                "raw_detections": n_before,
                "land_filtered": n_land_removed,
                "size_filtered": n_size_removed,
                "final_detections": len(regions),
                "area_km2": round(area_km2, 2),
            },
        )
