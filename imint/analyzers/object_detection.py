"""
imint/analyzers/object_detection.py — Object detection analyzer

Two modes:
  - heatmap (default): variance-based anomaly detection per patch.
    No model or extra dependencies needed.
  - model: YOLO inference via ultralytics. Requires a .pt weights file.
"""
from __future__ import annotations

import numpy as np
from .base import BaseAnalyzer, AnalysisResult


class ObjectDetectionAnalyzer(BaseAnalyzer):
    name = "object_detection"

    def analyze(self, rgb, bands=None, date=None, coords=None, output_dir="outputs"):
        mode = self.config.get("mode", "heatmap")
        if mode == "model":
            return self._run_model(rgb)
        return self._run_heatmap(rgb)

    def _run_heatmap(self, rgb):
        """Variance-based anomaly detection over image patches."""
        patch_size = self.config.get("patch_size", 32)
        std_threshold = self.config.get("std_threshold", 2.0)

        h, w = rgb.shape[:2]
        variances = []
        patches = []

        for y in range(0, h - patch_size + 1, patch_size):
            for x in range(0, w - patch_size + 1, patch_size):
                patch = rgb[y:y + patch_size, x:x + patch_size]
                var = float(np.var(patch))
                variances.append(var)
                patches.append((y, x, var))

        if not variances:
            return AnalysisResult(
                analyzer=self.name, success=True,
                outputs={"regions": [], "heatmap": np.zeros(rgb.shape[:2])},
                metadata={"mode": "heatmap", "n_patches": 0},
            )

        mean_var = np.mean(variances)
        std_var = np.std(variances)
        threshold = mean_var + std_threshold * std_var

        regions = []
        heatmap = np.zeros(rgb.shape[:2], dtype=np.float32)

        for y, x, var in patches:
            heatmap[y:y + patch_size, x:x + patch_size] = var
            if var > threshold:
                regions.append({
                    "bbox": {
                        "y_min": y, "y_max": y + patch_size,
                        "x_min": x, "x_max": x + patch_size,
                    },
                    "pixel_count": patch_size * patch_size,
                    "score": float(var),
                    "label": "anomaly",
                })

        return AnalysisResult(
            analyzer=self.name,
            success=True,
            outputs={"regions": regions, "heatmap": heatmap},
            metadata={
                "mode": "heatmap",
                "patch_size": patch_size,
                "variance_threshold": float(threshold),
                "n_detections": len(regions),
            },
        )

    def _run_model(self, rgb):
        """YOLO model inference. Requires ultralytics."""
        model_path = self.config.get("model_path", "yolov8n.pt")
        conf_threshold = self.config.get("confidence", 0.25)

        try:
            from ultralytics import YOLO
        except ImportError:
            return AnalysisResult(
                analyzer=self.name, success=False,
                error="ultralytics not installed. Install with: pip install ultralytics",
            )

        model = YOLO(model_path)
        img = (rgb * 255).clip(0, 255).astype(np.uint8)
        results = model(img, conf=conf_threshold, verbose=False)

        regions = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                regions.append({
                    "bbox": {
                        "y_min": int(y1), "y_max": int(y2),
                        "x_min": int(x1), "x_max": int(x2),
                    },
                    "pixel_count": int((x2 - x1) * (y2 - y1)),
                    "score": float(box.conf[0]),
                    "label": r.names[int(box.cls[0])],
                })

        return AnalysisResult(
            analyzer=self.name,
            success=True,
            outputs={"regions": regions},
            metadata={
                "mode": "model",
                "model_path": model_path,
                "n_detections": len(regions),
            },
        )
