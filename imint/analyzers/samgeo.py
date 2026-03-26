"""
imint/analyzers/samgeo.py — SAMGeo (Segment Anything for Geospatial) analyzer

Zero-shot segmentation using Meta's Segment Anything Model (SAM/SAM2)
applied to satellite imagery via the segment-geospatial package.

No training data required — the model segments objects from:
  - Automatic mode: discovers all segments in the image
  - Point prompts: segment around specified (x, y) pixel coords
  - Text prompts: segment objects matching a text description (requires
    Grounding DINO)

This enables rapid prototyping: point at a forest, lake, or building
and get an instant segmentation mask — no fine-tuning needed.

Requirements:
    pip install segment-geospatial

Config options:
    mode: "automatic" (default), "points", or "text"
    model_type: "vit_h" (default, best quality), "vit_l", "vit_b"
    points: [[x1,y1], [x2,y2], ...] — pixel coords for point prompts
    text_prompt: "buildings" — for text-prompted segmentation
    min_mask_area: 100 — minimum segment size in pixels (filters noise)
    device: "cpu", "cuda", or "mps" (auto-detected if omitted)
"""
from __future__ import annotations

import numpy as np
from .base import BaseAnalyzer, AnalysisResult


def _check_samgeo_available() -> bool:
    """Check if segment-geospatial is installed."""
    try:
        import samgeo  # noqa: F401
        return True
    except ImportError:
        return False


def _check_torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _get_device(preferred: str | None = None) -> str:
    """Auto-detect best available device."""
    import torch
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SAMGeoAnalyzer(BaseAnalyzer):
    """Segment Anything Model for geospatial data.

    Zero-shot segmentation — no training required. Discovers objects
    in satellite imagery automatically or from prompts.

    Config options:
        mode: "automatic" (default), "points", or "text"
        model_type: "vit_h" (default), "vit_l", "vit_b"
        sam_version: 2 (default, SAM2) or 1 (original SAM)
        points: [[x1,y1], [x2,y2]] — pixel coords for point prompts
        point_labels: [1, 1] — 1=foreground, 0=background per point
        text_prompt: str — for text-prompted segmentation
        min_mask_area: int — minimum segment size in pixels (default 100)
        device: "cpu", "cuda", or "mps" (auto-detected)
    """

    name = "samgeo"

    def analyze(self, rgb, bands=None, date=None, coords=None,
                output_dir="outputs", geo=None):
        if not _check_torch_available():
            return AnalysisResult(
                analyzer=self.name, success=False,
                error="PyTorch is not installed. Install with: pip install torch",
            )

        if not _check_samgeo_available():
            return AnalysisResult(
                analyzer=self.name, success=False,
                error=(
                    "segment-geospatial is not installed. "
                    "Install with: pip install segment-geospatial"
                ),
            )

        mode = self.config.get("mode", "automatic")

        if mode == "automatic":
            return self._run_automatic(rgb, output_dir)
        elif mode == "points":
            return self._run_points(rgb, output_dir)
        elif mode == "text":
            return self._run_text(rgb, output_dir)
        else:
            return AnalysisResult(
                analyzer=self.name, success=False,
                error=f"Unknown mode '{mode}'. Use: automatic, points, or text",
            )

    def _run_automatic(self, rgb: np.ndarray, output_dir: str) -> AnalysisResult:
        """Auto-segment all objects in the image."""
        import os
        import torch
        from PIL import Image

        device = _get_device(self.config.get("device"))
        model_type = self.config.get("model_type", "vit_h")
        min_mask_area = self.config.get("min_mask_area", 100)
        sam_version = self.config.get("sam_version", 2)

        # Save RGB as temporary image for SAMGeo input
        os.makedirs(output_dir, exist_ok=True)
        tmp_path = os.path.join(output_dir, "_samgeo_input.png")
        img = (rgb * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
        Image.fromarray(img).save(tmp_path)

        try:
            if sam_version == 2:
                from samgeo.samgeo import SamGeo2 as SamGeo
            else:
                from samgeo.samgeo import SamGeo
        except ImportError:
            from samgeo import SamGeo

        # Initialize model
        sam = SamGeo(
            model_type=model_type,
            device=device,
            automatic=True,
        )

        # Generate masks
        mask_path = os.path.join(output_dir, "_samgeo_masks.tif")
        sam.generate(tmp_path, output=mask_path)

        # Load result mask
        try:
            import rasterio
            with rasterio.open(mask_path) as src:
                seg_mask = src.read(1)
        except Exception:
            # Fallback: try loading as numpy
            seg_mask = np.zeros(rgb.shape[:2], dtype=np.int32)

        # Filter small segments
        if min_mask_area > 0:
            unique_ids = np.unique(seg_mask)
            for uid in unique_ids:
                if uid == 0:
                    continue
                if np.sum(seg_mask == uid) < min_mask_area:
                    seg_mask[seg_mask == uid] = 0

        # Compute segment statistics
        unique_ids, counts = np.unique(seg_mask, return_counts=True)
        n_segments = int(len(unique_ids[unique_ids > 0]))
        total_pixels = seg_mask.size
        segment_stats = {
            "n_segments": n_segments,
            "total_pixels": total_pixels,
            "segmented_fraction": round(
                float(np.sum(seg_mask > 0)) / total_pixels, 4
            ),
            "largest_segment_pixels": int(counts[1:].max()) if n_segments > 0 else 0,
            "median_segment_pixels": int(np.median(counts[1:])) if n_segments > 0 else 0,
        }

        # Cleanup temp files
        for tmp in [tmp_path, mask_path]:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

        return AnalysisResult(
            analyzer=self.name,
            success=True,
            outputs={
                "seg_mask": seg_mask,
                "segment_stats": segment_stats,
            },
            metadata={
                "mode": "automatic",
                "model_type": model_type,
                "sam_version": sam_version,
                "device": device,
                "n_segments": n_segments,
                "min_mask_area": min_mask_area,
                "image_size": list(rgb.shape[:2]),
            },
        )

    def _run_points(self, rgb: np.ndarray, output_dir: str) -> AnalysisResult:
        """Segment regions around specified point prompts."""
        import os
        import torch
        from PIL import Image

        points = self.config.get("points")
        if not points:
            return AnalysisResult(
                analyzer=self.name, success=False,
                error="Point mode requires 'points' config: [[x1,y1], [x2,y2], ...]",
            )

        point_labels = self.config.get("point_labels", [1] * len(points))
        device = _get_device(self.config.get("device"))
        model_type = self.config.get("model_type", "vit_h")
        sam_version = self.config.get("sam_version", 2)

        os.makedirs(output_dir, exist_ok=True)
        tmp_path = os.path.join(output_dir, "_samgeo_input.png")
        img = (rgb * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
        Image.fromarray(img).save(tmp_path)

        try:
            if sam_version == 2:
                from samgeo.samgeo import SamGeo2 as SamGeo
            else:
                from samgeo.samgeo import SamGeo
        except ImportError:
            from samgeo import SamGeo

        sam = SamGeo(
            model_type=model_type,
            device=device,
            automatic=False,
        )
        sam.set_image(tmp_path)

        # Predict from point prompts
        point_coords = np.array(points)
        point_labels_np = np.array(point_labels)

        masks, scores, logits = sam.predict(
            point_coords=point_coords,
            point_labels=point_labels_np,
            multimask_output=True,
        )

        # Take highest-confidence mask
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx].astype(np.uint8)
        best_score = float(scores[best_idx])

        # Cleanup
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        return AnalysisResult(
            analyzer=self.name,
            success=True,
            outputs={
                "seg_mask": best_mask,
                "confidence": best_score,
                "all_masks": masks,
                "all_scores": [float(s) for s in scores],
            },
            metadata={
                "mode": "points",
                "model_type": model_type,
                "sam_version": sam_version,
                "device": device,
                "n_points": len(points),
                "points": points,
                "best_score": best_score,
                "image_size": list(rgb.shape[:2]),
            },
        )

    def _run_text(self, rgb: np.ndarray, output_dir: str) -> AnalysisResult:
        """Segment objects matching a text description (requires Grounding DINO)."""
        import os
        from PIL import Image

        text_prompt = self.config.get("text_prompt")
        if not text_prompt:
            return AnalysisResult(
                analyzer=self.name, success=False,
                error="Text mode requires 'text_prompt' config, e.g. 'buildings'",
            )

        device = _get_device(self.config.get("device"))
        model_type = self.config.get("model_type", "vit_h")

        os.makedirs(output_dir, exist_ok=True)
        tmp_path = os.path.join(output_dir, "_samgeo_input.png")
        img = (rgb * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
        Image.fromarray(img).save(tmp_path)

        try:
            from samgeo.text_sam import LangSAM
        except ImportError:
            return AnalysisResult(
                analyzer=self.name, success=False,
                error=(
                    "Text-prompted segmentation requires LangSAM. "
                    "Install with: pip install segment-geospatial groundingdino-py"
                ),
            )

        sam = LangSAM(model_type=model_type, device=device)

        mask_path = os.path.join(output_dir, "_samgeo_text_mask.tif")
        sam.predict(tmp_path, text_prompt, box_threshold=0.24, text_threshold=0.24)

        # Extract masks from prediction
        masks = sam.prediction.masks if hasattr(sam, "prediction") else None

        if masks is not None and len(masks) > 0:
            # Combine all detected masks into one
            combined = np.zeros(rgb.shape[:2], dtype=np.uint8)
            for i, m in enumerate(masks):
                mask_np = m.cpu().numpy() if hasattr(m, "cpu") else np.array(m)
                combined[mask_np > 0] = i + 1
            seg_mask = combined
            n_objects = int(len(masks))
        else:
            seg_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
            n_objects = 0

        # Cleanup
        for tmp in [tmp_path, mask_path]:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

        return AnalysisResult(
            analyzer=self.name,
            success=True,
            outputs={
                "seg_mask": seg_mask,
                "n_objects": n_objects,
            },
            metadata={
                "mode": "text",
                "model_type": model_type,
                "device": device,
                "text_prompt": text_prompt,
                "n_objects": n_objects,
                "image_size": list(rgb.shape[:2]),
            },
        )
