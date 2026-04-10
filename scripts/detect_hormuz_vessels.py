#!/usr/bin/env python3
"""Detect marine vessels at the Strait of Hormuz using YOLO11s + CDSE S2 data.

Fetches Sentinel-2 L2A reflectance bands (B02, B03, B04) + SCL from CDSE
via Sentinel Hub Process API, applies TCI formula, runs MarineVesselAnalyzer
with SAHI sliding window, and saves annotated image + detection JSON.

Output: ../leo-constellation/simulations/hormuz_s2_detections.jpg
        ../leo-constellation/simulations/hormuz_s2_detections.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Add ImintEngine to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imint.training.cdse_s2 import fetch_s2_scene_wgs84

# ── Config ────────────────────────────────────────────────────────
# Hormuz AOI in WGS84
BBOX = [54.0, 25.0, 58.0, 27.5]  # west, south, east, north
WIDTH, HEIGHT = 2048, 1280
DATE = "2026-02-26"  # Best coverage scene

# Output paths
LEO_DIR = Path(__file__).resolve().parent.parent.parent / "leo-constellation" / "simulations"
OUT_IMG = LEO_DIR / "hormuz_s2_detections.jpg"
OUT_JSON = LEO_DIR / "hormuz_s2_detections.json"
OUT_TCI = LEO_DIR / "hormuz_s2.jpg"  # Also update the globe overlay

# Detection config
CONFIDENCE = 0.15
CHIP_SIZE = 320
OVERLAP = 0.3


def main():
    # ── 1. Fetch L2A reflectance data via fetch_s2_scene_wgs84 ─
    west, south, east, north = BBOX
    print(f"Fetching S2 L2A for Hormuz ({DATE}) via fetch_s2_scene_wgs84...")
    result = fetch_s2_scene_wgs84(
        west, south, east, north,
        date=DATE,
        size_px=(HEIGHT, WIDTH),
        cloud_threshold=0.30,
    )
    if result is None:
        print("Fetch returned None (cloud/haze/nodata). Try a different date.")
        sys.exit(1)

    spectral, scl, cloud_frac = result
    # spectral: (6, H, W) — bands [B02, B03, B04, B8A, B11, B12]
    # Extract B04(red=idx2), B03(green=idx1), B02(blue=idx0) for TCI
    rgb_reflectance = np.stack([spectral[2], spectral[1], spectral[0]], axis=-1)  # (H, W, 3)

    print(f"  Shape: {rgb_reflectance.shape}, cloud: {cloud_frac:.1%}")
    print(f"  Reflectance range: {rgb_reflectance.min():.4f} - {rgb_reflectance.max():.4f}")

    # ── Save full-resolution spectral TIFF + SCL ──────────────
    try:
        import tifffile
        spectral_path = LEO_DIR / f"hormuz_s2_{DATE}_spectral.tif"
        scl_path = LEO_DIR / f"hormuz_s2_{DATE}_scl.tif"
        tifffile.imwrite(str(spectral_path), spectral)  # (6, H, W) float32
        tifffile.imwrite(str(scl_path), scl)             # (H, W) uint8
        print(f"  Saved spectral: {spectral_path} ({spectral_path.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"  Saved SCL: {scl_path} ({scl_path.stat().st_size / 1024:.0f} KB)")
        print(f"  Bands: B02, B03, B04, B8A, B11, B12 (float32 reflectance [0,1])")
    except ImportError:
        print("  tifffile not available — skipping spectral TIFF save")

    # ── 2. Apply TCI formula ──────────────────────────────────
    # TCI = clip(reflectance × 2.5 × 255, 0, 255) as uint8
    # This matches what the YOLO11s model was trained on
    tci = (rgb_reflectance * 2.5 * 255).clip(0, 255).astype(np.uint8)
    print(f"TCI image: {tci.shape}, dtype: {tci.dtype}")

    # Save TCI as the globe overlay too
    tci_pil = Image.fromarray(tci)
    tci_pil.save(OUT_TCI, quality=92)
    print(f"Saved TCI: {OUT_TCI}")

    # ── 3. Run MarineVesselAnalyzer ───────────────────────────
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    model_path = str(Path(__file__).parent.parent / "imint" / "fm" / "marine_vessels" / "yolo11s_tci.pt")
    print(f"\nLoading model: {Path(model_path).name}")

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=CONFIDENCE,
        device="cpu",
    )

    print(f"Running SAHI detection (chip={CHIP_SIZE}, overlap={OVERLAP}, conf>{CONFIDENCE})...")
    result = get_sliced_prediction(
        tci,
        detection_model,
        slice_height=CHIP_SIZE,
        slice_width=CHIP_SIZE,
        overlap_height_ratio=OVERLAP,
        overlap_width_ratio=OVERLAP,
        verbose=1,
    )

    # ── 4. Parse detections ───────────────────────────────────
    detections = []
    for pred in result.object_prediction_list:
        bbox = pred.bbox
        x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
        w, h = x2 - x1, y2 - y1

        # Size filter: skip huge false positives (> 100px = ~1km at 10m res)
        if w > 100 or h > 100:
            continue

        # Water filter using SCL if available
        if scl is not None:
            scl_2d = scl.squeeze() if scl.ndim > 2 else scl
            cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
            if 0 <= cy < scl_2d.shape[0] and 0 <= cx < scl_2d.shape[1]:
                if scl_2d[cy, cx] != 6:  # Not water
                    continue

        detections.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "score": round(float(pred.score.value), 3),
            "label": pred.category.name if pred.category else "vessel",
        })

    print(f"\nDetected {len(detections)} vessels (after size + water filter)")

    # ── 5. Draw bounding boxes ────────────────────────────────
    draw = ImageDraw.Draw(tci_pil)

    for d in detections:
        score = d["score"]
        if score >= 0.5:
            color = (0, 255, 255)     # cyan
        elif score >= 0.3:
            color = (255, 255, 0)     # yellow
        else:
            color = (255, 165, 0)     # orange

        # Draw box with padding for visibility
        pad = 3
        draw.rectangle(
            [d["x1"] - pad, d["y1"] - pad, d["x2"] + pad, d["y2"] + pad],
            outline=color, width=2,
        )
        # Score label
        draw.text((d["x1"] - pad, d["y1"] - pad - 10), f"{score:.2f}", fill=color)

    tci_pil.save(OUT_IMG, quality=95)
    print(f"Saved: {OUT_IMG}")

    # ── 6. Save metadata ──────────────────────────────────────
    meta = {
        "date": DATE,
        "bbox_wgs84": BBOX,
        "size_px": [WIDTH, HEIGHT],
        "model": "yolo11s_tci.pt (mayrajeo/marine-vessel-yolo)",
        "tci_formula": "clip(reflectance × 2.5 × 255, 0, 255)",
        "confidence_threshold": CONFIDENCE,
        "chip_size": CHIP_SIZE,
        "overlap": OVERLAP,
        "water_filter": scl is not None,
        "total_detections": len(detections),
        "detections": detections,
    }
    OUT_JSON.write_text(json.dumps(meta, indent=2))
    print(f"Saved: {OUT_JSON}")

    if detections:
        scores = [d["score"] for d in detections]
        print(f"\nStats: {len(detections)} vessels, score {min(scores):.3f}-{max(scores):.3f}, mean {np.mean(scores):.3f}")


if __name__ == "__main__":
    main()
