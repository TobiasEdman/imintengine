"""Generate showcase images for the Marine Commercial (Kalmarsund) tab.

Fetches Sentinel-2 data from DES for the Kalmar Strait area, runs
both YOLO and AI2 vessel detection models, generates multi-date
heatmaps, spectral indices, NMD overlay, and cloud analysis.

Usage:
    .venv/bin/python scripts/generate_marine_commercial_showcase.py

    DES credentials needed for Sentinel-2 fetching.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# ── Area: Helsingborg–Helsingör — narrowest part of Öresund ──────
COORDS = {
    "west":  12.560,
    "south": 56.010,
    "east":  12.700,
    "north": 56.060,
}
PRIMARY_DATE = "2025-07-15"
HEATMAP_START = "2025-07-01"
HEATMAP_END = "2025-07-31"


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate marine commercial showcase images (Kalmarsund)"
    )
    parser.add_argument(
        "--date", default=PRIMARY_DATE,
        help=f"Primary analysis date (default: {PRIMARY_DATE})",
    )
    parser.add_argument(
        "--cloud-threshold", type=float, default=0.3,
        help="Max cloud fraction (default: 0.3)",
    )
    parser.add_argument(
        "--skip-heatmap", action="store_true",
        help="Skip multi-date heatmap generation (faster for testing)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: outputs/showcase/marine_commercial)",
    )
    args = parser.parse_args()

    from imint.fetch import fetch_copernicus_data, fetch_vessel_heatmap
    from imint.analyzers.spectral import SpectralAnalyzer
    from imint.analyzers.nmd import NMDAnalyzer
    from imint.analyzers.marine_vessels import MarineVesselAnalyzer
    from imint.analyzers.ai2_vessels import AI2VesselAnalyzer
    from imint.exporters.export import (
        save_rgb_png,
        save_ndvi_clean_png,
        save_spectral_index_clean_png,
        save_nmd_overlay,
        save_vessel_overlay,
        save_ai2_vessel_overlay,
        save_regions_geojson,
    )

    out_dir = args.output_dir or str(
        PROJECT_ROOT / "outputs" / "showcase" / "marine_commercial"
    )
    os.makedirs(out_dir, exist_ok=True)

    date = args.date
    coords = COORDS.copy()

    # ── Step 1: Fetch Sentinel-2 ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Marine Commercial Showcase — {date}")
    print(f"  Area: {coords}")
    print(f"{'='*60}\n")

    print("[1/9] Fetching Sentinel-2 data from CDSE...")
    result = fetch_copernicus_data(
        date=date,
        coords=coords,
        cloud_threshold=args.cloud_threshold,
        date_window=5,
    )

    if result.cloud_fraction > args.cloud_threshold:
        print(
            f"  Cloud fraction {result.cloud_fraction:.1%} exceeds threshold "
            f"{args.cloud_threshold:.0%} — try a different date."
        )
        sys.exit(1)

    rgb = result.rgb
    bands = result.bands
    scl = result.scl
    geo = result.geo
    print(f"  RGB shape: {rgb.shape}")
    print(f"  Bands: {sorted(bands.keys())}")
    print(f"  Cloud fraction: {result.cloud_fraction:.1%}")

    # ── Step 2: Save RGB ─────────────────────────────────────────────
    print("\n[2/9] Saving RGB...")
    save_rgb_png(rgb, os.path.join(out_dir, "rgb.png"))

    # ── Step 3: Spectral, NMD ────────────────────────────────────────
    print("\n[3/9] Running spectral and NMD analyzers...")

    spectral = SpectralAnalyzer(config={"ndvi_threshold": 0.3, "ndwi_threshold": 0.3})
    spec_result = spectral.run(
        rgb, bands=bands, date=date, coords=coords,
        output_dir=out_dir, scl=scl, geo=geo,
    )
    indices = spec_result.outputs.get("indices", {})
    print(f"  Spectral: {spec_result.summary()}")

    ndvi = indices.get("NDVI")
    if ndvi is not None:
        save_ndvi_clean_png(ndvi, os.path.join(out_dir, "ndvi_clean.png"))

    ndwi = indices.get("NDWI")
    if ndwi is not None:
        save_spectral_index_clean_png(
            ndwi, os.path.join(out_dir, "ndwi_clean.png"),
            cmap_name="RdBu", vmin=-1, vmax=1,
        )

    nmd = NMDAnalyzer()
    nmd_result = nmd.run(
        rgb, bands=bands, date=date, coords=coords,
        output_dir=out_dir, scl=scl, geo=geo,
    )
    print(f"  NMD: {nmd_result.summary()}")

    # save_nmd_overlay takes (l2_raster, path) — use l2_raster for detailed classes
    l2_raster = nmd_result.outputs.get("l2_raster")
    if l2_raster is not None:
        save_nmd_overlay(l2_raster, os.path.join(out_dir, "nmd_overlay.png"))

    # ── Step 4: YOLO vessel detection ────────────────────────────────
    print("\n[4/9] Running YOLO marine vessel detection...")
    yolo = MarineVesselAnalyzer(config={
        "confidence": 0.286,
        "chip_size": 320,
        "overlap_ratio": 0.2,
        "water_filter": True,
        "max_bbox_m": 750,
    })
    yolo_result = yolo.run(rgb, bands=bands, scl=scl, geo=geo)
    yolo_regions = yolo_result.outputs.get("regions", [])
    print(f"  YOLO: {yolo_result.summary()}")
    print(f"  YOLO detections: {len(yolo_regions)}")

    save_vessel_overlay(rgb, yolo_regions, os.path.join(out_dir, "vessels_clean.png"))
    if yolo_regions:
        save_regions_geojson(
            yolo_regions, os.path.join(out_dir, "mc_vessels.geojson"),
        )

    # ── Step 5: AI2 vessel detection ─────────────────────────────────
    print("\n[5/9] Running AI2 vessel detection (Swin V2 B + attributes)...")
    ai2 = AI2VesselAnalyzer(config={
        "predict_attributes": True,
        "water_filter": True,
        "max_bbox_m": 750,
        "device": "cpu",
    })
    ai2_result = ai2.run(rgb, bands=bands, scl=scl, geo=geo)
    ai2_regions = ai2_result.outputs.get("regions", [])
    print(f"  AI2: {ai2_result.summary()}")
    print(f"  AI2 detections: {len(ai2_regions)}")

    save_ai2_vessel_overlay(
        rgb, ai2_regions, os.path.join(out_dir, "ai2_vessels_clean.png"),
    )

    # Flatten attributes into properties for GeoJSON popup support
    ai2_regions_flat = []
    for r in ai2_regions:
        flat = {k: v for k, v in r.items() if k != "attributes"}
        attrs = r.get("attributes", {})
        for ak, av in attrs.items():
            flat[ak] = av
        ai2_regions_flat.append(flat)

    if ai2_regions_flat:
        save_regions_geojson(
            ai2_regions_flat, os.path.join(out_dir, "mc_ai2_vessels.geojson"),
        )

    # ── Step 6: YOLO multi-date heatmap ──────────────────────────────
    yolo_heatmap_info = {}
    ai2_heatmap_info = {}

    if not args.skip_heatmap:
        print(f"\n[6/9] Generating YOLO heatmap ({HEATMAP_START} to {HEATMAP_END})...")
        yolo_heatmap_info = fetch_vessel_heatmap(
            coords=coords,
            date_start=HEATMAP_START,
            date_end=HEATMAP_END,
            output_dir=out_dir,
            cloud_threshold=args.cloud_threshold,
            analyzer_type="yolo",
            prefix="yolo_",
        )
        # Rename to expected filename
        yolo_hm = os.path.join(out_dir, "yolo_vessel_heatmap_clean.png")
        target_hm = os.path.join(out_dir, "vessel_heatmap_clean.png")
        if os.path.exists(yolo_hm):
            shutil.copy2(yolo_hm, target_hm)

        # ── Step 7: AI2 multi-date heatmap ───────────────────────────
        print(f"\n[7/9] Generating AI2 heatmap ({HEATMAP_START} to {HEATMAP_END})...")
        ai2_heatmap_info = fetch_vessel_heatmap(
            coords=coords,
            date_start=HEATMAP_START,
            date_end=HEATMAP_END,
            output_dir=out_dir,
            cloud_threshold=args.cloud_threshold,
            analyzer_type="ai2",
            predict_attributes=True,
            prefix="ai2_",
        )
        # Rename to expected filename
        ai2_hm = os.path.join(out_dir, "ai2_vessel_heatmap_clean.png")
        target_ai2_hm = os.path.join(out_dir, "ai2_vessel_heatmap_clean.png")
        if os.path.exists(ai2_hm) and ai2_hm != target_ai2_hm:
            shutil.copy2(ai2_hm, target_ai2_hm)
    else:
        print("\n[6/9] Skipped YOLO heatmap (--skip-heatmap)")
        print("[7/9] Skipped AI2 heatmap (--skip-heatmap)")

    # ── Step 8: Sjokort placeholder ──────────────────────────────────
    print("\n[8/9] Creating sjokort placeholder...")
    _create_sjokort_placeholder(rgb.shape[:2], os.path.join(out_dir, "sjokort.png"))

    # ── Step 9: Save summary + copy to docs ──────────────────────────
    print("\n[9/9] Saving summary and copying to docs...")

    summary = {
        "date": date,
        "coords": coords,
        "image_shape": list(rgb.shape),
        "cloud_fraction": float(result.cloud_fraction),
        "yolo_vessels": len(yolo_regions),
        "ai2_vessels": len(ai2_regions),
        "nmd_stats": nmd_result.outputs.get("class_stats", {}),
        "yolo_heatmap": yolo_heatmap_info,
        "ai2_heatmap": ai2_heatmap_info,
    }
    summary_path = os.path.join(out_dir, "showcase_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary: {summary_path}")

    # Copy to docs
    docs_showcase = PROJECT_ROOT / "docs" / "showcase" / "marine_commercial"
    docs_data = PROJECT_ROOT / "docs" / "data"
    os.makedirs(str(docs_showcase), exist_ok=True)
    os.makedirs(str(docs_data), exist_ok=True)

    # Copy PNGs
    for fname in [
        "rgb.png", "vessels_clean.png", "ai2_vessels_clean.png",
        "vessel_heatmap_clean.png", "ai2_vessel_heatmap_clean.png",
        "nmd_overlay.png", "ndvi_clean.png", "ndwi_clean.png",
        "sjokort.png",
    ]:
        src = os.path.join(out_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, str(docs_showcase / fname))
            print(f"  Copied: {fname}")

    # Copy GeoJSON
    for fname in ["mc_vessels.geojson", "mc_ai2_vessels.geojson"]:
        src = os.path.join(out_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, str(docs_data / fname))
            print(f"  Copied: {fname}")

    print(f"\n{'='*60}")
    print(f"  Done! Images: {docs_showcase}")
    print(f"  GeoJSON: {docs_data}")
    print(f"  Image dimensions: {rgb.shape[1]}w x {rgb.shape[0]}h")
    print(f"  Update tab-data.js with: imgW: {rgb.shape[1]}, imgH: {rgb.shape[0]}")
    print(f"{'='*60}\n")


def _create_sjokort_placeholder(shape: tuple, path: str) -> None:
    """Create a light-blue placeholder image for the sjokort toggle."""
    from PIL import Image, ImageDraw

    h, w = shape[:2]
    img = Image.new("RGBA", (w, h), (190, 215, 235, 255))
    draw = ImageDraw.Draw(img)
    text = "Sjokort\n(placeholder)"
    # Center the text
    try:
        bbox = draw.textbbox((0, 0), text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        tw, th = draw.textsize(text)
    x = (w - tw) // 2
    y = (h - th) // 2
    draw.text((x, y), text, fill=(100, 120, 140, 200))
    img.save(path)
    print(f"    saved: {path}")


if __name__ == "__main__":
    main()
