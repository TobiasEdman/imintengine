"""Regenerate Marine Leisure (Bohuslän) showcase — vessel analysis.

Fetches Sentinel-2 data from CDSE Sentinel Hub HTTP API (no openEO),
runs YOLO vessel detection, and generates both raster overlay and
GeoJSON from the same analysis run to ensure coordinate consistency.

Also regenerates RGB, spectral indices, NMD, and COT.

Usage:
    .venv/bin/python scripts/generate_marine_leisure_showcase.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Bohuslän — Hunnebostrand archipelago ─────────────────────────
COORDS_WGS84 = {
    "west":  11.25049,
    "south": 58.42763,
    "east":  11.30049,
    "north": 58.47763,
}
PRIMARY_DATE = "2025-07-10"


def main():
    import argparse
    from imint.config.env import load_env
    load_env("dev")

    parser = argparse.ArgumentParser(
        description="Regenerate Bohuslän marine leisure showcase"
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
        "--output-dir", default=None,
        help="Output directory (default: outputs/showcase/marine)",
    )
    args = parser.parse_args()

    from imint.fetch import _to_nmd_grid
    from imint.training.cdse_s2 import fetch_s2_scene
    from imint.analyzers.spectral import SpectralAnalyzer
    from imint.analyzers.nmd import NMDAnalyzer
    from imint.analyzers.cot import COTAnalyzer
    from imint.analyzers.marine_vessels import MarineVesselAnalyzer
    from imint.exporters.export import (
        save_rgb_png,
        save_ndvi_clean_png,
        save_spectral_index_clean_png,
        save_cot_clean_png,
        save_nmd_overlay,
        save_vessel_overlay,
        save_regions_geojson,
    )
    from imint.utils import bands_to_rgb

    out_dir = args.output_dir or str(
        PROJECT_ROOT / "outputs" / "showcase" / "marine"
    )
    os.makedirs(out_dir, exist_ok=True)

    date = args.date

    # ── Convert WGS84 → EPSG:3006 for Sentinel Hub ───────────────
    proj = _to_nmd_grid(COORDS_WGS84)
    west, south, east, north = proj["west"], proj["south"], proj["east"], proj["north"]
    w_px = int((east - west) / 10)
    h_px = int((north - south) / 10)

    print(f"\n{'='*60}")
    print(f"  Marine Leisure Showcase — Bohuslän — {date}")
    print(f"  WGS84: {COORDS_WGS84}")
    print(f"  EPSG:3006: W={west} S={south} E={east} N={north}")
    print(f"  Image size: {w_px}x{h_px} px (10m)")
    print(f"{'='*60}\n")

    # ── Step 1: Fetch Sentinel-2 via CDSE HTTP ────────────────────
    print("[1/7] Fetching Sentinel-2 data from CDSE...")
    result = fetch_s2_scene(west, south, east, north, date,
                            size_px=(h_px, w_px),
                            cloud_threshold=args.cloud_threshold)
    if result is None:
        print("  Fetch failed or too cloudy — try a different date.")
        sys.exit(1)

    spectral, scl, cloud_frac = result
    print(f"  Shape: {spectral.shape}")
    print(f"  Cloud fraction: {cloud_frac:.1%}")

    # Build bands dict from (6, H, W) spectral array
    band_names = ["B02", "B03", "B04", "B8A", "B11", "B12"]
    bands = {name: spectral[i] for i, name in enumerate(band_names)}

    # Build RGB
    rgb = bands_to_rgb(bands, scl=scl)
    print(f"  RGB shape: {rgb.shape}")

    # ── Step 2: Save RGB ──────────────────────────────────────────
    print("\n[2/7] Saving RGB...")
    save_rgb_png(rgb, os.path.join(out_dir, "rgb.png"))

    # ── Step 3: Spectral indices ──────────────────────────────────
    print("\n[3/7] Running spectral analyzer...")
    spectral_analyzer = SpectralAnalyzer(config={"ndvi_threshold": 0.3, "ndwi_threshold": 0.3})
    spec_result = spectral_analyzer.run(
        rgb, bands=bands, date=date, coords=COORDS_WGS84,
        output_dir=out_dir, scl=scl,
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

    # ── Step 4: NMD ───────────────────────────────────────────────
    print("\n[4/7] Running NMD analyzer...")
    nmd = NMDAnalyzer()
    nmd_result = nmd.run(
        rgb, bands=bands, date=date, coords=COORDS_WGS84,
        output_dir=out_dir, scl=scl,
    )
    print(f"  NMD: {nmd_result.summary()}")

    l2_raster = nmd_result.outputs.get("l2_raster")
    if l2_raster is not None:
        save_nmd_overlay(l2_raster, os.path.join(out_dir, "nmd_overlay.png"))

    # ── Step 5: COT ───────────────────────────────────────────────
    print("\n[5/7] Running COT analyzer...")
    cot_analyzer = COTAnalyzer(config={"device": "cpu"})
    cot_result = cot_analyzer.run(
        rgb, bands=bands, date=date, coords=COORDS_WGS84,
        output_dir=out_dir, scl=scl,
    )
    print(f"  COT: {cot_result.summary()}")

    cot_map = cot_result.outputs.get("cot_map")
    if cot_map is not None:
        save_cot_clean_png(cot_map, os.path.join(out_dir, "cot_clean.png"))

    # ── Step 6: YOLO vessel detection ─────────────────────────────
    print("\n[6/7] Running YOLO marine vessel detection...")
    yolo = MarineVesselAnalyzer(config={
        "confidence": 0.286,
        "chip_size": 320,
        "overlap_ratio": 0.2,
        "water_filter": True,
        "max_bbox_m": 750,
    })
    yolo_result = yolo.run(rgb, bands=bands, scl=scl)
    yolo_regions = yolo_result.outputs.get("regions", [])
    print(f"  YOLO: {yolo_result.summary()}")
    print(f"  Detections: {len(yolo_regions)}")

    # Save BOTH raster overlay AND GeoJSON from the SAME analysis run
    save_vessel_overlay(rgb, yolo_regions, os.path.join(out_dir, "vessels_clean.png"))
    if yolo_regions:
        save_regions_geojson(
            yolo_regions, os.path.join(out_dir, "vessels.geojson"),
        )
    else:
        print("  WARNING: No detections — vessels.geojson not written")

    # ── Step 7: Copy to docs ──────────────────────────────────────
    print("\n[7/7] Copying outputs to docs/...")
    docs_showcase = str(PROJECT_ROOT / "docs" / "showcase" / "marine")
    docs_data = str(PROJECT_ROOT / "docs" / "data")
    os.makedirs(docs_showcase, exist_ok=True)
    os.makedirs(docs_data, exist_ok=True)

    file_map = {
        "rgb.png": docs_showcase,
        "vessels_clean.png": docs_showcase,
        "ndvi_clean.png": docs_showcase,
        "ndwi_clean.png": docs_showcase,
        "nmd_overlay.png": docs_showcase,
        "cot_clean.png": docs_showcase,
        "vessels.geojson": docs_data,
    }

    for filename, dest_dir in file_map.items():
        src = os.path.join(out_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest_dir, filename))
            print(f"  {filename} → {dest_dir}/")
        else:
            print(f"  SKIP {filename} (not generated)")

    # Print image dimensions for tab-data.js reference
    from PIL import Image
    rgb_path = os.path.join(docs_showcase, "rgb.png")
    if os.path.exists(rgb_path):
        with Image.open(rgb_path) as im:
            w, h = im.size
            print(f"\n  Image dimensions: {w}x{h} (imgW: {w}, imgH: {h})")

    print("\nDone!")


if __name__ == "__main__":
    main()
