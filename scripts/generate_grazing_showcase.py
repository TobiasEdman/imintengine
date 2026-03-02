"""Generate showcase images for the Grazing (Betesmark) tab.

Fetches a single Sentinel-2 scene for a pasture area in Skåne,
runs spectral + NMD + COT analysis, overlays LPIS polygons, and
saves PNGs to outputs/showcase/grazing/.

Usage:
    DES_USER=testuser DES_PASSWORD=secretpassword \
    .venv/bin/python scripts/generate_grazing_showcase.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


def main():
    from imint.fetch import fetch_des_data, fetch_lpis_polygons, fetch_nmd_data
    from imint.analyzers.spectral import SpectralAnalyzer
    from imint.analyzers.cot import COTAnalyzer
    from imint.exporters.export import (
        save_rgb_png,
        save_ndvi_clean_png,
        save_spectral_index_clean_png,
        save_cot_clean_png,
        save_nmd_overlay,
        save_lpis_overlay,
        save_lpis_geojson,
    )

    # ── Area: pasture-rich zone near Lund, Skåne ──────────────────────
    coords = {
        "west": 13.42, "south": 55.935,
        "east": 13.48, "north": 55.965,
    }
    date = "2024-05-13"  # Known cloud-free date from our integration test

    out_dir = str(PROJECT_ROOT / "outputs" / "showcase" / "grazing")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("  Grazing Showcase Image Generator")
    print(f"  Area: {coords['west']:.3f}–{coords['east']:.3f}°E, "
          f"{coords['south']:.3f}–{coords['north']:.3f}°N")
    print(f"  Date: {date}")
    print(f"  Output: {out_dir}")
    print("=" * 60)

    # ── Step 1: Fetch S2 data ─────────────────────────────────────────
    print("\n[1] Fetching Sentinel-2 data from DES...")
    fetch_result = fetch_des_data(
        date=date,
        coords=coords,
        cloud_threshold=1.0,  # no cloud check — known clear
        include_scl=True,
        date_window=5,
    )
    print(f"    ✓ Shape: {fetch_result.rgb.shape}")
    print(f"    Cloud fraction: {fetch_result.cloud_fraction:.1%}")

    # ── Step 2: Save RGB ──────────────────────────────────────────────
    print("\n[2] Saving RGB...")
    save_rgb_png(fetch_result.rgb, os.path.join(out_dir, "rgb.png"))

    # ── Step 3: Spectral indices ──────────────────────────────────────
    print("\n[3] Running spectral analysis...")
    spectral = SpectralAnalyzer()
    spec_result = spectral.run(
        fetch_result.rgb,
        bands=fetch_result.bands,
        date=date,
        coords=coords,
        output_dir=out_dir,
    )
    indices = spec_result.outputs.get("indices", {})

    if "NDVI" in indices:
        save_ndvi_clean_png(indices["NDVI"], os.path.join(out_dir, "ndvi_clean.png"))
    if "NDWI" in indices:
        save_spectral_index_clean_png(
            indices["NDWI"], os.path.join(out_dir, "ndwi_clean.png"),
            cmap_name="RdBu", vmin=-1, vmax=1,
        )
    if "EVI" in indices:
        save_spectral_index_clean_png(
            indices["EVI"], os.path.join(out_dir, "evi_clean.png"),
            cmap_name="RdYlGn", vmin=-0.5, vmax=1,
        )

    # ── Step 4: COT ───────────────────────────────────────────────────
    print("\n[4] Running COT analysis...")
    try:
        cot = COTAnalyzer()
        cot_result = cot.run(
            fetch_result.rgb,
            bands=fetch_result.bands,
            date=date,
            coords=coords,
            output_dir=out_dir,
        )
        cot_map = cot_result.outputs.get("cot_map")
        if cot_map is not None:
            save_cot_clean_png(cot_map, os.path.join(out_dir, "cot_clean.png"))
    except Exception as e:
        print(f"    COT skipped: {e}")

    # ── Step 5: NMD ───────────────────────────────────────────────────
    print("\n[5] Fetching NMD land cover...")
    try:
        nmd_result = fetch_nmd_data(
            coords=coords,
            target_shape=fetch_result.rgb.shape[:2],
        )
        if nmd_result is not None:
            save_nmd_overlay(nmd_result.nmd_raster, os.path.join(out_dir, "nmd_overlay.png"))
    except Exception as e:
        print(f"    NMD skipped: {e}")

    # ── Step 6: LPIS overlay ──────────────────────────────────────────
    print("\n[6] Fetching LPIS pasture polygons...")
    try:
        lpis_gdf = fetch_lpis_polygons(
            bbox=coords,
            agoslag="Bete",
        )
        if len(lpis_gdf) > 0 and fetch_result.geo is not None:
            save_lpis_overlay(
                fetch_result.rgb,
                lpis_gdf,
                fetch_result.geo,
                os.path.join(out_dir, "lpis_overlay.png"),
            )
            # Vector GeoJSON for Leaflet CRS.Simple rendering
            save_lpis_geojson(
                lpis_gdf,
                fetch_result.geo,
                os.path.join(out_dir, "lpis_polygons.json"),
                img_shape=fetch_result.rgb.shape[:2],
            )
            print(f"    ✓ {len(lpis_gdf)} pasture polygons overlaid")
        else:
            print("    No LPIS polygons found or no geo context")
    except Exception as e:
        print(f"    LPIS overlay skipped: {e}")

    # ── Step 7: Save metadata ─────────────────────────────────────────
    import json
    meta = {
        "date": date,
        "coords": coords,
        "shape": list(fetch_result.rgb.shape[:2]),
        "cloud_fraction": fetch_result.cloud_fraction,
        "lpis_count": len(lpis_gdf) if 'lpis_gdf' in dir() else 0,
    }
    with open(os.path.join(out_dir, "grazing_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  Done! Images saved to {out_dir}")
    files = [f for f in os.listdir(out_dir) if f.endswith(".png")]
    print(f"  PNGs: {', '.join(sorted(files))}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
