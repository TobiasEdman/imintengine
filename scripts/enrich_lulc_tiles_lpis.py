#!/usr/bin/env python3
"""
Enrich LULC seasonal tiles with LPIS crop parcel segmentation masks.

LULC tiles have NMD labels (10-class) but NMD class 7 (cropland) has no
crop-type detail. This script overlays LPIS parcel data to add per-pixel
crop class (vete=1, korn=2, havre=3, ..., övrig=8) within NMD cropland.

Year matching:
  - LULC tiles have 2018/2019 spectral data
  - LPIS data available: 2022, 2023, 2024
  - We use 2022 LPIS as best available approximation
  - Parcel boundaries are fairly stable; crop types rotate annually
  - Flagged as lpis_approximate=True

Usage:
    python scripts/enrich_lulc_tiles_lpis.py --tiles-dir data/lulc_seasonal/tiles
    python scripts/enrich_lulc_tiles_lpis.py --tiles-dir data/lulc_seasonal/tiles --lpis-year 2022 --workers 4
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import the rasterization function from existing LPIS enrichment script
from scripts.enrich_tiles_lpis_mask import (
    get_lpis_gdf,
    rasterize_parcels,
    TILE_SIZE_PX,
)


def process_lulc_tile(
    tile_path: str,
    lpis_gdf,
    lpis_year: int,
    tile_size: int = TILE_SIZE_PX,
) -> dict:
    """Add LPIS mask to a single LULC tile.

    LULC tiles store coords as easting/northing (center of tile in EPSG:3006)
    and image shape (num_frames*6, H, W). We reconstruct the bbox from these.
    """
    name = os.path.basename(tile_path)
    try:
        data = dict(np.load(tile_path, allow_pickle=True))

        # Skip if already enriched
        if "label_mask" in data:
            return {"name": name, "status": "skip"}

        # Reconstruct bbox from easting/northing
        # LULC tiles are 256×256 at 10m resolution = 2560m × 2560m
        easting = int(data["easting"])
        northing = int(data["northing"])
        half = (tile_size * 10) // 2  # 1280m

        bbox_3006 = np.array([
            easting - half,       # west
            northing - half,      # south
            easting + half,       # east
            northing + half,      # north
        ], dtype=np.int64)

        # Rasterize LPIS parcels
        mask, n_parcels = rasterize_parcels(lpis_gdf, bbox_3006, tile_size)

        if n_parcels == 0:
            # No agricultural parcels in this tile — skip
            return {"name": name, "status": "no_parcels"}

        # Save enriched tile
        data["label_mask"] = mask.astype(np.uint8)
        data["lpis_year"] = np.int32(lpis_year)
        data["n_parcels"] = np.int32(n_parcels)
        data["lpis_approximate"] = np.bool_(False)  # Exact year match
        data["bbox_3006"] = bbox_3006  # Add bbox for unified dataset

        np.savez(tile_path, **data)

        crop_pct = float((mask > 0).mean() * 100)
        return {"name": name, "status": "ok", "n_parcels": n_parcels, "crop_pct": crop_pct}

    except Exception as e:
        return {"name": name, "status": "error", "msg": str(e)}


def _get_tile_year(tile_path: str) -> int:
    """Extract the dominant spectral year from a LULC tile's dates array."""
    try:
        d = np.load(tile_path, allow_pickle=True)
        dates = d.get("dates", [])
        from collections import Counter
        years = Counter()
        for dt in dates:
            s = str(dt).strip()
            if len(s) >= 4 and s[:4].isdigit():
                years[int(s[:4])] += 1
        if years:
            return years.most_common(1)[0][0]
    except Exception:
        pass
    return 0


def main():
    parser = argparse.ArgumentParser(description="Enrich LULC tiles with LPIS masks")
    parser.add_argument("--tiles-dir", type=str, default="data/lulc_seasonal/tiles")
    parser.add_argument("--lpis-dir", type=str, default="data/lpis")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    tiles = sorted(glob.glob(os.path.join(args.tiles_dir, "*.npz")))
    if args.limit > 0:
        tiles = tiles[:args.limit]

    from collections import Counter

    print(f"Enriching {len(tiles)} LULC tiles with LPIS year-matched masks...")

    # Scan tiles to find which years we need
    print("Scanning tile years...")
    tile_years = {}
    needed_years = set()
    for t in tiles:
        y = _get_tile_year(t)
        tile_years[t] = y
        if y > 0:
            needed_years.add(y)
    print(f"  Tile years: {dict(Counter(tile_years.values()))}")
    print(f"  Need LPIS for: {sorted(needed_years)}")

    # Preload all needed LPIS years
    lpis_gdfs = {}
    for year in sorted(needed_years):
        print(f"  Loading LPIS {year}...")
        gdf = get_lpis_gdf(year, args.lpis_dir)
        if gdf is not None:
            lpis_gdfs[year] = gdf
            print(f"    {len(gdf)} parcels")
        else:
            print(f"    NOT FOUND — tiles with year {year} will be skipped")

    ok, skip, no_parcels, no_lpis, error = 0, 0, 0, 0, 0
    t0 = time.time()

    for i, tile_path in enumerate(tiles):
        year = tile_years.get(tile_path, 0)

        if year not in lpis_gdfs:
            no_lpis += 1
            continue

        result = process_lulc_tile(tile_path, lpis_gdfs[year], year)

        if result["status"] == "ok":
            ok += 1
        elif result["status"] == "skip":
            skip += 1
        elif result["status"] == "no_parcels":
            no_parcels += 1
        else:
            error += 1
            if error <= 5:
                print(f"  ERROR {result['name']}: {result.get('msg', '?')}")

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(tiles) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(tiles)}] ok={ok} skip={skip} "
                  f"no_parcels={no_parcels} err={error} "
                  f"({rate:.0f} tiles/s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s!")
    print(f"  With LPIS crops: {ok}")
    print(f"  No ag parcels: {no_parcels}")
    print(f"  No LPIS year: {no_lpis}")
    print(f"  Already done: {skip}")
    print(f"  Errors: {error}")
    print(f"  Total: {ok + skip + no_parcels + no_lpis + error}")


if __name__ == "__main__":
    main()
