#!/usr/bin/env python3
"""
Enrich crop tiles with NMD (Nationella Marktäckedata) labels.

Fetches NMD raster data from DES openEO for each crop tile's bbox,
converts to 10-class LULC labels, and saves as 'nmd_label' in the .npz.

Usage:
    python scripts/enrich_tiles_nmd.py --data-dir data/crop_tiles
    python scripts/enrich_tiles_nmd.py --data-dir data/crop_tiles --workers 4
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imint.config.env import load_env
load_env()

from imint.fetch import fetch_nmd_data
from imint.training.class_schema import nmd_raster_to_lulc


def process_tile(tile_path: str, num_classes: int = 10) -> dict:
    """Fetch NMD label for a single crop tile and save it.

    Returns dict with status info.
    """
    name = os.path.basename(tile_path)
    try:
        data = dict(np.load(tile_path, allow_pickle=True))

        # Skip if already enriched
        if "nmd_label" in data:
            return {"name": name, "status": "skip", "msg": "already has nmd_label"}

        # Get bbox — crop tiles store as [west, south, east, north]
        bbox = data["bbox_3006"]
        if len(bbox) != 4:
            return {"name": name, "status": "error", "msg": f"bad bbox shape: {bbox.shape}"}

        west, south, east, north = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Crop tiles bbox is EPSG:3006 — convert to WGS84 for fetch_nmd_data
        from rasterio.crs import CRS
        from rasterio.warp import transform_bounds
        w84_w, w84_s, w84_e, w84_n = transform_bounds(
            CRS.from_epsg(3006), CRS.from_epsg(4326),
            west, south, east, north,
        )
        coords = {"west": w84_w, "south": w84_s, "east": w84_e, "north": w84_n}

        # Fetch NMD via DES openEO
        nmd_result = fetch_nmd_data(coords=coords, target_shape=(256, 256))

        if nmd_result is None or nmd_result.nmd_raster is None:
            return {"name": name, "status": "error", "msg": "NMD fetch returned None"}

        # Convert raw NMD codes to 10-class LULC
        nmd_label = nmd_raster_to_lulc(nmd_result.nmd_raster, num_classes=num_classes)

        # Save back
        data["nmd_label"] = nmd_label.astype(np.uint8)
        np.savez(tile_path, **data)

        unique = np.unique(nmd_label)
        return {"name": name, "status": "ok", "classes": unique.tolist()}

    except Exception as e:
        return {"name": name, "status": "error", "msg": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Enrich crop tiles with NMD labels")
    parser.add_argument("--data-dir", type=str, default="data/crop_tiles")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (1=sequential, safer for API)")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    tiles = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if args.limit > 0:
        tiles = tiles[:args.limit]

    print(f"Enriching {len(tiles)} tiles with NMD labels ({args.num_classes}-class)...")
    print(f"Workers: {args.workers}")

    ok, skip, error = 0, 0, 0
    t0 = time.time()

    if args.workers <= 1:
        # Sequential
        for i, tile_path in enumerate(tiles):
            result = process_tile(tile_path, args.num_classes)
            if result["status"] == "ok":
                ok += 1
            elif result["status"] == "skip":
                skip += 1
            else:
                error += 1
                print(f"  ERROR {result['name']}: {result.get('msg', '?')}")

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(tiles) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(tiles)}] ok={ok} skip={skip} err={error} "
                      f"({rate:.1f} tiles/s, ETA {eta:.0f}s)")
    else:
        # Parallel
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_tile, t, args.num_classes): t for t in tiles}
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                if result["status"] == "ok":
                    ok += 1
                elif result["status"] == "skip":
                    skip += 1
                else:
                    error += 1
                    print(f"  ERROR {result['name']}: {result.get('msg', '?')}")

                if (i + 1) % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"  [{i+1}/{len(tiles)}] ok={ok} skip={skip} err={error} "
                          f"({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s!")
    print(f"  OK: {ok}")
    print(f"  Skipped: {skip}")
    print(f"  Errors: {error}")


if __name__ == "__main__":
    main()
