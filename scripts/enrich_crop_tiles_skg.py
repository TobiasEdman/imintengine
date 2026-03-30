#!/usr/bin/env python3
"""
Enrich crop tiles with Skogliga grunddata (height, volume, basal_area, diameter).

LULC tiles already have these. Crop tiles are missing them.
Fetches from Skogsstyrelsen's ArcGIS ImageServer (~0.5s per tile).

Usage:
    python scripts/enrich_crop_tiles_skg.py --tiles-dir data/crop_tiles
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

from imint.training.skg_height import fetch_height_tile
from imint.training.skg_grunddata import (
    fetch_volume_tile,
    fetch_basal_area_tile,
    fetch_diameter_tile,
)

TILE_M = 2560


def get_bbox(data: dict) -> tuple[float, float, float, float] | None:
    if "bbox_3006" in data:
        b = data["bbox_3006"]
        return float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return None


def process_tile(tile_path: str) -> dict:
    name = os.path.basename(tile_path)
    try:
        data = dict(np.load(tile_path, allow_pickle=True))

        # Check what's missing
        needs = []
        if "height" not in data:
            needs.append("height")
        if "volume" not in data:
            needs.append("volume")
        if "basal_area" not in data:
            needs.append("basal_area")
        if "diameter" not in data:
            needs.append("diameter")

        if not needs:
            return {"name": name, "status": "skip"}

        bbox = get_bbox(data)
        if bbox is None:
            return {"name": name, "status": "error", "msg": "no bbox"}

        west, south, east, north = bbox

        for var in needs:
            try:
                if var == "height":
                    arr = fetch_height_tile(west, south, east, north, size_px=256)
                elif var == "volume":
                    arr = fetch_volume_tile(west, south, east, north, size_px=256)
                elif var == "basal_area":
                    arr = fetch_basal_area_tile(west, south, east, north, size_px=256)
                elif var == "diameter":
                    arr = fetch_diameter_tile(west, south, east, north, size_px=256)
                data[var] = arr.astype(np.float32)
            except Exception as e:
                # Fill with zeros if fetch fails (coastal/urban tiles)
                data[var] = np.zeros((256, 256), dtype=np.float32)

        np.savez(tile_path, **data)
        return {"name": name, "status": "ok", "fetched": needs}

    except Exception as e:
        return {"name": name, "status": "error", "msg": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Enrich crop tiles with SKG data")
    parser.add_argument("--tiles-dir", type=str, default="data/crop_tiles")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    tiles = sorted(glob.glob(os.path.join(args.tiles_dir, "*.npz")))
    if args.limit > 0:
        tiles = tiles[:args.limit]

    print(f"Enriching {len(tiles)} crop tiles with Skogliga grunddata...")
    ok, skip, error = 0, 0, 0
    t0 = time.time()

    for i, tile_path in enumerate(tiles):
        result = process_tile(tile_path)
        if result["status"] == "ok":
            ok += 1
        elif result["status"] == "skip":
            skip += 1
        else:
            error += 1
            if error <= 3:
                print(f"  ERROR {result['name']}: {result.get('msg', '?')}")

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{len(tiles)}] ok={ok} skip={skip} err={error} "
                  f"({rate:.1f} tiles/s)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s!")
    print(f"  OK: {ok}, Skip: {skip}, Error: {error}")


if __name__ == "__main__":
    main()
