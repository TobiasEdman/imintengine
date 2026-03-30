#!/usr/bin/env python3
"""
Enrich tiles with NMD labels from local GeoTIFF raster.

Much faster than openEO API (~1000 tiles/s vs 1 tile/15s).
Reads NMD codes directly from nmd2018bas_ogeneraliserad_v1_1.tif
using rasterio windowed reads.

Usage:
    python scripts/enrich_tiles_nmd_local.py --tiles-dir data/crop_tiles
    python scripts/enrich_tiles_nmd_local.py --tiles-dir data/lulc_seasonal/tiles
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from imint.training.class_schema import nmd_raster_to_lulc

NMD_RASTER = "data/nmd/nmd2018bas_ogeneraliserad_v1_1.tif"
TILE_SIZE = 256
TILE_M = 2560  # 256 * 10m


def get_tile_bbox(data: dict) -> tuple[float, float, float, float] | None:
    """Get [west, south, east, north] in EPSG:3006."""
    if "bbox_3006" in data:
        b = data["bbox_3006"]
        return float(b[0]), float(b[1]), float(b[2]), float(b[3])
    if "easting" in data and "northing" in data:
        e, n = float(data["easting"]), float(data["northing"])
        half = TILE_M / 2
        return e - half, n - half, e + half, n + half
    return None


def main():
    parser = argparse.ArgumentParser(description="Enrich tiles with NMD from local raster")
    parser.add_argument("--tiles-dir", type=str, required=True)
    parser.add_argument("--nmd-raster", type=str, default=NMD_RASTER)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Overwrite existing nmd_label")
    args = parser.parse_args()

    tiles = sorted(glob.glob(os.path.join(args.tiles_dir, "*.npz")))
    if args.limit > 0:
        tiles = tiles[:args.limit]

    print(f"Enriching {len(tiles)} tiles with NMD from {args.nmd_raster}")

    src = rasterio.open(args.nmd_raster)
    print(f"  NMD raster: {src.width}x{src.height}, CRS={src.crs}, bounds={src.bounds}")

    ok, skip, error, oob = 0, 0, 0, 0
    t0 = time.time()

    for i, tile_path in enumerate(tiles):
        name = os.path.basename(tile_path)
        try:
            data = dict(np.load(tile_path, allow_pickle=True))

            if "nmd_label" in data and not args.force:
                skip += 1
                continue

            bbox = get_tile_bbox(data)
            if bbox is None:
                error += 1
                continue

            west, south, east, north = bbox

            # Check if bbox is within NMD raster bounds
            if (west < src.bounds.left or east > src.bounds.right or
                south < src.bounds.bottom or north > src.bounds.top):
                oob += 1
                continue

            # Read NMD window
            window = from_bounds(west, south, east, north, src.transform)
            nmd_raw = src.read(1, window=window)

            # Resize to TILE_SIZE if needed
            if nmd_raw.shape != (TILE_SIZE, TILE_SIZE):
                from scipy.ndimage import zoom
                zy = TILE_SIZE / nmd_raw.shape[0]
                zx = TILE_SIZE / nmd_raw.shape[1]
                nmd_raw = zoom(nmd_raw, (zy, zx), order=0)  # nearest neighbor

            # Convert raw NMD codes to 10-class
            nmd_label = nmd_raster_to_lulc(nmd_raw, num_classes=args.num_classes)

            data["nmd_label"] = nmd_label.astype(np.uint8)
            np.savez(tile_path, **data)
            ok += 1

        except Exception as e:
            error += 1
            if error <= 3:
                print(f"  ERROR {name}: {e}")

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{len(tiles)}] ok={ok} skip={skip} oob={oob} err={error} "
                  f"({rate:.0f} tiles/s)")

    src.close()
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({ok / max(elapsed, 0.1):.0f} tiles/s)")
    print(f"  OK: {ok}")
    print(f"  Skipped: {skip}")
    print(f"  Out of bounds: {oob}")
    print(f"  Errors: {error}")


if __name__ == "__main__":
    main()
