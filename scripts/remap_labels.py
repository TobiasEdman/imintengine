#!/usr/bin/env python3
"""Remap tile labels from one class schema to another.

Reads raw NMD codes from the .nmd_cache/ and re-applies
nmd_raster_to_lulc() with the target num_classes.

Usage:
    python scripts/remap_labels.py --data-dir data/lulc_seasonal_vpp --num-classes 12
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imint.fetch import fetch_nmd_data, FetchError
from imint.training.class_schema import nmd_raster_to_lulc, get_class_names


def remap_tile(tile_path: Path, num_classes: int, dry_run: bool = False) -> bool:
    """Remap a single tile's labels using raw NMD from cache.

    Returns True if the tile was successfully remapped.
    """
    data = dict(np.load(tile_path, allow_pickle=True))

    label = data.get("label")
    if label is None:
        return False

    # Reconstruct WGS84 bbox from stored lat/lon
    lat = float(data.get("lat", 0))
    lon = float(data.get("lon", 0))
    easting = float(data.get("easting", 0))
    northing = float(data.get("northing", 0))

    if lat == 0 and lon == 0:
        return False

    # Approximate WGS84 bbox from easting/northing (EPSG:3006)
    # Tile is 2560m x 2560m (256 px * 10m)
    half = 1280.0
    try:
        from rasterio.crs import CRS
        from rasterio.warp import transform_bounds

        w3006 = easting - half
        s3006 = northing - half
        e3006 = easting + half
        n3006 = northing + half

        w84, s84, e84, n84 = transform_bounds(
            CRS.from_epsg(3006), CRS.from_epsg(4326),
            w3006, s3006, e3006, n3006,
        )
        coords = {"west": w84, "south": s84, "east": e84, "north": n84}
    except ImportError:
        # Fallback: rough approximation
        dlat = 0.023  # ~2560m in latitude
        dlon = 0.046  # ~2560m in longitude at lat 60
        coords = {
            "west": lon - dlon / 2,
            "south": lat - dlat / 2,
            "east": lon + dlon / 2,
            "north": lat + dlat / 2,
        }

    target_shape = label.shape

    try:
        nmd_result = fetch_nmd_data(coords=coords, target_shape=target_shape)
    except (FetchError, Exception) as e:
        return False

    new_labels = nmd_raster_to_lulc(nmd_result.nmd_raster, num_classes=num_classes)

    # Check if labels actually changed
    if np.array_equal(label, new_labels):
        return True  # Already correct

    if dry_run:
        old_uniq = np.unique(label)
        new_uniq = np.unique(new_labels)
        print(f"  Would remap {tile_path.name}: {old_uniq} -> {new_uniq}")
        return True

    data["label"] = new_labels

    # Atomic write
    tmp = tile_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **data)
    tmp.rename(tile_path)
    return True


def main():
    parser = argparse.ArgumentParser(description="Remap tile labels to new class schema")
    parser.add_argument("--data-dir", required=True, help="Data directory")
    parser.add_argument("--num-classes", type=int, required=True, help="Target class count")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes only")
    args = parser.parse_args()

    tiles_dir = Path(args.data_dir) / "tiles"
    tile_files = sorted(tiles_dir.glob("tile_*.npz"))
    print(f"Found {len(tile_files)} tiles to remap to {args.num_classes}-class schema")
    print(f"Classes: {get_class_names(args.num_classes)}")

    if args.dry_run:
        # Just check first 10
        for f in tile_files[:10]:
            remap_tile(f, args.num_classes, dry_run=True)
        return

    ok = 0
    fail = 0
    for i, f in enumerate(tile_files):
        success = remap_tile(f, args.num_classes)
        if success:
            ok += 1
        else:
            fail += 1
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(tile_files)}] remapped={ok} failed={fail}")

    print(f"Done: {ok} remapped, {fail} failed out of {len(tile_files)}")


if __name__ == "__main__":
    main()
