#!/usr/bin/env python3
"""Remap tile labels from one class schema to another.

Reads raw NMD codes directly from the .nmd_cache/ directory (no DES
connection needed) and re-applies nmd_raster_to_lulc() with the target
num_classes.

Uses the exact same coordinate flow as prepare_data.py to reconstruct
NMD cache keys: _sweref99_to_wgs84() → _to_nmd_grid() → _nmd_cache_key().

Usage:
    python scripts/remap_labels.py --data-dir data/lulc_seasonal_vpp --num-classes 12
    IMINT_NMD_WORKERS=6 python scripts/remap_labels.py ...  # more parallelism
"""
from __future__ import annotations

import os
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imint.fetch import _to_nmd_grid, _nmd_cache_key, _resample_nearest
from imint.training.sampler import _sweref99_to_wgs84
from imint.training.class_schema import nmd_raster_to_lulc, get_class_names


# NMD is fetched for the actual tile extent (256px × 10m = 2560m),
# not the full grid cell. The half-width is 1280m.
_TILE_HALF_M = 1280.0


def _tile_to_nmd_cache_path(
    easting: float,
    northing: float,
    nmd_cache_dir: Path,
) -> Path | None:
    """Compute the NMD cache file path for a tile.

    Replicates the exact flow from prepare_data.py:
      1. _sweref99_to_wgs84 to get WGS84 bbox (same as grid_to_wgs84)
      2. _to_nmd_grid: project WGS84 → 3006 via rasterio, snap to 10m
      3. _nmd_cache_key: hash the snapped coords
    """
    # Same corner logic as grid_to_wgs84: sw=(east,south), ne=(west,north)
    sw = _sweref99_to_wgs84(easting + _TILE_HALF_M, northing - _TILE_HALF_M)
    ne = _sweref99_to_wgs84(easting - _TILE_HALF_M, northing + _TILE_HALF_M)

    coords_wgs84 = {
        "west": min(sw[1], ne[1]),
        "south": min(sw[0], ne[0]),
        "east": max(sw[1], ne[1]),
        "north": max(sw[0], ne[0]),
    }

    projected = _to_nmd_grid(coords_wgs84)
    cache_key = _nmd_cache_key(projected)
    return nmd_cache_dir / ("nmd_%s.npy" % cache_key)


def remap_tile(
    tile_path: Path,
    num_classes: int,
    nmd_cache_dir: Path,
    dry_run: bool = False,
) -> str:
    """Remap a single tile's labels using raw NMD from cache.

    Returns 'ok', 'skip' (already correct), or 'fail'.
    """
    data = dict(np.load(tile_path, allow_pickle=True))

    label = data.get("label")
    if label is None:
        return "fail"

    easting = float(data.get("easting", 0))
    northing = float(data.get("northing", 0))
    if easting == 0 and northing == 0:
        return "fail"

    cache_path = _tile_to_nmd_cache_path(
        easting, northing, nmd_cache_dir,
    )
    if cache_path is None or not cache_path.exists():
        return "fail"

    nmd_raster = np.load(cache_path)

    # Resample to label shape if needed
    if nmd_raster.shape != label.shape:
        nmd_raster = _resample_nearest(nmd_raster, label.shape)

    new_labels = nmd_raster_to_lulc(nmd_raster, num_classes=num_classes)

    if np.array_equal(label, new_labels):
        return "skip"

    if dry_run:
        old_uniq = np.unique(label)
        new_uniq = np.unique(new_labels)
        print("  Would remap %s: %s -> %s" % (tile_path.name, old_uniq, new_uniq))
        return "ok"

    data["label"] = new_labels

    # Atomic write
    tmp = tile_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **data)
    tmp.rename(tile_path)
    return "ok"


def main():
    parser = argparse.ArgumentParser(description="Remap tile labels to new class schema")
    parser.add_argument("--data-dir", required=True, help="Data directory")
    parser.add_argument("--num-classes", type=int, required=True, help="Target class count")
    parser.add_argument("--nmd-cache", default=None, help="NMD cache dir (default: .nmd_cache/)")
    parser.add_argument("--workers", type=int, default=None, help="Parallel workers")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes only")
    args = parser.parse_args()

    tiles_dir = Path(args.data_dir) / "tiles"
    tile_files = sorted(tiles_dir.glob("tile_*.npz"))

    project_root = Path(__file__).resolve().parents[1]
    nmd_cache_dir = Path(args.nmd_cache) if args.nmd_cache else project_root / ".nmd_cache"
    n_cached = len(list(nmd_cache_dir.glob("nmd_*.npy"))) if nmd_cache_dir.exists() else 0

    n_workers = args.workers or int(os.environ.get("IMINT_NMD_WORKERS", "4"))

    print("Found %d tiles to remap to %d-class schema" % (len(tile_files), args.num_classes))
    print("NMD cache: %s (%d entries)" % (nmd_cache_dir, n_cached))
    print("Workers: %d" % n_workers)
    print("Classes: %s" % get_class_names(args.num_classes))

    if args.dry_run:
        for f in tile_files[:10]:
            remap_tile(f, args.num_classes, nmd_cache_dir, dry_run=True)
        return

    ok = 0
    fail = 0
    skip = 0

    def _process(tile_path):
        return remap_tile(tile_path, args.num_classes, nmd_cache_dir)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process, f): f for f in tile_files}
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                if result == "ok":
                    ok += 1
                elif result == "skip":
                    skip += 1
                else:
                    fail += 1
            except Exception:
                fail += 1
            if (i + 1) % 200 == 0:
                print("  [%d/%d] remapped=%d skip=%d failed=%d" % (
                    i + 1, len(tile_files), ok, skip, fail))

    print("Done: %d remapped, %d already correct, %d failed out of %d" % (
        ok, skip, fail, len(tile_files)))


if __name__ == "__main__":
    main()
