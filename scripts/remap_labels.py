#!/usr/bin/env python3
"""Remap tile labels from one class schema to another.

Reads raw NMD codes directly from the .nmd_cache/ directory (no DES
connection needed) and re-applies nmd_raster_to_lulc() with the target
num_classes.

Usage:
    python scripts/remap_labels.py --data-dir data/lulc_seasonal_vpp --num-classes 12
    IMINT_NMD_WORKERS=6 python scripts/remap_labels.py ...  # more parallelism
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imint.training.class_schema import nmd_raster_to_lulc, get_class_names

# NMD 10m grid snapping — matches imint/fetch.py _to_nmd_grid()
_NMD_GRID = 10


def _snap_to_nmd_grid(w, s, e, n):
    """Snap EPSG:3006 bbox to NMD 10m grid boundaries."""
    return (
        math.floor(w / _NMD_GRID) * _NMD_GRID,
        math.floor(s / _NMD_GRID) * _NMD_GRID,
        math.ceil(e / _NMD_GRID) * _NMD_GRID,
        math.ceil(n / _NMD_GRID) * _NMD_GRID,
    )


def _nmd_cache_key(w84, s84, e84, n84):
    """Compute NMD cache key from WGS84 bbox (matches fetch.py)."""
    try:
        from rasterio.crs import CRS
        from rasterio.warp import transform_bounds

        w3006, s3006, e3006, n3006 = transform_bounds(
            CRS.from_epsg(4326), CRS.from_epsg(3006),
            w84, s84, e84, n84,
        )
    except ImportError:
        # Without rasterio, use easting/northing directly (caller provides)
        return None

    ws, ss, es, ns = _snap_to_nmd_grid(w3006, s3006, e3006, n3006)
    key_str = f"{ws:.6f}_{ss:.6f}_{es:.6f}_{ns:.6f}"
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def _nmd_cache_key_from_3006(easting, northing, half=1280.0):
    """Compute NMD cache key from EPSG:3006 center coordinates.

    Reconstructs the WGS84 bbox, projects back to 3006, snaps to grid,
    and hashes — matching the exact flow in fetch.py.
    """
    try:
        from rasterio.crs import CRS
        from rasterio.warp import transform_bounds

        w3006 = easting - half
        s3006 = northing - half
        e3006 = easting + half
        n3006 = northing + half

        # To WGS84
        w84, s84, e84, n84 = transform_bounds(
            CRS.from_epsg(3006), CRS.from_epsg(4326),
            w3006, s3006, e3006, n3006,
        )
        # Back to 3006 (this is what fetch.py does via _to_nmd_grid)
        w2, s2, e2, n2 = transform_bounds(
            CRS.from_epsg(4326), CRS.from_epsg(3006),
            w84, s84, e84, n84,
        )
        # Snap
        ws, ss, es, ns = _snap_to_nmd_grid(w2, s2, e2, n2)
        key_str = f"{ws:.6f}_{ss:.6f}_{es:.6f}_{ns:.6f}"
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    except ImportError:
        return None


def _resample_nearest(arr, target_shape):
    """Nearest-neighbor resample (matches fetch.py _resample_nearest)."""
    h_out, w_out = target_shape
    h_in, w_in = arr.shape
    row_idx = np.round(np.linspace(0, h_in - 1, h_out)).astype(int)
    col_idx = np.round(np.linspace(0, w_in - 1, w_out)).astype(int)
    return arr[np.ix_(row_idx, col_idx)]


def remap_tile(
    tile_path: Path,
    num_classes: int,
    nmd_cache_dir: Path,
    dry_run: bool = False,
) -> bool:
    """Remap a single tile's labels using raw NMD from cache.

    Returns True if the tile was successfully remapped.
    """
    data = dict(np.load(tile_path, allow_pickle=True))

    label = data.get("label")
    if label is None:
        return False

    easting = float(data.get("easting", 0))
    northing = float(data.get("northing", 0))
    if easting == 0 and northing == 0:
        return False

    # Compute NMD cache key
    cache_key = _nmd_cache_key_from_3006(easting, northing)
    if cache_key is None:
        return False

    cache_path = nmd_cache_dir / f"nmd_{cache_key}.npy"
    if not cache_path.exists():
        return False

    # Load raw NMD codes from cache
    nmd_raster = np.load(cache_path)

    # Resample to label shape if needed
    if nmd_raster.shape != label.shape:
        nmd_raster = _resample_nearest(nmd_raster, label.shape)

    new_labels = nmd_raster_to_lulc(nmd_raster, num_classes=num_classes)

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
    parser.add_argument("--nmd-cache", default=None, help="NMD cache dir (default: .nmd_cache/)")
    parser.add_argument("--workers", type=int, default=None, help="Parallel workers")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes only")
    args = parser.parse_args()

    tiles_dir = Path(args.data_dir) / "tiles"
    tile_files = sorted(tiles_dir.glob("tile_*.npz"))

    # Find NMD cache
    project_root = Path(__file__).resolve().parents[1]
    nmd_cache_dir = Path(args.nmd_cache) if args.nmd_cache else project_root / ".nmd_cache"
    n_cached = len(list(nmd_cache_dir.glob("nmd_*.npy"))) if nmd_cache_dir.exists() else 0

    n_workers = args.workers or int(os.environ.get("IMINT_NMD_WORKERS", "4"))

    print(f"Found {len(tile_files)} tiles to remap to {args.num_classes}-class schema")
    print(f"NMD cache: {nmd_cache_dir} ({n_cached} entries)")
    print(f"Workers: {n_workers}")
    print(f"Classes: {get_class_names(args.num_classes)}")

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
                success = future.result()
                if success:
                    ok += 1
                else:
                    fail += 1
            except Exception:
                fail += 1
            if (i + 1) % 200 == 0:
                print(f"  [{i+1}/{len(tile_files)}] remapped={ok} failed={fail}")

    print(f"Done: {ok} remapped, {fail} failed out of {len(tile_files)}")


if __name__ == "__main__":
    main()
