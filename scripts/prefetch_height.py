#!/usr/bin/env python3
"""Prefetch Skogsstyrelsen tree height for all existing training tiles.

Scans an existing tile directory (data/lulc_full/tiles/) and adds a
``height`` array to every .npz tile that doesn't already have one.
Runs independently of the main training pipeline so height data can be
downloaded in the background while training continues.

Usage:
    python scripts/prefetch_height.py [--data-dir data/lulc_full]
                                       [--workers 4]
                                       [--patch-size-m 2560]

The script is fully resumable — tiles that already contain a ``height``
key are skipped, and progress is printed every 50 tiles.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import numpy as np

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imint.training.skg_height import fetch_height_tile


# ── Helpers ──────────────────────────────────────────────────────────────

_TILE_RE = re.compile(r"tile_(\d+)_(\d+)\.npz$")


def _bbox_from_tile(tile_path: Path, half_m: int) -> tuple[int, int, int, int]:
    """Extract EPSG:3006 bbox from tile filename.

    Tile names are ``tile_{easting}_{northing}.npz`` where easting and
    northing are the grid-cell center.  The bbox extends ±half_m around
    that center.

    Returns (west, south, east, north) in EPSG:3006 meters.
    """
    m = _TILE_RE.search(tile_path.name)
    if not m:
        raise ValueError(f"Cannot parse easting/northing from {tile_path.name}")
    easting = int(m.group(1))
    northing = int(m.group(2))
    return (
        easting - half_m,
        northing - half_m,
        easting + half_m,
        northing + half_m,
    )


def _tile_has_height(tile_path: Path) -> bool:
    """Quick check whether a tile already contains a height channel."""
    try:
        with np.load(tile_path, allow_pickle=True) as d:
            return "height" in d
    except Exception:
        return False


def _add_height_to_tile(
    tile_path: Path,
    half_m: int,
    cache_dir: Path | None,
) -> dict:
    """Fetch height and write it into an existing .npz tile.

    The tile is rewritten atomically (tmp → rename) to avoid corruption.
    """
    west, south, east, north = _bbox_from_tile(tile_path, half_m)

    # Load existing tile data
    with np.load(tile_path, allow_pickle=True) as d:
        if "height" in d:
            return {"status": "skip", "tile": tile_path.name}
        existing = dict(d)

    # Determine output pixel size from the image shape (C, H, W)
    img = existing.get("spectral", existing.get("image"))
    if img is not None and img.ndim >= 2:
        h_px, w_px = img.shape[-2], img.shape[-1]
    else:
        h_px, w_px = 256, 256

    # Fetch height matching the tile's spatial dimensions exactly
    height = fetch_height_tile(
        west, south, east, north,
        size_px=(h_px, w_px),
        cache_dir=cache_dir,
    )

    existing["height"] = height

    # Atomic rewrite — np.savez_compressed does NOT auto-append .npz,
    # so the tmp file name IS what we pass.
    tmp = tile_path.parent / (tile_path.stem + "_tmp.npz")
    np.savez_compressed(tmp, **existing)
    tmp.rename(tile_path)

    return {
        "status": "ok",
        "tile": tile_path.name,
        "mean_h": float(height.mean()),
        "max_h": float(height.max()),
        "nonzero_pct": float((height > 0).mean() * 100),
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prefetch tree height for all training tiles"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/lulc_full",
        help="Training data directory (default: data/lulc_full)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel download threads (default: 4)",
    )
    parser.add_argument(
        "--patch-size-m", type=int, default=2560,
        help="Tile spatial extent in meters (default: 2560 = 256px × 10m)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable height tile caching",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only count tiles needing height, don't fetch",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tiles_dir = data_dir / "tiles"
    cache_dir = data_dir / "cache" / "height" if not args.no_cache else None
    half_m = args.patch_size_m // 2

    if not tiles_dir.exists():
        print(f"ERROR: Tiles directory not found: {tiles_dir}")
        sys.exit(1)

    # ── Scan tiles ──
    all_tiles = sorted(tiles_dir.glob("tile_*.npz"))
    print(f"Found {len(all_tiles)} tiles in {tiles_dir}")

    # Filter to tiles without height
    print("Scanning for tiles without height channel...")
    todo = []
    already = 0
    for tp in all_tiles:
        if _tile_has_height(tp):
            already += 1
        else:
            todo.append(tp)

    print(f"  Already have height: {already}")
    print(f"  Need height:         {len(todo)}")

    if not todo:
        print("\nAll tiles already have height data. Nothing to do!")
        return

    if args.dry_run:
        print(f"\n[dry-run] Would fetch height for {len(todo)} tiles.")
        return

    # ── Progress tracking ──
    progress_file = data_dir / "height_prefetch_progress.json"
    progress_lock = Lock()
    stats = {"ok": 0, "skip": 0, "fail": 0, "total": len(todo)}
    t_start = time.time()

    def _save_progress():
        elapsed = time.time() - t_start
        done = stats["ok"] + stats["skip"]
        rate = done / max(elapsed, 0.01)
        remaining = (len(todo) - done) / max(rate, 0.001)
        with open(progress_file, "w") as f:
            json.dump({
                **stats,
                "elapsed_s": round(elapsed, 1),
                "rate_tiles_per_s": round(rate, 2),
                "eta_s": round(remaining, 0),
            }, f, indent=2)

    # ── Parallel fetch ──
    print(f"\nFetching height for {len(todo)} tiles "
          f"({args.workers} workers)...")
    print(f"Cache: {cache_dir or 'disabled'}")
    print()

    def _worker(tile_path):
        try:
            return _add_height_to_tile(tile_path, half_m, cache_dir)
        except Exception as e:
            return {
                "status": "fail",
                "tile": tile_path.name,
                "error": f"{type(e).__name__}: {e}",
            }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, tp): tp for tp in todo}

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            status = result["status"]

            with progress_lock:
                stats[status] = stats.get(status, 0) + 1

            if status == "fail":
                print(f"  ✗ {result['tile']}: {result.get('error', '?')}")
            elif status == "ok" and i <= 5:
                # Show first few successes
                print(f"  ✓ {result['tile']}: "
                      f"mean={result['mean_h']:.1f}m "
                      f"max={result['max_h']:.1f}m "
                      f"forest={result['nonzero_pct']:.0f}%")

            # Progress every 50 tiles
            if i % 50 == 0 or i == len(todo):
                elapsed = time.time() - t_start
                rate = i / max(elapsed, 0.01)
                eta = (len(todo) - i) / max(rate, 0.001)
                print(f"  [{i}/{len(todo)}] "
                      f"{stats['ok']} ok, {stats['fail']} fail — "
                      f"{rate:.1f} tiles/s, "
                      f"ETA {eta/60:.0f}min")
                _save_progress()

    # ── Summary ──
    elapsed = time.time() - t_start
    _save_progress()
    print(f"\n{'='*60}")
    print(f"  Height prefetch complete!")
    print(f"  OK:      {stats['ok']}")
    print(f"  Skipped: {stats['skip']}")
    print(f"  Failed:  {stats['fail']}")
    print(f"  Time:    {elapsed/60:.1f} min ({elapsed:.0f}s)")
    print(f"  Rate:    {stats['ok']/max(elapsed, 1):.1f} tiles/s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
