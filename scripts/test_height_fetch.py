#!/usr/bin/env python3
"""Quick integration test for Skogsstyrelsen tree height fetching.

Tests the ArcGIS ImageServer exportImage endpoint with a small
known-forested bbox in southern Sweden.

Usage:
    python scripts/test_height_fetch.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    from imint.training.skg_height import fetch_height_tile, fetch_height_for_coords

    print("=" * 60)
    print("  Skogsstyrelsen Tree Height Fetch Test")
    print("=" * 60)

    # ── Test 1: Direct EPSG:3006 bbox (forested area near Ljungby) ───
    # This is a known forested area in Småland
    west, south = 396000, 6306000
    east, north = west + 2560, south + 2560  # 256px × 10m
    print(f"\n── Test 1: Direct EPSG:3006 bbox ──")
    print(f"  Bbox: ({west}, {south}) → ({east}, {north})")
    print(f"  Size: 256×256 @ 10m")

    t0 = time.time()
    try:
        height = fetch_height_tile(west, south, east, north, size_px=256)
        dt = time.time() - t0
        print(f"  ✓ Shape: {height.shape}")
        print(f"  ✓ dtype: {height.dtype}")
        print(f"  ✓ Range: [{height.min():.1f}, {height.max():.1f}] meters")
        print(f"  ✓ Mean: {height.mean():.1f} m")
        print(f"  ✓ Nonzero: {(height > 0).sum()} / {height.size} pixels "
              f"({(height > 0).mean():.1%})")
        print(f"  ✓ Time: {dt:.1f}s")
    except Exception as e:
        print(f"  ✗ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    # ── Test 2: WGS84 coords (same area as Sentinel-2 test bbox) ────
    coords = {
        "west": 13.0, "south": 55.5,
        "east": 13.1, "north": 55.6,
    }
    print(f"\n── Test 2: WGS84 coords ──")
    print(f"  Coords: {coords}")

    t0 = time.time()
    try:
        height2 = fetch_height_for_coords(coords, size_px=256)
        dt = time.time() - t0
        print(f"  ✓ Shape: {height2.shape}")
        print(f"  ✓ Range: [{height2.min():.1f}, {height2.max():.1f}] meters")
        print(f"  ✓ Mean: {height2.mean():.1f} m")
        print(f"  ✓ Nonzero: {(height2 > 0).sum()} / {height2.size} pixels "
              f"({(height2 > 0).mean():.1%})")
        print(f"  ✓ Time: {dt:.1f}s")
    except Exception as e:
        print(f"  ✗ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    # ── Test 3: Cache test (should be instant on second call) ────────
    print(f"\n── Test 3: Cache test ──")
    cache_dir = Path("/tmp/imint_height_test_cache")

    t0 = time.time()
    try:
        h3a = fetch_height_tile(west, south, east, north,
                                 size_px=256, cache_dir=cache_dir)
        dt1 = time.time() - t0
        print(f"  First call: {dt1:.2f}s")

        t0 = time.time()
        h3b = fetch_height_tile(west, south, east, north,
                                 size_px=256, cache_dir=cache_dir)
        dt2 = time.time() - t0
        print(f"  Cached call: {dt2:.4f}s")

        import numpy as np
        match = np.array_equal(h3a, h3b)
        print(f"  Match: {match}")
        print(f"  Speedup: {dt1/max(dt2, 0.0001):.0f}×")
    except Exception as e:
        print(f"  ✗ FAILED: {type(e).__name__}: {e}")

    # Cleanup
    import shutil
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    print(f"\n{'='*60}")
    print("  Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
