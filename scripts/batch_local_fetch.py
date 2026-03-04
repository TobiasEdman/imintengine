#!/usr/bin/env python3
"""
scripts/batch_local_fetch.py — Batch-fetch seasonal tiles locally.

Runs the SeasonalFetchExecutor in auto mode (dynamic 2:1 CDSE:DES with
fallback) for multiple tiles, simulating what ColonyOS would do.

Usage:
    # Fetch 10 tiles (default):
    source .env && python scripts/batch_local_fetch.py

    # Fetch 50 tiles:
    source .env && python scripts/batch_local_fetch.py --count 50

    # Resume (skips already-completed tiles):
    source .env && python scripts/batch_local_fetch.py --count 50
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyproj import Transformer
from imint.training.sampler import generate_grid, filter_land_cells
from imint.training.config import TrainingConfig

# SWEREF99 TM (EPSG:3006) → WGS84 (EPSG:4326)
_tf = Transformer.from_crs("EPSG:3006", "EPSG:4326", always_xy=True)


def _fill_wgs84(cell):
    """Compute WGS84 bbox from SWEREF99 coordinates."""
    cell.west_wgs84, cell.south_wgs84 = _tf.transform(cell.west_3006, cell.south_3006)
    cell.east_wgs84, cell.north_wgs84 = _tf.transform(cell.east_3006, cell.north_3006)
    return cell


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Batch seasonal fetch (local)")
    parser.add_argument("--count", type=int, default=10,
                        help="Number of tiles to fetch (default: 10)")
    parser.add_argument("--tiles-dir", default="data/seasonal_tiles",
                        help="Output directory")
    parser.add_argument("--years", default="2019,2018")
    parser.add_argument("--windows", default="4-5,6-7,8-9,1-2")
    parser.add_argument("--cloud-threshold", type=float, default=0.10)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--b02-haze", type=float, default=0.06)
    args = parser.parse_args()

    tiles_dir = Path(args.tiles_dir)
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # Generate grid
    config = TrainingConfig(grid_spacing_m=10000)
    patch_size_m = config.fetch_pixels * 10
    cells = generate_grid(spacing_m=10000, patch_size_m=patch_size_m)
    land_cells = filter_land_cells(cells)

    # Compute WGS84 coords and skip already-completed tiles
    remaining = []
    for cell in land_cells:
        _fill_wgs84(cell)
        key = f"{cell.easting}_{cell.northing}"
        tile_path = tiles_dir / f"tile_{key}.npz"
        if not tile_path.exists():
            remaining.append(cell)

    print("=" * 60)
    print(f"  Batch Seasonal Fetch — Local")
    print(f"  Total land cells:  {len(land_cells)}")
    print(f"  Already completed: {len(land_cells) - len(remaining)}")
    print(f"  Remaining:         {len(remaining)}")
    print(f"  Fetching:          {min(args.count, len(remaining))}")
    print(f"  Mode:              auto (2:1 CDSE:DES + fallback)")
    print(f"  Output:            {tiles_dir}")
    print("=" * 60)

    to_fetch = remaining[:args.count]
    ok_count = 0
    fail_count = 0
    total_time = 0.0

    for i, cell in enumerate(to_fetch):
        key = f"{cell.easting}_{cell.northing}"
        print(f"\n[{i+1}/{len(to_fetch)}] Tile {key}")

        # Set environment for this tile
        os.environ["EASTING"] = str(cell.easting)
        os.environ["NORTHING"] = str(cell.northing)
        os.environ["WEST_WGS84"] = str(cell.west_wgs84)
        os.environ["SOUTH_WGS84"] = str(cell.south_wgs84)
        os.environ["EAST_WGS84"] = str(cell.east_wgs84)
        os.environ["NORTH_WGS84"] = str(cell.north_wgs84)
        os.environ["FETCH_SOURCE"] = "auto"
        os.environ["YEARS"] = args.years
        os.environ["SEASONAL_WINDOWS"] = args.windows
        os.environ["SEASONAL_CLOUD_THRESHOLD"] = str(args.cloud_threshold)
        os.environ["TILES_DIR"] = str(tiles_dir)
        os.environ["NUM_CLASSES"] = str(args.num_classes)
        os.environ["B02_HAZE_THRESHOLD"] = str(args.b02_haze)

        from executors.seasonal_fetch import SeasonalFetchExecutor

        t0 = time.monotonic()
        try:
            executor = SeasonalFetchExecutor()
            executor.execute()
            elapsed = time.monotonic() - t0
            total_time += elapsed
            ok_count += 1
            print(f"  ✓ Done in {elapsed:.0f}s")
        except Exception as e:
            elapsed = time.monotonic() - t0
            total_time += elapsed
            fail_count += 1
            print(f"  ✗ Failed ({elapsed:.0f}s): {e}")

        # Progress summary
        done = ok_count + fail_count
        rate = total_time / done if done else 0
        print(f"  [{ok_count} ok / {fail_count} fail / "
              f"{len(to_fetch) - done} left / "
              f"{rate:.0f}s avg]")

    print("\n" + "=" * 60)
    print(f"  Complete: {ok_count}/{ok_count + fail_count} succeeded")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f}m)")
    if ok_count:
        print(f"  Avg per tile: {total_time / ok_count:.0f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
