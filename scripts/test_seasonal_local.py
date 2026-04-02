#!/usr/bin/env python3
"""
scripts/test_seasonal_local.py — Test seasonal fetch executor locally.

Simulates what ColonyOS would do by setting environment variables and
running the SeasonalFetchExecutor directly (no Docker, no ColonyOS).

Usage:
    # Quick test — single tile, CDSE backend:
    python scripts/test_seasonal_local.py

    # Test with DES backend:
    python scripts/test_seasonal_local.py --source des

    # Custom tile:
    python scripts/test_seasonal_local.py \\
        --easting 371280 --northing 6241280 \\
        --west 16.1 --south 58.5 --east 16.15 --north 58.55

    # Test both backends and compare:
    python scripts/test_seasonal_local.py --compare
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import numpy as np
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _set_env(args, source: str, tiles_dir: str):
    """Set environment variables to simulate ColonyOS."""
    os.environ["EASTING"] = str(args.easting)
    os.environ["NORTHING"] = str(args.northing)
    os.environ["WEST_WGS84"] = str(args.west)
    os.environ["SOUTH_WGS84"] = str(args.south)
    os.environ["EAST_WGS84"] = str(args.east)
    os.environ["NORTH_WGS84"] = str(args.north)
    os.environ["FETCH_SOURCE"] = source
    os.environ["YEARS"] = args.years
    os.environ["SEASONAL_WINDOWS"] = args.windows
    os.environ["SEASONAL_CLOUD_THRESHOLD"] = str(args.cloud_threshold)
    os.environ["TILES_DIR"] = tiles_dir
    os.environ["NUM_CLASSES"] = str(args.num_classes)
    os.environ["B02_HAZE_THRESHOLD"] = str(args.b02_haze)


def _run_fetch(args, source: str, tiles_dir: str) -> tuple[bool, float]:
    """Run one seasonal fetch, return (success, elapsed_seconds)."""
    _set_env(args, source, tiles_dir)

    from executors.seasonal_fetch import SeasonalFetchExecutor

    t0 = time.monotonic()
    try:
        executor = SeasonalFetchExecutor()
        executor.execute()
        elapsed = time.monotonic() - t0
        return True, elapsed
    except Exception as e:
        elapsed = time.monotonic() - t0
        print(f"\n  FAILED ({source.upper()}): {e}")
        return False, elapsed


def _inspect_tile(path: Path):
    """Print summary of a saved tile."""
    if not path.exists():
        print(f"  Tile not found: {path}")
        return

    data = np.load(path, allow_pickle=True)
    image = data.get("spectral", data.get("image"))
    label = data["label"]
    dates = data["dates"]
    mask = data["temporal_mask"]
    source = str(data.get("source", "unknown"))

    print(f"\n  Tile: {path.name}")
    print(f"  Source: {source}")
    print(f"  Image shape: {image.shape}  (T*bands, H, W)")
    print(f"  Label shape: {label.shape}")
    print(f"  Dates: {list(dates)}")
    print(f"  Temporal mask: {list(mask)}")
    print(f"  Valid frames: {sum(mask)}/{len(mask)}")

    n_bands = int(data.get("num_bands", 6))
    n_frames = int(data.get("num_frames", 4))
    for t in range(n_frames):
        if mask[t]:
            frame = image[t * n_bands:(t + 1) * n_bands]
            print(f"  Frame {t} ({dates[t]}): "
                  f"mean={frame.mean():.4f}, "
                  f"range=[{frame.min():.4f}, {frame.max():.4f}]")


def _compare_tiles(path_cdse: Path, path_des: Path):
    """Compare tiles from CDSE and DES."""
    if not path_cdse.exists() or not path_des.exists():
        print("  Cannot compare — one or both tiles missing")
        return

    cdse = np.load(path_cdse, allow_pickle=True)
    des = np.load(path_des, allow_pickle=True)

    print("\n  ── Cross-source comparison ──")
    cdse_mask = cdse["temporal_mask"]
    des_mask = des["temporal_mask"]

    n_bands = int(cdse.get("num_bands", 6))
    n_frames = int(cdse.get("num_frames", 4))

    for t in range(n_frames):
        if cdse_mask[t] and des_mask[t]:
            cdse_img = cdse.get("spectral", cdse.get("image"))
            des_img = des.get("spectral", des.get("image"))
            cdse_frame = cdse_img[t * n_bands:(t + 1) * n_bands]
            des_frame = des_img[t * n_bands:(t + 1) * n_bands]
            diff = np.abs(cdse_frame - des_frame).mean()
            print(f"  Frame {t}: CDSE date={cdse['dates'][t]}, "
                  f"DES date={des['dates'][t]}, "
                  f"mean abs diff={diff:.4f}")
        else:
            print(f"  Frame {t}: CDSE={'OK' if cdse_mask[t] else 'MISS'}, "
                  f"DES={'OK' if des_mask[t] else 'MISS'}")


def main():
    parser = argparse.ArgumentParser(
        description="Test seasonal fetch executor locally (no Docker/ColonyOS)"
    )
    parser.add_argument("--source", default="copernicus",
                        choices=["copernicus", "des"],
                        help="Fetch backend (default: copernicus)")
    parser.add_argument("--compare", action="store_true",
                        help="Fetch from both backends and compare")

    # Tile coordinates (default: a known good training tile)
    parser.add_argument("--easting", type=int, default=361280)
    parser.add_argument("--northing", type=int, default=6231280)
    parser.add_argument("--west", type=float, default=16.0)
    parser.add_argument("--south", type=float, default=58.4)
    parser.add_argument("--east", type=float, default=16.05)
    parser.add_argument("--north", type=float, default=58.45)

    # Seasonal config
    parser.add_argument("--years", default="2019,2018")
    parser.add_argument("--windows", default="4-5,6-7,8-9,1-2")
    parser.add_argument("--cloud-threshold", type=float, default=0.10)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--b02-haze", type=float, default=0.06)

    # Output
    parser.add_argument("--tiles-dir", default="data/test_tiles")

    args = parser.parse_args()

    print("=" * 60)
    print("  Seasonal Fetch — Local Test")
    print(f"  Tile: {args.easting}_{args.northing}")
    print(f"  Mode: {'COMPARE (both)' if args.compare else args.source.upper()}")
    print("=" * 60)

    tiles_dir = Path(args.tiles_dir)
    tiles_dir.mkdir(parents=True, exist_ok=True)
    cell_key = f"{args.easting}_{args.northing}"

    if args.compare:
        # Fetch from both and compare
        for source in ["copernicus", "des"]:
            out_dir = tiles_dir / source
            out_dir.mkdir(parents=True, exist_ok=True)

            tile_path = out_dir / f"tile_{cell_key}.npz"
            if tile_path.exists():
                print(f"\n  {source.upper()}: already exists, skipping")
                _inspect_tile(tile_path)
                continue

            print(f"\n  Fetching via {source.upper()}...")
            ok, elapsed = _run_fetch(args, source, str(out_dir))
            if ok:
                print(f"  {source.upper()}: done in {elapsed:.1f}s")
                _inspect_tile(tile_path)
            else:
                print(f"  {source.upper()}: failed after {elapsed:.1f}s")

        # Compare
        _compare_tiles(
            tiles_dir / "copernicus" / f"tile_{cell_key}.npz",
            tiles_dir / "des" / f"tile_{cell_key}.npz",
        )
    else:
        # Single source
        tile_path = tiles_dir / f"tile_{cell_key}.npz"
        if tile_path.exists():
            tile_path.unlink()  # re-fetch for testing

        print(f"\n  Fetching via {args.source.upper()}...")
        ok, elapsed = _run_fetch(args, args.source, str(tiles_dir))
        if ok:
            print(f"\n  Done in {elapsed:.1f}s")
            _inspect_tile(tile_path)
        else:
            print(f"\n  Failed after {elapsed:.1f}s")
            sys.exit(1)


if __name__ == "__main__":
    main()
