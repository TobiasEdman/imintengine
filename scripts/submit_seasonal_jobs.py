#!/usr/bin/env python3
"""
scripts/submit_seasonal_jobs.py — ColonyOS job coordinator for seasonal fetch.

Generates the tile grid, applies NMD land pre-filter, and submits one
ColonyOS job per tile.  Jobs are load-balanced between DES and CDSE
with a 2:1 CDSE:DES ratio (CDSE is ~30% faster).

Resumable: tiles already present in CFS are skipped automatically.

Usage:
    python scripts/submit_seasonal_jobs.py \\
        --sources copernicus,des \\
        --years 2019,2018 \\
        --tiles-dir seasonal-tiles \\
        --max-jobs 100 \\
        --dry-run

    # Monitor progress:
    python scripts/submit_seasonal_jobs.py --status --tiles-dir seasonal-tiles
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import os
import time
import tempfile
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imint.training.sampler import (
    generate_grid, grid_to_wgs84, split_by_latitude,
    filter_land_cells,
)
from imint.training.config import TrainingConfig


def _list_existing_tiles(cfs_dir: str) -> set[str]:
    """List tiles already present in CFS, return set of cell keys."""
    try:
        result = subprocess.run(
            ["colonies", "fs", "ls", "--label", cfs_dir, "--insecure"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            print(f"  Warning: colonies fs list failed: {result.stderr.strip()}")
            return set()

        existing = set()
        for line in result.stdout.strip().splitlines():
            name = line.strip().split()[-1] if line.strip() else ""
            if name.startswith("tile_") and name.endswith(".npz"):
                # tile_361280_6231280.npz → 361280_6231280
                key = name[5:-4]  # strip "tile_" and ".npz"
                existing.add(key)
        return existing
    except FileNotFoundError:
        print("  Warning: 'colonies' CLI not found — skipping CFS check")
        return set()
    except subprocess.TimeoutExpired:
        print("  Warning: colonies fs list timed out")
        return set()


def _build_job_spec(
    cell,
    years: list[str],
    windows: str,
    cloud_threshold: float,
    cfs_dir: str,
    num_classes: int,
    b02_haze_threshold: float,
    des_token: str = "",
    des_user: str = "",
    des_password: str = "",
    cdse_client_id: str = "",
    cdse_client_secret: str = "",
) -> dict:
    """Build a ColonyOS job spec dict for one tile.

    FETCH_SOURCE is set to "auto" — the executor picks the source
    dynamically (2:1 CDSE:DES weighting) with fallback on failure.
    Both CDSE and DES credentials are included so fallback works.
    """
    cell_key = f"{cell.easting}_{cell.northing}"
    return {
        "conditions": {
            "colonyname": "imint",
            "executortype": "container-executor",
            "nodes": 1,
            "processes": 1,
            "processespernode": 1,
            "walltime": 900,
            "cpu": "1000m",
            "mem": "2Gi",
        },
        "env": {
            "PYTHONUNBUFFERED": "1",
            "EASTING": str(cell.easting),
            "NORTHING": str(cell.northing),
            "WEST_WGS84": str(cell.west_wgs84),
            "SOUTH_WGS84": str(cell.south_wgs84),
            "EAST_WGS84": str(cell.east_wgs84),
            "NORTH_WGS84": str(cell.north_wgs84),
            "FETCH_SOURCE": "auto",
            "YEARS": ",".join(years),
            "SEASONAL_WINDOWS": windows,
            "SEASONAL_CLOUD_THRESHOLD": str(cloud_threshold),
            "TILES_DIR": "/cfs/tiles",
            "NUM_CLASSES": str(num_classes),
            "B02_HAZE_THRESHOLD": str(b02_haze_threshold),
            "DES_TOKEN": des_token,
            "DES_USER": des_user,
            "DES_PASSWORD": des_password,
            "CDSE_CLIENT_ID": cdse_client_id,
            "CDSE_CLIENT_SECRET": cdse_client_secret,
        },
        "funcname": "execute",
        "kwargs": {
            "docker-image": "localhost:5000/imint-engine:latest",
            "rebuild-image": False,
            "cmd": "python executors/seasonal_fetch.py",
        },
        "maxexectime": 900,
        "maxretries": 3,
        "maxwaittime": -1,
        "fs": {
            "mount": "/cfs/tiles",
            "dirs": [
                {
                    "label": cfs_dir,
                    "dir": "/cfs/tiles",
                    "keepfiles": False,
                    "onconflicts": {
                        "onstart": {"keeplocal": False},
                        "onclose": {"keeplocal": False},
                    },
                }
            ],
        },
    }


def _submit_job(spec: dict, dry_run: bool = False) -> bool:
    """Submit a single ColonyOS job. Returns True on success."""
    if dry_run:
        cell_key = f"{spec['env']['EASTING']}_{spec['env']['NORTHING']}"
        mode = spec["env"]["FETCH_SOURCE"]
        print(f"  [DRY-RUN] Would submit: {cell_key} (source={mode})")
        return True

    # Write spec to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(spec, f)
        spec_path = f.name

    try:
        result = subprocess.run(
            ["colonies", "function", "submit", "--spec", spec_path, "--insecure"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            print(f"  Submit failed: {result.stderr.strip()}")
            return False
        return True
    except FileNotFoundError:
        print("  Error: 'colonies' CLI not found")
        return False
    finally:
        os.unlink(spec_path)


def main():
    parser = argparse.ArgumentParser(
        description="Submit seasonal tile fetch jobs to ColonyOS"
    )
    parser.add_argument(
        "--sources", default="copernicus,des",
        help="Comma-separated fetch sources (default: copernicus,des)",
    )
    parser.add_argument(
        "--years", default="2019,2018",
        help="Comma-separated years to search (default: 2019,2018)",
    )
    parser.add_argument(
        "--windows", default="4-5,6-7,8-9,1-2",
        help="Seasonal windows (default: 4-5,6-7,8-9,1-2)",
    )
    parser.add_argument(
        "--cloud-threshold", type=float, default=0.10,
        help="Max cloud fraction per scene (default: 0.10)",
    )
    parser.add_argument(
        "--tiles-dir", default="seasonal-tiles",
        help="CFS directory name for tiles (default: seasonal-tiles)",
    )
    parser.add_argument(
        "--grid-spacing", type=int, default=10_000,
        help="Grid spacing in meters (default: 10000)",
    )
    parser.add_argument(
        "--max-jobs", type=int, default=0,
        help="Max jobs to submit (0 = unlimited, default: 0)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be submitted without actually submitting",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show progress status and exit",
    )
    parser.add_argument(
        "--num-classes", type=int, default=10,
        help="NMD class count (default: 10)",
    )
    args = parser.parse_args()

    sources = [s.strip() for s in args.sources.split(",")]
    years = [y.strip() for y in args.years.split(",")]

    print("=" * 60)
    print("  Seasonal Tile Fetch — ColonyOS Coordinator")
    print(f"  Sources: {', '.join(s.upper() for s in sources)}")
    print(f"  Years: {', '.join(years)}")
    print(f"  Windows: {args.windows}")
    print(f"  Cloud threshold: {args.cloud_threshold:.0%}")
    print(f"  Grid spacing: {args.grid_spacing}m")
    print("=" * 60)

    # ── Status mode ──────────────────────────────────────────────────
    if args.status:
        existing = _list_existing_tiles(args.tiles_dir)
        print(f"\n  Completed tiles in CFS: {len(existing)}")
        return

    # ── Generate grid ────────────────────────────────────────────────
    config = TrainingConfig(grid_spacing_m=args.grid_spacing)
    patch_size_m = config.fetch_pixels * 10

    print("\n  Generating grid...")
    cells = generate_grid(
        spacing_m=args.grid_spacing,
        patch_size_m=patch_size_m,
    )
    print(f"  Total grid cells: {len(cells)}")

    # ── NMD pre-filter ───────────────────────────────────────────────
    print("  Filtering land cells...")
    land_cells = filter_land_cells(cells)
    print(f"  Land cells: {len(land_cells)}")

    # ── Compute WGS84 coordinates from SWEREF99 ─────────────────────
    land_cells = grid_to_wgs84(land_cells)

    # ── Check CFS for existing tiles ─────────────────────────────────
    print("  Checking CFS for existing tiles...")
    existing = _list_existing_tiles(args.tiles_dir)
    print(f"  Already completed: {len(existing)}")

    # ── Filter out already-done tiles ────────────────────────────────
    todo = []
    for cell in land_cells:
        key = f"{cell.easting}_{cell.northing}"
        if key not in existing:
            todo.append(cell)
    print(f"  Remaining to fetch: {len(todo)}")

    if not todo:
        print("\n  All tiles already fetched!")
        return

    if args.max_jobs > 0:
        todo = todo[:args.max_jobs]
        print(f"  Limiting to {args.max_jobs} jobs")

    # ── Load credentials (both CDSE and DES for dynamic fallback) ────
    des_token = os.environ.get("DES_TOKEN", "")
    des_user = os.environ.get("DES_USER", "")
    des_password = os.environ.get("DES_PASSWORD", "")
    cdse_client_id = os.environ.get("CDSE_CLIENT_ID", "")
    cdse_client_secret = os.environ.get("CDSE_CLIENT_SECRET", "")

    # Try .cdse_credentials file
    creds_path = Path(__file__).parent.parent / ".cdse_credentials"
    if creds_path.exists() and not cdse_client_id:
        lines = creds_path.read_text().strip().splitlines()
        if len(lines) >= 4:
            cdse_client_id = lines[2].strip()
            cdse_client_secret = lines[3].strip()

    has_cdse = bool(cdse_client_id and cdse_client_secret)
    has_des = bool(des_token or (des_user and des_password))
    print(f"\n  Credentials: CDSE={'OK' if has_cdse else 'MISSING'}, "
          f"DES={'OK' if has_des else 'MISSING'}")
    print(f"  Source mode: auto (executor picks 2:1 CDSE:DES with fallback)")

    # ── Submit jobs ───────────────────────────────────────────────────
    print(f"\n  Submitting {len(todo)} jobs "
          f"({'DRY RUN' if args.dry_run else 'LIVE'})...")

    submitted = 0
    failed = 0

    for i, cell in enumerate(todo):
        spec = _build_job_spec(
            cell=cell,
            years=years,
            windows=args.windows,
            cloud_threshold=args.cloud_threshold,
            cfs_dir=args.tiles_dir,
            num_classes=args.num_classes,
            b02_haze_threshold=config.b02_haze_threshold,
            des_token=des_token,
            des_user=des_user,
            des_password=des_password,
            cdse_client_id=cdse_client_id,
            cdse_client_secret=cdse_client_secret,
        )

        if _submit_job(spec, dry_run=args.dry_run):
            submitted += 1
        else:
            failed += 1

        # Progress every 100 jobs
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(todo)} submitted")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n  {'=' * 50}")
    print(f"  Submitted: {submitted}")
    print(f"  Failed:    {failed}")
    print(f"  Source: auto (dynamic 2:1 CDSE:DES with fallback)")
    print(f"  {'=' * 50}")

    if not args.dry_run and submitted > 0:
        print(f"\n  Monitor progress:")
        print(f"    python scripts/submit_seasonal_jobs.py "
              f"--status --tiles-dir {args.tiles_dir}")


if __name__ == "__main__":
    main()
