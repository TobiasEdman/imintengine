#!/usr/bin/env python3
"""
scripts/submit_s2_jobs.py — ColonyOS job coordinator for S2 Process API fetch.

Generates the tile grid, applies NMD land pre-filter, and submits one
ColonyOS job per tile using the Sentinel Hub Process API path.

This replaces submit_seasonal_jobs.py for Process API mode.  Key differences:
  - Only needs CDSE credentials (no DES for spectral data)
  - Uses executors/s2_seasonal_fetch.py instead of seasonal_fetch.py
  - Simpler job spec (no DES tokens, no source selection)
  - Lower walltime (600s vs 900s) since Process API is faster

Also supports --local mode for running directly with ThreadPoolExecutor.

Resumable: tiles already present in CFS/local dir are skipped.

Usage:
    # Dry-run: see what jobs would be submitted
    python scripts/submit_s2_jobs.py --dry-run

    # Submit first 100 jobs
    python scripts/submit_s2_jobs.py --max-jobs 100

    # Submit all
    python scripts/submit_s2_jobs.py

    # Check progress
    python scripts/submit_s2_jobs.py --status

    # Local mode (no ColonyOS — uses ThreadPoolExecutor)
    python scripts/submit_s2_jobs.py --local --workers 4

    # Local mode on existing tiles directory
    python scripts/submit_s2_jobs.py --local --tiles-dir data/lulc_full/tiles
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imint.training.sampler import (
    generate_grid, grid_to_wgs84, split_by_latitude,
    filter_land_cells,
)
from imint.training.config import TrainingConfig

_TILE_RE = re.compile(r"tile_(\d+)_(\d+)\.npz$")


def _list_existing_tiles_cfs(cfs_dir: str) -> set[str]:
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
                key = name[5:-4]  # strip "tile_" and ".npz"
                existing.add(key)
        return existing
    except FileNotFoundError:
        print("  Warning: 'colonies' CLI not found — skipping CFS check")
        return set()
    except subprocess.TimeoutExpired:
        print("  Warning: colonies fs list timed out")
        return set()


def _list_existing_tiles_local(tiles_dir: Path) -> set[str]:
    """List tiles already present locally, return set of cell keys."""
    existing = set()
    for f in tiles_dir.glob("tile_*.npz"):
        m = _TILE_RE.search(f.name)
        if m:
            existing.add(f"{m.group(1)}_{m.group(2)}")
    return existing


def _build_job_spec(
    cell,
    years: list[str],
    windows: str,
    cloud_threshold: float,
    haze_threshold: float,
    cfs_dir: str,
    num_classes: int,
    tile_size_px: int,
    cdse_client_id: str,
    cdse_client_secret: str,
) -> dict:
    """Build a ColonyOS job spec dict for one S2 Process API tile."""
    return {
        "conditions": {
            "colonyname": "imint",
            "executortype": "container-executor",
            "nodes": 1,
            "processes": 1,
            "processespernode": 1,
            "walltime": 600,
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
            "YEARS": ",".join(years),
            "SEASONAL_WINDOWS": windows,
            "SEASONAL_CLOUD_THRESHOLD": str(cloud_threshold),
            "B02_HAZE_THRESHOLD": str(haze_threshold),
            "TILES_DIR": "/cfs/tiles",
            "NUM_CLASSES": str(num_classes),
            "TILE_SIZE_PX": str(tile_size_px),
            "CDSE_CLIENT_ID": cdse_client_id,
            "CDSE_CLIENT_SECRET": cdse_client_secret,
        },
        "funcname": "execute",
        "kwargs": {
            "docker-image": "localhost:5000/imint-engine:latest",
            "rebuild-image": False,
            "cmd": "python executors/s2_seasonal_fetch.py",
        },
        "maxexectime": 600,
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
        print(f"  [DRY-RUN] Would submit S2 fetch: {cell_key}")
        return True

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


def _run_local(
    cells: list,
    tiles_dir: Path,
    years: list[str],
    windows: list[tuple[int, int]],
    cloud_threshold: float,
    haze_threshold: float,
    num_classes: int,
    tile_size_px: int,
    workers: int,
) -> None:
    """Run S2 Process API fetch locally with ThreadPoolExecutor."""
    import numpy as np
    from imint.training.cdse_s2 import fetch_s2_seasonal_tile
    from imint.fetch import fetch_nmd_data, FetchError
    from imint.training.class_schema import nmd_raster_to_lulc

    # Filter to tiles that don't exist yet
    todo = []
    for cell in cells:
        key = f"{cell.easting}_{cell.northing}"
        tile_path = tiles_dir / f"tile_{key}.npz"
        if not tile_path.exists():
            todo.append(cell)

    print(f"  Tiles to fetch: {len(todo)}")
    if not todo:
        print("  All tiles already exist!")
        return

    t_start = time.time()
    stats = {"ok": 0, "fail": 0, "skip": 0}

    def _fetch_one(cell):
        key = f"{cell.easting}_{cell.northing}"
        tile_path = tiles_dir / f"tile_{key}.npz"

        if tile_path.exists():
            return {"status": "skip", "key": key}

        coords_wgs84 = {
            "west": cell.west_wgs84,
            "south": cell.south_wgs84,
            "east": cell.east_wgs84,
            "north": cell.north_wgs84,
        }

        try:
            # Fetch spectral data via Process API
            result = fetch_s2_seasonal_tile(
                easting=cell.easting,
                northing=cell.northing,
                coords_wgs84=coords_wgs84,
                windows=windows,
                years=years,
                size_px=tile_size_px,
                cloud_threshold=cloud_threshold,
                haze_threshold=haze_threshold,
            )

            if result is None:
                return {"status": "fail", "key": key, "error": "no valid frames"}

            image = result["spectral"]
            ref_shape = (image.shape[-2], image.shape[-1])
            n_valid = int(result["temporal_mask"].sum())

            # Fetch NMD labels
            try:
                nmd_result = fetch_nmd_data(
                    coords=coords_wgs84,
                    target_shape=ref_shape,
                )
                labels = nmd_raster_to_lulc(
                    nmd_result.nmd_raster,
                    num_classes=num_classes,
                )
            except Exception:
                labels = np.zeros(ref_shape, dtype=np.uint8)

            # Save (atomic)
            save_data = dict(result)
            save_data["label"] = labels

            tiles_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = tile_path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp_path, **save_data)
            tmp_path.rename(tile_path)

            return {
                "status": "ok",
                "key": key,
                "frames": f"{n_valid}/{len(windows)}",
            }

        except Exception as e:
            return {"status": "fail", "key": key, "error": str(e)[:200]}

    print(f"\n  Fetching S2 tiles for {len(todo)} cells ({workers} workers)...\n")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one, cell): cell for cell in todo}

        for i, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
                status = result["status"]
                stats[status] = stats.get(status, 0) + 1

                if status == "ok" and i <= 10:
                    print(f"  ✓ {result['key']}: {result['frames']} frames")
                elif status == "fail":
                    print(f"  ✗ {result['key']}: {result.get('error', '?')}")

            except Exception as e:
                stats["fail"] += 1
                cell = futures[future]
                key = f"{cell.easting}_{cell.northing}"
                print(f"  ✗ {key}: {type(e).__name__}: {e}")

            if i % 50 == 0 or i == len(todo):
                elapsed = time.time() - t_start
                rate = i / max(elapsed, 0.01)
                eta = (len(todo) - i) / max(rate, 0.001)
                print(f"  [{i}/{len(todo)}] "
                      f"{stats['ok']} ok, {stats['fail']} fail — "
                      f"{rate:.1f} tiles/s, ETA {eta/60:.0f}min")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  S2 Process API fetch complete!")
    print(f"  OK:      {stats['ok']}")
    print(f"  Failed:  {stats['fail']}")
    print(f"  Skipped: {stats['skip']}")
    print(f"  Time:    {elapsed/60:.1f} min")
    print(f"  Rate:    {stats['ok']/max(elapsed, 1):.1f} tiles/s")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Submit S2 Process API seasonal fetch jobs to ColonyOS"
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
        "--haze-threshold", type=float, default=0.06,
        help="Max mean B02 reflectance for haze gate (default: 0.06)",
    )
    parser.add_argument(
        "--tiles-dir", default="seasonal-tiles",
        help="CFS directory name (or local dir with --local) for tiles "
             "(default: seasonal-tiles)",
    )
    parser.add_argument(
        "--grid-spacing", type=int, default=10_000,
        help="Grid spacing in meters (default: 10000)",
    )
    parser.add_argument(
        "--tile-size-px", type=int, default=256,
        help="Tile size in pixels (default: 256)",
    )
    parser.add_argument(
        "--max-jobs", type=int, default=0,
        help="Max jobs to submit (0 = unlimited, default: 0)",
    )
    parser.add_argument(
        "--num-classes", type=int, default=10,
        help="NMD class count (default: 10)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be submitted without submitting",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show progress status and exit",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Run locally with ThreadPoolExecutor instead of ColonyOS",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel workers for local mode (default: 4)",
    )
    parser.add_argument(
        "--from-existing", type=str, default=None,
        help="Read tile coordinates from existing tile directory instead of "
             "generating grid (e.g. data/lulc_full). Reuses the same "
             "geographic coverage including any densification.",
    )
    args = parser.parse_args()

    years = [y.strip() for y in args.years.split(",")]

    print("=" * 60)
    print("  S2 Seasonal Fetch — Sentinel Hub Process API")
    print(f"  Mode: {'LOCAL' if args.local else 'ColonyOS'}")
    print(f"  Years: {', '.join(years)}")
    print(f"  Windows: {args.windows}")
    print(f"  Cloud threshold: {args.cloud_threshold:.0%}")
    print(f"  Haze threshold: {args.haze_threshold}")
    print(f"  Grid spacing: {args.grid_spacing}m")
    print(f"  Tile size: {args.tile_size_px}px")
    print("=" * 60)

    # ── Generate grid ────────────────────────────────────────────────
    patch_size_m = args.tile_size_px * 10  # 256 px × 10 m = 2560 m
    half = patch_size_m // 2

    if args.from_existing:
        # Read tile coordinates from existing directory
        from imint.training.sampler import GridCell
        existing_tiles_dir = Path(args.from_existing)
        if (existing_tiles_dir / "tiles").is_dir():
            existing_tiles_dir = existing_tiles_dir / "tiles"
        print(f"\n  Reading tile coordinates from {existing_tiles_dir}...")
        cells = []
        for f in sorted(existing_tiles_dir.glob("tile_*.npz")):
            m = _TILE_RE.search(f.name)
            if m:
                e, n = int(m.group(1)), int(m.group(2))
                cells.append(GridCell(
                    easting=e, northing=n,
                    west_3006=e - half, east_3006=e + half,
                    south_3006=n - half, north_3006=n + half,
                ))
        print(f"  Found {len(cells)} tile locations")
        land_cells = grid_to_wgs84(cells)
    else:
        print("\n  Generating grid...")
        cells = generate_grid(
            spacing_m=args.grid_spacing,
            patch_size_m=patch_size_m,
        )
        print(f"  Total grid cells: {len(cells)}")

        # ── NMD pre-filter ───────────────────────────────────────────
        print("  Filtering land cells...")
        land_cells = filter_land_cells(cells)
        print(f"  Land cells: {len(land_cells)}")
        land_cells = grid_to_wgs84(land_cells)

    # Parse windows for local mode
    windows_list = []
    for part in args.windows.split(","):
        start, end = part.strip().split("-")
        windows_list.append((int(start), int(end)))

    # ── Local mode ───────────────────────────────────────────────────
    if args.local:
        tiles_dir = Path(args.tiles_dir)
        if not tiles_dir.exists():
            if args.from_existing:
                # Writing to a new directory — create it
                tiles_dir.mkdir(parents=True, exist_ok=True)
                print(f"  Created output directory: {tiles_dir}")
            else:
                tiles_dir = Path("data/lulc_full/tiles")
        if not tiles_dir.exists():
            tiles_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Created tiles directory: {tiles_dir}")

        print(f"\n  Found {len(land_cells)} land cells")

        if args.dry_run:
            # Count how many tiles would be fetched
            existing_local = _list_existing_tiles_local(tiles_dir)
            todo_count = sum(
                1 for c in land_cells
                if f"{c.easting}_{c.northing}" not in existing_local
            )
            print(f"  Already exist: {len(existing_local)}")
            print(f"  [DRY-RUN] Would fetch: {todo_count} tiles")
            return

        _run_local(
            cells=land_cells,
            tiles_dir=tiles_dir,
            years=years,
            windows=windows_list,
            cloud_threshold=args.cloud_threshold,
            haze_threshold=args.haze_threshold,
            num_classes=args.num_classes,
            tile_size_px=args.tile_size_px,
            workers=args.workers,
        )
        return

    # ── Status mode ──────────────────────────────────────────────────
    if args.status:
        existing = _list_existing_tiles_cfs(args.tiles_dir)
        print(f"\n  Completed tiles in CFS: {len(existing)}")
        print(f"  Total land cells: {len(land_cells)}")
        if land_cells:
            pct = len(existing) / len(land_cells) * 100
            print(f"  Progress: {pct:.1f}%")
        return

    # ── Check CFS for existing tiles ─────────────────────────────────
    print("  Checking CFS for existing tiles...")
    existing = _list_existing_tiles_cfs(args.tiles_dir)
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

    # ── Load CDSE credentials ────────────────────────────────────────
    cdse_client_id = os.environ.get("CDSE_CLIENT_ID", "")
    cdse_client_secret = os.environ.get("CDSE_CLIENT_SECRET", "")

    creds_path = Path(__file__).parent.parent / ".cdse_credentials"
    if creds_path.exists() and not cdse_client_id:
        lines = creds_path.read_text().strip().splitlines()
        if len(lines) >= 4:
            cdse_client_id = lines[2].strip()
            cdse_client_secret = lines[3].strip()

    has_cdse = bool(cdse_client_id and cdse_client_secret)
    print(f"\n  CDSE credentials: {'OK' if has_cdse else 'MISSING'}")

    if not has_cdse:
        print("  ERROR: CDSE credentials required for Process API")
        print("  Set CDSE_CLIENT_ID + CDSE_CLIENT_SECRET env vars")
        print("  or add client_id/secret to .cdse_credentials (lines 3-4)")
        sys.exit(1)

    # ── Submit jobs ──────────────────────────────────────────────────
    print(f"\n  Submitting {len(todo)} S2 jobs "
          f"({'DRY RUN' if args.dry_run else 'LIVE'})...")

    submitted = 0
    failed = 0

    for i, cell in enumerate(todo):
        spec = _build_job_spec(
            cell=cell,
            years=years,
            windows=args.windows,
            cloud_threshold=args.cloud_threshold,
            haze_threshold=args.haze_threshold,
            cfs_dir=args.tiles_dir,
            num_classes=args.num_classes,
            tile_size_px=args.tile_size_px,
            cdse_client_id=cdse_client_id,
            cdse_client_secret=cdse_client_secret,
        )

        if _submit_job(spec, dry_run=args.dry_run):
            submitted += 1
        else:
            failed += 1

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(todo)} submitted")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n  {'=' * 50}")
    print(f"  Submitted: {submitted}")
    print(f"  Failed:    {failed}")
    print(f"  Backend:   Sentinel Hub Process API (CDSE)")
    print(f"  {'=' * 50}")

    if not args.dry_run and submitted > 0:
        print(f"\n  Monitor progress:")
        print(f"    python scripts/submit_s2_jobs.py "
              f"--status --tiles-dir {args.tiles_dir}")


if __name__ == "__main__":
    main()
