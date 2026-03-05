#!/usr/bin/env python3
"""
scripts/submit_vpp_jobs.py — ColonyOS job coordinator for VPP enrichment.

Scans existing training tiles (from seasonal fetch) and submits ColonyOS
jobs to add HR-VPP vegetation phenology channels to each tile.

Resumable: tiles that already contain VPP channels are skipped.

Usage:
    # Dry-run: see what jobs would be submitted
    python scripts/submit_vpp_jobs.py --dry-run

    # Submit first 100 VPP jobs
    python scripts/submit_vpp_jobs.py --max-jobs 100

    # Submit all
    python scripts/submit_vpp_jobs.py

    # Check progress
    python scripts/submit_vpp_jobs.py --status

    # Local mode (no ColonyOS — uses ThreadPoolExecutor directly)
    python scripts/submit_vpp_jobs.py --local --workers 4
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

# VPP band names stored in .npz tiles
_VPP_CHANNEL_NAMES = ["vpp_sosd", "vpp_eosd", "vpp_length", "vpp_maxv", "vpp_minv"]
_TILE_RE = re.compile(r"tile_(\d+)_(\d+)\.npz$")


def _list_cfs_tiles(cfs_dir: str) -> list[tuple[int, int]]:
    """List tile (easting, northing) pairs from CFS."""
    try:
        result = subprocess.run(
            ["colonies", "fs", "ls", "--label", cfs_dir, "--insecure"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            print(f"  Warning: colonies fs list failed: {result.stderr.strip()}")
            return []

        tiles = []
        for line in result.stdout.strip().splitlines():
            name = line.strip().split()[-1] if line.strip() else ""
            m = _TILE_RE.search(name)
            if m:
                tiles.append((int(m.group(1)), int(m.group(2))))
        return tiles
    except FileNotFoundError:
        print("  Warning: 'colonies' CLI not found")
        return []
    except subprocess.TimeoutExpired:
        print("  Warning: colonies fs list timed out")
        return []


def _list_local_tiles(tiles_dir: Path) -> list[tuple[int, int]]:
    """List tile (easting, northing) pairs from a local directory."""
    tiles = []
    for f in sorted(tiles_dir.glob("tile_*.npz")):
        m = _TILE_RE.search(f.name)
        if m:
            tiles.append((int(m.group(1)), int(m.group(2))))
    return tiles


def _check_tile_has_vpp(tiles_dir: Path, easting: int, northing: int) -> bool:
    """Check if a local tile already has VPP channels."""
    import numpy as np
    tile_path = tiles_dir / f"tile_{easting}_{northing}.npz"
    if not tile_path.exists():
        return False
    try:
        with np.load(tile_path, allow_pickle=True) as data:
            return all(ch in data for ch in _VPP_CHANNEL_NAMES)
    except Exception:
        return False


def _build_vpp_job_spec(
    easting: int,
    northing: int,
    cfs_dir: str,
    vpp_year: int,
    patch_size_m: int,
    cdse_client_id: str,
    cdse_client_secret: str,
) -> dict:
    """Build a ColonyOS job spec for one VPP fetch job."""
    return {
        "conditions": {
            "colonyname": "imint",
            "executortype": "container-executor",
            "nodes": 1,
            "processes": 1,
            "processespernode": 1,
            "walltime": 300,
            "cpu": "500m",
            "mem": "1Gi",
        },
        "env": {
            "PYTHONUNBUFFERED": "1",
            "EASTING": str(easting),
            "NORTHING": str(northing),
            "TILES_DIR": "/cfs/tiles",
            "VPP_YEAR": str(vpp_year),
            "PATCH_SIZE_M": str(patch_size_m),
            "CDSE_CLIENT_ID": cdse_client_id,
            "CDSE_CLIENT_SECRET": cdse_client_secret,
        },
        "funcname": "execute",
        "kwargs": {
            "docker-image": "localhost:5000/imint-engine:latest",
            "rebuild-image": False,
            "cmd": "python executors/vpp_fetch.py",
        },
        "maxexectime": 300,
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
        print(f"  [DRY-RUN] Would submit VPP: {cell_key}")
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
    tiles: list[tuple[int, int]],
    tiles_dir: Path,
    vpp_year: int,
    patch_size_m: int,
    workers: int,
) -> None:
    """Run VPP fetching locally with ThreadPoolExecutor (no ColonyOS)."""
    import numpy as np
    from imint.training.cdse_vpp import fetch_vpp_tiles

    half_m = patch_size_m // 2

    # Filter to tiles that need VPP
    todo = []
    for easting, northing in tiles:
        if not _check_tile_has_vpp(tiles_dir, easting, northing):
            todo.append((easting, northing))

    print(f"  Tiles needing VPP: {len(todo)}")
    if not todo:
        print("  All tiles already have VPP channels!")
        return

    t_start = time.time()
    stats = {"ok": 0, "fail": 0, "skip": 0}

    def _fetch_one(item):
        easting, northing = item
        tile_path = tiles_dir / f"tile_{easting}_{northing}.npz"

        if not tile_path.exists():
            return {"status": "skip", "key": f"{easting}_{northing}"}

        west = easting - half_m
        south = northing - half_m
        east = easting + half_m
        north = northing + half_m

        # Load existing tile
        with np.load(tile_path, allow_pickle=True) as data:
            existing = dict(data)
            if all(ch in existing for ch in _VPP_CHANNEL_NAMES):
                return {"status": "skip", "key": f"{easting}_{northing}"}
            img = existing.get("image")
            if img is not None and img.ndim >= 2:
                h_px, w_px = img.shape[-2], img.shape[-1]
            else:
                h_px, w_px = 256, 256

        # Fetch VPP
        vpp = fetch_vpp_tiles(
            west, south, east, north,
            size_px=(h_px, w_px),
            year=vpp_year,
        )

        # Add to tile
        for raw_name, arr in vpp.items():
            existing[f"vpp_{raw_name}"] = arr

        # Atomic rewrite
        tmp_path = tile_path.with_suffix(".vpp_tmp.npz")
        np.savez_compressed(tmp_path, **existing)
        tmp_path.rename(tile_path)

        return {
            "status": "ok",
            "key": f"{easting}_{northing}",
            "maxv_mean": float(vpp["maxv"].mean()),
        }

    print(f"\n  Fetching VPP for {len(todo)} tiles ({workers} workers)...\n")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one, item): item for item in todo}

        for i, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
                status = result["status"]
                stats[status] = stats.get(status, 0) + 1

                if status == "ok" and i <= 5:
                    print(f"  ✓ {result['key']}: maxv_mean={result['maxv_mean']:.4f}")
                elif status == "fail":
                    print(f"  ✗ {result['key']}: {result.get('error', '?')}")

            except Exception as e:
                stats["fail"] += 1
                key = f"{futures[future][0]}_{futures[future][1]}"
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
    print(f"  VPP enrichment complete!")
    print(f"  OK:      {stats['ok']}")
    print(f"  Failed:  {stats['fail']}")
    print(f"  Skipped: {stats['skip']}")
    print(f"  Time:    {elapsed/60:.1f} min")
    print(f"  Rate:    {stats['ok']/max(elapsed, 1):.1f} tiles/s")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Submit VPP phenology enrichment jobs to ColonyOS"
    )
    parser.add_argument(
        "--tiles-dir", default="seasonal-tiles",
        help="CFS directory (or local dir with --local) for tiles "
             "(default: seasonal-tiles)",
    )
    parser.add_argument(
        "--vpp-year", type=int, default=2021,
        help="VPP product year (default: 2021)",
    )
    parser.add_argument(
        "--patch-size-m", type=int, default=2560,
        help="Tile spatial extent in meters (default: 2560)",
    )
    parser.add_argument(
        "--max-jobs", type=int, default=0,
        help="Max jobs to submit (0 = unlimited, default: 0)",
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
    args = parser.parse_args()

    print("=" * 60)
    print("  VPP Phenology Enrichment — ColonyOS Coordinator")
    print(f"  VPP Year: {args.vpp_year}")
    print(f"  Mode: {'LOCAL' if args.local else 'ColonyOS'}")
    print("=" * 60)

    # ── Local mode ────────────────────────────────────────────────────
    if args.local:
        tiles_dir = Path(args.tiles_dir)
        if not tiles_dir.exists():
            # Try as relative to data dir
            tiles_dir = Path("data/lulc_full/tiles")
        if not tiles_dir.exists():
            print(f"  ERROR: Tiles directory not found: {tiles_dir}")
            sys.exit(1)

        tiles = _list_local_tiles(tiles_dir)
        print(f"\n  Found {len(tiles)} tiles in {tiles_dir}")
        _run_local(tiles, tiles_dir, args.vpp_year, args.patch_size_m, args.workers)
        return

    # ── Status mode ───────────────────────────────────────────────────
    if args.status:
        tiles = _list_cfs_tiles(args.tiles_dir)
        # We can't easily check VPP presence in CFS without downloading
        print(f"\n  Total tiles in CFS: {len(tiles)}")
        print(f"  (VPP enrichment status requires local tile inspection)")
        return

    # ── ColonyOS submission ───────────────────────────────────────────
    tiles = _list_cfs_tiles(args.tiles_dir)
    print(f"\n  Found {len(tiles)} tiles in CFS")

    if not tiles:
        print("  No tiles found — run seasonal fetch first")
        return

    if args.max_jobs > 0:
        tiles = tiles[:args.max_jobs]
        print(f"  Limiting to {args.max_jobs} jobs")

    # Load CDSE credentials
    cdse_client_id = os.environ.get("CDSE_CLIENT_ID", "")
    cdse_client_secret = os.environ.get("CDSE_CLIENT_SECRET", "")

    creds_path = Path(__file__).parent.parent / ".cdse_credentials"
    if creds_path.exists() and not cdse_client_id:
        lines = creds_path.read_text().strip().splitlines()
        if len(lines) >= 4:
            cdse_client_id = lines[2].strip()
            cdse_client_secret = lines[3].strip()

    if not cdse_client_id or not cdse_client_secret:
        print("  ERROR: CDSE credentials required for VPP fetch")
        print("  Set CDSE_CLIENT_ID + CDSE_CLIENT_SECRET env vars")
        print("  or add client_id/secret to .cdse_credentials (lines 3-4)")
        sys.exit(1)

    print(f"  CDSE credentials: OK")

    # ── Submit jobs ───────────────────────────────────────────────────
    print(f"\n  Submitting {len(tiles)} VPP jobs "
          f"({'DRY RUN' if args.dry_run else 'LIVE'})...")

    submitted = 0
    failed = 0

    for i, (easting, northing) in enumerate(tiles):
        spec = _build_vpp_job_spec(
            easting=easting,
            northing=northing,
            cfs_dir=args.tiles_dir,
            vpp_year=args.vpp_year,
            patch_size_m=args.patch_size_m,
            cdse_client_id=cdse_client_id,
            cdse_client_secret=cdse_client_secret,
        )

        if _submit_job(spec, dry_run=args.dry_run):
            submitted += 1
        else:
            failed += 1

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(tiles)} submitted")

    print(f"\n  {'=' * 50}")
    print(f"  Submitted: {submitted}")
    print(f"  Failed:    {failed}")
    print(f"  {'=' * 50}")


if __name__ == "__main__":
    main()
