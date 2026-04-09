"""Add 2016 (or 2015) summer background frame to all existing unified tiles.

Each tile receives a new key ``frame_2016``: the single best Sentinel-2
scene found in the Jun 1–Aug 16 window for 2016.  If no qualifying 2016
scene is available the script falls back to 2015.  Tiles that already
carry ``has_frame_2016 = 1`` are skipped (idempotent).

Fetch strategy per tile:
    1. Query CDSE Catalog STAC for all S2A acquisitions in window
       (catalogue.dataspace.copernicus.eu/stac — covers 2015+, no auth)
    2. Sort candidates by scene cloud % ascending
    3. For each candidate (up to 5): fetch via Sentinel Hub Process API
       and check tile-level quality gates (valid_pct > 80 %, cloud < 15 %)
    4. Accept first passing scene; fall back to summer 2015 if needed
    5. On complete failure: write has_frame_2016 = 0 + zero array

Keys written to each .npz:
    frame_2016         (6, 256, 256) float32  Prithvi bands [0, 1]
    frame_2016_date    bytes "YYYY-MM-DD"
    frame_2016_doy     int32
    frame_2016_cloud_pct float32 (0–1 tile level)
    frame_2016_year    int32 (actual year used; 2015 or 2016)
    has_frame_2016     int32 (1 = success, 0 = failed)

Usage::

    python scripts/add_background_frame.py \\
        --data-dir /data/unified_v2 \\
        --workers 32 \\
        [--tile-ids 471280_6621280 443520_6283520 ...]
        [--skip-existing]   # default: True

Credentials (same as main fetch pipeline):
    CDSE_CLIENT_ID, CDSE_CLIENT_SECRET — OAuth2 for Process API
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import numpy as np

# Ensure the repo root is on sys.path so `imint` can be imported
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from imint.training.tile_fetch import (
    TILE_SIZE_PX,
    N_BANDS,
    fetch_background_frame,
)

# ── Counter (thread-safe) ─────────────────────────────────────────────────

_lock = threading.Lock()
_n_done = 0
_n_ok = 0
_n_fail = 0
_n_skip = 0


def _inc(done: int = 1, ok: int = 0, fail: int = 0, skip: int = 0) -> None:
    global _n_done, _n_ok, _n_fail, _n_skip
    with _lock:
        _n_done += done
        _n_ok += ok
        _n_fail += fail
        _n_skip += skip


# ── Per-tile worker ───────────────────────────────────────────────────────

def _process_tile(path: Path, skip_existing: bool) -> str:
    """Add ``frame_2016`` to a single tile .npz file.

    Returns a short status string (for logging).
    """
    name = path.stem  # e.g. "tile_471280_6621280"

    # Load existing data
    try:
        data = dict(np.load(path, allow_pickle=False))
    except Exception as exc:
        _inc(done=1, fail=1)
        return f"LOAD_ERR {name}: {exc}"

    # Skip if already done
    if skip_existing and int(data.get("has_frame_2016", 0)) == 1:
        _inc(done=1, skip=1)
        return f"SKIP {name}"

    # Derive bbox_3006
    bbox_3006 = _get_bbox(data, name)
    if bbox_3006 is None:
        _inc(done=1, fail=1)
        return f"NO_BBOX {name}"

    # Fetch background frame
    result = fetch_background_frame(bbox_3006)

    if result is not None:
        data.update(result)
        status = (
            f"OK {name}: {result['frame_2016_date'].decode()} "
            f"year={int(result['frame_2016_year'])} "
            f"cloud={float(result['frame_2016_cloud_pct']):.0%}"
        )
        ok = 1
    else:
        # Write sentinel so we don't retry endlessly
        data["has_frame_2016"] = np.int32(0)
        data["frame_2016"] = np.zeros((N_BANDS, TILE_SIZE_PX, TILE_SIZE_PX), dtype=np.float32)
        data["frame_2016_date"] = np.bytes_("")
        data["frame_2016_doy"] = np.int32(0)
        data["frame_2016_cloud_pct"] = np.float32(0.0)
        data["frame_2016_year"] = np.int32(0)
        status = f"FAIL {name}: no qualifying scene (2016 + 2015)"
        ok = 0

    # Atomic write: save to temp file then rename.
    # NOTE: np.savez_compressed appends ".npz" only if the filename does NOT
    # already end in ".npz".  Use ".tmp.npz" so numpy writes to exactly that
    # path, then rename to the final ".npz" destination.
    tmp_path = path.with_name(path.stem + ".tmp.npz")
    try:
        np.savez_compressed(tmp_path, **data)
        tmp_path.rename(path)
    except Exception as exc:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        _inc(done=1, fail=1)
        return f"WRITE_ERR {name}: {exc}"

    _inc(done=1, ok=ok, fail=1 - ok)
    return status


def _get_bbox(data: dict, name: str) -> dict | None:
    """Extract bbox_3006 from tile data or filename."""
    # Method 1: bbox_3006 array stored in tile
    if "bbox_3006" in data:
        b = np.asarray(data["bbox_3006"]).ravel()
        if b.shape[0] >= 4:
            return {"west": int(b[0]), "south": int(b[1]),
                    "east": int(b[2]), "north": int(b[3])}

    # Method 2: easting/northing scalars
    if "easting" in data and "northing" in data:
        cx = int(data["easting"])
        cy = int(data["northing"])
        half = 1280
        return {"west": cx - half, "south": cy - half,
                "east": cx + half, "north": cy + half}

    # Method 3: parse from filename  "tile_<easting>_<northing>.npz"
    import re
    m = re.search(r"tile_(\d+)_(\d+)", name)
    if m:
        cx, cy = int(m.group(1)), int(m.group(2))
        half = 1280
        return {"west": cx - half, "south": cy - half,
                "east": cx + half, "north": cy + half}

    # Method 4: old-style numeric filename "<id>.npz"
    # These tiles have easting/northing stored differently
    return None


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Add 2016 summer background frame to all unified tiles."
    )
    p.add_argument("--data-dir", required=True,
                   help="Directory containing unified_v2 .npz tiles")
    p.add_argument("--workers", type=int, default=32,
                   help="Parallel worker threads (default: 32)")
    p.add_argument("--tile-ids", nargs="+", default=None,
                   help="Process only these tile IDs (e.g. 471280_6621280)")
    p.add_argument("--no-skip-existing", action="store_true",
                   help="Re-fetch even if has_frame_2016 == 1 already")
    p.add_argument("--primary-year", type=int, default=2016)
    p.add_argument("--fallback-year", type=int, default=2015)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        sys.exit(f"ERROR: --data-dir {data_dir} does not exist")

    skip_existing = not args.no_skip_existing

    # Collect tiles
    all_paths = sorted(data_dir.glob("*.npz"))
    if not all_paths:
        sys.exit(f"ERROR: no .npz files found in {data_dir}")

    if args.tile_ids:
        id_set = set(args.tile_ids)
        # Match by substring (tile ID appears in stem)
        all_paths = [
            p for p in all_paths
            if any(tid in p.stem for tid in id_set)
        ]
        if not all_paths:
            sys.exit(f"ERROR: none of the specified tile IDs found")

    total = len(all_paths)
    print(f"=== Add Background Frame ===")
    print(f"  data_dir     : {data_dir}")
    print(f"  tiles        : {total:,}")
    print(f"  workers      : {args.workers}")
    print(f"  primary year : {args.primary_year}")
    print(f"  fallback year: {args.fallback_year}")
    print(f"  skip existing: {skip_existing}")
    print()

    t0 = time.time()
    log_interval = max(1, total // 20)   # log every ~5 %

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_process_tile, path, skip_existing): path
            for path in all_paths
        }
        for i, fut in enumerate(as_completed(futures), 1):
            status = fut.result()
            if i % log_interval == 0 or i == total:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total - i) / rate if rate > 0 else 0
                with _lock:
                    n_ok, n_fail, n_skip = _n_ok, _n_fail, _n_skip
                print(
                    f"  [{i:5d}/{total}] {status[:90]}  "
                    f"| ok={n_ok} fail={n_fail} skip={n_skip} "
                    f"| {rate:.1f} t/s | ETA {eta/60:.1f}m"
                )
            elif "FAIL" in status or "ERR" in status:
                print(f"  [{i:5d}/{total}] {status}")

    elapsed = time.time() - t0
    with _lock:
        n_ok, n_fail, n_skip = _n_ok, _n_fail, _n_skip
    print()
    print(f"=== Done in {elapsed/60:.1f} min ===")
    print(f"  OK   : {n_ok:,}")
    print(f"  Fail : {n_fail:,}")
    print(f"  Skip : {n_skip:,}")
    print(f"  Total: {total:,}")
    if n_fail > 0:
        print(f"  NOTE: {n_fail} tiles written with has_frame_2016=0 — "
              "can be retried by running with --no-skip-existing")


if __name__ == "__main__":
    main()
