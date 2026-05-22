#!/usr/bin/env python3
"""One-time backfill: populate b08_dates / rededge_dates on existing tiles.

The D3 date-aware-mismatch design (enrich_tiles_b08 + enrich_tiles_rededge)
needs a per-frame ``b08_dates`` / ``rededge_dates`` marker to detect
spectral-date drift and re-fetch just the affected slots. Tiles enriched
before D3 don't have these fields — but their aux data WAS aligned to the
spectral ``dates`` field at enrichment time, so on backfill we can simply
copy ``dates`` into the missing markers without any new fetches.

Idempotent: skips tiles where both markers already exist (or where the
aux field itself is missing — nothing to mark).

Usage (k8s pod):
    python scripts/backfill_aux_dates.py \\
        --data-dir /cephfs/unified_v2_512 \\
        --workers 8

Atomic write: writes to FOO.tmp.npz, renames onto FOO.npz.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def backfill_one_tile(tile_path: str) -> dict:
    """Add b08_dates / rededge_dates to one .npz if missing."""
    name = Path(tile_path).stem
    try:
        data = dict(np.load(tile_path, allow_pickle=True))
    except Exception as e:
        return {"name": name, "status": "failed", "reason": str(e)[:200]}

    if "dates" not in data:
        return {"name": name, "status": "skipped", "reason": "no_dates"}

    dates = [str(d)[:10] for d in data["dates"]]
    changed = False

    if "b08" in data and "b08_dates" not in data:
        data["b08_dates"] = np.array(dates)
        changed = True

    if "rededge" in data and "rededge_dates" not in data:
        data["rededge_dates"] = np.array(dates)
        changed = True

    if not changed:
        return {"name": name, "status": "skipped",
                "reason": "already_has_markers_or_no_aux"}

    # Atomic write
    tmp_base = tile_path + ".tmp"
    np.savez_compressed(tmp_base, **data)
    os.replace(tmp_base + ".npz", tile_path)
    return {"name": name, "status": "ok", "added": [
        k for k in ("b08_dates", "rededge_dates") if k in data
    ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True,
                        help="Directory of .npz tiles (e.g. /cephfs/unified_v2_512)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-tiles", type=int, default=None,
                        help="For testing; processes the first N tiles only")
    args = parser.parse_args()

    tiles = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if args.max_tiles:
        tiles = tiles[:args.max_tiles]

    print(f"=== backfill aux dates ===", flush=True)
    print(f"  data-dir: {args.data_dir}", flush=True)
    print(f"  tiles:    {len(tiles)}", flush=True)
    print(f"  workers:  {args.workers}", flush=True)

    stats = {"ok": 0, "skipped": 0, "failed": 0}
    lock = threading.Lock()
    completed = 0
    t0 = time.time()

    def _run(path):
        nonlocal completed
        r = backfill_one_tile(path)
        with lock:
            completed += 1
            stats[r.get("status", "failed")] = stats.get(r.get("status", "failed"), 0) + 1
            if completed % 200 == 0 or completed == len(tiles):
                elapsed = time.time() - t0
                rate = completed / max(elapsed / 3600, 1e-6)
                print(f"  [{completed}/{len(tiles)}] status={stats} rate={rate:.0f}/h", flush=True)
        return r

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(as_completed([ex.submit(_run, p) for p in tiles]))

    elapsed = time.time() - t0
    print(f"\n=== done in {elapsed/60:.1f} min ===", flush=True)
    print(f"  by status: {stats}", flush=True)


if __name__ == "__main__":
    main()
