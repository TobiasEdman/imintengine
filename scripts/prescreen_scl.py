#!/usr/bin/env python3
"""Pre-screen tiles via SH Process API SCL-only fetch.

For each tile without a best_dates entry, fetches SCL band (250 KB per scene)
for all candidate dates in each VPP frame window. Counts cloud pixels locally,
picks best date per frame. Saves to best_dates.json (resumable).

Runs independently of fetch_unified_tiles.py — both read/write best_dates.json.
The fetch script picks up new entries on restart.

Usage:
    python scripts/prescreen_scl.py \
        --tile-locations /data/tile_locations_full.json \
        --vpp-cache /data/unified_v2_512/.vpp_cache.json \
        --output-dir /data/unified_v2_512 \
        --workers 6 \
        --year 2022
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SCL_CLOUD_CLASSES = {3, 8, 9, 10}


def doy_to_date(year: int, doy: int) -> str:
    return (datetime(year, 1, 1) + timedelta(days=max(doy - 1, 0))).strftime("%Y-%m-%d")


def screen_tile(tile: dict, frame_windows: list, year: int) -> dict:
    """Screen one tile: fetch SCL per candidate date, pick clearest per frame.

    Uses SH Process API SCL-only fetch (250 KB per scene) + STAC catalog
    for candidate dates. No spectral data downloaded.

    Returns: {frame_idx: {date, cloud_frac}} or empty dict.
    """
    from imint.training.tile_fetch import (
        TILE_SIZE_PX, bbox_3006_to_wgs84, _CDSE_SEMAPHORE,
    )
    from imint.training.cdse_s2 import _prescreen_scl, _get_token
    from imint.fetch import _stac_available_dates

    bbox = tile["bbox_3006"]
    coords = bbox_3006_to_wgs84(bbox)

    results = {}
    for fi, (doy_start, doy_end) in enumerate(frame_windows):
        date_start = doy_to_date(year, max(doy_start, 1))
        date_end = doy_to_date(year, min(doy_end, 365))

        # Get candidate dates from STAC catalog (metadata only, fast)
        try:
            candidates = _stac_available_dates(
                coords, date_start, date_end, scene_cloud_max=50,
            )
        except Exception:
            candidates = []

        if not candidates:
            # Generate synthetic candidates every 5 days
            d0 = datetime.strptime(date_start, "%Y-%m-%d")
            d1 = datetime.strptime(date_end, "%Y-%m-%d")
            candidates = [
                ((d0 + timedelta(days=i)).strftime("%Y-%m-%d"), 50.0)
                for i in range(0, (d1 - d0).days + 1, 5)
            ]

        candidates.sort(key=lambda x: x[1])

        # Check top 5 candidates via SCL-only fetch
        best_date = None
        best_frac = 1.0
        token = _get_token()

        for cand_date, _scene_cloud in candidates[:5]:
            _CDSE_SEMAPHORE.acquire()
            try:
                scl, cloud_frac = _prescreen_scl(
                    bbox["west"], bbox["south"], bbox["east"], bbox["north"],
                    cand_date,
                    TILE_SIZE_PX, TILE_SIZE_PX,
                    token, "http://www.opengis.net/def/crs/EPSG/0/3006",
                    cloud_threshold=1.0,  # don't reject, just measure
                )
                _CDSE_SEMAPHORE.report_success()
            except Exception:
                _CDSE_SEMAPHORE.report_failure()
                cloud_frac = 1.0
            finally:
                _CDSE_SEMAPHORE.release()

            if cloud_frac < best_frac:
                best_frac = cloud_frac
                best_date = cand_date

            # Good enough — don't check more
            if best_frac < 0.05:
                break

        if best_date:
            results[str(fi)] = {
                "date": best_date,
                "cloud_frac": round(best_frac, 4),
            }

    return results


def main():
    parser = argparse.ArgumentParser(description="SCL pre-screening via SH Process API")
    parser.add_argument("--tile-locations", required=True)
    parser.add_argument("--vpp-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--year", type=int, default=2022)
    args = parser.parse_args()

    with open(args.tile_locations) as f:
        all_tiles = json.load(f)
    with open(args.vpp_cache) as f:
        vpp_cache = json.load(f)

    output_dir = Path(args.output_dir)
    best_dates_path = output_dir / "best_dates.json"

    # Load existing
    best_dates = {}
    if best_dates_path.exists():
        with open(best_dates_path) as f:
            best_dates = json.load(f)

    # Filter: skip already-screened and already-fetched tiles
    existing = {p.stem for p in output_dir.glob("*.npz")}
    tiles = [
        t for t in all_tiles
        if t["name"] not in best_dates and t["name"] not in existing
    ]
    print(f"Total: {len(all_tiles)}, existing: {len(existing)}, "
          f"already screened: {len(best_dates)}, to screen: {len(tiles)}")

    if not tiles:
        print("Nothing to screen")
        return

    # Override tile size for 512px
    import imint.training.tile_fetch as _tf
    if args.output_dir and "512" in args.output_dir:
        _tf.TILE_SIZE_PX = 512
        _tf.TILE_SIZE_M = 5120
        print(f"Tile size: {_tf.TILE_SIZE_PX}px")

    lock = threading.Lock()
    completed = 0
    t0 = time.time()

    def _screen_one(tile):
        nonlocal completed
        name = tile["name"]
        windows = vpp_cache.get(name)
        if not windows:
            windows = [[91, 150], [151, 210], [211, 270]]
        result = screen_tile(tile, windows, args.year)
        with lock:
            completed += 1
            if result:
                best_dates[name] = result
                elapsed = time.time() - t0
                rate = completed / elapsed * 3600
                dates = " ".join(f"f{k}={v['date']}({v['cloud_frac']:.0%})"
                                 for k, v in result.items())
                print(f"[{completed}/{len(tiles)}] {name}: {dates} | {rate:.0f}/h",
                      flush=True)
            else:
                print(f"[{completed}/{len(tiles)}] {name}: no clear scenes", flush=True)
            # Save every 20 tiles
            if completed % 20 == 0:
                with open(best_dates_path, "w") as f:
                    json.dump(best_dates, f)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_screen_one, t): t for t in tiles}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                print(f"  Error: {e}")

    # Final save
    with open(best_dates_path, "w") as f:
        json.dump(best_dates, f, indent=2)
    print(f"\n=== Done: {len(best_dates)} tiles screened ===")


if __name__ == "__main__":
    main()
