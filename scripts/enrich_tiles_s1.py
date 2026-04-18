#!/usr/bin/env python3
"""Add Sentinel-1 SAR (VV/VH) to existing tiles.

For each tile, fetches S1 GRD scenes matching the S2 frame dates (±3 days).
Stores as separate keys in .npz — does not modify existing spectral data.

Idempotent: skips tiles with has_s1=1.

Uses separate CDSE S1 collection — does NOT compete with S2 rate limits.
Can run in parallel with the S2 spectral fetch pipeline.

Keys written:
    s1_vv_vh          (T*2, H, W) float32 — VV/VH per temporal frame
    s1_temporal_mask  (T,) uint8 — 1=valid S1 scene found, 0=no data
    s1_dates          (T,) object — actual S1 acquisition dates
    has_s1            int32 — 1 if any frame has S1 data

Usage:
    python scripts/enrich_tiles_s1.py \
        --data-dir /data/unified_v2_512 \
        --workers 6 \
        --skip-existing
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def enrich_one_tile(tile_path: str, skip_existing: bool = True) -> dict:
    """Add S1 VV/VH to one tile .npz file.

    Tile geometry is derived from the on-disk raster (``spectral`` shape)
    and either the persisted ``tile_size_px`` key or the pixel dimension
    itself. No module-level constants — fully size-agnostic.
    """
    from imint.training.cdse_s1 import fetch_s1_scene
    from imint.training.tile_config import TileConfig
    from imint.training.tile_bbox import resolve_tile_bbox

    name = Path(tile_path).stem
    try:
        data = dict(np.load(tile_path, allow_pickle=True))
    except Exception as e:
        return {"name": name, "status": "failed", "reason": str(e)}

    if skip_existing and int(data.get("has_s1", 0)) == 1:
        return {"name": name, "status": "skipped"}

    dates = data.get("dates", [])
    spectral = data.get("spectral", data.get("image"))
    if spectral is None:
        return {"name": name, "status": "failed", "reason": "no_spectral"}

    n_bands = 6
    h, w = spectral.shape[1], spectral.shape[2]
    n_frames = spectral.shape[0] // n_bands

    # Prefer persisted tile_size_px; fall back to raster dimension
    size_px = int(data.get("tile_size_px", h))
    tile_cfg = TileConfig(size_px=size_px)

    bbox = resolve_tile_bbox(name=name, tile=tile_cfg, npz_data=data)
    if bbox is None:
        return {"name": name, "status": "failed", "reason": "no_bbox"}
    tile_cfg.assert_bbox_matches(bbox)

    # Fetch S1 for each frame date (±3 day window)
    s1_frames = []
    s1_mask = []
    s1_dates_out = []

    for fi in range(n_frames):
        date_str = str(dates[fi])[:10] if fi < len(dates) and dates[fi] else ""
        if not date_str or date_str == "":
            s1_frames.append(np.zeros((2, h, w), dtype=np.float32))
            s1_mask.append(0)
            s1_dates_out.append("")
            continue

        # Search ±3 days for S1 scene
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        got_scene = False
        for offset in [0, -1, 1, -2, 2, -3, 3]:
            try_date = (dt + timedelta(days=offset)).strftime("%Y-%m-%d")
            try:
                result = fetch_s1_scene(
                    bbox["west"], bbox["south"], bbox["east"], bbox["north"],
                    date=try_date,
                    size_px=size_px,
                    nodata_threshold=0.30,
                )
                if result is not None:
                    sar, orbit = result
                    # Ensure correct shape
                    if sar.shape == (2, h, w):
                        s1_frames.append(sar)
                    else:
                        # Resize if needed
                        from scipy.ndimage import zoom
                        zoomed = np.stack([
                            zoom(sar[0], (h / sar.shape[1], w / sar.shape[2]), order=1),
                            zoom(sar[1], (h / sar.shape[1], w / sar.shape[2]), order=1),
                        ])
                        s1_frames.append(zoomed.astype(np.float32))
                    s1_mask.append(1)
                    s1_dates_out.append(try_date)
                    got_scene = True
                    break
            except Exception:
                continue

        if not got_scene:
            s1_frames.append(np.zeros((2, h, w), dtype=np.float32))
            s1_mask.append(0)
            s1_dates_out.append("")

    # Stack and save
    s1_vv_vh = np.concatenate(s1_frames, axis=0)  # (T*2, H, W)
    s1_temporal_mask = np.array(s1_mask, dtype=np.uint8)

    data["s1_vv_vh"] = s1_vv_vh
    data["s1_temporal_mask"] = s1_temporal_mask
    data["s1_dates"] = np.array(s1_dates_out)
    data["has_s1"] = np.int32(1 if sum(s1_mask) > 0 else 0)

    np.savez_compressed(tile_path, **data)

    valid = sum(s1_mask)
    return {"name": name, "status": "ok", "valid_frames": valid}


def main():
    parser = argparse.ArgumentParser(description="Enrich tiles with S1 SAR VV/VH")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--max-tiles", type=int, default=None)
    args = parser.parse_args()

    tiles = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if args.max_tiles:
        tiles = tiles[:args.max_tiles]
    print(f"=== S1 SAR Enrichment ===")
    print(f"  Tiles: {len(tiles)}")
    print(f"  Workers: {args.workers}")

    stats = {"ok": 0, "skipped": 0, "failed": 0}
    lock = threading.Lock()
    completed = 0
    t0 = time.time()

    def _run(path):
        nonlocal completed
        r = enrich_one_tile(path, skip_existing=args.skip_existing)
        with lock:
            completed += 1
            stats[r.get("status", "failed")] = stats.get(r.get("status", "failed"), 0) + 1
            elapsed = time.time() - t0
            rate = completed / elapsed * 3600 if elapsed > 0 else 0
            valid = r.get("valid_frames", "")
            reason = r.get("reason", "")
            reason_str = f" [{reason}]" if r.get("status") == "failed" and reason else ""
            print(f"  [{completed}/{len(tiles)}] {r['name']}: {r['status']}"
                  f"{f' ({valid}/4 frames)' if valid != '' else ''}"
                  f"{reason_str}"
                  f" | {rate:.0f}/h", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_run, t): t for t in tiles}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                print(f"  Error: {e}")

    elapsed = time.time() - t0
    print(f"\n=== Done in {elapsed/60:.1f} min ===")
    print(f"  OK={stats['ok']}  Skipped={stats['skipped']}  Failed={stats['failed']}")


if __name__ == "__main__":
    main()
