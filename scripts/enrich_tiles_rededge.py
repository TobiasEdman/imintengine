#!/usr/bin/env python3
"""Add Sentinel-2 red-edge bands (B05, B06, B07) to existing tiles.

Clay v1.5 expects 10 S2 bands; we already have 7 on disk (6-band
Prithvi spectral + B08). This script fills the remaining 3:
    B05 (rededge1, 705 nm, 20 m)
    B06 (rededge2, 740 nm, 20 m)
    B07 (rededge3, 783 nm, 20 m)

Sentinel Hub resamples 20 m bands to the tile's pixel grid (10 m), so
the output is 512×512 aligned with spectral. Uses a single evalscript
fetching all 3 bands per call → 3× API quota savings vs per-band.

Idempotent: skips tiles with has_rededge=1.

Keys written:
    rededge       (T*3, H, W) float32 — B05, B06, B07 per temporal frame
    has_rededge   int32 — 1 if any frame has red-edge data

Usage:
    python scripts/enrich_tiles_rededge.py \\
        --data-dir /data/unified_v2_512 \\
        --workers 4 \\
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
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


_REDEDGE_EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B05", "B06", "B07"], units: "DN" }],
    output: { bands: 3, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(sample) {
  return [sample.B05 / 10000, sample.B06 / 10000, sample.B07 / 10000];
}
"""


def _fetch_rededge_frame(
    west: float, south: float, east: float, north: float,
    date_str: str, size_px: int,
) -> np.ndarray | None:
    """Fetch B05/B06/B07 (3 bands) for one date via CDSE Process API.

    Returns (3, H, W) float32 or None on failure.
    """
    from imint.training.cdse_s2 import _fetch_s2_tiff, _parse_multiband_tiff, _get_token
    from imint.training.tile_fetch import _CDSE_SEMAPHORE

    _CDSE_SEMAPHORE.acquire()
    try:
        token = _get_token()
        tiff_bytes = _fetch_s2_tiff(
            west, south, east, north, size_px, size_px,
            date=date_str, token=token,
            evalscript=_REDEDGE_EVALSCRIPT,
        )
        bands = _parse_multiband_tiff(tiff_bytes, size_px, size_px, 3)
        if bands and len(bands) >= 3:
            _CDSE_SEMAPHORE.report_success()
            return np.stack(bands[:3], axis=0).astype(np.float32)
        _CDSE_SEMAPHORE.report_success()
    except Exception:
        _CDSE_SEMAPHORE.report_failure()
    finally:
        _CDSE_SEMAPHORE.release()
    return None


def enrich_one_tile(tile_path: str, skip_existing: bool = True) -> dict:
    """Add red-edge (B05/B06/B07) to one tile .npz file."""
    from imint.training.tile_config import TileConfig
    from imint.training.tile_bbox import resolve_tile_bbox

    name = Path(tile_path).stem
    try:
        data = dict(np.load(tile_path, allow_pickle=True))
    except Exception as e:
        return {"name": name, "status": "failed", "reason": str(e)}

    if skip_existing and int(data.get("has_rededge", 0)) == 1:
        return {"name": name, "status": "skipped"}

    dates = data.get("dates", [])
    spectral = data.get("spectral", data.get("image"))
    if spectral is None:
        return {"name": name, "status": "failed", "reason": "no_spectral"}

    h, w = spectral.shape[1], spectral.shape[2]
    n_frames = spectral.shape[0] // 6

    size_px = int(data.get("tile_size_px", h))
    tile_cfg = TileConfig(size_px=size_px)

    bbox = resolve_tile_bbox(name=name, tile=tile_cfg, npz_data=data)
    if bbox is None:
        return {"name": name, "status": "failed", "reason": "no_bbox"}
    tile_cfg.assert_bbox_matches(bbox)

    rededge_frames = []  # each (3, H, W) float32
    valid = 0
    for fi in range(n_frames):
        date_str = str(dates[fi])[:10] if fi < len(dates) and dates[fi] else ""
        if not date_str:
            rededge_frames.append(np.zeros((3, h, w), dtype=np.float32))
            continue

        frame = _fetch_rededge_frame(
            bbox["west"], bbox["south"], bbox["east"], bbox["north"],
            date_str, size_px,
        )
        if frame is not None and frame.shape == (3, h, w):
            rededge_frames.append(frame)
            valid += 1
        else:
            rededge_frames.append(np.zeros((3, h, w), dtype=np.float32))

    # Stack along band × frame axis → (T*3, H, W) matching spectral convention
    # Frame 0: B05, B06, B07; Frame 1: B05, B06, B07; ...
    data["rededge"] = np.concatenate(rededge_frames, axis=0)  # (T*3, H, W)
    data["has_rededge"] = np.int32(1 if valid > 0 else 0)

    np.savez_compressed(tile_path, **data)
    return {"name": name, "status": "ok", "valid_frames": valid}


def main():
    parser = argparse.ArgumentParser(description="Enrich tiles with S2 red-edge (B05/B06/B07)")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--max-tiles", type=int, default=None)
    args = parser.parse_args()

    tiles = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if args.max_tiles:
        tiles = tiles[:args.max_tiles]
    print(f"=== Red-Edge Enrichment (B05/B06/B07) ===")
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
