#!/usr/bin/env python3
"""Add Sentinel-1 SAR (VV/VH) to existing tiles.

For each tile, fetches S1 GRD scenes matching the S2 frame dates (±3 days).
Stores as separate keys in .npz — does not modify existing spectral data.

Idempotent: skips tiles with has_s1=1.

Backends (``--s1-backend``):
    sh    Sentinel Hub Process API via ``imint.training.cdse_s1``.
          Bills CDSE Processing Units (10k/month free tier, 100 PU
          minimum per S1 request — unsuitable for bulk enrichment at
          our scale).
    stac  CDSE STAC + direct COG + local σ⁰ calibration via
          ``imint.training.cdse_s1_stac``. Bills OData bandwidth
          (12 TB/month) and HTTP COG requests (50k/month). Requires
          ``pystac-client`` + ``scipy`` + ``rasterio``. Preferred for
          full dataset enrichment — see docs/training/s1_fetch.md.

Atomicity:
    Writes go to ``<tile>.npz.tmp`` and are atomically renamed on
    success. Killing the job mid-write leaves the original .npz
    untouched. Stale .tmp files from prior aborted runs are deleted
    on next pass.

Keys written:
    s1_vv_vh          (T*2, H, W) float32 — VV/VH per temporal frame
    s1_temporal_mask  (T,) uint8 — 1=valid S1 scene found, 0=no data
    s1_dates          (T,) object — actual S1 acquisition dates
    has_s1            int32 — 1 if any frame has S1 data

Usage:
    python scripts/enrich_tiles_s1.py \\
        --data-dir /data/unified_v2_512 \\
        --s1-backend stac \\
        --workers 6 \\
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


def _resolve_fetch_fn(backend: str):
    """Return the ``fetch_s1_scene`` implementation for the given backend.

    Raises ValueError for unknown names. Import errors for optional
    extras (pystac-client, scipy) surface here rather than per tile.
    """
    if backend == "sh":
        from imint.training.cdse_s1 import fetch_s1_scene
        return fetch_s1_scene
    if backend == "stac":
        from imint.training.cdse_s1_stac import fetch_s1_scene
        return fetch_s1_scene
    raise ValueError(
        f"Unknown --s1-backend {backend!r}. "
        "Valid choices: 'sh' (Sentinel Hub Process API), "
        "'stac' (CDSE STAC + direct COG)."
    )


def enrich_one_tile(
    tile_path: str,
    *,
    fetch_s1_scene,
    skip_existing: bool = True,
) -> dict:
    """Add S1 VV/VH to one tile .npz file.

    Tile geometry is derived from the on-disk raster (``spectral`` shape)
    and either the persisted ``tile_size_px`` key or the pixel dimension
    itself. No module-level constants — fully size-agnostic.

    Writes atomically via ``<tile>.npz.tmp`` → ``os.replace(...)`` so
    that killing the job mid-write leaves the original tile intact
    instead of producing the BadZipFile / EOF-truncated archives we
    saw after the previous aborted run.
    """
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

    # Atomic write: compress into <tile>.npz.tmp, fsync, rename.
    # np.savez_compressed appends ".npz" automatically unless the path
    # already ends with ".npz", so give it a path without extension and
    # rename the produced file.
    tmp_path = tile_path + ".tmp.npz"
    try:
        np.savez_compressed(tmp_path[:-4], **data)  # strips .npz, then re-adds
        # Best-effort flush before rename. savez_compressed closes the
        # file before returning; rename itself is atomic on POSIX.
        os.replace(tmp_path, tile_path)
    except Exception:
        # Never leave a half-written .tmp.npz behind to confuse the next run.
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise

    valid = sum(s1_mask)
    return {"name": name, "status": "ok", "valid_frames": valid}


def main():
    parser = argparse.ArgumentParser(description="Enrich tiles with S1 SAR VV/VH")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument(
        "--s1-backend", choices=["sh", "stac"], default="sh",
        help="S1 fetch backend. 'sh' = Sentinel Hub Process API (PU-billed, "
             "blown the 10k/month free quota), 'stac' = CDSE STAC + direct "
             "COG + local σ⁰ calibration (OData bandwidth quota, preferred "
             "for bulk enrichment). See docs/training/s1_fetch.md.",
    )
    args = parser.parse_args()

    tiles = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if args.max_tiles:
        tiles = tiles[:args.max_tiles]

    # Clean up stale .tmp.npz files from any prior aborted run — never
    # let half-written archives poison skip_existing or get picked up
    # as real tiles.
    stale = glob.glob(os.path.join(args.data_dir, "*.npz.tmp.npz"))
    stale += glob.glob(os.path.join(args.data_dir, "*.tmp.npz"))
    for s in stale:
        try:
            os.unlink(s)
        except FileNotFoundError:
            pass
    if stale:
        print(f"  Cleaned {len(stale)} stale .tmp.npz file(s) from prior runs")

    fetch_s1_scene = _resolve_fetch_fn(args.s1_backend)

    print(f"=== S1 SAR Enrichment ===")
    print(f"  Tiles:   {len(tiles)}")
    print(f"  Workers: {args.workers}")
    print(f"  Backend: {args.s1_backend}")

    stats = {"ok": 0, "skipped": 0, "failed": 0}
    lock = threading.Lock()
    completed = 0
    t0 = time.time()

    def _run(path):
        nonlocal completed
        r = enrich_one_tile(
            path,
            fetch_s1_scene=fetch_s1_scene,
            skip_existing=args.skip_existing,
        )
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
