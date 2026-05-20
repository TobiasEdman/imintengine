#!/usr/bin/env python3
"""Add Sentinel-2 B08 (broad NIR, 842nm, 10m) to existing tiles.

Required by Clay v1.5 and Croma which expect B08 alongside the 6-band
``spectral`` tensor. Fetches B08 from Digital Earth Sweden (DES) via
openEO — DES has no equivalent of CDSE's Process API + evalscript, so
``imint.fetch.fetch_des_data`` is used (returns all spectral bands; we
keep only B08). The CDSE PU quota is exhausted, so this path is what
we use; DES openEO is free for RISE.

Idempotent: skips tiles with ``has_b08 == 1``.

Keys written:
    b08       (T, H, W) float32 — B08 reflectance [0, 1] per temporal frame
    has_b08   int32 — 1 if any frame has B08 data

Usage:
    python scripts/enrich_tiles_b08.py \\
        --data-dir /data/unified_v2_512 \\
        --workers 6 \\
        --skip-existing

Credentials: DES_USER + DES_PASSWORD env vars (basic auth), DES_TOKEN,
or the .des_token file — see ``imint.fetch._get_des_token``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


_DES_LOCAL = threading.local()


def _get_des_conn():
    """Thread-local authenticated DES openEO connection.

    Auth happens once per worker thread; subsequent calls reuse the
    connection. On any error, the next call rebuilds — covers session
    expiry over long runs.
    """
    conn = getattr(_DES_LOCAL, "conn", None)
    if conn is None:
        from imint.fetch import _connect
        conn = _connect()
        _DES_LOCAL.conn = conn
    return conn


def _drop_des_conn() -> None:
    """Drop the thread's cached connection — call after an error."""
    if hasattr(_DES_LOCAL, "conn"):
        del _DES_LOCAL.conn


def _fetch_b08_frame(
    west: float, south: float, east: float, north: float,
    date_str: str, size_px: int,
) -> np.ndarray | None:
    """Fetch B08 for one date via DES openEO — band-specific.

    Asks DES for ONLY band B08 over the exact known date (the tile
    already knows which date was used for that frame), via
    ``load_collection(collection_id='s2_msi_l2a', bands=['B08'],
    temporal_extent=[d, d+1])``. Returns reflectance [0, 1] on the
    requested ``(size_px, size_px)`` grid in EPSG:3006.
    """
    from datetime import datetime, timedelta
    import os
    import tempfile

    try:
        d0 = datetime.fromisoformat(date_str)
    except Exception:
        return None
    d1 = (d0 + timedelta(days=1)).strftime("%Y-%m-%d")
    spatial = {
        "west": west, "south": south, "east": east, "north": north,
        "crs": "EPSG:3006",
    }

    tmp_path = tempfile.mktemp(suffix=".tif")
    try:
        conn = _get_des_conn()
        cube = conn.load_collection(
            collection_id="s2_msi_l2a",
            spatial_extent=spatial,
            temporal_extent=[date_str, d1],
            bands=["B08"],
        )
        cube.download(tmp_path, format="GTiff")
    except Exception as e:
        print(f"    [b08-fetch] DES openEO failed for "
              f"{date_str} bbox={spatial}: {type(e).__name__}: {e}",
              flush=True)
        _drop_des_conn()
        return None

    try:
        import rasterio
        with rasterio.open(tmp_path) as ds:
            arr = ds.read(1).astype(np.float32)
    except Exception as e:
        print(f"    [b08-fetch] rasterio read failed: "
              f"{type(e).__name__}: {e}", flush=True)
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    # Resample to the tile grid if openEO returned a different shape.
    if arr.shape != (size_px, size_px):
        try:
            from scipy.ndimage import zoom
            zy = size_px / arr.shape[0]
            zx = size_px / arr.shape[1]
            arr = zoom(arr, (zy, zx), order=1).astype(np.float32)
        except Exception:
            return None

    # DN -> reflectance [0, 1] — matches the convention `spectral` is
    # stored in (the original 6-band fetch divides by 10000 too).
    return arr / 10000.0


def enrich_one_tile(tile_path: str, skip_existing: bool = True) -> dict:
    """Add B08 to one tile .npz file."""
    name = Path(tile_path).stem
    try:
        data = dict(np.load(tile_path, allow_pickle=True))
    except Exception as e:
        return {"name": name, "status": "failed", "reason": str(e)}

    if skip_existing and int(data.get("has_b08", 0)) == 1:
        return {"name": name, "status": "skipped"}

    dates = data.get("dates", [])
    spectral = data.get("spectral", data.get("image"))
    if spectral is None:
        return {"name": name, "status": "failed", "reason": "no_spectral"}

    h, w = spectral.shape[1], spectral.shape[2]
    n_frames = spectral.shape[0] // 6

    # Derive TileConfig from persisted tile_size_px or fall back to raster dim
    from imint.training.tile_config import TileConfig
    from imint.training.tile_bbox import resolve_tile_bbox
    size_px = int(data.get("tile_size_px", h))
    tile_cfg = TileConfig(size_px=size_px)

    bbox = resolve_tile_bbox(name=name, tile=tile_cfg, npz_data=data)
    if bbox is None:
        return {"name": name, "status": "failed", "reason": "no_bbox"}
    tile_cfg.assert_bbox_matches(bbox)

    b08_frames = []
    valid = 0
    for fi in range(n_frames):
        date_str = str(dates[fi])[:10] if fi < len(dates) and dates[fi] else ""
        if not date_str:
            b08_frames.append(np.zeros((h, w), dtype=np.float32))
            continue

        frame = _fetch_b08_frame(
            bbox["west"], bbox["south"], bbox["east"], bbox["north"],
            date_str, size_px,
        )
        if frame is not None and frame.shape == (h, w):
            b08_frames.append(frame)
            valid += 1
        else:
            b08_frames.append(np.zeros((h, w), dtype=np.float32))

    data["b08"] = np.stack(b08_frames, axis=0)  # (T, H, W)
    data["has_b08"] = np.int32(1 if valid > 0 else 0)

    np.savez_compressed(tile_path, **data)
    return {"name": name, "status": "ok", "valid_frames": valid}


def main():
    parser = argparse.ArgumentParser(description="Enrich tiles with S2 B08 band")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--max-tiles", type=int, default=None)
    args = parser.parse_args()

    tiles = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if args.max_tiles:
        tiles = tiles[:args.max_tiles]
    print(f"=== B08 Enrichment ===")
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
            print(f"  [{completed}/{len(tiles)}] {r['name']}: {r['status']}"
                  f"{f' ({valid}/4 frames)' if valid != '' else ''}"
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
