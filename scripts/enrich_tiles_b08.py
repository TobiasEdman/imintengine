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

from imint.utils import dn_to_reflectance


# Per-source openEO config — collection id + band name case differ
# across backends. Keep the script source-agnostic by dispatching here.
# "des" = Digital Earth Sweden (lowercase bands, free for RISE).
# "vito" = openEO Platform (openeo.cloud) — federated, includes VITO;
#          supports client_credentials for unattended runs. The native
#          Terrascope endpoint (openeo.vito.be) only supports
#          interactive OIDC, which doesn't fit a k8s job.
_SOURCE_CONFIG = {
    "des": {"collection_id": "s2_msi_l2a", "band_b08": "b08"},
    "vito": {"collection_id": "SENTINEL2_L2A", "band_b08": "B08"},
}

_CONN_LOCAL = threading.local()


def _get_conn(source: str):
    """Thread-local authenticated openEO connection for ``source``.

    Auth happens once per worker thread; subsequent calls reuse the
    connection. On any error, the next call rebuilds — covers session
    expiry over long runs.
    """
    conn = getattr(_CONN_LOCAL, "conn", None)
    if conn is None:
        if source == "des":
            from imint.fetch import _connect
            conn = _connect()
        elif source == "vito":
            import openeo
            url = os.environ.get("OPENEO_URL", "openeo.cloud")
            conn = openeo.connect(url)
            # Uses OPENEO_AUTH_CLIENT_ID / OPENEO_AUTH_CLIENT_SECRET /
            # OPENEO_AUTH_PROVIDER_ID env vars (openeo-python-client 0.18+).
            conn.authenticate_oidc_client_credentials()
        else:
            raise ValueError(f"Unknown source '{source}' — use 'des' or 'vito'")
        _CONN_LOCAL.conn = conn
    return conn


def _drop_conn() -> None:
    """Drop the thread's cached connection — call after an error."""
    if hasattr(_CONN_LOCAL, "conn"):
        del _CONN_LOCAL.conn


def _fetch_b08_frame(
    west: float, south: float, east: float, north: float,
    date_str: str, size_px: int,
    source: str = "des",
) -> np.ndarray | None:
    """Fetch B08 for one date via ``source``'s openEO — band-specific.

    Asks the backend for ONLY band B08 over the exact known date (the
    tile already knows which date was used for that frame), via
    ``load_collection(collection_id=..., bands=[B08], temporal_extent=
    [d, d+1])``. Returns reflectance [0, 1] on the requested
    ``(size_px, size_px)`` grid in EPSG:3006.
    """
    from datetime import datetime, timedelta
    import os
    import tempfile

    cfg = _SOURCE_CONFIG[source]
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
        conn = _get_conn(source)
        cube = conn.load_collection(
            collection_id=cfg["collection_id"],
            spatial_extent=spatial,
            temporal_extent=[date_str, d1],
            bands=[cfg["band_b08"]],
        )
        cube.download(tmp_path, format="GTiff")
    except Exception as e:
        print(f"    [b08-fetch:{source}] failed for "
              f"{date_str} bbox={spatial}: {type(e).__name__}: {e}",
              flush=True)
        _drop_conn()
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

    # DN -> reflectance [0, 1]. DES bakes the PB04.00 -1000 BOA offset into
    # its COGs, so subtract it (via dn_to_reflectance(source="des")) to match
    # the spectral cube. VITO/CDSE openEO apply the offset server-side → plain
    # /10000. Must agree with the spectral fetch or b08 lands +0.1 high.
    if source == "des":
        return dn_to_reflectance(arr, source="des")
    return arr / 10000.0


def enrich_one_tile(tile_path: str, skip_existing: bool = True,
                    source: str = "des") -> dict:
    """Add B08 to one tile .npz file.

    D3 design (date-aware mismatch detection):
    - Per-frame `b08_dates[i]` is compared against the spectral cube's
      `dates[i]`. Any slot where they disagree (or where existing b08
      slice is all-zeros) gets re-fetched.
    - Slots whose b08_dates match dates AND have non-zero spectral data
      are preserved untouched.
    - `skip_existing=True` (default) takes a fast path when ALL slots are
      already date-aligned + non-zero — skips the tile entirely.
    - `skip_existing=False` forces the per-slot inspection even when the
      fast-path would skip.

    This makes b08-enrichment idempotent and self-healing: any operation
    that changes the spectral `dates` field (e.g. repair_to_canonical_layout)
    automatically becomes a no-op for already-aligned slots and a re-fetch
    trigger for slots whose date changed. The refetch path doesn't need
    to know about b08 — it just writes new dates and the next enrich pass
    converges.
    """
    name = Path(tile_path).stem
    try:
        data = dict(np.load(tile_path, allow_pickle=True))
    except Exception as e:
        return {"name": name, "status": "failed", "reason": str(e)}

    dates = data.get("dates", [])
    spectral = data.get("spectral", data.get("image"))
    if spectral is None:
        return {"name": name, "status": "failed", "reason": "no_spectral"}

    h, w = spectral.shape[1], spectral.shape[2]
    n_frames = spectral.shape[0] // 6
    new_dates = [
        str(dates[fi])[:10] if fi < len(dates) and dates[fi] else ""
        for fi in range(n_frames)
    ]

    existing_b08 = data.get("b08")
    existing_b08_dates = data.get("b08_dates")
    if existing_b08_dates is not None:
        existing_b08_dates = [str(x)[:10] for x in existing_b08_dates]

    # Per-slot decision: keep (date-aligned + non-zero) or fetch.
    frames_to_fetch: list[int] = []
    for fi in range(n_frames):
        already_aligned = (
            existing_b08 is not None
            and existing_b08_dates is not None
            and fi < len(existing_b08_dates)
            and existing_b08_dates[fi] == new_dates[fi]
            and getattr(existing_b08, "ndim", 0) == 3
            and existing_b08.shape[0] > fi
            and bool(np.any(existing_b08[fi]))
        )
        if not already_aligned:
            frames_to_fetch.append(fi)

    if skip_existing and not frames_to_fetch:
        return {"name": name, "status": "skipped",
                "reason": "all_slots_date_aligned"}

    # Derive TileConfig from persisted tile_size_px or fall back to raster dim
    from imint.training.tile_config import TileConfig
    from imint.training.tile_bbox import resolve_tile_bbox
    size_px = int(data.get("tile_size_px", h))
    tile_cfg = TileConfig(size_px=size_px)

    bbox = resolve_tile_bbox(name=name, tile=tile_cfg, npz_data=data)
    if bbox is None:
        return {"name": name, "status": "failed", "reason": "no_bbox"}
    tile_cfg.assert_bbox_matches(bbox)

    # Seed b08_frames + b08_dates_out with existing data where alignment
    # held; we'll overwrite to-fetch slots below.
    b08_frames: list[np.ndarray] = []
    b08_dates_out: list[str] = []
    for fi in range(n_frames):
        if fi in frames_to_fetch:
            b08_frames.append(np.zeros((h, w), dtype=np.float32))
            b08_dates_out.append("")
        else:
            b08_frames.append(np.asarray(existing_b08[fi], dtype=np.float32))
            b08_dates_out.append(existing_b08_dates[fi])

    for fi in frames_to_fetch:
        date_str = new_dates[fi]
        if not date_str:
            continue
        frame = _fetch_b08_frame(
            bbox["west"], bbox["south"], bbox["east"], bbox["north"],
            date_str, size_px, source=source,
        )
        if frame is not None and frame.shape == (h, w):
            b08_frames[fi] = frame
            b08_dates_out[fi] = date_str

    data["b08"] = np.stack(b08_frames, axis=0)               # (T, H, W)
    data["b08_dates"] = np.array(b08_dates_out)              # (T,) ISO strings
    valid = sum(1 for f in b08_frames if bool(np.any(f)))
    data["has_b08"] = np.int32(1 if valid > 0 else 0)

    np.savez_compressed(tile_path, **data)
    return {"name": name, "status": "ok", "valid_frames": valid,
            "refetched_frames": frames_to_fetch}


def main():
    parser = argparse.ArgumentParser(description="Enrich tiles with S2 B08 band")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument(
        "--source", default="des", choices=list(_SOURCE_CONFIG),
        help="openEO backend: 'des' (Digital Earth Sweden, default) or "
             "'vito' (openEO Platform, federated, includes VITO). vito "
             "requires OPENEO_AUTH_CLIENT_ID/SECRET/PROVIDER_ID env vars.",
    )
    args = parser.parse_args()

    tiles = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if args.max_tiles:
        tiles = tiles[:args.max_tiles]
    print(f"=== B08 Enrichment ({args.source}) ===")
    print(f"  Tiles:  {len(tiles)}")
    print(f"  Workers:{args.workers}")
    print(f"  Source: {args.source}")

    stats = {"ok": 0, "skipped": 0, "failed": 0}
    lock = threading.Lock()
    completed = 0
    t0 = time.time()

    def _run(path):
        nonlocal completed
        r = enrich_one_tile(
            path, skip_existing=args.skip_existing, source=args.source)
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
