#!/usr/bin/env python3
"""Build unified 20-class labels from scratch for all tiles.

Reads each tile's bbox_3006, fetches NMD from local raster, rasterizes
LPIS parcels and SKS harvest polygons, runs merge_all(), saves back.

Data sources (all local, no API calls):
  - NMD: GeoTIFF raster (10m, EPSG:3006)
  - LPIS: GeoParquet per year (Jordbruksverket)
  - SKS: GeoParquet (Skogsstyrelsen utförda avverkningar + anmälningar)

Usage:
    python scripts/build_labels.py --data-dir /data/unified_v2 \
        --nmd-raster data/nmd/nmd2018bas_ogeneraliserad_v1_1.tif \
        --lpis-dir data/lpis --sks-dir data/sks --workers 4
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import threading
import time
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path

import numpy as np
from scipy.ndimage import label as nd_label

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imint.training.tile_fetch import fetch_nmd_label_local
from imint.training.unified_schema import merge_all, UNIFIED_CLASS_NAMES

def _compute_nmd_area_map(nmd_label: np.ndarray, pixel_ha: float = 0.01) -> np.ndarray:
    """Per-pixel area map derived from NMD raster via connected components.

    Each pixel receives the area (ha) of its contiguous same-class region,
    matching the inverse-area weighting applied to LPIS parcels.

    At 10 m Sentinel-2 resolution one pixel = 100 m² = 0.01 ha (default).
    A 25-pixel patch = 0.25 ha = NMD MMU → weight floor of 1.0.
    A 5-pixel fragment = 0.05 ha → weight 4.0 (max).

    Background (class 0) pixels are left at 0.0 — the loss already ignores
    them via ignore_index.
    """
    area_map = np.zeros(nmd_label.shape, dtype=np.float32)
    for cls in np.unique(nmd_label):
        if cls == 0:
            continue
        labeled, _ = nd_label(nmd_label == cls)
        flat = labeled.ravel()
        counts = np.bincount(flat).astype(np.float32)
        counts[0] = 0.0                          # background region → 0 ha
        area_map.ravel()[:] += (counts * pixel_ha)[flat]
    return area_map


# Shared state — function handles only. Data is queried on-demand via
# SpatialParquet bbox-filtered reads; we never hold the full files in
# memory. Each tile pulls only the polygons whose bbox intersects it.
_rasterize_parcels = None
_rasterize_sks = None
_compute_harvest_probability = None

# SpatialParquet handles (SKS + per-year LPIS). Populated on first use.
# The handles themselves are thread-safe (each thread gets its own
# pyarrow ParquetFile via threading.local() inside SpatialParquet —
# see imint/training/spatial_parquet.py). We only need a lock around
# the dict mutation below to avoid racy "first-use" double-init.
_sks_utforda_sp = None    # spatial parquet for executed clearcuts
_sks_anmalda_sp = None    # spatial parquet for announced clearcuts
_lpis_sp: dict = {}       # year → SpatialParquet for LPIS
_handles_lock = threading.Lock()

# Sentinels so we don't retry missing files on every query
_SKS_UTFORDA_MISSING = object()
_SKS_ANMALDA_MISSING = object()
_SENTINEL_UNSET = object()  # used by _lpis_handle to distinguish "not in cache" vs "cached as None"


def _ensure_helpers_loaded():
    """Lazy-import rasterize helpers once."""
    global _rasterize_parcels, _rasterize_sks, _compute_harvest_probability
    if _rasterize_parcels is None:
        from scripts.enrich_tiles_lpis_mask import rasterize_parcels
        _rasterize_parcels = rasterize_parcels
    if _rasterize_sks is None:
        from scripts.enrich_tiles_sks import rasterize_sks, compute_harvest_probability
        _rasterize_sks = rasterize_sks
        _compute_harvest_probability = compute_harvest_probability


def _sks_utforda_handle(sks_dir: str):
    """Return a SpatialParquet for utförda clearcuts, or None if missing.

    Prefers the preprocessed ``*_spatial.parquet``; falls back to the
    legacy full-file parquet with a warning (still works, just slow).
    """
    global _sks_utforda_sp
    # Read outside the lock — safe because writes are atomic for these
    # singleton bindings on CPython, and the lock below ensures no
    # double-init.
    if _sks_utforda_sp is _SKS_UTFORDA_MISSING:
        return None
    if _sks_utforda_sp is not None:
        return _sks_utforda_sp

    with _handles_lock:
        # Re-check under the lock to avoid races between threads that
        # both reached the slow path before either could initialise.
        if _sks_utforda_sp is _SKS_UTFORDA_MISSING:
            return None
        if _sks_utforda_sp is not None:
            return _sks_utforda_sp

        from imint.training.spatial_parquet import SpatialParquet
        spatial = os.path.join(sks_dir, "utforda_avverkningar_spatial.parquet")
        fallback = os.path.join(sks_dir, "utforda_avverkningar.parquet")
        if os.path.exists(spatial) or os.path.exists(fallback):
            _sks_utforda_sp = SpatialParquet(
                spatial, fallback_path=fallback if os.path.exists(fallback) else None,
            )
            return _sks_utforda_sp

        _sks_utforda_sp = _SKS_UTFORDA_MISSING
        return None


def _sks_anmalda_handle(sks_dir: str):
    """SpatialParquet for anmälda clearcuts, or None."""
    global _sks_anmalda_sp
    if _sks_anmalda_sp is _SKS_ANMALDA_MISSING:
        return None
    if _sks_anmalda_sp is not None:
        return _sks_anmalda_sp

    with _handles_lock:
        if _sks_anmalda_sp is _SKS_ANMALDA_MISSING:
            return None
        if _sks_anmalda_sp is not None:
            return _sks_anmalda_sp

        from imint.training.spatial_parquet import SpatialParquet
        spatial = os.path.join(sks_dir, "avverkningsanmalningar_spatial.parquet")
        fallback = os.path.join(sks_dir, "avverkningsanmalningar.parquet")
        if os.path.exists(spatial) or os.path.exists(fallback):
            _sks_anmalda_sp = SpatialParquet(
                spatial, fallback_path=fallback if os.path.exists(fallback) else None,
            )
            return _sks_anmalda_sp

        _sks_anmalda_sp = _SKS_ANMALDA_MISSING
        return None


def _lpis_handle(year: int, lpis_dir: str):
    """SpatialParquet for LPIS year, or None if parquet missing."""
    # Cheap read; lock-free path for the common case.
    cached = _lpis_sp.get(year, _SENTINEL_UNSET)
    if cached is not _SENTINEL_UNSET:
        return cached

    with _handles_lock:
        cached = _lpis_sp.get(year, _SENTINEL_UNSET)
        if cached is not _SENTINEL_UNSET:
            return cached

        from imint.training.spatial_parquet import SpatialParquet
        spatial = os.path.join(lpis_dir, f"jordbruksskiften_{year}_spatial.parquet")
        fallback = os.path.join(lpis_dir, f"jordbruksskiften_{year}.parquet")
        if os.path.exists(spatial) or os.path.exists(fallback):
            _lpis_sp[year] = SpatialParquet(
                spatial, fallback_path=fallback if os.path.exists(fallback) else None,
            )
            return _lpis_sp[year]

        _lpis_sp[year] = None
        return None


def build_tile_label(
    tile_path: str,
    nmd_raster: str,
    lpis_dir: str,
    sks_dir: str,
) -> dict:
    """Build unified 23-class label for one tile from scratch.

    Tile geometry is derived from the on-disk raster (``spectral`` shape)
    or ``tile_size_px`` persisted by the fetcher — no global tile-size
    constants.
    """
    from imint.training.tile_config import TileConfig
    from imint.training.tile_bbox import resolve_tile_bbox

    name = os.path.basename(tile_path).replace(".npz", "")
    try:
        data = dict(np.load(tile_path, allow_pickle=True))

        # Derive tile size from raster (authoritative) or persisted key
        sp = data.get("spectral", data.get("image"))
        size_px = int(data.get("tile_size_px", sp.shape[-1] if sp is not None else 256))
        tile_cfg = TileConfig(size_px=size_px)

        bbox_3006 = resolve_tile_bbox(name=name, tile=tile_cfg, npz_data=data)
        if bbox_3006 is None:
            return {"name": name, "status": "failed", "reason": "no_bbox"}

        # Determine tile year — use the most common year in the
        # date stack, NOT dates[0]. Per CLAUDE.md, frame 0 is the
        # autumn background from *year-1* (Sep-Oct, used by the
        # hygges-pipeline). Frames 1-3 are the primary year. The
        # naive `dates[0][:4]` picks the year-1 background, which
        # makes _lpis_handle look up the wrong year — e.g. dates
        # = [2021-09, 2022-04, 2022-06, 2022-07] triggered a search
        # for LPIS-2021 (which doesn't exist on disk) instead of
        # LPIS-2022 where the parcels actually live. That dropped
        # 1000 crop_*-tiles in the 2026-05-07 run.
        tile_year = None
        if "year" in data:
            tile_year = int(data["year"])
        elif "lpis_year" in data:
            tile_year = int(data["lpis_year"])
        elif "dates" in data:
            from collections import Counter
            years = []
            for d in data["dates"]:
                s = str(d)
                if s and len(s) >= 4:
                    try:
                        years.append(int(s[:4]))
                    except ValueError:
                        pass
            if years:
                # Pick the modal year. Ties broken by most recent year
                # (a sane default — 2022 over 2021 if equal counts).
                counts = Counter(years)
                top_count = counts.most_common(1)[0][1]
                tied = [y for y, c in counts.items() if c == top_count]
                tile_year = max(tied)
        if tile_year is None:
            tile_year = 2022  # default

        # --- Step 1: NMD label from local raster ---
        nmd_label = fetch_nmd_label_local(bbox_3006, tile_cfg, nmd_raster=nmd_raster)
        if nmd_label is None:
            return {"name": name, "status": "failed", "reason": "no_nmd"}

        # Connected-component area map: same inverse-area weighting as LPIS.
        # Stored separately; unified_dataset.py merges with parcel_area_ha.
        data["nmd_area_ha"] = _compute_nmd_area_map(nmd_label)

        _ensure_helpers_loaded()

        # --- Step 2: LPIS crop mask (bbox-filtered SpatialParquet query) ---
        lpis_mask = None
        lpis_sp = _lpis_handle(tile_year, lpis_dir)
        if lpis_sp is not None:
            gdf = lpis_sp.query(bbox_3006)
            if len(gdf) > 0:
                bbox_arr = np.array([bbox_3006["west"], bbox_3006["south"],
                                     bbox_3006["east"], bbox_3006["north"]])
                lpis_mask, area_map, n_parcels = _rasterize_parcels(
                    gdf, bbox_arr, tile_size=tile_cfg.size_px,
                )
                data["label_mask"]     = lpis_mask   # uint16 raw SJV codes
                data["parcel_area_ha"] = area_map    # float32 ha/pixel
                data["n_parcels"]      = np.int32(n_parcels)

        # --- Step 3: SKS harvest mask (filtered by tile year + tile bbox) ---
        # Hygge = avverkat inom 5 år före tile-året
        harvest_mask = None
        bbox_tuple = (bbox_3006["west"], bbox_3006["south"],
                      bbox_3006["east"], bbox_3006["north"])

        sks_utforda_sp = _sks_utforda_handle(sks_dir)
        if sks_utforda_sp is not None:
            import pandas as pd
            sks_utforda_local = sks_utforda_sp.query(bbox_3006)
            if len(sks_utforda_local) > 0:
                # Parse Avvdatum lazily — only for polygons that intersect this tile
                sks_utforda_local = sks_utforda_local.copy()
                sks_utforda_local["Avvdatum"] = pd.to_datetime(
                    sks_utforda_local["Avvdatum"], errors="coerce",
                )
                min_date = pd.Timestamp(f"{tile_year - 5}-01-01")
                max_date = pd.Timestamp(f"{tile_year}-12-31")
                sks_filtered = sks_utforda_local[
                    (sks_utforda_local["Avvdatum"] >= min_date) &
                    (sks_utforda_local["Avvdatum"] <= max_date)
                ]
                if len(sks_filtered) > 0:
                    harvest_mask, n_harvest = _rasterize_sks(
                        sks_filtered, bbox_tuple, tile_size=tile_cfg.size_px,
                    )
                    data["harvest_mask"] = harvest_mask
                    data["n_harvest_polygons"] = np.int32(n_harvest)

        sks_anmalda_sp = _sks_anmalda_handle(sks_dir)
        if sks_anmalda_sp is not None:
            sks_anmalda_local = sks_anmalda_sp.query(bbox_3006)
            if len(sks_anmalda_local) > 0:
                mature_mask, n_mature = _rasterize_sks(
                    sks_anmalda_local, bbox_tuple, tile_size=tile_cfg.size_px,
                )
                data["n_mature_polygons"] = np.int32(n_mature)

                if _compute_harvest_probability is not None and harvest_mask is not None:
                    data["harvest_probability"] = _compute_harvest_probability(
                        harvest_mask, mature_mask,
                    )

        # --- Step 4: Merge all → unified 20-class ---
        unified = merge_all(nmd_label, lpis_mask, harvest_mask)
        data["label"] = unified
        data["nmd_label_raw"] = nmd_label

        # ── Invariant assertions (catch bugs at write time, not 3h later) ──
        # Shape match: label and spectral must agree spatially.
        if sp is not None:
            sp_h, sp_w = sp.shape[-2], sp.shape[-1]
            if unified.shape != (sp_h, sp_w):
                raise AssertionError(
                    f"label.shape={unified.shape} != spectral HW=({sp_h},{sp_w})"
                )
        # NMD label must be in the 19-class sequential range.
        if nmd_label.dtype != np.uint8 or int(nmd_label.max()) > 19:
            raise AssertionError(
                f"nmd_label_raw out of range: dtype={nmd_label.dtype} "
                f"max={int(nmd_label.max())} (expected uint8, max <= 19)"
            )
        # Unified label must be in the 23-class range.
        if int(unified.max()) > 22:
            raise AssertionError(
                f"unified label out of range: max={int(unified.max())} "
                f"(expected <= 22)"
            )
        # Crop-named tiles must end up with at least one parcel — these
        # were specifically fetched at LPIS centroids, so empty is a bug.
        if name.startswith("crop_") and lpis_mask is None:
            raise AssertionError(
                f"crop-named tile {name} got no LPIS overlay — likely a "
                f"thread-safety race or a bbox / parquet alignment bug"
            )

        # ── Atomic write: tmp + os.replace ──────────────────────────────
        # Without atomic write, a failure mid-savez_compressed leaves a
        # truncated .npz on disk (we hit BadZipFile / EOFError on those
        # earlier). os.replace is atomic on POSIX so the original tile
        # stays usable until the new write completes.
        #
        # np.savez_compressed unconditionally appends ".npz" to its path
        # argument unless the path already ends in ".npz". We pass a
        # path WITHOUT the .npz suffix (`tile.npz.tmp`) so the produced
        # file lands at `tile.npz.tmp.npz`, then rename onto `tile.npz`.
        tmp_base = tile_path + ".tmp"          # e.g. /…/foo.npz.tmp
        np.savez_compressed(tmp_base, **data)  # writes /…/foo.npz.tmp.npz
        os.replace(tmp_base + ".npz", tile_path)
        return {"name": name, "status": "ok"}

    except Exception as e:
        # Clean up any half-written tmp file (.tmp.npz from savez)
        stale = tile_path + ".tmp.npz"
        if os.path.exists(stale):
            try:
                os.unlink(stale)
            except OSError:
                pass
        return {"name": name, "status": "failed", "reason": str(e)[:120]}


def main():
    p = argparse.ArgumentParser(description="Build unified 20-class labels from scratch")
    p.add_argument("--data-dir", required=True, help="Directory with .npz tiles")
    p.add_argument("--nmd-raster", default="data/nmd/nmd2018bas_ogeneraliserad_v1_1.tif")
    p.add_argument("--lpis-dir", default="data/lpis")
    p.add_argument("--sks-dir", default="data/sks")
    p.add_argument("--tile-ids", nargs="+",
                   help="Only process these tile IDs (filename stems, e.g. 45843596)")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel workers (use 1 to minimize memory)")
    p.add_argument(
        "--executor",
        choices=["thread", "process"],
        default="thread",
        help="Concurrency model. 'thread' uses ThreadPoolExecutor with "
             "per-thread rasterio + pyarrow handles via threading.local() "
             "(default; lower memory, fast startup). 'process' uses "
             "ProcessPoolExecutor (each worker has its own address space, "
             "guaranteed isolation against any not-thread-safe library; "
             "higher memory, slower startup). Use 'process' as a safety "
             "fallback if a future regression breaks the thread-safety "
             "invariants in tile_fetch / spatial_parquet.",
    )
    args = p.parse_args()
    # Tile size is derived per-tile from the raster shape / tile_size_px key.
    # No CLI flag needed — the script adapts to whatever fetch_unified_tiles wrote.

    tiles = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if args.tile_ids:
        ids = set(args.tile_ids)
        tiles = [t for t in tiles if os.path.basename(t).replace(".npz", "") in ids]
    print(f"=== Build Labels from Scratch ===")
    print(f"  Tiles: {len(tiles)}")
    print(f"  NMD: {args.nmd_raster}")
    print(f"  LPIS: {args.lpis_dir}")
    print(f"  SKS: {args.sks_dir}")
    print(f"  Schema: {len(UNIFIED_CLASS_NAMES)} classes")
    print()

    # LPIS and SKS are read on-demand per tile via SpatialParquet — no
    # startup preload. Each tile query pulls only the row groups that
    # overlap its bbox.
    # Pre-loading all years at once would exceed memory on small pods.
    # Pre-open SpatialParquet handles once so we don't re-open per tile.
    # Only matters for the thread executor — process workers re-init
    # their own globals.
    if args.executor == "thread":
        _sks_utforda_handle(args.sks_dir)
        _sks_anmalda_handle(args.sks_dir)

    stats = {"ok": 0, "failed": 0}
    t0 = time.time()

    print(f"  Executor:  {args.executor} (workers={args.workers})")
    Executor = ProcessPoolExecutor if args.executor == "process" else ThreadPoolExecutor

    with Executor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(build_tile_label, t, args.nmd_raster,
                        args.lpis_dir, args.sks_dir): t
            for t in tiles
        }
        for i, f in enumerate(as_completed(futs)):
            r = f.result()
            stats[r.get("status", "failed")] = stats.get(r.get("status", "failed"), 0) + 1
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(tiles)}] {r['name']}: {r['status']} "
                      f"| {(i+1)/elapsed*3600:.0f}/h", flush=True)
            if r["status"] == "failed" and (i + 1) <= 10:
                print(f"  FAIL: {r['name']} — {r.get('reason', '?')}", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  OK={stats['ok']}  Failed={stats['failed']}")


if __name__ == "__main__":
    main()
