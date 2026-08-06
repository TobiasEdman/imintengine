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
from collections import Counter
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
from imint.training.unified_schema import merge_all, merge_all_2023, UNIFIED_CLASS_NAMES

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
    global _rasterize_parcels, _rasterize_sks
    if _rasterize_parcels is None:
        from scripts.enrich_tiles_lpis_mask import rasterize_parcels
        _rasterize_parcels = rasterize_parcels
    if _rasterize_sks is None:
        from scripts.enrich_tiles_sks import rasterize_sks
        _rasterize_sks = rasterize_sks


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
    nmd_version: int = 2018,
    label_out_dir: str | None = None,
) -> dict:
    """Build the unified label for one tile from scratch.

    Tile geometry is derived from the on-disk raster (``spectral`` shape)
    or ``tile_size_px`` persisted by the fetcher — no global tile-size
    constants.

    ``nmd_version`` selects the NMD base: 2018 → the 19-class chain + merge_all
    (23-class output); 2023 → raw NMD2023 codes + merge_all_2023 (29-class
    output, with the NMD2023-only fine classes). LPIS/SKS overlays are identical.
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
        # 2023 reads raw uint16 codes (mapped later by merge_all_2023); 2018
        # reads the 19-class sequential label.
        nmd_label = fetch_nmd_label_local(
            bbox_3006, tile_cfg, nmd_raster=nmd_raster, raw=(nmd_version == 2023),
        )
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
                _, n_mature = _rasterize_sks(
                    sks_anmalda_local, bbox_tuple, tile_size=tile_cfg.size_px,
                )
                data["n_mature_polygons"] = np.int32(n_mature)

        # --- Step 4: Merge all → unified label ---
        if nmd_version == 2023:
            unified = merge_all_2023(nmd_label, lpis_mask, harvest_mask)
        else:
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
        # NMD base sanity: 2018 → 19-class uint8; 2023 → raw uint16 codes.
        if nmd_version == 2023:
            if nmd_label.dtype != np.uint16:
                raise AssertionError(
                    f"nmd2023 raw must be uint16, got {nmd_label.dtype}"
                )
        elif nmd_label.dtype != np.uint8 or int(nmd_label.max()) > 19:
            raise AssertionError(
                f"nmd_label_raw out of range: dtype={nmd_label.dtype} "
                f"max={int(nmd_label.max())} (expected uint8, max <= 19)"
            )
        # Unified label must be in range: 2018 → <=22, 2023 → <=27.
        _max_unified = 27 if nmd_version == 2023 else 22
        if int(unified.max()) > _max_unified:
            raise AssertionError(
                f"unified label out of range: max={int(unified.max())} "
                f"(expected <= {_max_unified})"
            )
        # Crop-named tiles must end up with at least one parcel — these
        # were specifically fetched at LPIS centroids, so empty is a bug.
        if name.startswith("crop_") and lpis_mask is None:
            raise AssertionError(
                f"crop-named tile {name} got no LPIS overlay — likely a "
                f"thread-safety race or a bbox / parquet alignment bug"
            )

        # ── Persist: sidecar label-only, or overwrite the tile atomically ──
        # label_out_dir writes ONLY the label fields to a separate directory,
        # leaving the source tile (spectral/aux) untouched — this is how the
        # NMD2023 relabel avoids overwriting the NMD2018 labels and avoids a
        # full 806G dataset copy on a tight PVC. The training loader reads
        # spectral/aux from the source tile and the label from the sidecar.
        if label_out_dir is not None:
            _LABEL_KEYS = ("label", "nmd_label_raw", "nmd_area_ha", "label_mask",
                           "parcel_area_ha", "n_parcels", "harvest_mask",
                           "n_harvest_polygons", "n_mature_polygons")
            payload = {k: data[k] for k in _LABEL_KEYS if k in data}
            payload["tile_size_px"] = np.int32(size_px)
            out_path = os.path.join(label_out_dir, name + ".npz")
        else:
            payload = data
            out_path = tile_path

        # Atomic write: tmp + os.replace. Without it, a failure mid-savez leaves
        # a truncated .npz (BadZipFile / EOFError). np.savez_compressed appends
        # ".npz" unless the path already ends in it, so we pass a suffix-less
        # tmp base and rename the produced ".npz" onto the target.
        tmp_base = out_path + ".tmp"
        np.savez_compressed(tmp_base, **payload)
        os.replace(tmp_base + ".npz", out_path)
        return {"name": name, "status": "ok"}

    except Exception as e:
        # Clean up any half-written tmp file (.tmp.npz from savez)
        stale = (os.path.join(label_out_dir, name + ".npz")
                 if label_out_dir is not None else tile_path) + ".tmp.npz"
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
    p.add_argument("--nmd-version", type=int, choices=[2018, 2023], default=2018,
                   help="NMD base: 2018 (19-class chain, 23-class out) or 2023 "
                        "(raw codes, 29-class out with fine open-land classes)")
    p.add_argument("--label-out-dir", default=None,
                   help="if set, write ONLY the label fields to this dir "
                        "(sidecar) instead of overwriting the source tile — "
                        "non-destructive relabel, no full dataset copy")
    p.add_argument("--lpis-dir", default="data/lpis")
    p.add_argument("--sks-dir", default="data/sks")
    p.add_argument("--tile-ids", nargs="+",
                   help="Only process these tile IDs (filename stems, e.g. 45843596)")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel workers (use 1 to minimize memory)")
    p.add_argument("--max-failed", type=int, default=0,
                   help="Exit non-zero if more than this many tiles fail "
                        "(default 0 — any failure fails the run; a silent "
                        "OK=872/Failed=174 hid the missing-LPIS-2021 and "
                        "axis-swap bugs for two days, 2026-07-18..20)")
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

    if args.label_out_dir:
        os.makedirs(args.label_out_dir, exist_ok=True)

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
    fail_reasons: Counter[str] = Counter()
    t0 = time.time()

    print(f"  Executor:  {args.executor} (workers={args.workers})")
    Executor = ProcessPoolExecutor if args.executor == "process" else ThreadPoolExecutor

    with Executor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(build_tile_label, t, args.nmd_raster,
                        args.lpis_dir, args.sks_dir, args.nmd_version,
                        args.label_out_dir): t
            for t in tiles
        }
        for i, f in enumerate(as_completed(futs)):
            r = f.result()
            stats[r.get("status", "failed")] = stats.get(r.get("status", "failed"), 0) + 1
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(tiles)}] {r['name']}: {r['status']} "
                      f"| {(i+1)/elapsed*3600:.0f}/h", flush=True)
            if r["status"] == "failed":
                fail_reasons[r.get("reason", "?")] += 1
                if sum(fail_reasons.values()) <= 10:
                    print(f"  FAIL: {r['name']} — {r.get('reason', '?')}",
                          flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  OK={stats['ok']}  Failed={stats['failed']}")
    if fail_reasons:
        # Per-reason counts — 10 example lines is not a failure report; a
        # systematic error later in the list must be visible in the log.
        print("  Failure reasons:")
        for reason, n in fail_reasons.most_common():
            print(f"    {n:5d} × {reason}")
    if stats["failed"] > args.max_failed:
        print(f"FAILED: {stats['failed']} tiles failed "
              f"(> --max-failed {args.max_failed})")
        sys.exit(1)


if __name__ == "__main__":
    main()
