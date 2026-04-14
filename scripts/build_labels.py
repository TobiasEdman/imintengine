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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


# Lazy-loaded shared data (loaded once, reused across threads)
_lpis_gdfs: dict = {}  # year → GeoDataFrame
_sks_utforda = None
_sks_anmalda = None
_rasterize_parcels = None
_rasterize_sks = None
_compute_harvest_probability = None


def _load_lpis(year: int, lpis_dir: str):
    """Load LPIS GeoDataFrame for a year (cached)."""
    global _lpis_gdfs, _rasterize_parcels
    if year in _lpis_gdfs:
        return _lpis_gdfs[year]

    if _rasterize_parcels is None:
        from scripts.enrich_tiles_lpis_mask import get_lpis_gdf, rasterize_parcels
        _rasterize_parcels = rasterize_parcels

    from scripts.enrich_tiles_lpis_mask import get_lpis_gdf
    gdf = get_lpis_gdf(year, lpis_dir)
    if gdf is not None and len(gdf) > 0:
        _lpis_gdfs[year] = gdf
        print(f"  Loaded LPIS {year}: {len(gdf):,} parcels", flush=True)
    return _lpis_gdfs.get(year)


def _load_sks(sks_dir: str):
    """Load SKS GeoDataFrames (cached)."""
    global _sks_utforda, _sks_anmalda, _rasterize_sks, _compute_harvest_probability
    if _sks_utforda is not None:
        return

    from scripts.enrich_tiles_sks import rasterize_sks, compute_harvest_probability
    _rasterize_sks = rasterize_sks
    _compute_harvest_probability = compute_harvest_probability

    import geopandas as gpd

    for name in ["utforda_avverkningar_spatial.parquet", "utforda_avverkningar.parquet"]:
        path = os.path.join(sks_dir, name)
        if os.path.exists(path):
            import pandas as pd
            _sks_utforda = gpd.read_parquet(path)
            _sks_utforda["Avvdatum"] = pd.to_datetime(
                _sks_utforda["Avvdatum"], errors="coerce"
            )
            _ = _sks_utforda.sindex
            print(f"  Loaded SKS utförda: {len(_sks_utforda):,} polygons", flush=True)
            break

    path = os.path.join(sks_dir, "avverkningsanmalningar.parquet")
    if os.path.exists(path):
        _sks_anmalda = gpd.read_parquet(path)
        _ = _sks_anmalda.sindex
        print(f"  Loaded SKS anmälda: {len(_sks_anmalda):,} polygons", flush=True)


def build_tile_label(
    tile_path: str,
    nmd_raster: str,
    lpis_dir: str,
    sks_dir: str,
) -> dict:
    """Build unified 23-class label for one tile from scratch."""
    name = os.path.basename(tile_path).replace(".npz", "")
    try:
        data = dict(np.load(tile_path, allow_pickle=True))

        # Extract bbox
        bbox_3006 = None
        if "bbox_3006" in data:
            b = data["bbox_3006"].flatten()
            bbox_3006 = {"west": int(b[0]), "south": int(b[1]),
                         "east": int(b[2]), "north": int(b[3])}
        elif "easting" in data and "northing" in data:
            e, n = int(data["easting"]), int(data["northing"])
            half = 1280
            bbox_3006 = {"west": e - half, "south": n - half,
                         "east": e + half, "north": n + half}
        else:
            # Try filename
            import re
            m = re.search(r'tile_(\d+)_(\d+)', name)
            if m:
                e, n = int(m.group(1)), int(m.group(2))
                half = 1280
                bbox_3006 = {"west": e - half, "south": n - half,
                             "east": e + half, "north": n + half}

        if bbox_3006 is None:
            return {"name": name, "status": "failed", "reason": "no_bbox"}

        # Determine tile year
        tile_year = None
        if "year" in data:
            tile_year = int(data["year"])
        elif "lpis_year" in data:
            tile_year = int(data["lpis_year"])
        elif "dates" in data:
            for d in data["dates"]:
                s = str(d)
                if s and len(s) >= 4:
                    tile_year = int(s[:4])
                    break
        if tile_year is None:
            tile_year = 2022  # default

        # --- Step 1: NMD label from local raster ---
        nmd_label = fetch_nmd_label_local(bbox_3006, nmd_raster=nmd_raster)
        if nmd_label is None:
            return {"name": name, "status": "failed", "reason": "no_nmd"}

        # Connected-component area map: same inverse-area weighting as LPIS.
        # Stored separately; unified_dataset.py merges with parcel_area_ha.
        data["nmd_area_ha"] = _compute_nmd_area_map(nmd_label)

        # --- Step 2: LPIS crop mask ---
        lpis_mask = None
        gdf = _load_lpis(tile_year, lpis_dir)
        if gdf is not None:
            bbox_arr = np.array([bbox_3006["west"], bbox_3006["south"],
                                 bbox_3006["east"], bbox_3006["north"]])
            lpis_mask, area_map, n_parcels = _rasterize_parcels(gdf, bbox_arr)
            data["label_mask"]     = lpis_mask   # uint16 raw SJV codes
            data["parcel_area_ha"] = area_map    # float32 ha/pixel (for loss weighting)
            data["n_parcels"]      = np.int32(n_parcels)

        # --- Step 3: SKS harvest mask (filtered by tile year) ---
        # Hygge = avverkat inom 5 år före tile-året
        harvest_mask = None
        _load_sks(sks_dir)
        bbox_tuple = (bbox_3006["west"], bbox_3006["south"],
                      bbox_3006["east"], bbox_3006["north"])
        if _sks_utforda is not None:
            import pandas as pd
            min_date = pd.Timestamp(f"{tile_year - 5}-01-01")
            max_date = pd.Timestamp(f"{tile_year}-12-31")
            sks_filtered = _sks_utforda[
                (_sks_utforda["Avvdatum"] >= min_date) &
                (_sks_utforda["Avvdatum"] <= max_date)
            ]
            if len(sks_filtered) > 0:
                harvest_mask, n_harvest = _rasterize_sks(sks_filtered, bbox_tuple)
                data["harvest_mask"] = harvest_mask
                data["n_harvest_polygons"] = np.int32(n_harvest)

        if _sks_anmalda is not None:
            mature_mask, n_mature = _rasterize_sks(_sks_anmalda, bbox_tuple)
            data["n_mature_polygons"] = np.int32(n_mature)

            if _compute_harvest_probability is not None and harvest_mask is not None:
                data["harvest_probability"] = _compute_harvest_probability(
                    harvest_mask, mature_mask,
                )

        # --- Step 4: Merge all → unified 20-class ---
        unified = merge_all(nmd_label, lpis_mask, harvest_mask)
        data["label"] = unified
        data["nmd_label_raw"] = nmd_label

        # Save back
        np.savez_compressed(tile_path, **data)
        return {"name": name, "status": "ok"}

    except Exception as e:
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
    p.add_argument("--tile-size-px", type=int, default=256,
                   help="Tile pixel resolution (256 or 512)")
    args = p.parse_args()

    if args.tile_size_px != 256:
        import imint.training.tile_fetch as _tf
        _tf.TILE_SIZE_PX = args.tile_size_px
        _tf.TILE_SIZE_M = args.tile_size_px * 10
        print(f"  Tile size: {_tf.TILE_SIZE_PX}px ({_tf.TILE_SIZE_M}m)")

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

    # LPIS and SKS are lazy-loaded on demand per tile (see _load_lpis / _load_sks).
    # Pre-loading all years at once would exceed memory on small pods.
    _load_sks(args.sks_dir)

    stats = {"ok": 0, "failed": 0}
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
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
