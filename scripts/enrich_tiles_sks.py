#!/usr/bin/env python3
"""
Enrich LULC tiles with Skogsstyrelsen harvest data.

Adds two masks per tile:
  - harvest_mask:        (256,256) uint8 — binary, 1 = utförd avverkning
                         (skog som avverkats EFTER tilens spektrala datum)
                         → segmenteringsklass 18 ("hygge") i unified schema
  - mature_forest_mask:  (256,256) uint8 — binary, 1 = avverkningsmogen skog
                         (skog som anmälts för avverkning EFTER tilens datum)
                         → auxiliary channel (inte egen klass)

Temporal logic:
  LULC tiles have 2018/2019 spectral data.
  SKS data covers 2021-2026.

  For a tile with image date 2019-07-15:
    - An avverkning utförd 2022 → the forest was standing in 2019
      but has since been harvested. At tile time = forest.
      Used as "avverkningsmogen skog" proxy (aux channel).
    - An avverkning anmäld 2021 → the forest was standing in 2019,
      owner intends to harvest. At tile time = mature forest (aux).

  For "hygge" class: We need avverkningar utförda BEFORE tile date.
  Since SKS data starts 2021, no actual hyggen match 2018/2019 tiles.
  → harvest_mask will be empty for 2018/2019 tiles (as expected).
  → mature_forest_mask captures future harvest intent (useful aux).

Usage:
    python scripts/enrich_tiles_sks.py --tiles-dir data/lulc_seasonal/tiles --limit 50
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imint.training.tile_bbox import resolve_fetch_bbox


def _get_tile_dates(data: dict) -> list[str]:
    """Extract date strings from tile data."""
    dates = data.get("dates", [])
    return [str(d).strip() for d in dates if str(d).strip() and str(d).strip() != "nan"]


def _get_tile_max_year(data: dict) -> int:
    """Get the latest year from a tile's dates."""
    dates = _get_tile_dates(data)
    years = []
    for d in dates:
        if len(d) >= 4 and d[:4].isdigit():
            years.append(int(d[:4]))
    return max(years) if years else 0


def rasterize_sks(
    gdf,
    bbox: tuple[int, int, int, int],
    tile_size: int,
) -> tuple[np.ndarray, int]:
    """Rasterize SKS polygons within bbox to a binary mask.

    Returns (mask, n_polygons).
    """
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    from shapely.geometry import box as shapely_box

    west, south, east, north = bbox
    tile_box = shapely_box(west, south, east, north)

    # Spatial index query
    candidates = list(gdf.sindex.intersection(tile_box.bounds))
    if not candidates:
        return np.zeros((tile_size, tile_size), dtype=np.uint8), 0

    clipped = gdf.iloc[candidates]
    clipped = clipped[clipped.geometry.intersects(tile_box)]

    if clipped.empty:
        return np.zeros((tile_size, tile_size), dtype=np.uint8), 0

    n = len(clipped)
    shapes = [(geom, 1) for geom in clipped.geometry]
    transform = from_bounds(west, south, east, north, tile_size, tile_size)

    mask = rasterize(
        shapes,
        out_shape=(tile_size, tile_size),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )
    # SKS uses standard E,N (EPSG:3006) — rasterize() with from_bounds(west,south,east,north)
    # already produces a correct North-up array. No rotation needed (unlike LPIS which is N,E).
    return mask, n


def process_tile(
    tile_path: str,
    utforda_gdf,
    anmalda_gdf,
) -> dict:
    """Enrich a single tile with SKS harvest + mature forest masks."""
    name = os.path.basename(tile_path)
    try:
        data = dict(np.load(tile_path, allow_pickle=True))

        # Skip if already enriched
        if "harvest_mask" in data and "mature_forest_mask" in data:
            return {"name": name, "status": "skip"}

        # bbox + raster size via the shared SSOT resolver (extent + size coupled
        # to the tile's own pixel grid at 10 m GSD) — no module TILE_SIZE_PX.
        bbox_d, size = resolve_fetch_bbox(name=Path(tile_path).stem, npz_data=data)
        if bbox_d is None:
            return {"name": name, "status": "error", "msg": "no bbox"}
        bbox = (bbox_d["west"], bbox_d["south"], bbox_d["east"], bbox_d["north"])

        # Rasterize utförda avverkningar → harvest_mask (binary)
        harvest_mask, n_harvest = rasterize_sks(utforda_gdf, bbox, tile_size=size)

        # Rasterize avverkningsanmälningar → anmalda_mask (binary)
        anmalda_mask, n_mature = rasterize_sks(anmalda_gdf, bbox, tile_size=size)

        # Save the real SKS layers only (the synthetic harvest_probability
        # channel was dropped — it leaked the harvest target into the input).
        data["harvest_mask"] = harvest_mask           # Binary: utförd avverkning
        data["n_harvest_polygons"] = np.int32(n_harvest)
        data["n_mature_polygons"] = np.int32(n_mature)

        np.savez(tile_path, **data)

        h_pct = float((harvest_mask > 0).mean() * 100)
        m_pct = float((anmalda_mask > 0).mean() * 100)
        return {
            "name": name, "status": "ok",
            "harvest_pct": h_pct, "mature_pct": m_pct,
            "n_harvest": n_harvest, "n_mature": n_mature,
        }

    except Exception as e:
        return {"name": name, "status": "error", "msg": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Enrich tiles with SKS harvest + mature forest masks"
    )
    parser.add_argument("--tiles-dir", type=str, default="data/lulc_seasonal/tiles")
    parser.add_argument("--sks-dir", type=str, default="data/sks")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    import geopandas as gpd

    tiles = sorted(glob.glob(os.path.join(args.tiles_dir, "*.npz")))
    if args.limit > 0:
        tiles = tiles[:args.limit]

    print(f"Enriching {len(tiles)} tiles with SKS harvest data...")

    # Paths
    utforda_path = os.path.join(args.sks_dir, "utforda_avverkningar_spatial.parquet")
    if not os.path.exists(utforda_path):
        utforda_path = os.path.join(args.sks_dir, "utforda_avverkningar.parquet")
    anmalda_path = os.path.join(args.sks_dir, "avverkningsanmalningar.parquet")

    # Load both datasets into RAM with spatial index
    print(f"Loading utförda avverkningar ({utforda_path})...")
    t0 = time.time()
    utforda = gpd.read_parquet(utforda_path)
    _ = utforda.sindex  # Build R-tree spatial index
    print(f"  {len(utforda)} polygoner loaded in {time.time()-t0:.0f}s")

    print(f"Loading avverkningsanmälningar ({anmalda_path})...")
    t0 = time.time()
    anmalda = gpd.read_parquet(anmalda_path)
    _ = anmalda.sindex
    print(f"  {len(anmalda)} anmälningar loaded in {time.time()-t0:.0f}s")

    ok, skip, error = 0, 0, 0
    total_harvest_pct = 0.0
    total_mature_pct = 0.0
    t0 = time.time()

    for i, tile_path in enumerate(tiles):
        result = process_tile(tile_path, utforda, anmalda)

        if result["status"] == "ok":
            ok += 1
            total_harvest_pct += result.get("harvest_pct", 0)
            total_mature_pct += result.get("mature_pct", 0)
        elif result["status"] == "skip":
            skip += 1
        else:
            error += 1
            if error <= 3:
                print(f"  ERROR {result['name']}: {result.get('msg', '?')}")

        if result["status"] == "ok":
            ok += 1
            total_harvest_pct += result.get("harvest_pct", 0)
            total_mature_pct += result.get("mature_pct", 0)
        elif result["status"] == "skip":
            skip += 1
        else:
            error += 1
            if error <= 5:
                print(f"  ERROR {result['name']}: {result.get('msg', '?')}")

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            avg_h = total_harvest_pct / max(ok, 1)
            avg_m = total_mature_pct / max(ok, 1)
            print(f"  [{i+1}/{len(tiles)}] ok={ok} skip={skip} err={error} "
                  f"avg_harvest={avg_h:.1f}% avg_mature={avg_m:.1f}% "
                  f"({rate:.0f} tiles/s)")

    elapsed = time.time() - t0
    avg_h = total_harvest_pct / max(ok, 1)
    avg_m = total_mature_pct / max(ok, 1)
    print(f"\nDone in {elapsed:.0f}s!")
    print(f"  OK: {ok} (avg harvest={avg_h:.1f}%, avg mature={avg_m:.1f}%)")
    print(f"  Skipped: {skip}")
    print(f"  Errors: {error}")


if __name__ == "__main__":
    main()
