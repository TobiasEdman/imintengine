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


TILE_SIZE_PX = 256
TILE_SIZE_M = 2560  # 256 * 10m


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


def _bbox_from_tile(data: dict) -> tuple[int, int, int, int] | None:
    """Reconstruct bbox [west, south, east, north] from tile easting/northing."""
    if "bbox_3006" in data:
        bbox = data["bbox_3006"]
        return int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

    if "easting" in data and "northing" in data:
        easting = int(data["easting"])
        northing = int(data["northing"])
        half = TILE_SIZE_M // 2
        return easting - half, northing - half, easting + half, northing + half

    return None


def rasterize_sks(
    gdf,
    bbox: tuple[int, int, int, int],
    tile_size: int = TILE_SIZE_PX,
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


def compute_harvest_probability(
    utforda_mask: np.ndarray,
    anmalda_mask: np.ndarray,
    decay_distance_m: float = 500.0,
    pixel_size_m: float = 10.0,
) -> np.ndarray:
    """Compute per-pixel harvest probability as a continuous float32 channel.

    Logic:
      - Within anmäld avverkning polygon: p = 1.0
      - Within utförd avverkning polygon: p = 0.9
      - Near either (within decay_distance): p decays with distance
      - Elsewhere: p = 0.0

    This creates a smooth gradient around harvest areas, giving the model
    context about proximity to planned/executed harvests.

    Args:
        utforda_mask: (H, W) uint8, 1=utförd avverkning
        anmalda_mask: (H, W) uint8, 1=anmäld avverkning
        decay_distance_m: Distance in meters over which probability decays to 0
        pixel_size_m: Pixel resolution in meters (10m for S2)

    Returns:
        (H, W) float32, harvest probability [0.0, 1.0]
    """
    from scipy.ndimage import distance_transform_edt

    h, w = utforda_mask.shape
    prob = np.zeros((h, w), dtype=np.float32)

    # Start with anmälda (highest: p=1.0 inside polygon)
    if anmalda_mask.any():
        # Distance from nearest anmäld pixel (in pixels)
        dist_anm = distance_transform_edt(anmalda_mask == 0) * pixel_size_m
        decay_pixels = decay_distance_m
        # Inside polygon: 1.0, decaying to 0 at decay_distance
        anm_prob = np.clip(1.0 - dist_anm / decay_pixels, 0.0, 1.0)
        prob = np.maximum(prob, anm_prob)

    # Add utförda (p=0.9 inside, decaying)
    if utforda_mask.any():
        dist_utf = distance_transform_edt(utforda_mask == 0) * pixel_size_m
        utf_prob = np.clip(0.9 * (1.0 - dist_utf / decay_distance_m), 0.0, 0.9)
        prob = np.maximum(prob, utf_prob)

    return prob


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

        bbox = _bbox_from_tile(data)
        if bbox is None:
            return {"name": name, "status": "error", "msg": "no bbox"}

        # Rasterize utförda avverkningar → harvest_mask (binary)
        harvest_mask, n_harvest = rasterize_sks(utforda_gdf, bbox)

        # Rasterize avverkningsanmälningar → anmalda_mask (binary)
        anmalda_mask, n_mature = rasterize_sks(anmalda_gdf, bbox)

        # Compute continuous harvest probability (float32, 0-1)
        # Combines both sources with distance decay
        harvest_prob = compute_harvest_probability(harvest_mask, anmalda_mask)

        # Save
        data["harvest_mask"] = harvest_mask           # Binary: utförd avverkning
        data["harvest_probability"] = harvest_prob    # Float32 [0,1]: aux channel
        data["n_harvest_polygons"] = np.int32(n_harvest)
        data["n_mature_polygons"] = np.int32(n_mature)

        np.savez(tile_path, **data)

        h_pct = float((harvest_mask > 0).mean() * 100)
        m_pct = float((anmalda_mask > 0).mean() * 100)
        prob_mean = float(harvest_prob[harvest_prob > 0].mean()) if (harvest_prob > 0).any() else 0.0
        return {
            "name": name, "status": "ok",
            "harvest_pct": h_pct, "mature_pct": m_pct,
            "prob_mean": prob_mean,
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
