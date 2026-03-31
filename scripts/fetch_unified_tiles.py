#!/usr/bin/env python3
"""
scripts/fetch_unified_tiles.py — Unified 4-frame tile fetcher

Fetches S2 tiles with a consistent temporal pattern for ALL sources:
  - LULC grid (uniform across Sweden)
  - Rare crops (LPIS oljeväxter + havre centroids)
  - Urban areas (SCB tätort)

Frame pattern (1 autumn + 3 growing season):
  0: Sep–Oct   post-harvest stubble, autumn-sown crops
  1: Apr–May   bare soil vs green, spring plowing
  2: Jun–Jul   peak NDVI, max crop differentiation
  3: Aug       harvest timing, rapeseed done vs cereals

Usage:
    python scripts/fetch_unified_tiles.py --mode all \\
        --output-dir data/unified_v2 --years 2022 2023
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imint.config.env import load_env

load_env()

from imint.training.tile_fetch import (
    TILE_SIZE_M,
    N_BANDS,
    bbox_3006_to_wgs84,
    fetch_4frame_scenes,
    fetch_aux_channels,
    fetch_nmd_label_local,
    stack_frames,
)
from imint.training.sampler import generate_grid, grid_to_wgs84
from imint.training.scb_tatort import generate_scb_densification_regions

NUM_FRAMES = 4  # 1 autumn (year-1) + 3 VPP-guided growing season

# Rare crop grödkoder
SJV_OLJEVAXTER = {85, 86, 87, 88, 90, 91, 92}
SJV_HAVRE = {5}


# ── Location Generators ──────────────────────────────────────────────────────


def gen_lulc(spacing_m: int = 2500, max_tiles: int | None = None) -> list[dict]:
    """LULC grid locations across Sweden."""
    print("  Generating LULC grid...")
    cells = grid_to_wgs84(generate_grid(
        spacing_m=spacing_m, patch_size_m=TILE_SIZE_M, land_filter=True,
    ))
    print(f"  {len(cells)} land cells at {spacing_m}m")

    locs = [{
        "name": f"tile_{c.easting}_{c.northing}",
        "source": "lulc",
        "bbox_3006": {"west": c.west_3006, "south": c.south_3006,
                      "east": c.east_3006, "north": c.north_3006},
        "coords_wgs84": {"west": c.west_wgs84, "south": c.south_wgs84,
                         "east": c.east_wgs84, "north": c.north_wgs84},
    } for c in cells]

    if max_tiles:
        random.shuffle(locs)
        locs = locs[:max_tiles]
    return locs


def gen_rare_crops(
    lpis_dir: str = "data/lpis",
    n_oljevaxter: int = 500,
    n_havre: int = 500,
    min_area_ha: float = 5.0,
    lpis_years: list[int] | None = None,
) -> list[dict]:
    """Tile locations centred on rare LPIS crop parcels."""
    import geopandas as gpd

    if lpis_years is None:
        lpis_years = [2022, 2023, 2024]

    frames = []
    for year in lpis_years:
        pq = Path(lpis_dir) / f"jordbruksskiften_{year}.parquet"
        if not pq.exists():
            continue
        print(f"  Loading LPIS {year}...")
        gdf = gpd.read_parquet(pq)

        # Find grödkod + area columns
        grodkod_col = next(
            (c for c in gdf.columns if "grd" in c.lower() or "grod" in c.lower()),
            None,
        )
        area_col = next(
            (c for c in gdf.columns if "areal" in c.lower()), None,
        )
        if grodkod_col is None:
            continue

        gdf["_gk"] = gdf[grodkod_col].astype(int)
        gdf["_year"] = year
        if area_col:
            gdf = gdf[gdf[area_col].astype(float) >= min_area_ha]

        rare = gdf[gdf["_gk"].isin(SJV_OLJEVAXTER | SJV_HAVRE)]
        if area_col:
            rare = rare.sort_values(area_col, ascending=False)
        print(f"    {len(rare)} rare parcels (≥{min_area_ha} ha)")
        frames.append(rare)

    if not frames:
        return []
    combined = gpd.pd.concat(frames, ignore_index=True)
    if combined.crs and combined.crs.to_epsg() != 3006:
        combined = combined.to_crs(epsg=3006)

    locs = []
    for target_codes, n, label in [
        (SJV_OLJEVAXTER, n_oljevaxter, "oljevaxter"),
        (SJV_HAVRE, n_havre, "havre"),
    ]:
        sub = combined[combined["_gk"].isin(target_codes)].head(n)
        print(f"  {label}: {len(sub)} parcels")
        for _, row in sub.iterrows():
            cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
            half = TILE_SIZE_M // 2
            bbox = {"west": int(cx - half), "south": int(cy - half),
                    "east": int(cx + half), "north": int(cy + half)}
            locs.append({
                "name": f"crop_{label}_{int(cx)}_{int(cy)}",
                "source": "crop",
                "bbox_3006": bbox,
                "coords_wgs84": bbox_3006_to_wgs84(bbox),
                "year": int(row["_year"]),
            })
    return locs


def gen_urban(
    cache_dir: str = "data/cache",
    min_pop: int = 2000,
    spacing_m: int = 2500,
    max_tiles: int = 500,
) -> list[dict]:
    """Tile locations within SCB tätort urban areas."""
    from rasterio.crs import CRS
    from rasterio.warp import transform

    print("  Fetching SCB tätort regions...")
    regions = generate_scb_densification_regions(
        cache_dir=Path(cache_dir), min_population=min_pop,
        patch_size_m=TILE_SIZE_M,
    )
    print(f"  {len(regions)} tätorter (pop ≥ {min_pop})")

    locs = []
    half = TILE_SIZE_M // 2
    for region in regions:
        w, e, s, n = region["bbox_3006"]
        cx = w + half
        while cx + half <= e:
            cy = s + half
            while cy + half <= n:
                _, lats = transform(
                    CRS.from_epsg(3006), CRS.from_epsg(4326), [cx], [cy],
                )
                if lats[0] >= 64.0:
                    cy += spacing_m
                    continue
                bbox = {"west": cx - half, "south": cy - half,
                        "east": cx + half, "north": cy + half}
                locs.append({
                    "name": f"urban_{int(cx)}_{int(cy)}",
                    "source": "urban",
                    "bbox_3006": bbox,
                    "coords_wgs84": bbox_3006_to_wgs84(bbox),
                    "region": region["label"],
                })
                cy += spacing_m
            cx += spacing_m

    if len(locs) > max_tiles:
        random.shuffle(locs)
        locs = locs[:max_tiles]
    print(f"  {len(locs)} urban tiles")
    return locs


# ── Core Fetch ────────────────────────────────────────────────────────────────


def fetch_tile(
    loc: dict,
    years: list[str],
    output_dir: str,
    cloud_max: float = 30.0,
) -> dict:
    """Fetch one tile: 4 seasonal frames + NMD + aux → .npz."""
    name = loc["name"]
    out_path = os.path.join(output_dir, f"{name}.npz")
    if os.path.exists(out_path):
        return {"name": name, "status": "skipped"}

    bbox = loc["bbox_3006"]
    coords = loc.get("coords_wgs84") or bbox_3006_to_wgs84(bbox)
    tile_years = [str(loc["year"])] if "year" in loc else years

    scene_results = fetch_4frame_scenes(
        bbox, coords, tile_years,
        scene_cloud_max=cloud_max,
    )

    image, temporal_mask, doy, dates = stack_frames(scene_results, NUM_FRAMES)
    if int(temporal_mask.sum()) == 0:
        return {"name": name, "status": "failed", "reason": "no_scenes"}

    nmd_label = fetch_nmd_label_local(bbox)
    aux = fetch_aux_channels(bbox)

    save = {
        "image": image,
        "temporal_mask": temporal_mask,
        "doy": doy,
        "dates": np.array(dates),
        "multitemporal": np.int32(1),
        "num_frames": np.int32(NUM_FRAMES),
        "num_bands": np.int32(N_BANDS),
        "bbox_3006": np.array([bbox["west"], bbox["south"],
                               bbox["east"], bbox["north"]], dtype=np.int32),
        "easting": np.int32((bbox["west"] + bbox["east"]) // 2),
        "northing": np.int32((bbox["south"] + bbox["north"]) // 2),
        "source": loc["source"],
    }
    if nmd_label is not None:
        save["label"] = nmd_label
    save.update(aux)

    np.savez_compressed(out_path, **save)
    return {"name": name, "status": "ok",
            "valid_frames": int(temporal_mask.sum())}


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="Unified 4-frame tile fetcher")
    p.add_argument("--mode", required=True,
                   choices=["lulc", "rare-crops", "urban", "all"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--years", nargs="+", default=["2022", "2023"])
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--cloud-max", type=float, default=30.0)
    p.add_argument("--max-tiles", type=int, default=None)
    p.add_argument("--grid-spacing", type=int, default=2500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lpis-dir", default="data/lpis")
    p.add_argument("--n-oljevaxter", type=int, default=500)
    p.add_argument("--n-havre", type=int, default=500)
    p.add_argument("--min-population", type=int, default=2000)
    args = p.parse_args()
    random.seed(args.seed)

    print(f"=== Unified 4-Frame Tile Fetcher ===")
    print(f"  Mode: {args.mode}  Years: {args.years}")
    print(f"  Frame 0: Autumn (Sep-Oct, year-1)")
    print(f"  Frames 1-3: VPP-guided growing season\n")

    work: list[tuple[dict, str]] = []

    if args.mode in ("lulc", "all"):
        d = os.path.join(args.output_dir, "lulc") if args.mode == "all" else args.output_dir
        os.makedirs(d, exist_ok=True)
        for loc in gen_lulc(args.grid_spacing, args.max_tiles):
            work.append((loc, d))

    if args.mode in ("rare-crops", "all"):
        d = os.path.join(args.output_dir, "crop") if args.mode == "all" else args.output_dir
        os.makedirs(d, exist_ok=True)
        for loc in gen_rare_crops(args.lpis_dir, args.n_oljevaxter, args.n_havre):
            work.append((loc, d))

    if args.mode in ("urban", "all"):
        d = os.path.join(args.output_dir, "urban") if args.mode == "all" else args.output_dir
        os.makedirs(d, exist_ok=True)
        for loc in gen_urban(max_tiles=args.max_tiles or 500,
                             min_pop=args.min_population):
            work.append((loc, d))

    print(f"Total: {len(work)} tiles\n")
    if not work:
        return

    stats = {"ok": 0, "skipped": 0, "failed": 0}
    t0 = time.time()

    def _run(item):
        loc, d = item
        return fetch_tile(loc, args.years, d, args.cloud_max)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_run, w): w for w in work}
        for i, f in enumerate(as_completed(futs)):
            r = f.result()
            if r:
                stats[r.get("status", "failed")] = \
                    stats.get(r.get("status", "failed"), 0) + 1
                if (i + 1) % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"  [{i+1}/{len(work)}] {r['name']}: {r['status']} "
                          f"| {(i+1)/elapsed*3600:.0f}/h")

    elapsed = time.time() - t0
    print(f"\n=== Done in {elapsed/60:.1f} min ===")
    print(f"  OK={stats['ok']}  Skipped={stats['skipped']}  Failed={stats['failed']}")

    json.dump({
        "mode": args.mode, "years": args.years,
        "strategy": "autumn(year-1) + 3 VPP-guided growing season",
        "stats": stats,
    }, open(os.path.join(args.output_dir, "manifest.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
