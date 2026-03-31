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


def gen_from_existing(tiles_dir: str, max_tiles: int | None = None) -> list[dict]:
    """Read tile locations from existing .npz files on disk."""
    import glob
    import re
    tiles = sorted(glob.glob(os.path.join(tiles_dir, "*.npz")))
    print(f"  Found {len(tiles)} existing tiles in {tiles_dir}")

    locs = []
    skipped = 0
    for path in tiles:
        try:
            data = np.load(path, allow_pickle=True)
            bbox = None

            # Method 1: bbox_3006 array
            bbox_arr = data.get("bbox_3006", None)
            if bbox_arr is not None:
                b = bbox_arr.flatten()
                bbox = {"west": int(b[0]), "south": int(b[1]),
                        "east": int(b[2]), "north": int(b[3])}

            # Method 2: easting/northing keys
            if bbox is None and "easting" in data and "northing" in data:
                e, n = int(data["easting"]), int(data["northing"])
                half = TILE_SIZE_M // 2
                bbox = {"west": e - half, "south": n - half,
                        "east": e + half, "north": n + half}

            # Method 3: parse from filename (tile_EASTING_NORTHING.npz)
            if bbox is None:
                m = re.search(r'tile_(\d+)_(\d+)', os.path.basename(path))
                if m:
                    e, n = int(m.group(1)), int(m.group(2))
                    half = TILE_SIZE_M // 2
                    bbox = {"west": e - half, "south": n - half,
                            "east": e + half, "north": n + half}

            if bbox is None:
                skipped += 1
                continue

            # Read tile's base year (LPIS/LUCAS survey year)
            tile_year = None
            if "year" in data:
                tile_year = int(data["year"])
            elif "lpis_year" in data:
                tile_year = int(data["lpis_year"])
            elif "dates" in data:
                # Infer from first valid date
                dates = data["dates"]
                for d in dates:
                    d_str = str(d)
                    if d_str and len(d_str) >= 4:
                        tile_year = int(d_str[:4])
                        break

            # Check if tile has LPIS crop labels
            has_lpis = "label_mask" in data or "lpis_year" in data

            name = os.path.basename(path).replace(".npz", "")
            loc = {
                "name": name,
                "source": str(data.get("source", "lulc")),
                "bbox_3006": bbox,
                "coords_wgs84": bbox_3006_to_wgs84(bbox),
                "_existing_path": path,
                "_has_lpis": has_lpis,
            }
            if tile_year:
                loc["year"] = tile_year
            locs.append(loc)
        except Exception:
            continue

    if skipped > 0:
        print(f"  Skipped {skipped} tiles (no bbox recoverable)")

    if max_tiles and len(locs) > max_tiles:
        random.shuffle(locs)
        locs = locs[:max_tiles]

    print(f"  Using {len(locs)} tile locations")
    return locs


def refetch_tile(
    loc: dict,
    years: list[str],
    output_dir: str,
    cloud_max: float = 30.0,
) -> dict:
    """Re-fetch spectral data for an existing tile, keep all other fields."""
    name = loc["name"]
    existing_path = loc.get("_existing_path")
    out_path = os.path.join(output_dir, f"{name}.npz")

    # Skip if already re-fetched (check for multitemporal flag)
    if os.path.exists(out_path):
        try:
            d = np.load(out_path, allow_pickle=True)
            if d.get("multitemporal", 0) == 1 and d.get("num_frames", 0) == 4:
                return {"name": name, "status": "skipped"}
        except Exception:
            pass

    bbox = loc["bbox_3006"]
    coords = loc.get("coords_wgs84") or bbox_3006_to_wgs84(bbox)

    # Determine fetch years based on tile type
    # Crop tiles: strict year match only (LPIS labels are year-specific)
    # Forest/water tiles: tile year first, other years as fallback
    tile_year = loc.get("year")
    has_crop_labels = loc.get("source") == "crop" or loc.get("_has_lpis", False)

    if tile_year and has_crop_labels:
        # Crop tile — NO year fallback, spectral must match label year
        fetch_years = [str(tile_year)]
    elif tile_year:
        # LULC tile — tile year first, others as fallback (forest/water OK)
        fetch_years = [str(tile_year)] + [y for y in years if y != str(tile_year)]
    else:
        fetch_years = years

    # Fetch 4 new spectral frames
    scene_results = fetch_4frame_scenes(
        bbox, coords, fetch_years, scene_cloud_max=cloud_max,
    )
    image, temporal_mask, doy, dates = stack_frames(scene_results, NUM_FRAMES)

    if int(temporal_mask.sum()) == 0:
        return {"name": name, "status": "failed", "reason": "no_scenes"}

    # Load existing tile data (labels, aux, etc.)
    save = {}
    if existing_path and os.path.exists(existing_path):
        try:
            old = dict(np.load(existing_path, allow_pickle=True))
            # Keep everything except old spectral/temporal
            for k, v in old.items():
                if k not in ("image", "spectral", "temporal_mask", "doy",
                             "dates", "multitemporal", "num_frames",
                             "num_bands", "seasons_valid"):
                    save[k] = v
        except Exception:
            pass

    # Write new spectral + temporal metadata
    save["image"] = image
    save["temporal_mask"] = temporal_mask
    save["doy"] = doy
    save["dates"] = np.array(dates)
    save["multitemporal"] = np.int32(1)
    save["num_frames"] = np.int32(NUM_FRAMES)
    save["num_bands"] = np.int32(N_BANDS)

    np.savez_compressed(out_path, **save)
    return {"name": name, "status": "ok",
            "valid_frames": int(temporal_mask.sum())}


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="Unified 4-frame tile fetcher")
    p.add_argument("--mode", required=True,
                   choices=["lulc", "rare-crops", "urban", "all", "refetch"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--years", nargs="+", default=["2018", "2019", "2022", "2023"])
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--cloud-max", type=float, default=30.0)
    p.add_argument("--max-tiles", type=int, default=None)
    p.add_argument("--grid-spacing", type=int, default=2500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lpis-dir", default="data/lpis")
    p.add_argument("--n-oljevaxter", type=int, default=500)
    p.add_argument("--n-havre", type=int, default=500)
    p.add_argument("--min-population", type=int, default=2000)
    p.add_argument("--from-existing", nargs="+", default=None,
                   help="Directories with existing tiles to re-fetch spectral for")
    args = p.parse_args()
    random.seed(args.seed)

    print(f"=== Unified 4-Frame Tile Fetcher ===")
    print(f"  Mode: {args.mode}  Years: {args.years}")
    print(f"  Frame 0: Autumn (Sep-Oct, year-1)")
    print(f"  Frames 1-3: VPP-guided growing season\n")

    use_refetch = args.mode == "refetch" or args.from_existing
    work: list[tuple[dict, str]] = []

    # Refetch mode: read existing tile locations, re-fetch spectral only
    if use_refetch:
        dirs = args.from_existing or [args.output_dir]
        for d in dirs:
            os.makedirs(args.output_dir, exist_ok=True)
            for loc in gen_from_existing(d, args.max_tiles):
                work.append((loc, args.output_dir))
    else:
        # All modes write to same output dir — no subdirs.
        # LULC, crop, and urban tiles are handled identically.
        os.makedirs(args.output_dir, exist_ok=True)

        if args.mode in ("lulc", "all"):
            for loc in gen_lulc(args.grid_spacing, args.max_tiles):
                work.append((loc, args.output_dir))

        if args.mode in ("rare-crops", "all"):
            for loc in gen_rare_crops(args.lpis_dir, args.n_oljevaxter, args.n_havre):
                work.append((loc, args.output_dir))

        if args.mode in ("urban", "all"):
            for loc in gen_urban(max_tiles=args.max_tiles or 500,
                                 min_pop=args.min_population):
                work.append((loc, args.output_dir))

    print(f"Total: {len(work)} tiles\n")
    if not work:
        return

    stats = {"ok": 0, "skipped": 0, "failed": 0}
    t0 = time.time()

    def _run(item):
        loc, d = item
        if use_refetch:
            return refetch_tile(loc, args.years, d, args.cloud_max)
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
