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
import glob
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
    TILE_SIZE_PX,
    N_BANDS,
    bbox_3006_to_wgs84,
    fetch_4frame_scenes,
    fetch_aux_channels,
    fetch_background_frame,
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
            # EPSG:3006 (SWEREF99 TM) axis-order fix:
            #   EPSG authority defines axis order as (northing, easting).
            #   Older pyproj/geopandas gave centroid.x = easting (~250k–920k).
            #   Newer pyproj (≥2.2) respects authority order: centroid.x = northing
            #   (~6.1M–7.7M).  Detect and normalise to (easting, northing).
            if cx > 2_000_000:   # cx is clearly a northing value → swap
                cx, cy = cy, cx  # cx = easting, cy = northing
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


# ── VPP Prefetch ──────────────────────────────────────────────────────────────


def prefetch_vpp_batch(
    work_items: list[tuple[dict, str]],
    workers: int = 4,
    cache_path: str | None = None,
) -> dict[str, list[tuple[int, int]]]:
    """Pre-fetch VPP growing-season windows for all tiles in parallel.

    Results are cached to disk so restarts skip the VPP phase entirely.

    Args:
        work_items: List of (loc, output_dir) from the main work queue.
        workers: Concurrent VPP requests (default 4; VPP is light-weight).
        cache_path: Path to JSON cache file.  If None, derived from
            the output directory.

    Returns:
        Dict mapping tile name → list of (doy_start, doy_end) tuples.
    """
    from imint.training.tile_fetch import _get_vpp_doy_windows

    # Try loading from disk cache
    if cache_path is None and work_items:
        cache_path = os.path.join(work_items[0][1], ".vpp_cache.json")

    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        # Convert lists back to tuples
        vpp_cache = {k: [tuple(w) for w in v] for k, v in cached.items()}
        # Check how many of our tiles are cached
        needed = {loc["name"] for loc, _ in work_items}
        hits = needed & set(vpp_cache.keys())
        if len(hits) >= len(needed) * 0.95:
            print(f"\n=== VPP cache: {len(hits)}/{len(needed)} tiles from {cache_path} ===\n")
            return vpp_cache
        print(f"  VPP cache: {len(hits)}/{len(needed)} (partial, re-fetching missing)")

    total = len(work_items)
    print(f"\n=== VPP prefetch: {total} tiles ({workers} workers) ===")
    vpp_cache: dict[str, list[tuple[int, int]]] = {}

    # Keep any existing cached entries
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            for k, v in json.load(f).items():
                vpp_cache[k] = [tuple(w) for w in v]

    # Only fetch tiles not already cached
    to_fetch = [(loc, d) for loc, d in work_items if loc["name"] not in vpp_cache]

    from imint.training.tile_fetch import _CDSE_SEMAPHORE

    def _fetch_one(item: tuple[dict, str]) -> tuple[str, list | None]:
        loc, _ = item
        _CDSE_SEMAPHORE.acquire()
        try:
            windows = _get_vpp_doy_windows(loc["bbox_3006"], num_growing_frames=3)
            if windows:
                _CDSE_SEMAPHORE.report_success()
            else:
                _CDSE_SEMAPHORE.report_failure()
            return loc["name"], windows
        except Exception:
            _CDSE_SEMAPHORE.report_failure()
            return loc["name"], None
        finally:
            _CDSE_SEMAPHORE.release()

    if to_fetch:
        # Use more threads than CDSE permits — the semaphore queues excess
        with ThreadPoolExecutor(max_workers=max(workers, 8)) as pool:
            futs = {pool.submit(_fetch_one, item): item for item in to_fetch}
            done = 0
            for f in as_completed(futs):
                name, windows = f.result()
                if windows:
                    vpp_cache[name] = windows
                done += 1
                if done % 200 == 0 or done == len(to_fetch):
                    print(f"  VPP {done}/{len(to_fetch)}  hits={len(vpp_cache)}"
                          f"  (CDSE permits={_CDSE_SEMAPHORE.permits})",
                          flush=True)

    # Save to disk
    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump({k: list(v) for k, v in vpp_cache.items()}, f)
        except Exception:
            pass

    miss = total - len(vpp_cache)
    print(f"  VPP done — {len(vpp_cache)} hits, {miss} misses\n")
    return vpp_cache


# ── Core Fetch ────────────────────────────────────────────────────────────────


def _fetch_frames_from_best_dates(
    bbox: dict,
    coords_wgs84: dict,
    best: dict,
    n_frames: int = 4,
) -> list:
    """Fetch spectral frames using pre-screened best dates (from openEO stage 1).

    Races CDSE + DES in parallel for each frame — first result wins.
    Skips STAC search, VPP lookup, and cloud screening entirely.

    Args:
        bbox: Tile bbox in EPSG:3006.
        coords_wgs84: Tile center in WGS84.
        best: {frame_idx: {date, cloud_frac}} from best_dates.json.
        n_frames: Number of frames.

    Returns:
        List of (spectral, date_str) tuples matching stack_frames input.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from imint.training.cdse_s2 import fetch_s2_scene
    from imint.training.tile_fetch import (
        _CDSE_SEMAPHORE, _DES_SEMAPHORE, PRITHVI_BANDS,
    )
    from imint.fetch import fetch_seasonal_image

    def _cdse_fetch(date_str):
        _CDSE_SEMAPHORE.acquire()
        try:
            result = fetch_s2_scene(
                bbox["west"], bbox["south"], bbox["east"], bbox["north"],
                date=date_str,
                size_px=TILE_SIZE_PX,
                cloud_threshold=1.0,
                haze_threshold=1.0,
                nodata_threshold=None,
            )
            _CDSE_SEMAPHORE.report_success()
            if result is not None:
                return result[0], date_str
        except Exception:
            _CDSE_SEMAPHORE.report_failure()
        finally:
            _CDSE_SEMAPHORE.release()
        return None, date_str

    def _des_fetch(date_str):
        _DES_SEMAPHORE.acquire()
        try:
            result = fetch_seasonal_image(
                date=date_str,
                coords=coords_wgs84,
                prithvi_bands=PRITHVI_BANDS,
                source="des",
            )
            _DES_SEMAPHORE.report_success()
            if result is not None:
                return result[0], date_str
        except Exception:
            _DES_SEMAPHORE.report_failure()
        finally:
            _DES_SEMAPHORE.release()
        return None, date_str

    results = []
    for fi in range(n_frames):
        info = best.get(str(fi))
        if not info or not info.get("date"):
            results.append((None, ""))
            continue

        date_str = info["date"]

        # Race CDSE + DES — first success wins
        with ThreadPoolExecutor(max_workers=2) as pool:
            futs = [
                pool.submit(_cdse_fetch, date_str),
                pool.submit(_des_fetch, date_str),
            ]
            got_result = False
            for f in as_completed(futs):
                spectral, d = f.result()
                if spectral is not None and not got_result:
                    results.append((spectral, d))
                    got_result = True
                    for pending in futs:
                        pending.cancel()
            if not got_result:
                results.append((None, date_str))

    return results


def fetch_tile(
    loc: dict,
    years: list[str],
    output_dir: str,
    cloud_max: float = 30.0,
    vpp_cache: dict | None = None,
    best_dates: dict | None = None,
) -> dict:
    """Fetch one tile: 4 seasonal frames + NMD + aux → .npz."""
    name = loc["name"]
    out_path = os.path.join(output_dir, f"{name}.npz")
    if os.path.exists(out_path):
        return {"name": name, "status": "skipped"}

    bbox = loc["bbox_3006"]
    coords = loc.get("coords_wgs84") or bbox_3006_to_wgs84(bbox)

    # Fast path: use pre-screened best dates from openEO stage 1
    tile_best = best_dates.get(name) if best_dates else None
    if tile_best:
        scene_results = _fetch_frames_from_best_dates(bbox, coords, tile_best, NUM_FRAMES)
    else:
        tile_years = [str(loc["year"])] if "year" in loc else years
        vpp_windows = vpp_cache.get(name) if vpp_cache is not None else ...
        scene_results = fetch_4frame_scenes(
            bbox, coords, tile_years,
            scene_cloud_max=cloud_max,
            vpp_windows=vpp_windows,
        )

    image, temporal_mask, doy, dates = stack_frames(scene_results, NUM_FRAMES)
    if int(temporal_mask.sum()) == 0:
        return {"name": name, "status": "failed", "reason": "no_scenes"}

    nmd_label = fetch_nmd_label_local(bbox)
    aux = fetch_aux_channels(bbox)

    save = {
        "spectral": image,
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

    # Background frame (2016 summer) for clearcut change detection
    bg_result = fetch_background_frame(bbox, coords)
    save.update(bg_result)

    np.savez_compressed(out_path, **save)
    has_bg = int(save.get("has_frame_2016", 0))
    return {"name": name, "status": "ok",
            "valid_frames": int(temporal_mask.sum()),
            "has_bg": has_bg}


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
            # Corrupt .npz — still recover bbox from filename
            m = re.search(r'tile_(\d+)_(\d+)', os.path.basename(path))
            if m:
                e, n = int(m.group(1)), int(m.group(2))
                half = TILE_SIZE_M // 2
                bbox = {"west": e - half, "south": n - half,
                        "east": e + half, "north": n + half}
                name = os.path.basename(path).replace(".npz", "")
                locs.append({
                    "name": name,
                    "source": "lulc",
                    "bbox_3006": bbox,
                    "coords_wgs84": bbox_3006_to_wgs84(bbox),
                    "_existing_path": path,
                    "_has_lpis": False,
                })
            else:
                skipped += 1
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
    vpp_cache: dict | None = None,
    best_dates: dict | None = None,
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

    # Fast path: use pre-screened best dates from openEO stage 1
    tile_best = best_dates.get(name) if best_dates else None
    if tile_best:
        scene_results = _fetch_frames_from_best_dates(bbox, coords, tile_best, NUM_FRAMES)
    else:
        vpp_windows = vpp_cache.get(loc["name"]) if vpp_cache is not None else ...
        scene_results = fetch_4frame_scenes(
            bbox, coords, fetch_years,
            scene_cloud_max=cloud_max,
            vpp_windows=vpp_windows,
        )
    image, temporal_mask, doy, dates = stack_frames(scene_results, NUM_FRAMES)

    if int(temporal_mask.sum()) == 0:
        return {"name": name, "status": "failed", "reason": "no_scenes"}

    # Load existing tile data (labels, aux, etc.)
    save = {}
    if existing_path and os.path.exists(existing_path):
        try:
            old = dict(np.load(existing_path, allow_pickle=True))
            # Spectral arrays are always overwritten by the new fetch.
            # Label arrays are always regenerated by build_labels.py — never
            # carry them forward, or a stale label (wrong schema, wrong axis
            # orientation, etc.) silently survives a refetch indefinitely.
            _DROP = frozenset((
                "image", "spectral", "temporal_mask", "doy",
                "dates", "multitemporal", "num_frames", "num_bands",
                "seasons_valid",
                # label keys — always rebuilt by build_labels, never propagated
                "label", "label_mask", "label_year",
            ))
            for k, v in old.items():
                if k not in _DROP:
                    save[k] = v
        except Exception:
            pass

    # Write new spectral + temporal metadata
    save["spectral"] = image
    save["temporal_mask"] = temporal_mask
    save["doy"] = doy
    save["dates"] = np.array(dates)
    save["multitemporal"] = np.int32(1)
    save["num_frames"] = np.int32(NUM_FRAMES)
    save["num_bands"] = np.int32(N_BANDS)

    np.savez_compressed(out_path, **save)

    # NOTE: Do NOT delete source tiles — they contain bbox info needed
    # for future re-fetches. Labels are built separately by build_labels.py.

    return {"name": name, "status": "ok",
            "valid_frames": int(temporal_mask.sum())}


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="Unified 4-frame tile fetcher")
    p.add_argument("--mode", required=True,
                   choices=["lulc", "rare-crops", "urban", "all", "refetch"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--years", nargs="+", default=["2018", "2019", "2022", "2023"])
    p.add_argument("--workers", type=int, default=1)
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
    p.add_argument("--from-json", default=None,
                   help="JSON file with tile locations [{name, bbox, year, source}, ...]")
    p.add_argument("--tile-size-px", type=int, default=256,
                   help="Tile resolution in pixels (256 or 512). Sets TILE_SIZE_PX and TILE_SIZE_M.")
    args = p.parse_args()
    random.seed(args.seed)

    # Override tile_fetch module constants for non-default tile sizes
    if args.tile_size_px != 256:
        import imint.training.tile_fetch as _tf
        _tf.TILE_SIZE_PX = args.tile_size_px
        _tf.TILE_SIZE_M = args.tile_size_px * 10
        print(f"  Tile size overridden: {_tf.TILE_SIZE_PX}px ({_tf.TILE_SIZE_M}m)")

    print(f"=== Unified 4-Frame Tile Fetcher ===")
    print(f"  Mode: {args.mode}  Years: {args.years}")
    print(f"  Frame 0: Autumn (Sep-Oct, year-1)")
    print(f"  Frames 1-3: VPP-guided growing season\n")

    use_refetch = args.mode == "refetch" or args.from_existing or args.from_json
    work: list[tuple[dict, str]] = []

    # JSON mode: read tile locations from a JSON file (bbox + year)
    if args.from_json:
        import json as _json
        os.makedirs(args.output_dir, exist_ok=True)
        with open(args.from_json) as f:
            tile_locs = _json.load(f)
        print(f"  Loaded {len(tile_locs)} tile locations from {args.from_json}")

        # Skip tiles already fetched in output dir
        existing = set(os.path.basename(f).replace(".npz", "")
                       for f in glob.glob(os.path.join(args.output_dir, "*.npz")))
        skipped = 0
        for t in tile_locs:
            if t["name"] in existing:
                skipped += 1
                continue
            # Support both JSON formats:
            #   gen_lulc format: bbox_3006 = {"west":…,"south":…,"east":…,"north":…}
            #   legacy format:   bbox = [west, south, east, north]
            raw_bbox = t.get("bbox_3006") or t.get("bbox")
            if raw_bbox is None:
                continue
            if isinstance(raw_bbox, dict):
                bbox = raw_bbox
            else:
                bbox = {"west": raw_bbox[0], "south": raw_bbox[1],
                        "east": raw_bbox[2], "north": raw_bbox[3]}
            loc = {
                "name": t["name"],
                "source": t.get("source", "lulc"),
                "bbox_3006": bbox,
                "coords_wgs84": t.get("coords_wgs84") or bbox_3006_to_wgs84(bbox),
                "_has_lpis": t.get("source") == "crop",
            }
            if t.get("year"):
                loc["year"] = t["year"]
            work.append((loc, args.output_dir))
        print(f"  Skipped {skipped} already-fetched, {len(work)} to fetch")

    # Refetch mode: read existing tile locations from .npz dirs
    elif use_refetch:
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

    # ── Load best_dates from openEO screening (if available) ───────────────
    best_dates_path = os.path.join(args.output_dir, "best_dates.json")
    best_dates_cache = None
    if os.path.exists(best_dates_path):
        with open(best_dates_path) as f:
            best_dates_cache = json.load(f)
        n_with_dates = sum(1 for loc, _ in work if loc["name"] in best_dates_cache)
        print(f"  best_dates.json: {n_with_dates}/{len(work)} tiles have pre-screened dates")

    # ── VPP prefetch (skip for tiles with best_dates) ─────────────────────
    if best_dates_cache:
        # Only prefetch VPP for tiles WITHOUT pre-screened dates
        vpp_work = [(loc, d) for loc, d in work if loc["name"] not in best_dates_cache]
        if vpp_work:
            vpp_cache = prefetch_vpp_batch(vpp_work, workers=args.workers)
        else:
            vpp_cache = {}
            print("  VPP prefetch skipped — all tiles have best_dates")
    else:
        vpp_cache = prefetch_vpp_batch(work, workers=args.workers)

    stats = {"ok": 0, "skipped": 0, "failed": 0}
    t0 = time.time()

    max_w = args.workers
    active_workers = max_w

    def _run(item):
        loc, d = item
        if use_refetch:
            return refetch_tile(loc, args.years, d, args.cloud_max,
                                vpp_cache=vpp_cache, best_dates=best_dates_cache)
        return fetch_tile(loc, args.years, d, args.cloud_max,
                          vpp_cache=vpp_cache, best_dates=best_dates_cache)

    CHUNK = max(max_w * 2, len(work) // 10)
    completed = 0
    with ThreadPoolExecutor(max_workers=max_w) as pool:
        for chunk_start in range(0, len(work), CHUNK):
            chunk = work[chunk_start:chunk_start + CHUNK]
            futs = {pool.submit(_run, w): w for w in chunk}
            for f in as_completed(futs):
                r = f.result()
                completed += 1
                if r:
                    stats[r.get("status", "failed")] = \
                        stats.get(r.get("status", "failed"), 0) + 1
                    if True:  # log every tile
                        elapsed = time.time() - t0
                        print(f"  [{completed}/{len(work)}] {r['name']}: {r['status']} "
                              f"| {completed/elapsed*3600:.0f}/h | workers={active_workers}",
                              flush=True)

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
