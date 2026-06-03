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
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as _FuturesTimeout,
    as_completed,
)
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imint.config.env import load_env

load_env()

from imint.training.tile_fetch import (
    N_BANDS,
    bbox_3006_to_wgs84,
    fetch_4frame_scenes,
    fetch_aux_channels,
    fetch_background_frame,
    fetch_nmd_label_local,
    stack_frames,
)
from imint.training.tile_config import TileConfig
from imint.training.sampler import generate_grid, grid_to_wgs84
from imint.training.scb_tatort import generate_scb_densification_regions

NUM_FRAMES = 4  # 1 autumn (year-1) + 3 VPP-guided growing season

# Rare crop grödkoder
SJV_OLJEVAXTER = {85, 86, 87, 88, 90, 91, 92}
SJV_HAVRE = {5}


# ── Location Generators ──────────────────────────────────────────────────────


def gen_lulc(tile: TileConfig, spacing_m: int = 2500, max_tiles: int | None = None) -> list[dict]:
    """LULC grid locations across Sweden."""
    print("  Generating LULC grid...")
    cells = grid_to_wgs84(generate_grid(
        spacing_m=spacing_m, patch_size_m=tile.size_m, land_filter=True,
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
    tile: TileConfig,
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
            bbox = tile.bbox_from_center(cx, cy)
            locs.append({
                "name": f"crop_{label}_{int(cx)}_{int(cy)}",
                "source": "crop",
                "bbox_3006": bbox,
                "coords_wgs84": bbox_3006_to_wgs84(bbox),
                "year": int(row["_year"]),
            })
    return locs


def gen_urban(
    tile: TileConfig,
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
        patch_size_m=tile.size_m,
    )
    print(f"  {len(regions)} tätorter (pop ≥ {min_pop})")

    locs = []
    half = tile.half_m
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
                bbox = tile.bbox_from_center(cx, cy)
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
    tile: TileConfig,
    n_frames: int = 4,
    sources: tuple[str, ...] = ("cdse", "des"),
) -> list:
    """Fetch spectral frames using pre-screened best dates (from openEO stage 1).

    Races selected providers in parallel for each frame — first result wins.
    Skips STAC search, VPP lookup, and cloud screening entirely.

    Args:
        bbox: Tile bbox in EPSG:3006.
        coords_wgs84: Tile center in WGS84.
        best: {frame_idx: {date, cloud_frac}} from best_dates.json.
        n_frames: Number of frames.
        sources: Which backends to race. Subset of ("cdse", "des").
            Pass ("des",) to skip CDSE during a CDSE outage.

    Returns:
        List of (spectral, date_str) tuples matching stack_frames input.
    """
    from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as _FuturesTimeout,
    as_completed,
)
    from imint.training.cdse_s2 import fetch_s2_scene
    from imint.training.tile_fetch import (
        _CDSE_SEMAPHORE, _DES_SEMAPHORE, _CDSE_OPENEO_SEMAPHORE,
        PRITHVI_BANDS,
    )
    from imint.fetch import fetch_seasonal_image

    def _cdse_fetch(date_str):
        _CDSE_SEMAPHORE.acquire()
        try:
            result = fetch_s2_scene(
                bbox["west"], bbox["south"], bbox["east"], bbox["north"],
                date=date_str,
                size_px=tile.size_px,
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

    def _cdse_openeo_fetch(date_str):
        # CDSE openEO (openeo.dataspace.copernicus.eu) uses a separate
        # monthly-credit pool from SH Process PUs.
        #
        # Session-scoped 402-guard (same pattern as the unified flow
        # in fetch_spectral._fetch_via_openeo): on first PaymentRequired we
        # mark the source dead for this process; subsequent attempts
        # short-circuit without any HTTP roundtrip. Cleared by next
        # pod restart, so credit-reset / package-purchase auto-recovers.
        from imint.training.openeo_tile_graph import (
            is_source_dead, mark_source_dead, _is_payment_required_error,
        )
        if is_source_dead("cdse-openeo"):
            return None, date_str
        _CDSE_OPENEO_SEMAPHORE.acquire()
        try:
            result = fetch_seasonal_image(
                date=date_str,
                coords=coords_wgs84,
                prithvi_bands=PRITHVI_BANDS,
                source="copernicus",
            )
            _CDSE_OPENEO_SEMAPHORE.report_success()
            if result is not None:
                return result[0], date_str
        except Exception as exc:
            if _is_payment_required_error(exc):
                mark_source_dead(
                    "cdse-openeo",
                    f"402 in _cdse_openeo_fetch: {str(exc)[:160]}",
                )
            _CDSE_OPENEO_SEMAPHORE.report_failure()
        finally:
            _CDSE_OPENEO_SEMAPHORE.release()
        return None, date_str

    results = []
    for fi in range(n_frames):
        info = best.get(str(fi))
        if not info or not info.get("date"):
            results.append((None, ""))
            continue

        date_str = info["date"]

        # Race selected providers — first success wins
        submit_map = {}
        if "cdse" in sources:
            submit_map["cdse"] = _cdse_fetch
        if "des" in sources:
            submit_map["des"] = _des_fetch
        if "cdse-openeo" in sources:
            submit_map["cdse-openeo"] = _cdse_openeo_fetch
        if not submit_map:
            results.append((None, date_str))
            continue

        # Race selected providers — first success wins. Hard 180 s
        # timeout per frame and break on first success so one stalled
        # provider (e.g. DES openEO process-graph hang) cannot block
        # tile completion. shutdown(wait=False) skips waiting for the
        # remaining (possibly hung) threads; they die naturally when
        # the socket layer eventually times out. cancel_futures=True
        # cancels queued-but-not-started futures.
        pool = ThreadPoolExecutor(max_workers=max(len(submit_map), 1))
        futs = [pool.submit(fn, date_str) for fn in submit_map.values()]
        got_result = False
        try:
            for f in as_completed(futs, timeout=180):
                spectral, d = f.result()
                if spectral is not None:
                    results.append((spectral, d))
                    got_result = True
                    break
        except _FuturesTimeout:
            pass  # 3-min hard cap reached — treat as no-result for this frame
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        if not got_result:
            results.append((None, date_str))

    return results


def fetch_tile(
    loc: dict,
    years: list[str],
    output_dir: str,
    tile: TileConfig,
    cloud_max: float = 30.0,
    vpp_cache: dict | None = None,
    best_dates: dict | None = None,
    sources: tuple[str, ...] = ("cdse", "des"),
) -> dict:
    """Fetch one tile: 4 seasonal frames + NMD + aux → .npz."""
    name = loc["name"]
    out_path = os.path.join(output_dir, f"{name}.npz")
    if os.path.exists(out_path):
        return {"name": name, "status": "skipped"}

    # Always normalize bbox to the current tile size. Manifest/loc bboxes
    # may carry stale 256-era extents; trust only the center.
    raw_bbox = loc["bbox_3006"]
    cx = (raw_bbox["west"] + raw_bbox["east"]) // 2
    cy = (raw_bbox["south"] + raw_bbox["north"]) // 2
    bbox = tile.bbox_from_center(cx, cy)
    coords = bbox_3006_to_wgs84(bbox)
    tile.assert_bbox_matches(bbox)

    # Fast path: use pre-screened best dates from openEO stage 1
    tile_best = best_dates.get(name) if best_dates else None
    if tile_best:
        scene_results = _fetch_frames_from_best_dates(
            bbox, coords, tile_best, tile, NUM_FRAMES, sources=sources,
        )
    else:
        tile_years = [str(loc["year"])] if "year" in loc else years
        vpp_windows = vpp_cache.get(name) if vpp_cache is not None else ...
        scene_results = fetch_4frame_scenes(
            bbox, coords, tile_years, tile,
            scene_cloud_max=cloud_max,
            vpp_windows=vpp_windows,
            sources=sources,
        )

    image, temporal_mask, doy, dates = stack_frames(scene_results, NUM_FRAMES, tile)
    if int(temporal_mask.sum()) == 0:
        return {"name": name, "status": "failed", "reason": "no_scenes"}

    nmd_label = fetch_nmd_label_local(bbox, tile)
    aux = fetch_aux_channels(bbox, tile)

    save = {
        "spectral": image,
        "temporal_mask": temporal_mask,
        "doy": doy,
        "dates": np.array(dates),
        "multitemporal": np.int32(1),
        "num_frames": np.int32(NUM_FRAMES),
        "num_bands": np.int32(N_BANDS),
        "bbox_3006": np.array(
            [bbox["west"], bbox["south"], bbox["east"], bbox["north"]],
            dtype=np.int32,
        ),
        "easting": np.int32(cx),
        "northing": np.int32(cy),
        "tile_size_px": np.int32(tile.size_px),
        "source": loc["source"],
    }
    if nmd_label is not None:
        save["label"] = nmd_label
    save.update(aux)

    # Background frame (2016 summer) for clearcut change detection
    bg_result = fetch_background_frame(bbox, tile)
    if bg_result is not None:
        save.update(bg_result)

    np.savez_compressed(out_path, **save)
    has_bg = int(save.get("has_frame_2016", 0))
    return {"name": name, "status": "ok",
            "valid_frames": int(temporal_mask.sum()),
            "has_bg": has_bg}


def gen_from_existing(
    tiles_dir: str, tile: TileConfig, max_tiles: int | None = None,
) -> list[dict]:
    """Read tile locations from existing .npz files on disk.

    Always normalizes bbox to the current tile size via resolve_tile_bbox,
    so stale bboxes from earlier fetches (e.g. 256-era 2560m) are silently
    upgraded to the caller's current TileConfig extent.
    """
    import glob
    from imint.training.tile_bbox import resolve_tile_bbox

    tiles = sorted(glob.glob(os.path.join(tiles_dir, "*.npz")))
    print(f"  Found {len(tiles)} existing tiles in {tiles_dir}")

    locs = []
    skipped = 0
    for path in tiles:
        name = os.path.basename(path).replace(".npz", "")
        try:
            data = np.load(path, allow_pickle=True)
            bbox = resolve_tile_bbox(name=name, tile=tile, npz_data=data)
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
                dates = data["dates"]
                for d in dates:
                    d_str = str(d)
                    if d_str and len(d_str) >= 4:
                        tile_year = int(d_str[:4])
                        break

            has_lpis = "label_mask" in data or "lpis_year" in data

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
            # Corrupt .npz — still try filename-based recovery
            bbox = resolve_tile_bbox(name=name, tile=tile, npz_data=None)
            if bbox is None:
                skipped += 1
                continue
            locs.append({
                "name": name,
                "source": "lulc",
                "bbox_3006": bbox,
                "coords_wgs84": bbox_3006_to_wgs84(bbox),
                "_existing_path": path,
                "_has_lpis": False,
            })
            continue

    if skipped > 0:
        print(f"  Skipped {skipped} tiles (no bbox recoverable)")

    if max_tiles and len(locs) > max_tiles:
        random.shuffle(locs)
        locs = locs[:max_tiles]

    print(f"  Using {len(locs)} tile locations")
    return locs


def repair_to_canonical_layout(
    loc: dict,
    output_dir: str,
    tile: TileConfig,
    *,
    cloud_max: float = 30.0,
    max_aoi_cloud: float = 0.10,
    sources: tuple[str, ...] = ("cdse", "des"),
    cap_doy: int = 244,
) -> dict:
    """Repair a tile to canonical 4-frame layout via per-slot classify + fill-gap.

    Canonical layout (calendar-monotonic across all 4 slots):
        slot 0: autumn year-1            (DOY 228-300, hardcoded Aug 15..Oct 31)
        slot 1: spring year              (VPP window 0, capped at cap_doy)
        slot 2: early-summer year        (VPP window 1, capped at cap_doy)
        slot 3: late-summer year         (VPP window 2, capped at cap_doy)

    Algorithm:
      1. Read existing dates/doy/tmask + VPP arrays from .npz
      2. Compute capped VPP windows locally (no API call)
      3. Classify each valid existing frame into a slot. Drop frames
         with DOY > cap_doy in tile_year (the bug) or that match no slot.
      4. Fetch only the missing slots.
      5. Build new spectral cube in canonical slot order.
      6. Verify calendar-date monotonicity. Strict-fail otherwise.
      7. Atomic save (preserving all aux/label fields).

    Per-tile cost: 1 ``fetch_spectral`` call per missing slot, retrying
    through the slot's ranked candidate list until one passes the AOI
    cloud gate. Returns ``status=skipped`` if all 4 slots were already
    filled correctly.

    Year derivation hierarchy (loc agnostic — read straight from .npz):
        tessera_year → lpis_year → year → first valid date.year
    `tessera_year` is the verified primary (cross-checked against
    LPIS-parcel counts for all 20 audit-flagged tiles in this batch).
    """
    from imint.training.tile_fetch import (
        doy_to_date_range, N_BANDS,
    )
    from imint.training.vpp_windows import compute_growing_season_windows

    name = loc["name"]
    existing_path = loc.get("_existing_path")
    out_path = os.path.join(output_dir, f"{name}.npz")

    if not existing_path or not os.path.exists(existing_path):
        return {"name": name, "status": "error", "reason": "no_existing_tile"}

    try:
        old = dict(np.load(existing_path, allow_pickle=True))
    except Exception as e:
        return {"name": name, "status": "error", "reason": f"npz_load: {e!s}"[:200]}

    # Year derivation: tessera_year (LPIS-cross-checked) → lpis_year → year → dates
    tile_year: int | None = None
    for key in ("tessera_year", "lpis_year", "year"):
        if key in old:
            try:
                tile_year = int(old[key])
                break
            except Exception:
                pass
    if tile_year is None:
        # Last-resort: parse year from first valid date
        for d in old.get("dates", []):
            s = str(d)
            if s and len(s) >= 4:
                try:
                    tile_year = int(s[:4])
                    break
                except ValueError:
                    pass
    if tile_year is None:
        return {"name": name, "status": "error", "reason": "no_year_field"}

    # Normalise bbox
    raw_bbox = loc["bbox_3006"]
    cx = (raw_bbox["west"] + raw_bbox["east"]) // 2
    cy = (raw_bbox["south"] + raw_bbox["north"]) // 2
    bbox = tile.bbox_from_center(cx, cy)
    coords = bbox_3006_to_wgs84(bbox)
    tile.assert_bbox_matches(bbox)

    # Validate required fields
    for req in ("spectral", "doy", "dates", "temporal_mask", "vpp_sosd", "vpp_eosd"):
        if req not in old:
            return {"name": name, "status": "error", "reason": f"missing_{req}"}

    old_image = old["spectral"]                   # (NUM_FRAMES*N_BANDS, H, W)
    old_doys = [int(x) for x in old["doy"]]
    old_dates = [str(x) for x in old["dates"]]
    old_tmask = [int(x) for x in old["temporal_mask"]]

    # Compute VPP windows locally (no CDSE API call)
    vpp_windows = compute_growing_season_windows(
        old["vpp_sosd"], old["vpp_eosd"], num_frames=3,
    )
    # Cap against cap_doy (defends against fallback windows which aren't capped)
    capped = [(s, min(e, cap_doy)) for s, e in vpp_windows]
    capped = [(s, e) for s, e in capped if s <= e]
    if len(capped) < 3:
        return {"name": name, "status": "error",
                "reason": f"vpp_windows_collapsed: orig={vpp_windows}"}

    # Define 4 canonical slots
    # Autumn (slot 0): hardcoded Aug 15..Oct 31 of year-1 per fetch_4frame_scenes
    AUTUMN_DOY_MIN, AUTUMN_DOY_MAX = 228, 304
    slot_defs = [
        ("autumn_y_minus_1", tile_year - 1, AUTUMN_DOY_MIN, AUTUMN_DOY_MAX),
        ("spring",           tile_year,     capped[0][0], capped[0][1]),
        ("early_summer",     tile_year,     capped[1][0], capped[1][1]),
        ("late_summer",      tile_year,     capped[2][0], capped[2][1]),
    ]

    # Classify existing frames into slots. slot_idx -> (src_idx_in_old, year, doy)
    slot_assignments: dict[int, tuple[int, int, int]] = {}
    dropped: list[tuple[int, int, str]] = []  # (src_idx, doy, reason)
    for src_idx in range(4):
        if old_tmask[src_idx] == 0:
            continue
        # Silent-corruption guard: a previous repair_to_canonical_layout
        # run on the pre-refactor race-pool may have accepted an all-zeros
        # scene as if valid (tmask=1 but spectral cube slice is empty).
        # The unified ``fetch_spectral`` flow returns ``None`` for empty
        # scenes, so new runs cannot regress; this guard cleans up legacy
        # tiles touched by the old code path.
        frame_slice = old_image[src_idx * N_BANDS:(src_idx + 1) * N_BANDS]
        if not np.any(frame_slice):
            dropped.append((src_idx, old_doys[src_idx], "all_zeros_spectral"))
            continue
        doy = old_doys[src_idx]
        date_s = old_dates[src_idx]
        try:
            existing_year = int(date_s[:4])
        except Exception:
            dropped.append((src_idx, doy, "bad_date"))
            continue
        # Bug detection: late-autumn DOY in tile_year
        if existing_year == tile_year and doy > cap_doy:
            dropped.append((src_idx, doy, f"doy_gt_{cap_doy}"))
            continue
        # First-fit slot assignment
        assigned = False
        for slot_idx, (_, slot_year, slot_min, slot_max) in enumerate(slot_defs):
            if (existing_year == slot_year
                    and slot_min <= doy <= slot_max
                    and slot_idx not in slot_assignments):
                slot_assignments[slot_idx] = (src_idx, existing_year, doy)
                assigned = True
                break
        if not assigned:
            dropped.append((src_idx, doy, "no_slot_match"))

    missing_slots = [i for i in range(4) if i not in slot_assignments]

    if not missing_slots and not dropped:
        return {"name": name, "status": "skipped", "reason": "all_slots_filled"}

    from imint.training.optimal_fetch import rank_stac_era5_candidates
    growing_dates: list[str] | None = None
    autumn_dates: list[str] | None = None
    # Ranked (date, granule_cc, overpass_cloud) tuples per window — fed to
    # the lazy per-slot SCL verification in the tile-graph block below.
    ranked_by_window: dict[str, list[tuple[str, float, float]]] = {
        "autumn": [], "growing": [],
    }
    needs_growing = any(i in missing_slots for i in (1, 2, 3))
    needs_autumn = (0 in missing_slots)

    # Window bounds — computed once.
    gs_min = min(slot_defs[1][2], slot_defs[2][2], slot_defs[3][2])
    gs_max = max(slot_defs[1][3], slot_defs[2][3], slot_defs[3][3])
    gs_ds, gs_de = doy_to_date_range(tile_year, gs_min, gs_max)
    au_ds, au_de = doy_to_date_range(
        tile_year - 1, slot_defs[0][2], slot_defs[0][3],
    )

    # Date selection — CHEAP-FIRST, LAZY-EXPENSIVE chain (minimal calls):
    #
    #   1. DES STAC (1 free HTTP)      → real S2 passes + granule cloud
    #   2. ERA5 overpass (1 free HTTP) → drop overcast-at-10:30, rank by cloud
    #   3. SCL verify (openEO, LAZY)   → in the tile-graph block below, the
    #      ranked candidates are SCL-checked best-first and we STOP at the
    #      first AOI-clean date — so we pay for SCL only on the most-
    #      promising date(s) per slot, not the whole window.
    #
    # This replaces the whole-window SCL stack (which screened ~80 dates'
    # SCL even though we use 4) and the season-aggregate (>300 s on CDSE).
    # STAC + ERA5 are free; SCL is the expensive one, now called lazily.
    # SCL verification runs on DES (pixel path, bug-free) to keep CDSE's
    # single connection free for the spectral tile-graph.
    if needs_growing:
        try:
            ranked_by_window["growing"] = rank_stac_era5_candidates(
                coords, gs_ds, gs_de, overpass_cloud_max=50.0,
            )
        except Exception as exc:
            print(f"    [rank] growing candidates failed for {name}: "
                  f"{type(exc).__name__}: {str(exc)[:140]}", flush=True)
    if needs_autumn:
        try:
            # Autumn cloudier — looser overpass ceiling.
            ranked_by_window["autumn"] = rank_stac_era5_candidates(
                coords, au_ds, au_de, overpass_cloud_max=65.0,
            )
        except Exception as exc:
            print(f"    [rank] autumn candidates failed for {name}: "
                  f"{type(exc).__name__}: {str(exc)[:140]}", flush=True)
    # Date-only lists for the per-slot fallback's prefetched_dates= param.
    autumn_dates = [d for d, _ in ranked_by_window["autumn"]]
    growing_dates = [d for d, _ in ranked_by_window["growing"]]

    # Fetch each missing slot
    fetched: dict[int, tuple[np.ndarray, str]] = {}
    failed_slots: list[int] = []

    # ── Unified per-slot spectral fetch ──
    # ONE flow, ONE backend per tile. The lazy STAC+ERA5 chain (computed
    # above as ``ranked_by_window``) ranks candidate dates per slot; we
    # iterate best-first, hand each to :func:`fetch_spectral` (which
    # encapsulates the backend's verify+fetch), and stop at the first
    # AOI-clean scene. No backend race. No cross-backend fallback within
    # a slot. No separate "tile-graph vs race-pool" path. If the chosen
    # backend exhausts the slot's ranked candidates, the slot fails —
    # surfaced cleanly rather than masked by a silent fall-through to a
    # second code path with different semantics (which was the previous
    # design's failure mode — see commit history before this refactor).
    #
    # Backend selection happens once per tile: first ``--sources`` token
    # that's both supported and not marked dead this session. Mid-tile
    # health re-check skips slots cleanly if the backend dies (e.g.
    # cdse-openeo 402 PaymentRequired) without spamming doomed calls.
    from imint.training.fetch_spectral import fetch_spectral, SUPPORTED_BACKENDS
    from imint.training.openeo_tile_graph import is_source_dead
    from imint.training.optimal_fetch import era5_to_scl_gate

    unknown_sources = [s for s in sources if s not in SUPPORTED_BACKENDS]
    primary_backend: str | None = None
    for src in sources:
        if src in SUPPORTED_BACKENDS and not is_source_dead(src):
            primary_backend = src
            break

    if primary_backend is None:
        dead = [s for s in sources
                if s in SUPPORTED_BACKENDS and is_source_dead(s)]
        return {
            "name": name, "status": "failed",
            "reason": (f"no_healthy_backend (sources={list(sources)}, "
                       f"dead={dead}, unknown={unknown_sources})"),
            "kept_slots": sorted(slot_assignments.keys()),
            "fetched_slots": [],
            "dropped_count": len(dropped),
        }

    for sidx in missing_slots:
        _slot_name, syear, smin, smax = slot_defs[sidx]
        _ds, _de = doy_to_date_range(syear, smin, smax)
        is_autumn = (sidx == 0)
        ranked = ranked_by_window["autumn" if is_autumn else "growing"]
        # Per-candidate ERA5 overpass cloud drives the adaptive SCL gate
        # (see era5_to_scl_gate). Tuple shape from the ranker:
        # (date_str, era5_overpass_pct).
        candidates = [(d, oc) for d, oc in ranked if _ds <= d <= _de]

        # Pre-2018 catalogue gap: explorer.digitalearth.se's STAC starts
        # in 2018, so DES rank returns no candidates for year=2018 tiles'
        # slot 0 (autumn 2017) even though the data exists. Fall back to
        # earth-search (AWS, full S2 history) and re-rank with real ERA5
        # values — same ranker logic, different STAC source. No synthetic
        # date generation, no ERA5=50 placeholder.
        if not candidates and _de < "2018-01-01":
            try:
                era5_ceiling = 65.0 if is_autumn else 50.0
                aws_ranked = rank_stac_era5_candidates(
                    coords, _ds, _de,
                    overpass_cloud_max=era5_ceiling,
                    stac_backend="earth-search",
                )
                candidates = [(d, oc) for d, oc in aws_ranked]
            except Exception as exc:
                print(f"    [rank:earth-search] {name} slot {sidx} "
                      f"({_ds}..{_de}): {type(exc).__name__}: "
                      f"{str(exc)[:140]}", flush=True)

        # Mid-tile health re-check: if the chosen backend was marked
        # dead while a previous slot was processing (e.g. cdse-openeo
        # hit 402 PaymentRequired), bail this slot cleanly instead of
        # firing doomed calls.
        if is_source_dead(primary_backend):
            failed_slots.append(sidx)
            continue

        for cand_date, era5_oc in candidates:
            # ERA5-adaptive SCL gate: tighter when ERA5 says clear,
            # looser when ERA5 says cloudy. Replaces the previous
            # static max_aoi_cloud * (3 / 1.5) ceiling — that constant
            # was a poor approximation of "trust ERA5 in proportion
            # to how confident it is in cleanliness".
            ceiling = era5_to_scl_gate(era5_oc, is_autumn=is_autumn)
            scene = fetch_spectral(
                bbox, coords, cand_date,
                backend=primary_backend,
                size_px=tile.size_px,
                cloud_threshold=ceiling,
            )
            if scene is None:
                continue
            # Defensive resize (centre-place, no resample). The dispatcher
            # already returns ``None`` for zero/empty scenes, so we only
            # land here with a real array.
            if scene.shape[1] != tile.size_px or scene.shape[2] != tile.size_px:
                padded = np.zeros((N_BANDS, tile.size_px, tile.size_px),
                                  dtype=np.float32)
                h = min(scene.shape[1], tile.size_px)
                w = min(scene.shape[2], tile.size_px)
                padded[:, :h, :w] = scene[:, :h, :w]
                scene = padded
            fetched[sidx] = (scene, cand_date)
            break
        else:
            failed_slots.append(sidx)

    if failed_slots:
        return {
            "name": name, "status": "failed",
            "reason": f"fetch_failed_for_slots: {failed_slots}",
            "kept_slots": sorted(slot_assignments.keys()),
            "fetched_slots": sorted(fetched.keys()),
            "dropped_count": len(dropped),
        }

    # Build canonical (4*N_BANDS, H, W) cube
    H = W = tile.size_px
    new_image = np.zeros((4 * N_BANDS, H, W), dtype=np.float32)
    new_dates = ["", "", "", ""]
    new_doys = [0, 0, 0, 0]
    new_tmask = [0, 0, 0, 0]

    for slot_idx in range(4):
        dst = slot_idx * N_BANDS
        if slot_idx in slot_assignments:
            src_idx, _, _ = slot_assignments[slot_idx]
            src = src_idx * N_BANDS
            new_image[dst:dst + N_BANDS] = old_image[src:src + N_BANDS]
            new_dates[slot_idx] = old_dates[src_idx]
            new_doys[slot_idx] = old_doys[src_idx]
            new_tmask[slot_idx] = 1
        elif slot_idx in fetched:
            scene, date_str = fetched[slot_idx]
            new_image[dst:dst + N_BANDS] = scene
            new_dates[slot_idx] = date_str
            new_doys[slot_idx] = datetime.strptime(date_str, "%Y-%m-%d").timetuple().tm_yday
            new_tmask[slot_idx] = 1

    # Calendar-date monotonicity (autumn-y-1 < spring < early-summer < late-summer)
    cal_dates = []
    for slot_idx in range(4):
        if not new_dates[slot_idx]:
            return {"name": name, "status": "failed",
                    "reason": f"slot_{slot_idx}_empty_after_build"}
        cal_dates.append(datetime.strptime(new_dates[slot_idx], "%Y-%m-%d"))
    if not (cal_dates[0] < cal_dates[1] < cal_dates[2] < cal_dates[3]):
        return {
            "name": name, "status": "failed",
            "reason": f"non_monotonic_calendar",
            "dates_after": new_dates,
        }

    # Save: preserve all aux/label fields, replace only the 4 mutated keys
    save = {k: v for k, v in old.items() if k not in (
        "spectral", "dates", "doy", "temporal_mask",
    )}
    save["spectral"] = new_image
    save["dates"] = np.array(new_dates)
    save["doy"] = np.array(new_doys, dtype=np.int32)
    save["temporal_mask"] = np.array(new_tmask, dtype=np.uint8)

    # D3 design: refetch does NOT touch per-frame aux fields (b08/rededge).
    # Aux fields carry their own *_dates marker; the enrich-* scripts compare
    # b08_dates[i] / rededge_dates[i] vs dates[i] on every run and re-fetch
    # any slot that drifted. Refetch only needs to write the new `dates`
    # array — drift detection is the enrich pass's responsibility.

    # Atomic write — tmp must end in .npz, else np.savez_compressed auto-
    # appends ".npz" to the path, producing FOO.npz.tmp.npz and breaking
    # the subsequent os.replace("FOO.npz.tmp", "FOO.npz") rename.
    tmp_path = out_path[:-4] + ".tmp.npz"  # FOO.npz → FOO.tmp.npz
    np.savez_compressed(tmp_path, **save)
    os.replace(tmp_path, out_path)

    return {
        "name": name, "status": "ok",
        "kept_slots": sorted(slot_assignments.keys()),
        "fetched_slots": sorted(fetched.keys()),
        "dropped_count": len(dropped),
        "doys_after": new_doys,
        "dates_after": new_dates,
    }


def refetch_tile(
    loc: dict,
    years: list[str],
    output_dir: str,
    tile: TileConfig,
    cloud_max: float = 30.0,
    max_aoi_cloud: float = 0.10,
    vpp_cache: dict | None = None,
    best_dates: dict | None = None,
    sources: tuple[str, ...] = ("cdse", "des"),
    force: bool = False,
) -> dict:
    """Re-fetch spectral data for an existing tile, keep all other fields.

    Args:
        force: When True, re-fetch even if the tile already has
            multitemporal=1 and num_frames=4. Required for upgrading
            tiles fetched with a buggy VPP-window logic (see PR #15
            and IM-017: ~58% of tiles ended up with a growing-season
            frame in late-autumn DOY > 244 before that fix landed).
    """
    name = loc["name"]
    existing_path = loc.get("_existing_path")
    out_path = os.path.join(output_dir, f"{name}.npz")

    # Skip if already re-fetched (check for multitemporal flag) —
    # unless force=True, in which case re-fetch regardless.
    if not force and os.path.exists(out_path):
        try:
            d = np.load(out_path, allow_pickle=True)
            if d.get("multitemporal", 0) == 1 and d.get("num_frames", 0) == 4:
                return {"name": name, "status": "skipped"}
        except Exception:
            pass

    # Always normalize bbox to the current TileConfig extent — never
    # trust the incoming loc bbox blindly; it may carry a stale 256-era
    # extent. Trust only the center.
    raw_bbox = loc["bbox_3006"]
    cx = (raw_bbox["west"] + raw_bbox["east"]) // 2
    cy = (raw_bbox["south"] + raw_bbox["north"]) // 2
    bbox = tile.bbox_from_center(cx, cy)
    coords = bbox_3006_to_wgs84(bbox)
    tile.assert_bbox_matches(bbox)

    # Determine fetch years based on tile type
    # Crop tiles: strict year match only (LPIS labels are year-specific)
    # Forest/water tiles: tile year first, other years as fallback
    tile_year = loc.get("year")
    has_crop_labels = loc.get("source") == "crop" or loc.get("_has_lpis", False)

    if tile_year and has_crop_labels:
        fetch_years = [str(tile_year)]
    elif tile_year:
        fetch_years = [str(tile_year)] + [y for y in years if y != str(tile_year)]
    else:
        fetch_years = years

    # Fast path: pre-screened best dates from openEO stage 1
    tile_best = best_dates.get(name) if best_dates else None
    if tile_best:
        scene_results = _fetch_frames_from_best_dates(
            bbox, coords, tile_best, tile, NUM_FRAMES, sources=sources,
        )
    else:
        vpp_windows = vpp_cache.get(loc["name"]) if vpp_cache is not None else ...
        scene_results = fetch_4frame_scenes(
            bbox, coords, fetch_years, tile,
            scene_cloud_max=cloud_max,
            max_aoi_cloud=max_aoi_cloud,
            vpp_windows=vpp_windows,
            sources=sources,
        )
    image, temporal_mask, doy, dates = stack_frames(scene_results, NUM_FRAMES, tile)

    if int(temporal_mask.sum()) == 0:
        return {"name": name, "status": "failed", "reason": "no_scenes"}

    save = {}
    if existing_path and os.path.exists(existing_path):
        try:
            old = dict(np.load(existing_path, allow_pickle=True))
            # Spectral is always overwritten; labels always rebuilt.
            _DROP = frozenset((
                "image", "spectral", "temporal_mask", "doy",
                "dates", "multitemporal", "num_frames", "num_bands",
                "seasons_valid",
                "label", "label_mask", "label_year",
            ))
            for k, v in old.items():
                if k not in _DROP:
                    save[k] = v
        except Exception:
            pass

    save["spectral"] = image
    save["temporal_mask"] = temporal_mask
    save["doy"] = doy
    save["dates"] = np.array(dates)
    save["multitemporal"] = np.int32(1)
    save["num_frames"] = np.int32(NUM_FRAMES)
    save["num_bands"] = np.int32(N_BANDS)

    # Persist bbox that matches the raster we just wrote.
    save["bbox_3006"] = np.array(
        [bbox["west"], bbox["south"], bbox["east"], bbox["north"]],
        dtype=np.int32,
    )
    save["easting"] = np.int32(cx)
    save["northing"] = np.int32(cy)
    save["tile_size_px"] = np.int32(tile.size_px)
    save["source"] = loc.get("source", "lulc")

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
                   help="Tile resolution in pixels (256, 512, 1024, …). "
                        "Sets the runtime TileConfig.size_px.")
    p.add_argument("--fetch-sources", default="cdse,des",
                   help="Comma-separated list of S2 backends to race. "
                        "Valid: cdse (SH Process, PU-billed), des (Digital "
                        "Earth Sweden openEO), cdse-openeo (CDSE openEO, "
                        "uses the separate 10k-credits/mo pool). Use "
                        "--fetch-sources=des,cdse-openeo to skip SH Process "
                        "during a CDSE PU outage.")
    args = p.parse_args()
    random.seed(args.seed)

    # Parse fetch sources
    sources_tuple = tuple(
        s.strip().lower() for s in args.fetch_sources.split(",") if s.strip()
    )
    valid = {"cdse", "des", "cdse-openeo"}
    unknown = set(sources_tuple) - valid
    if unknown:
        p.error(f"Unknown fetch source(s): {sorted(unknown)}. Valid: {sorted(valid)}")
    if not sources_tuple:
        p.error("At least one fetch source required")
    print(f"  Fetch sources: {', '.join(sources_tuple)}")

    # Single source of truth for tile geometry — threaded explicitly
    # through every function that needs it. No monkey-patching, no
    # module mutation.
    tile = TileConfig(size_px=args.tile_size_px)
    print(f"=== Unified 4-Frame Tile Fetcher ===")
    print(f"  TileConfig: {tile.size_px}px × {tile.size_m}m")
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
            # Support both JSON formats (gen_lulc dict or legacy list).
            # TileConfig.bbox_from_center normalizes to the current size —
            # manifests produced for another tile size are fine, we trust
            # only the center.
            raw_bbox = t.get("bbox_3006") or t.get("bbox")
            if raw_bbox is None:
                continue
            if isinstance(raw_bbox, dict):
                cx = (raw_bbox["west"] + raw_bbox["east"]) // 2
                cy = (raw_bbox["south"] + raw_bbox["north"]) // 2
            else:
                cx = (raw_bbox[0] + raw_bbox[2]) // 2
                cy = (raw_bbox[1] + raw_bbox[3]) // 2
            bbox = tile.bbox_from_center(cx, cy)
            loc = {
                "name": t["name"],
                "source": t.get("source", "lulc"),
                "bbox_3006": bbox,
                "coords_wgs84": bbox_3006_to_wgs84(bbox),
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
            for loc in gen_from_existing(d, tile, args.max_tiles):
                work.append((loc, args.output_dir))
    else:
        os.makedirs(args.output_dir, exist_ok=True)

        if args.mode in ("lulc", "all"):
            for loc in gen_lulc(tile, args.grid_spacing, args.max_tiles):
                work.append((loc, args.output_dir))

        if args.mode in ("rare-crops", "all"):
            for loc in gen_rare_crops(tile, args.lpis_dir,
                                      args.n_oljevaxter, args.n_havre):
                work.append((loc, args.output_dir))

        if args.mode in ("urban", "all"):
            for loc in gen_urban(tile,
                                 max_tiles=args.max_tiles or 500,
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
            return refetch_tile(loc, args.years, d, tile,
                                cloud_max=args.cloud_max,
                                vpp_cache=vpp_cache, best_dates=best_dates_cache,
                                sources=sources_tuple)
        return fetch_tile(loc, args.years, d, tile,
                          cloud_max=args.cloud_max,
                          vpp_cache=vpp_cache, best_dates=best_dates_cache,
                          sources=sources_tuple)

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
                        from imint.training.tile_fetch import _CDSE_SEMAPHORE, _DES_SEMAPHORE
                        elapsed = time.time() - t0
                        print(f"  [{completed}/{len(work)}] {r['name']}: {r['status']} "
                              f"| {completed/elapsed*3600:.0f}/h "
                              f"| {_CDSE_SEMAPHORE.stats} | {_DES_SEMAPHORE.stats}",
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
