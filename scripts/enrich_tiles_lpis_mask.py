#!/usr/bin/env python3
"""
scripts/enrich_tiles_lpis_mask.py — Add LPIS parcel segmentation masks to crop tiles

Reads existing crop tile .npz files (from data/crop_tiles/) and enriches each
tile with a (256, 256) uint8 segmentation mask derived from LPIS parcel data.

For each tile:
  1. Read bbox_3006, year, label from the .npz
  2. Load the matching year's LPIS shapefile (cached in memory)
  3. Clip LPIS parcels to tile bbox using spatial index
  4. Map each parcel's grdkod_mar → crop class (1-8) via SJV_TO_CROP
  5. Rasterize all parcels to (256, 256) uint8 mask
  6. Save label_mask, lpis_year, n_parcels back into the .npz

LPIS sources (Jordbruksverket, CC BY 4.0):
  2022: data/lpis/jordbruksskiften_2022.zip (shapefile)
  2023: data/lpis/jordbruksskiften_2023.zip (shapefile)
  2024: data/lpis/jordbruksskiften_2024.zip (shapefile)
  2025: data/lpis/jordbruksskiften_2025.gml.zip (GML)
  2018: no LPIS available — tiles are skipped (scalar label only)

Usage:
    python scripts/enrich_tiles_lpis_mask.py \\
        --tiles-dir data/crop_tiles \\
        --lpis-dir data/lpis \\
        --workers 4

    # Skip tiles that already have a label_mask
    python scripts/enrich_tiles_lpis_mask.py \\
        --tiles-dir data/crop_tiles \\
        --lpis-dir data/lpis \\
        --workers 4 \\
        --skip-existing
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import numpy as np

# ── Project imports ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_crop_dataset import SJV_TO_CROP

# ── Constants ────────────────────────────────────────────────────────────

TILE_SIZE_PX = 256
TILE_SIZE_M = 2560  # 256 px * 10m

# LPIS file mapping: year -> (filename, format)
LPIS_FILES = {
    2018: ("jordbruksskiften_2018.zip", "shp"),
    2019: ("jordbruksskiften_2019.zip", "shp"),
    2022: ("jordbruksskiften_2022.zip", "shp"),
    2023: ("jordbruksskiften_2023.zip", "shp"),
    2024: ("jordbruksskiften_2024.zip", "shp"),
    2025: ("jordbruksskiften_2025.gml.zip", "gml"),
}

# Years without LPIS data
NO_LPIS_YEARS = set()

# Column name for grödkod in LPIS data
COL_GRODKOD = "grdkod_mar"


def sjv_grodkod_to_class(grodkod: int) -> int:
    """Map SJV grödkod to crop_schema class index (0 = unmapped)."""
    return SJV_TO_CROP.get(grodkod, 0)


# ── LPIS loading and caching ────────────────────────────────────────────

_lpis_cache: dict[int, "gpd.GeoDataFrame"] = {}
_lpis_cache_lock = Lock()


def _load_lpis_parquet(parquet_path: str, year: int) -> "gpd.GeoDataFrame":
    """Load LPIS from GeoParquet (fast, preferred)."""
    import geopandas as gpd
    print(f"  Loading GeoParquet: {parquet_path}")
    gdf = gpd.read_parquet(parquet_path)
    print(f"    {len(gdf)} features loaded")
    return _process_lpis_gdf(gdf, year)


def _load_lpis_shapefile(zip_path: str, year: int) -> "gpd.GeoDataFrame":
    """Load an LPIS shapefile from a zip archive (fallback if no parquet)."""
    import geopandas as gpd

    abs_path = str(Path(zip_path).resolve())
    print(f"  Loading shapefile: {abs_path}")

    gdf = None
    for uri in [abs_path, f"zip://{abs_path}"]:
        try:
            gdf = gpd.read_file(uri, layer="arslager_skiftePolygon")
            print(f"    Read via: {uri[:80]}...")
            break
        except Exception:
            continue

    if gdf is None:
        print("    Fallback: extracting zip to temp directory...")
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(abs_path, "r") as zf:
                zf.extractall(tmpdir)
            shp_files = list(Path(tmpdir).rglob("*.shp"))
            if not shp_files:
                raise FileNotFoundError(f"No .shp found inside {zip_path}")
            gdf = gpd.read_file(str(shp_files[0]))

    print(f"    Raw features: {len(gdf)}")
    return _process_lpis_gdf(gdf, year)


def _load_lpis_gml(zip_path: str, year: int) -> "gpd.GeoDataFrame":
    """Load LPIS GML from a zip archive (2025 format)."""
    import geopandas as gpd

    abs_path = str(Path(zip_path).resolve())
    print(f"  Loading GML: {abs_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(abs_path, "r") as zf:
            zf.extractall(tmpdir)
        gml_files = list(Path(tmpdir).rglob("*.gml"))
        if not gml_files:
            raise FileNotFoundError(f"No .gml found inside {zip_path}")
        gml_path = gml_files[0]
        print(f"    Found GML: {gml_path.name}")
        gdf = gpd.read_file(str(gml_path))

    print(f"    Raw features: {len(gdf)}")
    return _process_lpis_gdf(gdf, year)


def _process_lpis_gdf(gdf: "gpd.GeoDataFrame", year: int) -> "gpd.GeoDataFrame":
    """Process raw LPIS GeoDataFrame: normalize columns, map grödkoder, ensure CRS."""
    import geopandas as gpd

    if gdf.empty:
        return gdf

    # Normalize column names to lowercase
    gdf.columns = [c.lower() for c in gdf.columns]

    # Find the grödkod column
    grodkod_col = None
    for candidate in [COL_GRODKOD, "grodkod", "grdkod", "grodkod_mar"]:
        if candidate in gdf.columns:
            grodkod_col = candidate
            break

    if grodkod_col is None:
        print(f"    WARNING: No grödkod column found. Columns: {list(gdf.columns)}")
        return gpd.GeoDataFrame()

    # Drop rows without grödkod
    gdf = gdf.dropna(subset=[grodkod_col])

    # Store raw SJV grödkod as int
    gdf["grodkod_int"] = gdf[grodkod_col].astype(int)

    # Keep all parcels with a grödkod (mapping done in unified_schema.py)
    gdf = gdf[gdf["grodkod_int"] > 0].copy()
    print(f"    After filtering: {len(gdf)} parcels with grödkod")

    # Parcel area in hectares (CRS 3006 is metric — geometry.area is m²)
    gdf["area_ha"] = gdf.geometry.area / 10_000.0

    gdf["year"] = year

    # Ensure CRS is EPSG:3006
    if gdf.crs is None:
        print("    WARNING: No CRS detected, assuming EPSG:3006")
        gdf = gdf.set_crs("EPSG:3006")
    elif gdf.crs.to_epsg() != 3006:
        print(f"    Reprojecting from {gdf.crs} to EPSG:3006")
        gdf = gdf.to_crs("EPSG:3006")

    # Build spatial index (sindex) — geopandas does this lazily,
    # but we access it here to trigger creation while loading
    _ = gdf.sindex

    return gdf


def get_lpis_gdf(year: int, lpis_dir: str) -> "gpd.GeoDataFrame | None":
    """Get LPIS GeoDataFrame for a given year, loading and caching as needed.

    Thread-safe: uses a lock to prevent duplicate loads.

    Args:
        year: The LPIS year (2022-2025).
        lpis_dir: Directory containing LPIS zip files.

    Returns:
        GeoDataFrame with parcels, or None if no LPIS for this year.
    """
    if year in NO_LPIS_YEARS:
        return None

    if year not in LPIS_FILES:
        print(f"  WARNING: No LPIS file configured for year {year}")
        return None

    with _lpis_cache_lock:
        if year in _lpis_cache:
            return _lpis_cache[year]

    # Load outside lock (slow I/O), but check again after
    filename, fmt = LPIS_FILES[year]
    zip_path = os.path.join(lpis_dir, filename)
    parquet_path = os.path.join(lpis_dir, f"jordbruksskiften_{year}.parquet")

    # Prefer GeoParquet (much faster) over shapefile/GML
    if not os.path.exists(parquet_path) and not os.path.exists(zip_path):
        print(f"  WARNING: No LPIS data for {year} (checked {parquet_path} and {zip_path})")
        return None

    print(f"\nLoading LPIS {year}...")
    t0 = time.time()

    if os.path.exists(parquet_path):
        gdf = _load_lpis_parquet(parquet_path, year)
    elif fmt == "gml":
        gdf = _load_lpis_gml(zip_path, year)
    else:
        gdf = _load_lpis_shapefile(zip_path, year)

    elapsed = time.time() - t0
    print(f"  LPIS {year}: {len(gdf)} parcels loaded in {elapsed:.1f}s")

    with _lpis_cache_lock:
        # Another thread may have loaded it in the meantime
        if year not in _lpis_cache:
            _lpis_cache[year] = gdf
        return _lpis_cache[year]


# ── Rasterization ────────────────────────────────────────────────────────

def rasterize_parcels(
    gdf: "gpd.GeoDataFrame",
    bbox_3006: np.ndarray,
    tile_size: int = TILE_SIZE_PX,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Rasterize LPIS parcels within a bbox to a segmentation mask + area map.

    Args:
        gdf: Full LPIS GeoDataFrame (with spatial index, must have area_ha column).
        bbox_3006: [west, south, east, north] in EPSG:3006.
        tile_size: Output raster size in pixels (default 256).

    Returns:
        (mask, area_map, n_parcels):
          mask     — uint16 (tile_size, tile_size), raw SJV grödkod per pixel (0 = no parcel)
          area_map — float32 (tile_size, tile_size), parcel area in hectares per pixel
                     (0.0 for background; used for inverse-area loss weighting)
          n_parcels — int, number of parcels intersecting the tile
    """
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    from shapely.geometry import box

    _empty_mask = np.zeros((tile_size, tile_size), dtype=np.uint16)
    _empty_area = np.zeros((tile_size, tile_size), dtype=np.float32)

    west, south, east, north = bbox_3006

    # Clip LPIS parcels to tile bbox using spatial index
    tile_box = box(south, west, north, east)
    candidates_idx = list(gdf.sindex.intersection(tile_box.bounds))

    if not candidates_idx:
        return _empty_mask, _empty_area, 0

    clipped = gdf.iloc[candidates_idx]

    # Further filter: actual intersection (sindex is bbox-based)
    clipped = clipped[clipped.geometry.intersects(tile_box)]

    if clipped.empty:
        return _empty_mask, _empty_area, 0

    n_parcels = len(clipped)

    # Affine transform: maps pixel coordinates to EPSG:3006
    transform = from_bounds(south, west, north, east, tile_size, tile_size)

    # --- Crop-class mask (uint16, raw SJV grödkod) ---
    code_col = "grodkod_int" if "grodkod_int" in clipped.columns else "crop_class"
    code_shapes = [
        (geom, int(code))
        for geom, code in zip(clipped.geometry, clipped[code_col])
        if code > 0
    ]
    mask = rasterize(
        code_shapes,
        out_shape=(tile_size, tile_size),
        transform=transform,
        fill=0,
        dtype=np.uint16,  # SJV codes can be >255 (e.g. 300-318)
        all_touched=False,
    )

    # --- Parcel area map (float32, hectares per pixel) ---
    # Each pixel gets the area of its parcel — used for inverse-area loss weighting
    # so small parcels (<0.25 ha) receive proportionally higher gradient.
    area_col = "area_ha" if "area_ha" in clipped.columns else None
    if area_col is not None:
        area_shapes = [
            (geom, float(area))
            for geom, area in zip(clipped.geometry, clipped[area_col])
            if area > 0
        ]
        area_map = rasterize(
            area_shapes,
            out_shape=(tile_size, tile_size),
            transform=transform,
            fill=0.0,
            dtype=np.float32,
            all_touched=False,
        ).astype(np.float32)
    else:
        area_map = _empty_area

    # LPIS N,E axis order → S2 pixel grid: rot180 + transpose
    mask     = np.rot90(mask,     2).T
    area_map = np.rot90(area_map, 2).T

    return mask, area_map, n_parcels


# ── Per-tile processing ──────────────────────────────────────────────────

def process_tile(
    tile_path: str,
    lpis_dir: str,
    skip_existing: bool = False,
) -> dict:
    """Enrich a single crop tile .npz with LPIS segmentation mask.

    Args:
        tile_path: Path to the .npz tile file.
        lpis_dir: Directory containing LPIS zip files.
        skip_existing: If True, skip tiles that already have label_mask.

    Returns:
        Dict with processing result metadata.
    """
    tile_name = os.path.basename(tile_path)

    try:
        data = dict(np.load(tile_path, allow_pickle=True))
    except Exception as e:
        return {"tile": tile_name, "status": "error", "reason": f"Load failed: {e}"}

    # Check if already enriched
    if skip_existing and "label_mask" in data:
        return {"tile": tile_name, "status": "skipped", "reason": "already has label_mask"}

    # Read required fields
    if "bbox_3006" not in data:
        return {"tile": tile_name, "status": "error", "reason": "missing bbox_3006"}

    bbox_3006 = data["bbox_3006"]

    # Read year — default to 2022 (most tiles are from LUCAS 2022)
    if "year" in data:
        year = int(data["year"])
    else:
        year = 2022

    label = int(data["label"]) if "label" in data else 0
    point_id = str(data["point_id"]) if "point_id" in data else tile_name

    # Skip years without LPIS
    if year in NO_LPIS_YEARS:
        return {
            "tile": tile_name,
            "status": "skipped",
            "reason": f"no LPIS for year {year}",
            "year": year,
        }

    # Get LPIS data for this year
    gdf = get_lpis_gdf(year, lpis_dir)
    if gdf is None or gdf.empty:
        return {
            "tile": tile_name,
            "status": "skipped",
            "reason": f"LPIS not available for year {year}",
            "year": year,
        }

    # Rasterize parcels
    mask, n_parcels = rasterize_parcels(gdf, bbox_3006)

    # Update tile data
    data["label_mask"] = mask
    data["lpis_year"] = np.int32(year)
    data["n_parcels"] = np.int32(n_parcels)

    # Save back (overwrite with all fields)
    np.savez_compressed(tile_path, **data)

    return {
        "tile": tile_name,
        "status": "enriched",
        "year": year,
        "n_parcels": n_parcels,
        "label": label,
        "point_id": point_id,
    }


# ── Preloading ───────────────────────────────────────────────────────────

def preload_lpis_years(tiles_dir: str, lpis_dir: str) -> set[int]:
    """Scan tiles to find which LPIS years are needed, then preload them.

    This ensures each large LPIS file is loaded exactly once before
    parallel processing begins.

    Args:
        tiles_dir: Directory containing .npz tile files.
        lpis_dir: Directory containing LPIS zip files.

    Returns:
        Set of years that were successfully preloaded.
    """
    print("Scanning tiles for required LPIS years...")
    needed_years: set[int] = set()

    tile_paths = sorted(Path(tiles_dir).glob("*.npz"))
    for tp in tile_paths:
        try:
            data = np.load(str(tp), allow_pickle=True)
            y = int(data["year"]) if "year" in data else 2022
            if y not in NO_LPIS_YEARS and y in LPIS_FILES:
                needed_years.add(y)
        except Exception:
            continue

    print(f"  LPIS years needed: {sorted(needed_years)}")

    loaded = set()
    for year in sorted(needed_years):
        gdf = get_lpis_gdf(year, lpis_dir)
        if gdf is not None and not gdf.empty:
            loaded.add(year)

    return loaded


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Enrich crop tile .npz files with LPIS parcel segmentation masks"
    )
    parser.add_argument(
        "--tiles-dir",
        default="data/crop_tiles",
        help="Directory containing .npz crop tile files (default: data/crop_tiles)",
    )
    parser.add_argument(
        "--lpis-dir",
        default="data/lpis",
        help="Directory containing LPIS zip files (default: data/lpis)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tiles that already have a label_mask field",
    )
    args = parser.parse_args()

    tiles_dir = args.tiles_dir
    lpis_dir = args.lpis_dir

    # Discover tiles
    tile_paths = sorted(Path(tiles_dir).glob("*.npz"))
    tile_paths = [str(tp) for tp in tile_paths if tp.stem != "manifest"]

    if not tile_paths:
        print(f"No .npz tiles found in {tiles_dir}")
        sys.exit(1)

    print(f"Found {len(tile_paths)} tiles in {tiles_dir}")

    # Preload all needed LPIS files (one-time, sequential)
    preload_lpis_years(tiles_dir, lpis_dir)

    # Process tiles in parallel
    print(f"\nProcessing tiles with {args.workers} workers...")
    t0 = time.time()

    results = []
    status_counts: Counter = Counter()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_tile, tp, lpis_dir, args.skip_existing): tp
            for tp in tile_paths
        }

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            status_counts[result["status"]] += 1

            # Progress
            if i % 50 == 0 or i == len(tile_paths):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                print(
                    f"  [{i}/{len(tile_paths)}] "
                    f"enriched={status_counts.get('enriched', 0)} "
                    f"skipped={status_counts.get('skipped', 0)} "
                    f"errors={status_counts.get('error', 0)} "
                    f"({rate:.1f} tiles/s)"
                )

    elapsed = time.time() - t0

    # Summary
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Total tiles: {len(tile_paths)}")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    # Parcels stats for enriched tiles
    enriched = [r for r in results if r["status"] == "enriched"]
    if enriched:
        parcel_counts = [r["n_parcels"] for r in enriched]
        print(f"\n  Enriched tiles: {len(enriched)}")
        print(f"  Parcels per tile: min={min(parcel_counts)}, "
              f"max={max(parcel_counts)}, "
              f"mean={sum(parcel_counts)/len(parcel_counts):.1f}")

        year_counts = Counter(r["year"] for r in enriched)
        print(f"  By year: {dict(sorted(year_counts.items()))}")

    # Report errors
    errors = [r for r in results if r["status"] == "error"]
    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"    {e['tile']}: {e.get('reason', 'unknown')}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more")


if __name__ == "__main__":
    main()
