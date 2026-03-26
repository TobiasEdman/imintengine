#!/usr/bin/env python3
"""
scripts/fetch_lucas_tiles.py — Fetch Sentinel-2 multitemporal tiles for LUCAS-SE crop points

Builds the training dataset for Swedish crop classification by:
  1. Loading LUCAS Copernicus 2018 + 2022 points filtered for Sweden
  2. For each point, fetching 3 seasonal Sentinel-2 scenes (spring, summer, autumn)
  3. Stacking into (18, 224, 224) multitemporal tiles with crop class labels
  4. Saving as .npz files ready for Prithvi fine-tuning

Data fetch strategy (per docstring in imint/training/cdse_s2.py):
  Primary:  CDSE Sentinel Hub Process API (fast, single HTTP POST)
  Fallback: DES openEO (if CDSE fails)

STAC discovery always uses DES STAC (explorer.digitalearth.se).

Usage:
    # Both 2018 and 2022 surveys (recommended — maximizes Swedish crop points)
    python scripts/fetch_lucas_tiles.py \\
        --lucas-csv data/lucas_copernicus_2018.csv data/lucas_copernicus_2022.csv \\
        --output-dir data/crop_tiles \\
        --workers 3

    # Single survey
    python scripts/fetch_lucas_tiles.py \\
        --lucas-csv data/lucas_copernicus_2022.csv \\
        --output-dir data/crop_tiles

Prerequisites:
    - CDSE credentials: export CDSE_CLIENT_ID=... CDSE_CLIENT_SECRET=...
    - DES token (for STAC + fallback): .des_token in project root
    - LUCAS 2018 CSV: https://doi.org/10.6084/m9.figshare.12382667
    - LUCAS 2022 CSV: https://doi.org/10.6084/m9.figshare.24090553
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imint.training.crop_schema import (
    load_lucas_sweden,
    summarize_lucas_sweden,
    CLASS_NAMES,
    NUM_CLASSES,
)
from imint.fetch import _to_nmd_grid, _stac_available_dates


# ── Constants ─────────────────────────────────────────────────────────────

# Seasonal windows: (start_month, end_month)
SEASONAL_WINDOWS = [
    (4, 5),   # Spring: April–May
    (6, 7),   # Summer: June–July
    (8, 9),   # Autumn: August–September
]

TILE_SIZE_M = 2560   # 256 pixels × 10m = 2560m bounding box
TILE_SIZE_PX = 256
PRITHVI_BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]


# ── Tile fetching ─────────────────────────────────────────────────────────

def _point_to_bbox_3006(lat: float, lon: float) -> dict:
    """Convert a WGS84 point to a 2560m × 2560m EPSG:3006 bounding box."""
    from rasterio.crs import CRS
    from rasterio.warp import transform

    xs, ys = transform(
        CRS.from_epsg(4326), CRS.from_epsg(3006), [lon], [lat],
    )
    cx, cy = xs[0], ys[0]
    half = TILE_SIZE_M / 2
    return {
        "west": int(cx - half),
        "south": int(cy - half),
        "east": int(cx + half),
        "north": int(cy + half),
    }


def _fetch_seasonal_scenes(
    bbox_3006: dict,
    coords_wgs84: dict,
    years: list[str],
    *,
    scene_cloud_max: float = 30.0,
    max_candidates: int = 3,
) -> list[np.ndarray | None]:
    """Fetch 3 seasonal S2 scenes: CDSE HTTP primary, DES openEO fallback.

    Returns:
        List of 3 arrays, each (6, 256, 256) float32 or None if unavailable.
    """
    from imint.training.cdse_s2 import fetch_s2_scene

    scenes = []
    for window_start, window_end in SEASONAL_WINDOWS:
        scene = None

        # STAC discovery (all years, sorted by cloud)
        candidates = []
        for year in years:
            date_start = f"{year}-{window_start:02d}-01"
            if window_end in (1, 3, 5, 7, 8, 10, 12):
                date_end = f"{year}-{window_end:02d}-31"
            elif window_end in (4, 6, 9, 11):
                date_end = f"{year}-{window_end:02d}-30"
            else:
                date_end = f"{year}-{window_end:02d}-28"

            try:
                dates = _stac_available_dates(
                    coords_wgs84, date_start, date_end,
                    scene_cloud_max=scene_cloud_max,
                )
                candidates.extend(dates)
            except Exception:
                pass

        candidates.sort(key=lambda x: x[1])  # sort by cloud fraction

        # Try top candidates via CDSE HTTP (primary)
        for date_str, cloud_pct in candidates[:max_candidates]:
            try:
                result = fetch_s2_scene(
                    bbox_3006["west"], bbox_3006["south"],
                    bbox_3006["east"], bbox_3006["north"],
                    date=date_str,
                    size_px=TILE_SIZE_PX,
                )
                if result is not None:
                    spectral, scl, cloud_frac = result
                    scene = spectral  # (6, H, W) float32
                    break
            except Exception:
                continue

        # Fallback: DES openEO
        if scene is None and candidates:
            best_date = candidates[0][0]
            try:
                from imint.fetch import fetch_seasonal_image
                result = fetch_seasonal_image(
                    date=best_date,
                    coords=coords_wgs84,
                    prithvi_bands=PRITHVI_BANDS,
                    source="des",
                )
                if result is not None:
                    scene = result[0]  # (6, H, W)
            except Exception:
                pass

        scenes.append(scene)

    return scenes


def _process_point(
    point: dict,
    years_override: list[str] | None,
    output_dir: str,
    scene_cloud_max: float = 30.0,
) -> dict:
    """Process a single LUCAS point: fetch S2 tiles and save .npz.

    S2 imagery is matched to the LUCAS survey year by default — a point
    surveyed in 2018 fetches S2 from 2018, a 2022 point from 2022.
    Use years_override to search across multiple years instead.

    Returns:
        Status dict with point_id, success, path, etc.
    """
    point_id = point["point_id"]
    lat, lon = point["lat"], point["lon"]
    crop_class = point["crop_class"]
    survey_year = point.get("year", 2018)

    # Match S2 year to LUCAS survey year (default), or use override
    years = years_override or [str(survey_year)]

    # Bounding box in EPSG:3006
    bbox_3006 = _point_to_bbox_3006(lat, lon)
    coords_wgs84 = {
        "west": lon - 0.015,
        "south": lat - 0.012,
        "east": lon + 0.015,
        "north": lat + 0.012,
    }

    # Fetch 3 seasonal scenes
    scenes = _fetch_seasonal_scenes(
        bbox_3006, coords_wgs84, years,
        scene_cloud_max=scene_cloud_max,
    )

    # Check completeness — need at least 2 of 3 seasons
    valid = [s is not None for s in scenes]
    if sum(valid) < 2:
        return {
            "point_id": point_id,
            "success": False,
            "reason": f"Only {sum(valid)}/3 seasons available",
        }

    # Stack: fill missing seasons with zeros
    stacked = []
    for s in scenes:
        if s is not None:
            stacked.append(s)
        else:
            stacked.append(np.zeros((6, TILE_SIZE_PX, TILE_SIZE_PX), dtype=np.float32))
    multitemporal = np.concatenate(stacked, axis=0)  # (18, 256, 256)

    # Save as .npz
    out_path = os.path.join(output_dir, f"{point_id}.npz")
    np.savez_compressed(
        out_path,
        spectral=multitemporal,              # (18, 256, 256) float32
        label=np.uint8(crop_class),          # scalar
        label_name=CLASS_NAMES[crop_class],
        lat=np.float64(lat),
        lon=np.float64(lon),
        point_id=point_id,
        seasons_valid=np.array(valid),
        bbox_3006=np.array([
            bbox_3006["west"], bbox_3006["south"],
            bbox_3006["east"], bbox_3006["north"],
        ]),
    )

    return {
        "point_id": point_id,
        "success": True,
        "path": out_path,
        "crop_class": crop_class,
        "seasons_valid": valid,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch S2 multitemporal tiles for LUCAS-SE crop points"
    )
    parser.add_argument(
        "--lucas-csv", nargs="+", default=None,
        help="Path(s) to LUCAS Copernicus CSV files (2018 and/or 2022)",
    )
    parser.add_argument(
        "--balanced-json", default=None,
        help="Path to balanced_points.json (overrides --lucas-csv)",
    )
    parser.add_argument(
        "--output-dir", default="data/crop_tiles",
        help="Output directory for .npz tiles (default: data/crop_tiles)",
    )
    parser.add_argument(
        "--workers", type=int, default=3,
        help="Parallel fetch workers (default: 3)",
    )
    parser.add_argument(
        "--years", nargs="+", default=None,
        help=(
            "Override S2 search years. Default: match LUCAS survey year "
            "(2018 point → S2 2018, 2022 point → S2 2022). "
            "Use --years 2018 2019 to search across multiple years."
        ),
    )
    parser.add_argument(
        "--cloud-max", type=float, default=30.0,
        help="Max scene cloud cover %% for STAC candidates (default: 30)",
    )
    parser.add_argument(
        "--max-points", type=int, default=None,
        help="Max points to process (for testing)",
    )
    parser.add_argument(
        "--crop-only", action="store_true", default=True,
        help="Only include agricultural points (default: True)",
    )
    args = parser.parse_args()

    # Load points: balanced JSON or raw CSV
    if args.balanced_json:
        print(f"Loading balanced points from {args.balanced_json}...")
        with open(args.balanced_json) as f:
            data = json.load(f)
        points = data["points"]
        print(f"  {data['metadata']['total_balanced']} balanced points "
              f"(from {data['metadata']['total_original']} original)")
    elif args.lucas_csv:
        csv_paths = args.lucas_csv if len(args.lucas_csv) > 1 else args.lucas_csv[0]
        print(f"Loading LUCAS points from {args.lucas_csv}...")
        points = load_lucas_sweden(csv_paths, crop_only=args.crop_only)
    else:
        parser.error("Provide --balanced-json or --lucas-csv")
    summary = summarize_lucas_sweden(points)
    print(f"  Found {summary['total']} Swedish crop points:")
    for cls_name, count in summary["per_class"].items():
        if count > 0:
            print(f"    {cls_name}: {count}")

    if args.max_points:
        points = points[: args.max_points]
        print(f"  Limited to {len(points)} points")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Skip already-fetched points
    existing = {
        p.stem for p in Path(args.output_dir).glob("*.npz")
    }
    remaining = [p for p in points if p["point_id"] not in existing]
    print(f"  {len(existing)} already fetched, {len(remaining)} remaining")

    if not remaining:
        print("All points already fetched!")
        return

    # Fetch in parallel
    year_mode = args.years if args.years else "matched to LUCAS survey year"
    print(f"\nFetching S2 tiles ({args.workers} workers, years={year_mode})...")
    print(f"  Primary: CDSE Sentinel Hub HTTP")
    print(f"  Fallback: DES openEO")

    t0 = time.time()
    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _process_point, pt, args.years, args.output_dir, args.cloud_max,
            ): pt
            for pt in remaining
        }

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result["success"]:
                success_count += 1
                print(
                    f"  [{i}/{len(remaining)}] {result['point_id']} "
                    f"→ {CLASS_NAMES[result['crop_class']]} "
                    f"(seasons: {result['seasons_valid']})"
                )
            else:
                fail_count += 1
                if i % 50 == 0 or fail_count < 5:
                    print(
                        f"  [{i}/{len(remaining)}] {result['point_id']} "
                        f"FAILED: {result['reason']}"
                    )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Success: {success_count} / {len(remaining)}")
    print(f"  Failed:  {fail_count} / {len(remaining)}")
    print(f"  Tiles:   {args.output_dir}/")

    # Save manifest
    manifest = {
        "lucas_csv": args.lucas_csv,
        "output_dir": args.output_dir,
        "years": args.years,
        "total_points": len(points),
        "fetched": success_count,
        "failed": fail_count,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "tile_size_px": TILE_SIZE_PX,
        "tile_size_m": TILE_SIZE_M,
        "bands": PRITHVI_BANDS,
        "seasonal_windows": SEASONAL_WINDOWS,
        "fetch_strategy": "CDSE HTTP primary, DES openEO fallback",
    }
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
