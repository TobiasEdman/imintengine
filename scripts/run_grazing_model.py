"""Run the pib-ml-grazing CNN-biLSTM model on LPIS pasture polygons.

Fetches Sentinel-2 timeseries ONCE for the full bounding box, then crops
per polygon and runs the grazing activity classifier on each.

Usage:
    .venv/bin/python scripts/run_grazing_model.py [--year 2025] [--max-polygons 80]

Requires DES_USER and DES_PASSWORD environment variables.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Run grazing model on LPIS polygons")
    parser.add_argument("--year", type=int, default=2025, help="Year to fetch S2 data")
    parser.add_argument("--max-polygons", type=int, default=80,
                        help="Max number of LPIS polygons to process")
    parser.add_argument("--cloud-threshold", type=float, default=0.05,
                        help="Max cloud fraction per timestep (default: 5%%)")
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "outputs" / "grazing_model"),
                        help="Output directory")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device (cpu/cuda, auto-detected)")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    import geopandas as gpd
    from shapely.geometry import box as shapely_box
    from rasterio.transform import from_origin

    from imint.fetch import (
        fetch_lpis_polygons,
        fetch_grazing_timeseries,
        GrazingTimeseriesResult,
        GeoContext,
    )
    from imint.analyzers.grazing import GrazingAnalyzer

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Area: pasture-rich zone near Lund, Skåne ──────────────────────
    bbox = {
        "west": 13.42, "south": 55.935,
        "east": 13.48, "north": 55.965,
    }

    print("=" * 60)
    print("  Grazing Model — pib-ml-grazing CNN-biLSTM")
    print(f"  Area: {bbox['west']:.3f}–{bbox['east']:.3f}°E, "
          f"{bbox['south']:.3f}–{bbox['north']:.3f}°N")
    print(f"  Year: {args.year}")
    print(f"  Output: {args.output_dir}")
    print("=" * 60)

    # ── Step 1: Fetch LPIS pasture polygons ────────────────────────────
    print("\n[1] Fetching LPIS pasture polygons...")
    lpis_gdf = fetch_lpis_polygons(
        bbox=bbox,
        agoslag="Bete",
        max_features=500,
    )
    print(f"    Found {len(lpis_gdf)} pasture polygons")

    if len(lpis_gdf) == 0:
        print("    No polygons found — exiting.")
        return

    # Clip to bbox
    clip_box_wgs84 = shapely_box(bbox["west"], bbox["south"],
                                 bbox["east"], bbox["north"])
    clip_gdf = gpd.GeoDataFrame(
        geometry=[clip_box_wgs84], crs="EPSG:4326",
    ).to_crs(lpis_gdf.crs)
    lpis_gdf = gpd.clip(lpis_gdf, clip_gdf)
    print(f"    {len(lpis_gdf)} polygons after bbox clip")

    if len(lpis_gdf) > args.max_polygons:
        lpis_gdf = lpis_gdf.head(args.max_polygons)
        print(f"    Limited to {args.max_polygons} polygons")

    # ── Step 2: Fetch S2 timeseries ONCE for the full bbox ────────────
    cache_npz = os.path.join(args.output_dir, f"bbox_timeseries_{args.year}.npz")
    cache_json = os.path.join(args.output_dir, f"bbox_timeseries_{args.year}_meta.json")

    if os.path.isfile(cache_npz) and os.path.isfile(cache_json):
        print(f"\n[2] Loading cached S2 timeseries from {cache_npz}...")
        cached = np.load(cache_npz)
        data_array = cached["data"]
        with open(cache_json) as _f:
            cache_meta = json.load(_f)
        from rasterio.transform import from_origin
        full_ts = GrazingTimeseriesResult(
            data=data_array,
            dates=cache_meta["dates"],
            cloud_fractions=cache_meta["cloud_fractions"],
            polygon_id=0,
            geo=GeoContext(
                crs="EPSG:3006",
                transform=from_origin(
                    cache_meta["west"], cache_meta["north"],
                    cache_meta["pixel_size"], cache_meta["pixel_size"],
                ),
                bounds_projected={"west": cache_meta["west"], "south": cache_meta["south"],
                                  "east": cache_meta["east"], "north": cache_meta["north"]},
                bounds_wgs84=None,
                shape=(data_array.shape[2], data_array.shape[3]),
            ),
            shape_hw=(data_array.shape[2], data_array.shape[3]),
        )
        print(f"    Loaded: {data_array.shape[0]} dates, shape {data_array.shape}")
    else:
        print(f"\n[2] Fetching S2 timeseries for full bbox ({args.year})...")
        bbox_polygon = shapely_box(bbox["west"], bbox["south"],
                                   bbox["east"], bbox["north"])
        bbox_ts = fetch_grazing_timeseries(
            bbox_polygon,
            year=args.year,
            cloud_threshold=args.cloud_threshold,
            scene_cloud_max=50.0,
            buffer_m=0.0,
            chunk_days=14,
            polygon_crs="EPSG:4326",
        )
        if not bbox_ts or bbox_ts[0].data.shape[0] == 0:
            print("    No cloud-free dates found — exiting.")
            with open(os.path.join(args.output_dir, "grazing_results.json"), "w") as f:
                json.dump({"year": args.year, "predictions": [],
                            "error": "No valid timeseries"}, f, indent=2)
            return

        full_ts = bbox_ts[0]
        # Cache to disk
        np.savez_compressed(cache_npz, data=full_ts.data)
        tf = full_ts.geo.transform
        cache_meta = {
            "dates": full_ts.dates,
            "cloud_fractions": full_ts.cloud_fractions,
            "west": tf[2], "north": tf[5],
            "east": tf[2] + full_ts.data.shape[3] * abs(tf[0]),
            "south": tf[5] - full_ts.data.shape[2] * abs(tf[4]),
            "pixel_size": abs(tf[0]),
        }
        with open(cache_json, "w") as _f:
            json.dump(cache_meta, _f, indent=2)
        print(f"    Cached to {cache_npz}")

    print(f"    Full bbox: {full_ts.data.shape[0]} cloud-free dates, "
          f"shape {full_ts.data.shape}")

    # ── Step 3: Crop per polygon ──────────────────────────────────────
    print(f"\n[3] Cropping per polygon ({len(lpis_gdf)} polygons)...")
    full_transform = full_ts.geo.transform
    pixel_size = abs(full_transform[0])  # 10m
    full_west = full_transform[2]  # xoff
    full_north = full_transform[5]  # yoff (top)

    T, C, H_full, W_full = full_ts.data.shape
    timeseries_results = []

    for _, row in lpis_gdf.iterrows():
        pid = row.get("blockid", row.name)
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # Polygon bbox in EPSG:3006 (LPIS is already in EPSG:3006)
        minx, miny, maxx, maxy = geom.bounds

        # Add 20m buffer for model context
        buffer_m = 20.0
        minx -= buffer_m
        miny -= buffer_m
        maxx += buffer_m
        maxy += buffer_m

        # Convert to pixel coords in full array
        col_start = int((minx - full_west) / pixel_size)
        col_end = int((maxx - full_west) / pixel_size)
        row_start = int((full_north - maxy) / pixel_size)
        row_end = int((full_north - miny) / pixel_size)

        # Clamp to array bounds
        col_start = max(0, col_start)
        col_end = min(W_full, col_end)
        row_start = max(0, row_start)
        row_end = min(H_full, row_end)

        if col_end <= col_start or row_end <= row_start:
            print(f"    Polygon {pid}: out of bounds, skip")
            continue

        # Crop: (T, 12, H_poly, W_poly)
        crop = full_ts.data[:, :, row_start:row_end, col_start:col_end]
        H_poly, W_poly = crop.shape[2], crop.shape[3]

        # Create per-polygon GeoContext
        poly_west = full_west + col_start * pixel_size
        poly_north = full_north - row_start * pixel_size
        poly_tf = from_origin(poly_west, poly_north, pixel_size, pixel_size)

        ts_result = GrazingTimeseriesResult(
            data=crop,
            dates=full_ts.dates,
            cloud_fractions=full_ts.cloud_fractions,
            polygon_id=pid,
            geo=GeoContext(
                crs="EPSG:3006",
                transform=poly_tf,
                bounds_projected={"west": poly_west, "south": poly_north - H_poly * pixel_size,
                                  "east": poly_west + W_poly * pixel_size, "north": poly_north},
                bounds_wgs84=None,
                shape=(H_poly, W_poly),
            ),
            shape_hw=(H_poly, W_poly),
        )
        timeseries_results.append(ts_result)

    print(f"    Cropped {len(timeseries_results)} polygon timeseries")

    # Filter out polygons with too few dates (model pads small polygons)
    valid_ts = [ts for ts in timeseries_results
                if ts.data.shape[0] >= 4]
    print(f"    {len(valid_ts)} polygons with >= 4 cloud-free dates")

    if not valid_ts:
        print("    No valid timeseries — exiting.")
        with open(os.path.join(args.output_dir, "grazing_results.json"), "w") as f:
            json.dump({"year": args.year, "predictions": [],
                        "error": "No valid timeseries"}, f, indent=2)
        return

    # ── Step 4: Run grazing model ──────────────────────────────────────
    print(f"\n[4] Running grazing model on {len(valid_ts)} polygons...")
    analyzer = GrazingAnalyzer(device=args.device)
    predictions = analyzer.predict_batch(valid_ts)

    # ── Step 5: Print and save results ─────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  RESULTS")
    print(f"{'=' * 60}")

    results = []
    for pred, ts in zip(predictions, valid_ts):
        T = ts.data.shape[0]
        print(f"  Polygon {pred.polygon_id}: {pred.class_label} "
              f"(conf={pred.confidence:.1%}, T={T} dates, "
              f"shape={ts.data.shape[2]}x{ts.data.shape[3]})")
        results.append({
            "polygon_id": str(pred.polygon_id),
            "predicted_class": pred.predicted_class,
            "class_label": pred.class_label,
            "confidence": round(pred.confidence, 4),
            "probabilities": [round(p, 4) for p in pred.probabilities],
            "num_dates": T,
            "dates": ts.dates,
        })

    # Summary stats
    n_grazing = sum(1 for p in predictions if p.predicted_class == 1)
    n_no = sum(1 for p in predictions if p.predicted_class == 0)
    n_err = sum(1 for p in predictions if p.predicted_class == -1)
    print(f"\n  Summary: {n_grazing} active grazing, {n_no} no activity, "
          f"{n_err} errors out of {len(predictions)} polygons")

    # Save results
    output = {
        "year": args.year,
        "bbox": bbox,
        "model": "pib-ml-grazing CNN-biLSTM (RISE, 2025)",
        "predictions": results,
    }
    out_path = os.path.join(args.output_dir, "grazing_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {out_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import functools
    print = functools.partial(print, flush=True)
    main()
