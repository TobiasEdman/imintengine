"""Generate showcase images for the Kustlinje (Shoreline) tab.

Fetches Sentinel-2 timeseries for 2018–2025 using the same
``fetch_grazing_timeseries()`` pipeline (cloud filtering via SCL,
co-registration, NMD 10 m grid snapping), runs spectral-index
shoreline analysis (NDWI/MNDWI + Otsu) per year, and saves PNGs to
outputs/showcase/kustlinje/.

Usage:
    .venv/bin/python scripts/generate_kustlinje_showcase.py

    DES credentials needed for Sentinel-2 fetching.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# Band order in cached timeseries: B01(0), B02(1), B03(2), B04(3),
# B05(4), B06(5), B07(6), B08(7), B8A(8), B09(9), B11(10), B12(11)
_BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
               "B08", "B8A", "B09", "B11", "B12"]

YEARS = list(range(2018, 2026))


def _pick_best_date(dates: list[str], cloud_fractions: list[float]) -> int:
    """Pick the best cloud-free date, preferring summer months (Jun-Aug)."""
    best_idx = 0
    best_score = -1.0
    for i, (d, cf) in enumerate(zip(dates, cloud_fractions)):
        month = int(d[5:7])
        season_bonus = 1.0 if 6 <= month <= 8 else 0.5 if month in (5, 9) else 0.0
        score = (1.0 - cf) + season_bonus
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def _build_rgb(snapshot: np.ndarray) -> np.ndarray:
    """Build contrast-stretched RGB from a (12, H, W) band stack."""
    # B04(3)=Red, B03(2)=Green, B02(1)=Blue
    rgb = np.stack([snapshot[3], snapshot[2], snapshot[1]], axis=-1)  # (H, W, 3)
    p2, p98 = np.percentile(rgb, [2, 98])
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-8), 0.0, 1.0).astype(np.float32)
    return rgb


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate kustlinje (shoreline) showcase images"
    )
    parser.add_argument(
        "--cloud-threshold", type=float, default=0.05,
        help="Max cloud fraction within bbox to keep a date (default 0.05)",
    )
    args = parser.parse_args()

    from shapely.geometry import box as shapely_box
    from rasterio.transform import from_origin
    from imint.fetch import fetch_grazing_timeseries, GeoContext
    from imint.analyzers.spectral import SpectralAnalyzer
    from imint.analyzers.cot import COTAnalyzer
    from imint.analyzers.shoreline import ShorelineAnalyzer
    from imint.exporters.export import (
        save_rgb_png,
        save_ndvi_clean_png,
        save_spectral_index_clean_png,
        save_cot_clean_png,
        save_segmentation_clean_png,
        save_shoreline_overlay,
        save_shoreline_change_png,
        save_coastline_geojson,
    )

    # ── Area: Skåne coast near Kåseberga ─────────────────────────────
    coords = {
        "west": 14.13, "south": 55.355,
        "east": 14.22, "north": 55.400,
    }

    out_dir = str(PROJECT_ROOT / "outputs" / "showcase" / "kustlinje")
    cache_dir = str(PROJECT_ROOT / "outputs" / "kustlinje_model")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    bbox_polygon = shapely_box(
        coords["west"], coords["south"], coords["east"], coords["north"]
    )

    print("=" * 60)
    print("  Kustlinje Showcase Image Generator")
    print(f"  Area: {coords['west']:.3f}–{coords['east']:.3f}°E, "
          f"{coords['south']:.3f}–{coords['north']:.3f}°N")
    print(f"  Years: {YEARS[0]}–{YEARS[-1]}")
    print(f"  Output: {out_dir}")
    print("=" * 60)

    # ── Step 1: Fetch / load S2 timeseries per year ──────────────────
    print("\n[1] Loading Sentinel-2 timeseries (2018–2025)...")
    yearly_data = {}       # year → (T, 12, H, W)
    yearly_meta = {}       # year → {dates, cloud_fractions, ...}
    geo = None

    for year in YEARS:
        cache_npz = os.path.join(cache_dir, f"bbox_timeseries_{year}.npz")
        cache_json = os.path.join(cache_dir, f"bbox_timeseries_{year}_meta.json")

        if os.path.isfile(cache_npz) and os.path.isfile(cache_json):
            print(f"    {year}: Loading from cache...")
            data = np.load(cache_npz)["data"]
            with open(cache_json) as f:
                meta = json.load(f)
        else:
            print(f"    {year}: Fetching from DES...")
            try:
                ts_list = fetch_grazing_timeseries(
                    bbox_polygon,
                    year=year,
                    cloud_threshold=args.cloud_threshold,
                    scene_cloud_max=50.0,
                    buffer_m=0.0,
                    chunk_days=14,
                    polygon_crs="EPSG:4326",
                )
            except Exception as e:
                print(f"    {year}: FAILED — {e}")
                continue

            if not ts_list or ts_list[0].data.shape[0] == 0:
                print(f"    {year}: No cloud-free dates found")
                continue

            ts = ts_list[0]
            data = ts.data

            meta = {
                "dates": ts.dates,
                "cloud_fractions": ts.cloud_fractions,
                "west": float(ts.geo.bounds_projected["west"]),
                "south": float(ts.geo.bounds_projected["south"]),
                "east": float(ts.geo.bounds_projected["east"]),
                "north": float(ts.geo.bounds_projected["north"]),
                "pixel_size": 10,
            }

            # Cache
            np.savez_compressed(cache_npz, data=data)
            with open(cache_json, "w") as f:
                json.dump(meta, f, indent=2)
            print(f"    {year}: Cached ({data.shape[0]} dates)")

            if geo is None:
                geo = ts.geo

        yearly_data[year] = data
        yearly_meta[year] = meta
        T = data.shape[0]
        print(f"    {year}: {T} dates, shape {data.shape}")

    if not yearly_data:
        print("ERROR: No data for any year. Aborting.")
        return

    # Build GeoContext from first available year if not set
    if geo is None:
        first_meta = next(iter(yearly_meta.values()))
        H, W = next(iter(yearly_data.values())).shape[2:]
        pixel_size = first_meta.get("pixel_size", 10)
        geo = GeoContext(
            crs="EPSG:3006",
            transform=from_origin(
                first_meta["west"], first_meta["north"],
                pixel_size, pixel_size,
            ),
            bounds_projected={
                "west": first_meta["west"],
                "south": first_meta["south"],
                "east": first_meta["east"],
                "north": first_meta["north"],
            },
            bounds_wgs84=None,
            shape=(H, W),
        )

    # ── Step 2: Pick best summer RGB per year ────────────────────────
    print("\n[2] Selecting best summer date per year...")
    yearly_rgbs = {}     # year → (H, W, 3) float32
    yearly_bands = {}    # year → {name: (H, W)}
    yearly_dates = {}    # year → str

    for year in sorted(yearly_data.keys()):
        data = yearly_data[year]
        meta = yearly_meta[year]
        best_idx = _pick_best_date(meta["dates"], meta["cloud_fractions"])
        date = meta["dates"][best_idx]
        cloud = meta["cloud_fractions"][best_idx]
        snapshot = data[best_idx]  # (12, H, W)

        rgb = _build_rgb(snapshot)
        yearly_rgbs[year] = rgb
        yearly_bands[year] = {name: snapshot[i] for i, name in enumerate(_BAND_NAMES)}
        yearly_dates[year] = date
        print(f"    {year}: {date} (cloud: {cloud:.1%})")

    # Use latest year as reference
    ref_year = max(yearly_rgbs.keys())
    ref_rgb = yearly_rgbs[ref_year]
    ref_date = yearly_dates[ref_year]
    ref_bands = yearly_bands[ref_year]
    H, W = ref_rgb.shape[:2]

    # ── Step 3: Save reference RGB ───────────────────────────────────
    print(f"\n[3] Saving reference RGB ({ref_year})...")
    save_rgb_png(ref_rgb, os.path.join(out_dir, "rgb.png"))

    # ── Step 4: Spectral indices (reference year) ────────────────────
    print("\n[4] Running spectral analysis...")
    spectral = SpectralAnalyzer()
    spec_result = spectral.run(
        ref_rgb, bands=ref_bands, date=ref_date,
        coords=coords, output_dir=out_dir,
    )
    indices = spec_result.outputs.get("indices", {})

    if "NDVI" in indices:
        save_ndvi_clean_png(indices["NDVI"], os.path.join(out_dir, "ndvi_clean.png"))
    if "NDWI" in indices:
        save_spectral_index_clean_png(
            indices["NDWI"], os.path.join(out_dir, "ndwi_clean.png"),
            cmap_name="RdBu", vmin=-1, vmax=1,
        )
    if "EVI" in indices:
        save_spectral_index_clean_png(
            indices["EVI"], os.path.join(out_dir, "evi_clean.png"),
            cmap_name="RdYlGn", vmin=-0.5, vmax=1,
        )

    # ── Step 5: COT ──────────────────────────────────────────────────
    print("\n[5] Running COT analysis...")
    try:
        cot = COTAnalyzer()
        cot_result = cot.run(
            ref_rgb, bands=ref_bands, date=ref_date,
            coords=coords, output_dir=out_dir,
        )
        cot_map = cot_result.outputs.get("cot_map")
        if cot_map is not None:
            save_cot_clean_png(cot_map, os.path.join(out_dir, "cot_clean.png"))
    except Exception as e:
        print(f"    COT skipped: {e}")

    # ── Step 6: Shoreline segmentation (all years) ───────────────────
    print("\n[6] Running NDWI/MNDWI shoreline segmentation...")
    analyzer = ShorelineAnalyzer()

    # Segmentation per year
    yearly_seg = {}         # year → seg_map (H, W)
    yearly_info = {}        # year → info dict (thresholds, fractions)
    yearly_shorelines = {}  # year → shoreline_mask (H, W)
    yearly_contours = {}    # year → [contour arrays]

    for year in sorted(yearly_bands.keys()):
        print(f"    {year}...", end=" ", flush=True)
        seg_map, info = analyzer.classify(yearly_bands[year], yearly_rgbs[year])
        shoreline = analyzer.extract_shoreline(seg_map)
        contours = analyzer.extract_contours(shoreline, min_length=10)

        yearly_seg[year] = seg_map
        yearly_info[year] = info
        yearly_shorelines[year] = shoreline
        yearly_contours[year] = contours

        print(f"water={info['water_fraction']:.1%}, "
              f"{len(contours)} contours, "
              f"NDWI thresh={info['ndwi_threshold']:.3f}")

    # Save reference year segmentation
    save_segmentation_clean_png(
        yearly_seg[ref_year],
        os.path.join(out_dir, "coastline_clean.png"),
    )

    # Save shoreline overlay on RGB
    save_shoreline_overlay(
        ref_rgb, yearly_shorelines[ref_year],
        os.path.join(out_dir, "shoreline_overlay.png"),
    )

    # Save multi-year shoreline change
    save_shoreline_change_png(
        yearly_shorelines,
        ref_rgb,
        os.path.join(out_dir, "shoreline_change.png"),
    )

    # Save GeoJSON with all years' contours (pixel coords for Leaflet)
    save_coastline_geojson(
        yearly_contours,
        geo,
        os.path.join(out_dir, "coastline_vectors.json"),
        img_shape=(H, W),
        pixel_coords=True,
        smooth_sigma=3.0,
        subsample_step=3,
    )

    # ── Step 7: Compute statistics and save metadata ─────────────────
    print("\n[7] Computing statistics...")

    # Per-year stats
    per_year_stats = {}
    for year in sorted(yearly_seg.keys()):
        info = yearly_info[year]
        per_year_stats[year] = {
            "date": yearly_dates[year],
            "water_fraction": round(info["water_fraction"], 4),
            "deep_water_fraction": round(info["deep_water_fraction"], 4),
            "shallow_water_fraction": round(info["shallow_water_fraction"], 4),
            "sediment_fraction": round(info["sediment_fraction"], 4),
            "land_fraction": round(info["land_fraction"], 4),
            "ndwi_threshold": info["ndwi_threshold"],
            "mndwi_threshold": info.get("mndwi_threshold"),
            "n_contours": len(yearly_contours[year]),
        }
        print(f"    {year}: water={per_year_stats[year]['water_fraction']:.1%}, "
              f"NDWI thresh={info['ndwi_threshold']:.3f}")

    # NDWI stats for reference year
    ndwi_mean = 0.0
    if "NDWI" in indices:
        ndwi_mean = float(np.nanmean(indices["NDWI"]))

    kustlinje_meta = {
        "date": ref_date,
        "reference_year": ref_year,
        "coords": coords,
        "shape": [H, W],
        "years_analyzed": sorted(yearly_seg.keys()),
        "num_years": len(yearly_seg),
        "method": "NDWI/MNDWI + Otsu (CoastSat approach)",
        "reference": "Vos et al. (2019) CoastSat",
        "per_year": per_year_stats,
        "ndwi_mean": round(ndwi_mean, 4),
    }

    with open(os.path.join(out_dir, "kustlinje_meta.json"), "w") as f:
        json.dump(kustlinje_meta, f, indent=2, ensure_ascii=False)
    print("    Saved kustlinje_meta.json")

    print(f"\n{'=' * 60}")
    print(f"  Done! Images saved to {out_dir}")
    files = [f for f in os.listdir(out_dir)
             if f.endswith((".png", ".json"))]
    print(f"  Files: {', '.join(sorted(files))}")
    print(f"  Years: {sorted(yearly_seg.keys())}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import functools
    print = functools.partial(print, flush=True)
    main()
