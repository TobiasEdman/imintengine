"""Generate showcase images for the Grazing (Betesmark) tab.

Uses the cached Sentinel-2 timeseries from run_grazing_model.py
(bbox_timeseries_YYYY.npz) to pick the best cloud-free date,
runs spectral + NMD analysis, overlays LPIS polygons with
grazing model predictions, and saves PNGs to outputs/showcase/grazing/.

Usage:
    .venv/bin/python scripts/generate_grazing_showcase.py [--year 2025]

    DES credentials only needed if NMD fetch is used (no cached NMD).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


# Band order in cached timeseries: B01(0), B02(1), B03(2), B04(3),
# B05(4), B06(5), B07(6), B08(7), B8A(8), B09(9), B11(10), B12(11)
_BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
               "B08", "B8A", "B09", "B11", "B12"]


def _pick_best_date(dates: list[str], cloud_fractions: list[float]) -> int:
    """Pick the best cloud-free date, preferring summer months (Jun–Aug)."""
    best_idx = 0
    best_score = -1.0
    for i, (d, cf) in enumerate(zip(dates, cloud_fractions)):
        month = int(d[5:7])
        # Prefer summer months for greener RGB
        season_bonus = 1.0 if 6 <= month <= 8 else 0.5 if month in (5, 9) else 0.0
        score = (1.0 - cf) + season_bonus
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate grazing showcase images")
    parser.add_argument("--year", type=int, default=2025)
    args = parser.parse_args()

    from rasterio.transform import from_origin
    from imint.fetch import fetch_lpis_polygons, fetch_nmd_data, GeoContext
    from imint.analyzers.spectral import SpectralAnalyzer
    from imint.exporters.export import (
        save_rgb_png,
        save_ndvi_clean_png,
        save_spectral_index_clean_png,
        save_nmd_overlay,
        save_lpis_overlay,
        save_lpis_geojson,
    )

    # ── Area: pasture-rich zone near Lund, Skåne ──────────────────────
    coords = {
        "west": 13.42, "south": 55.935,
        "east": 13.48, "north": 55.965,
    }

    out_dir = str(PROJECT_ROOT / "outputs" / "showcase" / "grazing")
    os.makedirs(out_dir, exist_ok=True)

    # ── Step 1: Load cached timeseries ──────────────────────────────────
    cache_dir = str(PROJECT_ROOT / "outputs" / "grazing_model")
    cache_npz = os.path.join(cache_dir, f"bbox_timeseries_{args.year}.npz")
    cache_json = os.path.join(cache_dir, f"bbox_timeseries_{args.year}_meta.json")

    if not os.path.isfile(cache_npz) or not os.path.isfile(cache_json):
        print(f"ERROR: Cached timeseries not found at {cache_npz}")
        print("       Run scripts/run_grazing_model.py first.")
        return

    print("=" * 60)
    print("  Grazing Showcase Image Generator")
    print(f"  Area: {coords['west']:.3f}–{coords['east']:.3f}°E, "
          f"{coords['south']:.3f}–{coords['north']:.3f}°N")
    print(f"  Output: {out_dir}")
    print("=" * 60)

    print(f"\n[1] Loading cached S2 timeseries ({args.year})...")
    data = np.load(cache_npz)["data"]  # (T, 12, H, W)
    with open(cache_json) as f:
        meta = json.load(f)

    dates = meta["dates"]
    cloud_fractions = meta["cloud_fractions"]
    pixel_size = meta["pixel_size"]
    T, C, H, W = data.shape
    print(f"    {T} cloud-free dates, shape ({T}, {C}, {H}, {W})")

    # Pick best date
    best_idx = _pick_best_date(dates, cloud_fractions)
    date = dates[best_idx]
    cloud_frac = cloud_fractions[best_idx]
    print(f"    Best date: {date} (cloud: {cloud_frac:.1%})")

    # Extract single timestep: (12, H, W)
    snapshot = data[best_idx]

    # Build RGB: B04(3)=Red, B03(2)=Green, B02(1)=Blue
    rgb = np.stack([snapshot[3], snapshot[2], snapshot[1]], axis=-1)  # (H, W, 3)
    # Percentile contrast stretch for better visualisation
    p2, p98 = np.percentile(rgb, [2, 98])
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-8), 0.0, 1.0).astype(np.float32)
    print(f"    RGB shape: {rgb.shape}, stretch: [{p2:.4f}, {p98:.4f}]")

    # Build bands dict for analyzers
    bands = {name: snapshot[i] for i, name in enumerate(_BAND_NAMES)}

    # Build GeoContext
    geo = GeoContext(
        crs="EPSG:3006",
        transform=from_origin(meta["west"], meta["north"], pixel_size, pixel_size),
        bounds_projected={
            "west": meta["west"], "south": meta["south"],
            "east": meta["east"], "north": meta["north"],
        },
        bounds_wgs84=None,
        shape=(H, W),
    )

    # ── Step 2: Save RGB ──────────────────────────────────────────────
    print("\n[2] Saving RGB...")
    save_rgb_png(rgb, os.path.join(out_dir, "rgb.png"))

    # ── Step 3: Spectral indices ──────────────────────────────────────
    print("\n[3] Running spectral analysis...")
    spectral = SpectralAnalyzer()
    spec_result = spectral.run(
        rgb, bands=bands, date=date, coords=coords, output_dir=out_dir,
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

    # ── Step 4: NMD ───────────────────────────────────────────────────
    print("\n[4] Fetching NMD land cover...")
    nmd_result = None
    try:
        nmd_result = fetch_nmd_data(coords=coords, target_shape=(H, W))
        if nmd_result is not None:
            save_nmd_overlay(nmd_result.nmd_raster, os.path.join(out_dir, "nmd_overlay.png"))
    except Exception as e:
        print(f"    NMD skipped: {e}")

    # ── Step 5: LPIS overlay with predictions ─────────────────────────
    print("\n[5] Fetching LPIS pasture polygons...")
    lpis_gdf = None
    try:
        import geopandas as gpd
        from shapely.geometry import box as shapely_box

        lpis_gdf = fetch_lpis_polygons(bbox=coords, agoslag="Bete")
        print(f"    {len(lpis_gdf)} pasture polygons fetched")

        # Clip polygons to the showcase bounding box
        clip_box_wgs84 = shapely_box(
            coords["west"], coords["south"],
            coords["east"], coords["north"],
        )
        clip_gdf = gpd.GeoDataFrame(
            geometry=[clip_box_wgs84], crs="EPSG:4326",
        ).to_crs(lpis_gdf.crs)
        lpis_gdf = gpd.clip(lpis_gdf, clip_gdf)
        print(f"    {len(lpis_gdf)} polygons after bbox clip")

        # Load grazing model predictions
        pred_map = None
        pred_path = os.path.join(cache_dir, "grazing_results.json")
        if os.path.isfile(pred_path):
            with open(pred_path) as _f:
                pred_data = json.load(_f)
            pred_map = {}
            for p in pred_data.get("predictions", []):
                pred_map[str(p["polygon_id"])] = SimpleNamespace(
                    predicted_class=p["predicted_class"],
                    class_label=p["class_label"],
                    confidence=p["confidence"],
                )
            print(f"    Loaded {len(pred_map)} grazing model predictions")

        if len(lpis_gdf) > 0:
            save_lpis_overlay(
                rgb, lpis_gdf, geo,
                os.path.join(out_dir, "lpis_overlay.png"),
                predictions=pred_map,
            )
            save_lpis_geojson(
                lpis_gdf, geo,
                os.path.join(out_dir, "lpis_polygons.json"),
                img_shape=(H, W),
                predictions=pred_map,
            )
            n_pred = sum(1 for _, f in lpis_gdf.iterrows()
                         if pred_map and str(f.get("blockid", "")) in pred_map)
            print(f"    {len(lpis_gdf)} polygons overlaid ({n_pred} with predictions)")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"    LPIS overlay skipped: {e}")

    # ── Step 6: Compute statistics and save metadata ──────────────────
    print("\n[6] Computing statistics...")
    lpis_count = len(lpis_gdf) if lpis_gdf is not None else 0

    # Compute LPIS total area (ha) from clipped geometries in EPSG:3006
    lpis_total_area_ha = 0.0
    if lpis_gdf is not None and len(lpis_gdf) > 0:
        lpis_total_area_ha = lpis_gdf.geometry.area.sum() / 10_000  # m² → ha

    # NDVI mean within LPIS polygons
    ndvi_mean_inside = 0.0
    ndvi_std_inside = 0.0
    ndvi = indices.get("NDVI")
    if ndvi is not None and lpis_gdf is not None and len(lpis_gdf) > 0:
        try:
            from rasterio.features import rasterize
            from rasterio.transform import from_origin as _fo

            tf = geo.transform
            lpis_3006 = lpis_gdf.to_crs("EPSG:3006") if lpis_gdf.crs != "EPSG:3006" else lpis_gdf
            mask = rasterize(
                [(g, 1) for g in lpis_3006.geometry if g is not None and not g.is_empty],
                out_shape=(H, W),
                transform=tf,
                fill=0,
                dtype=np.uint8,
            )
            inside = ndvi[mask == 1]
            if len(inside) > 0:
                ndvi_mean_inside = float(np.nanmean(inside))
                ndvi_std_inside = float(np.nanstd(inside))
                print(f"    NDVI inside LPIS: {ndvi_mean_inside:.3f} +/- {ndvi_std_inside:.3f}")
        except Exception as e:
            print(f"    NDVI stats skipped: {e}")

    # NMD class distribution within LPIS
    nmd_within_lpis = {}
    if nmd_result is not None and lpis_gdf is not None and len(lpis_gdf) > 0:
        try:
            from rasterio.features import rasterize
            from imint.exporters.export import _NMD_L1_PALETTE

            lpis_3006 = lpis_gdf.to_crs("EPSG:3006") if lpis_gdf.crs != "EPSG:3006" else lpis_gdf
            mask = rasterize(
                [(g, 1) for g in lpis_3006.geometry if g is not None and not g.is_empty],
                out_shape=(H, W),
                transform=geo.transform,
                fill=0,
                dtype=np.uint8,
            )
            nmd = nmd_result.nmd_raster
            inside = nmd[mask == 1]
            total = len(inside)
            if total > 0:
                for code, info in _NMD_L1_PALETTE.items():
                    count = int((inside == code).sum())
                    if count > 0:
                        nmd_within_lpis[info["name"]] = {
                            "fraction": round(count / total, 4),
                            "pixel_count": count,
                        }
                # Sort by fraction descending
                nmd_within_lpis = dict(
                    sorted(nmd_within_lpis.items(),
                           key=lambda x: x[1]["fraction"], reverse=True)
                )
        except Exception as e:
            print(f"    NMD within LPIS skipped: {e}")

    # Grazing prediction summary
    grazing_summary = {}
    if pred_map:
        n_active = sum(1 for p in pred_map.values() if p.predicted_class == 1)
        n_inactive = sum(1 for p in pred_map.values() if p.predicted_class == 0)
        n_error = sum(1 for p in pred_map.values() if p.predicted_class == -1)
        avg_conf = np.mean([p.confidence for p in pred_map.values()
                            if p.predicted_class >= 0]) if pred_map else 0
        grazing_summary = {
            "total_polygons": len(pred_map),
            "active_grazing": n_active,
            "no_activity": n_inactive,
            "errors": n_error,
            "mean_confidence": round(float(avg_conf), 3),
            "model": "pib-ml-grazing CNN-biLSTM (RISE, 2025)",
            "year": args.year,
            "num_dates": T,
        }
        print(f"    Grazing: {n_active} active, {n_inactive} inactive, {n_error} errors")

    grazing_meta = {
        "date": date,
        "coords": coords,
        "shape": [H, W],
        "cloud_fraction": cloud_frac,
        "lpis_count": lpis_count,
        "lpis_total_area_ha": round(lpis_total_area_ha, 1),
        "ndvi_mean_inside": round(ndvi_mean_inside, 4),
        "ndvi_std_inside": round(ndvi_std_inside, 4),
        "nmd_within_lpis": nmd_within_lpis,
        "grazing_predictions": grazing_summary,
    }
    with open(os.path.join(out_dir, "grazing_meta.json"), "w") as f:
        json.dump(grazing_meta, f, indent=2, ensure_ascii=False)
    print(f"    Saved grazing_meta.json")

    print(f"\n{'=' * 60}")
    print(f"  Done! Images saved to {out_dir}")
    files = [f for f in os.listdir(out_dir) if f.endswith(".png")]
    print(f"  PNGs: {', '.join(sorted(files))}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import functools
    print = functools.partial(print, flush=True)
    main()
