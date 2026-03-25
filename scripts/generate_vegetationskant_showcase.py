"""Generate showcase images for the Vegetationskant (Vegetation Edge) tab.

Pipeline:
  1. DES STAC — discover best cloud-free summer date per year (metadata only)
  2. CDSE — fetch single-date Sentinel-2 bands for each selected date
  3. Compute NDVI, NDWI, EVI from bands
  4. Run VegetationEdgeAnalyzer (NDVI threshold + edge extraction) per year
  5. Save PNGs + GeoJSON to outputs/showcase/vegetationskant/

Target area: southern shore of Lake Vanern near Lidkoping/Lacko.

Usage:
    .venv/bin/python scripts/generate_vegetationskant_showcase.py
    .venv/bin/python scripts/generate_vegetationskant_showcase.py --workers 5

References:
    Muir et al. (2024) VedgeSat
    Nugraha et al. (2026) Tropical VedgeSat validation
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

YEARS = list(range(2018, 2026))


def _save_vegetation_seg_png(seg_map: np.ndarray, path: str) -> str:
    """Save 3-class vegetation segmentation as coloured PNG."""
    from PIL import Image

    colors = {
        0: (0.12, 0.47, 0.71),   # water
        1: (0.85, 0.78, 0.60),   # non-vegetated
        2: (0.17, 0.63, 0.17),   # vegetated
    }
    h, w = seg_map.shape
    out = np.zeros((h, w, 3), dtype=np.float32)
    for cls, color in colors.items():
        mask = seg_map == cls
        for c in range(3):
            out[:, :, c][mask] = color[c]

    img = (out * 255).astype(np.uint8)
    Image.fromarray(img).save(path)
    print(f"    saved: {path}")
    return path


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate vegetationskant (vegetation edge) showcase images"
    )
    parser.add_argument(
        "--cloud-threshold", type=float, default=0.05,
        help="Max cloud fraction to accept a date (default 0.05)",
    )
    parser.add_argument(
        "--threshold-method", choices=["otsu", "weighted_peaks", "fixed"],
        default="otsu",
        help="NDVI threshold method (default: otsu)",
    )
    parser.add_argument(
        "--ndvi-threshold", type=float, default=None,
        help="Fixed NDVI threshold (only with --threshold-method fixed)",
    )
    parser.add_argument(
        "--workers", type=int, default=3,
        help="Number of parallel CDSE fetch workers (default: 3)",
    )
    args = parser.parse_args()

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from imint.fetch import (
        _stac_available_dates,
        fetch_sentinel2_data,
        GeoContext,
    )
    from imint.analyzers.spectral import SpectralAnalyzer
    from imint.analyzers.vegetation_edge import VegetationEdgeAnalyzer
    from imint.analyzers.nmd import NMDAnalyzer
    from imint.exporters.export import (
        save_rgb_png,
        save_ndvi_clean_png,
        save_spectral_index_clean_png,
        save_shoreline_overlay,
        save_shoreline_change_png,
        save_coastline_geojson,
        save_nmd_overlay,
    )

    # -- Area: Lake Vanern southern shore near Lidkoping/Lacko --------
    coords = {
        "west": 13.15, "south": 58.50,
        "east": 13.35, "north": 58.60,
    }

    out_dir = str(PROJECT_ROOT / "outputs" / "showcase" / "vegetationskant")
    cache_dir = str(PROJECT_ROOT / "outputs" / "vegetationskant_cache")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    print("=" * 60)
    print("  Vegetationskant Showcase Image Generator")
    print(f"  Area: Vanern — {coords['west']:.3f}–{coords['east']:.3f}°E, "
          f"{coords['south']:.3f}–{coords['north']:.3f}°N")
    print(f"  Years: {YEARS[0]}–{YEARS[-1]}")
    print(f"  Dates: DES STAC  |  Bands: CDSE ({args.workers} workers)")
    print(f"  Threshold: {args.threshold_method}")
    print(f"  Output: {out_dir}")
    print("=" * 60)

    # ── Step 1: DES STAC — find best summer date per year ────────────
    print("\n[1] Discovering dates via DES STAC...")
    yearly_dates: dict[int, str] = {}

    for year in YEARS:
        date_start = f"{year}-05-15"
        date_end = f"{year}-08-31"
        try:
            stac_dates = _stac_available_dates(
                coords, date_start, date_end,
                scene_cloud_max=50.0,
            )
        except Exception as e:
            print(f"    {year}: STAC failed — {e}")
            continue

        if not stac_dates:
            print(f"    {year}: no dates found")
            continue

        # Pick best: lowest cloud, prefer Jun-Aug
        best_date, best_cloud = stac_dates[0]  # already sorted by cloud
        for d, cf in stac_dates:
            month = int(d[5:7])
            if 6 <= month <= 8 and cf < 15.0:
                best_date, best_cloud = d, cf
                break

        yearly_dates[year] = best_date
        print(f"    {year}: {best_date} (scene cloud {best_cloud:.1f}%, "
              f"{len(stac_dates)} candidates)")

    if not yearly_dates:
        print("ERROR: No dates found for any year. Aborting.")
        return

    # ── Step 2: CDSE — fetch bands for each date (parallel) ─────────
    print(f"\n[2] Fetching {len(yearly_dates)} dates via CDSE "
          f"({args.workers} workers)...")

    yearly_results: dict[int, dict] = {}  # year -> {bands, rgb, geo, ...}
    geo = None

    def _fetch_year(year_date):
        yr, date = year_date
        cache_npz = os.path.join(cache_dir, f"scene_{yr}_{date}.npz")
        cache_json = os.path.join(cache_dir, f"scene_{yr}_{date}_meta.json")

        # Check cache
        if os.path.isfile(cache_npz) and os.path.isfile(cache_json):
            with open(cache_json) as f:
                meta = json.load(f)
            if meta.get("cloud_fraction", 0) > 0.05:
                print(f"    {yr}: {date} cached but cloudy "
                      f"({meta['cloud_fraction']:.1%}), skip", flush=True)
                return None
            if meta.get("nodata_fraction", 0) > 0.01:
                print(f"    {yr}: {date} cached but partial "
                      f"({meta['nodata_fraction']:.1%}), skip", flush=True)
                return None
            print(f"    {yr}: Loading {date} from cache", flush=True)
            cached = np.load(cache_npz)
            bands = {name: cached[name] for name in meta["band_names"]}
            rgb = cached["rgb"]
            return yr, date, bands, rgb, meta

        print(f"    {yr}: Fetching {date} from CDSE...", flush=True)
        result = fetch_sentinel2_data(
            source="copernicus",
            date=date,
            coords=coords,
            cloud_threshold=0.05,  # 5% SCL-based cloud filter
            date_window=3,
        )

        # Reject partial coverage (SCL=0 is nodata)
        if result.scl is not None:
            nodata_frac = float((result.scl == 0).sum()) / result.scl.size
            if nodata_frac > 0.01:  # >1% nodata = partial tile
                print(f"    {yr}: {date} rejected — partial coverage "
                      f"({nodata_frac:.1%} nodata)", flush=True)
                return None

        if result.cloud_fraction > 0.05:
            print(f"    {yr}: {date} rejected — SCL cloud "
                  f"{result.cloud_fraction:.1%} > 5%", flush=True)
            return None
        bands = result.bands
        rgb = result.rgb

        # Cache bands + rgb
        save_dict = {name: arr for name, arr in bands.items()}
        save_dict["rgb"] = rgb
        np.savez_compressed(cache_npz, **save_dict)

        nodata_frac = float((result.scl == 0).sum()) / result.scl.size if result.scl is not None else 0.0
        meta = {
            "date": date,
            "year": yr,
            "band_names": list(bands.keys()),
            "cloud_fraction": result.cloud_fraction,
            "nodata_fraction": nodata_frac,
            "shape": list(rgb.shape[:2]),
        }
        if result.geo:
            meta["geo"] = {
                "crs": str(result.geo.crs),
                "bounds_projected": result.geo.bounds_projected,
                "bounds_wgs84": result.geo.bounds_wgs84,
                "transform": list(result.geo.transform)[:6],
            }
        with open(cache_json, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"    {yr}: {date} OK — cloud={result.cloud_fraction:.1%}, "
              f"shape={rgb.shape[:2]}", flush=True)
        return yr, date, bands, rgb, meta

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_fetch_year, (yr, d)): yr
            for yr, d in sorted(yearly_dates.items())
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                if result is None:
                    continue  # rejected by cloud filter
                yr, date, bands, rgb, meta = result
                yearly_results[yr] = {
                    "bands": bands, "rgb": rgb, "meta": meta,
                }
                # Build GeoContext from first result
                if geo is None and "geo" in meta:
                    from rasterio.transform import Affine
                    g = meta["geo"]
                    t = g["transform"]
                    geo = GeoContext(
                        crs=g["crs"],
                        transform=Affine(t[0], t[1], t[2], t[3], t[4], t[5]),
                        bounds_projected=g.get("bounds_projected", {}),
                        bounds_wgs84=g.get("bounds_wgs84"),
                        shape=tuple(meta["shape"]),
                    )
            except Exception as e:
                yr = futures[future]
                print(f"    {yr}: FAILED — {e}", flush=True)

    if not yearly_results:
        print("ERROR: No data fetched. Aborting.")
        return

    print(f"    Fetched {len(yearly_results)} years successfully")

    # Build convenience dicts
    yearly_rgbs = {yr: r["rgb"] for yr, r in yearly_results.items()}
    yearly_bands = {yr: r["bands"] for yr, r in yearly_results.items()}
    yearly_date_str = {yr: yearly_dates[yr] for yr in yearly_results}

    ref_year = max(yearly_results.keys())
    ref_rgb = yearly_rgbs[ref_year]
    ref_bands = yearly_bands[ref_year]
    ref_date = yearly_date_str[ref_year]
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

    # ── Step 5: NMD land cover ──────────────────────────────────────
    print("\n[5] Running NMD analysis...")
    try:
        nmd = NMDAnalyzer()
        nmd_result = nmd.run(
            ref_rgb, bands=ref_bands, date=ref_date,
            coords=coords, output_dir=out_dir, geo=geo,
        )
        if nmd_result.success and nmd_result.outputs.get("nmd_available"):
            l2_raster = nmd_result.outputs.get("l2_raster")
            if l2_raster is not None:
                save_nmd_overlay(l2_raster, os.path.join(out_dir, "nmd_overlay.png"))
    except Exception as e:
        print(f"    NMD skipped: {e}")

    # ── Step 6: Vegetation edge detection (all years) ────────────────
    print("\n[6] Running NDVI vegetation edge detection...")

    ve_config = {}
    if args.threshold_method == "weighted_peaks":
        ve_config["weighted_peaks"] = True
    elif args.threshold_method == "fixed" and args.ndvi_threshold is not None:
        ve_config["ndvi_threshold"] = args.ndvi_threshold

    analyzer = VegetationEdgeAnalyzer(config=ve_config)

    yearly_seg = {}
    yearly_info = {}
    yearly_edges = {}
    yearly_contours = {}

    for year in sorted(yearly_bands.keys()):
        print(f"    {year}...", end=" ", flush=True)
        seg_map, info = analyzer.classify(yearly_bands[year], yearly_rgbs[year])
        edge = analyzer.extract_vegetation_edge(seg_map)
        contours = analyzer.extract_contours(edge, min_length=10)

        yearly_seg[year] = seg_map
        yearly_info[year] = info
        yearly_edges[year] = edge
        yearly_contours[year] = contours

        print(f"veg={info['vegetation_fraction']:.1%}, "
              f"{len(contours)} contours, "
              f"NDVI thresh={info['ndvi_threshold']:.3f} "
              f"({info['threshold_method']})")

    # Save outputs
    _save_vegetation_seg_png(
        yearly_seg[ref_year],
        os.path.join(out_dir, "vegetation_seg.png"),
    )
    save_shoreline_overlay(
        ref_rgb, yearly_edges[ref_year],
        os.path.join(out_dir, "vegetation_edge_overlay.png"),
        color=(0.13, 0.87, 0.27),
    )
    save_shoreline_change_png(
        yearly_edges, ref_rgb,
        os.path.join(out_dir, "vegetation_edge_change.png"),
    )
    if geo is not None:
        save_coastline_geojson(
            yearly_contours, geo,
            os.path.join(out_dir, "vegetation_edge_vectors.json"),
            img_shape=(H, W),
            pixel_coords=True,
            smooth_sigma=3.0,
            subsample_step=3,
        )

    # ── Step 7: Multitemporal spectral stability ──────────────────────
    print("\n[7] Computing multitemporal spectral stability...")
    from imint.analyzers.change_detection import CHANGE_BANDS, _build_stack
    from imint.exporters.export import save_change_gradient_png

    sorted_years = sorted(yearly_bands.keys())
    baseline_year = sorted_years[0]
    baseline_stack, _ = _build_stack(
        yearly_rgbs[baseline_year], yearly_bands[baseline_year],
    )

    # Compute spectral distance (L2 norm) from baseline for each year
    yearly_diffs = {}
    for year in sorted_years[1:]:
        current_stack, _ = _build_stack(yearly_rgbs[year], yearly_bands[year])
        # Crop to common shape if needed
        min_h = min(baseline_stack.shape[0], current_stack.shape[0])
        min_w = min(baseline_stack.shape[1], current_stack.shape[1])
        diff = np.linalg.norm(
            current_stack[:min_h, :min_w].astype(np.float32)
            - baseline_stack[:min_h, :min_w].astype(np.float32),
            axis=-1,
        )
        yearly_diffs[year] = diff
        mean_change = float(np.mean(diff))
        print(f"    {baseline_year}→{year}: mean spectral change = {mean_change:.4f}")

    # Mean stability map (average change magnitude across all years)
    if yearly_diffs:
        diff_stack = np.stack(list(yearly_diffs.values()), axis=0)
        stability_map = np.mean(diff_stack, axis=0)  # (H, W) mean L2 dist
        save_change_gradient_png(
            stability_map,
            os.path.join(out_dir, "spectral_stability.png"),
        )

        # Max change map (worst-case change across any year)
        max_change = np.max(diff_stack, axis=0)
        save_change_gradient_png(
            max_change,
            os.path.join(out_dir, "spectral_max_change.png"),
        )

        print(f"    Mean stability: {float(np.mean(stability_map)):.4f}")
        print(f"    Max change pixel: {float(np.max(max_change)):.4f}")

    # ── Step 8: Statistics + metadata ────────────────────────────────
    print("\n[8] Computing statistics...")

    per_year_stats = {}
    for year in sorted(yearly_seg.keys()):
        info = yearly_info[year]
        per_year_stats[year] = {
            "date": yearly_date_str[year],
            "water_fraction": round(info["water_fraction"], 4),
            "non_vegetated_fraction": round(info["non_vegetated_fraction"], 4),
            "vegetation_fraction": round(info["vegetation_fraction"], 4),
            "ndvi_threshold": info["ndvi_threshold"],
            "threshold_method": info["threshold_method"],
            "n_contours": len(yearly_contours[year]),
        }
        print(f"    {year}: veg={per_year_stats[year]['vegetation_fraction']:.1%}, "
              f"NDVI thresh={info['ndvi_threshold']:.3f}")

    ndvi_mean = float(np.nanmean(indices["NDVI"])) if "NDVI" in indices else 0.0

    vegetationskant_meta = {
        "date": ref_date,
        "reference_year": ref_year,
        "coords": coords,
        "area_name": "Vanern — Lidkoping/Lacko",
        "shape": [H, W],
        "years_analyzed": sorted(yearly_seg.keys()),
        "num_years": len(yearly_seg),
        "method": "NDVI + Otsu/Weighted Peaks (VedgeSat approach)",
        "reference": "Muir et al. (2024) VedgeSat; Nugraha et al. (2026)",
        "threshold_method": args.threshold_method,
        "fetch_pipeline": "DES STAC (dates) + CDSE (bands)",
        "per_year": per_year_stats,
        "ndvi_mean": round(ndvi_mean, 4),
    }

    with open(os.path.join(out_dir, "vegetationskant_meta.json"), "w") as f:
        json.dump(vegetationskant_meta, f, indent=2, ensure_ascii=False)
    print("    Saved vegetationskant_meta.json")

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
