"""Generate showcase images for the Vattenkvalitet (Water Quality) tab.

Fetches Sentinel-2 timeseries for 2018–2026 using the same
``fetch_grazing_timeseries()`` pipeline as the other showcases (cloud
filtering via SCL, co-registration, NMD 10 m grid snapping), runs the
:class:`WaterQualityAnalyzer` per year on a spring-bloom date, and saves
PNGs to ``outputs/showcase/water_quality/<year>/`` (mirrored to
``docs/showcase/water_quality/<year>/`` so the dashboard serves them).

The 2026 entry is pinned to **2026-04-08**, the date of the inspiring
Kattegatt true-color scene; other years pick the cleanest April–June
date available.

Usage:
    .venv/bin/python scripts/generate_water_quality_showcase.py

DES credentials needed for Sentinel-2 fetching.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# Band order in cached timeseries (same as other showcases):
_BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
               "B08", "B8A", "B09", "B11", "B12"]

YEARS = list(range(2018, 2027))

# 2026-04-08 — the inspiring image. If the timeseries contains this date,
# prefer it for the 2026 frame regardless of cloud cover.
PINNED_DATE_BY_YEAR = {2026: "2026-04-08"}

# AOI v3 (2026-04-28): user-picked Marstrand + S Tjörn + nearshore Skagerrak.
# 25 × 11 km, aspect ~2.21:1 landscape in projected EPSG:3006.
COORDS = {
    "west": 11.3187, "south": 58.0416,
    "east": 11.7403, "north": 58.1424,
}


def _pick_best_spring_date(
    dates: list[str],
    cloud_fractions: list[float],
    year: int,
) -> int:
    """Pick the cleanest April–June date; honour the per-year pin if present."""
    pinned = PINNED_DATE_BY_YEAR.get(year)
    if pinned and pinned in dates:
        return dates.index(pinned)

    best_idx = 0
    best_score = -1.0
    for i, (d, cf) in enumerate(zip(dates, cloud_fractions)):
        month = int(d[5:7])
        # Spring bloom window peaks April–May in Skagerrak/Kattegatt
        season_bonus = 1.0 if 4 <= month <= 5 else 0.5 if month == 6 else 0.0
        score = (1.0 - cf) + season_bonus
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def _build_rgb(snapshot: np.ndarray) -> np.ndarray:
    """Build contrast-stretched RGB from a (12, H, W) band stack."""
    rgb = np.stack([snapshot[3], snapshot[2], snapshot[1]], axis=-1)  # B04, B03, B02
    p2, p98 = np.percentile(rgb, [2, 98])
    return np.clip((rgb - p2) / (p98 - p2 + 1e-8), 0.0, 1.0).astype(np.float32)


def _scl_from_snapshot(snapshot: np.ndarray) -> np.ndarray | None:
    """No SCL stored in the kustlinje cache format — return None.

    The analyzer falls back to MNDWI water masking. If a richer cache
    schema becomes available later, populate this from the appropriate
    band index.
    """
    return None


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate water quality (Vattenkvalitet) showcase images"
    )
    parser.add_argument(
        "--cloud-threshold", type=float, default=0.10,
        help="Max cloud fraction within bbox to keep a date (default 0.10)",
    )
    parser.add_argument(
        "--years", type=int, nargs="+", default=YEARS,
        help="Years to render (default 2018–2026)",
    )
    args = parser.parse_args()

    from shapely.geometry import box as shapely_box
    from rasterio.transform import from_origin
    from imint.fetch import fetch_grazing_timeseries, GeoContext
    from imint.analyzers.water_quality import WaterQualityAnalyzer
    from imint.exporters.export import (
        save_rgb_png,
        save_water_quality_png,
        save_water_mask_png,
    )

    out_root = PROJECT_ROOT / "outputs" / "showcase" / "water_quality"
    docs_root = PROJECT_ROOT / "docs" / "showcase" / "water_quality"
    cache_dir = PROJECT_ROOT / "outputs" / "water_quality_cache"
    out_root.mkdir(parents=True, exist_ok=True)
    docs_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    bbox_polygon = shapely_box(
        COORDS["west"], COORDS["south"], COORDS["east"], COORDS["north"]
    )

    print("=" * 60)
    print("  Vattenkvalitet (Water Quality) Showcase")
    print(f"  Area:  {COORDS['west']:.2f}–{COORDS['east']:.2f}°E, "
          f"{COORDS['south']:.2f}–{COORDS['north']:.2f}°N")
    print(f"  Years: {args.years[0]}–{args.years[-1]}")
    print(f"  Out:   {out_root}")
    print("=" * 60)

    # 1. Fetch / load S2 timeseries per year
    print("\n[1] Loading Sentinel-2 timeseries...")
    yearly_data: dict[int, np.ndarray] = {}
    yearly_meta: dict[int, dict] = {}
    geo: GeoContext | None = None

    for year in args.years:
        cache_npz = cache_dir / f"bbox_timeseries_{year}.npz"
        cache_json = cache_dir / f"bbox_timeseries_{year}_meta.json"

        if cache_npz.exists() and cache_json.exists():
            print(f"    {year}: cached")
            data = np.load(cache_npz)["data"]
            with open(cache_json) as f:
                meta = json.load(f)
        else:
            print(f"    {year}: fetching...")
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
                print(f"    {year}: no cloud-free dates")
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
            np.savez_compressed(cache_npz, data=data)
            with open(cache_json, "w") as f:
                json.dump(meta, f, indent=2)
            if geo is None:
                geo = ts.geo

        yearly_data[year] = data
        yearly_meta[year] = meta
        print(f"    {year}: {data.shape[0]} dates, shape {data.shape}")

    if not yearly_data:
        print("ERROR: no data fetched. Aborting.")
        return

    if geo is None:
        first_meta = next(iter(yearly_meta.values()))
        H, W = next(iter(yearly_data.values())).shape[2:]
        pixel_size = first_meta.get("pixel_size", 10)
        geo = GeoContext(
            crs="EPSG:3006",
            transform=from_origin(
                first_meta["west"], first_meta["north"], pixel_size, pixel_size,
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

    # 2. Per year: pick spring date, run analyzer, render PNGs
    analyzer = WaterQualityAnalyzer(config={
        "aoi_geojson": str(
            PROJECT_ROOT / "imint" / "analyzers" / "water_quality"
            / "aoi" / "stigfjorden_skagerrak.geojson"
        ),
        "methods": {
            # In the showcase environment, MDN/C2RCC will skip-and-warn unless
            # the executing host has weights + acolite installed. NDCI/MCI are
            # always available.
            "mdn": {"enabled": True, "weights_url": None},
            "c2rcc": {"enabled": True},
            "ndci": {"enabled": True},
            "mci": {"enabled": True},
        },
    })

    layer_specs = [
        ("ndci",                  {"cmap_name": "RdBu_r", "vmin": -0.2, "vmax": 0.6}),
        ("mci",                   {"cmap_name": "magma", "vmin": -0.02, "vmax": 0.08}),
        ("chlorophyll_a_mdn",     {"cmap_name": "viridis", "log_scale": True}),
        ("chlorophyll_a_uncertainty_mdn", {"cmap_name": "Greys"}),
        ("tss_mdn",               {"cmap_name": "cividis", "log_scale": True}),
        ("acdom_mdn",             {"cmap_name": "copper", "log_scale": True}),
        ("chlorophyll_a_c2rcc",   {"cmap_name": "viridis", "log_scale": True}),
        ("tsm_c2rcc",             {"cmap_name": "cividis", "log_scale": True}),
        ("cdom_c2rcc",            {"cmap_name": "copper", "log_scale": True}),
        ("chlorophyll_spread",    {"cmap_name": "magma"}),
    ]

    print("\n[2] Running analyzer + saving PNGs...")
    for year in sorted(yearly_data.keys()):
        data = yearly_data[year]
        meta = yearly_meta[year]
        idx = _pick_best_spring_date(meta["dates"], meta["cloud_fractions"], year)
        date = meta["dates"][idx]
        cloud = meta["cloud_fractions"][idx]
        snapshot = data[idx]
        bands = {name: snapshot[i] for i, name in enumerate(_BAND_NAMES)}
        rgb = _build_rgb(snapshot)
        scl = _scl_from_snapshot(snapshot)

        year_dir = out_root / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        print(f"    {year}: {date} (cloud {cloud:.1%}) → {year_dir}")

        save_rgb_png(rgb, str(year_dir / "rgb.png"))

        result = analyzer.run(
            rgb=rgb, bands=bands, scl=scl, geo=geo,
            date=date, output_dir=str(out_root.parent.parent / "water_quality_runs"),
        )
        if not result.success:
            print(f"      analyzer failed: {result.error}")
            continue

        for layer, kwargs in layer_specs:
            arr = result.outputs.get(layer)
            if arr is None:
                continue
            save_water_quality_png(arr, str(year_dir / f"{layer}.png"), **kwargs)

        wm = result.outputs.get("water_mask")
        if wm is not None:
            save_water_mask_png(wm, str(year_dir / "water_mask.png"))

        # Mirror to docs/showcase/water_quality/<year>/ for the dashboard
        docs_year_dir = docs_root / str(year)
        docs_year_dir.mkdir(parents=True, exist_ok=True)
        for png in year_dir.glob("*.png"):
            shutil.copy2(png, docs_year_dir / png.name)

    print("\nDone.")


if __name__ == "__main__":
    main()
