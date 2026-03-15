#!/usr/bin/env python3
"""
scripts/prepare_lulc_data.py — Fetch and cache LULC training data

Generates a geographic grid across Sweden, fetches Sentinel-2 + NMD
tile pairs from DES, and saves them as .npz files ready for training.

Supports two modes:
  * Single-date (default): One best cloud-free growing season image per tile.
  * Multitemporal (--multitemporal): T seasonal frames per tile for
    phenological time-series classification with Prithvi-EO-2.0.

Usage:
    python scripts/prepare_lulc_data.py
    python scripts/prepare_lulc_data.py --grid-spacing 50000   # Small test (~50 tiles)
    python scripts/prepare_lulc_data.py --data-dir /path/to/data --years 2017 2018

    # Multitemporal: 4 seasonal frames (spring/summer/autumn/winter)
    python scripts/prepare_lulc_data.py --multitemporal --data-dir data/lulc_mt

    # Custom seasonal windows (3 frames, growing season only)
    python scripts/prepare_lulc_data.py --multitemporal --num-frames 3 \\
        --seasonal-windows 5-5 7-7 9-9

    # With Skogsstyrelsen tree height as extra channel
    python scripts/prepare_lulc_data.py --enable-height
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imint.training.config import TrainingConfig
from imint.training.prepare_data import prepare_training_data


def main():
    parser = argparse.ArgumentParser(
        description="Fetch and cache LULC training data from DES",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/lulc_training",
        help="Directory for training data (default: data/lulc_training)",
    )
    parser.add_argument(
        "--grid-spacing", type=int, default=10_000,
        help="Grid spacing in meters (default: 10000, use 50000 for test)",
    )
    parser.add_argument(
        "--years", nargs="+", default=["2019", "2018"],
        help="Years to fetch data from (default: 2019 2018)",
    )
    parser.add_argument(
        "--num-classes", type=int, default=19,
        help="Number of classes: 19 (full NMD L2) or 10 (grouped)",
    )
    parser.add_argument(
        "--cloud-threshold", type=float, default=0.10,
        help="Max cloud fraction per tile (default: 0.10)",
    )

    # ── Grid densification flags ──────────────────────────────────────
    parser.add_argument(
        "--enable-densification", action="store_true",
        help="Enable grid densification in predefined rare-class areas",
    )
    parser.add_argument(
        "--enable-scb-densification", action="store_true",
        help="Enable SCB tätort urban densification (downloads ~60 MB)",
    )
    parser.add_argument(
        "--scb-min-population", type=int, default=2_000,
        help="Min tätort population for SCB densification (default: 2000)",
    )
    parser.add_argument(
        "--scb-densify-spacing", type=int, default=2_500,
        help="Grid spacing in SCB tätort regions in meters (default: 2500)",
    )
    parser.add_argument(
        "--enable-sea-densification", action="store_true",
        help="Enable coastal water densification (uses HaV EEZ data)",
    )
    parser.add_argument(
        "--max-sea-distance", type=int, default=5_000,
        help="Max distance from land for sea cells in meters (default: 5000)",
    )
    parser.add_argument(
        "--enable-sumpskog-densification", action="store_true",
        help="Enable Skogsstyrelsen sumpskog (swamp forest) densification",
    )
    parser.add_argument(
        "--sumpskog-min-density", type=float, default=5.0,
        help="Min sumpskog density %% per 25km cell for inclusion (default: 5.0)",
    )
    parser.add_argument(
        "--sumpskog-densify-spacing", type=int, default=10_000,
        help="Grid spacing in sumpskog regions in meters (default: 10000)",
    )

    # ── Multitemporal / seasonal flags ─────────────────────────────────
    parser.add_argument(
        "--multitemporal", action="store_true",
        help="Enable multitemporal fetching (4 seasonal frames per tile)",
    )
    parser.add_argument(
        "--num-frames", type=int, default=4,
        help="Number of temporal frames (default: 4, max for Prithvi)",
    )
    parser.add_argument(
        "--seasonal-windows", nargs="+", default=None,
        help="Seasonal windows as M1-M2 pairs, e.g. '4-5 6-7 8-9 1-2' "
             "(default: spring/summer/autumn/winter)",
    )
    parser.add_argument(
        "--seasonal-cloud-threshold", type=float, default=0.10,
        help="Cloud threshold for seasonal images (default: 0.10)",
    )
    parser.add_argument(
        "--require-all-seasons", action="store_true",
        help="Require all seasonal frames (skip tiles with missing seasons)",
    )

    # ── Auxiliary raster channels ─────────────────────────────────────
    parser.add_argument(
        "--enable-height", action="store_true",
        help="Fetch Skogsstyrelsen tree height as extra channel per tile",
    )
    parser.add_argument(
        "--enable-volume", action="store_true",
        help="Fetch Skogliga grunddata timber volume (m³sk/ha) per tile",
    )
    parser.add_argument(
        "--enable-basal-area", action="store_true",
        help="Fetch Skogliga grunddata basal area / grundyta (m²/ha) per tile",
    )
    parser.add_argument(
        "--enable-diameter", action="store_true",
        help="Fetch Skogliga grunddata mean diameter / medeldiameter (cm) per tile",
    )
    parser.add_argument(
        "--enable-dem", action="store_true",
        help="Fetch Copernicus DEM GLO-30 terrain elevation (meters) per tile",
    )

    # ── Fetch backend selection ───────────────────────────────────────
    parser.add_argument(
        "--fetch-sources", nargs="+", default=None,
        help="Data fetch backends: 'copernicus' (CDSE) and/or 'des' "
             "(default: copernicus des). Use 'copernicus' alone if no DES creds.",
    )

    args = parser.parse_args()

    # Parse seasonal windows from CLI (e.g. "4-5 6-7 8-9 1-2")
    seasonal_windows = None
    if args.seasonal_windows:
        seasonal_windows = []
        for w in args.seasonal_windows:
            parts = w.split("-")
            if len(parts) == 2:
                seasonal_windows.append((int(parts[0]), int(parts[1])))
            else:
                parser.error(f"Invalid seasonal window '{w}', use M1-M2 format")

    # Build config kwargs
    config_kwargs = dict(
        data_dir=args.data_dir,
        grid_spacing_m=args.grid_spacing,
        years=args.years,
        num_classes=args.num_classes,
        cloud_threshold=args.cloud_threshold,
        enable_grid_densification=args.enable_densification,
        enable_scb_densification=args.enable_scb_densification,
        scb_min_population=args.scb_min_population,
        scb_densify_spacing_m=args.scb_densify_spacing,
        enable_sea_densification=args.enable_sea_densification,
        max_sea_distance_m=args.max_sea_distance,
        enable_sumpskog_densification=args.enable_sumpskog_densification,
        sumpskog_min_density_pct=args.sumpskog_min_density,
        sumpskog_densify_spacing_m=args.sumpskog_densify_spacing,
        # Multitemporal
        enable_multitemporal=args.multitemporal,
        num_temporal_frames=args.num_frames,
        seasonal_cloud_threshold=args.seasonal_cloud_threshold,
        seasonal_require_all=args.require_all_seasons,
        # Auxiliary channels
        enable_height_channel=args.enable_height,
        enable_volume_channel=args.enable_volume,
        enable_basal_area_channel=args.enable_basal_area,
        enable_diameter_channel=args.enable_diameter,
        enable_dem_channel=args.enable_dem,
    )
    if seasonal_windows is not None:
        config_kwargs["seasonal_windows"] = seasonal_windows
    if args.fetch_sources is not None:
        config_kwargs["fetch_sources"] = args.fetch_sources

    config = TrainingConfig(**config_kwargs)

    prepare_training_data(config)


if __name__ == "__main__":
    main()
