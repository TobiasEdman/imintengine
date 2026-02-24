#!/usr/bin/env python3
"""
scripts/prepare_lulc_data.py — Fetch and cache LULC training data

Generates a geographic grid across Sweden, fetches Sentinel-2 + NMD
tile pairs from DES, and saves them as .npz files ready for training.

Usage:
    python scripts/prepare_lulc_data.py
    python scripts/prepare_lulc_data.py --grid-spacing 50000   # Small test (~50 tiles)
    python scripts/prepare_lulc_data.py --data-dir /path/to/data --years 2017 2018
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
        "--years", nargs="+", default=["2017", "2018"],
        help="Years to fetch data from (default: 2017 2018)",
    )
    parser.add_argument(
        "--num-classes", type=int, default=19,
        help="Number of classes: 19 (full NMD L2) or 10 (grouped)",
    )
    parser.add_argument(
        "--cloud-threshold", type=float, default=0.10,
        help="Max cloud fraction per tile (default: 0.10)",
    )

    args = parser.parse_args()

    config = TrainingConfig(
        data_dir=args.data_dir,
        grid_spacing_m=args.grid_spacing,
        years=args.years,
        num_classes=args.num_classes,
        cloud_threshold=args.cloud_threshold,
    )

    prepare_training_data(config)


if __name__ == "__main__":
    main()
