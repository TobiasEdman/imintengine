#!/usr/bin/env python3
"""Compute mean/std normalization statistics for auxiliary channels.

Scans all training tiles and computes per-channel mean and std
from non-zero pixels (zero = nodata for Skogsstyrelsen rasters).

Usage:
    python scripts/compute_aux_stats.py --data-dir data/lulc_full

Output: Prints aux_norm dict ready to paste into config.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

AUX_CHANNELS = [
    "height", "volume", "basal_area", "diameter", "dem",
    "vpp_sosd", "vpp_eosd", "vpp_length", "vpp_maxv", "vpp_minv",
]


def main():
    parser = argparse.ArgumentParser(
        description="Compute aux channel normalization statistics",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/lulc_full",
        help="Training data directory (default: data/lulc_full)",
    )
    args = parser.parse_args()

    tiles_dir = Path(args.data_dir) / "tiles"
    if not tiles_dir.exists():
        print(f"ERROR: Tiles directory not found: {tiles_dir}")
        sys.exit(1)

    all_tiles = sorted(tiles_dir.glob("tile_*.npz"))
    print(f"Found {len(all_tiles)} tiles in {tiles_dir}")

    # Accumulators for Welford's online algorithm
    stats: dict[str, dict] = {}
    for ch in AUX_CHANNELS:
        stats[ch] = {
            "sum": 0.0,
            "sq_sum": 0.0,
            "n": 0,
            "tiles_with": 0,
            "min": float("inf"),
            "max": float("-inf"),
        }

    for i, tile_path in enumerate(all_tiles):
        try:
            with np.load(tile_path, allow_pickle=True) as d:
                for ch in AUX_CHANNELS:
                    if ch in d:
                        arr = d[ch].astype(np.float64)
                        # Use non-zero pixels only (0 = nodata)
                        mask = arr > 0
                        if mask.any():
                            vals = arr[mask]
                            stats[ch]["sum"] += vals.sum()
                            stats[ch]["sq_sum"] += (vals ** 2).sum()
                            stats[ch]["n"] += vals.size
                            stats[ch]["min"] = min(stats[ch]["min"], vals.min())
                            stats[ch]["max"] = max(stats[ch]["max"], vals.max())
                        stats[ch]["tiles_with"] += 1
        except Exception as e:
            print(f"  skip {tile_path.name}: {e}")

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(all_tiles)}]...")

    # Compute final statistics
    print(f"\n{'='*60}")
    print(f"  Auxiliary Channel Statistics ({len(all_tiles)} tiles)")
    print(f"{'='*60}\n")

    norm_dict = {}
    for ch in AUX_CHANNELS:
        s = stats[ch]
        if s["n"] > 0:
            mean = s["sum"] / s["n"]
            variance = s["sq_sum"] / s["n"] - mean ** 2
            std = max(variance ** 0.5, 1e-6)
            norm_dict[ch] = (round(mean, 2), round(std, 2))

            print(f"  {ch}:")
            print(f"    tiles:   {s['tiles_with']}")
            print(f"    pixels:  {s['n']:,} (non-zero)")
            print(f"    mean:    {mean:.2f}")
            print(f"    std:     {std:.2f}")
            print(f"    min:     {s['min']:.2f}")
            print(f"    max:     {s['max']:.2f}")
            print()
        else:
            print(f"  {ch}: no data found\n")

    # Print config-ready dict
    print(f"{'='*60}")
    print("  Config-ready aux_norm dict:")
    print(f"{'='*60}")
    print()
    print("    aux_norm: dict = field(default_factory=lambda: {")
    for ch in AUX_CHANNELS:
        if ch in norm_dict:
            mean, std = norm_dict[ch]
            print(f'        "{ch}": ({mean}, {std}),')
    print("    })")


if __name__ == "__main__":
    main()
