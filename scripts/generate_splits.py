#!/usr/bin/env python3
"""
scripts/generate_splits.py — Generate train/val/test splits and class statistics

Standalone script that generates latitude-based splits and per-class pixel
statistics for an existing tile directory. Useful when tiles were fetched
via submit_s2_jobs.py (which doesn't write splits).

Usage:
    python scripts/generate_splits.py --data-dir data/lulc_seasonal
    python scripts/generate_splits.py --data-dir data/lulc_seasonal --num-classes 19
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imint.training.sampler import GridCell, grid_to_wgs84, split_by_latitude
from imint.training.class_schema import _LUT_19_TO_10

_TILE_RE = re.compile(r"tile_(\d+)_(\d+)\.npz")


def main():
    parser = argparse.ArgumentParser(
        description="Generate train/val/test splits and class statistics",
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Data directory (contains tiles/ subfolder)",
    )
    parser.add_argument(
        "--num-classes", type=int, default=10,
        help="Number of classes: 10 (grouped, default) or 19 (full NMD L2)",
    )
    parser.add_argument(
        "--patch-size-m", type=int, default=2560,
        help="Tile size in meters (default: 2560 = 256px × 10m)",
    )
    parser.add_argument(
        "--val-lat-min", type=float, default=64.0,
        help="Val zone southern boundary (default: 64.0)",
    )
    parser.add_argument(
        "--val-lat-max", type=float, default=66.0,
        help="Val zone northern boundary (default: 66.0)",
    )
    parser.add_argument(
        "--test-lat-min", type=float, default=66.0,
        help="Test zone southern boundary (default: 66.0)",
    )
    parser.add_argument(
        "--rare-threshold", type=float, default=0.02,
        help="Classes below this pixel fraction are 'rare' (default: 0.02)",
    )
    parser.add_argument(
        "--max-weight", type=float, default=5.0,
        help="Maximum tile oversampling weight (default: 5.0)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tiles_dir = data_dir / "tiles"
    if not tiles_dir.is_dir():
        print(f"  ERROR: {tiles_dir} not found")
        sys.exit(1)

    remap_to_grouped = args.num_classes == 10
    ignore_index = 0
    half = args.patch_size_m // 2

    # ── Step 1: Parse tile coordinates ────────────────────────────────────
    print(f"\n  Scanning {tiles_dir}...")
    cells = []
    tile_files = sorted(tiles_dir.glob("tile_*.npz"))
    for f in tile_files:
        m = _TILE_RE.search(f.name)
        if m:
            e, n = int(m.group(1)), int(m.group(2))
            cells.append(GridCell(
                easting=e, northing=n,
                west_3006=e - half, east_3006=e + half,
                south_3006=n - half, north_3006=n + half,
            ))
    print(f"  Found {len(cells)} tiles")

    if not cells:
        print("  ERROR: No tiles found")
        sys.exit(1)

    # ── Step 2: Convert to WGS84 and split by latitude ───────────────────
    print("  Converting to WGS84...")
    cells = grid_to_wgs84(cells)

    train_cells, val_cells, test_cells = split_by_latitude(
        cells,
        val_lat_min=args.val_lat_min,
        val_lat_max=args.val_lat_max,
        test_lat_min=args.test_lat_min,
    )
    print(f"  Split: train={len(train_cells)}, val={len(val_cells)}, test={len(test_cells)}")

    # Build lookup: (easting, northing) → split name
    cell_to_split: dict[tuple[int, int], str] = {}
    for c in train_cells:
        cell_to_split[(c.easting, c.northing)] = "train"
    for c in val_cells:
        cell_to_split[(c.easting, c.northing)] = "val"
    for c in test_cells:
        cell_to_split[(c.easting, c.northing)] = "test"

    # ── Step 3: Scan labels → class counts + per-tile histograms ─────────
    print("  Computing class statistics...")
    t0 = time.time()
    class_counts: dict[int, int] = defaultdict(int)
    tile_histograms: dict[str, dict[int, int]] = {}
    split_tiles: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for i, f in enumerate(tile_files):
        m = _TILE_RE.search(f.name)
        if not m:
            continue
        e, n = int(m.group(1)), int(m.group(2))
        split_name = cell_to_split.get((e, n))
        if not split_name:
            continue

        split_tiles[split_name].append(f.name)

        # Load label only (memory efficient)
        try:
            with np.load(f, allow_pickle=True) as data:
                label = data["label"]
        except Exception as exc:
            print(f"  WARNING: failed to load {f.name}: {exc}")
            continue

        # Remap to grouped classes if labels are in 19-class schema
        # (auto-detect: if max label > num_classes, remap is needed)
        if remap_to_grouped and label.max() > args.num_classes:
            label = _LUT_19_TO_10[np.clip(label, 0, 19)]

        # Count pixels per class
        hist: dict[int, int] = {}
        unique, counts = np.unique(label, return_counts=True)
        for cls_idx, count in zip(unique, counts):
            cls_idx = int(cls_idx)
            hist[cls_idx] = int(count)
            class_counts[cls_idx] += int(count)

        tile_histograms[f.name] = hist

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(tile_files) - i - 1) / rate
            print(f"  [{i+1}/{len(tile_files)}] {rate:.0f} tiles/s, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"  Scanned {len(tile_histograms)} tiles in {elapsed:.1f}s")

    # ── Step 4: Compute tile weights for rare-class oversampling ─────────
    total_px = sum(v for k, v in class_counts.items() if k != ignore_index)
    rare_classes = set()
    if total_px > 0:
        for cls_idx in range(args.num_classes + 1):
            if cls_idx == ignore_index:
                continue
            frac = class_counts.get(cls_idx, 0) / total_px
            if frac < args.rare_threshold:
                rare_classes.add(cls_idx)

    tile_weights: dict[str, float] = {}
    for tile_name, hist in tile_histograms.items():
        t_px = sum(v for k, v in hist.items() if k != ignore_index)
        if t_px == 0 or not rare_classes:
            tile_weights[tile_name] = 1.0
            continue
        rare_px = sum(v for k, v in hist.items() if k in rare_classes)
        rarity_score = rare_px / t_px
        w = 1.0 + (args.max_weight - 1.0) * rarity_score
        tile_weights[tile_name] = round(w, 3)

    # ── Step 5: Write split files ────────────────────────────────────────
    for split_name, names in split_tiles.items():
        split_path = data_dir / f"split_{split_name}.txt"
        with open(split_path, "w") as f:
            f.write("\n".join(sorted(names)) + "\n")
        print(f"  {split_name}: {len(names)} tiles → {split_path}")

    # ── Step 6: Write class_stats.json ───────────────────────────────────
    stats = {
        "class_counts": {str(k): v for k, v in sorted(class_counts.items())},
        "tile_weights": tile_weights,
        "rare_classes": sorted(rare_classes),
        "total_tiles": len(tile_histograms),
        "num_classes": args.num_classes,
    }
    stats_path = data_dir / "class_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Class stats → {stats_path}")

    # Report
    if rare_classes:
        print(f"\n  Rare classes (<{args.rare_threshold:.0%} of pixels):")
        for cls_idx in sorted(rare_classes):
            frac = class_counts.get(cls_idx, 0) / max(total_px, 1)
            print(f"    class {cls_idx:2d}: {frac:.4%}")
        n_boosted = sum(1 for w in tile_weights.values() if w > 1.01)
        print(f"  Tiles with boosted weight: {n_boosted}/{len(tile_weights)}")

    print(f"\n  Done! Ready for training with --data-dir {data_dir}")


if __name__ == "__main__":
    main()
