#!/usr/bin/env python3
"""Post-processing QC: nodata filter + frame check + class stats.

Run AFTER remap_labels.py. Removes bad tiles and generates class_stats.json.

Usage:
    python scripts/postprocess_qc.py --data-dir /data/unified_v2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imint.training.unified_schema import UNIFIED_CLASS_NAMES


def main():
    p = argparse.ArgumentParser(description="QC: filter nodata, check frames, class stats")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--nodata-threshold", type=float, default=0.05,
                   help="Max fraction of zero pixels per frame (default 0.05)")
    p.add_argument("--min-valid-frames", type=int, default=3,
                   help="Min valid temporal frames (default 3 of 4)")
    args = p.parse_args()

    tiles = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    print(f"Scanning {len(tiles)} tiles...")

    removed_nodata = 0
    removed_frames = 0
    removed_corrupt = 0
    kept = 0
    pixel_counts: Counter = Counter()
    tile_dominant: Counter = Counter()
    tiles_with_crop = 0

    for i, f in enumerate(tiles):
        try:
            d = np.load(f, allow_pickle=True)
            img = d.get("spectral", d.get("image"))
            n_frames = img.shape[0] // 6

            # Nodata filter: >threshold zero pixels in any frame
            has_nodata = False
            for fi in range(n_frames):
                frame = img[fi * 6:(fi + 1) * 6]
                nodata_frac = float((frame.max(axis=0) == 0).mean())
                if nodata_frac > args.nodata_threshold:
                    has_nodata = True
                    break
            if has_nodata:
                os.remove(f)
                removed_nodata += 1
                continue

            # Frame check: need min valid frames
            tmask = d.get("temporal_mask", None)
            if tmask is not None and int(tmask.sum()) < args.min_valid_frames:
                os.remove(f)
                removed_frames += 1
                continue

            # Class stats
            label = d.get("label", None)
            if label is not None:
                vals, counts = np.unique(label.flatten(), return_counts=True)
                for v, c in zip(vals, counts):
                    if 0 <= v < len(UNIFIED_CLASS_NAMES):
                        pixel_counts[int(v)] += int(c)
                dominant = int(vals[np.argmax(counts)])
                if dominant < len(UNIFIED_CLASS_NAMES):
                    tile_dominant[dominant] += 1
                if any(11 <= v <= 18 for v in vals):
                    tiles_with_crop += 1
            kept += 1

            if (i + 1) % 500 == 0:
                print(f"  [{i+1}/{len(tiles)}] kept={kept} "
                      f"rm_nodata={removed_nodata} rm_frames={removed_frames}",
                      flush=True)

        except Exception:
            os.remove(f)
            removed_corrupt += 1

    print()
    print("=== Results ===")
    print(f"  Kept:                    {kept}")
    print(f"  Removed (nodata >5%):    {removed_nodata}")
    print(f"  Removed (<3 frames):     {removed_frames}")
    print(f"  Removed (corrupt):       {removed_corrupt}")
    print(f"  Tiles with crops:        {tiles_with_crop}")
    print()
    print("=== Class Distribution ===")
    total = sum(pixel_counts.values())
    for k in sorted(pixel_counts.keys()):
        name = UNIFIED_CLASS_NAMES[k] if k < len(UNIFIED_CLASS_NAMES) else f"class_{k}"
        pct = pixel_counts[k] / total * 100 if total > 0 else 0
        print(f"  {k:2d} {name:25s} {pct:5.1f}%  ({pixel_counts[k]:>10,} px)")

    # Save
    result = {
        "pixel_counts": {
            UNIFIED_CLASS_NAMES[k]: v
            for k, v in pixel_counts.items()
            if k < len(UNIFIED_CLASS_NAMES)
        },
        "tile_dominant_class": {
            UNIFIED_CLASS_NAMES[k]: v
            for k, v in tile_dominant.items()
            if k < len(UNIFIED_CLASS_NAMES)
        },
        "tiles_with_crop": tiles_with_crop,
        "tiles_kept": kept,
        "tiles_removed_nodata": removed_nodata,
        "tiles_removed_frames": removed_frames,
        "tiles_removed_corrupt": removed_corrupt,
        "class_names": list(UNIFIED_CLASS_NAMES),
    }
    out_path = os.path.join(args.data_dir, "class_stats.json")
    json.dump(result, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
