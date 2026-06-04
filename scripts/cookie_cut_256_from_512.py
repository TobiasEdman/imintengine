#!/usr/bin/env python3
"""Derive 256-tiles from freshly-refetched 512-tiles via center crop.

For each tile name present in both /cephfs/unified_v2_512 and the
256-dataset dir (e.g. /td/unified_v2), reads the 512 tile's spatial
arrays, extracts the center 256×256 (or 256 px equivalent per the
tile's actual dimensions), and writes the 256 version atomically.

Designed to avoid re-fetching the same Sentinel-2 scenes twice: the
512-set is fetched first (with correct VPP windows), and 256-tiles
that share a center are derived locally.

Usage (k8s pod):
    python scripts/cookie_cut_256_from_512.py \\
        --tiles-json /checkpoints/audits/overlap_512_in_256.json \\
        --src-dir /cephfs/unified_v2_512 \\
        --dst-dir /td/unified_v2 \\
        --workers 4

The tiles-json file is a JSON array of tile basenames (with or
without .npz suffix). The 256-PVC is mounted RW so the destination
.npz can be atomically replaced.

Per-tile workflow:
  1. Read src .npz (512-version, freshly refetched)
  2. Read dst .npz (existing 256-version, for non-spatial metadata)
  3. Crop spatial arrays from 512 → 256 center slice
  4. Copy temporal metadata (dates/doy/temporal_mask) from src — it
     reflects the new fetch
  5. Copy 256-specific metadata from dst (bbox_3006 differs! it's
     2560m on 256 vs 5120m on 512, even at the same center)
  6. Atomic write to dst path
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


# Keys that are spatial (need cropping) — both H and W dims
SPATIAL_2D_KEYS = (
    "label", "label_mask", "nmd_label_raw",
    "dem", "height", "volume", "basal_area", "diameter",
    "harvest_mask",
    "vpp_sosd", "vpp_eosd", "vpp_length", "vpp_maxv", "vpp_minv",
    "nmd_area_ha", "parcel_area_ha",
)
# Keys that are (T*C, H, W) or (T, H, W) — crop last 2 dims
SPATIAL_3D_KEYS = (
    "image", "b08", "s1_vv_vh", "frame_2016",
)
# Keys that take their value from the SRC (512) — they reflect the
# new fetch's temporal coverage
SRC_OVERRIDES = (
    "dates", "doy", "temporal_mask", "multitemporal", "num_frames",
    "num_bands",
    "s1_temporal_mask", "s1_dates", "has_s1",
    "has_b08",
    "frame_2016_date", "frame_2016_doy", "frame_2016_cloud_pct",
    "frame_2016_year", "has_frame_2016",
)


def center_crop_2d(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Center-crop a 2D-or-higher array's last two dims."""
    if arr.ndim < 2:
        return arr
    h, w = arr.shape[-2], arr.shape[-1]
    y0 = (h - target_h) // 2
    x0 = (w - target_w) // 2
    if y0 < 0 or x0 < 0:
        raise ValueError(
            f"src smaller than target: src=({h},{w}) target=({target_h},{target_w})"
        )
    sl = (slice(None),) * (arr.ndim - 2) + (
        slice(y0, y0 + target_h), slice(x0, x0 + target_w),
    )
    return arr[sl]


def cookie_cut_one(name: str, src_dir: str, dst_dir: str) -> dict:
    """Cookie-cut one 256-tile from its 512 parent."""
    name = name.replace(".npz", "")
    src_path = os.path.join(src_dir, f"{name}.npz")
    dst_path = os.path.join(dst_dir, f"{name}.npz")

    if not os.path.exists(src_path):
        return {"name": name, "status": "skip_no_src"}
    if not os.path.exists(dst_path):
        return {"name": name, "status": "skip_no_dst"}

    try:
        src = dict(np.load(src_path, allow_pickle=True))
        dst = dict(np.load(dst_path, allow_pickle=True))
    except Exception as e:
        return {"name": name, "status": "load_error", "reason": str(e)[:200]}

    # Determine target spatial size from dst's label
    if "label" not in dst:
        return {"name": name, "status": "skip_no_dst_label"}
    target_h, target_w = dst["label"].shape[-2], dst["label"].shape[-1]

    # Build output: start from dst (keeps dst-only metadata), then
    # override with cropped src spatial arrays + src temporal metadata.
    out = dict(dst)

    cropped = 0
    for k in SPATIAL_2D_KEYS + SPATIAL_3D_KEYS:
        if k not in src:
            continue
        try:
            out[k] = center_crop_2d(src[k], target_h, target_w)
            cropped += 1
        except Exception as e:
            return {"name": name, "status": "crop_error", "key": k, "reason": str(e)[:200]}

    overridden = 0
    for k in SRC_OVERRIDES:
        if k in src:
            out[k] = src[k]
            overridden += 1

    # Atomic write: tmp + os.replace
    tmp_base = dst_path + ".cookie.tmp"
    np.savez_compressed(tmp_base, **out)
    tmp_actual = tmp_base + ".npz"
    if not os.path.exists(tmp_actual):
        return {"name": name, "status": "tmp_missing"}
    os.replace(tmp_actual, dst_path)
    return {"name": name, "status": "ok", "cropped": cropped, "overridden": overridden}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tiles-json", required=True,
                   help="JSON array of tile basenames OR object with 'tiles' key")
    p.add_argument("--src-dir", required=True,
                   help="Source dir (512-set, e.g. /cephfs/unified_v2_512)")
    p.add_argument("--dst-dir", required=True,
                   help="Destination dir (256-set, e.g. /td/unified_v2)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-tiles", type=int, default=None)
    args = p.parse_args()

    print(f"=== cookie_cut_256_from_512 ===", flush=True)
    print(f"  tiles-json: {args.tiles_json}", flush=True)
    print(f"  src-dir:    {args.src_dir}", flush=True)
    print(f"  dst-dir:    {args.dst_dir}", flush=True)
    print(f"  workers:    {args.workers}", flush=True)

    with open(args.tiles_json) as f:
        data = json.load(f)
    if isinstance(data, dict):
        tiles = data.get("tiles") or data.get("unique_affected_tiles", [])
    else:
        tiles = data
    if args.max_tiles:
        tiles = tiles[: args.max_tiles]
    print(f"  tiles: {len(tiles)}", flush=True)

    t0 = time.time()
    by_status: dict[str, int] = {}
    completed = 0
    results = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(cookie_cut_one, n, args.src_dir, args.dst_dir): n
                for n in tiles}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            by_status[r["status"]] = by_status.get(r["status"], 0) + 1
            completed += 1
            if completed % 50 == 0 or completed == len(tiles):
                elapsed = time.time() - t0
                rate = completed / max(elapsed, 1e-6)
                eta = (len(tiles) - completed) / max(rate, 1e-6)
                print(
                    f"  [{completed}/{len(tiles)}] "
                    f"status={by_status} "
                    f"rate={rate*60:.0f}/min ETA={eta/60:.1f}min",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(f"\n=== done in {elapsed/60:.1f} min ===", flush=True)
    print(f"  status breakdown: {by_status}", flush=True)
    errors = [r for r in results if r["status"] not in ("ok", "skip_no_src", "skip_no_dst")]
    if errors:
        print(f"\n  first 10 errors:", flush=True)
        for r in errors[:10]:
            print(f"    {r['name']}: {r}", flush=True)


if __name__ == "__main__":
    main()
