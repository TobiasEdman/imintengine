#!/usr/bin/env python3
"""Preprocess SKS + LPIS parquets into spatially-partitioned GeoParquets.

Why:
    Loading the full 3.5 GB utforda_avverkningar.parquet into memory eats
    ~20 GB after geometry deserialization. Multiply by 32 workers and OOM
    is guaranteed. This script sorts polygons by a coarse EPSG:3006 grid
    and writes small row groups so pyarrow's row-group statistics can
    filter to the handful of row groups that overlap a tile bbox.

What:
    For each input parquet, we:
      1. Load via geopandas (one-time cost, offline).
      2. Compute polygon bbox → bucket into a 100 km × 100 km grid cell
         (``grid_x = bbox_cx // 100_000``, etc).
      3. Sort by (grid_x, grid_y) so polygons in the same cell are
         contiguous.
      4. Write back as GeoParquet with small row groups (10 000 rows).
         pyarrow stores per-row-group min/max stats for the bbox columns,
         so runtime queries can skip entire row groups by tile bbox.

Outputs (alongside originals, suffix _spatial):
    /data/sks/utforda_avverkningar_spatial.parquet
    /data/sks/avverkningsanmalningar_spatial.parquet
    /data/lpis/jordbruksskiften_<year>_spatial.parquet

Usage:
    python scripts/preprocess_sks_lpis_spatial.py \\
        --sks-dir /data/sks \\
        --lpis-dir /data/lpis \\
        --grid-m 100000 \\
        --row-group-size 10000
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def preprocess_parquet(
    in_path: str,
    out_path: str,
    *,
    grid_m: int = 100_000,
    row_group_size: int = 10_000,
) -> dict:
    """Load, bucket, sort, and write a GeoParquet with small row groups.

    Adds columns:
        _bbox_minx, _bbox_miny, _bbox_maxx, _bbox_maxy (float64) —
            per-polygon bbox in EPSG:3006; row-group statistics on
            these enable bbox-based row-group filtering at query time.
        _grid_x, _grid_y (int32) — bucket indices used for sort.

    Args:
        in_path: Source parquet.
        out_path: Destination spatial parquet.
        grid_m: Bucket size in meters (default 100 km).
        row_group_size: Rows per row group (default 10 000).
    """
    import geopandas as gpd
    import pandas as pd

    t0 = time.time()
    print(f"  Loading {in_path} ({os.path.getsize(in_path)/1e9:.2f} GB)...")
    gdf = gpd.read_parquet(in_path)
    n = len(gdf)
    print(f"    {n:,} polygons, CRS={gdf.crs}")

    if gdf.crs is None or gdf.crs.to_epsg() != 3006:
        print(f"    Reprojecting to EPSG:3006...")
        gdf = gdf.to_crs(epsg=3006)

    # Per-polygon bbox
    b = gdf.geometry.bounds  # DataFrame with minx, miny, maxx, maxy
    gdf["_bbox_minx"] = b["minx"].astype("float64")
    gdf["_bbox_miny"] = b["miny"].astype("float64")
    gdf["_bbox_maxx"] = b["maxx"].astype("float64")
    gdf["_bbox_maxy"] = b["maxy"].astype("float64")

    # Bucket by polygon centroid (not bbox corner — stable across shapes)
    cx = (gdf["_bbox_minx"] + gdf["_bbox_maxx"]) / 2
    cy = (gdf["_bbox_miny"] + gdf["_bbox_maxy"]) / 2
    gdf["_grid_x"] = (cx // grid_m).astype("int32")
    gdf["_grid_y"] = (cy // grid_m).astype("int32")

    print(f"    Sorting by (_grid_x, _grid_y)...")
    gdf = gdf.sort_values(by=["_grid_x", "_grid_y"]).reset_index(drop=True)

    n_cells = len(gdf[["_grid_x", "_grid_y"]].drop_duplicates())
    print(f"    Unique grid cells: {n_cells} ({grid_m/1000:.0f} km × {grid_m/1000:.0f} km)")

    print(f"    Writing {out_path} (row_group_size={row_group_size})...")
    gdf.to_parquet(
        out_path,
        index=False,
        compression="snappy",
        row_group_size=row_group_size,
        # Ensure write_covering_bbox statistics so our filter predicates hit stats.
    )
    size_out = os.path.getsize(out_path) / 1e9
    elapsed = time.time() - t0
    print(f"    ✓ wrote {size_out:.2f} GB in {elapsed:.1f}s")
    return {
        "in_path": in_path,
        "out_path": out_path,
        "polygons": n,
        "grid_cells": n_cells,
        "size_gb": size_out,
        "elapsed_s": elapsed,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sks-dir", default="/data/sks")
    p.add_argument("--lpis-dir", default="/data/lpis")
    p.add_argument("--grid-m", type=int, default=100_000,
                   help="Bucket size in meters (default 100 km)")
    p.add_argument("--row-group-size", type=int, default=10_000,
                   help="Rows per row group (default 10 000)")
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="Skip outputs that already exist")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing outputs")
    args = p.parse_args()

    if args.force:
        args.skip_existing = False

    targets: list[tuple[str, str]] = []

    # SKS
    for name in ["utforda_avverkningar.parquet", "avverkningsanmalningar.parquet"]:
        src = os.path.join(args.sks_dir, name)
        dst = os.path.join(args.sks_dir, name.replace(".parquet", "_spatial.parquet"))
        if os.path.exists(src):
            targets.append((src, dst))

    # LPIS — all yearly files
    import glob as _glob
    for src in sorted(_glob.glob(os.path.join(args.lpis_dir, "jordbruksskiften_*.parquet"))):
        base = os.path.basename(src)
        if base.endswith("_spatial.parquet"):
            continue
        dst = src.replace(".parquet", "_spatial.parquet")
        targets.append((src, dst))

    print(f"=== Spatial preprocess ===")
    print(f"  Targets: {len(targets)}")
    for s, d in targets:
        print(f"    {os.path.basename(s)} → {os.path.basename(d)}")
    print()

    results = []
    for src, dst in targets:
        print(f"[{len(results)+1}/{len(targets)}] {os.path.basename(src)}")
        if args.skip_existing and os.path.exists(dst):
            print(f"  Skipping (already exists): {dst}")
            continue
        try:
            r = preprocess_parquet(
                src, dst,
                grid_m=args.grid_m,
                row_group_size=args.row_group_size,
            )
            results.append(r)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=== Summary ===")
    for r in results:
        print(f"  {os.path.basename(r['out_path'])}: "
              f"{r['polygons']:,} polys, {r['grid_cells']} cells, "
              f"{r['size_gb']:.2f} GB, {r['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
