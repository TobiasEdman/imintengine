#!/usr/bin/env python3
"""Probe TESSERA pre-computed embedding coverage for Sweden.

TESSERA (arXiv:2506.20380, ucam-eo/tessera) publishes pre-computed
128-D per-pixel Sentinel-1/2 annual embeddings via the geotessera
Python library. Coverage is advertised as 2017-2025, but actual
availability for specific regions varies.

This script answers:
    1. How many of our Swedish bounding boxes have TESSERA embeddings?
    2. Which years?
    3. What does a sample embedding look like (shape, dtype, stats)?

Before we build an enrichment pipeline, we want to know if TESSERA is
usable for our 8261 tiles or if coverage is too sparse to justify.

Usage:
    python scripts/probe_tessera_coverage.py \\
        --tile-locations /data/tile_locations_full.json \\
        --sample-size 100 \\
        --years 2018 2019 2022 2023 2024 \\
        --output /tmp/tessera_coverage.json

Depends on: pip install geotessera
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def bbox_to_wgs84(bbox_3006: dict) -> tuple[float, float, float, float]:
    """Convert EPSG:3006 bbox → WGS84 (lon_min, lat_min, lon_max, lat_max)."""
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds
    w, s, e, n = transform_bounds(
        CRS.from_epsg(3006), CRS.from_epsg(4326),
        bbox_3006["west"], bbox_3006["south"],
        bbox_3006["east"], bbox_3006["north"],
    )
    return (w, s, e, n)


def probe_tile(gt, tile_name: str, bbox_wgs84: tuple, year: int) -> dict:
    """Query GeoTessera for the given bbox+year. Returns result dict."""
    lon_min, lat_min, lon_max, lat_max = bbox_wgs84
    info = {
        "name": tile_name,
        "year": year,
        "bbox_wgs84": list(bbox_wgs84),
        "available": False,
        "n_tiles_returned": 0,
        "sample_shape": None,
        "sample_dtype": None,
        "error": None,
    }
    try:
        tiles = gt.registry.load_blocks_for_region(
            bounds=(lon_min, lat_min, lon_max, lat_max), year=year,
        )
        info["n_tiles_returned"] = len(tiles) if tiles else 0
        if tiles:
            info["available"] = True
            # Fetch the first one to verify it's actually downloadable
            first = tiles[0]
            fetched = gt.fetch_embeddings([first])
            for (y, tl, tlat, arr, crs, tform) in fetched:
                info["sample_shape"] = list(arr.shape)
                info["sample_dtype"] = str(arr.dtype)
                info["sample_mean"] = float(arr.mean())
                info["sample_std"] = float(arr.std())
                break
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tile-locations", default="/data/tile_locations_full.json")
    p.add_argument("--sample-size", type=int, default=100)
    p.add_argument("--years", nargs="+", type=int,
                   default=[2018, 2019, 2022, 2023, 2024])
    p.add_argument("--output", default="/tmp/tessera_coverage.json")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    try:
        from geotessera import GeoTessera
    except ImportError:
        print("ERROR: geotessera not installed. Run: pip install geotessera",
              file=sys.stderr)
        sys.exit(1)

    with open(args.tile_locations) as f:
        all_tiles = json.load(f)
    print(f"Loaded {len(all_tiles)} tile locations")

    random.seed(args.seed)
    sample = random.sample(all_tiles, min(args.sample_size, len(all_tiles)))
    print(f"Probing {len(sample)} random tiles × {len(args.years)} years "
          f"= {len(sample) * len(args.years)} total queries")
    print()

    gt = GeoTessera()
    results = []
    t0 = time.time()

    for i, t in enumerate(sample):
        bbox = t.get("bbox_3006") or t.get("bbox")
        if isinstance(bbox, list):
            bbox = {
                "west": bbox[0], "south": bbox[1],
                "east": bbox[2], "north": bbox[3],
            }
        if not bbox:
            continue
        try:
            bbox_wgs = bbox_to_wgs84(bbox)
        except Exception as e:
            print(f"  [{i+1}/{len(sample)}] {t['name']}: bbox conv failed: {e}")
            continue
        for year in args.years:
            info = probe_tile(gt, t["name"], bbox_wgs, year)
            results.append(info)
            mark = "✓" if info["available"] else ("·" if info["error"] is None else "✗")
            print(f"  [{i+1}/{len(sample)}] {t['name']} year={year} "
                  f"{mark} n={info['n_tiles_returned']}"
                  f"{f' err={info[\"error\"][:60]}' if info['error'] else ''}",
                  flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")

    # Summarize
    by_year = {}
    for r in results:
        y = r["year"]
        by_year.setdefault(y, {"available": 0, "unavailable": 0, "error": 0})
        if r["available"]:
            by_year[y]["available"] += 1
        elif r["error"]:
            by_year[y]["error"] += 1
        else:
            by_year[y]["unavailable"] += 1

    print()
    print("=== Coverage summary ===")
    for y in sorted(by_year):
        s = by_year[y]
        total = s["available"] + s["unavailable"] + s["error"]
        pct = 100 * s["available"] / max(total, 1)
        print(f"  {y}: {s['available']}/{total} available ({pct:.0f}%) "
              f"— {s['unavailable']} unavailable, {s['error']} errors")

    with open(args.output, "w") as f:
        json.dump({"sample_size": len(sample), "years": args.years,
                   "results": results, "by_year": by_year}, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
