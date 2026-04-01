#!/usr/bin/env python3
"""Enrich existing tiles with SKS hygge labels (date-filtered).

Reads each tile's year from dates/year field, filters SKS utförda
avverkningar to within 5 years, rasterizes, and overlays on label.

Usage:
    python scripts/enrich_hygge.py --data-dir /data/unified_v2 --sks-dir /data/sks
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imint.training.unified_schema import HARVEST_CLASS, UNIFIED_CLASS_NAMES

_sks_utforda = None
_sks_anmalda = None
_rasterize_sks = None


def _load_sks(sks_dir: str):
    global _sks_utforda, _sks_anmalda, _rasterize_sks
    if _sks_utforda is not None:
        return

    from scripts.enrich_tiles_sks import rasterize_sks
    _rasterize_sks = rasterize_sks

    import geopandas as gpd

    for name in ["utforda_avverkningar_spatial.parquet", "utforda_avverkningar.parquet"]:
        path = os.path.join(sks_dir, name)
        if os.path.exists(path):
            _sks_utforda = gpd.read_parquet(path)
            # Parse Avvdatum to datetime
            import pandas as pd
            _sks_utforda["Avvdatum"] = pd.to_datetime(_sks_utforda["Avvdatum"], errors="coerce")
            _ = _sks_utforda.sindex
            print(f"  Loaded SKS utförda: {len(_sks_utforda):,} polygons", flush=True)
            break

    path = os.path.join(sks_dir, "avverkningsanmalningar.parquet")
    if os.path.exists(path):
        _sks_anmalda = gpd.read_parquet(path)
        _ = _sks_anmalda.sindex
        print(f"  Loaded SKS anmälda: {len(_sks_anmalda):,} polygons", flush=True)


def _get_tile_year(data: dict) -> int:
    """Extract tile year from .npz data."""
    if "year" in data:
        return int(data["year"])
    if "lpis_year" in data:
        return int(data["lpis_year"])
    if "dates" in data:
        for d in data["dates"]:
            s = str(d)
            if s and len(s) >= 4:
                try:
                    return int(s[:4])
                except ValueError:
                    continue
    return 2018  # fallback


def _get_bbox(data: dict, name: str) -> dict | None:
    """Extract bbox from tile data or filename."""
    if "bbox_3006" in data:
        b = data["bbox_3006"].flatten()
        return {"west": int(b[0]), "south": int(b[1]),
                "east": int(b[2]), "north": int(b[3])}
    if "easting" in data and "northing" in data:
        e, n = int(data["easting"]), int(data["northing"])
        return {"west": e - 1280, "south": n - 1280,
                "east": e + 1280, "north": n + 1280}
    m = re.search(r'tile_(\d+)_(\d+)', name)
    if m:
        e, n = int(m.group(1)), int(m.group(2))
        return {"west": e - 1280, "south": n - 1280,
                "east": e + 1280, "north": n + 1280}
    return None


def enrich_tile(tile_path: str, sks_dir: str) -> dict:
    """Add hygge overlay to one tile."""
    name = os.path.basename(tile_path).replace(".npz", "")
    try:
        data = dict(np.load(tile_path, allow_pickle=True))

        label = data.get("label", None)
        if label is None or label.ndim < 2:
            return {"name": name, "status": "skipped", "reason": "no_label"}

        bbox = _get_bbox(data, name)
        if bbox is None:
            return {"name": name, "status": "skipped", "reason": "no_bbox"}

        tile_year = _get_tile_year(data)
        bbox_tuple = (bbox["west"], bbox["south"], bbox["east"], bbox["north"])

        _load_sks(sks_dir)
        if _sks_utforda is None:
            return {"name": name, "status": "skipped", "reason": "no_sks"}

        # Filter by date: hygge = avverkat inom 5 år före tile-året
        import pandas as pd
        min_date = pd.Timestamp(f"{tile_year - 5}-01-01")
        max_date = pd.Timestamp(f"{tile_year}-12-31")
        sks_filtered = _sks_utforda[
            (_sks_utforda["Avvdatum"] >= min_date) &
            (_sks_utforda["Avvdatum"] <= max_date)
        ]

        if len(sks_filtered) == 0:
            return {"name": name, "status": "ok", "hygge_pixels": 0}

        harvest_mask, n_harvest = _rasterize_sks(sks_filtered, bbox_tuple)
        data["harvest_mask"] = harvest_mask
        data["n_harvest_polygons"] = np.int32(n_harvest)

        # Avverkningsmogen (anmälda)
        if _sks_anmalda is not None:
            mature_mask, n_mature = _rasterize_sks(_sks_anmalda, bbox_tuple)
            data["n_mature_polygons"] = np.int32(n_mature)

        # Overlay hygge on label — unconditionally (no is_forest check)
        label = label.copy()
        label[harvest_mask > 0] = HARVEST_CLASS
        data["label"] = label

        hygge_pixels = int((harvest_mask > 0).sum())
        np.savez_compressed(tile_path, **data)
        return {"name": name, "status": "ok", "hygge_pixels": hygge_pixels}

    except Exception as e:
        return {"name": name, "status": "failed", "reason": str(e)[:120]}


def main():
    p = argparse.ArgumentParser(description="Enrich tiles with SKS hygge labels")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--sks-dir", default="data/sks")
    args = p.parse_args()

    tiles = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    print(f"=== Enrich Hygge ===")
    print(f"  Tiles: {len(tiles)}")
    print(f"  SKS: {args.sks_dir}")
    print(f"  Hygge window: tile_year-5 → tile_year")

    stats = {"ok": 0, "skipped": 0, "failed": 0}
    total_hygge_px = 0
    tiles_with_hygge = 0
    t0 = time.time()

    for i, t in enumerate(tiles):
        r = enrich_tile(t, args.sks_dir)
        stats[r.get("status", "failed")] = stats.get(r.get("status", "failed"), 0) + 1
        hp = r.get("hygge_pixels", 0)
        total_hygge_px += hp
        if hp > 0:
            tiles_with_hygge += 1
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(tiles)}] {r['name']}: {r['status']} "
                  f"| {(i+1)/elapsed*3600:.0f}/h | hygge_tiles={tiles_with_hygge}",
                  flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  OK={stats['ok']}  Skipped={stats['skipped']}  Failed={stats['failed']}")
    print(f"  Tiles with hygge: {tiles_with_hygge}")
    print(f"  Total hygge pixels: {total_hygge_px:,}")


if __name__ == "__main__":
    main()
