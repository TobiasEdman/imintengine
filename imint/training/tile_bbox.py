"""
imint/training/tile_bbox.py — Shared tile bbox recovery / reconstruction.

Single source of truth for "given a tile .npz file, what is its EPSG:3006
bounding box?". Used by the fetcher's refetch path and every enrichment
script. Matches the 3-method recovery chain already implemented inline in
``scripts/fetch_unified_tiles.py`` lines 492-513, plus an optional fallback
to ``/data/tile_locations_full.json`` when filename-based recovery fails.

Resolution order:
    1. bbox_3006 key in the .npz (when fetcher persisted it)
    2. easting/northing keys + TILE_SIZE_M//2 (ditto)
    3. Parse filename tile_{EASTING}_{NORTHING}.npz + TILE_SIZE_M//2
    4. Optional lookup in tile_locations_full.json by tile name,
       re-expanded to the current TILE_SIZE_M (handles 256-era
       manifest entries by taking the center and re-centering).

Use this exclusively — do not reimplement bbox logic in new scripts.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any

import numpy as np


# Pattern: tile_{EASTING_3006}_{NORTHING_3006}.npz — used consistently by
# the fetcher since the 256-era. Supports 5-7 digit eastings/northings.
_FILENAME_RE = re.compile(r"tile_(\d+)_(\d+)")


def _expand_center(cx: int | float, cy: int | float, tile_size_m: int) -> dict[str, int]:
    half = tile_size_m // 2
    return {
        "west":  int(cx - half),
        "east":  int(cx + half),
        "south": int(cy - half),
        "north": int(cy + half),
    }


@lru_cache(maxsize=1)
def _load_manifest(path: str) -> dict[str, dict[str, Any]]:
    """Load tile_locations_full.json once, index by tile name."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        entries = json.load(f)
    return {t["name"]: t for t in entries if "name" in t}


def resolve_tile_bbox(
    *,
    name: str,
    npz_data: dict | None = None,
    tile_size_m: int,
    manifest_path: str | None = None,
) -> dict[str, int] | None:
    """Resolve a tile's EPSG:3006 bbox using the 4-method fallback chain.

    Args:
        name: Tile name, e.g. "tile_281280_6471280" or legacy "43963942".
        npz_data: Optional dict-like of .npz contents (any of np.load, dict).
        tile_size_m: Current tile width in metres (e.g. 2560 for 256-px,
            5120 for 512-px). Used to expand center coordinates.
        manifest_path: Optional path to tile_locations_full.json. Needed
            only for legacy-numeric names. Defaults to
            ``/data/tile_locations_full.json`` when omitted.

    Returns:
        Dict with integer keys ``{"west","south","east","north"}`` or
        ``None`` if no recovery method succeeded.
    """
    # Method 1: bbox_3006 array (fresh-fetch path persists this)
    if npz_data is not None:
        bbox_arr = npz_data.get("bbox_3006", None) if hasattr(npz_data, "get") else None
        if bbox_arr is not None:
            b = np.asarray(bbox_arr).flatten()
            if b.size >= 4:
                return {
                    "west":  int(b[0]), "south": int(b[1]),
                    "east":  int(b[2]), "north": int(b[3]),
                }

        # Method 2: easting/northing keys + tile_size_m
        try:
            east_val = npz_data.get("easting")
            north_val = npz_data.get("northing")
        except AttributeError:
            east_val = north_val = None
        if east_val is not None and north_val is not None:
            return _expand_center(int(east_val), int(north_val), tile_size_m)

    # Method 3: parse filename
    m = _FILENAME_RE.search(name)
    if m:
        return _expand_center(int(m.group(1)), int(m.group(2)), tile_size_m)

    # Method 4: manifest lookup (for legacy-numeric names)
    manifest_path = manifest_path or "/data/tile_locations_full.json"
    manifest = _load_manifest(manifest_path)
    entry = manifest.get(name)
    if entry is not None:
        b = entry.get("bbox_3006")
        if isinstance(b, dict):
            # Manifest bbox is authoritative for center; re-expand to current
            # tile_size_m to handle the 256-era manifest + 512-era rasters.
            cx = (b["west"] + b["east"]) / 2
            cy = (b["south"] + b["north"]) / 2
            return _expand_center(cx, cy, tile_size_m)

    return None
