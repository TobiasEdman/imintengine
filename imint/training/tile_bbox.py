"""
imint/training/tile_bbox.py — Shared tile bbox recovery / reconstruction.

Single source of truth for "given a tile .npz file, what is its EPSG:3006
bounding box?". Used by the fetcher's refetch path and every enrichment
script.

Resolution order (each method trusts the center; extent is always
rebuilt via ``tile.bbox_from_center()`` so the returned bbox always
satisfies ``tile.assert_bbox_matches()``):

    1. easting/northing keys in the .npz (preferred, persisted by fetcher)
    2. Parse filename ``tile_{EASTING}_{NORTHING}.npz``
    3. ``bbox_3006`` key in the .npz — center extracted, extent rebuilt
       to the current tile size (handles legacy 256-era bboxes inside
       512-era rasters)
    4. Optional lookup in ``tile_locations_full.json`` by tile name —
       center extracted, extent rebuilt (for legacy numeric filenames
       that don't carry coordinates)

Use this exclusively — do not reimplement bbox logic in new scripts.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from imint.training.tile_config import TileConfig


# Pattern: tile_{EASTING_3006}_{NORTHING_3006}.npz — used consistently by
# the fetcher since the 256-era. Supports 5-7 digit eastings/northings.
_FILENAME_RE = re.compile(r"tile_(\d+)_(\d+)")


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
    tile: "TileConfig",
    npz_data: dict | None = None,
    manifest_path: str | None = None,
) -> dict[str, int] | None:
    """Resolve a tile's EPSG:3006 bbox using the 4-method fallback chain.

    Every successful path returns a bbox built by ``tile.bbox_from_center()``
    so the returned bbox always has extent exactly ``tile.size_m``
    regardless of the source's own opinion about tile size.

    Args:
        name: Tile name, e.g. "tile_281280_6471280" or legacy "43963942".
        tile: Current TileConfig. Defines the output bbox extent.
        npz_data: Optional dict-like of .npz contents (np.load result or dict).
        manifest_path: Optional path to tile_locations_full.json. Needed
            only for legacy-numeric names. Defaults to
            ``/data/tile_locations_full.json`` when omitted.

    Returns:
        Dict with integer keys ``{"west","south","east","north"}`` or
        ``None`` if no recovery method succeeded.
    """
    # Method 1: easting/northing keys (preferred — direct center)
    if npz_data is not None:
        try:
            east_val = npz_data.get("easting")
            north_val = npz_data.get("northing")
        except AttributeError:
            east_val = north_val = None
        if east_val is not None and north_val is not None:
            return tile.bbox_from_center(int(east_val), int(north_val))

    # Method 2: filename parse
    m = _FILENAME_RE.search(name)
    if m:
        return tile.bbox_from_center(int(m.group(1)), int(m.group(2)))

    # Method 3: bbox_3006 array in .npz — extract center, rebuild extent
    if npz_data is not None:
        bbox_arr = npz_data.get("bbox_3006", None) if hasattr(npz_data, "get") else None
        if bbox_arr is not None:
            b = np.asarray(bbox_arr).flatten()
            if b.size >= 4:
                cx = (int(b[0]) + int(b[2])) // 2
                cy = (int(b[1]) + int(b[3])) // 2
                return tile.bbox_from_center(cx, cy)

    # Method 4: manifest lookup (legacy-numeric names)
    manifest_path = manifest_path or "/data/tile_locations_full.json"
    manifest = _load_manifest(manifest_path)
    entry = manifest.get(name)
    if entry is not None:
        b = entry.get("bbox_3006")
        if isinstance(b, dict):
            cx = (b["west"] + b["east"]) / 2
            cy = (b["south"] + b["north"]) / 2
            return tile.bbox_from_center(cx, cy)

    return None


def tile_size_px(npz_data: Any | None, *, default: int = 512) -> int:
    """Square pixel size of a tile, read from its stored spectral/image cube.

    Every aux fetcher must derive the output size the same way — from the tile's
    own data, never a CLI flag or module constant — so the fetched channel always
    matches the tile's pixel grid. Falls back to ``default`` only when no
    spectral/image cube is present (tiles are square; the last axis is returned).
    """
    if npz_data is not None and hasattr(npz_data, "get"):
        img = npz_data.get("spectral", npz_data.get("image"))
        if img is not None:
            arr = np.asarray(img)
            if arr.ndim >= 2:
                return int(arr.shape[-1])
    return default


def resolve_fetch_bbox(
    *,
    name: str,
    npz_data: Any | None = None,
    manifest_path: str | None = None,
    default_size_px: int = 512,
) -> tuple[dict[str, int] | None, int]:
    """Resolve ``(bbox_3006, size_px)`` for fetching tile-aligned aux data.

    THE single entry point every per-pixel aux/spectral fetcher must use. It
    derives ``size_px`` from the tile's own grid (:func:`tile_size_px`) and then
    calls :func:`resolve_tile_bbox` with a matching :class:`TileConfig`, so the
    fetched ground extent is ALWAYS coupled to the output pixel count at the 10 m
    NMD GSD. This replaces the per-script bbox math (``--patch-size-m`` and
    private ``_bbox_from_tile`` / ``_tile_bbox_3006`` helpers) that decoupled
    extent from pixels and produced the 256/512 aux-misalignment.

    Returns ``(bbox, size_px)``; ``bbox`` is ``None`` only when the tile's
    location cannot be resolved by any :func:`resolve_tile_bbox` method.
    """
    from imint.training.tile_config import TileConfig

    size = tile_size_px(npz_data, default=default_size_px)
    bbox = resolve_tile_bbox(
        name=name, tile=TileConfig(size_px=size),
        npz_data=npz_data, manifest_path=manifest_path,
    )
    return bbox, size
