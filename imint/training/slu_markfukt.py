"""SLU Markfuktighetskarta — Pirinen 2023 lager #4 — from the open-data mirror.

Soil moisture (William Lidberg & Anneli Ågren, SLU Umeå), published by
Skogsstyrelsen as per-county 2 m GeoTIFFs and mirrored to the PVC
(``imint.training.skg_open``, backlog IM-022). Each 10 m NMD cell takes
the nearest 2 m sample at its centre, which is what the retired
ImageServer proxy returned for ``RSP_NearestNeighbor`` requests.

Encoding (uint8, the service encoding every stored tile uses):
    0        no data / outside Sweden
    1–100    soil moisture probability (low → high), %
    101      saturated water / lake fill
    255      fill value present in the source for some areas (callers map
             it to NaN together with 0)

Note: the published county files are not the same model version the proxy
served in 2026-06 (same nodata and water masks, index values differ by ~5
on average). Tiles enriched from the mirror therefore differ from the
stored ones in this channel; see ``docs/data/skg_open_mirror.md``.

License: Open (Skogsstyrelsen + SLU).

Typical usage::

    from imint.training.slu_markfukt import fetch_markfukt_tile
    sm = fetch_markfukt_tile(west, south, east, north, size_px=512)
    # sm.shape == (512, 512), dtype uint8
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import skg_open
from .nmd_grid import to_nmd_grid_bounds
from .skg_grunddata import _cached


def fetch_markfukt_tile(
    west: float,
    south: float,
    east: float,
    north: float,
    *,
    size_px: int | tuple[int, int] = 1000,
    cache_dir: Path | None = None,
) -> np.ndarray:
    """SLU Markfuktighet on the NMD 10 m grid.

    Args:
        west, south, east, north: Bounding box in EPSG:3006 (metres), on
            the 10 m lattice, with ``extent == size_px × 10 m``.
        size_px: Output size — int for square or ``(H, W)`` tuple.
        cache_dir: Optional ``.npy`` cache directory (key
            ``markfukt_<west>_<south>_<east>_<north>.npy``).

    Returns:
        ``(H, W)`` uint8 array in the service encoding.
    """
    return _cached(
        "markfukt", west, south, east, north, size_px, cache_dir,
        lambda: skg_open.read_markfukt(west, south, east, north, size_px=size_px),
    )


def fetch_markfukt_for_coords(
    coords: dict,
    *,
    size_px: int = 1000,
    cache_dir: Path | None = None,
) -> np.ndarray:
    """SLU Markfuktighet for a WGS84 bbox snapped outwards to the NMD 10 m
    grid; ``size_px`` must match the snapped extent at 10 m/px."""
    projected = to_nmd_grid_bounds(coords)
    return fetch_markfukt_tile(
        projected["west"], projected["south"],
        projected["east"], projected["north"],
        size_px=size_px,
        cache_dir=cache_dir,
    )
