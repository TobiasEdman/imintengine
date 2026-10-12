"""The NMD 10 m lattice in EPSG:3006 (SWEREF99 TM).

Every per-pixel aux raster in the training pipeline (Sentinel-2 frames,
NMD labels, Skogsstyrelsen forestry, SLU markfukt, DEM, VPP) is sampled on
this lattice: cell edges at multiples of ``GRID_M`` metres. Snapping a bbox
here is what keeps the fetchers pixel-aligned with each other.
"""

from __future__ import annotations

GRID_M = 10


def snap_bounds(west: float, south: float, east: float, north: float) -> tuple[int, int, int, int]:
    """Expand an EPSG:3006 bbox outwards to the nearest 10 m cell edges."""
    w = int(west // GRID_M) * GRID_M
    s = int(south // GRID_M) * GRID_M
    e = int(-(-east // GRID_M)) * GRID_M   # ceil
    n = int(-(-north // GRID_M)) * GRID_M
    return w, s, e, n


def to_nmd_grid_bounds(coords: dict) -> dict:
    """Convert a WGS84 bbox (``west/south/east/north`` in degrees) to an
    EPSG:3006 bbox snapped outwards to the 10 m NMD grid."""
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds

    w, s, e, n = transform_bounds(
        CRS.from_epsg(4326), CRS.from_epsg(3006),
        coords["west"], coords["south"], coords["east"], coords["north"],
    )
    w, s, e, n = snap_bounds(w, s, e, n)
    return {"west": w, "south": s, "east": e, "north": n}


def is_on_grid(*values: float) -> bool:
    """True when every value is a multiple of the 10 m cell size."""
    return all(float(v) % GRID_M == 0 for v in values)
