"""Geoid undulation N from the EGM96 15' grid shipped with the package.

``us_nga_egm96_15.tif`` is PROJ's EGM96 grid (cdn.proj.org; NGA, public
domain): the height of the EGM96 geoid above the WGS84 ellipsoid on a 15'
lattice of nodes at lon = -180 + 0.25 i, lat = 90 - 0.25 j. Ellipsoidal height
h and EGM96 height H relate as ``h = H + N``. The mean sea surface lies on the
geoid to within ~1 m, so a target on the sea is at ellipsoidal height ~N.

Offline and deterministic: no PROJ network grid, no fallback. A missing grid
or an invalid coordinate raises; N is never silently 0.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np

EGM96_GRID = Path(__file__).resolve().parent / "us_nga_egm96_15.tif"
EGM96_MODEL = "EGM96 15' (PROJ us_nga_egm96_15.tif), bilinear"


@lru_cache(maxsize=1)
def _egm96() -> tuple[np.ndarray, object]:
    """(grid, affine) of the EGM96 grid, read once (1440 x 721 float32, 4 MB)."""
    if not EGM96_GRID.is_file():
        raise FileNotFoundError(f"EGM96 geoid grid missing: {EGM96_GRID} (package data of "
                                f"imint.geoid; reinstall the package)")
    import rasterio

    with rasterio.open(EGM96_GRID) as src:
        grid = src.read(1).astype(np.float64)
        transform = src.transform
    if not np.isfinite(grid).all():
        raise ValueError(f"{EGM96_GRID} holds non-finite undulations")
    # A global grid: one column per 0.25 deg from -180, nodes on pixel centres.
    if grid.shape != (721, 1440) or (transform.a, transform.e) != (0.25, -0.25):
        raise ValueError(f"{EGM96_GRID}: unexpected layout {grid.shape}, {transform}")
    return grid, transform


@lru_cache(maxsize=1)
def egm96_grid_sha256() -> str:
    """sha256 of the shipped grid file (recorded in provenance)."""
    _egm96()                                   # fails loud if the file is missing
    return hashlib.sha256(EGM96_GRID.read_bytes()).hexdigest()


def egm96_undulation(lon, lat) -> np.ndarray:
    """EGM96 geoid undulation N (metres above WGS84), bilinear between grid nodes.

    Args:
        lon, lat: WGS84 degrees, scalars or arrays of one shape. Longitude
            wraps (the grid is global); latitude must lie in [-90, 90].

    Raises:
        FileNotFoundError: the grid file is missing.
        ValueError: a non-finite coordinate or a latitude outside [-90, 90].
    """
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    lon, lat = np.broadcast_arrays(lon, lat)
    if not (np.isfinite(lon).all() and np.isfinite(lat).all()):
        raise ValueError("non-finite coordinate passed to egm96_undulation")
    if (np.abs(lat) > 90.0).any():
        raise ValueError(f"latitude outside [-90, 90]: {lat[np.abs(lat) > 90.0][:3]}")
    grid, t = _egm96()
    # Node (j, i) sits at the centre of pixel (j, i): subtract half a pixel.
    col = (lon - t.c) / t.a - 0.5
    row = (lat - t.f) / t.e - 0.5
    n_rows, n_cols = grid.shape
    i0 = np.floor(col).astype(np.int64)
    j0 = np.clip(np.floor(row).astype(np.int64), 0, n_rows - 2)
    fc = col - i0
    fr = row - j0
    i0 %= n_cols
    i1 = (i0 + 1) % n_cols
    return ((1 - fr) * ((1 - fc) * grid[j0, i0] + fc * grid[j0, i1])
            + fr * ((1 - fc) * grid[j0 + 1, i0] + fc * grid[j0 + 1, i1]))
