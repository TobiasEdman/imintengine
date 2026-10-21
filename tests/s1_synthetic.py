"""Synthetic Sentinel-1 GRD rasters for offline tests (shared by several test files)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def synthetic_grd(path: Path, width: int = 2400, height: int = 1800, *,
                  rows: tuple[int, int] | None = None) -> dict:
    """A GCP-referenced uint16 'GRD' across 12 E, rotated like a descending pass.

    The first GCP lies in UTM 33 and the footprint centroid in UTM 32, so the
    chain warps 33 -> 32 as for the full Skagen scene.

    ``rows=(r0, r1)`` writes only lines ``r0:r1`` of that scene, with GCPs of
    its own every ~360 lines: an azimuth slice, as ESA cuts one datatake into
    consecutive products. ``None`` is the whole scene (6 GCP lines).

    The TIFF holds the raw GCPs (z = 0, as ESA's over the sea); the returned
    header adds an IW-like incidence angle per GCP (30.5 -> 46 deg across the
    swath), as ``_s1_scene_header`` reads it from the product annotation.
    """
    import rasterio
    from rasterio.control import GroundControlPoint

    rng = np.random.default_rng(7)
    data = rng.gamma(2.0, 25.0, size=(height, width)).astype(np.uint16) + 1
    data[rng.integers(0, height, 40), rng.integers(0, width, 40)] = 3000   # bright targets
    r0, r1 = rows if rows is not None else (0, height)
    n_lines = 6 if rows is None else max(2, round((r1 - r0) / 360) + 1)
    gcps = []
    for row in np.linspace(r0, r1, n_lines):
        for col in np.linspace(0, width, 7):
            # ~10 m pixels, ~12 deg rotation, a little curvature so the TPS is not affine.
            lon = 12.10 - 1.68e-4 * col * math.cos(0.21) + 1.68e-4 * row * math.sin(0.21)
            lat = 57.95 - 9.0e-5 * row * math.cos(0.21) - 9.0e-5 * col * math.sin(0.21) \
                + 2e-10 * col ** 2
            gcps.append(GroundControlPoint(row=float(row - r0), col=float(col), x=lon, y=lat,
                                           z=0.0, id=str(len(gcps) + 1)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=width, height=r1 - r0, count=1,
                       dtype="uint16", nodata=0, gcps=gcps, crs="EPSG:4326") as dst:
        dst.write(data[r0:r1], 1)
    return {"scene_id": "SYNTH", "href": str(path), "width": width, "height": r1 - r0,
            "dtype": "uint16", "nodata": 0.0, "gcp_crs": "EPSG:4326",
            "gcps": [[g.id, g.col, g.row, g.x, g.y, g.z] for g in gcps],
            "gcp_incidence_deg": [30.5 + 15.5 * g.col / width for g in gcps]}


def sea_surface_copy(path: Path, header: dict, out: Path) -> Path:
    """``path``'s raster with the header's GCPs moved onto the sea surface.

    The full-scene reference for the chain: same pixels, the GCPs the fetch
    uses (``fetch._s1_sea_surface_gcps``) written by rasterio, not by the VRT
    builder under test.
    """
    import rasterio
    from rasterio.control import GroundControlPoint

    from imint import fetch

    gcps = [GroundControlPoint(row=r, col=c, x=x, y=y, z=z, id=i)
            for i, c, r, x, y, z in fetch._s1_sea_surface_gcps(header)]
    with rasterio.open(path) as src:
        data, profile = src.read(1), src.profile
    profile.pop("transform", None)
    profile.update(gcps=gcps, crs="EPSG:4326")
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(data, 1)
    return out
