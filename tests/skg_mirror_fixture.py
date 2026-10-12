"""Helpers that build a tiny Skogsstyrelsen open-data mirror for tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def write_raster(path: Path, data: np.ndarray, west: float, north: float,
                 res: float, *, nodata) -> Path:
    """Write ``data`` as an EPSG:3006 GeoTIFF at ``path``.

    The mirror reader resolves ``*.vrt`` paths; GDAL identifies formats by
    content, so a GeoTIFF stored under that name is read the same way.
    """
    import rasterio
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", width=data.shape[1], height=data.shape[0],
        count=1, dtype=data.dtype, crs="EPSG:3006",
        transform=from_origin(west, north, res, res), nodata=nodata,
    ) as dst:
        dst.write(data, 1)
    return path
