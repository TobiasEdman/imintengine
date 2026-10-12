#!/usr/bin/env python3
"""CPU coverage only under the sealed Rasterio interpreter; emit no classes.

JSON coordinates and raster paths enter stdin. Boolean support masks exit
stdout. Used before quota selection; never construct a model or score NMD.
"""
from __future__ import annotations

import json
import sys

import numpy as np


def nmd_covered(raw) -> np.ndarray:
    """Shared support predicate: finite, unmasked and nonzero raster codes."""
    values = np.ma.asarray(raw)
    return (~np.ma.getmaskarray(values) & np.isfinite(values.data) & (values.data != 0))


def sample_nmd_codes(path, coordinates) -> np.ndarray:
    import rasterio

    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or not np.isfinite(coordinates).all():
        raise ValueError("NMD coverage requires finite 2D observation coordinates")
    with rasterio.open(path) as raster:
        if raster.crs != rasterio.crs.CRS.from_epsg(3006):
            raise ValueError("NMD coverage requires EPSG:3006")
        values = list(raster.sample(coordinates, indexes=1, masked=True))
    if not values:
        return np.array([], dtype=np.int64)
    raw = np.ma.concatenate(values)
    # Preserve the established zero=uncovered return contract for consumers.
    return np.where(nmd_covered(raw), raw.data, 0).astype(np.int64)


def coverage_masks(request: dict) -> dict[str, list[bool]]:
    return {name: nmd_covered(sample_nmd_codes(path, request["coordinates"])).tolist()
            for name, path in request["baselines"].items()}


if __name__ == "__main__":
    json.dump(coverage_masks(json.load(sys.stdin)), sys.stdout, allow_nan=False)
    sys.stdout.write("\n")
