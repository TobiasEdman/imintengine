#!/usr/bin/env python3
"""CPU coverage only under the sealed Rasterio interpreter; emit no classes.

JSON coordinates and raster paths enter stdin. Boolean support masks exit
stdout. Used before quota selection; never construct a model or score NMD.
"""
from __future__ import annotations

import json
import sys

import numpy as np


def coverage_masks(request: dict) -> dict[str, list[bool]]:
    import rasterio

    coordinates = np.asarray(request["coordinates"], dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or not np.isfinite(coordinates).all():
        raise ValueError("NMD coverage requires finite 2D observation coordinates")
    result = {}
    for name, path in request["baselines"].items():
        with rasterio.open(path) as raster:
            if raster.crs != rasterio.crs.CRS.from_epsg(3006):
                raise ValueError(f"{name}: NMD coverage requires EPSG:3006")
            result[name] = [
                bool(not np.ma.getmaskarray(value)[0] and value[0] != 0)
                for value in raster.sample(coordinates, indexes=1, masked=True)
            ]
    return result


if __name__ == "__main__":
    json.dump(coverage_masks(json.load(sys.stdin)), sys.stdout, allow_nan=False)
    sys.stdout.write("\n")
