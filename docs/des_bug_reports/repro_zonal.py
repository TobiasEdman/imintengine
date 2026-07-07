"""Minimal repro of the DES aggregate_spatial failure path:
xvec.zonal_stats over a small raster cube with one polygon — the exact
call chain from openeo_processes_dask aggregate.py:311. Exercises BOTH
xvec strategies (the iterative one is where DES's traceback dies:
_zonal_stats_iterative -> xr.concat -> np.dtype(GeometryDtype))."""
import sys

import numpy as np
import pandas as pd
import xarray as xr
import xvec  # noqa: F401  (registers the .xvec accessor)
import geopandas as gpd
from shapely.geometry import box

import importlib.metadata as md
vers = {p: md.version(p) for p in ("xarray", "xvec", "geopandas", "pandas", "numpy", "shapely")}
print("STACK:", vers)

# 3 timesteps x 20x20 raster, WGS84-ish coords
da = xr.DataArray(
    np.random.default_rng(0).uniform(0, 1, (3, 20, 20)).astype("float32"),
    dims=("time", "y", "x"),
    coords={
        "time": pd.date_range("2018-04-01", periods=3, freq="10D"),
        "y": np.linspace(60.87, 60.83, 20),
        "x": np.linspace(13.79, 13.89, 20),
    },
)
gdf = gpd.GeoDataFrame(geometry=[box(13.80, 60.84, 13.88, 60.86)], crs="EPSG:4326")

ok = True
for method in ("rasterize", "iterate"):
    try:
        out = da.xvec.zonal_stats(
            gdf.geometry, x_coords="x", y_coords="y", stats="mean",
            method=method,
        )
        vals = np.asarray(out).ravel()
        print(f"method={method}: OK — {len(vals)} values, mean={vals.mean():.3f}")
    except Exception as e:
        ok = False
        print(f"method={method}: FAILED — {type(e).__name__}: {e}")

print("VERDICT:", "ALL OK" if ok else "REPRODUCED FAILURE")
sys.exit(0 if ok else 1)
