"""Minimal reproducer for the DES aggregate_spatial TypeError — no xvec,
no rioxarray, no openEO: just the failing operation itself.

xvec<=0.3.x _zonal_stats_iterative ends with

    xr.concat(zonal, dim=xr.DataArray(geometry, name=name, dims=name))

where `geometry` is a geopandas GeoSeries (pandas extension array,
GeometryDtype). concat promotes the dim coordinate to an INDEX →
pd.Index keeps the extension dtype → xarray's PandasIndexingAdapter
calls np.dtype(GeometryDtype) → TypeError: Cannot interpret
'<geopandas.array.GeometryDtype …>' as a data type.

Run: python repro_geometry_index.py   → exit 1 + the TypeError when the
stack is affected; exit 0 ("not affected") otherwise.
"""
import sys
import importlib.metadata as md

import xarray as xr
import geopandas as gpd
from shapely.geometry import box

vers = {p: md.version(p) for p in ("xarray", "geopandas", "pandas", "numpy", "shapely")}
print("STACK:", vers)

geoms = gpd.GeoSeries([box(0, 0, 1, 1), box(1, 1, 2, 2)], crs="EPSG:4326")
pieces = [xr.DataArray([float(i)], dims="geometry") for i in range(2)]

try:
    out = xr.concat(
        pieces,
        dim=xr.DataArray(geoms, name="geometry", dims="geometry"),
    )
    # Touch the index the way downstream code does.
    _ = out.indexes.get("geometry")
    _ = repr(out)
    print("NOT AFFECTED — concat over GeometryDtype coordinate succeeded")
    sys.exit(0)
except TypeError as e:
    print(f"REPRODUCED — TypeError: {e}")
    sys.exit(1)
