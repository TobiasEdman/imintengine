# DES openEO bug report — `aggregate_spatial` fails with geopandas GeometryDtype TypeError

**Date observed:** 2026-05-26
**Reporter:** Tobias Edman (tobias.edman@ri.se), RISE — ImintEngine project
**Endpoint:** `https://openeo.digitalearth.se/openeo/1.1.0/`
**Auth:** Basic (`DES_USER` / `DES_PASSWORD`)
**openEO Python client:** 0.45.x (current PyPI)
**Severity:** Blocks all server-side spatial-reduction workflows on DES; users have to fall back to pixel-level downloads (10-100× more bandwidth).

This is the **second** bug report from the same project this month. The
first was the silent process-graph stall reported earlier (workers blocked
on `socket.recv()` for hours with no error returned to client). This
report is separately filed at the protocol level — the bugs may share
a root cause in the DES worker's dependency stack, but the symptoms
differ.

---

## Symptom

Any synchronous openEO process graph that calls
`aggregate_spatial(geometries=…, reducer=…)` returns HTTP 500 with a
worker-side `TypeError` originating in the geopandas / xarray / xvec
chain.

```
[500] Internal: Server error: Syncronouns job failed with
  stdout=[Executing process graph: ]
  stderr=[Job <uuid> Synchronous Job job got an Exception
          <class 'TypeError'>
          Cannot interpret '<geopandas.array.GeometryDtype object …>'
          as a data type.

Traceback (most recent call last):
  …
  File "/app/lib/python3.12/site-packages/openeo_processes_dask/
        process_implementations/cubes/aggregate.py", line 311,
        in aggregate_spatial
    vec_cube = data.xvec.zonal_stats(…)
  …
  File "/app/lib/python3.12/site-packages/xvec/zonal.py", line 193,
        in _zonal_stats_iterative
    vec_cube = xr.concat(…)
  …
  File "/app/lib/python3.12/site-packages/xarray/core/indexing.py",
        line 1667, in __init__
    self._dtype = np.dtype(dtype)
TypeError: Cannot interpret
    '<geopandas.array.GeometryDtype object at 0x7fae247b28a0>'
    as a data type
] (ref: b97ab156-0c23-42e1-91d7-65f276d04125)
```

The error fires before any pixel processing — `Executing process graph:`
is the only stdout line. So this is a graph-validation-stage failure,
not a data-availability or auth issue.

Same behaviour observed across:
* Different bboxes (tested several West Sweden tiles).
* Different temporal extents (single date, 1-month, 6-month windows).
* Multiple SCL band-name castings (`"scl"` and `"SCL"`).
* Different geometry sources (GeoJSON Polygon literal, shapely
  `box().__geo_interface__`, FeatureCollection wrapper).
* Different reducers (`"mean"`, `"sum"`).

Equivalent code paths against CDSE openEO
(`https://openeo.dataspace.copernicus.eu/openeo/1.2/`) work correctly
with the **same** process graph and the **same** Python client version.

---

## Minimal reproduction

```python
import openeo
from shapely.geometry import box, mapping

conn = openeo.connect("https://openeo.digitalearth.se/")
conn.authenticate_basic(username=<user>, password=<pass>)

# Any small West Sweden bbox — WGS84
west, south, east, north = 13.79, 60.83, 13.89, 60.87

scl = conn.load_collection(
    "s2_msi_l2a",
    spatial_extent={"west": west, "south": south,
                    "east": east, "north": north, "crs": "EPSG:4326"},
    temporal_extent=["2018-04-01", "2018-09-30"],
    bands=["scl"],
)

# Cloud mask: SCL ∈ {3 shadow, 8 cloud_medium, 9 cloud_high, 10 cirrus}
scl_b = scl.band("scl")
cloud_flag = (scl_b == 3) | (scl_b == 8) | (scl_b == 9) | (scl_b == 10)

poly = box(west, south, east, north)
result = cloud_flag.aggregate_spatial(geometries=mapping(poly),
                                       reducer="mean")
result.execute()        # → HTTP 500 TypeError above
```

Expected return: `{"YYYY-MM-DD": [[cloud_frac]], ...}` JSON.

---

## Likely root cause

The traceback path goes through:

```
openeo_processes_dask/process_implementations/cubes/aggregate.py:311
  → xvec.accessor.zonal_stats
    → xvec.zonal._zonal_stats_iterative
      → xarray.core.concat._dataset_concat
        → xarray.core.indexes.PandasIndex.create_variables
          → np.dtype(GeometryDtype)  ← TypeError here
```

The `np.dtype()` constructor rejects `geopandas.array.GeometryDtype`.
This is a known interaction between specific minor versions of geopandas,
xarray, and xvec — `np.dtype` only accepts extension dtypes via xarray's
own indexing layer when the versions are aligned (see e.g.
xarray PR #8407, geopandas issue #2929).

Best guess: the DES worker image pins an older `xvec` or `xarray` that
doesn't yet know how to wrap `GeometryDtype` correctly for `np.dtype()`,
or has a `geopandas` newer than its `xvec` was tested against. CDSE
runs the same `openeo_processes_dask` library successfully, so they
likely have a different (newer / compatible) `xvec` + `xarray` pin.

---

## Impact

`aggregate_spatial` is the standard openEO process for server-side
zonal statistics. Without it, any workflow that needs "metric per AOI
per timestep" — cloud fraction screening, NDVI averaging, change-
detection summaries — has to:

1. Download the full raster stack pixel-by-pixel.
2. Compute the reduction client-side.

For our use case (Sentinel-2 SCL cloud-fraction scoring of S2 dates
within a 5×5 km AOI), this changes the network payload from O(scenes)
~8-byte floats to O(scenes × 512 × 512) ~1-byte SCL classes — roughly
100 000× more bandwidth, plus the 19-timestep `save_result` cap forces
chunking. The fallback works (we've shipped around it for a year) but
it taxes the DES openEO download capacity that has been already
strained, and burns wall-time we'd otherwise spend on the spectral
download.

---

## Workaround in our code (for context)

We've kept the legacy pixel-level path as a fallback. The new
`score_dates_aoi_cloud` call in our `imint/training/openeo_tile_graph.py`
catches the `OpenEoApiError`, logs it, and routes to
`optimal_fetch_dates(mode="era5_then_scl_ranked")` which downloads
the SCL stack in 19-day chunks and aggregates client-side. End-users
of our refetch pipeline don't see the failure, only slower throughput
on DES tiles. We'd switch to the server-side aggregate the moment the
bug is fixed — no code change needed on our side.

---

## Asks

1. Confirm whether this is reproducible on your end (it should be —
   the graph is trivial) and that it's a worker-stack version-pin
   issue rather than something specific to our auth or bbox.
2. Indicate whether a worker-image rebuild with newer `xvec` /
   `xarray` is on a roadmap. We'd be happy to test a staging endpoint.
3. Cross-link with the silent-stall report (the one I sent earlier
   this week) — they may share a root cause in the DES worker
   dependency stack and benefit from a coordinated fix.

Happy to provide additional process graphs, traces, or to test a
staging build at any time.

---

**Reference IDs to include in DES-side tracking:**

| openEO job ref               | Bbox (WGS84)                          | Window               |
|------------------------------|---------------------------------------|----------------------|
| `b97ab156-0c23-42e1-91d7-65f276d04125` | 13.79 / 60.83 / 13.89 / 60.87 (West Sweden) | 2018-04-01 → 2018-09-30 |
