# DES openEO — fix proposal for the `aggregate_spatial` GeometryDtype TypeError

**Date:** 2026-07-07
**From:** Tobias Edman (tobias.edman@ri.se), RISE — ImintEngine project
**Follow-up to:** our bug report `2026-05-26_aggregate_spatial_geopandas_dtype.md`
(HTTP 500, `TypeError: Cannot interpret '<geopandas.array.GeometryDtype …>'
as a data type`, openEO job ref `b97ab156-0c23-42e1-91d7-65f276d04125`)

This is a solution proposal, not a new bug. We have located the exact
failing statement in the worker's dependency stack and verified two
independent remedies plus a zero-rebuild hotfix.

---

## UPDATE 2026-07-07 (pm): reproducer delivered, affected range bracketed

You asked for a reproducer and noted that **xvec is hard-locked
upstream** — confirmed on our side: `openeo_processes_dask` (checked
2026.6.4) declares `Requires-Dist: xvec (==0.2.0)`. Everything below
works *within* that lock.

### Minimal reproducer — 6 lines, no xvec, no rioxarray, no openEO

`repro_geometry_index.py` (attached). Core:

```python
import xarray as xr, geopandas as gpd
from shapely.geometry import box

geoms = gpd.GeoSeries([box(0, 0, 1, 1), box(1, 1, 2, 2)], crs="EPSG:4326")
pieces = [xr.DataArray([float(i)], dims="geometry") for i in range(2)]
xr.concat(pieces, dim=xr.DataArray(geoms, name="geometry", dims="geometry"))
```

On an affected xarray this raises your exact error:

```
TypeError: Cannot interpret '<geopandas.array.GeometryDtype object at …>'
    as a data type
```

### End-to-end reproducer — through your exact xvec pin

`repro_zonal.py` with **xvec==0.2.0** + xarray 2025.1.0 +
geopandas 1.0.1 / pandas 2.2.3:

```
method=iterate: FAILED — TypeError: Cannot interpret
    '<geopandas.array.GeometryDtype object at 0x…>' as a data type
```

— identical to production. The trigger requires the geometries to reach
`zonal_stats` as a **GeoSeries** (pandas extension array), which is what
your `aggregate_spatial` version does. Current upstream
`openeo_processes_dask` passes `list(gdf.geometry.values)` instead
(commented *"addressing potential issues with xvec's zonal_stats
expecting list input"*) — i.e. upstream has already worked around this
very bug at the call site.

### Affected xarray range (bisected, all with xvec 0.2.0-era stack)

| xarray | verdict |
|---|---|
| ≤ 2024.9.0 | ✅ not affected (extension dtype object-coerced before indexing) |
| **2024.10.0 – 2025.3.0** | ❌ **affected** — `np.dtype(GeometryDtype)` TypeError |
| ≥ 2025.4.0 | ✅ not affected (fixed in the indexing adapter) |

This also explains the timeline: the worker image picked up an xarray
in the affected window at some rebuild (xarray is unpinned-floor
`>=2022.11.0` in openeo_processes_dask), and the bug appeared without
any xvec change.

### Revised remedies, respecting the xvec hard-lock

1. **Pin `xarray>=2025.4`** in the worker image — one constraint line.
   **Verified compatible with the locked `xvec==0.2.0`**: `iterate`
   zonal stats green on xarray 2025.4.0 + xvec 0.2.0.
2. **Or bump `openeo_processes_dask`** to a version with the
   `list(gdf.geometry.values)` call-site conversion — defuses the
   trigger without touching xarray or xvec.
3. **Or the 3-line hotfix** below (object-dtype coercion) — works on
   every combination, no dependency changes at all.

---

## The failing statement (pinpointed)

Your traceback (`xvec/zonal.py:193 in _zonal_stats_iterative`) matches
xvec **0.2.x–0.3.x** exactly. The statement is (xvec 0.3.0 source):

```python
vec_cube = xr.concat(
    zonal,
    dim=xr.DataArray(geometry, name=name, dims=name),   # ← here
).xvec.set_geom_indexes(name, crs=crs)
```

When `geometry` arrives as a **GeoSeries / GeometryArray** (pandas
extension array with `GeometryDtype`) — which is what
`openeo_processes_dask.aggregate_spatial` passes — `xr.DataArray(...)`
wraps it in xarray's `PandasIndexingAdapter`, whose `__init__` does
`self._dtype = np.dtype(dtype)`. `np.dtype()` cannot interpret a pandas
extension dtype → the exact `TypeError` in your log
(`xarray/core/indexing.py:1667`).

The bug therefore needs the *conjunction* of: xvec ≤ 0.3.x (the concat
above) **and** an xarray without extension-dtype handling in
`PandasIndexingAdapter` (pre-2024.03) **and** geometries reaching the
call as a pandas extension array (openeo_processes_dask does this; our
plain-shapely local repros did not, which is why the trivial repro
passes on some client stacks while your worker fails consistently).

## Remedy A — dependency bump (recommended)

Either leg is sufficient; both together is the comfortable choice:

| Package | Fix boundary | Why |
|---|---|---|
| `xarray` | **≥ 2024.03** | `PandasIndexingAdapter` gained pandas-ExtensionDtype support (upstream PR #8723 line) — `np.dtype()` is no longer called on `GeometryDtype`, so even old xvec's concat works. |
| `xvec` | **≥ 0.4** (we verified **0.5.2**) | `_zonal_stats_iterative` was rewritten; the offending `xr.concat(dim=xr.DataArray(geometry, …))` construction no longer exists in that form. |

**Verified matrix (our lab, py3.11, `zonal_stats` both methods,
in-memory AND dask-backed cubes):**

| xarray | xvec | geopandas | pandas | numpy | iterate | rasterize |
|---|---|---|---|---|---|---|
| 2026.4.0 | 0.5.2 | 1.1.4 | 3.0.3 | 2.4.6 | ✅ | ✅ |
| 2024.2.0 | 0.3.0 | 1.0.1 | 2.2.3 | 1.26.4 | ✅* | ❌ NaN→int (separate bug, see below) |
| 2023.12.0 | 0.2.0 | 1.0.1 | 2.2.3 | 1.26.4 | ✅* | ❌ NaN→int |

\* our local repro feeds plain shapely arrays; your worker feeds a
pandas GeometryArray via openeo_processes_dask, which is the missing
trigger ingredient — see "failing statement" above. The structural fix
claim does not depend on our repro: the code path is eliminated /
neutralised by the version bounds.

**Incidental finding you get for free:** on the old stack the
*rasterize* strategy fails differently
(`ValueError: cannot convert float NaN to integer` in the
transform/shape path) — also gone on the verified modern stack. Two
bugs, one upgrade.

## Remedy B — zero-rebuild hotfix (if an image rebuild must wait)

Coerce the concat dim coordinate to a plain object-dtype numpy array so
xarray never sees the pandas extension dtype. Three lines in
`xvec/zonal.py` (0.2.x/0.3.x), immediately before the `xr.concat`:

```python
import numpy as np
geom_arr = np.empty(len(geometry), dtype=object)
geom_arr[:] = np.asarray(geometry)

vec_cube = xr.concat(
    zonal,
    dim=xr.DataArray(geom_arr, name=name, dims=name),   # was: geometry
).xvec.set_geom_indexes(name, crs=crs)
```

`set_geom_indexes` re-attaches xvec's own `GeometryIndex` from the
values afterwards, so downstream behaviour (CRS, spatial indexing) is
unchanged. Equivalent as a sitecustomize monkeypatch if you prefer not
to touch site-packages:

```python
# /app/sitecustomize.py — remove after the image rebuild
import numpy as np, xarray as xr
import xvec.zonal as _z
_orig = _z._zonal_stats_iterative

def _patched(acc, geometry, *a, **kw):
    arr = np.empty(len(geometry), dtype=object)
    arr[:] = np.asarray(geometry)
    return _orig(acc, arr, *a, **kw)

_z._zonal_stats_iterative = _patched
```

(The monkeypatch loses GeoSeries `.crs` carry-through — the file patch
in-place is the better hotfix; the monkeypatch is listed for
completeness.)

## Verification script

`repro_zonal.py` (attached / available on request): builds a 3-timestep
cube + 1 polygon and runs `zonal_stats` with both strategies, in-memory
and dask-backed. Green on the Remedy-A stack. For a worker-side test,
the minimal openEO graph from our 2026-05-26 report (§ Minimal
reproduction) is still the truest end-to-end check — expected result is
`{"YYYY-MM-DD": [[cloud_frac]], …}` instead of HTTP 500.

## Why we keep caring

Our date-selection pipeline screens AOI cloud fractions for ~1 100
tiles × 4-5 windows. Without server-side `aggregate_spatial` we
download SCL raster stacks in ≤19-timestep chunks and reduce
client-side — 10⁴–10⁵× the bandwidth of the scalar answer, on the same
DES capacity that serves our spectral fetches. We have since added
client-side caching (complete-screen memo + disk), which reduces our
re-screening load substantially — but first-time screens remain
raster-sized until `aggregate_spatial` works. We will switch over the
day it does, with zero code change on our side (the call path already
exists behind a fallback).

Happy to test a staging endpoint, provide process graphs, or pair on
the worker-image pins.
