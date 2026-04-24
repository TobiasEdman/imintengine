# Sentinel-1 GRD fetch — STAC + direct COG path

## Why we switched

The original fetcher (`imint/training/cdse_s1.py`) posts one request per tile
per date to the Sentinel Hub Process API at
`https://sh.dataspace.copernicus.eu/api/v1/process`. That endpoint bills
Processing Units (PUs) per request.

For this project's scale:

- ~8 260 tiles × 4 temporal frames = **~33 000 frames**
- `scripts/enrich_tiles_s1.py` tries up to 7 dates per frame (±3-day window)
- Worst case: **~230 000 SH Process calls**

The free-tier quota on CDSE is **10 000 PUs/month**, which the S1 enrichment
exhausted well before covering the full dataset.

CDSE exposes other free quotas on the same account; the one that fits
a one-time bulk backfill of S1 is the **OData / STAC catalog with direct COG
access**:

| Quota           | Budget            |
| --------------- | ----------------- |
| Bandwidth       | **12 TB / month** |
| HTTP requests   | **50 000 / month** |
| PU consumption  | **0** (no PU accounting) |

A Sentinel-1 GRD IW product is ~2 GB. Sweden covered by our 4-frame-per-tile
schedule needs on the order of **100–300 unique products** total, i.e. well
under 1 TB of bandwidth and a few hundred HTTP requests when each product is
downloaded exactly once and then re-read for every tile it covers.

## The new path

`imint/training/cdse_s1_stac.py` implements a drop-in replacement for
`fetch_s1_scene` with identical signature and return contract. Internally:

1. **STAC search** against `https://stac.dataspace.copernicus.eu/v1`,
   collection `SENTINEL-1-GRD`, filtered to IW / GRDH and the requested
   orbit direction.
2. **Product cache** keyed by product id. On first hit the module streams
   the VV / VH measurement COGs and their calibration XMLs to
   `$S1_CACHE_DIR/<product_id>/` (default `/data/s1_cache/`). Every
   subsequent tile that needs the same product re-uses the cache — one
   download per product, not per tile.
3. **Window read** with `rasterio`: reproject the input bbox into the
   product CRS, build a `Window` via `from_bounds`, read at the requested
   `size_px` with bilinear resampling.
4. **Local σ⁰ calibration**: parse `<sigmaNought>` from the calibration
   XML, bilinearly interpolate the LUT onto the window's pixel/line grid,
   apply `σ⁰ = DN² / LUT²`. This is sigma-nought on the WGS84 ellipsoid —
   no terrain correction, no SNAP, pure Python.

The fetcher is safe to run concurrently from many tile-workers; a
per-product lock prevents duplicate 2 GB downloads when N workers race to
the same product.

## Operating the cache

- **Location**: controlled by `S1_CACHE_DIR` (defaults to `/data/s1_cache`).
- **Shape**: one directory per product id, each containing
  `measurement_vv.tiff`, `measurement_vh.tiff`, `calibration_vv.xml`,
  `calibration_vh.xml`.
- **Expected size on Sweden**: ~600 GB–1.5 TB of products total.

Clear it programmatically:

```python
from imint.training.cdse_s1_stac import clear_cache

clear_cache()                              # wipe everything
clear_cache(product_id="S1A_IW_GRDH_...")  # remove one product
```

Or from the shell on the PVC:

```bash
rm -rf /data/s1_cache/*
```

## Smoke-testing

Run the module's built-in `__main__` block against the mounted PVC:

```bash
kubectl exec -it <debug-pod> -- \
    python -m imint.training.cdse_s1_stac
```

It fetches a known Swedish bbox on `2023-06-15`, prints the result shape
and the VV/VH dynamic range. A successful run looks like:

```
[smoke] shape=(2, 256, 256) dtype=float32 orbit=DESCENDING
[smoke] VV  min=...  max=...  mean=...  nonzero_frac=0.998
[smoke] VH  min=...  max=...  mean=...  nonzero_frac=0.998
```

## Wiring the new backend in

`scripts/enrich_tiles_s1.py` currently imports the Process-API fetcher:

```python
from imint.training.cdse_s1 import fetch_s1_scene
```

Switching to the STAC path is a single-line change:

```python
from imint.training.cdse_s1_stac import fetch_s1_scene
```

The return contract, bbox convention, and nodata semantics are identical,
so no other code needs to change. Do the swap after a smoke-test run
against the PVC confirms the cache dir is writable and at least one
product round-trips successfully.

## Caveats

- The module imports `pystac-client`, `rasterio`, and `scipy` lazily
  (either inside `fetch_s1_scene` or inside the helper that needs them),
  so the module itself can be imported on machines without those libs, but
  calling `fetch_s1_scene` requires all three.
- σ⁰ on the ellipsoid (not terrain-corrected) is what the old Process API
  call also returned when `orthorectify=False` — but the old call here
  used `orthorectify=True` with `COPERNICUS_30`. Accept this as a
  deliberate trade-off: going terrain-corrected in pure Python means
  building a DEM pipeline, which is out of scope for a quota-driven
  rewrite.
- First-run cost on a fresh cache is dominated by product downloads
  (~2 GB each). The cache directory should live on the PVC, not in the
  pod's ephemeral storage.
