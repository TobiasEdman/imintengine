#!/usr/bin/env python3
"""Reproduce AI2's materialised model input from the radar-geometry GRD crops (dossier c).

An implementation independent of ``imint.fetch`` of the recipe a port's fetch
must follow:
  1. GRD measurement TIFF (uint16 DN, GCPs in EPSG:4326, nodata 0) ->
     WarpedVRT(crs = UTM zone of the FIRST GCP, cubic, MAX_GCP_ORDER=-1 [TPS]),
     GDAL-suggested grid, uint16;
  2. -> WarpedVRT onto the window grid (UTM zone of the GCP-bbox centroid,
     10 m, origin on multiples of 10 m), cubic, src_nodata 0, uint16 -> float32.
Compared pixel for pixel with the rasters AI2 materialised (``run1/scratch_dir``).

    python scripts/marine/s1_port/reproduce_inputs.py --record <record> --out reproduce_inputs.json
"""
from __future__ import annotations

import argparse

import numpy as np

import _record as rec


def utm_epsg(lon: float, lat: float) -> int:
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def step1(path):
    import rasterio
    from pyproj import CRS
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    with rasterio.open(path) as src:
        gcps, _ = src.gcps
        crs = CRS.from_epsg(utm_epsg(gcps[0].x, gcps[0].y))
        with WarpedVRT(src, crs=crs, resampling=Resampling.cubic, MAX_GCP_ORDER="-1") as vrt:
            return vrt.read(), vrt.crs, vrt.transform, src.nodata


def step2(arr, crs, transform, nodata, dst_crs, dst_transform, w, h):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    profile = dict(driver="GTiff", tiled=True, blockxsize=256, blockysize=256, compress="lzw",
                   width=arr.shape[2], height=arr.shape[1], count=arr.shape[0],
                   dtype=arr.dtype, crs=crs, transform=transform, nodata=nodata)
    with rasterio.MemoryFile() as mf:
        with mf.open(**profile) as ds:
            ds.write(arr)
        with mf.open() as ds, WarpedVRT(ds, crs=dst_crs, transform=dst_transform, width=w,
                                        height=h, resampling=Resampling.cubic,
                                        src_nodata=nodata) as vrt:
            return vrt.read().astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    rec.add_record_arg(ap)
    ap.add_argument("--out", type=rec.Path)
    a = ap.parse_args()

    import json

    import rasterio
    from pyproj import CRS
    from rasterio.transform import Affine

    meta = json.loads((rec.ai2_window_dir(a.record) / "metadata.json").read_text())
    c0, r0, c1, r1 = meta["bounds"]
    res = meta["projection"]["x_resolution"]
    dst_crs = CRS.from_user_input(meta["projection"]["crs"])
    dst_transform = Affine(res, 0, c0 * res, 0, -res, r0 * -res)
    w, h = c1 - c0, r1 - r0
    out = {}
    for scene in rec.SCENE_LAYERS:
        for pol in rec.POLS:
            arr, crs, tr, nd = step1(a.record / "inputs" / f"{scene}_{pol}.tif")
            mine = step2(arr, crs, tr, nd, dst_crs, dst_transform, w, h)[0]
            with rasterio.open(rec.ai2_band_path(a.record, scene, pol)) as ref:
                theirs = ref.read(1).astype(np.float32)
                same_grid = (ref.crs == dst_crs and ref.transform.almost_equals(dst_transform)
                             and ref.width == w and ref.height == h)
            d = np.abs(mine - theirs)
            valid = theirs > 0
            out[f"{scene}_{pol}"] = dict(
                step1_crs=str(crs), step1_res=tr.a, same_grid=bool(same_grid),
                exact_equal_frac=float((d == 0).mean()), max_abs_diff=float(d.max()),
                ref_valid_frac=float(valid.mean()), ref_p50_valid=float(np.median(theirs[valid])),
                ref_p99_valid=float(np.percentile(theirs[valid], 99)),
                frac_valid_gt_1000=float((theirs[valid] > 1000).mean()))
    rec.write_json(a.out, out)


if __name__ == "__main__":
    main()
