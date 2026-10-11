#!/usr/bin/env python3
"""Does a windowed read of a GDAL WarpedVRT equal the same region of a full read?
(dossier: "Why a small warp is not enough")

Step 1 of rslearn's chain (TPS, cubic) on the reference target VV crop, read
whole, by windows of three sizes, block by block, and with the exact
transformer (``tolerance=0``). GDAL splits each request into chunks with their
own kernel scale and approximate-transformer segments, so the answer is no.

    python scripts/marine/s1_port/warp_window_dependence.py --record <record> --out window.json
"""
from __future__ import annotations

import argparse

import numpy as np

import _record as rec

WINDOWS = [(1000, 1000, 700, 500), (1537, 1311, 1500, 900), (3000, 2000, 2048, 1024)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    rec.add_record_arg(ap)
    ap.add_argument("--out", type=rec.Path)
    a = ap.parse_args()

    import rasterio
    from pyproj import CRS
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import Window

    out = {"rasterio": rasterio.__version__, "gdal": rasterio.__gdal_version__, "windows": []}
    warp = dict(crs=CRS.from_epsg(32632), resampling=Resampling.cubic, MAX_GCP_ORDER="-1")
    with rasterio.open(a.record / "inputs" / "target_vv.tif") as src:
        with WarpedVRT(src, **warp) as vrt:
            full = vrt.read(1)
            for c, r, w, h in WINDOWS:
                d = np.abs(vrt.read(1, window=Window(c, r, w, h)).astype(int)
                           - full[r:r + h, c:c + w].astype(int))
                out["windows"].append({"window": [c, r, w, h], "diff_frac": float((d > 0).mean()),
                                       "max_abs_dn": int(d.max())})
            H, W = full.shape
            blocks = np.zeros_like(full)
            for r in range(0, H, 128):
                for c in range(0, W, 512):
                    w, h = min(512, W - c), min(128, H - r)
                    blocks[r:r + h, c:c + w] = vrt.read(1, window=Window(c, r, w, h))
            out["block_by_block_512x128_diff_frac"] = float((blocks != full).mean())
        with WarpedVRT(src, tolerance=0, **warp) as vrt:
            exact = vrt.read(1)
            c, r, w, h = WINDOWS[1]
            sub = vrt.read(1, window=Window(c, r, w, h))
            out["exact_transformer_window_diff_frac"] = float((sub != exact[r:r + h, c:c + w]).mean())
            out["exact_vs_approx_full_diff_frac"] = float((exact != full).mean())
    rec.write_json(a.out, out)


if __name__ == "__main__":
    main()
