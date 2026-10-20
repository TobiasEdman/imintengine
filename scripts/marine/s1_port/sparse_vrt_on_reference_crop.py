#!/usr/bin/env python3
"""The fetch's sparse full-size VRT on real data: the reference crop as a stand-in scene.

Uses the production functions of ``imint.fetch`` (radar window, VRT XML, both
warps) on AI2's radar-geometry crop and window: the crop plays the full scene,
only a radar window around an AOI inside AI2's window is kept, and the AOI
result must equal both the non-sparse chain and AI2's materialised raster bit
for bit. ``target`` is warped once (first GCP in zone 32); ``historical1``
33 -> 32.

Chain 1 only: AI2's raster is rslearn's warp of the ellipsoid-height GCPs, and
the crop's GCPs carry no incidence angles. Under chain 2 (sea-surface GCPs) the
script refuses to run; reproduce its results on a chain-1 checkout (e.g.
864e75e). The chain-independent claim, sparse VRT = full-scene chain, is
checked under chain 2 by ``full_scene_check.py`` and
``tests/test_ai2_vessels_s1.py::test_sparse_scene_vrt_equals_full_scene_chain``.

    python scripts/marine/s1_port/sparse_vrt_on_reference_crop.py --record <record> \\
        --scene historical1 --pol vh --work-dir /tmp/sparse --out sparse_historical1_vh.json
"""
from __future__ import annotations

import argparse

import numpy as np

import _record as rec

# AOI inside AI2's reference window, window-local pixels (col0, row0, col1, row1).
AOI_LOCAL = (3000, 4000, 4400, 5100)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    rec.add_record_arg(ap)
    ap.add_argument("--scene", choices=tuple(rec.SCENE_LAYERS), required=True)
    ap.add_argument("--pol", choices=rec.POLS, required=True)
    ap.add_argument("--work-dir", type=rec.Path, required=True)
    ap.add_argument("--out", type=rec.Path)
    a = ap.parse_args()

    import rasterio
    from rasterio.windows import Window

    from imint.fetch import (S1_VESSEL_CHAIN_VERSION, _ai2_utm_epsg, _s1_grd_to_window_grid,
                             _s1_radar_window, _s1_sparse_vrt_xml)

    if S1_VESSEL_CHAIN_VERSION != rec.AI2_PIXEL_CHAIN:
        raise SystemExit(f"chain-{rec.AI2_PIXEL_CHAIN} check, this code is chain "
                         f"{S1_VESSEL_CHAIN_VERSION}: run it on a chain-1 checkout (see the "
                         f"docstring); full_scene_check.py covers chain 2")

    crop = a.record / "inputs" / f"{a.scene}_{a.pol}.tif"
    with rasterio.open(crop) as src:
        gcps, gcp_crs = src.gcps
        header = {"scene_id": f"crop:{a.scene}_{a.pol}", "width": src.width,
                  "height": src.height, "dtype": src.dtypes[0], "nodata": src.nodata,
                  "gcp_crs": str(gcp_crs),
                  "gcps": [[g.id, g.col, g.row, g.x, g.y, g.z] for g in gcps]}
    c0, r0, c1, r1 = window = rec.ai2_window_bounds(a.record)
    aoi_px = (c0 + AOI_LOCAL[0], r0 + AOI_LOCAL[1], c0 + AOI_LOCAL[2], r0 + AOI_LOCAL[3])
    rwin = _s1_radar_window(header, 32632, aoi_px)
    a.work_dir.mkdir(parents=True, exist_ok=True)
    win_tif = a.work_dir / f"{a.scene}_{a.pol}_win.tif"
    with rasterio.open(crop) as src:
        data = src.read(1, window=Window(rwin["col_off"], rwin["row_off"],
                                         rwin["width"], rwin["height"]))
    with rasterio.open(win_tif, "w", driver="GTiff", width=data.shape[1], height=data.shape[0],
                       count=1, dtype=data.dtype, nodata=header["nodata"], tiled=True,
                       blockxsize=512, blockysize=512, compress="zstd") as dst:
        dst.write(data, 1)
    vrt = a.work_dir / f"{a.scene}_{a.pol}_sparse.vrt"
    vrt.write_text(_s1_sparse_vrt_xml(header, rwin, win_tif.name))

    step1_epsg = _ai2_utm_epsg(gcps[0].x, gcps[0].y)
    sparse = _s1_grd_to_window_grid(vrt, step1_epsg, 32632, window, aoi_px)
    dense = _s1_grd_to_window_grid(crop, step1_epsg, 32632, window, aoi_px)
    with rasterio.open(rec.ai2_band_path(a.record, a.scene, a.pol)) as src:
        ai2 = src.read(1, window=Window(*AOI_LOCAL[:2], AOI_LOCAL[2] - AOI_LOCAL[0],
                                        AOI_LOCAL[3] - AOI_LOCAL[1]))
    rec.write_json(a.out, {
        "scene": a.scene, "pol": a.pol, "step1_epsg": step1_epsg, "aoi_px": list(aoi_px),
        "radar_window": rwin, "crop_px": [header["width"], header["height"]],
        "n": int(sparse.size),
        "n_diff_vs_dense_chain": int((sparse != dense).sum()),
        "n_diff_vs_ai2_materialised": int((sparse.astype(np.float32) != ai2).sum()),
    })


if __name__ == "__main__":
    main()
