#!/usr/bin/env python3
"""Ground truth for "the sparse fetch equals the full-scene chain" (dossier: Fetch design).

Downloads the WHOLE target VV GRD from MPC once (~420 MB, kept at ``--full-tif``),
runs the same two warps on it without any sparsity, and compares inside the AOI
with the fetch's cached model input (target VV, band 1).

    python scripts/marine/s1_port/full_scene_check.py \\
        --port-input <cache>/s1_vessel_inputs/2026-06-28_<key>.tif \\
        --full-tif <record>/s1port/full_target_vv.tif --out full_scene_check.json
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

import _record as rec


def download_full_vv(item_id: str, date: str, bbox, path: rec.Path) -> None:
    import rasterio

    from imint.fetch import _S1_GDAL_ENV
    from imint.training import mpc_s1, s1_shared

    t = time.time()
    (item,) = [it for it in mpc_s1.search_grd_items(f"{date}T00:00:00Z/{date}T23:59:59Z",
                                                    bbox_4326=tuple(bbox)) if it.id == item_id]
    href = s1_shared.pick_measurement_urls(mpc_s1.sign_item(item))[0]
    with rasterio.Env(**_S1_GDAL_ENV), rasterio.open(href) as src:
        data = src.read(1)
        gcps, gcp_crs = src.gcps
        profile = dict(driver="GTiff", width=src.width, height=src.height, count=1,
                       dtype="uint16", nodata=src.nodata, compress="zstd", tiled=True,
                       blockxsize=1024, blockysize=1024, BIGTIFF="YES", gcps=gcps, crs=gcp_crs)
    tmp = path.with_name(path.name + ".part")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(data, 1)
    os.replace(tmp, path)
    print("downloaded the full VV GRD in", round(time.time() - t, 1), "s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port-input", type=rec.Path, required=True)
    ap.add_argument("--full-tif", type=rec.Path, required=True)
    ap.add_argument("--out", type=rec.Path)
    a = ap.parse_args()

    from imint.fetch import _s1_grd_to_window_grid

    port, _, meta = rec.read_port_input(a.port_input)
    prov = meta["provenance"]
    item_id = meta["target_scene"]["scene_id"]
    if not a.full_tif.exists():
        req = prov["request"]
        download_full_vv(item_id, req["date"],
                         [req["coords"][k] for k in ("west", "south", "east", "north")], a.full_tif)
    t = time.time()
    full = _s1_grd_to_window_grid(a.full_tif, prov["radar_windows"][f"{item_id}/vv"]["step1_epsg"],
                                  prov["window_epsg"], tuple(prov["ai2_window_px"]),
                                  tuple(prov["aoi_px"]))
    d = full != port[0]
    rec.write_json(a.out, {"n_diff": int(d.sum()), "n": int(d.size),
                           "max_abs": int(np.abs(full.astype(int) - port[0].astype(int)).max()),
                           "seconds_full_chain": round(time.time() - t, 1)})


if __name__ == "__main__":
    main()
