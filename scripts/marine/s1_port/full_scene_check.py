#!/usr/bin/env python3
"""Ground truth for "the sparse fetch equals the full-scene chain" (dossier: Fetch design).

Downloads the WHOLE target VV GRD from MPC once (~420 MB, kept at ``--full-tif``),
runs the same two warps on it without any sparsity, and compares inside the AOI
with the fetch's cached model input (target VV, band 1).

The full scene gets the chain's own GCPs: a VRT over the full TIFF written by
the fetch's ``_s1_sparse_vrt_xml`` with the "radar window" set to the whole
raster, from the scene header in the fetch's cache (chain 2: sea-surface GCPs;
the TIFF's own GCPs are not used). The port input must come from this code's
chain and geocoding model, else the script refuses.

    python scripts/marine/s1_port/full_scene_check.py \\
        --port-input <cache>/s1_vessel_inputs/2026-06-28_<key>.tif \\
        --full-tif <record>/s1port/full_target_vv.tif --work-dir <tmp> --out full_scene_check.json
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
    from imint.data import mpc_s1, s1_shared

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
    ap.add_argument("--work-dir", type=rec.Path, required=True,
                    help="where the full-scene VRT with the chain's GCPs is written")
    ap.add_argument("--out", type=rec.Path)
    a = ap.parse_args()

    import json

    import rasterio

    from imint.fetch import (
        S1_VESSEL_CHAIN_VERSION,
        _s1_geocoding_model,
        _s1_grd_to_window_grid,
        _s1_sparse_vrt_xml,
    )

    port, _, meta = rec.read_port_input(a.port_input, chain=S1_VESSEL_CHAIN_VERSION)
    prov = meta["provenance"]
    if prov.get("geocoding") != _s1_geocoding_model():
        raise SystemExit(f"{a.port_input}: geocoding {prov.get('geocoding')} is not this "
                         f"code's {_s1_geocoding_model()}; re-run the fetch")
    item_id = meta["target_scene"]["scene_id"]
    header = json.loads((a.port_input.parent.parent / "s1_grd" / item_id / "vv_header.json")
                        .read_text())
    if not a.full_tif.exists():
        req = prov["request"]
        download_full_vv(item_id, req["date"],
                         [req["coords"][k] for k in ("west", "south", "east", "north")], a.full_tif)
    with rasterio.open(a.full_tif) as src:
        assert (src.width, src.height) == (header["width"], header["height"]), a.full_tif
    whole = {"col_off": 0, "row_off": 0, "width": header["width"], "height": header["height"]}
    a.work_dir.mkdir(parents=True, exist_ok=True)
    vrt = a.work_dir / f"{item_id}_vv_full.vrt"
    vrt.write_text(_s1_sparse_vrt_xml(header, whole, str(a.full_tif.resolve())))
    t = time.time()
    full = _s1_grd_to_window_grid(vrt, prov["radar_windows"][f"{item_id}/vv"]["step1_epsg"],
                                  prov["window_epsg"], tuple(prov["ai2_window_px"]),
                                  tuple(prov["aoi_px"]))
    d = full != port[0]
    rec.write_json(a.out, {"chain_version": prov["chain_version"],
                           "geocoding": prov["geocoding"],
                           "n_diff": int(d.sum()), "n": int(d.size),
                           "max_abs": int(np.abs(full.astype(int) - port[0].astype(int)).max()),
                           "seconds_full_chain": round(time.time() - t, 1)})


if __name__ == "__main__":
    main()
