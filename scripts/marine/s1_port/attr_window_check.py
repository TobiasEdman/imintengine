#!/usr/bin/env python3
"""Analyzer attribute crops vs AI2-style separately warped windows (dossier: Attribute windows).

AI2 materialises each 128 px attribute window with its own cubic warp from the
step-1 (first-GCP-zone) raster; the analyzer crops the detector-window raster.
For the full-scene 33 -> 32 chain the two can differ. Probe: AI2's reference
detections at least 64 px inside the port AOI.

    python scripts/marine/s1_port/attr_window_check.py --record <record> \\
        --port-input <cache>/s1_vessel_inputs/2026-06-28_<key>.tif --cache-dir <cache> \\
        --out attr_window_check.json
"""
from __future__ import annotations

import argparse

import numpy as np

import _record as rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    rec.add_record_arg(ap)
    ap.add_argument("--port-input", type=rec.Path, required=True)
    ap.add_argument("--cache-dir", type=rec.Path, required=True,
                    help="the fetch cache holding s1_grd/<scene>/ radar windows + VRTs")
    ap.add_argument("--out", type=rec.Path)
    a = ap.parse_args()

    import rasterio
    import torch
    from rasterio.enums import Resampling
    from rasterio.io import MemoryFile
    from rasterio.transform import Affine
    from rasterio.vrt import WarpedVRT

    from imint.analyzers import ai2_vessels_s1 as s1
    from imint.analyzers.ai2_rslearn_blocks import crop_centered

    port, (pc0, pr0), meta = rec.read_port_input(a.port_input)
    prov, tid = meta["provenance"], meta["target_scene"]["scene_id"]
    target = port[:2]
    h, w = target.shape[1:]
    half = s1.ATTR_CROP_PX // 2
    pts = [(p["col"], p["row"]) for p in rec.reference_properties(a.record)
           if half <= p["col"] - pc0 <= w - half and half <= p["row"] - pr0 <= h - half]
    separate = {i: [] for i in range(len(pts))}
    for pol in rec.POLS:
        rw = prov["radar_windows"][f"{tid}/{pol}"]
        vrt = (a.cache_dir / "s1_grd" / tid /
               f"{pol}_c{rw['col_off']}_r{rw['row_off']}_{rw['width']}x{rw['height']}.vrt")
        with rasterio.open(vrt) as src, WarpedVRT(src, crs=f"EPSG:{rw['step1_epsg']}",
                                                  resampling=Resampling.cubic,
                                                  MAX_GCP_ORDER="-1") as v:
            step1 = v.read()
            prof = dict(driver="GTiff", width=v.width, height=v.height, count=1, dtype="uint16",
                        crs=v.crs, transform=v.transform, nodata=src.nodata, tiled=True,
                        SPARSE_OK=True)
        with MemoryFile() as mf:
            with mf.open(**prof) as t:
                t.write(step1)
            del step1
            with mf.open() as t:
                for i, (c, r) in enumerate(pts):
                    with WarpedVRT(t, crs=f"EPSG:{prov['window_epsg']}",
                                   transform=Affine(10, 0, (c - half) * 10, 0, -10, (r - half) * -10),
                                   width=s1.ATTR_CROP_PX, height=s1.ATTR_CROP_PX,
                                   resampling=Resampling.cubic) as v:
                        separate[i].append(v.read(1))
    crops = np.stack([crop_centered(target, r - pr0, c - pc0, s1.ATTR_CROP_PX) for c, r in pts])
    seps = np.stack([np.stack(separate[i]) for i in range(len(pts))])
    d = np.abs(crops.astype(np.float32) - seps.astype(np.float32))
    model = s1.load_attribute_model(s1.DEFAULT_ATTR_CKPT)

    def attrs(x: np.ndarray) -> list[dict]:
        out = []
        with torch.inference_mode():
            for i in range(0, len(x), s1.ATTR_BATCH_SIZE):
                batch = torch.from_numpy(s1.normalize_attribute(x[i:i + s1.ATTR_BATCH_SIZE]))
                raw = {k: v.numpy() for k, v in s1.attribute_forward(model, batch).items()}
                out += s1.decode_attributes(raw)
        return out

    a_crop, a_sep = attrs(crops), attrs(seps)

    def med_max(values) -> list[float]:
        return [float(np.median(values)), float(np.max(values))]

    pairs = list(zip(a_crop, a_sep, strict=True))
    rec.write_json(a.out, {
        "n": len(pts), "px_equal_frac": float((d == 0).mean()), "px_mean_abs": float(d.mean()),
        "px_max_abs": float(d.max()),
        "length_m": med_max([abs(x["length_m"] - y["length_m"]) for x, y in pairs]),
        "rel_length": med_max([abs(x["length_m"] - y["length_m"]) / y["length_m"] for x, y in pairs]),
        "width_m": med_max([abs(x["width_m"] - y["width_m"]) for x, y in pairs]),
        "speed_kn": med_max([abs(x["speed_knots"] - y["speed_knots"]) for x, y in pairs]),
        "heading_deg": med_max([abs((x["heading_deg"] - y["heading_deg"] + 180) % 360 - 180)
                                for x, y in pairs]),
        "type_agree": sum(x["vessel_type"] == y["vessel_type"] for x, y in pairs),
    })


if __name__ == "__main__":
    main()
