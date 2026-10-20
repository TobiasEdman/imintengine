#!/usr/bin/env python3
"""Model-input comparison, dossier table (a).

Part 1 (``--part1``): the production warp code (``imint.fetch._s1_grd_to_window_grid``)
on AI2's own crop inputs and window must equal AI2's materialised rasters bit
for bit. Part 2: the production fetch (full-scene chain) against AI2's rasters
on the same pixels, differences quantified.

    python scripts/marine/s1_port/compare_inputs.py --record <record> \\
        --port-input <cache>/s1_vessel_inputs/2026-06-28_<key>.tif --part1 --out cmp_inputs.json
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import _record as rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    rec.add_record_arg(ap)
    ap.add_argument("--port-input", type=rec.Path, required=True)
    ap.add_argument("--part1", action="store_true")
    ap.add_argument("--out", type=rec.Path)
    a = ap.parse_args()

    import rasterio

    from imint.fetch import _ai2_utm_epsg, _s1_grd_to_window_grid

    ai2, (c0, r0, c1, r1) = rec.read_ai2_stack(a.record)
    names = [(scene, pol) for scene in rec.SCENE_LAYERS for pol in rec.POLS]
    out = {}
    if a.part1:
        for i, (scene, pol) in enumerate(names):
            src = a.record / "inputs" / f"{scene}_{pol}.tif"
            with rasterio.open(src) as s:
                g = s.gcps[0][0]
            step1_epsg = _ai2_utm_epsg(g.x, g.y)
            t = time.time()
            mine = _s1_grd_to_window_grid(src, step1_epsg, 32632, (c0, r0, c1, r1), (c0, r0, c1, r1))
            d = mine.astype(np.float32) != ai2[i]
            out[f"part1_{scene}_{pol}"] = {"n_diff": int(d.sum()), "n": int(d.size),
                                           "seconds": round(time.time() - t, 1),
                                           "step1_epsg": step1_epsg}
            print(f"part1_{scene}_{pol}", out[f"part1_{scene}_{pol}"], flush=True)
    port, (pc0, pr0), _ = rec.read_port_input(a.port_input, chain=rec.AI2_PIXEL_CHAIN)
    for i, (scene, pol) in enumerate(names):
        p = port[i].astype(np.float64)
        ref = ai2[i, pr0 - r0:pr0 - r0 + p.shape[0], pc0 - c0:pc0 - c0 + p.shape[1]].astype(np.float64)
        both = (p > 0) & (ref > 0)
        d = np.abs(p - ref)[both]
        out[f"part2_{scene}_{pol}"] = {
            "n_px": int(both.sum()), "valid_mismatch": int(((p > 0) != (ref > 0)).sum()),
            "exact_equal_frac": float((d == 0).mean()), "mean_abs_dn": float(d.mean()),
            "p50_abs_dn": float(np.median(d)), "p99_abs_dn": float(np.percentile(d, 99)),
            "max_abs_dn": float(d.max()), "median_rel": float(np.median(d / ref[both])),
            "mean_signed_dn": float((p[both] - ref[both]).mean()),
            "corr_log": float(np.corrcoef(np.log(p[both]), np.log(ref[both]))[0, 1]),
            "ref_median_dn": float(np.median(ref[both]))}
    rec.write_json(a.out, out)


if __name__ == "__main__":
    main()
