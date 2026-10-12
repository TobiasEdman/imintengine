#!/usr/bin/env python3
"""Are two runs (and optionally their cached model inputs) identical?

For refactors that must not change results: every region of ``run_b`` must
equal the one at the same index of ``run_a`` exactly (boxes, scores, points,
lon/lat, attributes), and ``--inputs`` tifs must be bit-identical (pixels,
dtype, grid). Exit status 1 on any difference.

    python scripts/marine/s1_port/parity_check.py run_c_before.json run_c_after.json \\
        --inputs before.tif after.tif --out parity.json
"""
from __future__ import annotations

import argparse
import json
import sys

import _record as rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_a", type=rec.Path)
    ap.add_argument("run_b", type=rec.Path)
    ap.add_argument("--inputs", nargs=2, type=rec.Path)
    ap.add_argument("--out", type=rec.Path)
    a = ap.parse_args()

    ra, rb = (json.loads(p.read_text())["outputs"]["regions"] for p in (a.run_a, a.run_b))
    differing = [i for i, (x, y) in enumerate(zip(ra, rb)) if x != y]
    out = {"n_regions": [len(ra), len(rb)], "regions_identical": len(ra) == len(rb) and not differing,
           "differing_region_indices": differing}
    if a.inputs:
        import numpy as np
        import rasterio

        meta = []
        arrays = []
        for path in a.inputs:
            with rasterio.open(path) as src:
                arrays.append(src.read())
                meta.append((src.crs, src.transform, src.dtypes, src.nodata))
        out["inputs_identical"] = bool(meta[0] == meta[1] and arrays[0].shape == arrays[1].shape
                                       and np.array_equal(arrays[0], arrays[1]))
        out["inputs_shape_dtype"] = [list(arrays[0].shape), str(arrays[0].dtype)]
    rec.write_json(a.out, out)
    return 0 if out["regions_identical"] and out.get("inputs_identical", True) else 1


if __name__ == "__main__":
    sys.exit(main())
