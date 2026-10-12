#!/usr/bin/env python3
"""Do two runs of AI2's reference pipeline give the same result? (run_ai2_reference.sh)

Compares two record roots: AI2's materialised model input (4 rasters, pixel for
pixel and grid) and ``run1/reference.geojson`` feature by feature, ignoring only
``crop_fname`` / ``crop_fnames`` (absolute paths of the PNG crops, which name the
run directory). Exit status 1 on any other difference.

    python scripts/marine/s1_port/compare_reference_runs.py --record <record> \\
        --other <re-run record> --out ai2_rerun_check.json
"""
from __future__ import annotations

import argparse
import json
import sys

import _record as rec

PATH_FIELDS = ("crop_fname", "crop_fnames")


def features(record: rec.Path) -> list[dict]:
    out = []
    for f in json.loads((record / "run1" / "reference.geojson").read_text())["features"]:
        props = {k: v for k, v in f["properties"].items() if k not in PATH_FIELDS}
        out.append({"geometry": f["geometry"], "properties": props})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    rec.add_record_arg(ap)
    ap.add_argument("--other", type=rec.Path, required=True, help="the second record root")
    ap.add_argument("--out", type=rec.Path)
    a = ap.parse_args()

    import rasterio

    rasters = {}
    for scene in rec.SCENE_LAYERS:
        for pol in rec.POLS:
            with rasterio.open(rec.ai2_band_path(a.record, scene, pol)) as x, \
                    rasterio.open(rec.ai2_band_path(a.other, scene, pol)) as y:
                same_grid = x.crs == y.crs and x.transform == y.transform and x.shape == y.shape
                rasters[f"{scene}_{pol}"] = {
                    "same_grid": bool(same_grid),
                    "n_diff": int((x.read(1) != y.read(1)).sum()) if same_grid else None}
    fa, fb = features(a.record), features(a.other)
    differing = [i for i, (x, y) in enumerate(zip(fa, fb)) if x != y]
    same_features = len(fa) == len(fb) and not differing
    same_rasters = all(r["same_grid"] and r["n_diff"] == 0 for r in rasters.values())
    rec.write_json(a.out, {"materialised_input": rasters, "n_features": [len(fa), len(fb)],
                           "features_identical_except_crop_paths": same_features,
                           "differing_feature_indices": differing})
    return 0 if same_features and same_rasters else 1


if __name__ == "__main__":
    sys.exit(main())
