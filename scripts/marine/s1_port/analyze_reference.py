#!/usr/bin/env python3
"""Summarise AI2's reference run: counts, scores, attributes, filter effects (dossier g, h).

    python scripts/marine/s1_port/analyze_reference.py --record <record> --out reference_summary.json
"""
from __future__ import annotations

import argparse
import collections
import json

import numpy as np

import _record as rec


def min_median_max(values: list[float]) -> list[float]:
    return [float(np.min(values)), float(np.median(values)), float(np.max(values))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    rec.add_record_arg(ap)
    ap.add_argument("--bbox", nargs=4, type=float, default=rec.SKAGEN_BBOX)
    ap.add_argument("--out", type=rec.Path)
    a = ap.parse_args()

    from pyproj import Transformer
    from shapely.geometry import Point, box

    from imint.utils import haversine_m

    props = rec.reference_properties(a.record)
    s = np.array([p["score"] for p in props])
    att = [p["attributes"] for p in props]
    raw = json.loads((rec.ai2_window_dir(a.record) / "layers" / "output" / "data.geojson")
                     .read_text())["features"]
    infra = json.loads((a.record / "inputs" / "marine_infrastructure.geojson").read_text())
    ilonlat = np.array([f["geometry"]["coordinates"][:2] for f in infra["features"]])
    meta = json.loads((rec.ai2_window_dir(a.record) / "metadata.json").read_text())
    to_wgs = Transformer.from_crs(meta["projection"]["crs"], 4326, always_xy=True).transform
    res = meta["projection"]["x_resolution"]
    raw_pts, sizes = [], []
    for f in raw:            # post-NMS detector output, before the edge drop and infra filter
        xs = [c[0] for c in f["geometry"]["coordinates"][0]]
        ys = [c[1] for c in f["geometry"]["coordinates"][0]]
        col, row = int((min(xs) + max(xs)) / 2), int((min(ys) + max(ys)) / 2)
        raw_pts.append((col, row, *to_wgs(col * res, row * -res)))
        sizes.append((max(xs) - min(xs), max(ys) - min(ys)))
    d_inf_km = np.array([haversine_m(lon, lat, ilonlat[:, 0], ilonlat[:, 1]).min() / 1000
                         for _, _, lon, lat in raw_pts])
    final = {(p["col"], p["row"]) for p in props}
    sizes = np.array(sizes)
    rec.write_json(a.out, {
        "n_final": len(props),
        "n_in_bbox": sum(box(*a.bbox).contains(Point(p["longitude"], p["latitude"]))
                         for p in props),
        "n_raw_post_nms": len(raw),
        "n_removed_by_filters": sum((c, r) not in final for c, r, *_ in raw_pts),
        "score_quantiles": dict(zip(["min", "p25", "p50", "p75", "max"],
                                    [float(s.min()), *np.percentile(s, [25, 50, 75]), float(s.max())])),
        "score_hist_0.70_1.00_step_0.05": np.histogram(s, bins=np.arange(0.70, 1.0001, 0.05))[0].tolist(),
        "attributes_min_median_max": {k: min_median_max([x[k] for x in att])
                                      for k in ("length", "width", "speed", "heading")},
        "vessel_types": dict(collections.Counter(x["vessel_type"] for x in att).most_common()),
        "nearest_infra_km_min": float(d_inf_km.min()),
        "n_within_infra_km": {str(t): int((d_inf_km < t).sum()) for t in (0.05, 0.1, 0.2, 0.5, 1.0)},
        "raw_box_px_w_h_min_max": [float(sizes[:, 0].min()), float(sizes[:, 0].max()),
                                   float(sizes[:, 1].min()), float(sizes[:, 1].max())],
        "scene_id_ts_in_output": [sorted({str(p["scene_id"]) for p in props}),
                                  sorted({str(p["ts"]) for p in props})],
    })


if __name__ == "__main__":
    main()
