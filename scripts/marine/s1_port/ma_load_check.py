#!/usr/bin/env python3
"""Read an S1 GeoJSON with marine-awareness's own ``load_detections`` (cross-repo seam).

The reader is taken read-only from a marine-awareness checkout with
``git show <ref>:src/marine_awareness/detect/ais_match.py`` into a temp dir and
imported from there; the marine-awareness repo is never modified. Checks that
every feature is read, that ``score`` and ``pixel_count`` round-trip, how far
the position it reads (ring vertex mean) is from the box centre and AI2's point,
that the rings are counter-clockwise (RFC 7946) and headings lie in [0, 360).

    python scripts/marine/s1_port/ma_load_check.py surv_aoi_2026-06-28_s1.geojson run_c.json \\
        --ma-repo ../marine-awareness --out ma_load_check.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile

import numpy as np

import _record as rec

READER = "src/marine_awareness/detect/ais_match.py"


def load_reader(repo: rec.Path, ref: str):
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", ref], check=True,
                         capture_output=True, text=True).stdout.strip()
    src = subprocess.run(["git", "-C", str(repo), "show", f"{ref}:{READER}"], check=True,
                         capture_output=True, text=True).stdout
    path = rec.Path(tempfile.mkdtemp()) / "ais_match_reader.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("ais_match_reader", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module          # dataclasses resolve annotations through it
    spec.loader.exec_module(module)
    return module, sha


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("geojson", type=rec.Path)
    ap.add_argument("run", type=rec.Path, help="--run-json of scripts/detect_vessels_s1.py")
    ap.add_argument("--ma-repo", type=rec.Path, required=True)
    ap.add_argument("--ma-ref", default="origin/main")
    ap.add_argument("--out", type=rec.Path)
    a = ap.parse_args()

    from pyproj import Transformer

    from imint.exporters.export import s1_regions_in_bbox
    from imint.utils import haversine_m

    reader, sha = load_reader(a.ma_repo, a.ma_ref)
    dets = reader.load_detections(a.geojson)
    fc = json.loads(a.geojson.read_text())
    run = json.loads(a.run.read_text())
    regions = s1_regions_in_bbox(run["outputs"]["regions"], fc["properties"]["bbox"])
    prov = run["provenance"]
    c0, r0 = prov["aoi_px"][:2]
    to_wgs = Transformer.from_crs(f"EPSG:{prov['window_epsg']}", 4326, always_xy=True).transform
    if not len(dets) == len(fc["features"]) == len(regions):
        raise SystemExit(f"count mismatch: load_detections {len(dets)}, features "
                         f"{len(fc['features'])}, regions in bbox {len(regions)}")
    off_box, off_ai2 = [], []
    for det, region, feat in zip(dets, regions, fc["features"], strict=True):
        x1, y1, x2, y2 = region["box_px"]
        lon, lat = to_wgs((c0 + (x1 + x2) / 2) * 10, (r0 + (y1 + y2) / 2) * -10)
        off_box.append(float(haversine_m(det.lon, det.lat, lon, lat)))
        off_ai2.append(float(haversine_m(det.lon, det.lat, region["lon"], region["lat"])))
        if det.score != round(region["score"], 4):
            raise SystemExit(f"score not round-tripped: {det.score} vs {region['score']}")
        if not det.pixel_count == region["pixel_count"] == feat["properties"]["pixel_count"] > 0:
            raise SystemExit(f"pixel_count not round-tripped: {det.pixel_count} vs "
                             f"{region['pixel_count']}")
    rings = [np.array(f["geometry"]["coordinates"][0])[:-1] for f in fc["features"]]
    signed_area = [0.5 * np.sum(r[:, 0] * np.roll(r[:, 1], -1) - np.roll(r[:, 0], -1) * r[:, 1])
                   for r in rings]
    headings = [f["properties"]["pred_heading_deg"] for f in fc["features"]
                if f["properties"]["pred_heading_deg"] is not None]
    rec.write_json(a.out, {
        "marine_awareness_ref": f"{a.ma_ref} ({sha})",
        "rings_counter_clockwise": all(area > 0 for area in signed_area),
        "pred_heading_deg_min_max": [min(headings), max(headings)],
        "n_load_detections": len(dets), "n_features": len(fc["features"]),
        "n_outside_bbox_dropped": fc["properties"]["n_outside_bbox"],
        "score_and_pixel_count_round_trip": True,
        "centroid_vs_box_centre_m_max": max(off_box),
        "centroid_vs_ai2_point_m_median_max": [float(np.median(off_ai2)), max(off_ai2)],
        "scene_time_utc": fc["properties"]["scene_time_utc"],
    })


if __name__ == "__main__":
    main()
