#!/usr/bin/env python3
"""Post-hoc OSM road distances for the pilot points (plan B for D2).

Runs independently of the embedding extraction: reads chosen_points.json,
fetches OSM highways per 1-degree cell (disk-cached, resumable), computes
per-point distance to nearest road, and writes data/road_dist.json keyed
by "lon,lat" rounded to 6 decimals. probe_train.py picks it up if present.

Resumable: cells already in osm_cache/ are never refetched, so this can
be killed and rerun until all cells are in.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from extract_embeddings import fetch_cell_roads  # cached, retrying

HERE = Path(__file__).parent
DATA = HERE / "data"


def key(lon: float, lat: float) -> str:
    return f"{lon:.6f},{lat:.6f}"


def main() -> None:
    from pyproj import Transformer
    from shapely.geometry import LineString, Point
    from shapely.strtree import STRtree

    to3006 = Transformer.from_crs(4326, 3006, always_xy=True)
    pts = json.loads((DATA / "chosen_points.json").read_text())

    out_path = DATA / "road_dist.json"
    done: dict = json.loads(out_path.read_text()) if out_path.exists() else {}

    by_cell = defaultdict(list)
    for p in pts:
        if key(p["lon"], p["lat"]) in done:
            continue
        by_cell[(int(np.floor(p["lon"])), int(np.floor(p["lat"])))].append(p)
    print(f"{len(pts)} points, {len(done)} already done, "
          f"{len(by_cell)} cells remaining")

    for i, (cell, cpts) in enumerate(sorted(by_cell.items())):
        ways = fetch_cell_roads(cell)
        if not ways:
            print(f"[{i+1}/{len(by_cell)}] cell {cell}: FAILED, skipping "
                  f"({len(cpts)} pts stay unresolved)", flush=True)
            continue
        lines = [LineString([to3006.transform(lo, la) for lo, la in w])
                 for w in ways]
        rtree = STRtree(lines)
        for p in cpts:
            pt = Point(p["x"], p["y"])
            nearest = rtree.nearest(pt)
            geom = lines[nearest] if isinstance(nearest, (int, np.integer)) \
                else nearest
            done[key(p["lon"], p["lat"])] = round(float(pt.distance(geom)), 1)
        out_path.write_text(json.dumps(done))  # save after EVERY cell
        print(f"[{i+1}/{len(by_cell)}] cell {cell}: {len(ways)} ways, "
              f"{len(cpts)} pts -> saved ({len(done)} total)", flush=True)

    print(f"done: {len(done)}/{len(pts)} points have road distance")


if __name__ == "__main__":
    main()
