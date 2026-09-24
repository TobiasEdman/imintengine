#!/usr/bin/env python3
"""Select pilot points by ROAD VERGE rather than by block density.

The first pilot picked the densest 0.1-degree blocks, which is why only
~2 % of its points ended up near a road with a real verge. Habitat then
had to be matched after the fact with a distance threshold, and the
threshold that was used (30 m from any OSM highway) silently admitted
forest edges along skogsbilvägar: NDVI of the negatives sat 0,039 above
the positives there, versus 0,008 at 10 m.

This selects on road class up front. Only the state network —
motorway/trunk/primary/secondary/tertiary — counts as a verge; track,
service and residential are excluded, which is exactly the contamination
the earlier filter let through. Both classes are then drawn from the same
verges, so habitat is matched by construction rather than by threshold.

Outputs, in ./data:
    verge_dist.json          metres to nearest verge road, per point
    chosen_points_verge.json the selection, schema-compatible with the
                             existing probe pipeline
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import requests as rq

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CACHE = HERE / "osm_verge_cache"
CACHE.mkdir(parents=True, exist_ok=True)

# State road network only. Excluding track|service|residential is the
# whole point: those are what made "within 30 m of a road" include
# spruce forest along a forest track.
VERGE_CLASSES = "motorway|trunk|primary|secondary|tertiary"
VERGE_LINK = "|motorway_link|trunk_link|primary_link|secondary_link"
OVERPASS = "https://overpass-api.de/api/interpreter"
HEADERS = {"User-Agent": "imint-lupin-study/0.1 (tobias.edman@ri.se)"}

TESSERA_YEARS = (2018, 2019, 2020, 2021, 2022, 2023, 2024)
VETO_M = 100.0        # negatives this close to any Lupinus record are out
PIXEL_M = 10          # TESSERA / S2 ground sample distance


def read_csv(name: str) -> list[dict]:
    with (DATA / name).open() as f:
        return [{"lat": float(r["lat"]), "lon": float(r["lon"]),
                 "year": int(r["year"]),
                 "month": int(r["month"]) if r["month"] else -1,
                 "speciesKey": r["speciesKey"]}
                for r in csv.DictReader(f)]


def cell_of(p: dict) -> tuple[int, int]:
    return (int(np.floor(p["lon"])), int(np.floor(p["lat"])))


def fetch_cell_roads(cell: tuple[int, int]) -> list[list]:
    """Verge-class road geometries for a 1-degree cell, cached on disk."""
    path = CACHE / f"verge_{cell[0]}_{cell[1]}.json"
    if path.exists():
        return json.loads(path.read_text())
    lon0, lat0 = cell
    q = (f'[out:json][timeout:180];'
         f'way["highway"~"^({VERGE_CLASSES}{VERGE_LINK})$"]'
         f"({lat0},{lon0},{lat0 + 1},{lon0 + 1});out geom;")
    for attempt in range(4):
        try:
            r = rq.post(OVERPASS, data={"data": q}, headers=HEADERS,
                        timeout=240)
            r.raise_for_status()
            ways = [[(nd["lon"], nd["lat"]) for nd in w.get("geometry", [])]
                    for w in r.json().get("elements", [])]
            ways = [w for w in ways if len(w) >= 2]
            path.write_text(json.dumps(ways))
            return ways
        except Exception as e:
            print(f"    cell {cell} försök {attempt+1}: {str(e)[:60]}",
                  flush=True)
            time.sleep(15 * (attempt + 1))
    path.write_text("[]")      # cache the miss so a rerun does not stall
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verge-max", type=float, default=15.0,
                    help="metres from a verge-class road centreline. 15 m "
                         "is roughly one pixel from the carriageway edge "
                         "on a two-lane state road.")
    ap.add_argument("--max-per-cell", type=int, default=120,
                    help="per class, to stop dense southern cells from "
                         "dominating the spatial folds")
    args = ap.parse_args()

    from pyproj import Transformer
    from scipy.spatial import cKDTree
    from shapely.geometry import LineString, Point
    from shapely.strtree import STRtree

    to3006 = Transformer.from_crs(4326, 3006, always_xy=True)

    pos = [p for p in read_csv("positives.csv") if p["year"] in TESSERA_YEARS]
    neg = [p for p in read_csv("negatives.csv") if p["year"] in TESSERA_YEARS]
    excl = read_csv("exclusion.csv")
    print(f"pool: {len(pos)} positiva, {len(neg)} negativa, "
          f"{len(excl)} i vetoset", flush=True)

    for group in (pos, neg, excl):
        for p in group:
            p["x"], p["y"] = to3006.transform(p["lon"], p["lat"])

    # Drop every Lupinus species from the negatives, not just polyphyllus:
    # L. nootkatensis and friends are spectrally lupine but carry a
    # different speciesKey, so a key-equality filter would keep them.
    genus_keys = {p["speciesKey"] for p in excl if p["speciesKey"]}
    before = len(neg)
    neg = [p for p in neg if p["speciesKey"] not in genus_keys]
    print(f"genusfilter: {before - len(neg)} negativa bort", flush=True)

    # Spatial veto: a verge pixel 60 m from a reported stand is very
    # likely lupine too, it just was not reported there.
    tree = cKDTree(np.array([(p["x"], p["y"]) for p in excl]))
    d, _ = tree.query(np.array([(p["x"], p["y"]) for p in neg]), k=1)
    for p, di in zip(neg, d):
        p["dist_lupin"] = float(di)
    neg = [p for p in neg if p["dist_lupin"] > VETO_M]
    for p in pos:
        p["dist_lupin"] = 0.0
    print(f"efter {VETO_M:.0f} m-veto: {len(neg)} negativa", flush=True)

    # --- verge distance, per 1-degree cell ---
    cells = sorted({cell_of(p) for p in pos} | {cell_of(p) for p in neg})
    print(f"hämtar vägnät för {len(cells)} celler", flush=True)
    by_cell = defaultdict(list)
    for p in pos + neg:
        by_cell[cell_of(p)].append(p)

    for i, cell in enumerate(cells, 1):
        ways = fetch_cell_roads(cell)
        pts = by_cell[cell]
        if not ways:
            for p in pts:
                p["verge_dist"] = float("inf")
            continue
        lines = [LineString([to3006.transform(lo, la) for lo, la in w])
                 for w in ways]
        rtree = STRtree(lines)
        for p in pts:
            pt = Point(p["x"], p["y"])
            g = rtree.nearest(pt)
            geom = lines[g] if isinstance(g, (int, np.integer)) else g
            p["verge_dist"] = float(pt.distance(geom))
        if i % 10 == 0 or i == len(cells):
            print(f"  [{i}/{len(cells)}] {cell}: {len(ways)} vägar, "
                  f"{len(pts)} punkter", flush=True)

    json.dump({f"{p['lon']:.6f},{p['lat']:.6f}": p["verge_dist"]
               for p in pos + neg if np.isfinite(p["verge_dist"])},
              (DATA / "verge_dist.json").open("w"))

    print(f"\n{'tröskel':>9s} {'lupin':>7s} {'negativ':>8s}")
    for thr in (10, 15, 20, 30, 50):
        np_ = sum(1 for p in pos if p["verge_dist"] <= thr)
        nn = sum(1 for p in neg if p["verge_dist"] <= thr)
        print(f"{thr:>7d} m {np_:7d} {nn:8d}")

    # --- selection: both classes from the verge, capped per cell ---
    sel_pos = [p for p in pos if p["verge_dist"] <= args.verge_max]
    sel_neg = [p for p in neg if p["verge_dist"] <= args.verge_max]

    # One point per 10 m pixel, positives claiming a pixel first: two
    # points in the same pixel share one embedding vector, and a pixel
    # carrying both labels is unusable rather than merely duplicated.
    taken: set = set()
    def dedup(points: list[dict]) -> list[dict]:
        out = []
        for p in points:
            k = (round(p["x"] / PIXEL_M), round(p["y"] / PIXEL_M))
            if k in taken:
                continue
            taken.add(k)
            out.append(p)
        return out
    sel_pos, sel_neg = dedup(sel_pos), dedup(sel_neg)

    rng = np.random.default_rng(42)
    chosen: list[dict] = []
    for label, pool in ((1, sel_pos), (0, sel_neg)):
        per_cell = defaultdict(list)
        for p in pool:
            per_cell[cell_of(p)].append(p)
        for cell, pts in per_cell.items():
            if len(pts) > args.max_per_cell:
                pts = list(rng.choice(pts, args.max_per_cell, replace=False))
            for p in pts:
                chosen.append({
                    "lon": p["lon"], "lat": p["lat"], "x": p["x"], "y": p["y"],
                    "year": p["year"], "month": p["month"], "label": label,
                    "block": f"{int(np.floor(p['lon']*10))}_"
                             f"{int(np.floor(p['lat']*10))}",
                    "dist_lupin": p["dist_lupin"],
                    "verge_dist": p["verge_dist"],
                })

    (DATA / "chosen_points_verge.json").write_text(json.dumps(chosen))
    y = np.array([c["label"] for c in chosen])
    yrs = Counter(c["year"] for c in chosen)
    print(f"\nvalt vid <={args.verge_max:g} m: {len(chosen)} punkter "
          f"({int(y.sum())} lupin, {int((y == 0).sum())} negativa)")
    print(f"block: {len({c['block'] for c in chosen})}, "
          f"celler: {len({(int(c['lon']), int(c['lat'])) for c in chosen})}")
    print("per år:", dict(sorted(yrs.items())))


if __name__ == "__main__":
    main()
