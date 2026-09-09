#!/usr/bin/env python3
"""Extract TESSERA 128-D embeddings at pilot label points — v3.

Review fixes baked in (savant B1-B5/W1-W2 + domain-reviewer 1-3):
    B1/B2  dedup to one point per 10 m SWEREF99 pixel within class; drop
           negatives sharing a pixel with any positive.
    B3     veto negatives against the WHOLE genus Lupinus (speciesKey set
           + spatial), not just L. polyphyllus.
    W1     hard assert len(embeddings) == len(points) per chunk.
    W2/D1  every point sampled at its OWN observation year (no modal year).
    D3     per-negative distance to nearest Lupinus record is SAVED, so the
           probe can sweep the veto threshold (100/250/500 m); only a
           minimal 100 m veto is applied here.
    D2     per-point distance to nearest OSM road is computed (Overpass,
           cached per block) → enables the habitat-matched control subset.

Output: data/features.npz with X (N,128), y, lat, lon, year, month,
block, dist_lupin (m; 0 for positives), dist_road (m; NaN if OSM failed).
Uses geotessera 0.10.2 (data.source.coop host).
"""
from __future__ import annotations

import argparse
import csv
import json
import signal
import socket
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests as rq

# geotessera issues HTTP calls without timeouts; a dead socket then hangs
# the process forever (observed: 55 min, 0.02 s CPU, no connections).
# A global default makes every socket operation fail loudly instead.
socket.setdefaulttimeout(120)


class _Timeout(Exception):
    pass


def _alarm(_sig, _frm):
    raise _Timeout("chunk exceeded wall-clock budget")


signal.signal(signal.SIGALRM, _alarm)

HERE = Path(__file__).parent
DATA = HERE / "data"
OSM_CACHE = HERE / "osm_cache"
OSM_CACHE.mkdir(exist_ok=True)

TESSERA_YEARS = (2018, 2019, 2020, 2021, 2022, 2023, 2024)
BASE_VETO_M = 100.0
MAX_PER_BLOCK = 20
MIN_POS = MIN_NEG = 3
CHUNK = 150

OVERPASS_ENDPOINTS = (  # kumi first: overpass-api.de 429s on bulk
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
)
# Overpass rejects requests without a descriptive UA (406/504)
OSM_HEADERS = {"User-Agent": "imint-lupin-pilot/0.1 (tobias.edman@ri.se)"}
HIGHWAY_RE = ("motorway|trunk|primary|secondary|tertiary|unclassified|"
              "residential|track|service")


def read_csv(name: str) -> list[dict]:
    with (DATA / name).open() as f:
        return [
            {"lat": float(r["lat"]), "lon": float(r["lon"]),
             "year": int(r["year"]),
             "month": int(r["month"]) if r["month"] else -1,
             "speciesKey": r["speciesKey"]}
            for r in csv.DictReader(f)
        ]


def block_key(p: dict) -> tuple[int, int]:
    return (int(np.floor(p["lon"] * 10)), int(np.floor(p["lat"] * 10)))


def dedup_by_pixel(pts: list[dict], taken: set) -> list[dict]:
    """One point per 10 m SWEREF99 pixel; also skips pixels in `taken`."""
    out = []
    for p in pts:
        k = (round(p["x"] / 10), round(p["y"] / 10))
        if k in taken:
            continue
        taken.add(k)
        out.append(p)
    return out


def fetch_cell_roads(cell: tuple[int, int]) -> list[list[tuple[float, float]]]:
    """OSM highway geometries for a 1-degree cell, cached on disk.

    One query per 1-degree cell (~20 total) instead of per 0.1-degree
    block (100): bulk Overpass behaves far better with few large queries
    than many small ones (429s on overpass-api.de, read timeouts on kumi).
    """
    cache = OSM_CACHE / f"roads_cell_{cell[0]}_{cell[1]}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    lon0, lat0 = float(cell[0]), float(cell[1])
    q = (f'[out:json][timeout:300];way["highway"~"^({HIGHWAY_RE})$"]'
         f"({lat0},{lon0},{lat0 + 1},{lon0 + 1});out geom;")
    for attempt in range(4):
        url = OVERPASS_ENDPOINTS[attempt % len(OVERPASS_ENDPOINTS)]
        try:
            r = rq.post(url, data={"data": q}, headers=OSM_HEADERS,
                        timeout=330)
            r.raise_for_status()
            ways = [[(nd["lon"], nd["lat"]) for nd in w.get("geometry", [])]
                    for w in r.json().get("elements", [])]
            ways = [w for w in ways if len(w) >= 2]
            cache.write_text(json.dumps(ways))
            return ways
        except Exception as e:
            print(f"  OSM cell {cell} attempt {attempt+1} "
                  f"({url.split('/')[2]}): {str(e)[:80]}", flush=True)
            time.sleep(10 * (attempt + 1))
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-blocks", type=int, default=100)
    ap.add_argument("--cache-dir", default=str(HERE / "tessera_cache"))
    ap.add_argument("--skip-osm", action="store_true")
    ap.add_argument("--select-only", action="store_true",
                    help="write chosen_points/point_years and exit (no network)")
    ap.add_argument("--embeddings-dir", default=str(HERE),
                    help="where prefetch_tiles.py put the tiles")
    ap.add_argument("--offline", action="store_true",
                    help="sample with auto_download=False (tiles prefetched)")
    args = ap.parse_args()

    from pyproj import Transformer
    from scipy.spatial import cKDTree
    from shapely.geometry import LineString, Point
    from shapely.strtree import STRtree

    to3006 = Transformer.from_crs(4326, 3006, always_xy=True)

    def proj(pts: list[dict]) -> None:
        for p in pts:
            p["x"], p["y"] = to3006.transform(p["lon"], p["lat"])

    pos = [p for p in read_csv("positives.csv") if p["year"] in TESSERA_YEARS]
    neg = [p for p in read_csv("negatives_raw.csv")
           if p["year"] in TESSERA_YEARS]
    excl = read_csv("exclusion_genus.csv")
    print(f"loaded: {len(pos)} pos, {len(neg)} neg_raw, "
          f"{len(excl)} genus-exclusion")

    # B3: drop ANY Lupinus species from negatives (speciesKey set from the
    # genus fetch — covers nootkatensis etc., not just polyphyllus).
    genus_keys = {p["speciesKey"] for p in excl if p["speciesKey"]}
    n0 = len(neg)
    neg = [p for p in neg if p["speciesKey"] not in genus_keys]
    print(f"B3 genus filter: dropped {n0 - len(neg)} negative rows")

    proj(pos), proj(neg), proj(excl)

    # B1+B2: pixel dedup, positives claim pixels first
    taken: set = set()
    pos = dedup_by_pixel(pos, taken)
    neg = dedup_by_pixel(neg, taken)
    print(f"after pixel dedup: {len(pos)} pos, {len(neg)} neg")

    # D3: distance to nearest Lupinus record per negative (saved for sweep)
    tree = cKDTree(np.array([(p["x"], p["y"]) for p in excl]))
    dist, _ = tree.query(np.array([(p["x"], p["y"]) for p in neg]), k=1)
    for p, di in zip(neg, dist):
        p["dist_lupin"] = float(di)
    neg = [p for p in neg if p["dist_lupin"] > BASE_VETO_M]
    for p in pos:
        p["dist_lupin"] = 0.0
    print(f"negatives after base {BASE_VETO_M} m veto: {len(neg)}")

    # block selection (bounded download budget)
    pos_by_b, neg_by_b = defaultdict(list), defaultdict(list)
    for p in pos:
        pos_by_b[block_key(p)].append(p)
    for p in neg:
        neg_by_b[block_key(p)].append(p)
    cands = [(len(v), b) for b, v in pos_by_b.items()
             if len(v) >= MIN_POS and len(neg_by_b[b]) >= MIN_NEG]
    cands.sort(reverse=True)
    blocks = [b for _, b in cands[: args.max_blocks]]
    print(f"blocks selected: {len(blocks)} of {len(cands)} candidates")

    # per-block capped sample; W2/D1: bucket by the POINT'S OWN year
    rng = np.random.default_rng(42)
    chosen: list[dict] = []
    for b in blocks:
        for label, pool in ((1, pos_by_b[b]), (0, neg_by_b[b])):
            sel = (list(rng.choice(pool, MAX_PER_BLOCK, replace=False))
                   if len(pool) > MAX_PER_BLOCK else pool)
            for p in sel:
                chosen.append({**p, "label": label,
                               "block": f"{b[0]}_{b[1]}"})

    # Dump chosen points so road distances (compute_road_dist.py) and tile
    # prefetch (prefetch_tiles.py) can run independently of this process.
    (DATA / "chosen_points.json").write_text(json.dumps(
        [{k: p[k] for k in ("lon", "lat", "x", "y", "label", "block",
                            "year", "month")} for p in chosen]))
    (DATA / "point_years.json").write_text(
        json.dumps([p["year"] for p in chosen]))
    if args.select_only:
        print(f"select-only: wrote {len(chosen)} points; exiting")
        return

    # D2: OSM road distance per chosen point, one query per 1-degree cell
    if not args.skip_osm:
        pts_by_cell = defaultdict(list)
        for p in chosen:
            pts_by_cell[(int(np.floor(p["lon"])),
                         int(np.floor(p["lat"])))].append(p)
        print(f"OSM: {len(pts_by_cell)} 1-degree cells to fetch", flush=True)
        for i, (cell, pts) in enumerate(sorted(pts_by_cell.items())):
            ways = fetch_cell_roads(cell)
            if not ways:
                for p in pts:
                    p["dist_road"] = float("nan")
                print(f"  OSM: {i+1}/{len(pts_by_cell)} cells "
                      f"(cell {cell}: NO ROADS/FAILED)", flush=True)
                continue
            lines = [LineString([to3006.transform(lo, la) for lo, la in w])
                     for w in ways]
            rtree = STRtree(lines)
            for p in pts:
                pt = Point(p["x"], p["y"])
                nearest = rtree.nearest(pt)
                geom = lines[nearest] if isinstance(nearest, (int, np.integer)) \
                    else nearest
                p["dist_road"] = float(pt.distance(geom))
            print(f"  OSM: {i+1}/{len(pts_by_cell)} cells "
                  f"({len(ways)} ways, {len(pts)} pts)", flush=True)
    else:
        for p in chosen:
            p["dist_road"] = float("nan")

    # embeddings, per point-year buckets
    from geotessera import GeoTessera
    gt = GeoTessera(cache_dir=args.cache_dir,
                    embeddings_dir=args.embeddings_dir)

    by_year = defaultdict(list)
    for p in chosen:
        by_year[p["year"]].append(p)

    if args.offline:
        # Offline sampling RAISES on a missing tile, which would poison the
        # whole chunk. Keep only points whose tile-year is already on disk;
        # the rest are picked up by a later run as the prefetch fills in.
        # A tile counts as present only when embedding, scales AND landmask
        # all exist — that is exactly tile_for_coord's contract. Checking
        # only the embedding let half-fetched tiles through, and one such
        # tile raises FileNotFoundError that kills the entire 150-pt chunk.
        from prefetch_tiles import (EMB_DIR_NAME, LM_DIR_NAME, grid_name,
                                    tile_coords)
        base = Path(args.embeddings_dir)
        kept, dropped = defaultdict(list), 0
        for yr, pts in by_year.items():
            for p in pts:
                g = grid_name(*tile_coords(p["lon"], p["lat"]))
                d = base / EMB_DIR_NAME / str(yr) / g
                if ((d / f"{g}.npy").exists()
                        and (d / f"{g}_scales.npy").exists()
                        and (base / LM_DIR_NAME / f"{g}.tiff").exists()):
                    kept[yr].append(p)
                else:
                    dropped += 1
        by_year = kept
        print(f"offline: {sum(len(v) for v in by_year.values())} points have "
              f"tiles on disk, {dropped} awaiting prefetch", flush=True)

    cols = defaultdict(list)
    n_nan = 0
    for year in sorted(by_year):
        pts = by_year[year]
        print(f"year {year}: sampling {len(pts)} points...", flush=True)
        for c0 in range(0, len(pts), CHUNK):
            chunk = pts[c0:c0 + CHUNK]
            coords = [(p["lon"], p["lat"]) for p in chunk]
            emb = None
            for attempt in range(3):
                try:
                    signal.alarm(600)  # hard ceiling per chunk attempt
                    emb = np.asarray(
                        gt.sample_embeddings_at_points(
                            coords, year=year,
                            auto_download=not args.offline),
                        dtype=np.float32)
                    break
                except Exception as e:
                    print(f"  chunk {c0//CHUNK} attempt {attempt+1}: "
                          f"{type(e).__name__}: {str(e)[:80]}; backoff",
                          flush=True)
                    time.sleep(15 * (attempt + 1))
                finally:
                    signal.alarm(0)
            if emb is None:
                print(f"  chunk {c0//CHUNK} y{year}: GAVE UP "
                      f"({len(chunk)} pts lost)", flush=True)
                continue
            # W1: silent misalignment guard
            assert len(emb) == len(chunk), \
                f"embedding count mismatch {len(emb)} != {len(chunk)}"
            for p, e in zip(chunk, emb):
                if np.isnan(e).any():
                    n_nan += 1
                    continue
                cols["X"].append(e)
                cols["y"].append(p["label"])
                cols["lat"].append(p["lat"]); cols["lon"].append(p["lon"])
                cols["year"].append(year); cols["month"].append(p["month"])
                cols["block"].append(p["block"])
                cols["dist_lupin"].append(p["dist_lupin"])
                cols["dist_road"].append(p["dist_road"])
            print(f"  y{year} +chunk{c0//CHUNK}: kept {len(cols['y'])} "
                  f"(NaN: {n_nan})", flush=True)
        np.savez_compressed(  # incremental save per year (CLAUDE.md p.12)
            DATA / "features_partial.npz",
            **{k: np.asarray(v) for k, v in cols.items()})

    X = np.asarray(cols.pop("X"), dtype=np.float32)
    arrs = {k: np.asarray(v) for k, v in cols.items()}
    yv = arrs["y"].astype(np.int8)
    print(f"total: X={X.shape}, pos={int(yv.sum())}, "
          f"neg={int((yv == 0).sum())}, nan_dropped={n_nan}")
    np.savez_compressed(DATA / "features.npz", X=X, **arrs)
    print(f"saved {DATA/'features.npz'}")


if __name__ == "__main__":
    main()
