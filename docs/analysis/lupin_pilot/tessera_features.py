#!/usr/bin/env python3
"""TESSERA embeddings for the verge selection: prefetch, then sample.

Two stages, deliberately separated. geotessera's own downloader is serial
and issues HTTP without timeouts, so one stalled socket blocks everything
(measured once at 55 min wall for 0,02 s of CPU). Stage 1 fetches the same
files with our own timeouts and 8-way parallelism, writing them into the
exact layout ``tile_for_coord`` expects. Stage 2 then samples with
``auto_download=False``, so it touches no network and cannot stall.

A tile counts as present only when embedding, scales AND landmask all
exist — that is tile_for_coord's contract. Checking only the embedding
let half-fetched tiles through, and a single missing tile raises inside
``sample_embeddings_at_points``, killing the whole chunk rather than the
one point.

Output: data/features_tessera_verge.npz in the probe's schema
(X, y, lat, lon, year, month, block, dist_lupin, dist_road).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests as rq

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EMB_ROOT = HERE / "tessera_tiles"
EMB_DIR = "global_0.1_degree_representation"
LM_DIR = "global_0.1_degree_tiff_all"
NPY_URL = "https://data.source.coop/tessera/tessera/npy/v1"
LM_URL = "https://data.source.coop/tessera/tessera/landmasks/v1"
HEADERS = {"User-Agent": "imint-lupin-study/0.1 (tobias.edman@ri.se)"}
CHUNK = 150


def tile_coords(lon: float, lat: float) -> tuple[float, float]:
    """0.1-degree tile centre on the 0.05 grid, as used in file names."""
    return (round(np.floor(lon * 10) / 10 + 0.05, 2),
            round(np.floor(lat * 10) / 10 + 0.05, 2))


def grid_name(tlon: float, tlat: float) -> str:
    return f"grid_{tlon:.2f}_{tlat:.2f}"


def get(url: str, dest: Path, timeout: int = 120) -> tuple[bool, str]:
    if dest.exists() and dest.stat().st_size > 0:
        return True, "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + f".tmp{os.getpid()}")
    for attempt in range(3):
        try:
            with rq.get(url, headers=HEADERS, timeout=timeout,
                        stream=True) as r:
                if r.status_code == 404:
                    return False, "404"
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for block in r.iter_content(1 << 20):
                        f.write(block)
            tmp.rename(dest)          # atomic: never a half-written tile
            return True, "downloaded"
        except Exception as e:
            tmp.unlink(missing_ok=True)
            if attempt == 2:
                return False, f"{type(e).__name__}"
    return False, "exhausted"


def fetch_tile_year(tlon: float, tlat: float, year: int) -> tuple[str, bool]:
    g = grid_name(tlon, tlat)
    root = EMB_ROOT / EMB_DIR / str(year) / g
    for url, dest in ((f"{NPY_URL}/{year}/{g}/{g}.npy", root / f"{g}.npy"),
                      (f"{NPY_URL}/{year}/{g}/{g}_scales.npy",
                       root / f"{g}_scales.npy"),
                      (f"{LM_URL}/{g}.tiff",
                       EMB_ROOT / LM_DIR / f"{g}.tiff")):
        ok, _ = get(url, dest)
        if not ok:
            return f"{g}@{year}", False
    return f"{g}@{year}", True


def tile_present(lon: float, lat: float, year: int) -> bool:
    g = grid_name(*tile_coords(lon, lat))
    d = EMB_ROOT / EMB_DIR / str(year) / g
    return ((d / f"{g}.npy").exists() and (d / f"{g}_scales.npy").exists()
            and (EMB_ROOT / LM_DIR / f"{g}.tiff").exists())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default=str(DATA / "chosen_points_verge.json"))
    ap.add_argument("--out", default=str(DATA / "features_tessera_verge.npz"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-prefetch", action="store_true")
    args = ap.parse_args()

    pts = json.loads(Path(args.points).read_text())
    print(f"{len(pts)} punkter", flush=True)

    needed = sorted({(*tile_coords(p["lon"], p["lat"]), p["year"])
                     for p in pts})
    if not args.skip_prefetch:
        print(f"förhämtar {len(needed)} tile-år", flush=True)
        ok = bad = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(fetch_tile_year, a, b, c) for a, b, c in needed]
            for i, f in enumerate(as_completed(futs), 1):
                tag, good = f.result()
                ok += good
                bad += not good
                if not good or i % 25 == 0:
                    print(f"  [{i}/{len(needed)}] {tag} "
                          f"{'OK' if good else 'FAIL'} (ok={ok} fail={bad})",
                          flush=True)
        print(f"förhämtning klar: {ok} ok, {bad} misslyckade", flush=True)

    from geotessera import GeoTessera
    gt = GeoTessera(cache_dir=str(HERE / "tessera_cache"),
                    embeddings_dir=str(EMB_ROOT))

    by_year = defaultdict(list)
    dropped = 0
    for p in pts:
        if tile_present(p["lon"], p["lat"], p["year"]):
            by_year[p["year"]].append(p)
        else:
            dropped += 1
    kept = sum(len(v) for v in by_year.values())
    print(f"tiles på disk för {kept} punkter, {dropped} utan", flush=True)

    cols: dict[str, list] = defaultdict(list)
    n_nan = 0
    for year in sorted(by_year):
        group = by_year[year]
        print(f"år {year}: samplar {len(group)} punkter", flush=True)
        for c0 in range(0, len(group), CHUNK):
            chunk = group[c0:c0 + CHUNK]
            coords = [(p["lon"], p["lat"]) for p in chunk]
            try:
                emb = np.asarray(gt.sample_embeddings_at_points(
                    coords, year=year, auto_download=False), dtype=np.float32)
            except Exception as e:
                print(f"  chunk {c0//CHUNK} y{year}: {type(e).__name__} "
                      f"({len(chunk)} punkter förlorade)", flush=True)
                continue
            assert len(emb) == len(chunk), \
                f"embedding count {len(emb)} != {len(chunk)}"
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
                # the probe's habitat filter reads dist_road; here every
                # point is already verge-selected, so this is the verge
                # distance and the filter becomes a no-op rather than a
                # second, different definition of "roadside"
                cols["dist_road"].append(p["verge_dist"])

    X = np.asarray(cols.pop("X"), dtype=np.float32)
    arrs = {k: np.asarray(v) for k, v in cols.items()}
    yv = arrs["y"].astype(int)
    print(f"\ntotalt: X={X.shape}, lupin={int(yv.sum())}, "
          f"negativa={int((yv == 0).sum())}, NaN-bortfall={n_nan}")
    print("per år:", dict(sorted(Counter(arrs['year'].tolist()).items())))
    np.savez_compressed(args.out, X=X, **arrs)
    print(f"sparade {args.out}")


if __name__ == "__main__":
    main()
