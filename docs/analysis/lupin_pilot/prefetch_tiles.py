#!/usr/bin/env python3
"""Parallel tile prefetch for the pilot points.

geotessera's own downloader is serial and issues HTTP without timeouts, so
a single stalled socket blocks the whole extraction indefinitely (observed:
55 min, 0.02 s CPU). This fetches the same files with our own timeouts,
retries and 8-way parallelism, writing them into the exact layout
``tile_for_coord`` expects so sampling can then run with
``auto_download=False`` and never touch the network.

Per tile-year three files are needed:
    <emb>/global_0.1_degree_representation/<year>/grid_x_y/grid_x_y.npy
    <emb>/global_0.1_degree_representation/<year>/grid_x_y/grid_x_y_scales.npy
    <emb>/global_0.1_degree_tiff_all/grid_x_y.tiff          (year-independent)

Idempotent and resumable: existing non-empty files are never refetched.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests as rq

HERE = Path(__file__).parent
DATA = HERE / "data"
EMB_DIR_NAME = "global_0.1_degree_representation"
LM_DIR_NAME = "global_0.1_degree_tiff_all"
NPY_URL = "https://data.source.coop/tessera/tessera/npy/v1"
LM_URL = "https://data.source.coop/tessera/tessera/landmasks/v1"
HEADERS = {"User-Agent": "imint-lupin-pilot/0.1 (tobias.edman@ri.se)"}


def tile_coords(lon: float, lat: float) -> tuple[float, float]:
    """0.1-degree tile centre on the 0.05 grid, as used in file names."""
    return (round(np.floor(lon * 10) / 10 + 0.05, 2),
            round(np.floor(lat * 10) / 10 + 0.05, 2))


def grid_name(tlon: float, tlat: float) -> str:
    return f"grid_{tlon:.2f}_{tlat:.2f}"


def get(url: str, dest: Path, timeout: int = 120) -> tuple[bool, str]:
    """Download to a temp file then rename; returns (ok, note)."""
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
            tmp.rename(dest)
            return True, "downloaded"
        except Exception as e:
            tmp.unlink(missing_ok=True)
            if attempt == 2:
                return False, f"{type(e).__name__}: {str(e)[:60]}"
    return False, "exhausted"


def fetch_tile_year(base: Path, tlon: float, tlat: float,
                    year: int) -> tuple[str, bool, str]:
    g = grid_name(tlon, tlat)
    tag = f"{g}@{year}"
    emb_root = base / EMB_DIR_NAME / str(year) / g
    ok1, n1 = get(f"{NPY_URL}/{year}/{g}/{g}.npy", emb_root / f"{g}.npy")
    if not ok1:
        return tag, False, f"emb {n1}"
    ok2, n2 = get(f"{NPY_URL}/{year}/{g}/{g}_scales.npy",
                  emb_root / f"{g}_scales.npy")
    if not ok2:
        return tag, False, f"scales {n2}"
    ok3, n3 = get(f"{LM_URL}/{g}.tiff", base / LM_DIR_NAME / f"{g}.tiff")
    if not ok3:
        return tag, False, f"landmask {n3}"
    return tag, True, n1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings-dir", default=str(HERE))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    base = Path(args.embeddings_dir)
    pts = json.loads((DATA / "chosen_points.json").read_text())
    years = json.loads((DATA / "point_years.json").read_text())
    assert len(years) == len(pts), "point_years.json out of sync with points"

    # Order by how many points each tile-year unlocks, densest first: the
    # interim probe needs ~1000 resolved points, and this reaches that
    # threshold after a fraction of the tile-years rather than all of them.
    from collections import Counter
    counts = Counter((*tile_coords(p["lon"], p["lat"]), y)
                     for p, y in zip(pts, years))
    needed = [t for t, _ in counts.most_common()]
    cum = np.cumsum([counts[t] for t in needed])
    n_for_1k = int(np.searchsorted(cum, 1000) + 1)
    print(f"{len(pts)} points -> {len(needed)} tile-years needed; "
          f"densest {n_for_1k} cover 1000 points", flush=True)

    done = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_tile_year, base, tlon, tlat, yr): (tlon, tlat, yr)
                for tlon, tlat, yr in needed}
        for i, fut in enumerate(as_completed(futs), 1):
            tag, ok, note = fut.result()
            done += ok
            failed += not ok
            if not ok or i % 20 == 0:
                print(f"[{i}/{len(needed)}] {tag}: "
                      f"{'OK' if ok else 'FAIL'} {note} "
                      f"(ok={done} fail={failed})", flush=True)
    print(f"prefetch done: {done} ok, {failed} failed of {len(needed)}")


if __name__ == "__main__":
    main()
