#!/usr/bin/env python3
"""Fetch pilot label points from GBIF (Artportalen mirror).

Outputs (CSV: lat, lon, year, month, uncert_m, gbifID):
    positives.csv   — Lupinus polyphyllus, SE, <=25 m, 2018-2024, Jun-Aug
    exclusion.csv   — L. polyphyllus, SE, <=100 m, 2018-2024, all months
                      (used only to veto pseudo-negatives near known lupine)
    negatives_raw.csv — target-group background: other Tracheophyta obs,
                      same filters as positives (shares observer/roadside
                      bias), Lupinus genus excluded.
"""
from __future__ import annotations

import csv
import random
import sys
import time
from pathlib import Path

import requests

API = "https://api.gbif.org/v1/occurrence/search"
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

LUPIN_KEY = 2964355
LUPINUS_GENUS_KEY = 2963774
TRACHEOPHYTA_KEY = 7707728
PAGE = 300
OFFSET_CAP = 100_000  # GBIF search API hard cap

session = requests.Session()


def _get(params: dict) -> dict:
    for attempt in range(5):
        try:
            r = session.get(API, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            wait = 2 ** attempt
            print(f"  retry {attempt+1} in {wait}s: {e}", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"GBIF gave up on {params}")


def _rows(results: list[dict]) -> list[list]:
    rows = []
    for rec in results:
        lat, lon = rec.get("decimalLatitude"), rec.get("decimalLongitude")
        year = rec.get("year")
        if lat is None or lon is None or year is None:
            continue
        rows.append([
            lat, lon, year, rec.get("month", ""),
            rec.get("coordinateUncertaintyInMeters", ""),
            rec.get("key", ""), rec.get("speciesKey", ""),
        ])
    return rows


def fetch_all(params: dict, out_path: Path, cap: int | None = None) -> int:
    """Fetch every page concurrently (4 workers), write in offset order."""
    from concurrent.futures import ThreadPoolExecutor

    if out_path.exists() and out_path.stat().st_size > 0:
        with out_path.open() as f:
            n_existing = sum(1 for _ in f) - 1
        # Only reuse a file whose fetch ran to completion (count matches).
        total_now = _get({**params, "limit": 0})["count"]
        if n_existing >= min(total_now, cap or total_now) * 0.98:
            print(f"{out_path.name}: reusing {n_existing} existing rows")
            return n_existing

    total = min(_get({**params, "limit": 0})["count"], OFFSET_CAP)
    if cap:
        total = min(total, cap)
    offsets = list(range(0, total, PAGE))
    n = 0
    with ThreadPoolExecutor(max_workers=4) as ex, \
            out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lat", "lon", "year", "month", "uncert_m", "gbifID", "speciesKey"])
        pages = ex.map(lambda off: _get({**params, "limit": PAGE, "offset": off}),
                       offsets)
        for i, d in enumerate(pages):
            rows = _rows(d.get("results", []))
            w.writerows(rows)
            n += len(rows)
            if (i + 1) % 20 == 0:
                print(f"  {out_path.name}: {n} rows ({i+1}/{len(offsets)} pages)",
                      flush=True)
    print(f"{out_path.name}: {n} rows total")
    return n


def fetch_random_offsets(params: dict, out_path: Path, n_pages: int, seed: int) -> int:
    """Sample pages at random offsets (for huge result sets)."""
    total = min(_get({**params, "limit": 0})["count"], OFFSET_CAP)
    rng = random.Random(seed)
    offsets = rng.sample(range(0, max(total - PAGE, 1), PAGE),
                         min(n_pages, max(total // PAGE, 1)))
    from concurrent.futures import ThreadPoolExecutor

    n = 0
    with ThreadPoolExecutor(max_workers=4) as ex, \
            out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lat", "lon", "year", "month", "uncert_m", "gbifID", "speciesKey"])
        pages = ex.map(lambda off: _get({**params, "limit": PAGE, "offset": off}),
                       sorted(offsets))
        for i, d in enumerate(pages):
            rows = _rows(d.get("results", []))
            w.writerows(rows)
            n += len(rows)
            if (i + 1) % 10 == 0:
                print(f"  {out_path.name}: page {i+1}/{len(offsets)}, {n} rows",
                      flush=True)
    print(f"{out_path.name}: {n} rows total (sampled from {total})")
    return n


BASE = {
    "country": "SE", "hasCoordinate": "true",
    "hasGeospatialIssue": "false", "year": "2018,2024",
    "occurrenceStatus": "PRESENT",
}

# Guarded: without this, `from fetch_gbif import ...` re-runs every fetch
# below. That is what silently rebuilt negatives_raw.csv with the
# single-shot random-offset sampler (2024-only) after the year-stratified
# fetch had replaced it, and what re-ran the negatives during the genus
# fetch. Importers must be able to reuse the helpers without side effects.
if __name__ == "__main__":
    print("== positives (lupin, <=25m, Jun-Aug) ==")
    fetch_all({**BASE, "taxonKey": LUPIN_KEY, "month": "6,8",
               "coordinateUncertaintyInMeters": "0,25"},
              OUT / "positives.csv")

    print("== exclusion set (lupin, <=100m, all months) ==")
    fetch_all({**BASE, "taxonKey": LUPIN_KEY,
               "coordinateUncertaintyInMeters": "0,100"},
              OUT / "exclusion.csv")

    print("== negatives raw (target-group background) ==")
    print("NOTE: superseded by fetch_neg_by_year.py — the single-shot "
          "random-offset draw returns one year only (GBIF ordering is not "
          "random across years) and confounds year with class label.")
    fetch_random_offsets({**BASE, "taxonKey": TRACHEOPHYTA_KEY, "month": "6,8",
                          "coordinateUncertaintyInMeters": "0,25"},
                         OUT / "negatives_raw.csv", n_pages=60, seed=42)
    print("done")
