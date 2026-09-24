#!/usr/bin/env python3
"""Fetch the full candidate pool from GBIF (Artportalen mirror).

Rebuilt for the verge-based selection. The first pilot selected points by
0.1-degree block density, which left only ~2 % of them near a road with a
real verge (measured: 54 lupine points within 20 m of a European route).
This fetches the whole usable pool so the selection can be made on road
class instead.

Negatives are fetched per year with an explicit year filter. GBIF's
default ordering is not random across years, and a single random-offset
draw returned negatives from 2024 only — which made year predict the
label almost perfectly and invalidated the first probe entirely.

Outputs, in ./data:
    positives.csv   L. polyphyllus, SE, <=25 m, 2018-2024, Jun-Aug
    negatives.csv   other vascular plants, same filters, 7 years balanced
    exclusion.csv   genus Lupinus, <=100 m, all months (negative veto)
"""
from __future__ import annotations

import csv
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

API = "https://api.gbif.org/v1/occurrence/search"
OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(parents=True, exist_ok=True)

LUPIN_KEY = 2964355
LUPINUS_GENUS_KEY = 2963774
TRACHEOPHYTA_KEY = 7707728
PAGE = 300
OFFSET_CAP = 100_000          # GBIF search API hard limit
YEARS = (2018, 2019, 2020, 2021, 2022, 2023, 2024)
FIELDS = ["lat", "lon", "year", "month", "uncert_m", "gbifID", "speciesKey"]

BASE = {"country": "SE", "hasCoordinate": "true",
        "hasGeospatialIssue": "false", "occurrenceStatus": "PRESENT"}

# One Session per thread. requests.Session is NOT thread-safe: sharing one
# across the pool corrupts its connection pool, which shows up as
# RemoteDisconnected storms and then a hang — measured here as 43 min wall
# for 7 s of CPU while GBIF itself answered in 0,7 s.
_local = threading.local()


def _session() -> requests.Session:
    s = getattr(_local, "s", None)
    if s is None:
        s = _local.s = requests.Session()
    return s


def _get(params: dict) -> dict:
    for attempt in range(5):
        try:
            r = _session().get(API, params=params, timeout=(10, 45))
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == 4:
                raise
            print(f"  retry {attempt+1}: {str(e)[:60]}", file=sys.stderr)
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def _rows(results: list[dict]) -> list[list]:
    out = []
    for rec in results:
        lat, lon, yr = (rec.get("decimalLatitude"),
                        rec.get("decimalLongitude"), rec.get("year"))
        if lat is None or lon is None or yr is None:
            continue
        out.append([lat, lon, yr, rec.get("month", ""),
                    rec.get("coordinateUncertaintyInMeters", ""),
                    rec.get("key", ""), rec.get("speciesKey", "")])
    return out


def fetch(params: dict, writer, cap_pages: int | None = None,
          seed: int | None = None, label: str = "") -> int:
    """Page through a query with 4 workers; optionally sample pages."""
    total = min(_get({**params, "limit": 0})["count"], OFFSET_CAP)
    offsets = list(range(0, max(total, 1), PAGE))
    if cap_pages and len(offsets) > cap_pages:
        offsets = sorted(random.Random(seed).sample(offsets, cap_pages))
    n = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        for d in ex.map(lambda o: _get({**params, "limit": PAGE,
                                        "offset": o}), offsets):
            rows = _rows(d.get("results", []))
            writer.writerows(rows)
            n += len(rows)
    print(f"{label}: {n} rader (av {total} träffar)", flush=True)
    return n


def write_csv(path: Path, fn) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        fn(w)


if __name__ == "__main__":
    print("== positiva: L. polyphyllus, <=25 m, juni-aug ==", flush=True)
    write_csv(OUT / "positives.csv", lambda w: fetch(
        {**BASE, "taxonKey": LUPIN_KEY, "year": "2018,2024",
         "month": "6,8", "coordinateUncertaintyInMeters": "0,25"},
        w, label="positives"))

    print("== vetoset: hela släktet Lupinus, <=100 m ==", flush=True)
    write_csv(OUT / "exclusion.csv", lambda w: fetch(
        {**BASE, "taxonKey": LUPINUS_GENUS_KEY, "year": "2018,2024",
         "coordinateUncertaintyInMeters": "0,100"},
        w, label="exclusion"))

    print("== negativa: kärlväxter, per år ==", flush=True)
    def _neg(w):
        tot = 0
        for yr in YEARS:
            tot += fetch({**BASE, "taxonKey": TRACHEOPHYTA_KEY,
                          "year": str(yr), "month": "6,8",
                          "coordinateUncertaintyInMeters": "0,25"},
                         w, cap_pages=120, seed=1000 + yr, label=f"  {yr}")
        print(f"negatives totalt: {tot}", flush=True)
    write_csv(OUT / "negatives.csv", _neg)
    print("klart")
