#!/usr/bin/env python3
"""Candidate-funnel diagnostic for failed-tile slot windows.

For a sample of N tiles from the audit list, computes per-slot:
  - raw_stac:   all S2 acquisitions over the tile bbox in the window
                (no cloud filter)
  - stac_pass:  scenes with granule eo:cloud_cover <= 30 (production
                ranker default)
  - era5_pass:  also overpass ERA5 cloud cover <= 50 % (production
                ranker default)
  - tried:      what fetch_spectral would iterate (== era5_pass under
                the current production thresholds)
  - era5_p25/50/75: ERA5 percentiles of the tried candidates (gives a
                sense of how strict the adaptive SCL gate would be on
                them — at ERA5=0 % the gate is 0.05 growing / 0.10
                autumn; at 25 % it's 0.15 / 0.30).

Aggregates across the sample + per-slot type. Final block answers:
"are we trying everything that exists, or are we cutting off candidates
before fetch_spectral even sees them?"

Reads:
  /cephfs/audits/frame_audit_512_*.json  (audit list)
  /cephfs/unified_v2_512/<name>.npz      (per-tile bbox + year)

Hits public APIs (DES STAC + Open-Meteo ERA5) — does NOT touch CDSE
PUs or the production semaphores. Safe to run alongside the refetch
job.
"""
from __future__ import annotations

import glob
import json
import os
import random
import statistics
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace/imintengine")

from imint.config.env import load_env
load_env()

from imint.fetch import _stac_available_dates
from imint.training.optimal_fetch import rank_stac_era5_candidates
from imint.training.tile_fetch import bbox_3006_to_wgs84


# ── Slot windows (production fallback when VPP not used) ────────────────────


def slot_windows_for_year(year: int) -> dict[int, tuple[str, str]]:
    """Standard 4-slot fallback windows for a given tile year.

    Slot 0 = autumn of year-1 (Aug 15 - Oct 31).
    Slots 1-3 = growing season VPP frames; we use a fixed approximation
    here for diagnostic purposes (real production shifts by VPP per
    tile, but the count of available scenes inside each window is
    approximately the same).
    """
    prev = year - 1
    return {
        0: (f"{prev}-08-15", f"{prev}-10-31"),
        1: (f"{year}-04-01", f"{year}-05-31"),
        2: (f"{year}-06-01", f"{year}-07-31"),
        3: (f"{year}-08-01", f"{year}-09-15"),
    }


# ── Single-tile diagnostic ──────────────────────────────────────────────────


def analyse_tile(name: str, tiles_dir: str) -> dict | None:
    """Compute the candidate funnel for one tile (4 slots)."""
    path = os.path.join(tiles_dir, f"{name}.npz")
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path, allow_pickle=True)
    except Exception:
        return None
    if "bbox_3006" not in data.files:
        return None
    bbox = data["bbox_3006"]
    bbox_dict = {"west": float(bbox[0]), "south": float(bbox[1]),
                 "east": float(bbox[2]), "north": float(bbox[3])}
    coords_wgs84 = bbox_3006_to_wgs84(bbox_dict)

    # Year derivation (same logic as refetch_affected_tiles.loc_from_existing)
    year = None
    if "year" in data.files:
        year = int(data["year"])
    elif "lpis_year" in data.files:
        year = int(data["lpis_year"])
    elif "dates" in data.files:
        for d in data["dates"]:
            s = str(d)
            if s and len(s) >= 4:
                try:
                    year = int(s[:4])
                    break
                except ValueError:
                    continue
    if year is None:
        return None

    windows = slot_windows_for_year(year)
    out: dict = {"name": name, "year": year, "slots": {}}
    for sidx, (ds, de) in windows.items():
        try:
            raw = _stac_available_dates(coords_wgs84, ds, de)
        except Exception as e:
            out["slots"][sidx] = {"error": f"stac: {type(e).__name__}: {str(e)[:80]}"}
            continue
        raw_n = len(raw)

        # Production ranker output: scene_cloud_max=30, overpass<=50
        try:
            ranked = rank_stac_era5_candidates(
                coords_wgs84, ds, de,
                scene_cloud_max=30.0, overpass_cloud_max=50.0,
            )
        except Exception as e:
            out["slots"][sidx] = {
                "raw_stac": raw_n,
                "error": f"rank: {type(e).__name__}: {str(e)[:80]}",
            }
            continue
        era5_vals = sorted(oc for _d, _cc, oc in ranked)
        out["slots"][sidx] = {
            "raw_stac": raw_n,
            "ranked": len(ranked),     # == what fetch_spectral iterates
            "era5_min": (era5_vals[0] if era5_vals else None),
            "era5_p50": (statistics.median(era5_vals) if era5_vals else None),
            "era5_max": (era5_vals[-1] if era5_vals else None),
        }
    return out


# ── Aggregate over a sample ─────────────────────────────────────────────────


def summarise(results: list[dict]) -> None:
    SLOT_LABELS = {0: "autumn", 1: "growing-1", 2: "growing-2", 3: "growing-3"}
    print()
    print("=" * 78)
    print(f"FUNNEL SUMMARY — {len(results)} tiles, 4 slots each")
    print("=" * 78)
    for sidx in (0, 1, 2, 3):
        raw = [r["slots"][sidx]["raw_stac"]
               for r in results
               if sidx in r["slots"] and "raw_stac" in r["slots"][sidx]]
        ranked = [r["slots"][sidx]["ranked"]
                  for r in results
                  if sidx in r["slots"] and "ranked" in r["slots"][sidx]]
        zero_ranked = sum(1 for v in ranked if v == 0)
        era5_mins = [r["slots"][sidx]["era5_min"]
                     for r in results
                     if sidx in r["slots"] and r["slots"][sidx].get("era5_min") is not None]
        print()
        print(f"--- SLOT {sidx} ({SLOT_LABELS[sidx]}) ---")
        if raw:
            print(f"  raw S2 acquisitions:  mean={statistics.mean(raw):5.1f}  "
                  f"median={statistics.median(raw):.0f}  "
                  f"min={min(raw)}  max={max(raw)}")
        if ranked:
            print(f"  ranked candidates:    mean={statistics.mean(ranked):5.1f}  "
                  f"median={statistics.median(ranked):.0f}  "
                  f"min={min(ranked)}  max={max(ranked)}")
            print(f"  tiles w/ 0 candidates after filter: {zero_ranked} / {len(ranked)}")
        if era5_mins:
            print(f"  ERA5 best-of-slot (%): mean={statistics.mean(era5_mins):5.1f}  "
                  f"median={statistics.median(era5_mins):.1f}  "
                  f"min={min(era5_mins):.1f}  max={max(era5_mins):.1f}")

    # Per-tile worst-cases (slots with 0 candidates after filter)
    print()
    print("=" * 78)
    print("TILES WITH ZERO RANKED CANDIDATES IN ONE OR MORE SLOTS")
    print("=" * 78)
    for r in results:
        zero_slots = [sidx for sidx, s in r["slots"].items() if s.get("ranked") == 0]
        if zero_slots:
            raw_for_zero = [r["slots"][s]["raw_stac"] for s in zero_slots]
            print(f"  {r['name']:32s} year={r['year']}  zero-slot(s)={zero_slots}  "
                  f"raw_stac in those slots={raw_for_zero}")


def main():
    audit_dir = "/cephfs/audits"
    audit_glob = sorted(glob.glob(f"{audit_dir}/frame_audit_512_*.json"))
    if not audit_glob:
        print(f"FATAL: no audit JSON in {audit_dir}", flush=True)
        sys.exit(1)
    audit_path = audit_glob[-1]
    print(f"audit: {audit_path}", flush=True)
    audit = json.load(open(audit_path))
    names = (audit.get("unique_affected_tiles")
             or audit.get("unique_problem_tiles") or [])
    names = [n[:-4] if n.endswith(".npz") else n for n in names]
    print(f"total affected tiles in audit: {len(names)}", flush=True)

    # Random sample
    N = int(os.environ.get("SAMPLE_N", "25"))
    random.seed(42)
    sample = random.sample(names, min(N, len(names)))
    print(f"analysing sample of {len(sample)} tiles\n", flush=True)

    results = []
    t0 = time.time()
    for i, name in enumerate(sample, 1):
        r = analyse_tile(name, "/cephfs/unified_v2_512")
        if r is None:
            print(f"  [{i}/{len(sample)}] {name}: NO TILE / NO BBOX", flush=True)
            continue
        # Compact per-tile line
        counts = []
        for sidx in (0, 1, 2, 3):
            s = r["slots"].get(sidx, {})
            if "error" in s:
                counts.append(f"s{sidx}:err")
            else:
                counts.append(f"s{sidx}:raw={s.get('raw_stac', '?')}/rank={s.get('ranked', '?')}")
        print(f"  [{i}/{len(sample)}] {name:30s} y={r['year']}  " + "  ".join(counts), flush=True)
        results.append(r)
    print(f"\nelapsed: {time.time()-t0:.0f}s", flush=True)

    summarise(results)


if __name__ == "__main__":
    main()
