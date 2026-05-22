#!/usr/bin/env python3
"""Comprehensive frame-quality audit for unified_v2_512 / unified_v2_256 tiles.

Goes beyond the single late-autumn-DOY check (audit-late-autumn-frames)
and inventories every kind of frame problem that's accumulated across
the various fetch / refetch / enrich passes. Output JSON lists per-tile
issues and a per-tile ``needs`` list that the next refetch can act on.

Checks (per tile)
-----------------
- ``spectral_shape``         spectral cube shape is (24, H, W)
- ``tmask_consistency``      tmask[i]=1 iff spectral[i*6:(i+1)*6] non-zero
- ``date_parseable``         each non-empty dates[i] is ISO YYYY-MM-DD
- ``doy_matches_date``       doy[i] == day-of-year(dates[i])
- ``monotonic_calendar``     calendar dates are strictly ascending
- ``all_4_slots_filled``     sum(tmask) == 4
- ``late_autumn_bug``        any growing-season DOY > cap_doy (244)
- ``wrong_layout``           frame 0 is year-of-tile not year-1 (no autumn)
- ``year_consistency``       frame 0 year == tessera_year - 1
- ``slot_3_too_early``       slot 3 DOY < 200 (target is Jul–Aug, DOY 201–244)
- ``b08_date_drift``         b08_dates[i] != dates[i] (post-D3 backfill only)
- ``rededge_date_drift``     rededge_dates[i] != dates[i]

Output
------
``/checkpoints/audits/frame_audit_<dataset>_<date>.json``::

    {
      "dataset":     "512",
      "data_dir":    "/cephfs/unified_v2_512",
      "scanned_at":  "2026-05-22T...",
      "cap_doy":     244,
      "total":       7070,
      "unreadable":  5,
      "ok":          NNN,
      "tiles_with_issues": [
        {"name": "44363944", "issues": [...], "needs": [...],
         "dates": [...], "doy": [...], "tmask": [...]},
        ...
      ],
      "summary": {"late_autumn_bug": NNN, "wrong_layout": NNN, ...},
      "unique_problem_tiles": [list of names]
    }

Usage (k8s pod)
---------------
    python scripts/audit_frame_problems.py \\
        --data-dir /cephfs/unified_v2_512 \\
        --tile-size-px 512 \\
        --output-json /checkpoints/audits/frame_audit_512_2026-05-22.json \\
        --workers 8

Read-only. Safe to run alongside other jobs touching the same tiles.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


N_BANDS = 6
NUM_FRAMES = 4
DEFAULT_CAP_DOY = 244
SLOT_3_TARGET_MIN_DOY = 200  # late-summer should be at least mid-Jul


def _doy_from_date(date_str: str) -> int | None:
    try:
        return _dt.datetime.strptime(date_str, "%Y-%m-%d").timetuple().tm_yday
    except Exception:
        return None


def _year_from_date(date_str: str) -> int | None:
    try:
        return int(date_str[:4])
    except Exception:
        return None


def audit_one_tile(tile_path: str, *, tile_size_px: int = 512,
                   cap_doy: int = DEFAULT_CAP_DOY) -> dict:
    name = Path(tile_path).stem
    try:
        d = np.load(tile_path, allow_pickle=True)
    except Exception as e:
        return {"name": name, "unreadable": True, "reason": str(e)[:200]}

    issues: list[str] = []
    needs: set[str] = set()

    # Required fields
    if "spectral" not in d.files:
        return {"name": name, "unreadable": True, "reason": "no_spectral"}

    spectral = d["spectral"]
    dates = [str(x)[:10] for x in d.get("dates", ["", "", "", ""])]
    doy = [int(x) for x in d.get("doy", [0, 0, 0, 0])]
    tmask = [int(x) for x in d.get("temporal_mask", [0, 0, 0, 0])]

    # 1. spectral_shape
    expected = (NUM_FRAMES * N_BANDS, tile_size_px, tile_size_px)
    if spectral.shape != expected:
        issues.append(f"spectral_shape:{spectral.shape}_vs_{expected}")
        needs.add("rebuild_tile")

    # 2. tmask_consistency
    for i in range(NUM_FRAMES):
        slot_nonzero = bool(np.any(spectral[i * N_BANDS:(i + 1) * N_BANDS]))
        if bool(tmask[i]) != slot_nonzero:
            issues.append(f"tmask_consistency:slot_{i}_tmask={tmask[i]}_nonzero={slot_nonzero}")
            needs.add(f"refetch_slot_{i}")

    # 3. date_parseable + 4. doy_matches_date
    for i in range(NUM_FRAMES):
        if not dates[i]:
            continue
        parsed_doy = _doy_from_date(dates[i])
        if parsed_doy is None:
            issues.append(f"date_parseable:slot_{i}_bad={dates[i]!r}")
            needs.add(f"refetch_slot_{i}")
            continue
        if doy[i] != parsed_doy:
            issues.append(f"doy_matches_date:slot_{i}_doy={doy[i]}_vs_date={parsed_doy}")

    # 5. monotonic_calendar
    valid_dates = []
    for i in range(NUM_FRAMES):
        if dates[i]:
            try:
                valid_dates.append((i, _dt.datetime.strptime(dates[i], "%Y-%m-%d")))
            except Exception:
                pass
    for j in range(len(valid_dates) - 1):
        i1, d1 = valid_dates[j]
        i2, d2 = valid_dates[j + 1]
        if d2 <= d1:
            issues.append(f"monotonic_calendar:slot_{i1}={d1.date()}_>=_slot_{i2}={d2.date()}")
            needs.add(f"refetch_slot_{i2}")

    # 6. all_4_slots_filled
    if sum(tmask) < NUM_FRAMES:
        missing = [i for i in range(NUM_FRAMES) if not tmask[i]]
        issues.append(f"all_4_slots_filled:missing={missing}")
        for i in missing:
            needs.add(f"refetch_slot_{i}")

    # Determine tile_year (tessera_year preferred, lpis_year, year, dates)
    tile_year = None
    for k in ("tessera_year", "lpis_year", "year"):
        if k in d.files:
            try:
                tile_year = int(d[k])
                break
            except Exception:
                pass
    if tile_year is None:
        for s in dates:
            if s:
                y = _year_from_date(s)
                if y is not None:
                    # frame 0 is year-1 by convention if it's autumn
                    parsed_doy = _doy_from_date(s)
                    if parsed_doy and parsed_doy >= 228:
                        tile_year = y + 1
                    else:
                        tile_year = y
                    break

    # 7. late_autumn_bug — any growing-season DOY > cap_doy in tile_year
    for i in range(1, NUM_FRAMES):
        if tmask[i] and dates[i]:
            yr = _year_from_date(dates[i])
            if yr == tile_year and doy[i] > cap_doy:
                issues.append(f"late_autumn_bug:slot_{i}_doy={doy[i]}")
                needs.add(f"refetch_slot_{i}")

    # 8. wrong_layout — frame 0 should be year-1 (autumn) if tile_year known
    if tile_year is not None and tmask[0] and dates[0]:
        yr0 = _year_from_date(dates[0])
        if yr0 == tile_year:
            # frame 0 is year-of-tile, not year-1 — old 3-frame layout
            issues.append(f"wrong_layout:frame_0_year={yr0}_eq_tile_year={tile_year}")
            needs.add("refetch_slot_0")

    # 9. year_consistency — explicit check
    if tile_year is not None and tmask[0] and dates[0]:
        yr0 = _year_from_date(dates[0])
        if yr0 != tile_year - 1:
            issues.append(f"year_consistency:frame_0_year={yr0}_expected={tile_year - 1}")

    # 10. slot_3_too_early — target Jul-Aug (DOY 200-244)
    if tmask[3] and doy[3] > 0 and doy[3] < SLOT_3_TARGET_MIN_DOY:
        issues.append(f"slot_3_too_early:doy={doy[3]}")
        needs.add("refetch_slot_3")

    # 11. b08_date_drift (only when b08_dates marker exists post-D3 backfill)
    if "b08_dates" in d.files and "b08" in d.files:
        b08_dates = [str(x)[:10] for x in d["b08_dates"]]
        for i in range(min(NUM_FRAMES, len(b08_dates))):
            if dates[i] and b08_dates[i] and b08_dates[i] != dates[i]:
                issues.append(f"b08_date_drift:slot_{i}_b08={b08_dates[i]}_vs_spectral={dates[i]}")
                needs.add(f"enrich_b08_slot_{i}")

    # 12. rededge_date_drift
    if "rededge_dates" in d.files and "rededge" in d.files:
        re_dates = [str(x)[:10] for x in d["rededge_dates"]]
        for i in range(min(NUM_FRAMES, len(re_dates))):
            if dates[i] and re_dates[i] and re_dates[i] != dates[i]:
                issues.append(f"rededge_date_drift:slot_{i}_re={re_dates[i]}_vs_spectral={dates[i]}")
                needs.add(f"enrich_rededge_slot_{i}")

    return {
        "name": name,
        "unreadable": False,
        "issues": issues,
        "needs": sorted(needs),
        "dates": dates,
        "doy": doy,
        "tmask": tmask,
        "tile_year": tile_year,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--tile-size-px", type=int, default=512)
    p.add_argument("--output-json", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cap-doy", type=int, default=DEFAULT_CAP_DOY)
    p.add_argument("--max-tiles", type=int, default=None)
    args = p.parse_args()

    tiles = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if args.max_tiles:
        tiles = tiles[: args.max_tiles]

    print(f"=== frame audit ===", flush=True)
    print(f"  data-dir:     {args.data_dir}", flush=True)
    print(f"  tile-size:    {args.tile_size_px}", flush=True)
    print(f"  cap-doy:      {args.cap_doy}", flush=True)
    print(f"  output:       {args.output_json}", flush=True)
    print(f"  workers:      {args.workers}", flush=True)
    print(f"  tiles:        {len(tiles)}", flush=True)

    results: list[dict] = []
    lock = threading.Lock()
    completed = 0
    t0 = time.time()

    def _run(path):
        nonlocal completed
        r = audit_one_tile(path, tile_size_px=args.tile_size_px, cap_doy=args.cap_doy)
        with lock:
            results.append(r)
            completed += 1
            if completed % 500 == 0 or completed == len(tiles):
                elapsed = time.time() - t0
                rate = completed / max(elapsed / 3600, 1e-6)
                print(f"  [{completed}/{len(tiles)}] rate={rate:.0f}/h", flush=True)
        return r

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(as_completed([ex.submit(_run, t) for t in tiles]))

    elapsed = time.time() - t0
    print(f"\n=== done in {elapsed/60:.1f} min ===", flush=True)

    # Tally
    unreadable = [r for r in results if r.get("unreadable")]
    readable = [r for r in results if not r.get("unreadable")]
    with_issues = [r for r in readable if r["issues"]]
    no_issues = [r for r in readable if not r["issues"]]

    # Issue-type summary
    summary: dict[str, int] = {}
    for r in with_issues:
        seen_keys = set()
        for issue in r["issues"]:
            key = issue.split(":")[0]
            if key not in seen_keys:
                seen_keys.add(key)
                summary[key] = summary.get(key, 0) + 1

    out = {
        "dataset":      os.path.basename(args.data_dir).replace("unified_v2_", ""),
        "data_dir":     args.data_dir,
        "scanned_at":   _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "cap_doy":      args.cap_doy,
        "total":        len(results),
        "unreadable":   len(unreadable),
        "ok":           len(no_issues),
        "with_issues":  len(with_issues),
        "summary":      summary,
        "tiles_with_issues": with_issues,
        "unique_problem_tiles": [r["name"] for r in with_issues],
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n=== summary ===", flush=True)
    print(f"  total:       {out['total']}", flush=True)
    print(f"  unreadable:  {out['unreadable']}", flush=True)
    print(f"  ok:          {out['ok']}", flush=True)
    print(f"  with_issues: {out['with_issues']}", flush=True)
    print(f"  by issue type:", flush=True)
    for k, v in sorted(summary.items(), key=lambda kv: -kv[1]):
        print(f"    {k}: {v}", flush=True)
    print(f"\n  written: {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
