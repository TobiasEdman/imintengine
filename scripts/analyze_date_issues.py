#!/usr/bin/env python3
"""
Analyze date inconsistencies across unified_v2 tiles and produce fix lists.

Expected frame pattern: [fall_y-1, spring_y, summer_y, late-summer_y]
  d[0] = Sep-Oct of year y-1  (autumn previous year)
  d[1] = Apr-May of year y    (spring)
  d[2] = Jun-Jul of year y    (summer)
  d[3] = Aug-Sep of year y    (late summer / post-harvest)

Produces two JSON files:
  --refetch-out   Tiles with scrambled dates — need full 4-frame re-fetch.
                  Format: fetch_unified_tiles.py --from-json compatible.
                  Fields: name (no .npz), bbox [W,S,E,N], year, source, …

  --patch-out     Tiles where all 4 dates are same year (d[0] ≠ y-1).
                  Only frame 0 (6 bands) needs to be replaced with Sep-Oct y-1.
                  Fields: name (with .npz), center_e/n, bbox, primary_year, …

LPIS vintages available: 2018, 2019, 2022, 2023, 2024

Usage:
    python scripts/analyze_date_issues.py \\
        --data-dir /data/unified_v2 \\
        --refetch-out /tmp/refetch_list.json \\
        --patch-out   /tmp/patch_list.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

TILE_RE        = re.compile(r"tile_(\d+)_(\d+)\.npz$")
LPIS_AVAILABLE = {2018, 2019, 2022, 2023, 2024}
HALF           = 1280   # 256px × 10m/px / 2


def best_lpis_year(y: int) -> int:
    """Return closest available LPIS vintage to target year y."""
    if y in LPIS_AVAILABLE:
        return y
    return min(LPIS_AVAILABLE, key=lambda a: abs(a - y))


def target_year_from_dates(dates: list[str]) -> int:
    """
    Infer intended primary growing-season year.
    Uses modal year among d[1,2,3]; tie-breaks to smallest (older) year.
    """
    years = []
    for d in dates:
        s = str(d).strip()
        years.append(int(s[:4]) if (len(s) >= 4 and s[:4].isdigit()) else 0)
    if len(years) != 4:
        return 0
    cnt = Counter(years[1:4])
    return sorted(cnt.items(), key=lambda x: (-x[1], x[0]))[0][0]


def classify(dates: list[str]) -> str:
    years = []
    for d in dates:
        s = str(d).strip()
        if len(s) >= 4 and s[:4].isdigit():
            years.append(int(s[:4]))
        else:
            return "bad"
    if len(years) != 4:
        return "bad_count"
    y0, y1, y2, y3 = years
    if y1 == y2 == y3:
        if y0 == y1 - 1:   return "ok"
        elif y0 == y1:      return "same_year"
        else:               return "inconsistent"
    return "inconsistent"


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze tile date inconsistencies")
    ap.add_argument("--data-dir",    default="/data/unified_v2")
    ap.add_argument("--refetch-out", default="/tmp/refetch_list.json",
                    help="Output JSON for full re-fetch (inconsistent tiles)")
    ap.add_argument("--patch-out",   default="/tmp/patch_list.json",
                    help="Output JSON for frame-0 patch (same-year tiles)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    tiles = sorted(f for f in data_dir.glob("tile_*.npz")
                   if TILE_RE.search(f.name))
    print(f"Scanning {len(tiles)} tiles in {data_dir}...")

    refetch: list[dict] = []
    patch:   list[dict] = []
    ok_count = 0

    for fp in tiles:
        m  = TILE_RE.search(fp.name)
        ce, cn = int(m.group(1)), int(m.group(2))
        W, S, E, N = ce - HALF, cn - HALF, ce + HALF, cn + HALF

        try:
            d = np.load(fp, allow_pickle=True)
        except Exception:
            continue

        if "dates" not in d.files:
            continue

        dates  = [str(x).strip() for x in d["dates"]]
        status = classify(dates)

        if status == "ok":
            ok_count += 1
            continue

        elif status == "same_year":
            yrs     = [int(s[:4]) for s in dates]
            primary = yrs[1]
            patch.append({
                # patch_autumn_frame.py fields:
                "name":            fp.name,
                "center_e":        ce,
                "center_n":        cn,
                "bbox":            [W, S, E, N],
                "primary_year":    primary,
                "autumn_year":     primary - 1,
                "fetch_from":      f"{primary-1}-09-01",
                "fetch_to":        f"{primary-1}-10-31",
                "current_d0_date": dates[0],
            })

        elif status == "inconsistent":
            target = target_year_from_dates(dates)
            lpis_y = best_lpis_year(target)
            yrs    = [int(s[:4]) for s in dates]
            refetch.append({
                # fetch_unified_tiles.py --from-json fields:
                "name":    fp.name.replace(".npz", ""),
                "bbox":    [W, S, E, N],
                "year":    target,
                "source":  "lulc",
                # Reference info:
                "lpis_year":   lpis_y,
                "lpis_file":   f"jordbruksskiften_{lpis_y}.parquet",
                "autumn_year": target - 1,
                "old_dates":   dates,
                "pattern":     f"{yrs[0]},{yrs[1]},{yrs[2]},{yrs[3]}",
            })

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n=== DATE CONSISTENCY SUMMARY ===")
    print(f"  OK (y-1,y,y,y):       {ok_count}")
    print(f"  Patch (frame-0 only): {len(patch)}")
    print(f"  Refetch (full):       {len(refetch)}")

    tc = Counter(r["year"] for r in refetch)
    print("\nRefetch by target year:")
    for y, n in sorted(tc.items()):
        lp = best_lpis_year(y)
        print(f"  {y}: {n} tiles → autumn {y-1} | LPIS {lp}")

    lc = Counter(r["lpis_year"] for r in refetch)
    print("\nLPIS assignment for refetch tiles:")
    for y, n in sorted(lc.items()):
        print(f"  jordbruksskiften_{y}.parquet: {n} tiles")

    pc = Counter(r["primary_year"] for r in patch)
    print("\nPatch by primary year:")
    for y, n in sorted(pc.items()):
        print(f"  {y}: {n} tiles → fetch autumn {y-1}")

    # ── Write outputs ─────────────────────────────────────────────────────────
    with open(args.refetch_out, "w") as f:
        json.dump(refetch, f, indent=2)
    print(f"\nWrote {args.refetch_out} ({len(refetch)} entries)")

    with open(args.patch_out, "w") as f:
        json.dump(patch, f, indent=2)
    print(f"Wrote {args.patch_out} ({len(patch)} entries)")


if __name__ == "__main__":
    main()
