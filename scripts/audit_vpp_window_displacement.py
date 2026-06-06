#!/usr/bin/env python3
"""Audit tiles whose growing-season frames are displaced by the VPP YYDDD fix.

After the HR-VPP date decode was corrected (commit: YYDDD instead of CNES),
per-tile growing-season windows change. A present growing frame (slots 1-3)
that no longer falls inside its corrected slot window will be DROPPED and
re-fetched by ``repair_to_canonical_layout``. This script flags exactly
those tiles so a scoped re-fetch can target them.

It mirrors repair_to_canonical_layout's classification: corrected windows
from ``compute_growing_season_windows`` (now YYDDD-decoded), capped at
``cap_doy`` (Sep 1 = DOY 244), year-matched per slot.

Output: an audit JSON with ``unique_affected_tiles`` (consumed directly by
scripts/refetch_affected_tiles.py).

Usage:
    python scripts/audit_vpp_window_displacement.py \\
        --data-dir /cephfs/unified_v2_512 \\
        --out /cephfs/audits/vpp_window_displacement_512.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from imint.training.vpp_windows import compute_growing_season_windows

CAP_DOY = 244  # Sep 1 — matches repair_to_canonical_layout default


def tile_year_of(d) -> int | None:
    for key in ("tessera_year", "lpis_year", "year"):
        if key in d:
            try:
                return int(d[key])
            except Exception:
                pass
    for v in d.get("dates", []):
        s = str(v)
        if len(s) >= 4:
            try:
                return int(s[:4])
            except ValueError:
                pass
    return None


def tile_is_displaced(d) -> bool:
    """True if >=1 present growing frame falls outside its corrected window."""
    if "vpp_sosd" not in d or "vpp_eosd" not in d:
        return False
    if not all(k in d for k in ("doy", "dates", "temporal_mask")):
        return False
    ty = tile_year_of(d)
    if ty is None:
        return False

    windows = compute_growing_season_windows(
        np.asarray(d["vpp_sosd"], np.float32),
        np.asarray(d["vpp_eosd"], np.float32),
        num_frames=3,
    )
    capped = [(s, min(e, CAP_DOY)) for s, e in windows]
    capped = [(s, e) for s, e in capped if s <= e]
    if len(capped) < 3:
        return False  # repair would return "error" — re-fetch can't help

    doy = [int(x) for x in d["doy"]]
    dates = [str(x) for x in d["dates"]]
    tmask = [int(x) for x in d["temporal_mask"]]

    # Growing frames live at canonical slots 1-3; slot 0 is autumn (hardcoded,
    # unaffected by the VPP fix).
    for src in (1, 2, 3):
        if src >= len(tmask) or tmask[src] == 0:
            continue
        frame_doy = doy[src]
        try:
            frame_year = int(dates[src][:4])
        except (ValueError, IndexError):
            return True  # unparseable date -> repair drops it
        if frame_year == ty and frame_doy > CAP_DOY:
            return True  # dropped: past the cap
        fits = any(
            frame_year == ty and lo <= frame_doy <= hi for lo, hi in capped
        )
        if not fits:
            return True
    return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None, help="For smoke tests.")
    args = p.parse_args()

    # Match UnifiedDataset's discovery: ALL *.npz (current "tile_<e>_<n>"
    # naming AND legacy numeric IDs like "44143910.npz"), minus atomic-write
    # leftovers.
    files = sorted(
        f for f in glob.glob(os.path.join(args.data_dir, "*.npz"))
        if not f.endswith("_tmp.npz")
    )
    if args.limit:
        files = files[: args.limit]
    print(f"scanning {len(files)} tiles in {args.data_dir}", flush=True)

    affected: list[str] = []
    errors = 0
    for i, f in enumerate(files):
        try:
            with np.load(f, allow_pickle=True) as d:
                if tile_is_displaced(d):
                    affected.append(os.path.basename(f)[:-4])
        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"  skip {os.path.basename(f)}: {e}", flush=True)
        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{len(files)}] affected so far: {len(affected)}",
                  flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    payload = {
        "audit": os.path.basename(args.out),
        "criterion": "growing-frame outside corrected YYDDD VPP window",
        "data_dir": args.data_dir,
        "n_scanned": len(files),
        "n_read_errors": errors,
        "unique_affected_tiles": sorted(affected),
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=0)
    print(f"\n=== done: {len(affected)}/{len(files)} affected "
          f"({len(affected)/max(len(files),1):.1%}); {errors} read errors ===",
          flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
