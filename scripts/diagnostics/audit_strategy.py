#!/usr/bin/env python3
"""What does each audit-flagged tile actually need?

Per-tile, per-slot validity check against the npz on /cephfs:
  - frame all-zero? (no spectral data)
  - date string empty? (slot was never fetched)
  - DOY > cap_doy (244)? (the pre-PR#15 late-autumn bug — the audit criterion)
  - tmask=0? (marked missing)

Aggregates: how many tiles need {0, 1, 2, 3, 4} slots refetched, and
which slots are most often broken. Estimates total per-slot fetch
volume vs the naive "4 fetches per tile" baseline.

Read-only on /cephfs. Does not call any API. Safe to run concurrent with
a fetch job.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

CAP_DOY = 244           # Sep 1 — matches manifest --cap-doy and PR #15
TILES_DIR = "/cephfs/unified_v2_512"

# Per-slot nominal DOY windows (production fallback when VPP not used);
# in practice production uses VPP-shifted windows per tile, but these are
# adequate for the validity check (DOY outside any plausible window =
# definitely broken).
SLOT_WINDOWS = {
    0: (228, 304),   # autumn — Aug 15 .. Oct 31
    1: (91, 152),    # growing-1 — Apr-May
    2: (152, 213),   # growing-2 — Jun-Jul
    3: (213, 258),   # growing-3 — Aug-mid-Sep
}


def doy_from_date(s: str) -> int | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s)[:10]).timetuple().tm_yday
    except Exception:
        return None


def analyse_tile(name: str) -> dict:
    """Return per-slot reasons (or [] if slot OK)."""
    path = os.path.join(TILES_DIR, f"{name}.npz")
    if not os.path.exists(path):
        return {"name": name, "exists": False, "slots": {}}
    try:
        d = np.load(path, allow_pickle=True)
    except Exception as e:
        return {"name": name, "exists": True, "load_error": str(e)[:80], "slots": {}}

    out = {"name": name, "exists": True, "slots": {}}
    spec = d["spectral"] if "spectral" in d.files else None
    dates = list(d["dates"]) if "dates" in d.files else []
    tmask = list(d["temporal_mask"]) if "temporal_mask" in d.files else [1] * 4
    doys = list(d["doy"]) if "doy" in d.files else []

    n_bands_per_slot = 6
    for s in range(4):
        reasons = []
        # tmask check
        if s < len(tmask) and int(tmask[s]) == 0:
            reasons.append("tmask=0")
        # all-zero spectral check
        if spec is not None:
            slice_ = spec[s * n_bands_per_slot:(s + 1) * n_bands_per_slot]
            if slice_.size and not np.any(slice_):
                reasons.append("all-zero")
        # date check
        date_str = str(dates[s]) if s < len(dates) else ""
        if not date_str or date_str == "":
            reasons.append("no-date")
        # DOY checks
        doy_val = None
        if s < len(doys):
            try:
                doy_val = int(doys[s])
            except Exception:
                pass
        if doy_val is None:
            doy_val = doy_from_date(date_str)
        if doy_val is not None:
            if s == 0 and doy_val > CAP_DOY:
                # The audit's exact criterion — pre-PR #15 late-autumn bug
                reasons.append(f"DOY {doy_val} > cap {CAP_DOY}")
            nmin, nmax = SLOT_WINDOWS[s]
            if not (nmin <= doy_val <= nmax):
                reasons.append(f"DOY {doy_val} out of nominal [{nmin},{nmax}]")
        out["slots"][s] = reasons
    return out


def main():
    audit_glob = sorted(glob.glob("/cephfs/audits/frame_audit_512_*.json"))
    if not audit_glob:
        print("FATAL: no audit JSON under /cephfs/audits/", flush=True)
        sys.exit(1)
    audit_path = audit_glob[-1]
    print(f"audit: {audit_path}", flush=True)
    audit = json.load(open(audit_path))
    names = (audit.get("unique_affected_tiles")
             or audit.get("unique_problem_tiles") or [])
    names = [n[:-4] if n.endswith(".npz") else n for n in names]
    print(f"audit lists: {len(names)} tiles", flush=True)

    results = []
    n_no_npz = 0
    for i, name in enumerate(names, 1):
        r = analyse_tile(name)
        if not r.get("exists"):
            n_no_npz += 1
            continue
        results.append(r)
        if i % 1000 == 0:
            print(f"  scanned {i}/{len(names)} ({i/len(names)*100:.0f}%)",
                  flush=True)
    print(f"\n{n_no_npz} tiles had no .npz on disk; analysing {len(results)} that do.\n",
          flush=True)

    # Aggregates
    broken_per_tile = Counter()
    broken_per_slot = Counter()
    reasons_per_slot = {s: Counter() for s in range(4)}
    fully_ok = 0
    for r in results:
        bad = [s for s, rs in r["slots"].items() if rs]
        broken_per_tile[len(bad)] += 1
        if not bad:
            fully_ok += 1
        for s in bad:
            broken_per_slot[s] += 1
        for s, rs in r["slots"].items():
            for reason in rs:
                # Bucket reasons (strip DOY values for grouping)
                key = reason.split(" out of")[0] if "out of" in reason \
                      else reason.split(">")[0].rstrip() if ">" in reason \
                      else reason
                reasons_per_slot[s][key] += 1

    n = len(results)
    print("=" * 70)
    print(f"BROKEN-SLOT DISTRIBUTION PER TILE  (n={n})")
    print("=" * 70)
    for k in range(5):
        c = broken_per_tile[k]
        bar = "█" * int(60 * c / n)
        label = "all 4 slots OK (script skips)" if k == 0 else f"{k} slot(s) broken"
        print(f"  {k}: {c:5d} ({100*c/n:>5.1f}%)  {label}  {bar}")

    print()
    print("=" * 70)
    print("BROKEN-SLOT FREQUENCY BY SLOT INDEX")
    print("=" * 70)
    slot_labels = {0: "autumn (y-1)", 1: "growing-1 spring",
                   2: "growing-2 summer", 3: "growing-3 late-summer"}
    for s in range(4):
        c = broken_per_slot[s]
        bar = "█" * int(60 * c / n)
        print(f"  slot {s} ({slot_labels[s]:22s}): {c:5d} "
              f"({100*c/n:>5.1f}%)  {bar}")
        for reason, count in reasons_per_slot[s].most_common(3):
            print(f"      └─ {reason}: {count}")

    total_slots_to_fetch = sum(broken_per_slot.values())
    naive_baseline = (n - fully_ok) * 4  # if we refetched all 4 slots per non-OK tile
    print()
    print("=" * 70)
    print("PU/FETCH ECONOMICS")
    print("=" * 70)
    print(f"  Tiles to skip (all 4 OK):          {fully_ok:5d}  →  0 slot fetches")
    print(f"  Tiles to refetch:                  {n-fully_ok:5d}")
    print(f"  Slot-fetches actually needed:      {total_slots_to_fetch:5d}")
    print(f"  Naive baseline (4 slots/tile):     {naive_baseline:5d}")
    if naive_baseline:
        saved = 100 * (1 - total_slots_to_fetch / naive_baseline)
        print(f"  Per-slot vs naive 4/tile:          {saved:.1f}% fewer slot fetches")
    print()
    print(f"  Rough PU cost @ ~7 PU per slot:    ~{total_slots_to_fetch * 7:,} PU")
    print(f"  (vs ~30k monthly budget)")


if __name__ == "__main__":
    main()
