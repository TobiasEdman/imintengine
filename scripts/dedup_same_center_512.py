#!/usr/bin/env python3
"""Dedup same-center 512 tiles: keep the most-complete, urban-preferred.

The tile ledger surfaced ~142 centers with two 512 tiles — a `tile_` (lulc)
and an `urban_` (2022 densification) tile, from DIFFERENT years (not the
same imagery, so not byte-duplicates). Per decision: keep ONE tile per
center, choosing:
    1. most frames_filled (full 4-frame tiles win — never lose frames),
    2. urban-preferred on ties (keep the newer densification pass),
    3. name as final deterministic tiebreak.
The other tile(s) at that center are dropped.

Reads the ledger (scripts/build_tile_ledger.py output) — no per-tile loads.
Default DRY-RUN (writes a plan, deletes nothing). --execute removes the
losers from --data-dir. Run only when no other 512 writer is active.

Usage:
  python scripts/dedup_same_center_512.py \\
    --ledger /cephfs/audits/tile_ledger.jsonl \\
    --data-dir /cephfs/unified_v2_512 \\
    --plan-out /cephfs/audits/dedup_plan_512.json            # dry-run
  ... --execute                                               # apply
"""
from __future__ import annotations

import argparse
import collections
import json
import os


def _key(row):
    """Sort key: most frames first, urban-preferred, then name."""
    return (
        -int(row.get("frames_filled") or 0),
        0 if row.get("cohort") == "urban" else 1,
        row.get("name", ""),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ledger", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--plan-out", required=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()

    by_center = collections.defaultdict(list)
    n = 0
    with open(args.ledger) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" in r or r.get("dataset") != "512" or r.get("center_e") is None:
                continue
            by_center[(r["center_e"], r["center_n"])].append(r)
            n += 1

    keep, drop = [], []
    for c, lst in by_center.items():
        if len(lst) < 2:
            continue
        ranked = sorted(lst, key=_key)
        winner = ranked[0]
        keep.append(winner["name"])
        for loser in ranked[1:]:
            drop.append({
                "drop": loser["name"], "keep": winner["name"],
                "center": f"{c[0]}_{c[1]}",
                "drop_frames": loser.get("frames_filled"),
                "keep_frames": winner.get("frames_filled"),
                "drop_cohort": loser.get("cohort"),
                "keep_cohort": winner.get("cohort"),
            })

    # Sanity: never drop a fuller tile than the one we keep.
    bad = [d for d in drop if (d["drop_frames"] or 0) > (d["keep_frames"] or 0)]
    by_keepcohort = collections.Counter(d["keep_cohort"] for d in drop)
    plan = {
        "data_dir": args.data_dir, "executed": args.execute,
        "n_512_tiles_scanned": n,
        "multi_center_groups": sum(1 for l in by_center.values() if len(l) > 1),
        "n_keep": len(keep), "n_drop": len(drop),
        "kept_cohort_counts": dict(by_keepcohort),
        "INVARIANT_VIOLATIONS_drop_fuller_than_keep": len(bad),
        "drops": drop,
    }
    os.makedirs(os.path.dirname(args.plan_out) or ".", exist_ok=True)
    with open(args.plan_out, "w") as fh:
        json.dump(plan, fh, indent=0)
    print(f"512 scanned={n}  multi-center groups={plan['multi_center_groups']}")
    print(f"  keep={len(keep)}  drop={len(drop)}  kept-cohort={dict(by_keepcohort)}")
    print(f"  invariant (never drop a fuller tile): violations={len(bad)}")
    if bad[:3]:
        print(f"  VIOLATION examples: {bad[:3]}")
    print(f"  wrote plan {args.plan_out}")

    if not args.execute:
        print("  DRY-RUN — nothing deleted. Re-run with --execute to apply.")
        return
    if bad:
        raise SystemExit("ABORT: invariant violated (would drop fuller tiles).")
    removed = 0
    for d in drop:
        path = os.path.join(args.data_dir, d["drop"] + ".npz")
        if os.path.exists(path):
            os.remove(path); removed += 1
    print(f"  EXECUTED: removed {removed} loser tiles")


if __name__ == "__main__":
    main()
