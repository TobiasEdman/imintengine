#!/usr/bin/env python3
"""Sample tiles from audit and run repair_to_canonical_layout in-process
to capture WHY they fail. Logs (status, reason, slot_breakdown) per tile.

Output JSON groups failures by reason → quick way to see if it's
geographic (north/south), VPP-related, DES-data-gap, or something else.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def diagnose_one(loc: dict, output_dir: str, tile, repair_fn) -> dict:
    """Run repair on one tile but capture status without writing."""
    # We can't easily prevent the write side-effect, so just run it
    # and capture the returned status dict. If status=="ok", the tile
    # was actually fixed — we count that too.
    try:
        return repair_fn(loc, output_dir, tile)
    except Exception as e:
        return {"name": loc.get("name", "?"), "status": "exception",
                "reason": f"{type(e).__name__}: {e!s}"[:300]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audit-json", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--tile-size-px", type=int, default=512)
    p.add_argument("--sample-size", type=int, default=30)
    p.add_argument("--output-json", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from imint.training.tile_config import TileConfig
    from scripts.refetch_affected_tiles import loc_from_existing
    from scripts.fetch_unified_tiles import repair_to_canonical_layout

    with open(args.audit_json) as f:
        audit = json.load(f)
    names = (audit.get("unique_affected_tiles")
             or audit.get("unique_problem_tiles")
             or [])
    names = [n[:-4] if n.endswith(".npz") else n for n in names]

    # Sample
    random.seed(args.seed)
    sample = random.sample(names, min(args.sample_size, len(names)))
    print(f"=== diagnosing {len(sample)} sampled tiles from {len(names)} audit ===",
          flush=True)

    tile = TileConfig(size_px=args.tile_size_px)
    results = []
    by_status: Counter = Counter()
    by_reason: Counter = Counter()
    geo_failures: list[tuple[str, int, int]] = []  # (name, easting, northing) for failed

    for i, name in enumerate(sample):
        loc = loc_from_existing(name, tile, args.output_dir)
        if loc is None:
            r = {"name": name, "status": "no_loc"}
        else:
            r = diagnose_one(loc, args.output_dir, tile, repair_to_canonical_layout)

        status = r.get("status", "?")
        reason = r.get("reason", "")
        # Reduce reason cardinality — keep just the prefix before first ":"
        reason_key = reason.split(":")[0] if reason else "(no reason)"
        by_status[status] += 1
        if status in ("failed", "error", "exception"):
            by_reason[reason_key] += 1
            # Get bbox for geo categorisation
            if loc:
                bbox = loc.get("bbox_3006", {})
                geo_failures.append((
                    name,
                    int((bbox.get("west", 0) + bbox.get("east", 0)) // 2),
                    int((bbox.get("south", 0) + bbox.get("north", 0)) // 2),
                ))
        results.append({
            "name": name,
            "status": status,
            "reason": reason[:200],
            "year": loc.get("year") if loc else None,
            "easting": (loc.get("bbox_3006", {}).get("west", 0)
                       + loc.get("bbox_3006", {}).get("east", 0)) // 2 if loc else None,
            "northing": (loc.get("bbox_3006", {}).get("south", 0)
                        + loc.get("bbox_3006", {}).get("north", 0)) // 2 if loc else None,
        })
        print(f"  [{i+1}/{len(sample)}] {name}: {status} {reason[:80]}", flush=True)

    # Geographic bucketing: easting < 5e5 = west, > 6e5 = east
    # northing < 6.5e6 = south, > 7e6 = north
    geo_buckets: dict[str, int] = {"NW": 0, "NE": 0, "SW": 0, "SE": 0, "C": 0}
    for name, e, n in geo_failures:
        ns = "N" if n >= 6_700_000 else "S"
        we = "W" if e <= 500_000 else ("E" if e >= 700_000 else "C")
        key = ns + we if we != "C" else "C"
        geo_buckets[key] = geo_buckets.get(key, 0) + 1

    out = {
        "audit_json": args.audit_json,
        "sample_size": len(sample),
        "scanned_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "by_status": dict(by_status),
        "by_failure_reason": dict(by_reason),
        "geographic_distribution_of_failures": geo_buckets,
        "results": results,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=1)

    print(f"\n=== summary ===", flush=True)
    print(f"  by status: {dict(by_status)}", flush=True)
    print(f"  by failure reason: {dict(by_reason)}", flush=True)
    print(f"  geographic distribution of failures: {geo_buckets}", flush=True)
    print(f"  written: {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
