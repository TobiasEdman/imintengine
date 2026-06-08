#!/usr/bin/env python3
"""Build the from-json fetch list for 512 parents of 256-only tiles.

The ledger shows ~1,147 centers present in 256 but missing from 512. To
make 512 a superset (Option B), fetch a 512 tile at each such center. One
512 parent per center: pick the representative 256 tile (most frames, then
urban-preferred) and carry its name, cohort→source, year, and center.

Year is preserved deliberately — spectral must match the label year
(temporal-matching rule); the 512 fetch reuses the 256 tile's year so the
re-built labels stay consistent.

Output: a JSON array consumable by
``fetch_unified_tiles.py --from-json ... --tile-size-px 512``.

Usage:
  python scripts/build_orphan_fetch_list.py \\
    --ledger /cephfs/audits/tile_ledger.jsonl \\
    --out /cephfs/audits/orphan_fetch_512.json
"""
from __future__ import annotations

import argparse
import collections
import json


def _key(r):
    return (-int(r.get("frames_filled") or 0),
            0 if r.get("cohort") == "urban" else 1,
            r.get("name", ""))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ledger", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    by_center = collections.defaultdict(lambda: {"256": [], "512": []})
    with open(args.ledger) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" in r or r.get("center_e") is None:
                continue
            ds = r.get("dataset")
            if ds in ("256", "512"):
                by_center[(r["center_e"], r["center_n"])][ds].append(r)

    out = []
    cohort_counts = collections.Counter()
    for (e, n), g in by_center.items():
        if g["512"] or not g["256"]:
            continue  # already has a 512 parent, or no 256 tile
        rep = sorted(g["256"], key=_key)[0]   # representative 256 tile
        cohort = rep.get("cohort", "lulc")
        source = "crop" if cohort == "crop" else "lulc"  # crop => LPIS path
        cohort_counts[cohort] += 1
        entry = {
            "name": rep["name"],
            "source": source,
            "cohort": cohort,                 # carried for the ledger
            # carry the center via a minimal bbox; the fetcher recomputes
            # the 512 bbox from the center, so exact extent here is irrelevant
            "bbox_3006": [e - 1, n - 1, e + 1, n + 1],
        }
        if rep.get("year"):
            entry["year"] = rep["year"]
        out.append(entry)

    with open(args.out, "w") as fh:
        json.dump(out, fh)
    print(f"orphan 512-parents to fetch: {len(out)}")
    print(f"  by cohort: {dict(cohort_counts)}")
    print(f"  with year: {sum(1 for e in out if 'year' in e)}/{len(out)}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
