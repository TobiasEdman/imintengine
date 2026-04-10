#!/usr/bin/env python3
"""Remap tile labels to unified 20-class schema.

Post-processing step after spectral fetching. Applies merge_all()
which combines NMD + LPIS + SKS harvest into unified labels.

Pipeline: NMD 19-class sequential → nmd19_to_unified()
         + LPIS label_mask SJV codes → crop classes 11-21
         + SKS harvest_mask → hygge class 22

Usage:
    python scripts/remap_labels.py --data-dir /data/unified_v2
    python scripts/remap_labels.py --data-dir /data/unified_v2 --workers 4
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imint.training.unified_schema import merge_all, UNIFIED_CLASS_NAMES


def remap_tile(path: str) -> dict:
    """Remap a single tile's label to unified 20-class."""
    name = os.path.basename(path).replace(".npz", "")
    try:
        data = dict(np.load(path, allow_pickle=True))

        if "label" not in data:
            return {"name": name, "status": "skipped", "reason": "no_label"}

        nmd_label = data["label"]
        lpis_mask = data.get("label_mask", None)
        harvest_mask = data.get("harvest_mask", None)

        # Apply unified remapping
        unified = merge_all(nmd_label, lpis_mask, harvest_mask)
        data["label"] = unified

        # Save back
        np.savez_compressed(path, **data)
        return {"name": name, "status": "ok"}

    except Exception as e:
        return {"name": name, "status": "failed", "reason": str(e)[:100]}


def main():
    p = argparse.ArgumentParser(description="Remap tile labels to unified 20-class")
    p.add_argument("--data-dir", required=True, help="Directory with .npz tiles")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    import glob
    tiles = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    print(f"Remapping {len(tiles)} tiles in {args.data_dir}")
    print(f"Schema: {len(UNIFIED_CLASS_NAMES)} classes")
    print(f"  Classes: {', '.join(UNIFIED_CLASS_NAMES[:5])} ... {', '.join(UNIFIED_CLASS_NAMES[-3:])}")

    stats = {"ok": 0, "skipped": 0, "failed": 0}
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(remap_tile, t): t for t in tiles}
        for i, f in enumerate(as_completed(futs)):
            r = f.result()
            stats[r["status"]] = stats.get(r["status"], 0) + 1
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(tiles)}] {r['name']}: {r['status']} "
                      f"| {(i+1)/elapsed*3600:.0f}/h", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  OK={stats['ok']}  Skipped={stats['skipped']}  Failed={stats['failed']}")


if __name__ == "__main__":
    main()
