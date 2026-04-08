#!/usr/bin/env python3
"""
Replace frame 0 (autumn y-1) for same-year tiles.

These tiles have all 4 seasonal composites from the same year (e.g. all 2018).
Frame 0 should be Sep-Oct of primary_year-1 (e.g. Sep-Oct 2017) but was
accidentally filled with a same-year autumn scene instead.

This script:
  1. Reads the patch list from analyze_date_issues.py (--patch-list)
  2. For each tile, fetches a cloud-free Sep-Oct (primary_year-1) scene from CDSE
  3. Replaces bands 0..N_BANDS-1 in the stored image array in-place
  4. Updates dates[0] to the new acquisition date
  5. Writes the updated tile back (atomic: .tmp.npz → rename)

Tiles where no cloud-free scene is available are written to --fail-log.

Usage (on pod):
    python scripts/patch_autumn_frame.py \\
        --patch-list /tmp/patch_list.json \\
        --data-dir   /data/unified_v2 \\
        --workers    6

    # Dry-run (no disk writes, just fetches and reports):
    python scripts/patch_autumn_frame.py --patch-list /tmp/patch_list.json --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imint.config.env import load_env
load_env()

from imint.training.tile_fetch import (
    _fetch_single_scene,
    bbox_3006_to_wgs84,
    N_BANDS,
)

CLOUD_MAX     = 0.30
MAX_CANDIDATES = 5


def patch_one(entry: dict, data_dir: Path, dry_run: bool) -> dict:
    name = entry["name"]
    fp   = data_dir / name

    if not fp.exists():
        return {"name": name, "status": "missing"}

    # Build bbox dicts from stored [W, S, E, N]
    W, S, E, N = entry["bbox"]
    bbox_3006  = {"west": W, "south": S, "east": E, "north": N}
    coords     = bbox_3006_to_wgs84(bbox_3006)

    # Fetch the autumn y-1 scene
    try:
        scene, date = _fetch_single_scene(
            bbox_3006, coords,
            entry["fetch_from"], entry["fetch_to"],
            scene_cloud_max=CLOUD_MAX,
            max_candidates=MAX_CANDIDATES,
        )
    except Exception as e:
        return {"name": name, "status": "fetch_error", "msg": str(e)}

    if scene is None:
        return {
            "name":   name,
            "status": "no_clear_scene",
            "range":  f"{entry['fetch_from']} – {entry['fetch_to']}",
        }

    if scene.shape[0] != N_BANDS:
        return {"name": name, "status": "wrong_bands",
                "got": scene.shape[0], "expected": N_BANDS}

    if dry_run:
        return {"name": name, "status": "dry_ok",
                "new_d0": date, "old_d0": entry["current_d0_date"]}

    try:
        d = dict(np.load(fp, allow_pickle=True))
        img = d["image"]   # (4*N_BANDS, H, W)
        H, W_px = img.shape[1], img.shape[2]

        # Resize if needed (should already be 256×256)
        if scene.shape[1] != H or scene.shape[2] != W_px:
            from PIL import Image as PILImage
            scene = np.stack([
                np.array(PILImage.fromarray(scene[b]).resize(
                    (W_px, H), PILImage.BILINEAR))
                for b in range(N_BANDS)
            ])

        img = img.copy()
        img[:N_BANDS] = scene  # replace frame 0

        dates      = list(d["dates"])
        old_d0     = str(dates[0])
        dates[0]   = date
        d["image"] = img
        d["dates"] = np.array(dates)

        tmp = fp.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **d)
        tmp.rename(fp)

        return {"name": name, "status": "ok",
                "old_d0": old_d0, "new_d0": date}

    except Exception as e:
        return {"name": name, "status": "write_error", "msg": str(e)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Patch autumn frame-0 for same-year tiles")
    ap.add_argument("--patch-list", required=True,
                    help="JSON from analyze_date_issues.py --patch-out")
    ap.add_argument("--data-dir",   default="/data/unified_v2")
    ap.add_argument("--fail-log",   default="/tmp/patch_autumn_failed.json")
    ap.add_argument("--done-log",   default="/tmp/patch_autumn_done.json")
    ap.add_argument("--workers",    type=int, default=4)
    ap.add_argument("--limit",      type=int, default=0,
                    help="Process only first N tiles (debug)")
    ap.add_argument("--dry-run",    action="store_true",
                    help="Fetch scenes but don't write to disk")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    with open(args.patch_list) as f:
        entries = json.load(f)

    if args.limit > 0:
        entries = entries[:args.limit]

    print(f"Patching frame-0 for {len(entries)} tiles")
    print(f"  data_dir={data_dir}  dry_run={args.dry_run}  workers={args.workers}")

    ok, fail, skip = 0, 0, 0
    failed_list: list[dict] = []
    done_list:   list[dict] = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(patch_one, e, data_dir, args.dry_run): e
                for e in entries}
        for i, fut in enumerate(as_completed(futs)):
            r = fut.result()
            s = r["status"]
            if s in ("ok", "dry_ok"):
                ok += 1
                done_list.append(r)
            elif s == "missing":
                skip += 1
            else:
                fail += 1
                failed_list.append(r)

            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                rate    = (i + 1) / elapsed
                print(f"  [{i+1}/{len(entries)}] ok={ok} fail={fail} skip={skip}"
                      f"  ({rate:.1f} tiles/s)", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  OK: {ok}  Failed: {fail}  Skipped: {skip}")

    with open(args.fail_log, "w") as f:
        json.dump(failed_list, f, indent=2)
    with open(args.done_log, "w") as f:
        json.dump(done_list, f, indent=2)
    print(f"  Failed → {args.fail_log}")
    print(f"  Done   → {args.done_log}")

    if failed_list:
        from collections import Counter
        reasons = Counter(r["status"] for r in failed_list)
        print("\n  Failure breakdown:")
        for r, n in reasons.most_common():
            print(f"    {r}: {n}")


if __name__ == "__main__":
    main()
