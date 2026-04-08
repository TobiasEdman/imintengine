#!/usr/bin/env python3
"""
For tiles that failed autumn-frame patching (no clear 2017 scene):
  - n_parcels == 0  →  non-crop tile: keep 2018 autumn frame as-is
  - n_parcels  > 0  →  crop tile: zero out frame 0, set dates[0]=''

Zeroing the crop-tile autumn frame signals "unavailable" to the model
instead of providing the wrong-year (2018) temporal context.

Reads: --fail-log (from patch_autumn_frame.py, default /data/patch_autumn_failed.json)
Writes: tiles in-place (atomic .tmp.npz → rename)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

N_BANDS = 6   # must match tile_fetch.py N_BANDS


def blank_one(name: str, data_dir: Path, dry_run: bool) -> dict:
    fp = data_dir / name
    if not fp.exists():
        return {"name": name, "status": "missing"}

    try:
        d = dict(np.load(fp, allow_pickle=True))
    except Exception as e:
        return {"name": name, "status": "load_error", "msg": str(e)}

    # Already blanked or already patched with 2017 data — skip
    dates = list(d["dates"])
    if str(dates[0]) == "" or not str(dates[0]).startswith("2018"):
        return {"name": name, "status": "skip", "d0": str(dates[0])}

    n_parcels = int(d.get("n_parcels", 0))
    if n_parcels == 0:
        return {"name": name, "status": "non_crop_kept", "n_parcels": 0}

    # Crop tile — zero frame 0 and mark as unavailable
    if dry_run:
        return {"name": name, "status": "dry_blanked", "n_parcels": n_parcels,
                "old_d0": str(dates[0])}

    try:
        img = d["image"].copy()           # (4*N_BANDS, H, W)
        img[:N_BANDS] = 0.0               # zero out frame 0 bands

        dates[0] = ""
        d["image"] = img
        d["dates"] = np.array(dates)

        # Also clear DOY and temporal_mask for frame 0
        if "doy" in d:
            doy = d["doy"].copy()
            doy[0] = 0
            d["doy"] = doy
        if "temporal_mask" in d:
            tm = d["temporal_mask"].copy()
            tm[0] = 0
            d["temporal_mask"] = tm

        tmp = fp.with_suffix(".tmp.npz")
        np.savez_compressed(str(tmp), **d)
        tmp.rename(fp)

        return {"name": name, "status": "blanked", "n_parcels": n_parcels}

    except Exception as e:
        return {"name": name, "status": "write_error", "msg": str(e)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fail-log",  default="/data/patch_autumn_failed.json")
    ap.add_argument("--data-dir",  default="/data/unified_v2")
    ap.add_argument("--out-log",   default="/data/blank_crop_result.json")
    ap.add_argument("--workers",   type=int, default=6)
    ap.add_argument("--dry-run",   action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    with open(args.fail_log) as f:
        failed = json.load(f)

    names = [e["name"] for e in failed]
    print(f"Processing {len(names)} failed tiles  dry_run={args.dry_run}")

    results: list[dict] = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(blank_one, n, data_dir, args.dry_run): n for n in names}
        for i, fut in enumerate(as_completed(futs)):
            r = fut.result()
            results.append(r)
            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"  [{i+1}/{len(names)}]  ({rate:.1f} tiles/s)", flush=True)

    from collections import Counter
    counts = Counter(r["status"] for r in results)
    print(f"\nDone in {time.time()-t0:.0f}s")
    for s, n in counts.most_common():
        print(f"  {s}: {n}")

    with open(args.out_log, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results → {args.out_log}")


if __name__ == "__main__":
    main()
