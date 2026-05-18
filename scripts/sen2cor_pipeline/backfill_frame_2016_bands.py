#!/usr/bin/env python3
"""Backfill the frame_2016_bands metadata field on existing tiles.

The frame_2016_bands field (added 2026-05-18) records the 6-band order
of a tile's frame_2016. Tiles written before it existed lack the field,
which would make select_scenes.py treat them as missing and trigger a
needless re-fetch.

Existing frame_2016 frames came from two sources:

  * CDSE burst (add_background_frame.py → fetch_background_frame →
    fetch_s2_scene). cdse_s2._PRITHVI_BANDS proves these are B8A — the
    canonical order. They only need the field stamped.

  * sen2cor pipeline (run_sen2cor_per_scene.py) — identifiable by the
    frame_2016_scene key. Pre-fix sen2cor tiles were B08 and must NOT be
    backfilled; they need a real re-fetch. Post-fix sen2cor tiles
    already carry frame_2016_bands. So any tile with frame_2016_scene is
    left untouched.

This is a metadata-only, in-place, idempotent rewrite — no network, no
sen2cor compute. Re-running it skips tiles already stamped.

Usage:
    python scripts/sen2cor_pipeline/backfill_frame_2016_bands.py \\
        --data-dir /data/unified_v2_512 --workers 8
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from imint.training.tile_fetch import PRITHVI_BANDS

_stats = {"stamped": 0, "skip_no_frame": 0, "skip_already": 0,
          "skip_sen2cor": 0, "corrupt": 0}
_lock = threading.Lock()


def _bump(key: str) -> None:
    with _lock:
        _stats[key] += 1


def _process(npz_path: str) -> None:
    try:
        with np.load(npz_path, allow_pickle=True) as d:
            keys = set(d.files)
            if "frame_2016" not in keys or int(d.get("has_frame_2016", 0)) != 1:
                _bump("skip_no_frame")
                return
            if "frame_2016_bands" in keys:
                _bump("skip_already")
                return
            if "frame_2016_scene" in keys:
                # sen2cor-origin without the field → pre-fix B08; leave it
                # for select_scenes to flag and re-fetch.
                _bump("skip_sen2cor")
                return
            data = {k: d[k] for k in d.files}
    except (zipfile.BadZipFile, EOFError, OSError, ValueError):
        _bump("corrupt")
        return

    data["frame_2016_bands"] = np.array(PRITHVI_BANDS)
    tmp = npz_path + ".tmp"
    np.savez_compressed(tmp, **data)
    os.replace(tmp + ".npz", npz_path)
    _bump("stamped")


def main() -> None:
    p = argparse.ArgumentParser(description="Backfill frame_2016_bands metadata")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    print(f"=== backfill frame_2016_bands ===")
    print(f"  data-dir: {args.data_dir}")
    print(f"  tiles:    {len(files)}")
    print(f"  bands:    {PRITHVI_BANDS}")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_process, f) for f in files]
        for i, fut in enumerate(as_completed(futs)):
            fut.result()
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(files)}  stamped={_stats['stamped']}  "
                      f"({time.time() - t0:.0f}s)", flush=True)

    print(f"\n=== done in {(time.time() - t0) / 60:.1f} min ===")
    for k, v in _stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
