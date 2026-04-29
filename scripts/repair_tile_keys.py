#!/usr/bin/env python3
"""Repair missing keys in tile .npz files using data already on disk.

The 2026-04-29 audit of /data/unified_v2 showed 6 distinct missing-key
signatures across 8290 tiles. Most are not damage — they are mid-pipeline
lifecycle states or legacy formats that can be repaired without re-fetching
spectral data.

This script handles the mechanical repairs (Step 1) only:

  • Alias  ``image`` → ``spectral`` (4 legacy tiles).
  • Derive ``easting``/``northing`` from either:
      - ``bbox_3006`` (preferred — exact center used at fetch time)
      - filename pattern ``tile_<east>_<north>.npz``
      - filename pattern ``crop_<crop>_<east>_<north>.npz`` /
        ``urban_<east>_<north>.npz``
  • Add ``tile_size_px`` if absent (derived from spectral or image shape)

It does NOT:
  • Fetch any spectral data (use Stage B if you want full enrichment)
  • Re-build labels (use scripts/build_labels.py for that — Step 2)
  • Touch tiles whose missing keys can't be derived from on-disk data

Atomic writes via tmp + os.replace (same pattern as build_labels.py:344
and the post-2026-04-28 enrich scripts).

Usage:
    python scripts/repair_tile_keys.py \\
        --data-dir /data/unified_v2 \\
        --workers 8 \\
        --report /data/debug/repair_report.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# Filename patterns — must match imint.training.tile_bbox._FILENAME_RE
# plus crop_/urban_ variants used in unified_v2.
_TILE_RE  = re.compile(r"^tile_(\d+)_(\d+)\.npz$")
_PREFIXED_RE = re.compile(r"^(?:crop|urban)_.*_(\d+)_(\d+)\.npz$")
_NUMERIC_RE = re.compile(r"^\d+\.npz$")


def _derive_center(name: str, data: dict) -> tuple[int | None, int | None, str]:
    """Best-effort recovery of (easting, northing) and the source used.

    Order:
      1. bbox_3006 array — exact, set at fetch time, no rounding
      2. tile_*  filename
      3. crop_/urban_ filename suffix
      4. Numeric filenames have no coordinate signal — fall through
    """
    bbox = data.get("bbox_3006", None)
    if bbox is not None:
        b = np.asarray(bbox).flatten()
        if b.size >= 4:
            cx = (int(b[0]) + int(b[2])) // 2
            cy = (int(b[1]) + int(b[3])) // 2
            return cx, cy, "bbox_3006"

    m = _TILE_RE.match(name)
    if m:
        return int(m.group(1)), int(m.group(2)), "tile_filename"

    m = _PREFIXED_RE.match(name)
    if m:
        return int(m.group(1)), int(m.group(2)), "prefixed_filename"

    return None, None, "unrecoverable"


def repair_one_tile(tile_path: str) -> dict:
    """Apply mechanical key repairs to a single .npz; return action log."""
    name = os.path.basename(tile_path)
    actions: list[str] = []

    try:
        with np.load(tile_path, allow_pickle=True) as d:
            data = {k: d[k] for k in d.files}
    except Exception as e:
        return {"name": name, "status": "load_error", "error": str(e)[:200]}

    keys_before = set(data.keys())

    # ── Alias image → spectral (4 legacy tiles) ──────────────────────
    if "spectral" not in data and "image" in data:
        arr = data["image"]
        if isinstance(arr, np.ndarray) and arr.ndim == 3:
            data["spectral"] = arr
            actions.append("alias_image_to_spectral")

    # ── Derive easting / northing if missing ─────────────────────────
    needs_easting = "easting" not in data
    needs_northing = "northing" not in data
    if needs_easting or needs_northing:
        e, n, src = _derive_center(name, data)
        if e is not None and n is not None:
            data["easting"] = np.int64(e)
            data["northing"] = np.int64(n)
            actions.append(f"derive_easting_northing_from_{src}")

    # ── Derive tile_size_px if missing ───────────────────────────────
    if "tile_size_px" not in data:
        for k in ("spectral", "image"):
            arr = data.get(k)
            if isinstance(arr, np.ndarray) and arr.ndim == 3:
                data["tile_size_px"] = np.int32(arr.shape[1])
                actions.append("derive_tile_size_px")
                break

    keys_after = set(data.keys())
    added = sorted(keys_after - keys_before)

    if not actions:
        return {"name": name, "status": "no_change", "added": []}

    # Atomic write — tmp + os.replace, same convention as build_labels.py:344
    tmp_base = tile_path + ".tmp"
    np.savez_compressed(tmp_base, **data)
    os.replace(tmp_base + ".npz", tile_path)

    return {
        "name": name,
        "status": "repaired",
        "actions": actions,
        "added_keys": added,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Mechanical key repair for tile .npz files."
    )
    p.add_argument("--data-dir", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-tiles", type=int, default=None)
    p.add_argument(
        "--report",
        default=None,
        help="If given, write a per-tile JSON action log here.",
    )
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if args.max_tiles:
        files = files[: args.max_tiles]

    print(f"=== Repair tile keys ===")
    print(f"  Tiles:   {len(files)}")
    print(f"  Workers: {args.workers}")
    print(f"  Report:  {args.report or '(none)'}")

    counters = {
        "repaired": 0,
        "no_change": 0,
        "load_error": 0,
    }
    action_counts: dict[str, int] = {}
    lock = threading.Lock()
    completed = 0
    t0 = time.time()
    report: list[dict] = []

    def _run(path: str) -> None:
        nonlocal completed
        r = repair_one_tile(path)
        with lock:
            completed += 1
            counters[r["status"]] = counters.get(r["status"], 0) + 1
            for a in r.get("actions", []):
                action_counts[a] = action_counts.get(a, 0) + 1
            if args.report:
                report.append(r)
            if completed % 500 == 0 or completed == len(files):
                elapsed = time.time() - t0
                rate = completed / elapsed * 60 if elapsed > 0 else 0
                print(
                    f"  [{completed}/{len(files)}] rate {rate:.0f}/min  "
                    f"repaired={counters['repaired']} "
                    f"no_change={counters['no_change']} "
                    f"errors={counters['load_error']}",
                    flush=True,
                )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_run, f): f for f in files}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                print(f"  worker error: {e}", flush=True)

    print(f"\n=== Done in {(time.time() - t0) / 60:.1f} min ===")
    print(f"  repaired:  {counters['repaired']}")
    print(f"  no_change: {counters['no_change']}")
    print(f"  errors:    {counters['load_error']}")
    print(f"  action breakdown:")
    for act, n in sorted(action_counts.items(), key=lambda kv: -kv[1]):
        print(f"    [{n}] {act}")

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w") as fh:
            json.dump(
                {
                    "summary": {
                        "total": len(files),
                        **counters,
                        "actions": action_counts,
                    },
                    "tiles": report,
                },
                fh,
                indent=2,
                default=str,
            )
        print(f"  Report written: {args.report}")


if __name__ == "__main__":
    main()
