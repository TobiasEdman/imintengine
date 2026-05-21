#!/usr/bin/env python3
"""Re-fetch tiles flagged by audit-late-autumn-frames as having a
growing-season frame with DOY > 244 (pre-PR#15 VPP-window bug).

Reads the audit JSON, iterates affected tiles in a ThreadPool, calls
``refetch_tile(..., force=True)`` so the existing tile is overwritten
with frames computed from the now-capped VPP windows.

Usage (k8s pod):
    python scripts/refetch_affected_tiles.py \\
        --audit-json /checkpoints/audits/late_autumn_frames_512.json \\
        --output-dir /cephfs/unified_v2_512 \\
        --tile-size-px 512 \\
        --workers 6

The audit JSON's ``unique_affected_tiles`` list defines the work.
Each tile is read from disk (output-dir), its bbox / year / source
metadata is extracted, and refetch_tile is invoked. Progress is
logged every 50 tiles plus a final summary.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imint.config.env import load_env
load_env()

from imint.training.tile_bbox import resolve_tile_bbox
from imint.training.tile_fetch import bbox_3006_to_wgs84
from imint.training.tile_config import TileConfig
from scripts.fetch_unified_tiles import repair_to_canonical_layout


def loc_from_existing(name: str, tile: TileConfig, tiles_dir: str) -> dict | None:
    """Build a refetch_tile-compatible loc dict from an existing .npz."""
    path = os.path.join(tiles_dir, f"{name}" if name.endswith(".npz") else f"{name}.npz")
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path, allow_pickle=True)
    except Exception:
        return None
    bbox = resolve_tile_bbox(name=name.replace(".npz", ""), tile=tile, npz_data=data)
    if bbox is None:
        return None

    # Read base year
    tile_year = None
    if "year" in data.files:
        tile_year = int(data["year"])
    elif "lpis_year" in data.files:
        tile_year = int(data["lpis_year"])
    elif "dates" in data.files:
        for d in data["dates"]:
            d_str = str(d)
            if d_str and len(d_str) >= 4:
                tile_year = int(d_str[:4])
                break

    has_lpis = "label_mask" in data.files or "lpis_year" in data.files

    return {
        "name": name.replace(".npz", ""),
        "source": str(data.get("source", "lulc")) if "source" in data.files else "lulc",
        "bbox_3006": bbox,
        "coords_wgs84": bbox_3006_to_wgs84(bbox),
        "year": tile_year,
        "_existing_path": path,
        "_has_lpis": has_lpis,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audit-json", required=True,
                   help="Audit-job output (unique_affected_tiles field)")
    p.add_argument("--output-dir", required=True,
                   help="Where re-fetched tiles land (e.g. /cephfs/unified_v2_512)")
    p.add_argument("--tile-size-px", type=int, required=True,
                   help="256 or 512 — must match output-dir")
    p.add_argument("--workers", type=int, default=6,
                   help="Concurrent fetches (default 6)")
    p.add_argument("--cloud-max", type=float, default=30.0,
                   help="STAC scene_cloud_max (legacy, ERA5 prefilter replaces it)")
    p.add_argument("--max-aoi-cloud", type=float, default=0.10,
                   help="SCL AOI cloud threshold (0.10 = 10%%, per Lund-benchmark M4 "
                        "ERA5->SCL — lowest mean COT 0.0086)")
    p.add_argument("--sources", default="des,cdse",
                   help="Comma-separated fetch backends in priority order")
    p.add_argument("--max-tiles", type=int, default=None,
                   help="For smoke-testing; limits to first N tiles")
    p.add_argument("--cap-doy", type=int, default=244,
                   help="Refetch any growing-season frame with DOY > cap (default 244=Sep 1, matches PR #15 VPP cap)")
    args = p.parse_args()

    sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())

    print(f"=== refetch_affected_tiles ===", flush=True)
    print(f"  audit-json:    {args.audit_json}", flush=True)
    print(f"  output-dir:    {args.output_dir}", flush=True)
    print(f"  tile-size-px:  {args.tile_size_px}", flush=True)
    print(f"  workers:       {args.workers}", flush=True)
    print(f"  sources:       {sources}", flush=True)
    print(f"  cloud-max:     {args.cloud_max}", flush=True)
    print(f"  max-aoi-cloud: {args.max_aoi_cloud}", flush=True)

    with open(args.audit_json) as f:
        audit = json.load(f)
    names = audit.get("unique_affected_tiles", [])
    if args.max_tiles:
        names = names[: args.max_tiles]
    print(f"  affected tiles to re-fetch: {len(names)}", flush=True)

    tile = TileConfig(size_px=args.tile_size_px)

    # Build locs (sequential — cheap I/O)
    print(f"\n=== building loc dicts ===", flush=True)
    locs = []
    skipped_no_loc = 0
    for n in names:
        loc = loc_from_existing(n, tile, args.output_dir)
        if loc is None:
            skipped_no_loc += 1
            continue
        locs.append(loc)
    print(f"  built {len(locs)} locs ({skipped_no_loc} skipped — no .npz or bbox)", flush=True)

    if not locs:
        print(f"  nothing to do; exiting"); return

    # Execute with thread pool
    print(f"\n=== re-fetching with {args.workers} workers ===", flush=True)
    t0 = time.time()
    results = []
    completed = 0
    by_status = {"ok": 0, "failed": 0, "skipped": 0, "error": 0}

    def task(loc):
        try:
            return repair_to_canonical_layout(
                loc, args.output_dir, tile,
                cloud_max=args.cloud_max,
                max_aoi_cloud=args.max_aoi_cloud,
                sources=sources,
                cap_doy=args.cap_doy,
            )
        except Exception as e:
            return {"name": loc["name"], "status": "error", "reason": str(e)[:200]}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(task, loc): loc for loc in locs}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            completed += 1
            status = r.get("status", "unknown")
            by_status[status] = by_status.get(status, 0) + 1
            if completed % 25 == 0 or completed == len(locs):
                elapsed = time.time() - t0
                rate = completed / max(elapsed / 3600, 1e-6)
                eta_min = (len(locs) - completed) / max(rate / 60, 1e-6)
                print(
                    f"  [{completed}/{len(locs)}] "
                    f"status={by_status} "
                    f"rate={rate:.0f}/h ETA={eta_min:.1f}min",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(f"\n=== done in {elapsed/60:.1f} min ===", flush=True)
    print(f"  by status: {by_status}", flush=True)
    failed = [r for r in results if r.get("status") == "failed"]
    errors = [r for r in results if r.get("status") == "error"]
    if failed:
        print(f"\n  failed ({len(failed)}): first 10:", flush=True)
        for r in failed[:10]:
            print(f"    {r['name']}: {r.get('reason', '?')}", flush=True)
    if errors:
        print(f"\n  errors ({len(errors)}): first 10:", flush=True)
        for r in errors[:10]:
            print(f"    {r['name']}: {r.get('reason', '?')}", flush=True)


if __name__ == "__main__":
    main()
