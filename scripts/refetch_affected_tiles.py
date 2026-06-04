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
import threading
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


class SkipIndex:
    """Persistent set of audit tiles that prior runs confirmed
    OK ("status": "skipped" or "ok" from ``repair_to_canonical_layout``).

    Eliminates the per-iteration skip-burst: instead of re-loading
    every audit tile's .npz to discover it's already in canonical
    layout, we record the verdict once and short-circuit the next
    run. Keyed by the audit-JSON basename so different audits don't
    cross-contaminate.

    Thread-safe (single lock around the in-memory set + flush). Flushes
    every ``flush_every`` updates via temp-file + atomic ``os.replace``
    so a pod kill mid-write never corrupts the on-disk file.

    The skip-index is *durable* across runs because the only thing that
    modifies a tile's .npz is a successful refetch — and a successful
    refetch leaves the tile in canonical layout, so it stays "known OK"
    regardless. If you ever need to invalidate the index (e.g. you
    deliberately corrupted a tile to force a re-run), delete the
    index file.
    """

    def __init__(self, path: str, audit_basename: str, *, flush_every: int = 100):
        self.path = path
        self.audit_basename = audit_basename
        self.flush_every = flush_every
        self._lock = threading.Lock()
        self._known: set[str] = set()
        self._pending = 0
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            data = json.load(open(self.path))
        except Exception as e:
            print(f"  [skip-index] load failed: {e}", flush=True)
            return
        # Cross-audit guard — different audit JSON ⇒ different work list,
        # don't reuse another audit's set.
        if data.get("audit") != self.audit_basename:
            print(
                f"  [skip-index] audit mismatch "
                f"(file: {data.get('audit')!r}, want: {self.audit_basename!r}); "
                f"ignoring",
                flush=True,
            )
            return
        self._known = set(data.get("tiles_ok", []))
        print(
            f"  [skip-index] loaded {len(self._known)} known-OK tiles "
            f"from {self.path}",
            flush=True,
        )

    def __contains__(self, name: str) -> bool:
        return name in self._known

    def __len__(self) -> int:
        return len(self._known)

    def add(self, name: str) -> None:
        with self._lock:
            if name in self._known:
                return
            self._known.add(name)
            self._pending += 1
            if self._pending >= self.flush_every:
                self._flush_locked()

    def _flush_locked(self) -> None:
        """Caller must hold ``self._lock``. Atomic via temp-file + rename."""
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(
                    {
                        "audit": self.audit_basename,
                        "updated_utc": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        ),
                        "n_tiles": len(self._known),
                        "tiles_ok": sorted(self._known),
                    },
                    f,
                )
            os.replace(tmp, self.path)
            self._pending = 0
        except Exception as e:
            print(f"  [skip-index] flush failed: {e}", flush=True)

    def flush_final(self) -> None:
        with self._lock:
            if self._pending > 0 or not os.path.exists(self.path):
                self._flush_locked()
        print(
            f"  [skip-index] final: {len(self._known)} known-OK tiles "
            f"persisted to {self.path}",
            flush=True,
        )


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
    p.add_argument("--skip-index-path", default=None,
                   help="Persistent skip-index JSON. Default: "
                        "{audit_dir}/skip_index_{audit_basename}. "
                        "Tiles confirmed OK by a prior run are fast-skipped "
                        "(no .npz load) on subsequent runs.")
    p.add_argument("--no-skip-index", action="store_true",
                   help="Disable the skip-index entirely (always full scan).")
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
    # Accept both audit formats:
    # - audit-late-autumn-frames: "unique_affected_tiles"
    # - audit-frame-problems:     "unique_problem_tiles"
    names = (audit.get("unique_affected_tiles")
             or audit.get("unique_problem_tiles")
             or [])
    # audit-frame-problems entries include ".npz"; strip for consistency.
    names = [n[:-4] if n.endswith(".npz") else n for n in names]
    if args.max_tiles:
        names = names[: args.max_tiles]
    print(f"  affected tiles to re-fetch: {len(names)}", flush=True)

    tile = TileConfig(size_px=args.tile_size_px)

    # Persistent skip-index. Default file colocates with the audit JSON:
    # /cephfs/audits/skip_index_<audit_basename>. Same audit across runs
    # ⇒ same skip-index ⇒ no re-loading of confirmed-OK tiles' .npz.
    audit_basename = os.path.basename(args.audit_json)
    default_skip_idx = os.path.join(
        os.path.dirname(args.audit_json) or ".",
        f"skip_index_{audit_basename}",
    )
    skip_index = (
        None
        if args.no_skip_index
        else SkipIndex(args.skip_index_path or default_skip_idx, audit_basename)
    )
    print(
        f"  skip-index:    "
        + (f"{skip_index.path} ({len(skip_index)} pre-known)"
           if skip_index else "DISABLED (--no-skip-index)"),
        flush=True,
    )

    # Build locs — fast-skip tiles already in skip-index (no .npz load).
    print(f"\n=== building loc dicts ===", flush=True)
    locs = []
    skipped_no_loc = 0
    fast_skip_count = 0
    for n in names:
        if skip_index is not None and n in skip_index:
            fast_skip_count += 1
            continue
        loc = loc_from_existing(n, tile, args.output_dir)
        if loc is None:
            skipped_no_loc += 1
            continue
        locs.append(loc)
    print(
        f"  built {len(locs)} locs "
        f"({fast_skip_count} fast-skipped via index, "
        f"{skipped_no_loc} skipped — no .npz or bbox)",
        flush=True,
    )

    # Pre-credit fast-skips into the by_status counter so the progress
    # line + downstream dashboards see the full picture from tile 0,
    # not "ok=0 skipped=0" until the first pool result lands.
    total = len(names)
    by_status = {
        "ok": 0,
        "failed": 0,
        "skipped": fast_skip_count,
        "error": 0,
    }
    pre_credit = fast_skip_count + skipped_no_loc
    if fast_skip_count or skipped_no_loc:
        print(
            f"  [{pre_credit}/{total}] status={by_status} "
            f"(pre-credit: {fast_skip_count} fast-skip + "
            f"{skipped_no_loc} no-npz)",
            flush=True,
        )

    if not locs:
        print(f"  nothing to do; exiting", flush=True)
        if skip_index is not None:
            skip_index.flush_final()
        return

    # Execute with thread pool
    print(f"\n=== re-fetching with {args.workers} workers ===", flush=True)
    t0 = time.time()
    results = []
    completed = 0

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

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(task, loc): loc for loc in locs}
            for fut in as_completed(futs):
                r = fut.result()
                results.append(r)
                completed += 1
                status = r.get("status", "unknown")
                by_status[status] = by_status.get(status, 0) + 1
                # Persist terminal-OK tiles (skipped + ok) to the index.
                # Tiles with status="failed" or "error" stay out so they
                # get retried on the next run.
                if (
                    skip_index is not None
                    and status in ("skipped", "ok")
                    and "name" in r
                ):
                    skip_index.add(r["name"])
                if completed % 25 == 0 or completed == len(locs):
                    elapsed = time.time() - t0
                    rate = completed / max(elapsed / 3600, 1e-6)
                    eta_min = (len(locs) - completed) / max(rate / 60, 1e-6)
                    total_completed = pre_credit + completed
                    print(
                        f"  [{total_completed}/{total}] "
                        f"status={by_status} "
                        f"rate={rate:.0f}/h ETA={eta_min:.1f}min",
                        flush=True,
                    )
    finally:
        if skip_index is not None:
            skip_index.flush_final()

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
