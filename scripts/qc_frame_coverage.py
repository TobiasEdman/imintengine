#!/usr/bin/env python3
"""Per-frame spectral coverage QC gate.

Scans a tile directory and asserts that every PRESENT temporal frame carries
real data — catching the Sentinel-2 swath-edge wedge and the empty-post-crop
frame (``temporal_mask=1`` but ~100% no-data) that the raw-halo all-zero guard
misses. Recomputed from the stored ``spectral`` cube, no re-fetch. See
:mod:`imint.training.frame_coverage_qc`.

Emits a ``CAMPAIGN_GATE_VERDICT`` line and exits non-zero when the failure rate
among evaluable tiles exceeds ``--max-fail-frac`` (a hard pre-promote gate). The
failing tiles + their bad slots are written to the JSON report as the refetch
list.

Usage:
    python scripts/qc_frame_coverage.py --data-dir /data/unified_v2_512_recoreg
    python scripts/qc_frame_coverage.py --data-dir <dir> --min-valid-frac 0.9 \\
        --report /data/debug/frame_coverage_qc.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imint.training.frame_coverage_qc import check_frame_coverage


def _check_one(path: str, *, min_valid_frac: float) -> dict:
    try:
        with np.load(path, allow_pickle=True) as d:
            r = check_frame_coverage(d, min_valid_frac=min_valid_frac)
    except Exception as e:  # noqa: BLE001 — a corrupt .npz must not abort the gate
        return {"name": Path(path).stem, "status": "error",
                "reason": f"{type(e).__name__}: {e}"}
    r["name"] = Path(path).stem
    return r


def run(
    data_dir: str,
    *,
    workers: int = 8,
    min_valid_frac: float = 0.90,
    max_fail_frac: float = 0.02,
    limit: int | None = None,
    report_path: str | None = None,
) -> dict:
    tiles = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if limit:
        tiles = tiles[:limit]
    print(f"=== frame-coverage QC ===  tiles={len(tiles)}  workers={workers}  "
          f"min_valid_frac={min_valid_frac}  max_fail_frac={max_fail_frac}",
          flush=True)

    stats = {"pass": 0, "fail": 0, "skipped": 0, "error": 0}
    failures: list[dict] = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_check_one, p, min_valid_frac=min_valid_frac): p
                for p in tiles}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            stats[r["status"]] = stats.get(r["status"], 0) + 1
            if r["status"] in ("fail", "error"):
                failures.append(r)
                if len(failures) <= 20:
                    print(f"  ✗ {r['name']}: {r.get('bad_frames') or r.get('reason')}",
                          flush=True)
            if i % 1000 == 0:
                print(f"  [{i}/{len(tiles)}] pass={stats['pass']} "
                      f"fail={stats['fail']} skip={stats['skipped']}", flush=True)

    evaluated = stats["pass"] + stats["fail"]
    fail_frac = stats["fail"] / evaluated if evaluated else 0.0
    ok = evaluated > 0 and fail_frac <= max_fail_frac
    if evaluated == 0:
        reason = "no evaluable tiles (no present frames?)"
    elif ok:
        reason = (f"{stats['pass']}/{evaluated} tiles fully-covered "
                  f"({fail_frac:.1%} fail <= {max_fail_frac:.1%})")
    else:
        reason = (f"{stats['fail']}/{evaluated} tiles have a no-data frame "
                  f"({fail_frac:.1%} > {max_fail_frac:.1%})")

    print(f"\n=== QC done in {(time.time()-t0)/60:.1f} min ===", flush=True)
    print(f"  pass={stats['pass']} fail={stats['fail']} "
          f"skipped={stats['skipped']} error={stats['error']} "
          f"(evaluated={evaluated})", flush=True)
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} — {reason}", flush=True)

    if report_path:
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        tmp = report_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"ok": ok, "reason": reason, "stats": stats,
                       "evaluated": evaluated, "fail_frac": fail_frac,
                       "refetch_list": failures[:500]}, f, indent=2)
        os.replace(tmp, report_path)
        print(f"  wrote {report_path}", flush=True)

    verdict = {"gate": "frame-coverage", "ok": ok, "reason": reason}
    print(f"CAMPAIGN_GATE_VERDICT={json.dumps(verdict)}", flush=True)
    return {"ok": ok, **stats, "evaluated": evaluated}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True,
                    help="Directory globbed as <dir>/*.npz")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-valid-frac", type=float, default=0.90,
                    help="A present frame below this non-no-data fraction fails "
                         "(default 0.90; the bulk is ~1.0)")
    ap.add_argument("--max-fail-frac", type=float, default=0.02,
                    help="Gate fails if more than this fraction of evaluable "
                         "tiles have a no-data frame (default 0.02)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--report", default=None, help="Write a JSON refetch list here")
    args = ap.parse_args()

    res = run(args.data_dir, workers=args.workers,
              min_valid_frac=args.min_valid_frac, max_fail_frac=args.max_fail_frac,
              limit=args.limit, report_path=args.report)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
