#!/usr/bin/env python3
"""Cross-raster aux-alignment QC gate.

Scans a tile directory and asserts that each stored tile's forestry aux
(height/volume/basal_area/diameter) is spatially aligned with the NMD ``label``
grid — the standing guard against the 256/512 aux-misalignment, where aux was
silently fetched on a 5 m/px central-quarter grid instead of the tile's 10 m/px
NMD lattice. See :mod:`imint.training.aux_alignment_qc`.

Emits a ``CAMPAIGN_GATE_VERDICT`` line for the campaign orchestrator and exits
non-zero when the failure rate among evaluable tiles exceeds ``--max-fail-frac``
(so it can sit before promote as a hard gate).

Usage:
    python scripts/qc_aux_alignment.py --data-dir /data/unified_v2_512_recoreg
    python scripts/qc_aux_alignment.py --data-dir <dir> --max-fail-frac 0.02 \\
        --report /data/debug/aux_alignment_qc.json
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

from imint.training.aux_alignment_qc import check_tile_alignment


def _check_one(path: str, *, min_phi: float) -> dict:
    try:
        with np.load(path, allow_pickle=True) as d:
            r = check_tile_alignment(d, min_phi=min_phi)
    except Exception as e:  # noqa: BLE001 — a corrupt .npz must not abort the gate
        return {"name": Path(path).stem, "status": "error",
                "reason": f"{type(e).__name__}: {e}"}
    r["name"] = Path(path).stem
    return r


def run(
    data_dir: str,
    *,
    workers: int = 8,
    min_phi: float = 0.15,
    max_fail_frac: float = 0.01,
    min_evaluated: int = 50,
    limit: int | None = None,
    report_path: str | None = None,
) -> dict:
    tiles = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if limit:
        tiles = tiles[:limit]
    print(f"=== aux-alignment QC ===  tiles={len(tiles)}  workers={workers}  "
          f"min_phi={min_phi}  max_fail_frac={max_fail_frac}", flush=True)

    stats = {"pass": 0, "fail": 0, "skipped": 0, "error": 0}
    failures: list[dict] = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_check_one, p, min_phi=min_phi): p for p in tiles}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            stats[r["status"]] = stats.get(r["status"], 0) + 1
            if r["status"] in ("fail", "error"):
                failures.append(r)
                if len(failures) <= 20:
                    print(f"  ✗ {r['name']}: {r.get('failed_aux') or r.get('reason')}"
                          f"  phi={r.get('phi')}", flush=True)
            if i % 500 == 0:
                print(f"  [{i}/{len(tiles)}] pass={stats['pass']} "
                      f"fail={stats['fail']} skip={stats['skipped']}", flush=True)

    evaluated = stats["pass"] + stats["fail"]
    fail_frac = stats["fail"] / evaluated if evaluated else 0.0

    # Gate logic: fail if too many evaluable tiles are misaligned. A run that
    # could evaluate almost nothing (e.g. labels not restored yet) is itself a
    # failure — the gate must not green-light on no evidence.
    if evaluated < min_evaluated:
        ok = False
        reason = (f"only {evaluated} evaluable tiles (<{min_evaluated}); "
                  f"cannot certify alignment — are labels present?")
    elif fail_frac > max_fail_frac:
        ok = False
        reason = (f"{stats['fail']}/{evaluated} tiles misaligned "
                  f"({fail_frac:.1%} > {max_fail_frac:.1%})")
    else:
        ok = True
        reason = (f"{stats['pass']}/{evaluated} tiles aligned "
                  f"({fail_frac:.1%} fail <= {max_fail_frac:.1%})")

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
                       "failures": failures[:200]}, f, indent=2)
        os.replace(tmp, report_path)
        print(f"  wrote {report_path}", flush=True)

    # Verdict line the campaign orchestrator parses (CLAUDE.md §8 protocol).
    verdict = {"gate": "aux-alignment", "ok": ok, "reason": reason}
    print(f"CAMPAIGN_GATE_VERDICT={json.dumps(verdict)}", flush=True)
    return {"ok": ok, **stats, "evaluated": evaluated}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True,
                    help="Directory globbed as <dir>/*.npz")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-phi", type=float, default=0.15,
                    help="Min phi(aux>0, forest) for a forestry aux to count as "
                         "aligned (default 0.15; aligned ~0.3-0.6, wrong-grid ~0)")
    ap.add_argument("--max-fail-frac", type=float, default=0.01,
                    help="Gate fails if more than this fraction of evaluable "
                         "tiles are misaligned (default 0.01)")
    ap.add_argument("--min-evaluated", type=int, default=50,
                    help="Gate fails if fewer than this many tiles are evaluable "
                         "(no evidence ⇒ no green light)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--report", default=None, help="Write a JSON report here")
    args = ap.parse_args()

    res = run(args.data_dir, workers=args.workers, min_phi=args.min_phi,
              max_fail_frac=args.max_fail_frac, min_evaluated=args.min_evaluated,
              limit=args.limit, report_path=args.report)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
