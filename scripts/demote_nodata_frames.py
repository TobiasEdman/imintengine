#!/usr/bin/env python3
"""scripts/demote_nodata_frames.py — clear no-data spectral frames out of
``temporal_mask`` for tiles that retain enough good frames (decision-2).

The frame-coverage gate (:mod:`imint.training.frame_coverage_qc`) fails a tile
when any PRESENT frame (``temporal_mask=1``) is a no-data wedge — a swath edge
or an empty-post-crop frame with ``valid_frac < min_valid_frac``. The
semantically-correct fix is to clear that frame from ``temporal_mask``: it was
never real data, so it should never have been marked present. This is *exactly*
what ``fetch_spectral`` does at fetch time for sub-threshold frames — both paths
use the same :func:`frame_valid_fraction` primitive, so the gate and the demote
can never disagree on "what counts as covered". No re-fetch, no spectral
mutation, zero quality loss.

Only tiles that RETAIN ``>= --min-good-frames`` present frames AFTER the bad
ones are cleared are touched. A tile that would drop below that floor is left
untouched and reported as ``residual`` — demoting it would leave too few frames
for multitemporal training, so it is a separate accept/drop call (no re-fetch).

Idempotent + resumable + atomic (tmp + ``os.replace``), modelled on
``restore_recoreg_labels.py``. Re-running is a no-op: an already-demoted frame
is no longer present, so the tile classifies ``clean``.

Usage:
    python scripts/demote_nodata_frames.py --data-dir /data/unified_v2_512_recoreg --dry-run
    python scripts/demote_nodata_frames.py --data-dir <dir> --workers 4 \\
        --report /data/debug/demote_nodata_frames.json
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

from imint.training.frame_coverage_qc import frame_valid_fractions


def plan_demote(npz, *, min_valid_frac: float, min_good_frames: int) -> dict:
    """Classify one tile without writing anything.

    Returns ``{action, bad_slots, present_slots, n_present, n_good}``:
      * ``demote``   — has bad present frame(s) AND ``n_good >= min_good_frames``.
      * ``residual`` — has bad present frame(s) but ``n_good < min_good_frames``.
      * ``clean``    — no bad present frame (nothing to do / already demoted).
      * ``skipped``  — no ``spectral`` cube / no present frames.

    ``n_good`` is the count of present frames that survive (``n_present`` minus
    the bad ones). Uses :func:`frame_valid_fractions` so this classification is
    identical to the gate's.
    """
    fracs = frame_valid_fractions(npz)
    if not fracs:
        return {"action": "skipped", "bad_slots": [], "present_slots": [],
                "n_present": 0, "n_good": 0}
    present_slots = [int(f) for f, _ in fracs]
    bad_slots = [int(f) for f, v in fracs if v < min_valid_frac]
    n_present = len(present_slots)
    n_good = n_present - len(bad_slots)
    if not bad_slots:
        action = "clean"
    elif n_good >= min_good_frames:
        action = "demote"
    else:
        action = "residual"
    return {"action": action, "bad_slots": bad_slots, "present_slots": present_slots,
            "n_present": n_present, "n_good": n_good}


def _demoted_mask(old_mask: np.ndarray | None, bad_slots: list[int],
                  present_slots: list[int]) -> np.ndarray:
    """Build the new ``temporal_mask``: a copy of ``old_mask`` (or all-ones when
    the tile carries none) with ``bad_slots`` zeroed.

    Padded to cover the highest present slot, so a bad frame past the stored
    mask length (a tile saved with a short mask + present-by-default tail) is
    still cleared rather than raising ``IndexError``.
    """
    need_len = (max(present_slots) + 1) if present_slots else 0
    if old_mask is None:
        mask = np.ones(need_len, np.uint8)
    else:
        mask = np.asarray(old_mask).astype(np.uint8).copy()
        if len(mask) < need_len:
            mask = np.concatenate([mask, np.ones(need_len - len(mask), np.uint8)])
    for s in bad_slots:
        mask[s] = 0
    return mask


def demote_one(path: str, *, min_valid_frac: float, min_good_frames: int,
               dry_run: bool = False) -> dict:
    """Demote the no-data frames of one tile, atomically, preserving every other
    field. Never raises on a per-tile error — the batch keeps going.

    Status mirrors :func:`plan_demote`'s ``action`` (``demote`` / ``residual`` /
    ``clean`` / ``skipped``) plus ``error``.
    """
    name = os.path.basename(path)[:-4]  # strip ".npz"
    try:
        with np.load(path, allow_pickle=True) as d:
            keys = d.files
            plan = plan_demote(d, min_valid_frac=min_valid_frac,
                               min_good_frames=min_good_frames)
            old_mask = (np.asarray(d["temporal_mask"]).astype(np.uint8).copy()
                        if "temporal_mask" in keys else None)
    except Exception as exc:  # noqa: BLE001 — a corrupt .npz must not abort the batch
        return {"name": name, "status": "error",
                "reason": f"load: {type(exc).__name__}: {str(exc)[:140]}"}

    res = {"name": name, "status": plan["action"], "n_present": plan["n_present"],
           "n_good": plan["n_good"], "bad_slots": plan["bad_slots"]}
    if plan["action"] != "demote" or dry_run:
        if dry_run:
            res["dry_run"] = True
        return res

    new_mask = _demoted_mask(old_mask, plan["bad_slots"], plan["present_slots"])

    # Reload + materialise every field, swap in the new mask, atomic write. The
    # tmp path MUST end in ".npz" (np.savez_compressed auto-appends it, which
    # would otherwise break the os.replace) — documented in restore_recoreg_labels.
    try:
        with np.load(path, allow_pickle=True) as d:
            save = {k: np.array(d[k]) for k in d.files}
    except Exception as exc:
        return {"name": name, "status": "error",
                "reason": f"reload: {type(exc).__name__}: {str(exc)[:140]}"}
    save["temporal_mask"] = new_mask

    tmp_path = path[:-4] + ".tmp.npz"
    try:
        np.savez_compressed(tmp_path, **save)
        os.replace(tmp_path, path)
    except Exception as exc:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        return {"name": name, "status": "error",
                "reason": f"write: {type(exc).__name__}: {str(exc)[:140]}"}
    return res


def run(data_dir: str, *, workers: int = 4, min_valid_frac: float = 0.90,
        min_good_frames: int = 3, dry_run: bool = False,
        report_path: str | None = None) -> dict:
    paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    total = len(paths)
    print(f"=== demote_nodata_frames{'  [DRY-RUN]' if dry_run else ''} ===")
    print(f"  data-dir:        {data_dir}")
    print(f"  tiles:           {total}  (workers={workers})")
    print(f"  min_valid_frac:  {min_valid_frac}   min_good_frames: {min_good_frames}\n",
          flush=True)

    counts = {"demote": 0, "residual": 0, "clean": 0, "skipped": 0, "error": 0}
    demoted: list[dict] = []
    residual: list[dict] = []
    if total == 0:
        print("  no tiles found — nothing to do.")
        return {"ok": True, **counts}

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {pool.submit(demote_one, p, min_valid_frac=min_valid_frac,
                            min_good_frames=min_good_frames, dry_run=dry_run): p
                for p in paths}
        for fut in as_completed(futs):
            r = fut.result()
            counts[r["status"]] = counts.get(r["status"], 0) + 1
            if r["status"] == "demote":
                demoted.append(r)
            elif r["status"] == "residual":
                residual.append(r)
            elif r["status"] == "error":
                print(f"  [ERROR] {r['name']}: {r.get('reason', '')}", flush=True)
            done += 1
            if done % 1000 == 0 or done == total:
                print(f"  [{done}/{total}] demote={counts['demote']} "
                      f"residual={counts['residual']} clean={counts['clean']} "
                      f"skip={counts['skipped']} err={counts['error']}", flush=True)

    print(f"\n=== Summary{'  [DRY-RUN — no writes]' if dry_run else ''} "
          f"({(time.time()-t0)/60:.1f} min) ===")
    print(f"  demoted (temporal_mask cleared): {counts['demote']}")
    print(f"  residual (<{min_good_frames} good, untouched): {counts['residual']}")
    print(f"  clean (no bad present frame):    {counts['clean']}")
    print(f"  skipped (no frames):             {counts['skipped']}")
    print(f"  errors:                          {counts['error']}", flush=True)

    ok = counts["error"] == 0
    if report_path:
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        tmp = report_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"ok": ok, "dry_run": dry_run, "counts": counts,
                       "min_valid_frac": min_valid_frac,
                       "min_good_frames": min_good_frames,
                       "demoted": sorted(demoted, key=lambda r: r["name"]),
                       "residual": sorted(residual, key=lambda r: r["name"])},
                      f, indent=2)
        os.replace(tmp, report_path)
        print(f"  wrote {report_path}", flush=True)

    return {"ok": ok, **counts}


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True,
                   help="Directory globbed as <dir>/*.npz.")
    p.add_argument("--workers", type=int, default=4,
                   help="Concurrent tiles (IO-bound; default 4).")
    p.add_argument("--min-valid-frac", type=float, default=0.90,
                   help="A present frame below this non-no-data fraction is a "
                        "no-data wedge to demote (matches the gate; default 0.90).")
    p.add_argument("--min-good-frames", type=int, default=3,
                   help="Only demote a tile that retains at least this many good "
                        "frames afterwards (default 3); fewer → residual, untouched.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report the demote/residual partition without writing npz.")
    p.add_argument("--report", default=None,
                   help="Write a JSON report (demoted + residual lists) here.")
    args = p.parse_args()

    res = run(args.data_dir, workers=args.workers, min_valid_frac=args.min_valid_frac,
              min_good_frames=args.min_good_frames, dry_run=args.dry_run,
              report_path=args.report)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
