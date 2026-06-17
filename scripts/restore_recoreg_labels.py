#!/usr/bin/env python3
"""scripts/restore_recoreg_labels.py — carry-forward the training label into the
re-coreg tiles.

``refetch_tile`` (scripts/fetch_unified_tiles.py) rebuilds every spectral-derived
field from the canonical M1+M2 entry and *drops* the three label fields listed in
``_REFETCH_DROP``::

    label, label_mask, label_year

so a ``unified_v2_512_recoreg`` tile lands with no training target. This script
restores them by COPYING those fields from the same-named ORIGINAL
``unified_v2_512`` tile (cross-dir).

This is valid because re-coreg keeps the canonical bbox/grid — M1 grid-snaps every
frame's transform onto the NMD 10 m lattice, M2 sub-pixel-shifts frames to the M1
anchor, M3 absolute is OFF — so the tile's bbox/centre is unchanged and the
original land-cover label still aligns pixel-for-pixel. The dashboard's
``campaign_dashboard.build_label`` already renders the original label over the
``_recoreg`` anchor and it lines up; this script makes that alignment a persisted
field rather than a render-time cross-dir read.

Idempotent + resumable:
  * skip a ``_recoreg`` tile that already carries ``label`` (re-runs are no-ops),
  * skip when the original lacks ``label`` (nothing to carry forward),
  * atomic write (tmp + ``os.replace``) preserving every other field.

Usage:
    python scripts/restore_recoreg_labels.py \\
        --recoreg-dir /data/unified_v2_512_recoreg \\
        --orig-dir    /data/unified_v2_512 \\
        --workers 4
    python scripts/restore_recoreg_labels.py --recoreg-dir <dir> --dry-run
"""
from __future__ import annotations

import argparse
import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

# The label fields refetch drops (mirrors fetch_unified_tiles._REFETCH_DROP). The
# canonical label is ``label``; ``label_mask`` (valid-pixel mask) and
# ``label_year`` (LPIS/SKS survey year) accompany it when present. Carried in this
# order so ``label`` gates the restore (a tile with a mask but no label is not a
# usable target and is treated as "original lacks label").
_LABEL_FIELDS = ("label", "label_mask", "label_year")


def restore_one(recoreg_path: str, orig_dir: str, *, dry_run: bool = False) -> dict:
    """Carry the label fields from the original tile into one ``_recoreg`` tile.

    Reads ``label``/``label_mask``/``label_year`` (whichever exist) from
    ``{orig_dir}/{name}.npz`` and writes them into ``recoreg_path`` atomically,
    preserving every other field. Returns a status dict — never raises on a
    per-tile error (the batch keeps going; the failure is surfaced in the summary).

    Status values:
      * ``restored``       — label fields copied in (or would be, under --dry-run).
      * ``skipped``        — the ``_recoreg`` tile already carries ``label``.
      * ``orig_missing``   — the original tile is absent or has no ``label``.
      * ``error``          — a tile-level read/write failure (``reason`` set).
    """
    name = os.path.basename(recoreg_path)[:-4]  # strip ".npz"

    # Idempotency gate 1: the _recoreg tile already has a label → no-op. Read only
    # the key listing (.files), not the arrays, so the common re-run case is cheap.
    try:
        with np.load(recoreg_path, allow_pickle=True) as rec:
            if "label" in rec.files:
                return {"name": name, "status": "skipped"}
    except Exception as exc:  # corrupt/unreadable _recoreg tile
        return {"name": name, "status": "error",
                "reason": f"recoreg_load: {type(exc).__name__}: {str(exc)[:140]}"}

    # Cross-dir read of the original label fields (same pattern as
    # campaign_dashboard.build_label): membership-check on .files, then load.
    orig_path = os.path.join(orig_dir, f"{name}.npz")
    if not os.path.exists(orig_path):
        return {"name": name, "status": "orig_missing", "reason": "no_orig_file"}
    try:
        with np.load(orig_path, allow_pickle=True) as orig:
            if "label" not in orig.files:
                return {"name": name, "status": "orig_missing",
                        "reason": "orig_no_label"}
            # Materialise only the label fields the original actually carries —
            # copy out of the NpzFile context so the arrays survive the close.
            carry = {k: np.array(orig[k]) for k in _LABEL_FIELDS if k in orig.files}
    except Exception as exc:
        return {"name": name, "status": "error",
                "reason": f"orig_load: {type(exc).__name__}: {str(exc)[:140]}"}

    if dry_run:
        return {"name": name, "status": "restored", "fields": sorted(carry),
                "dry_run": True}

    # Re-save the _recoreg tile with the label fields added, every other field
    # preserved verbatim. Atomic: write a sibling tmp then os.replace.
    try:
        with np.load(recoreg_path, allow_pickle=True) as rec:
            save = {k: np.array(rec[k]) for k in rec.files}
    except Exception as exc:
        return {"name": name, "status": "error",
                "reason": f"recoreg_reload: {type(exc).__name__}: {str(exc)[:140]}"}
    save.update(carry)

    # The tmp path MUST end in ``.npz`` — np.savez_compressed auto-appends ".npz"
    # to a path lacking it, producing ``FOO.npz.tmp.npz`` and breaking the
    # subsequent os.replace (documented quirk in fetch_unified_tiles).
    tmp_path = recoreg_path[:-4] + ".tmp.npz"  # FOO.npz → FOO.tmp.npz
    try:
        np.savez_compressed(tmp_path, **save)
        os.replace(tmp_path, recoreg_path)
    except Exception as exc:
        # Best-effort cleanup of a half-written tmp; never leave a dangling .tmp.
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        return {"name": name, "status": "error",
                "reason": f"write: {type(exc).__name__}: {str(exc)[:140]}"}

    return {"name": name, "status": "restored", "fields": sorted(carry)}


def restore_all(
    recoreg_dir: str, orig_dir: str, *, workers: int = 1, dry_run: bool = False,
) -> dict[str, int]:
    """Carry-forward labels for every ``*.npz`` in ``recoreg_dir`` in parallel.

    Returns the summary counts ``{restored, skipped, orig_missing, error}``.
    """
    paths = sorted(glob.glob(os.path.join(recoreg_dir, "*.npz")))
    total = len(paths)
    print(f"=== restore_recoreg_labels{'  [DRY-RUN]' if dry_run else ''} ===")
    print(f"  recoreg-dir: {recoreg_dir}")
    print(f"  orig-dir:    {orig_dir}")
    print(f"  tiles:       {total}  (workers={workers})\n")

    counts = {"restored": 0, "skipped": 0, "orig_missing": 0, "error": 0}
    if total == 0:
        print("  no _recoreg tiles found — nothing to do.")
        return counts

    def _run(path: str) -> dict:
        return restore_one(path, orig_dir, dry_run=dry_run)

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {pool.submit(_run, p): p for p in paths}
        for f in as_completed(futs):
            r = f.result()
            counts[r["status"]] = counts.get(r["status"], 0) + 1
            done += 1
            if r["status"] == "error":
                print(f"  [ERROR] {r['name']}: {r.get('reason', '')}", flush=True)
            if done % 500 == 0 or done == total:
                print(f"  [{done}/{total}] restored={counts['restored']} "
                      f"skipped={counts['skipped']} "
                      f"orig_missing={counts['orig_missing']} "
                      f"error={counts['error']}", flush=True)

    print(f"\n=== Summary{'  [DRY-RUN — no writes]' if dry_run else ''} ===")
    print(f"  restored:           {counts['restored']}")
    print(f"  skipped (has label):{counts['skipped']:>5}")
    print(f"  original-missing:   {counts['orig_missing']}")
    print(f"  errors:             {counts['error']}")
    return counts


def main() -> None:
    p = argparse.ArgumentParser(
        description="Carry-forward label/label_mask/label_year from the original "
                    "unified_v2_512 tiles into the same-named _recoreg tiles.")
    p.add_argument("--recoreg-dir", required=True,
                   help="Directory of _recoreg tiles to restore labels into "
                        "(e.g. /data/unified_v2_512_recoreg).")
    p.add_argument("--orig-dir", default="/data/unified_v2_512",
                   help="Directory of the original labelled tiles to copy from.")
    p.add_argument("--workers", type=int, default=1,
                   help="Concurrent tiles (label copy is IO-bound; default 1).")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be restored without writing any npz.")
    args = p.parse_args()

    restore_all(args.recoreg_dir, args.orig_dir,
                workers=args.workers, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
