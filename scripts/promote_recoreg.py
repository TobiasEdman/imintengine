#!/usr/bin/env python3
"""Promote the re-coreg campaign output dir to the live dataset.

Phases 1+2 write a NEW directory (``unified_v2_512_recoreg``) so the live
``unified_v2_512`` stays intact until an explicit, verified promote. This script
performs that swap via two directory renames on the same filesystem::

    unified_v2_512          -> unified_v2_512_pre_recoreg_<UTC-ts>   (backup)
    unified_v2_512_recoreg  -> unified_v2_512                        (promote)

Rename-with-backup is chosen over per-file copy (slower; needs stale-tile cleanup
to stay consistent) and symlink-swap (changes consumer path assumptions): renames
are metadata-only + atomic per call, rollback is a reverse rename, and the
timestamped backup is a safety net. Paths must be distinct, non-nested directories
on one filesystem. A failed second rename triggers an attempt to restore live;
if recovery fails, the error identifies the retained data and recovery paths.
The pair is NOT crash-atomic: live is briefly absent between renames, and process
termination or a machine crash can leave it absent. Operators must ensure
exclusive access to these paths during the operation (including recovery).

DRY-RUN by default: prints the plan and exits 0 without touching disk. Pass
``--execute`` to act. ``--rollback`` restores the most recent
``*_pre_recoreg_*`` backup over the (current) live dir.

Usage:
  # inspect the promote plan (no changes)
  python scripts/promote_recoreg.py \\
      --live /data/unified_v2_512 --recoreg /data/unified_v2_512_recoreg
  # perform the promote
  python scripts/promote_recoreg.py \\
      --live /data/unified_v2_512 --recoreg /data/unified_v2_512_recoreg --execute
  # undo (restore newest backup over live)
  python scripts/promote_recoreg.py --live /data/unified_v2_512 --rollback --execute
"""
from __future__ import annotations

import argparse
import errno
import glob
import os
import sys
from datetime import datetime, timezone

_BACKUP_SUFFIX = "_pre_recoreg_"


def _count_npz(d: str) -> int:
    return len(glob.glob(os.path.join(d, "*.npz")))


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _newest_backup(live: str) -> str | None:
    """Most recent ``{live}{_BACKUP_SUFFIX}*`` sibling, or None."""
    cands = sorted(glob.glob(live + _BACKUP_SUFFIX + "*"))
    return cands[-1] if cands else None


def _rename(src: str, dst: str, *, execute: bool) -> None:
    """os.rename with a same-filesystem guard and a dry-run echo."""
    print(f"    {'RENAME' if execute else 'would rename'}: {src} -> {dst}")
    if not execute:
        return
    try:
        os.rename(src, dst)
    except OSError as e:
        if getattr(e, "errno", None) == errno.EXDEV:  # cross-device link
            raise SystemExit(
                f"ERROR: {src} and {dst} are on different filesystems "
                f"(os.rename EXDEV). Promote requires one filesystem."
            ) from e
        raise


def _validate_swap(live: str, candidate: str, saved: str) -> None:
    """Reject unsafe paths before moving anything, including in dry-run mode."""
    for path in (live, candidate):
        if os.path.islink(path):
            sys.exit(f"ERROR: dataset path must not be a symlink: {path}")
        if os.path.lexists(path) and not os.path.isdir(path):
            sys.exit(f"ERROR: dataset path is not a directory: {path}")
    if not os.path.isdir(candidate):
        sys.exit(f"ERROR: candidate directory not found: {candidate}")
    if os.path.lexists(saved):
        sys.exit(f"ERROR: backup/aside path already exists: {saved}")

    paths = [os.path.realpath(p) for p in (live, candidate, saved)]
    for i, path in enumerate(paths):
        for other in paths[i + 1:]:
            if os.path.commonpath((path, other)) in (path, other):
                sys.exit(f"ERROR: paths must be distinct and non-nested: "
                         f"{path}, {other}")
    # Check both directories and parents: a mounted dataset directory may have
    # a different device from its parent, making the first rename impossible.
    devices = set()
    for path in (live, candidate, saved):
        parent = os.path.dirname(path)
        if not os.path.isdir(parent):
            sys.exit(f"ERROR: parent directory not found: {parent}")
        devices.add(os.stat(parent).st_dev)
        if os.path.isdir(path):
            devices.add(os.stat(path).st_dev)
    if len(devices) != 1:
        sys.exit("ERROR: swap paths are on different filesystems; "
                 "all directories and parents must share one filesystem.")


def _swap(live: str, candidate: str, saved: str, *, execute: bool) -> None:
    """Compensate a failed second rename; this is not a crash-atomic swap."""
    had_live = os.path.isdir(live)
    if had_live:
        _rename(live, saved, execute=execute)
    try:
        _rename(candidate, live, execute=execute)
    except (OSError, SystemExit) as original:
        if execute and had_live:
            try:
                # Never replace a path that appeared after the first rename.
                if os.path.lexists(live):
                    raise FileExistsError(f"live path reappeared: {live}")
                _rename(saved, live, execute=True)
            except (OSError, SystemExit) as recovery:
                raise RuntimeError(
                    f"Swap failed ({candidate} -> {live}): {original}; "
                    f"restoration failed ({saved} -> {live}): {recovery}. "
                    f"Manual recovery required; inspect {saved}, {candidate} "
                    f"and {live} before moving anything."
                ) from original
        raise


def _promote(live: str, recoreg: str, *, execute: bool, min_frac: float) -> int:
    live, recoreg = os.path.abspath(live), os.path.abspath(recoreg)
    backup = live + _BACKUP_SUFFIX + _utc_stamp()
    _validate_swap(live, recoreg, backup)
    n_recoreg = _count_npz(recoreg)
    if n_recoreg == 0:
        sys.exit(f"ERROR: recoreg dir has no .npz tiles: {recoreg}")
    n_live = _count_npz(live) if os.path.isdir(live) else 0

    print("=== promote plan ===")
    print(f"  live:    {live}  ({n_live} tiles)")
    print(f"  recoreg: {recoreg}  ({n_recoreg} tiles)")
    if n_live:
        frac = n_recoreg / n_live
        print(f"  recoreg/live ratio: {frac:.3f}  (delta {n_recoreg - n_live:+d})")
        # A recoreg far smaller than live signals Phase-1/2 dropped too many
        # tiles — block unless the operator overrides. recoreg can legitimately
        # be slightly larger (orphan additions) or smaller (frame-count drops).
        if execute and frac < min_frac:
            sys.exit(f"ERROR: recoreg has only {frac:.1%} of live's tiles "
                     f"(< --min-frac {min_frac:.0%}). Re-check the campaign or "
                     f"pass --min-frac to override.")

    print("  steps:")
    if not os.path.isdir(live):
        print("    (live dir absent — no backup; fresh promote)")
    _swap(live, recoreg, backup, execute=execute)

    if execute:
        print(f"=== promoted: {recoreg} is now {live} ===")
        if os.path.isdir(backup):
            print(f"    backup retained at {backup} (delete once verified)")
    else:
        print("=== DRY-RUN — nothing changed. Re-run with --execute. ===")
    return 0


def _rollback(live: str, *, execute: bool) -> int:
    live = os.path.abspath(live)
    backup = _newest_backup(live)
    if backup is None:
        sys.exit(f"ERROR: no {live}{_BACKUP_SUFFIX}* backup to roll back to.")
    aside = live + "_recoreg_rolledback_" + _utc_stamp()
    _validate_swap(live, backup, aside)
    print("=== rollback plan ===")
    print(f"  newest backup: {backup}  ({_count_npz(backup)} tiles)")
    print("  steps:")
    # Preserve the current (promoted) live by moving it back to a recoreg-style
    # name rather than deleting it — rollback must never destroy data.
    _swap(live, backup, aside, execute=execute)
    if execute:
        print(f"=== rolled back: {backup} is now {live} ===")
    else:
        print("=== DRY-RUN — nothing changed. Re-run with --execute. ===")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--live", default="/data/unified_v2_512",
                   help="The live dataset dir (promote target).")
    p.add_argument("--recoreg", default="/data/unified_v2_512_recoreg",
                   help="The campaign output dir to promote.")
    p.add_argument("--execute", action="store_true",
                   help="Actually perform the renames (default: dry-run).")
    p.add_argument("--rollback", action="store_true",
                   help="Restore the most recent *_pre_recoreg_* backup over live.")
    p.add_argument("--min-frac", type=float, default=0.9,
                   help="Refuse to promote if recoreg has fewer than this "
                        "fraction of live's tiles (safety rail; default 0.9).")
    args = p.parse_args()

    if args.rollback:
        sys.exit(_rollback(args.live, execute=args.execute))
    sys.exit(_promote(args.live, args.recoreg,
                      execute=args.execute, min_frac=args.min_frac))


if __name__ == "__main__":
    main()
