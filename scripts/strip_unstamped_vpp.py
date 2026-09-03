#!/usr/bin/env python3
"""Remove VPP bands from tiles that carry phenology of unverifiable year.

A tile written before ``ce2dfcd`` has the five ``vpp_*`` channels but no
``vpp_year`` stamp, so the year the phenology came from is unknowable from the
tile itself. Pre-fix code defaulted to ``year=2021`` (the bug fixed in
``72d4fd1``), so any such tile whose own year is not 2021 carries wrong-year
phenology. The 2026-08-26 audit measured the damage on ``unified_v2_512``::

    unstamped tile-year histogram: {2018: 41, 2021: 7, 2022: 27}

68 of 75 are wrong-year. They are invisible to every existing guard:
``_vpp_is_empty`` sees five populated channels and reports the tile as fine,
and ``backfill_vpp.py:241`` cannot re-fetch them because a failed fetch returns
before writing (``:264-266``), leaving the stale bands untouched.

Worse, they are excluded from the repair path. ``derive_missing_vpp_mgrs.py``
skips any tile where ``_vpp_is_empty`` is False, so these tiles never
contributed their (MGRS, year) pairs to the WEkEO gap-fill — which is why the
backfill kept reporting "no coverage" for them.

Stripping the bands makes them empty in the eyes of both tools, so the normal
repair chain can see them::

    strip_unstamped_vpp.py --apply      # bands removed, tiles become empty
    derive_missing_vpp_mgrs.py          # now includes them; emits their pairs
    prefetch_vpp_wekeo.py               # PU-free, fetch any missing COGs
    backfill_vpp.py                     # refills WITH a vpp_year stamp

Nothing is fabricated and nothing is zero-filled: the keys are deleted, so a
tile reads as honestly missing rather than confidently wrong. Writes are atomic
(temp + ``os.replace``), so an interrupted run cannot leave a half-written tile.

Dry-run by default; ``--apply`` is required to modify anything.
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from backfill_vpp import _atomic_savez, _tile_year  # noqa: E402
from imint.training.tile_fetch import list_tile_paths  # noqa: E402
from scripts.atomic_npz import (  # noqa: E402
    capture_npz,
    default_dataset_lock,
    exclusive_dataset_lock,
)

_VPP_KEYS = ("vpp_sosd", "vpp_eosd", "vpp_length", "vpp_maxv", "vpp_minv")


def scan(data_dir: str, *, keep_correct_2021: bool) -> tuple[list[str], dict]:
    """Return (paths to strip, report).

    A tile qualifies when every ``vpp_*`` band is present but ``vpp_year`` is
    absent. ``keep_correct_2021`` spares tiles whose own year is 2021, where the
    old default happened to be right — they keep correct values but stay
    unverifiable, so the default is to strip them too and let the refill stamp
    them.
    """
    targets: list[str] = []
    years: collections.Counter = collections.Counter()
    spared = 0
    scanned = 0

    for path in list_tile_paths(data_dir):
        scanned += 1
        try:
            z = np.load(path, allow_pickle=True)
        except Exception:
            continue
        try:
            keys = set(z.files)
            if not all(b in keys for b in _VPP_KEYS):
                continue          # no VPP at all — nothing to strip
            if "vpp_year" in keys:
                continue          # stamped: year is known, leave it alone
            meta = {k: z[k] for k in ("year", "lpis_year", "dates") if k in keys}
            ty = _tile_year(meta)
        finally:
            z.close()
        if keep_correct_2021 and ty == 2021:
            spared += 1
            continue
        years[ty] += 1
        targets.append(path)

    return targets, {
        "scanned": scanned,
        "to_strip": len(targets),
        "spared_correct_2021": spared,
        "year_histogram": dict(sorted(years.items(),
                                      key=lambda kv: (kv[0] is None, kv[0]))),
    }


def strip(
    path: str,
    *,
    keep_correct_2021: bool = False,
    dataset_lock: str | None = None,
    _lock_held: bool = False,
) -> bool:
    """Delete the five vpp_* keys from one tile, preserving everything else."""
    if not _lock_held:
        lock_path = (
            Path(dataset_lock)
            if dataset_lock is not None
            else default_dataset_lock(Path(path).parent)
        )
        with exclusive_dataset_lock(lock_path):
            return strip(
                path,
                keep_correct_2021=keep_correct_2021,
                dataset_lock=dataset_lock,
                _lock_held=True,
            )

    data, initial_identity = capture_npz(path)
    # Revalidate the scan predicate on the fd/hash-bound snapshot. An
    # uncoordinated replacement may have become stamped or otherwise
    # ineligible since the inventory pass; never strip that newer meaning.
    if "vpp_year" in data or not all(key in data for key in _VPP_KEYS):
        return False
    if keep_correct_2021 and _tile_year(data) == 2021:
        return False
    for k in _VPP_KEYS:
        del data[k]
    _atomic_savez(path, data, expected=initial_identity)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--apply", action="store_true",
                    help="Actually modify tiles (default: dry-run).")
    ap.add_argument("--keep-correct-2021", action="store_true",
                    help="Spare unstamped tiles whose own year is 2021, where "
                         "the old default was accidentally correct.")
    ap.add_argument(
        "--dataset-lock",
        default=None,
        help=(
            "Shared source-access lock (default: "
            "<data-dir-parent>/ops/crop-distill/source-access/locks/dataset.lock)"
        ),
    )
    args = ap.parse_args()

    lock_path = (
        Path(args.dataset_lock)
        if args.dataset_lock is not None
        else default_dataset_lock(args.data_dir)
    )
    lock_context = exclusive_dataset_lock(lock_path) if args.apply else nullcontext()
    with lock_context:
        # The apply inventory and every initial read/write remain inside the
        # same lock acquisition; a cooperating PLAN cannot start between scan
        # and replacement.
        targets, report = scan(
            args.data_dir,
            keep_correct_2021=args.keep_correct_2021,
        )
        print(f"scanned                : {report['scanned']:,}")
        print(f"unstamped w/ VPP bands : {report['to_strip']:,}")
        print(f"spared (correct 2021)  : {report['spared_correct_2021']:,}")
        print(f"tile-year histogram    : {report['year_histogram']}")
        for p in targets[:10]:
            print(f"    {os.path.basename(p)}")
        if len(targets) > 10:
            print(f"    ... and {len(targets) - 10} more")

        if not args.apply:
            print("\nDRY RUN — nothing modified. Re-run with --apply to strip.")
            return 0

        ok = fail = 0
        for p in targets:
            try:
                ok += bool(strip(
                    p,
                    keep_correct_2021=args.keep_correct_2021,
                    _lock_held=True,
                ))
            except Exception as e:  # noqa: BLE001 — isolate per-tile failures
                fail += 1
                print(f"  FAILED {os.path.basename(p)}: {type(e).__name__}: {e}")
    print(f"\nstripped={ok}  failed={fail}")
    print("Next: derive_missing_vpp_mgrs.py (they are now visible to it), "
          "then prefetch_vpp_wekeo.py, then backfill_vpp.py.")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
