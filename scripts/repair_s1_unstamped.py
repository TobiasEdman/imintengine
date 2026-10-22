#!/usr/bin/env python3
"""Re-enrich the S1 of tiles whose S1 the SAR gates reject (task 0114).

A read-only PVC count (2026-10-22) found tiles in ``/data/unified_v2_512``
that hold ``s1_vv_vh`` but fail the shared rule
(``imint.training.s1_enrichment.s1_usable``): 22 with no ``s1_enrich_v`` and
no ``s1_source`` (most likely v1 S1 from the retired GCP warp, task 0113)
and 1 at v3. Their other bands and labels are valid and in use, so the tiles
are not deleted: only their S1 is rebuilt from pc-rtc by
``scripts/enrich_tiles_s1.py:enrich_one_tile``, which rewrites the tile
atomically and keeps every other key.

Why not ``enrich_tiles_s1.py`` itself: its ``main()`` loads every tile in
full just to decide skips and deletes stale ``.tmp.npz`` files. This script
decides from the zip member list and the two scalar stamps only, touches
nothing but the selected tiles, and never deletes a file.

Selection
    Tiles (``*.npz``, temps excluded) that contain ``s1_vv_vh`` and fail
    ``s1_usable``. Tiles without S1 are out of scope and left alone.

Modes
    default / ``--dry-run``  print the selection as JSON; write nothing.
    ``--apply``              byte-copy each selected tile into
                             ``--backup-dir`` (refuses an existing name, never
                             overwrites), re-enrich each from pc-rtc, print a
                             line per tile, re-select and print the
                             after-count. Exit 1 if any enrich call failed or
                             any selected tile still fails the rule.
    ``--expect N``           refuse (exit 3) unless exactly N tiles are
                             selected; checked before anything is written.

Not to run concurrently with any other job writing the same directory.

Usage:
    python scripts/repair_s1_unstamped.py --data-dir /data/unified_v2_512
    python scripts/repair_s1_unstamped.py --data-dir /data/unified_v2_512 \\
        --apply --expect 23 --backup-dir /data/repair_backups/s1_0114_20261022
"""
from __future__ import annotations

import argparse
import filecmp
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imint.training.s1_enrichment import s1_unusable_reason
from imint.training.tile_fetch import list_tile_paths

S1_KEY = "s1_vv_vh"
STAMP_KEYS = ("s1_enrich_v", "s1_source")
BACKEND = "pc-rtc"

EXIT_FAILED = 1
EXIT_REFUSED = 3


def tile_s1_reason(path: str) -> tuple[bool, str | None]:
    """``(has_s1, reason)`` for one tile, from member names + stamps only.

    ``np.load`` on an ``.npz`` reads the zip directory lazily; only the two
    scalar stamp members are decompressed. ``reason`` is
    :func:`s1_unusable_reason` (``None`` = usable). Raises if unreadable.
    """
    with np.load(path, allow_pickle=True) as npz:
        members = set(npz.files)
        if S1_KEY not in members:
            return False, None
        stamps = {k: npz[k] for k in STAMP_KEYS if k in members}
    return True, s1_unusable_reason(stamps)


def scan(data_dir: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """``(selected, unreadable)`` over the tiles in ``data_dir``.

    ``selected``: ``(path, reason)`` for tiles with S1 that fail the rule.
    ``unreadable``: ``(path, error)`` for tiles that could not be opened;
    they are reported, never selected or touched.
    """
    selected: list[tuple[str, str]] = []
    unreadable: list[tuple[str, str]] = []
    for path in list_tile_paths(data_dir):
        try:
            has_s1, reason = tile_s1_reason(path)
        except Exception as exc:  # noqa: BLE001 — reported, not selected
            unreadable.append((path, f"{type(exc).__name__}: {exc}"))
            continue
        if has_s1 and reason is not None:
            selected.append((path, reason))
    return selected, unreadable


def select(data_dir: str) -> list[tuple[str, str]]:
    """Tiles that hold ``s1_vv_vh`` but fail ``s1_usable``: ``(path, reason)``."""
    return scan(data_dir)[0]


def selection_report(data_dir: str, scanned=None) -> dict:
    """JSON-ready selection: count, names + reasons, unreadable tiles."""
    selected, unreadable = scanned if scanned is not None else scan(data_dir)
    return {
        "data_dir": data_dir,
        "count": len(selected),
        "tiles": [
            {"name": os.path.basename(p), "reason": r} for p, r in selected
        ],
        "unreadable": [
            {"name": os.path.basename(p), "error": e} for p, e in unreadable
        ],
    }


def backup_conflicts(paths: list[str], backup_dir: str) -> list[str]:
    """Names already present in ``backup_dir`` (any one refuses the run)."""
    return sorted(
        os.path.basename(p) for p in paths
        if os.path.lexists(os.path.join(backup_dir, os.path.basename(p)))
    )


def backup_tile(path: str, backup_dir: str) -> str:
    """Byte-copy ``path`` into ``backup_dir``; never overwrite; verify bytes."""
    dst = os.path.join(backup_dir, os.path.basename(path))
    with open(path, "rb") as src, open(dst, "xb") as out:  # x: refuse existing
        shutil.copyfileobj(src, out, length=16 * 1024 * 1024)
    shutil.copystat(path, dst)
    if not filecmp.cmp(path, dst, shallow=False):
        raise OSError(f"backup of {path} differs from the source: {dst}")
    return dst


def _enrich(path: str) -> dict:
    # Imported here so a dry run needs none of the enricher's dependencies.
    from scripts.enrich_tiles_s1 import enrich_one_tile

    name = Path(path).stem
    try:
        return enrich_one_tile(path, skip_existing=True, backend=BACKEND)
    except Exception as exc:  # noqa: BLE001 — one tile must not stop the rest
        return {"name": name, "status": "failed",
                "reason": f"exception: {type(exc).__name__}: {exc}"}


def apply(data_dir: str, selected: list[tuple[str, str]], backup_dir: str,
          workers: int = 1) -> int:
    """Back up, re-enrich and re-check ``selected``; return the exit code."""
    paths = [p for p, _ in selected]
    os.makedirs(backup_dir, exist_ok=True)
    conflicts = backup_conflicts(paths, backup_dir)
    if conflicts:
        print(f"REFUSED: {len(conflicts)} name(s) already in {backup_dir}: "
              f"{conflicts}", flush=True)
        return EXIT_REFUSED

    for path in paths:
        dst = backup_tile(path, backup_dir)
        print(f"  backup {os.path.basename(path)} -> {dst}", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(_enrich, paths))

    failed = 0
    for (path, before), r in zip(selected, results):
        status = r.get("status", "failed")
        failed += status != "ok"
        line = {"name": os.path.basename(path), "before": before,
                "status": status}
        line.update({k: v for k, v in r.items() if k not in ("name", "status")})
        print("  result " + json.dumps(line, default=str), flush=True)

    still_bad = []
    for path in paths:
        try:
            has_s1, reason = tile_s1_reason(path)
        except Exception as exc:  # noqa: BLE001
            has_s1, reason = True, f"unreadable: {type(exc).__name__}: {exc}"
        if not has_s1:
            reason = f"{S1_KEY} missing"
        if reason is not None:
            still_bad.append({"name": os.path.basename(path), "reason": reason})

    after = selection_report(data_dir)
    summary = {"selected": len(paths), "enrich_failed": failed,
               "still_failing": still_bad, "after_count": after["count"],
               "after_tiles": after["tiles"]}
    print("AFTER " + json.dumps(summary), flush=True)
    return EXIT_FAILED if (failed or still_bad) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", dest="apply", action="store_false",
                      default=False,
                      help="Print the selection only (default).")
    mode.add_argument("--apply", dest="apply", action="store_true",
                      help="Back up and re-enrich the selected tiles.")
    parser.add_argument("--backup-dir",
                        help="Required with --apply; must not already hold a "
                             "selected tile's name.")
    parser.add_argument("--expect", type=int, default=None,
                        help="Refuse unless exactly N tiles are selected.")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.apply and not args.backup_dir:
        parser.error("--apply requires --backup-dir")
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    selected, unreadable = scan(args.data_dir)
    report = selection_report(args.data_dir, (selected, unreadable))
    print("SELECTION " + json.dumps(report), flush=True)
    if args.expect is not None and report["count"] != args.expect:
        print(f"REFUSED: selected {report['count']} tile(s), --expect "
              f"{args.expect}", flush=True)
        return EXIT_REFUSED
    if not args.apply:
        return 0

    return apply(args.data_dir, selected, args.backup_dir, args.workers)


if __name__ == "__main__":
    sys.exit(main())
