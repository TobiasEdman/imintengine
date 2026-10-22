#!/usr/bin/env python3
"""Re-enrich or strip the S1 of tiles whose S1 the SAR gates reject (0114).

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
    ``--strip``              for tiles RTC cannot re-enrich (owner decision
                             2026-10-22): remove every key in ``S1_KEYS``
                             and keep the rest. Backs up like ``--apply``,
                             but also reuses a backup of the same name that
                             is byte-identical to the current tile (and
                             refuses one that differs). Each tile is
                             rewritten member by member (the kept zip
                             members are copied, not re-serialized) to the
                             enricher's staging name, verified, and only
                             then renamed over the tile; it is verified
                             again against its backup afterwards. Exit 1 if
                             any tile failed or still holds an S1 key.
    ``--expect N``           refuse (exit 3) unless exactly N tiles are
                             selected; checked before anything is written.

``--apply`` and ``--strip`` also refuse (exit 3, nothing written, no backup
dir created) if a selected tile's staging file (``<tile>.npz.tmp.npz``, the
enricher's ``tile_staging_path``) already exists: the enricher would
overwrite it and then rename it away. The dry run lists such files as
``staging_conflicts``. ``--strip`` likewise refuses if a selected tile holds
an ``s1_*`` / ``has_s1*`` key outside ``S1_KEYS`` (dry run:
``unknown_s1_keys``): it would survive the strip with unknown meaning.

A stripped tile has no ``s1_vv_vh`` and is therefore no longer selected.

Not to run concurrently with any other job writing the same directory.

Usage:
    python scripts/repair_s1_unstamped.py --data-dir /data/unified_v2_512
    python scripts/repair_s1_unstamped.py --data-dir /data/unified_v2_512 \\
        --apply --expect 23 --backup-dir /data/repair_backups/s1_0114_20261022
    python scripts/repair_s1_unstamped.py --data-dir /data/unified_v2_512 \\
        --strip --expect 23 --backup-dir /data/repair_backups/s1_0114_20261022
"""
from __future__ import annotations

import argparse
import filecmp
import json
import os
import shutil
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imint.training.s1_enrichment import s1_unusable_reason
from imint.training.tile_fetch import list_tile_paths

S1_KEY = "s1_vv_vh"
STAMP_KEYS = ("s1_enrich_v", "s1_source")
BACKEND = "pc-rtc"

# Every tile key the S1 enricher owns (scripts/enrich_tiles_s1.py,
# enrich_one_tile): the eight it writes, plus s1_temporal_mask, the v1/v2
# leftover it drops on every write. --strip removes exactly these.
# tests/test_repair_s1_unstamped.py derives the set from the real producer.
S1_KEYS = (
    "s1_vv_vh", "s1_vv_vh_2016", "s1_dates", "s1_dates_2016", "s1_orbit",
    "s1_source", "has_s1", "s1_enrich_v", "s1_temporal_mask",
)
# Key prefixes treated as S1 when they are not in S1_KEYS: --strip refuses.
S1_PREFIXES = ("s1_", "has_s1")

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
    """JSON-ready selection: count, names + reasons, unreadable tiles and
    existing staging files of selected tiles (``--apply`` refuses those)."""
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
        "staging_conflicts": staging_conflicts([p for p, _ in selected]),
        "unknown_s1_keys": unknown_s1_keys([p for p, _ in selected]),
    }


def _key(member: str) -> str:
    """The ``np.load`` key of an ``.npz`` zip member (``.npy`` stripped)."""
    return member.removesuffix(".npy")


def unknown_s1_keys(paths: list[str]) -> dict[str, list[str]]:
    """``{tile name: keys}`` for S1-looking keys outside ``S1_KEYS``.

    From the zip member list only. Any entry refuses ``--strip``: such a key
    would survive the strip and nobody has said what it means.
    """
    found = {}
    for path in paths:
        with zipfile.ZipFile(path) as zf:
            keys = sorted(
                k for k in map(_key, zf.namelist())
                if k.startswith(S1_PREFIXES) and k not in S1_KEYS
            )
        if keys:
            found[os.path.basename(path)] = keys
    return found


def staging_conflicts(paths: list[str]) -> list[str]:
    """Existing staging files of ``paths`` (any one refuses the run).

    ``enrich_one_tile`` writes ``tile_staging_path(tile)`` and then renames
    it over the tile (or unlinks it on failure), so a file already at that
    name — e.g. left by an interrupted run — would be overwritten and
    removed. The name comes from the enricher's own rule.
    """
    from scripts.enrich_tiles_s1 import tile_staging_path

    return sorted(
        os.path.basename(tile_staging_path(p)) for p in paths
        if os.path.lexists(tile_staging_path(p))
    )


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


def backup_plan(paths: list[str], backup_dir: str
                ) -> tuple[list[str], list[str], list[str]]:
    """``(to_copy, reused, refused)`` tile paths for ``--strip``.

    A name absent from ``backup_dir`` is copied. A regular file of the same
    name that is byte-identical to the current tile is reused (e.g. the
    ``--apply`` backups of a failed re-enrich). Anything else at that name —
    different bytes, a link, a directory — refuses the run.
    """
    to_copy, reused, refused = [], [], []
    for path in paths:
        dst = os.path.join(backup_dir, os.path.basename(path))
        if not os.path.lexists(dst):
            to_copy.append(path)
        elif (os.path.isfile(dst) and not os.path.islink(dst)
              and filecmp.cmp(path, dst, shallow=False)):
            reused.append(path)
        else:
            refused.append(path)
    return to_copy, reused, refused


def verify_stripped(original: str, stripped: str) -> list[str]:
    """Check ``stripped`` is ``original`` minus ``S1_KEYS``; return removed keys.

    Raises ``ValueError`` unless the member list is exactly the original's
    minus the S1 members, no S1 key remains, every kept member's (uncompressed,
    CRC-checked) bytes equal the original's, and ``np.load`` reads every key.
    """
    with zipfile.ZipFile(original) as zo, zipfile.ZipFile(stripped) as zs:
        names = zo.namelist()
        removed = sorted(_key(n) for n in names if _key(n) in S1_KEYS)
        kept = [n for n in names if _key(n) not in S1_KEYS]
        if zs.namelist() != kept:
            raise ValueError(f"members {zs.namelist()} != expected {kept}")
        for name in kept:
            if zs.read(name) != zo.read(name):
                raise ValueError(f"member {name} differs from the original")
    with np.load(stripped, allow_pickle=True) as npz:
        left = sorted(set(npz.files) & set(S1_KEYS))
        if left:
            raise ValueError(f"S1 keys remain: {left}")
        for key in npz.files:
            npz[key]
    return removed


def strip_tile(path: str, backup: str) -> list[str]:
    """Rewrite ``path`` without ``S1_KEYS``; return the removed keys.

    The kept zip members are copied byte for byte (same name, compression and
    timestamp) to the enricher's staging name, created exclusively. The stage
    is verified against the tile and only then renamed over it; the result is
    verified again against ``backup`` (byte-identical to the original). On any
    failure before the rename the stage is removed and the tile is untouched.
    """
    from scripts.enrich_tiles_s1 import tile_staging_path

    staging = tile_staging_path(path)
    # Opened before the cleanup scope: an existing stage raises here and is
    # never removed — only a stage this call created is. Closed by the with.
    fh = open(staging, "xb")  # noqa: SIM115
    try:
        with fh, zipfile.ZipFile(fh, "w", allowZip64=True) as zout, \
                zipfile.ZipFile(path) as zin:
            for info in zin.infolist():
                if _key(info.filename) in S1_KEYS:
                    continue
                member = zipfile.ZipInfo(info.filename, info.date_time)
                member.compress_type = info.compress_type
                member.external_attr = info.external_attr
                zout.writestr(member, zin.read(info))
        verify_stripped(path, staging)
        os.replace(staging, path)
    except BaseException:
        try:
            os.unlink(staging)
        except FileNotFoundError:
            pass
        raise
    return verify_stripped(backup, path)


def strip(data_dir: str, selected: list[tuple[str, str]],
          backup_dir: str) -> int:
    """Back up (or reuse identical backups), strip S1, re-check; exit code."""
    paths = [p for p, _ in selected]
    # Every refusal is decided before the backup dir is created.
    staged = staging_conflicts(paths)
    if staged:
        print(f"REFUSED: {len(staged)} staging file(s) of selected tiles "
              f"already exist (would be overwritten and removed): {staged}",
              flush=True)
        return EXIT_REFUSED
    unknown = unknown_s1_keys(paths)
    if unknown:
        print(f"REFUSED: S1-looking keys outside S1_KEYS would survive the "
              f"strip: {unknown}", flush=True)
        return EXIT_REFUSED
    to_copy, reused, refused = backup_plan(paths, backup_dir)
    if refused:
        names = sorted(os.path.basename(p) for p in refused)
        print(f"REFUSED: {len(names)} name(s) in {backup_dir} are not "
              f"byte-identical to the current tile: {names}", flush=True)
        return EXIT_REFUSED

    os.makedirs(backup_dir, exist_ok=True)
    for path in to_copy:
        dst = backup_tile(path, backup_dir)
        print(f"  backup {os.path.basename(path)} -> {dst}", flush=True)
    for path in reused:
        print(f"  backup {os.path.basename(path)} = existing identical "
              f"{os.path.join(backup_dir, os.path.basename(path))}",
              flush=True)

    failed = 0
    for i, (path, before) in enumerate(selected, 1):
        name = os.path.basename(path)
        line = {"i": f"{i}/{len(paths)}", "name": name, "before": before}
        try:
            removed = strip_tile(path, os.path.join(backup_dir, name))
            line.update(status="ok", removed=removed)
        except Exception as exc:  # noqa: BLE001 — one tile must not stop the rest
            failed += 1
            line.update(status="failed",
                        reason=f"exception: {type(exc).__name__}: {exc}")
        print("  result " + json.dumps(line), flush=True)

    still_s1 = []
    for path in paths:
        try:
            with zipfile.ZipFile(path) as zf:
                left = sorted(set(map(_key, zf.namelist())) & set(S1_KEYS))
        except Exception as exc:  # noqa: BLE001
            left = [f"unreadable: {type(exc).__name__}: {exc}"]
        if left:
            still_s1.append({"name": os.path.basename(path), "keys": left})

    after = selection_report(data_dir)
    summary = {"selected": len(paths), "backed_up": len(to_copy),
               "backup_reused": len(reused), "strip_failed": failed,
               "still_holding_s1": still_s1, "after_count": after["count"],
               "after_tiles": after["tiles"]}
    print("AFTER " + json.dumps(summary), flush=True)
    return EXIT_FAILED if (failed or still_s1) else 0


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
    # Checked before the backup dir is created: a refusal touches nothing.
    staged = staging_conflicts(paths)
    if staged:
        print(f"REFUSED: {len(staged)} staging file(s) of selected tiles "
              f"already exist (would be overwritten and removed): {staged}",
              flush=True)
        return EXIT_REFUSED
    os.makedirs(backup_dir, exist_ok=True)
    conflicts = backup_conflicts(paths, backup_dir)
    if conflicts:
        print(f"REFUSED: {len(conflicts)} name(s) already in {backup_dir}: "
              f"{conflicts}", flush=True)
        return EXIT_REFUSED

    for path in paths:
        dst = backup_tile(path, backup_dir)
        print(f"  backup {os.path.basename(path)} -> {dst}", flush=True)

    # pool.map yields in order as tiles finish, so each line prints as soon
    # as its tile (and every earlier one) is done.
    failed = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = pool.map(_enrich, paths)
        for i, ((path, before), r) in enumerate(zip(selected, results), 1):
            status = r.get("status", "failed")
            failed += status != "ok"
            line = {"i": f"{i}/{len(paths)}", "name": os.path.basename(path),
                    "before": before, "status": status}
            line.update({k: v for k, v in r.items()
                         if k not in ("name", "status")})
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
    mode.add_argument("--dry-run", dest="mode", action="store_const",
                      const="dry-run", default="dry-run",
                      help="Print the selection only (default).")
    mode.add_argument("--apply", dest="mode", action="store_const",
                      const="apply",
                      help="Back up and re-enrich the selected tiles.")
    mode.add_argument("--strip", dest="mode", action="store_const",
                      const="strip",
                      help="Back up and remove S1_KEYS from the selected "
                           "tiles (for tiles RTC cannot re-enrich).")
    parser.add_argument("--backup-dir",
                        help="Required with --apply / --strip. --apply: must "
                             "not already hold a selected tile's name. "
                             "--strip: may hold it only byte-identical.")
    parser.add_argument("--expect", type=int, default=None,
                        help="Refuse unless exactly N tiles are selected.")
    parser.add_argument("--workers", type=int, default=1,
                        help="--apply only.")
    args = parser.parse_args(argv)
    if args.mode != "dry-run" and not args.backup_dir:
        parser.error(f"--{args.mode} requires --backup-dir")
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    selected, unreadable = scan(args.data_dir)
    report = selection_report(args.data_dir, (selected, unreadable))
    print("SELECTION " + json.dumps(report), flush=True)
    if args.expect is not None and report["count"] != args.expect:
        print(f"REFUSED: selected {report['count']} tile(s), --expect "
              f"{args.expect}", flush=True)
        return EXIT_REFUSED
    if args.mode == "dry-run":
        return 0
    if args.mode == "strip":
        return strip(args.data_dir, selected, args.backup_dir)
    return apply(args.data_dir, selected, args.backup_dir, args.workers)


if __name__ == "__main__":
    sys.exit(main())
