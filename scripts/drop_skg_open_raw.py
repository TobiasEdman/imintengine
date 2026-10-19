#!/usr/bin/env python3
"""Delete the grunddata and markfukt raw zips of the open-data mirror, keeping every derived raster.

Tobias decided on 2026-10-19 ("ja, radera råzip nu, behåll rastren") to
free the ~97 GiB of raw zips under ``<dest>/raw`` that
``scripts/stage_skg_open.py`` keeps for grunddata and markfukt. The derived
COGs and VRTs stay: the readers in ``imint.training.skg_open`` use only
those. Trädhöjd is not this script's business; its raw zips are handled
by the staging script itself.

For each manifest file entry of product ``grunddata`` or ``markfukt``
whose raw zip is still on disk, the zip is deleted only when:

* the entry records the zip's ``sha256``, ``bytes``, ``url`` and
  ``downloaded_at``;
* the zip on disk has the recorded size;
* exactly one derived raster records it as its source (``from``), that
  raster exists and its sha256 equals the recorded one.

Then ``raw_deleted_at`` is set, the marker ``stage_skg_open.is_done``
already honours, so a later staging run skips the item instead of
downloading it again. Only files directly under ``<dest>/raw`` are ever
deleted; derived rasters, VRTs and Trädhöjd files are never touched.

A run that dies between deleting a zip and saving its marker leaves the
zip gone without ``raw_deleted_at``. The next run does not skip such an
entry: if its one derived raster still matches its recorded sha256 it
records ``raw_deleted_at`` with ``raw_absent_at_drop: true``, otherwise it
reports it and exits 2.

The manifest is rewritten whole, so a concurrent writer would erase the
other's records. A real run takes ``<dest>/MANIFEST.lock``, which
``stage_skg_open`` also takes, and refuses to start while it exists. It
also refuses to save if the file on disk changed under it.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.stage_skg_open import (  # noqa: E402
    _git_sha, _now, load_manifest, manifest_lock, save_manifest, sha256_of)

PRODUCTS = ("grunddata", "markfukt")
REQUIRED = ("sha256", "bytes", "url", "downloaded_at")


class ConcurrentWriter(RuntimeError):
    """MANIFEST.json changed on disk while this script held it."""


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan(manifest: dict, dest: Path) -> tuple[list[tuple[str, str]], list[str], list[tuple[str, str]]]:
    """``(deletable, refusals, unmarked)``: ``[(zip, derived)]`` to delete, the
    refusals, and ``[(zip, derived)]`` already gone from disk without a marker
    (a run that died between the delete and the save)."""
    by_source: dict[str, list[str]] = {}
    for derived, rec in manifest.get("derived", {}).items():
        by_source.setdefault(rec.get("from", ""), []).append(derived)
    todo, refused, unmarked = [], [], []
    for name, rec in sorted(manifest.get("files", {}).items()):
        if rec.get("product") not in PRODUCTS or rec.get("raw_deleted_at"):
            continue
        raw = dest / "raw" / name
        if not raw.exists():
            derived = by_source.get(name, [])
            if len(derived) == 1:
                unmarked.append((name, derived[0]))
            else:
                refused.append(f"{name}: raw zip absent without a marker and {len(derived)} derived rasters")
            continue
        if name.startswith("sksTradhojd") or raw.parent != dest / "raw":
            refused.append(f"{name}: not a grunddata/markfukt raw zip")
            continue
        missing = [k for k in REQUIRED if not rec.get(k)]
        if missing:
            refused.append(f"{name}: record lacks {missing}")
            continue
        if raw.stat().st_size != rec["bytes"]:
            refused.append(f"{name}: {raw.stat().st_size} bytes on disk, {rec['bytes']} recorded")
            continue
        derived = by_source.get(name, [])
        if len(derived) != 1:
            refused.append(f"{name}: {len(derived)} derived rasters record it as their source")
            continue
        todo.append((name, derived[0]))
    return todo, refused, unmarked


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    dest: Path = args.dest
    manifest_path = dest / "MANIFEST.json"
    if not manifest_path.exists():
        sys.exit(f"REFUSING: {manifest_path} is missing")
    if args.dry_run:
        return _run(args, dest, manifest_path)
    with manifest_lock(dest, "drop_skg_open_raw"):
        return _run(args, dest, manifest_path)


def _run(args, dest: Path, manifest_path: Path) -> int:
    held = _file_sha(manifest_path)
    manifest = load_manifest(dest)
    free0 = shutil.disk_usage(dest).free
    print(f"free before: {free0 / 2**30:.1f} GiB", flush=True)
    todo, refused, unmarked = plan(manifest, dest)
    planned = sum(manifest["files"][n]["bytes"] for n, _ in todo)
    print(f"plan: {len(todo)} raw zips, {planned / 2**30:.1f} GiB; refused {len(refused)}; "
          f"already gone without a marker {len(unmarked)}", flush=True)
    for r in refused:
        print(f"  REFUSED {r}", flush=True)
    for name, _ in unmarked:
        print(f"  UNMARKED {name}: zip gone, no raw_deleted_at", flush=True)
    if args.dry_run:
        return 0 if not refused else 2

    manifest["history"].append({"at": _now(), "argv": ["drop_skg_open_raw", *sys.argv[1:]],
                                "git_sha": _git_sha()})
    deleted = mismatched = 0
    for name, derived in unmarked:
        # an earlier run deleted the zip and died before saving its marker
        out = dest / derived
        if not out.exists() or sha256_of(out) != manifest["derived"][derived].get("sha256"):
            print(f"  UNMARKED {name}: zip gone but {derived} missing or not its recorded sha256", flush=True)
            mismatched += 1
            continue
        manifest["files"][name]["raw_deleted_at"] = _now()
        manifest["files"][name]["raw_absent_at_drop"] = True
        print(f"  marked {name}: zip already gone, {derived} verified", flush=True)
    if unmarked:
        if _file_sha(manifest_path) != held:
            raise ConcurrentWriter(f"{manifest_path} changed during the run; another writer is active")
        save_manifest(dest, manifest)
        held = _file_sha(manifest_path)
    for i, (name, derived) in enumerate(todo, 1):
        out = dest / derived
        if not out.exists() or sha256_of(out) != manifest["derived"][derived].get("sha256"):
            print(f"  [{i}/{len(todo)}] KEPT {name}: {derived} missing or not its recorded sha256", flush=True)
            mismatched += 1
            continue
        if _file_sha(manifest_path) != held:
            raise ConcurrentWriter(f"{manifest_path} changed during the run; another writer is active")
        (dest / "raw" / name).unlink()
        manifest["files"][name]["raw_deleted_at"] = _now()
        save_manifest(dest, manifest)
        held = _file_sha(manifest_path)
        deleted += 1
        print(f"  [{i}/{len(todo)}] deleted {name} ({manifest['files'][name]['bytes'] / 2**30:.2f} GiB); "
              f"{derived} kept and verified", flush=True)

    free1 = shutil.disk_usage(dest).free
    print(f"deleted {deleted}, kept {mismatched + len(refused)}; free after: {free1 / 2**30:.1f} GiB "
          f"(freed {(free1 - free0) / 2**30:.1f} GiB)", flush=True)
    return 0 if not (refused or mismatched) else 2


if __name__ == "__main__":
    sys.exit(main())
