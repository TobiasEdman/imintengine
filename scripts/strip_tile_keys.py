#!/usr/bin/env python3
"""Remove one or more arrays from unified-dataset tiles, in place.

Drops the named key(s) from each `.npz` by copying every *other* zip member
verbatim into a temp file and atomically replacing the original. No numpy
round-trip — all retained arrays keep their exact bytes/dtype/compression.
Idempotent (tiles already lacking the key are skipped) and safe to re-run.

Usage:
    # verify on one tile (non-destructive: strips a copy, diffs with numpy):
    python scripts/strip_tile_keys.py --verify <tile.npz> --key harvest_probability

    # strip in place across a directory:
    python scripts/strip_tile_keys.py --dir /data/unified_v2 --key harvest_probability [--workers 8]
"""
from __future__ import annotations

import argparse
import glob
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor


def strip_tile(path: str, keys: set[str]) -> str:
    """Atomically rewrite `path` without the named arrays. Returns a status."""
    drop = {f"{k}.npy" for k in keys}
    with zipfile.ZipFile(path) as zin:
        members = zin.infolist()
        present = {m.filename for m in members} & drop
        if not present:
            return "skip"  # idempotent — nothing to remove
        tmp = path + ".tmp"
        with zipfile.ZipFile(tmp, "w") as zout:
            for m in members:
                if m.filename in drop:
                    continue
                # writestr(zinfo, ...) preserves the member's compress_type,
                # so retained arrays are byte-for-byte equivalent on reload.
                zout.writestr(m, zin.read(m.filename))
    os.replace(tmp, path)
    return "stripped"


def verify(tile: str, keys: set[str]) -> None:
    """Strip a COPY and prove only the named keys changed."""
    import shutil
    import tempfile
    import numpy as np

    # Work on a copy in a temp dir so the tile dir can be mounted read-only.
    work = os.path.join(tempfile.gettempdir(), os.path.basename(tile) + ".verify.npz")
    shutil.copy2(tile, work)
    try:
        before = dict(np.load(tile, allow_pickle=True))
        strip_tile(work, keys)
        after = dict(np.load(work, allow_pickle=True))
        removed = set(before) - set(after)
        added = set(after) - set(before)
        print(f"tile: {os.path.basename(tile)}")
        print(f"  keys before: {len(before)}  after: {len(after)}")
        print(f"  removed: {sorted(removed)}")
        print(f"  unexpectedly added: {sorted(added) or 'none'}")

        def arr_eq(a, b):
            if a.shape != b.shape or a.dtype != b.dtype:
                return False
            if a.dtype.kind in "fc":        # float/complex: NaN-aware
                return np.array_equal(a, b, equal_nan=True)
            try:                            # else: exact byte equality
                return a.tobytes() == b.tobytes()
            except Exception:
                return bool(np.array_equal(a, b))

        identical = all(arr_eq(before[k], after[k]) for k in after)
        print(f"  all retained arrays bit-identical: {identical}")
        ok = removed == keys and not added and identical
        print(f"  VERIFY {'PASS' if ok else 'FAIL'}")
    finally:
        if os.path.exists(work):
            os.remove(work)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", action="append", required=True,
                    help="array name to remove (repeatable)")
    ap.add_argument("--dir", help="tile directory to strip in place")
    ap.add_argument("--verify", help="single tile to verify (non-destructive)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    keys = set(a.key)

    if a.verify:
        verify(a.verify, keys)
        return

    files = sorted(glob.glob(os.path.join(a.dir, "*.npz")))
    print(f"scanning {len(files)} tiles for {sorted(keys)} ...")
    counts = {"stripped": 0, "skip": 0, "error": 0}

    def work(f: str) -> str:
        try:
            return strip_tile(f, keys)
        except Exception as e:  # keep going; report at end
            print(f"  ERROR {os.path.basename(f)}: {e}")
            return "error"

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, status in enumerate(ex.map(work, files), 1):
            counts[status] += 1
            if i % 1000 == 0:
                print(f"  {i}/{len(files)}  {counts}")
    print(f"done: {counts}")


if __name__ == "__main__":
    main()
