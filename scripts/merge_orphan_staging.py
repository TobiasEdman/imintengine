#!/usr/bin/env python3
"""Additive merge: orphan-staging tiles -> the live 512 dataset.

The orphan campaign (2026-06..07) fetched 512-tiles whose centers exist
only in the 256 dataset, into ``unified_v2_512_orphans_staging``. Unlike
the recoreg promote (``promote_recoreg.py`` — a whole-directory swap),
this merge is PURE ADDITIONS: every staging tile is new to live.

Contract (each violation aborts before anything moves):

1. **No collisions.** ``staging ∩ live`` must be empty — an overlap is a
   replace/remove decision, which belongs to a human, not this script.
2. **Key parity.** Every staging npz must carry the required keys
   (spectral, label, dates) — no half-fetched tiles enter live.
3. **Counted move.** Per-file ``os.rename`` (atomic, same PVC). After:
   ``live_after == live_before + staged`` and ``staging_after == 0``,
   or the script exits non-zero.

Sidecars (manifest.json, class_stats.json, vpp_known_empty.json, .claims,
cache/) stay in the staging dir as campaign archive. The 12 known-empty
VPP tiles move WITH their catalogue intact in the archived sidecar.

DRY-RUN by default — prints the full plan and QC verdicts, moves nothing.

Usage:
    python scripts/merge_orphan_staging.py \\
        --staging /data/unified_v2_512_orphans_staging \\
        --live /data/unified_v2_512 [--execute]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import zipfile

_REQUIRED_MEMBERS = ("spectral.npy", "label.npy", "dates.npy")


def _names(d: str) -> dict[str, str]:
    """{tile_name: path} for real tiles (excludes .tmp.npz artefacts)."""
    out = {}
    for p in glob.glob(os.path.join(d, "*.npz")):
        base = os.path.basename(p)
        if base.endswith(".tmp.npz"):
            continue
        out[base[:-4]] = p
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--staging", default="/data/unified_v2_512_orphans_staging")
    p.add_argument("--live", default="/data/unified_v2_512")
    p.add_argument("--execute", action="store_true",
                   help="perform the move (default: dry-run, print plan only)")
    a = p.parse_args()

    staging = _names(a.staging)
    live = _names(a.live)
    print(f"staging: {len(staging)} tiles   live: {len(live)} tiles")
    if not staging:
        raise SystemExit("staging är tom — inget att merga")

    # Gate 1 — pure additions.
    collisions = sorted(staging.keys() & live.keys())
    if collisions:
        print(f"ABORT: {len(collisions)} namn-kollisioner (replace är ett "
              f"mänskligt beslut): {collisions[:10]}")
        sys.exit(2)
    print("gate 1 OK: inga kollisioner (pure additions)")

    # Gate 2 — key parity on every staging tile (zip namelist: cheap).
    bad: list[str] = []
    for name, path in staging.items():
        try:
            with zipfile.ZipFile(path) as z:
                members = set(z.namelist())
            if any(m not in members for m in _REQUIRED_MEMBERS):
                bad.append(name)
        except (OSError, zipfile.BadZipFile):
            bad.append(name)
    if bad:
        print(f"ABORT: {len(bad)} tiles saknar obligatoriska nycklar "
              f"{_REQUIRED_MEMBERS}: {sorted(bad)[:10]}")
        sys.exit(3)
    print(f"gate 2 OK: alla {len(staging)} tiles bär {_REQUIRED_MEMBERS}")

    # Move.
    verb = "RENAME" if a.execute else "would rename"
    for name, src in sorted(staging.items()):
        dst = os.path.join(a.live, f"{name}.npz")
        if a.execute:
            os.rename(src, dst)
    print(f"{verb}: {len(staging)} tiles {a.staging} -> {a.live}")

    # Post-verification.
    live_after = len(_names(a.live))
    staging_after = len(_names(a.staging))
    expected = len(live) + (len(staging) if a.execute else 0)
    print(f"live efter: {live_after} (förväntat {expected})   "
          f"staging efter: {staging_after}")
    if a.execute and (live_after != len(live) + len(staging)
                      or staging_after != 0):
        print("ABORT-POST: räkningen stämmer inte — utred INNAN något mer görs")
        sys.exit(4)
    print("MERGE OK" if a.execute else "DRY-RUN OK — kör om med --execute")


if __name__ == "__main__":
    main()
