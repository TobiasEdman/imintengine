#!/usr/bin/env python3
"""scripts/check_tiles.py — Tile inventory and QC diagnostics.

Replaces ad-hoc inline python3 -c snippets in k8s job YAML files with
a single, testable, readable script.  Each subcommand exits 0 on pass
and non-zero on hard failure (use --warn to downgrade failures to warnings).

Subcommands
-----------
count
    Print the total number of .npz tiles in DATA_DIR.

frame2016
    Report how many tiles (out of --sample) carry has_frame_2016=1.
    Exits 1 if *none* of the sampled tiles have the key (hard failure).

labels
    Report how many tiles (out of --sample) carry a 'label' key.
    Exits 1 if *none* of the sampled tiles have the key (hard failure).

qc-frame2016
    Shape + non-zero assertion on frame_2016 arrays.  Exits 1 if all
    sampled tiles fail (used as a gate in add-background-frame job).

Usage::

    python scripts/check_tiles.py /data/unified_v2 count
    python scripts/check_tiles.py /data/unified_v2 frame2016 --sample 500
    python scripts/check_tiles.py /data/unified_v2 labels --sample 50
    python scripts/check_tiles.py /data/unified_v2 qc-frame2016 --sample 20 --warn
"""
from __future__ import annotations

import argparse
import glob
import random
import sys
from pathlib import Path

import numpy as np


N_BANDS = 6
TILE_PX = 256


def _tile_paths(data_dir: str, sample: int | None = None) -> list[str]:
    tiles = sorted(glob.glob(str(Path(data_dir) / "*.npz")))
    if sample and sample < len(tiles):
        random.shuffle(tiles)
        tiles = tiles[:sample]
    return tiles


# ── Subcommands ──────────────────────────────────────────────────────────

def cmd_count(data_dir: str, **_) -> int:
    tiles = sorted(glob.glob(str(Path(data_dir) / "*.npz")))
    print(f"  Tiles: {len(tiles):,}")
    return 0


def cmd_frame2016(data_dir: str, sample: int, warn: bool, **_) -> int:
    tiles = _tile_paths(data_dir, sample)
    if not tiles:
        print("  frame2016: no tiles found", file=sys.stderr)
        return 0 if warn else 1

    ok = 0
    for path in tiles:
        try:
            d = np.load(path, allow_pickle=False)
            if int(d.get("has_frame_2016", 0)) == 1:
                ok += 1
        except Exception:
            pass

    n = len(tiles)
    print(f"  frame_2016: {ok}/{n} sampled tiles have has_frame_2016=1")

    if ok == 0:
        msg = f"WARNING: no tiles with has_frame_2016=1 in {n}-tile sample!"
        print(f"  {msg}", file=sys.stderr)
        return 0 if warn else 1
    return 0


def cmd_labels(data_dir: str, sample: int, warn: bool, **_) -> int:
    tiles = _tile_paths(data_dir, sample)
    if not tiles:
        print("  labels: no tiles found", file=sys.stderr)
        return 0 if warn else 1

    ok = 0
    for path in tiles:
        try:
            d = np.load(path, allow_pickle=False)
            if "label" in d.files:
                ok += 1
        except Exception:
            pass

    n = len(tiles)
    print(f"  labels: {ok}/{n} sampled tiles have 'label' key")

    if ok == 0:
        print(f"  WARNING: no tiles with 'label' key in {n}-tile sample!", file=sys.stderr)
        return 0 if warn else 1
    return 0


def cmd_qc_frame2016(data_dir: str, sample: int, warn: bool, **_) -> int:
    tiles = _tile_paths(data_dir, sample)
    if not tiles:
        print("  qc-frame2016: no tiles found", file=sys.stderr)
        return 0 if warn else 1

    ok = fail = 0
    errors: list[str] = []
    for path in tiles:
        name = Path(path).name
        try:
            d = np.load(path, allow_pickle=False)
            if int(d.get("has_frame_2016", 0)) != 1:
                fail += 1
                continue
            frame = d["frame_2016"]
            assert frame.shape == (N_BANDS, TILE_PX, TILE_PX), \
                f"bad shape {frame.shape}, expected ({N_BANDS},{TILE_PX},{TILE_PX})"
            assert frame.max() > 0, "all-zero frame"
            ok += 1
        except AssertionError as exc:
            fail += 1
            errors.append(f"  FAIL {name}: {exc}")
        except Exception as exc:
            fail += 1
            errors.append(f"  ERR  {name}: {exc}")

    print(f"  QC frame_2016: {ok} OK, {fail} failed/missing (of {len(tiles)} sampled)")
    for e in errors[:5]:
        print(e)

    if ok == 0:
        print("  WARNING: no tiles passed frame_2016 QC in sample!", file=sys.stderr)
        return 0 if warn else 1
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────

COMMANDS = {
    "count": cmd_count,
    "frame2016": cmd_frame2016,
    "labels": cmd_labels,
    "qc-frame2016": cmd_qc_frame2016,
}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Tile inventory and QC diagnostics for k8s jobs"
    )
    p.add_argument("data_dir", help="Directory containing .npz tile files")
    p.add_argument(
        "check",
        choices=list(COMMANDS),
        help="Diagnostic to run",
    )
    p.add_argument(
        "--sample", type=int, default=500,
        help="Max tiles to sample for checks (default: 500)",
    )
    p.add_argument(
        "--warn", action="store_true",
        help="Downgrade hard failures to warnings (exit 0)",
    )
    args = p.parse_args()

    fn = COMMANDS[args.check]
    rc = fn(data_dir=args.data_dir, sample=args.sample, warn=args.warn)
    sys.exit(rc)


if __name__ == "__main__":
    main()
