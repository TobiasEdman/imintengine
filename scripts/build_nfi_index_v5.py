#!/usr/bin/env python3
"""Build the NFI plot index v5 over every declared tile root (task 0117).

Index v4 covered the training cohort and the nfi2024 campaign. v5 adds the
evaluation copy and the per-year top-up roots, so that the freeze
(``scripts/prepare_nfi_holdout.py``) can deduplicate plot-years across all
of them and exclude every plot-year that also lies on a cohort tile.

Rows and columns are v4's (``scripts/nfi_tile_coverage.py``): year-exact
colocation through ``colocate_plots``, tile size from ``bbox_3006``, and the
GPS/RTK tier. Unlike that scan, it fails on an unreadable tile and on a tile
name found in two roots, both of which the freeze also refuses, and it lists
each root the way the freeze does (``*.npz``, non-recursive, no
``.tmp.npz``). It never overwrites. The JSON sidecar holds only aggregates,
input hashes and the parquet's sha256: plot records stay on the volume.

    python scripts/build_nfi_index_v5.py --nfi-dir /data/nfi \\
        --root cohort:/data/unified_v2_512 \\
        --root campaign:/data/unified_v2_512_nfi2024_staging \\
        --root holdout_copy:/data/holdout_eval_512 \\
        --root topup2018:/data/unified_v2_512_nfi2018_staging \\
        --out /data/nfi/nfi_index_v5_2026-10-24.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imint.training.nfi_colocate import colocate_plots
from imint.training.slu_nfi import PARQUET_NAME, load_nfi_plots
from imint.training.tile_config import TileConfig

KEY = ["TractID", "PlotID", "Year"]
RTK_FROM_YEAR = 2024  # Emlid RTK from 2024, handheld GPS before (as in v4)
META_KEYS = ("easting", "northing", "bbox_3006", "tile_size_px", "year", "lpis_year", "dates")


def parse_root(value: str) -> tuple[str, Path]:
    role, sep, path = value.partition(":")
    if not sep or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", role) or not path:
        raise argparse.ArgumentTypeError("--root requires ROLE:PATH")
    return role, Path(path)


def root_tiles(directory: Path) -> dict[str, Path]:
    """The freeze's membership rule (``prepare_nfi_holdout.tile_files``)."""
    if not directory.is_dir():
        raise SystemExit(f"not a directory: {directory}")
    return {p.stem: p for p in sorted(directory.glob("*.npz")) if not p.name.endswith(".tmp.npz")}


def size_px(meta: dict) -> int:
    """Side in pixels: from the bbox extent at 10 m, else the stored tile_size_px.
    A tile with neither cannot be placed and stops the build."""
    if "bbox_3006" in meta:
        b = np.asarray(meta["bbox_3006"], dtype=float).ravel()
        if (b.size != 4 or not np.isfinite(b).all() or b[2] <= b[0] or b[3] <= b[1]
                or (b[2] - b[0]) % 10):
            raise SystemExit(f"invalid bbox_3006: {b.tolist()}")
        return int((b[2] - b[0]) // 10)
    if "tile_size_px" in meta:
        return int(meta["tile_size_px"])
    raise SystemExit("tile has neither bbox_3006 nor tile_size_px")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build(roots: list[tuple[str, Path]], plots: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    owner: dict[str, str] = {}
    frames, summary = [], []
    for role, directory in roots:
        tiles = root_tiles(directory)
        rows = with_plots = 0
        for name, path in tiles.items():
            if name in owner:
                raise SystemExit(f"tile name {name} is in roots {owner[name]} and {role}")
            owner[name] = role
            with np.load(path, allow_pickle=False) as npz:  # an unreadable tile raises
                meta = {k: npz[k] for k in META_KEYS if k in npz.files}
            got = colocate_plots(plots, name=name, npz_data=meta, tile=TileConfig(size_px=size_px(meta)))
            if got.empty:
                continue
            got.insert(0, "tile_name", name)
            got.insert(1, "tile_path", str(path))
            frames.append(got)
            rows += len(got)
            with_plots += 1
        summary.append({"role": role, "root": str(directory.resolve()), "tiles": len(tiles),
                        "tiles_with_plots": with_plots, "rows": rows})
    columns = ["tile_name", "tile_path", *plots.columns, "row", "col"]
    index = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    index["tier"] = np.where(index["Year"] >= RTK_FROM_YEAR, "RTK", "GPS")
    if index.duplicated(KEY + ["tile_name"]).any():
        raise SystemExit("a plot-year occurs twice on one tile")
    return index, summary


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=parse_root, action="append", required=True, help="repeat ROLE:PATH")
    ap.add_argument("--nfi-dir", type=Path, required=True, help="directory holding nfi_plots.parquet")
    ap.add_argument("--first-year", type=int, default=2018)
    ap.add_argument("--last-year", type=int, default=2025)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    sidecar = args.out.with_suffix(".json")
    for path in (args.out, sidecar):
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}")
    plots_path = args.nfi_dir / PARQUET_NAME
    if not plots_path.exists():
        raise SystemExit(f"missing {plots_path}; the index never builds the plot cache")
    plots_sha = sha256_file(plots_path)
    plots = load_nfi_plots(data_dir=args.nfi_dir, years=range(args.first_year, args.last_year + 1))
    if plots.empty:
        raise SystemExit("no NFI plots in the requested years")

    index, roots = build(args.root, plots)
    if sha256_file(plots_path) != plots_sha:
        raise SystemExit(f"{plots_path} changed during the build")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "xb") as f:      # exclusive create: never overwrite
        index.to_parquet(f, index=False)
    if len(pd.read_parquet(args.out)) != len(index):
        raise SystemExit("parquet read-back mismatch")
    record = {
        "schema": "nfi-index-v5",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": os.environ.get("IMINT_GIT_SHA", "unknown"),
        "nfi_plots": {"path": str(plots_path.resolve()), "sha256": plots_sha,
                      "years": [args.first_year, args.last_year], "plots": len(plots)},
        "roots": roots,
        "rows": len(index),
        "observations": int(len(index.drop_duplicates(KEY))),
        "rows_by_year": {str(int(y)): int(n) for y, n in index["Year"].value_counts().sort_index().items()},
        "parquet": {"path": str(args.out.resolve()), "sha256": sha256_file(args.out)},
    }
    with open(sidecar, "x") as f:
        f.write(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
    return record


if __name__ == "__main__":
    main()
