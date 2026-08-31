#!/usr/bin/env python3
"""Pin the shared NFI plot set for the ladder's distillability metric.

The ladder's ONE cross-backbone claim rides distillability: each rung-2
head's out-of-fold forest-type accuracy under a pinned protocol
(docs/experiments/label_source_ladder.md). StratifiedKFold's fold
assignment depends only on the sample count and the label vector, so the
comparison is controlled if and only if every column scores the SAME
plots in the SAME canonical order. Nothing enforced that
(docs/experiments/ladder_distill_stage.md, problem 2): CROMA and
TerraMind can only forward tiles that carry ``s1_vv_vh``, so their
feature parquets naturally cover fewer plots than the other four
columns'.

This script computes the pinned set ONCE, from tile-file properties
alone — no model, no GPU, no extraction. A plot is pinned iff its tile
satisfies EVERY column's input requirement (today: the SAR key; the
optical families read keys present on all tiles). Because the set is
derived from data, not from extraction outcomes, the six columns stay
free of any cross-column barrier: each just subsets its own parquet to
the pinned list (``nfi_head_cv.py --pinned-plots``) and fails loudly if
extraction dropped a pinned plot.

    python3 scripts/build_pinned_plot_set.py \
        --plot-index /cephfs/nfi/nfi_index_unified_v2_512.parquet \
        --data-dir /cephfs/unified_v2_512 \
        --out /cephfs/distill/pinned_plots.json
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Keys a tile must carry for EVERY ladder column to forward it.
# croma_base + terramind_v1_base require SAR; the optical-only families
# (prithvi, tessera, clay) build from spectral/b08/rededge, present on
# all tiles. Extend this list if a future column adds a modality.
REQUIRED_KEYS = ("s1_vv_vh",)


def npz_has_keys(path: Path, keys: tuple[str, ...]) -> bool:
    """Membership test on the npz's zip directory — no array is loaded.

    An unreadable tile counts as not qualifying rather than raising: the
    pinned set must under-claim, never over-claim, because a pinned plot
    that later fails extraction aborts that column's distillability run.
    """
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
    except (OSError, zipfile.BadZipFile):
        return False
    return all(f"{k}.npy" in names for k in keys)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot-index", required=True,
                    help="parquet with tile_name/tile_path/TractID/PlotID")
    ap.add_argument("--data-dir", required=True, help="tile (.npz) directory")
    ap.add_argument("--out", required=True, help="pinned_plots.json path")
    args = ap.parse_args()

    df = pd.read_parquet(args.plot_index)
    for col in ("tile_name", "TractID", "PlotID"):
        if col not in df.columns:
            raise SystemExit(f"plot index missing column {col!r}")
    dupes = df.groupby(["tile_name", "TractID", "PlotID"]).size()
    if (dupes > 1).any():
        raise SystemExit(
            f"(tile_name, TractID, PlotID) is not unique: "
            f"{int((dupes > 1).sum())} duplicate keys — cannot pin plots")

    data_dir = Path(args.data_dir)
    tiles = sorted(df["tile_name"].unique().tolist())
    qualifying: set[str] = set()
    missing_file = 0
    for name in tiles:
        path = data_dir / f"{name}.npz"
        if not path.exists():
            missing_file += 1
            continue
        if npz_has_keys(path, REQUIRED_KEYS):
            qualifying.add(name)

    kept = df[df["tile_name"].isin(qualifying)]
    # Canonical order — every consumer sorts the same way, so the fold
    # assignment is identical across columns by construction.
    kept = kept.sort_values(["tile_name", "TractID", "PlotID"])
    plots = [
        {"tile_name": str(t), "TractID": int(tr), "PlotID": int(p)}
        for t, tr, p in kept[["tile_name", "TractID", "PlotID"]].itertuples(
            index=False)
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "required_keys": list(REQUIRED_KEYS),
        "plot_index": str(args.plot_index),
        "n_tiles_indexed": len(tiles),
        "n_tiles_qualifying": len(qualifying),
        "n_tiles_missing_file": missing_file,
        "n_plots_indexed": int(len(df)),
        "n_plots_pinned": len(plots),
        "plots": plots,
    }, indent=1))

    print(f"tiles: {len(qualifying)}/{len(tiles)} qualify "
          f"({missing_file} npz missing on disk)")
    print(f"plots: {len(plots)}/{len(df)} pinned → {out}")
    if not plots:
        raise SystemExit("pinned set is EMPTY — wrong --data-dir?")


if __name__ == "__main__":
    main()
