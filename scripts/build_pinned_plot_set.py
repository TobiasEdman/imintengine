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
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Own dir for sibling-script imports (gen_ladder_manifests, validate_
# against_nfi) — what direct execution gets for free but importlib loads
# (the test suite) do not.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Keys a tile must carry for EVERY ladder column to forward it.
# croma_base + terramind_v1_base require SAR; the optical-only families
# (prithvi, tessera, clay) build from spectral/b08/rededge, present on
# all tiles. Extend this list if a future column adds a modality.
REQUIRED_KEYS = ("s1_vv_vh",)
# Presence is NOT sufficient: the dataset gates SAR on the enrichment
# VERSION (s1_enrich_v == 4, RTC γ⁰ season composite — f221f0c), so a
# tile with an old-version s1_vv_vh was never seen by SAR training and
# crashes extract/dense at forward time. Terramind's third submission
# died on exactly one such tile (47023938, v=0). key -> (version_key,
# required_value); checked wherever REQUIRED_KEYS is.
VERSION_REQUIREMENTS = {"s1_vv_vh": ("s1_enrich_v", 4)}


def npz_version_ok(path: Path, keys: tuple[str, ...]) -> bool:
    """True iff every required key's version stamp matches. Reads only
    the scalar stamps (lazy npz), not the arrays."""
    need = [VERSION_REQUIREMENTS[k] for k in keys if k in VERSION_REQUIREMENTS]
    if not need:
        return True
    try:
        with np.load(path) as z:
            for vkey, want in need:
                if vkey not in z.files or int(z[vkey]) != want:
                    return False
    except (OSError, ValueError, zipfile.BadZipFile):
        return False
    return True


def _crop_offset(tile_h: int, img_size: int) -> int:
    """Centre-crop top-left offset — MUST equal
    ``validate_against_nfi.crop_offset`` (pinned by
    ``test_crop_offset_parity``). Inlined rather than imported: that
    module pulls in ``imint.eval.metrics`` and its heavy dependencies,
    which this script's CPU pod deliberately does not install.
    """
    return (tile_h - min(img_size, tile_h)) // 2


def npz_key_names(path: Path) -> set[str] | None:
    """Array names in the npz's zip directory — no array is loaded.

    Returns ``None`` for an unreadable file so callers can tell "readable
    but lacks the key" from "cannot be read at all". The two must not be
    conflated: the first legitimately shrinks a SAR cohort, the second is
    a PVC problem that silently degrading the pinned set would hide.
    """
    try:
        with zipfile.ZipFile(path) as zf:
            return {n[:-4] for n in zf.namelist() if n.endswith(".npy")}
    except (OSError, zipfile.BadZipFile):
        return None


def npz_has_keys(path: Path, keys: tuple[str, ...]) -> bool:
    """True iff the npz is readable and carries every key."""
    names = npz_key_names(path)
    return names is not None and all(k in names for k in keys)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot-index", required=True,
                    help="parquet with tile_name/tile_path/TractID/PlotID")
    ap.add_argument("--data-dir", required=True, help="tile (.npz) directory")
    ap.add_argument("--out", required=True, help="pinned_plots.json path")
    args = ap.parse_args()

    df = pd.read_parquet(args.plot_index)
    for col in ("tile_name", "TractID", "PlotID", "row", "col"):
        if col not in df.columns:
            raise SystemExit(f"plot index missing column {col!r}")

    # Crop-window intersection. Each column's extract crops the 512 tile
    # to ITS img_size and border-drops plots outside the window — and the
    # crop differs per column (496 for prithvi300m/terramind, 504 for the
    # rest), so the drop set differs too. A pinned plot must survive
    # EVERY column's crop, i.e. sit inside the tightest window. Derived
    # from the generator's DISTILL table so a changed column regime
    # re-tightens this automatically. First submission failed exactly
    # here: 600m covered 935/964 pinned, 300m 904/964 — the exact-match
    # guard refused both, as designed.
    from gen_ladder_manifests import DISTILL

    min_img = min(cfg["img_size"] for cfg in DISTILL.values())
    off = _crop_offset(512, min_img)
    in_window = (
        (df["row"] >= off) & (df["row"] < off + min_img)
        & (df["col"] >= off) & (df["col"] < off + min_img)
    )
    border_dropped = int((~in_window).sum())
    df = df[in_window]
    print(f"crop window [{off}, {off + min_img}) from tightest img_size "
          f"{min_img}: {border_dropped} border plots excluded")
    dupes = df.groupby(["tile_name", "TractID", "PlotID"]).size()
    if (dupes > 1).any():
        raise SystemExit(
            f"(tile_name, TractID, PlotID) is not unique: "
            f"{int((dupes > 1).sum())} duplicate keys — cannot pin plots")

    data_dir = Path(args.data_dir)
    tiles = sorted(df["tile_name"].unique().tolist())
    qualifying: set[str] = set()
    missing_file = 0
    unreadable: list[str] = []
    for name in tiles:
        path = data_dir / f"{name}.npz"
        if not path.exists():
            missing_file += 1
            continue
        names = npz_key_names(path)
        if names is None:
            unreadable.append(name)
            continue
        if all(k in names for k in REQUIRED_KEYS) and npz_version_ok(
                path, REQUIRED_KEYS):
            qualifying.add(name)

    # An unreadable tile is a PVC problem, not a cohort property. Folding it
    # into the SAR-less remainder would shrink the pinned set silently —
    # every column would then compute an internally consistent
    # "controlled" distillability number over a degraded population, which
    # is worse than no number. Fail; rerun after the tiles are fixed.
    if unreadable:
        raise SystemExit(
            f"{len(unreadable)}/{len(tiles)} indexed tiles are unreadable "
            f"(first: {unreadable[:10]}). The pinned set must not be built "
            f"on a degraded PVC — fix or exclude these from the plot index "
            f"first.")

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
        "n_tiles_unreadable": 0,  # nonzero aborts above; recorded as attestation
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
