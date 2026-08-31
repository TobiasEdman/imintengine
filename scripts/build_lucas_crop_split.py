#!/usr/bin/env python3
"""Freeze the LUCAS crop distill/holdout split — ONCE, before any training.

LUCAS is the ladder's independent cross-validator ("never trained on by
any rung"). Distilling crop type from it burns that property unless the
data is split FIRST: a grouped-by-tile 70/30 freeze [user-approved
2026-08-31] keeps a holdout side that remains untouched-by-training, so
the cross-check survives as "LUCAS holdout never trained on".

Non-negotiables encoded here:
- **Grouped by tile** — points on one tile share context; a point-level
  split would leak tile context across sides (same isolation argument as
  the NFI head's grouped split).
- **The pre-existing index split is honoured**: the L1 index carries a
  'test' side (71 points) frozen by an earlier experiment; its tiles are
  FORCED into our holdout so no prior freeze leaks into distill-train.
- **Holdout must cover all 11 crop classes** (it is the future validator;
  a class absent there is a class we can never score). Seeded retry search, like the NFI grouped_split.
- Same physical pinning as the NFI set: tiles must carry s1_vv_vh (the
  SAR-cohort intersection) and points must sit inside every column's
  crop window (row/col in [off, off+min_img)).

Outputs (PVC):
- ``lucas_crop_distill_index.parquet`` — the 70% side, extract-ready
  (tile_name/tile_path/row/col/unified_class/point_id).
- ``lucas_crop_split.json`` — provenance + the pinned plot list with
  ``key_cols`` so consumers verify the exact-match guard, + holdout tile
  list (points NOT enumerated here; the validator reads the index).

    python3 scripts/build_lucas_crop_split.py \
        --lucas-index /cephfs/lucas/lucas_tile_index.parquet \
        --data-dir /cephfs/unified_v2_512 \
        --out-dir /cephfs/distill
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_pinned_plot_set import (npz_key_names, npz_version_ok,
                                   REQUIRED_KEYS, _crop_offset)

CROP_CLASSES = tuple(range(11, 22))  # vete..majs, unified schema v5
SEED = 42
HOLDOUT_FRAC = 0.30
MIN_HOLDOUT_PER_CLASS = 5


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lucas-index", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    from gen_ladder_manifests import DISTILL

    df = pd.read_parquet(args.lucas_index)
    crops = df[df["unified_class"].isin(CROP_CLASSES)].copy()
    print(f"crop points: {len(crops)}/{len(df)} on "
          f"{crops['tile_name'].nunique()} tiles")

    # Crop-window intersection — identical arithmetic to the NFI pinned set.
    min_img = min(cfg["img_size"] for cfg in DISTILL.values())
    off = _crop_offset(512, min_img)
    in_win = ((crops["row"] >= off) & (crops["row"] < off + min_img)
              & (crops["col"] >= off) & (crops["col"] < off + min_img))
    print(f"crop window [{off}, {off + min_img}): "
          f"{int((~in_win).sum())} border points excluded")
    crops = crops[in_win]

    # SAR-cohort tile qualification (unreadable aborts, as in the NFI set).
    data_dir = Path(args.data_dir)
    qual: dict[str, bool] = {}
    unreadable: list[str] = []
    for name in crops["tile_name"].unique():
        p = data_dir / f"{name}.npz"
        if not p.exists():
            qual[name] = False
            continue
        names = npz_key_names(p)
        if names is None:
            unreadable.append(name)
        else:
            qual[name] = (all(k in names for k in REQUIRED_KEYS)
                          and npz_version_ok(p, REQUIRED_KEYS))
    if unreadable:
        raise SystemExit(
            f"{len(unreadable)} unreadable tiles (first {unreadable[:5]}) — "
            f"no split is frozen on a degraded PVC.")
    crops = crops[crops["tile_name"].map(qual).fillna(False)]
    print(f"after SAR/window/existence pinning: {len(crops)} points on "
          f"{crops['tile_name'].nunique()} tiles")

    # The L1 index's own 'test' side is an earlier freeze — its tiles go
    # to OUR holdout unconditionally.
    forced_holdout = set(df.loc[df.get("split", "") == "test", "tile_name"])

    tiles = np.array(sorted(set(crops["tile_name"]) - forced_holdout))
    rng_base = args.seed
    best = None
    for trial in range(50):
        rng = np.random.default_rng(rng_base + trial)
        n_hold = max(1, int(round(len(tiles) * HOLDOUT_FRAC))
                     - len(forced_holdout & set(crops["tile_name"])))
        hold_tiles = set(rng.choice(tiles, size=n_hold, replace=False))
        hold_tiles |= (forced_holdout & set(crops["tile_name"]))
        hold = crops[crops["tile_name"].isin(hold_tiles)]
        support = hold["unified_class"].value_counts()
        cover = sum(1 for c in CROP_CLASSES
                    if support.get(c, 0) >= MIN_HOLDOUT_PER_CLASS)
        if best is None or cover > best[0]:
            best = (cover, trial, hold_tiles)
        if cover == len(CROP_CLASSES):
            break
    cover, trial, hold_tiles = best
    if cover < len(CROP_CLASSES):
        raise SystemExit(
            f"no seed in 50 trials gave every crop class >= "
            f"{MIN_HOLDOUT_PER_CLASS} holdout points (best {cover}/11) — "
            f"loosen MIN_HOLDOUT_PER_CLASS deliberately, do not ship a "
            f"validator that cannot score a class.")
    if trial:
        print(f"  note: seed+{trial} used for full holdout class coverage")

    hold = crops[crops["tile_name"].isin(hold_tiles)]
    dist = crops[~crops["tile_name"].isin(hold_tiles)]
    assert not (set(dist.tile_name) & set(hold.tile_name)), "tile leak"
    print(f"distill: {len(dist)} points / {dist.tile_name.nunique()} tiles; "
          f"holdout: {len(hold)} points / {hold.tile_name.nunique()} tiles")
    print("holdout class support:",
          hold["unified_class"].value_counts().sort_index().to_dict())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dist = dist.sort_values(["tile_name", "point_id"]).reset_index(drop=True)
    dist.to_parquet(out_dir / "lucas_crop_distill_index.parquet")

    (out_dir / "lucas_crop_split.json").write_text(json.dumps({
        "seed": args.seed, "trial_offset": trial,
        "holdout_frac": HOLDOUT_FRAC,
        "min_holdout_per_class": MIN_HOLDOUT_PER_CLASS,
        "required_keys": list(REQUIRED_KEYS),
        "crop_window": [off, off + min_img],
        "key_cols": ["tile_name", "point_id"],
        "truth_col": "unified_class",
        "n_distill": int(len(dist)),
        "n_holdout": int(len(hold)),
        "holdout_tiles": sorted(hold_tiles),
        "forced_holdout_tiles_from_prior_split": sorted(
            forced_holdout & set(crops["tile_name"])),
        "plots": [
            {"tile_name": str(t), "point_id": int(p)}
            for t, p in dist[["tile_name", "point_id"]].itertuples(index=False)
        ],
    }, indent=1))
    print(f"wrote {out_dir}/lucas_crop_distill_index.parquet + "
          f"lucas_crop_split.json")


if __name__ == "__main__":
    main()
