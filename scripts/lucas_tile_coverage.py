"""scripts/lucas_tile_coverage.py — co-locate LUCAS truth points onto tiles.

L1 of docs/experiments/lucas_validation_plan.md. Takes the LUCAS truth set
(scripts/build_lucas_truth.py) and the unified_v2_512 tile grid, and writes a
point → (tile, row, col) index — the LUCAS analogue of
scripts/nfi_tile_coverage.py. Reuses ``colocate_plots`` verbatim (year-matched:
a LUCAS 2022 point only lands on a 2022 tile, per the campaign temporal rule).

Adds the leakage-guard tag from ``distill_split.json`` so downstream L-head CV
and L-direct scoring never let a held-out point share a tile with training.

Run locally for a correctness proof on the sample tiles; run on the cluster
(CPU pod, /cephfs/unified_v2_512) for the full 7,882-tile coverage.

    python scripts/lucas_tile_coverage.py \
        --tile-dir /cephfs/unified_v2_512 --size-px 512 \
        --split data/distill/distill_split.json \
        --out data/lucas/lucas_tile_index.parquet
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imint.training.nfi_colocate import colocate_plots  # noqa: E402
from imint.training.tile_config import TileConfig  # noqa: E402

# The metadata keys colocate_plots reads — same set nfi_tile_coverage scans.
_META_KEYS = ("easting", "northing", "bbox_3006", "tile_size_px",
              "year", "lpis_year", "dates")


def load_truth(path: Path) -> pd.DataFrame:
    """Load the LUCAS truth parquet and expose the columns colocate_plots
    expects (``Easting``/``Northing``/``Year``) without dropping the LUCAS
    identity/label columns."""
    df = pd.read_parquet(path)
    need = {"easting", "northing", "year", "point_id", "unified_class"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{path}: truth set missing columns {missing}")
    return df.rename(columns={"easting": "Easting", "northing": "Northing",
                              "year": "Year"})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tile-dir", help="globbed recursively for *.npz (reads each tile)")
    src.add_argument("--manifest", help="tile_bbox_manifest.parquet from scan_tile_bboxes.py "
                     "(co-locate locally against tile centers, no tile reads)")
    ap.add_argument("--size-px", type=int, default=512)
    ap.add_argument("--infer-size", action="store_true",
                    help="infer size_px per tile from bbox_3006")
    ap.add_argument("--truth", default="data/lucas/lucas_truth_sweden.parquet")
    ap.add_argument("--split", default="data/distill/distill_split.json",
                    help="distill_split.json — its test_tiles tag the held-out points")
    ap.add_argument("--out", default="data/lucas/lucas_tile_index.parquet")
    args = ap.parse_args()

    truth = load_truth(Path(args.truth))
    # Scorable points only carry the index forward; EXCLUDE (-1) rows are
    # counted in L0 and never co-located (they have no valid target).
    scorable = truth[truth["unified_class"] >= 0]
    print(f"LUCAS truth: {len(truth):,} pts ({len(scorable):,} scorable, "
          f"{len(truth) - len(scorable):,} EXCLUDE)")

    # Each source yields (tile_name, npz_data-dict) pairs; co-location is the
    # same canonical colocate_plots either way.
    if args.manifest:
        man = pd.read_parquet(args.manifest)
        tiles = ((r.tile_name, {"easting": r.easting, "northing": r.northing,
                                "year": r.year, "lpis_year": r.lpis_year})
                 for r in man.itertuples(index=False))
        n_tiles = len(man)
        print(f"tiles:       {n_tiles:,} from manifest {args.manifest}")
    else:
        paths = sorted(glob.glob(os.path.join(args.tile_dir, "**", "*.npz"),
                                 recursive=True))
        n_tiles = len(paths)
        print(f"tiles:       {n_tiles:,} under {args.tile_dir}")
        if not paths:
            sys.exit("no tiles found")
        tiles = _iter_tile_reads(paths)

    test_tiles = set()
    if args.split and Path(args.split).exists():
        test_tiles = {str(t) for t in json.loads(Path(args.split).read_text())
                      .get("test_tiles", [])}
        print(f"split:       {len(test_tiles):,} held-out tiles from {args.split}")

    frames: list[pd.DataFrame] = []
    for name, data in tiles:
        size = _infer_size(data, args.size_px) if args.infer_size else args.size_px
        got = colocate_plots(scorable, name=str(name), npz_data=data,
                             tile=TileConfig(size_px=size))
        if got.empty:
            continue
        got.insert(0, "tile_name", str(name))
        frames.append(got)

    idx = (pd.concat(frames, ignore_index=True) if frames
           else pd.DataFrame(columns=list(scorable.columns) + ["tile_name", "row", "col"]))

    if len(idx):
        idx["split"] = np.where(idx["tile_name"].astype(str).isin(test_tiles),
                                "test", "train")

    print("\n=== COVERAGE ===")
    print(f"points co-located:   {len(idx):,}")
    print(f"tiles with ≥1 point: {idx['tile_name'].nunique() if len(idx) else 0:,} / {n_tiles:,}")
    if len(idx):
        by_cls = idx.groupby("unified_class").size().sort_index()
        print("\nper unified class (co-located):")
        for cls, n in by_cls.items():
            held = int((idx[idx["unified_class"] == cls]["split"] == "test").sum())
            print(f"  {int(cls):2d} {str(idx[idx.unified_class==cls]['unified_name'].iloc[0]):26s} "
                  f"{n:6,d}  ({held:,} held-out)")
        print(f"\nby year:  {idx['Year'].value_counts().sort_index().to_dict()}")
        print(f"by split: {idx['split'].value_counts().to_dict()}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    idx.to_parquet(out, index=False)
    print(f"\nLUCAS point→tile index → {out}  ({len(idx):,} rows)")


def _iter_tile_reads(paths: list[str]):
    """Yield (tile_name, metadata-dict) reading each npz's location keys only.

    One corrupt tile in a bulk scan is skipped, never fatal."""
    for p in paths:
        path = Path(p)
        try:
            with np.load(path, allow_pickle=True) as npz:
                data = {k: npz[k] for k in _META_KEYS if k in npz.files}
        except Exception:
            continue
        yield path.stem, data


def _infer_size(npz_data: dict, default: int) -> int:
    """Infer tile side in px from bbox_3006 span / 10 m GSD (fallback: default)."""
    bbox = npz_data.get("bbox_3006")
    if bbox is None:
        return default
    b = np.asarray(bbox).ravel()
    if b.size < 4:
        return default
    return int(round((float(b[2]) - float(b[0])) / 10.0))


if __name__ == "__main__":
    main()
