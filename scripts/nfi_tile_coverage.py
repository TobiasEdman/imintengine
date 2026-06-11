"""scripts/nfi_tile_coverage.py — how many NFI plots land on built tiles?

The Phase-1 feasibility gate for using SLU NFI plots as sparse training
targets / validation truth. Globs a tile directory, co-locates the NFI plot
table (year-matched) onto each tile's pixel grid, and reports how many plots
fall on tiles — overall, by inventory year, and by GPS-accuracy tier
(handheld Garmin ≤2023 vs Emlid RTK ≥2024). Persists the
plot→(tile, row, col) index to parquet so downstream steps never recompute
it.

Run locally for a correctness proof on the sample tiles; run where the full
``unified_v2/`` PVC is mounted for the authoritative number that gates the
training track.

    python scripts/nfi_tile_coverage.py --tile-dir data/unified_v2 --size-px 256
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imint.training.nfi_colocate import colocate_plots
from imint.training.slu_nfi import load_nfi_plots
from imint.training.tile_config import TileConfig

# Emlid RS3 RTK (2–5 cm) from 2024; earlier plots were navigated with a
# handheld Garmin (metre-level) — see docs/data/nfi_plotdata_DATA_CARD.md.
RTK_FROM_YEAR = 2024

# Only the small geo/temporal metadata is needed to co-locate — never
# materialise the big image/label arrays (np.load is lazy per key, so this
# keeps the scan cheap across thousands of PVC tiles).
_META_KEYS = ("easting", "northing", "bbox_3006", "tile_size_px", "year", "lpis_year", "dates")


def parse_years(spec: str) -> list[int]:
    spec = spec.strip()
    if "," in spec:
        return [int(x) for x in spec.split(",") if x.strip()]
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(spec)]


def infer_size_px(npz_data: dict, default: int) -> int:
    """Pixel side from the stored bbox extent (/10 m), else ``default``."""
    bbox = npz_data.get("bbox_3006")
    if bbox is not None:
        b = np.asarray(bbox).ravel()
        if b.size >= 4:
            ext = int(b[2]) - int(b[0])
            if ext > 0 and ext % 10 == 0:
                return ext // 10
    return default


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tile-dir", required=True, help="globbed recursively for *.npz")
    ap.add_argument("--size-px", type=int, default=256, help="tile side in px (default 256)")
    ap.add_argument("--infer-size", action="store_true", help="infer size_px per tile from bbox_3006")
    ap.add_argument("--years", default="2018-2025", help="e.g. '2018-2025' or '2022,2023'")
    ap.add_argument("--nfi-dir", default=None, help="dir with the NFI xlsx/parquet (default: repo data/nfi; on ICE: the PVC /data/nfi)")
    ap.add_argument("--out", default="data/nfi/nfi_plot_tile_index.parquet")
    args = ap.parse_args()

    years = parse_years(args.years)
    plots = load_nfi_plots(years=years, **({"data_dir": args.nfi_dir} if args.nfi_dir else {}))
    print(f"NFI plots: {len(plots):,} (years {min(years)}–{max(years)})")
    if plots.empty:
        sys.exit("no NFI plots in the requested years")

    paths = sorted(glob.glob(os.path.join(args.tile_dir, "**", "*.npz"), recursive=True))
    print(f"tiles:     {len(paths):,} under {args.tile_dir}")
    if not paths:
        sys.exit("no tiles found")

    out_cols = list(plots.columns) + ["tile_name", "tile_path", "row", "col"]
    frames: list[pd.DataFrame] = []
    n_unreadable = 0
    for path in paths:
        path = Path(path)
        try:
            with np.load(path, allow_pickle=True) as npz:
                data = {k: npz[k] for k in _META_KEYS if k in npz.files}
        except (EOFError, OSError, ValueError):
            n_unreadable += 1  # truncated / empty / corrupt — skip, don't die
            continue
        size = infer_size_px(data, args.size_px) if args.infer_size else args.size_px
        got = colocate_plots(plots, name=path.stem, npz_data=data, tile=TileConfig(size_px=size))
        if got.empty:
            continue
        got.insert(0, "tile_name", path.stem)
        got.insert(1, "tile_path", str(path))
        frames.append(got)

    idx = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=out_cols)

    n_tiles_with = idx["tile_name"].nunique() if len(idx) else 0
    print("\n=== COVERAGE ===")
    print(f"plots co-located:    {len(idx):,}")
    print(f"tiles with ≥1 plot:  {n_tiles_with:,} / {len(paths):,}")
    if n_unreadable:
        print(f"unreadable tiles:    {n_unreadable:,} (skipped)")
    if len(idx):
        idx["tier"] = np.where(idx["Year"] >= RTK_FROM_YEAR, "RTK", "GPS")
        per_tile = idx["tile_name"].value_counts()
        print(f"plots/tile:          max {per_tile.max()}, mean {per_tile.mean():.2f}")
        print(f"by year:             {idx['Year'].value_counts().sort_index().to_dict()}")
        print(f"by tier:             {idx['tier'].value_counts().to_dict()}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    idx.to_parquet(out, index=False)
    print(f"\nplot→tile index → {out}  ({len(idx):,} rows)")


if __name__ == "__main__":
    main()
