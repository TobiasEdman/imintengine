"""scripts/scan_tile_bboxes.py — dump the tile grid's centers to a manifest.

A lightweight metadata-only scan: for every tile ``.npz`` under a directory,
read just the location keys (``easting``/``northing``/``year``/``lpis_year``/
``dates``) and write a small parquet. ``np.load`` is lazy per-key, so the big
spectral/label arrays are never decompressed — a 7,882-tile scan is minutes,
not the hours a full read would take.

The manifest lets any point-in-tile co-location (NFI, LUCAS, …) run **locally**
against the grid without re-touching the PVC: ``resolve_tile_bbox`` rebuilds a
tile's bbox from its center via ``TileConfig.bbox_from_center``, so
center + year is all downstream co-location needs.

    python scripts/scan_tile_bboxes.py \
        --tile-dir /cephfs/unified_v2_512 --out /cephfs/lucas/tile_bbox_manifest.parquet
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

from imint.training.nfi_colocate import tile_year  # noqa: E402
from imint.training.tile_bbox import resolve_tile_bbox  # noqa: E402
from imint.training.tile_config import TileConfig  # noqa: E402

# Location keys only — never the pixel arrays.
_LOC_KEYS = ("easting", "northing", "bbox_3006", "year", "lpis_year", "dates")


# Center resolution is size-independent (bbox_from_center just needs the
# center), so any TileConfig works for the manifest scan.
_TILE = TileConfig(size_px=512)


def _scalar(v):
    """Coerce a 0-d / 1-element npz value to a python scalar, else None."""
    if v is None:
        return None
    a = np.asarray(v).ravel()
    return a[0].item() if a.size else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tile-dir", required=True, help="globbed recursively for *.npz")
    ap.add_argument("--out", default="data/lucas/tile_bbox_manifest.parquet")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.tile_dir, "**", "*.npz"),
                             recursive=True))
    print(f"tiles: {len(paths):,} under {args.tile_dir}")
    if not paths:
        sys.exit("no tiles found")

    rows: list[dict] = []
    n_unreadable = 0
    for p in paths:
        path = Path(p)
        try:
            with np.load(path, allow_pickle=True) as npz:
                data = {k: npz[k] for k in _LOC_KEYS if k in npz.files}
        except Exception:
            n_unreadable += 1
            continue
        # Resolve center + year with the SAME canonical helpers downstream
        # co-location uses, so the manifest is a faithful stand-in for the
        # tile: bbox via easting/northing (or bbox_3006 fallback), year via
        # year → lpis_year → dates. Store the resolved center so manifest-mode
        # never needs bbox_3006.
        bbox = resolve_tile_bbox(name=path.stem, tile=_TILE, npz_data=data)
        east = north = None
        if bbox is not None:
            east = (bbox["west"] + bbox["east"]) // 2
            north = (bbox["south"] + bbox["north"]) // 2
        rows.append({
            "tile_name": path.stem,
            "easting": east,
            "northing": north,
            "year": tile_year(data),
            "lpis_year": _scalar(data.get("lpis_year")),
        })

    df = pd.DataFrame(rows)
    have_center = df[["easting", "northing"]].notna().all(axis=1).sum()
    print(f"scanned:   {len(df):,}  | with center: {have_center:,}  "
          f"| unreadable: {n_unreadable:,}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"tile bbox manifest → {out}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
