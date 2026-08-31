#!/usr/bin/env python3
"""Convert an SJV INSPIRE WFS shape-zip (jordbruksskiften) to a GeoParquet.

Codifies the LPIS conversion that has now been solved TWICE ad hoc:

* 2026-04-25..27: ae5fe17 (rewrite parquets with swapped X/Y), ff0d51f
  (compensating rasterize bug), 38a18a4 (9-job debug investigation).
* 2026-07-20: the 2021 layer — never downloaded in April — was fetched
  fresh and the whole axis-swap was rediscovered from scratch, because
  no in-repo script encoded the fix.

The trap: SJV's WFS 2.0 shape-zip export delivers EPSG:3006 in the
EPSG-official axis order **(Northing, Easting)**. Every consumer in this
repo (SpatialParquet.query row-group pruning, _rasterize_parcels,
rasterio) assumes GIS-conventional (x=Easting, y=Northing). An unswapped
parquet yields ZERO parcels for every tile bbox — silently, since
build_labels only logs the first 10 failure reasons.

Pipeline: read zip (polygon layer) -> bounds-gate -> axis-swap if needed
-> schema check against a reference-year parquet -> write -> readback
assert. Follow with scripts/preprocess_sks_lpis_spatial.py to build the
``_spatial`` variant.

Usage:
    python scripts/convert_lpis_wfs_zip.py \\
        --zip /data/lpis/jordbruksskiften_2021.zip \\
        --ref-parquet /data/lpis/jordbruksskiften_2022.parquet \\
        --out /data/lpis/jordbruksskiften_2021.parquet
"""
from __future__ import annotations

import argparse

# Sweden in EPSG:3006 (SWEREF99 TM).
_EASTING_RANGE = (200_000, 1_000_000)
_NORTHING_RANGE = (6_000_000, 7_800_000)


def _axes_ok(bounds) -> bool:
    """True iff [minx, miny, maxx, maxy] reads as (easting, northing)."""
    e_lo, e_hi = _EASTING_RANGE
    n_lo, n_hi = _NORTHING_RANGE
    return (e_lo < bounds[0] < e_hi and e_lo < bounds[2] < e_hi
            and n_lo < bounds[1] < n_hi and n_lo < bounds[3] < n_hi)


def _ensure_en_axes(gdf):
    """Return ``gdf`` with GIS-conventional (Easting, Northing) geometry.

    No-op if already (E, N); swaps the (N, E) WFS-official order otherwise;
    raises if the bounds still don't read as Swedish 3006 after a swap (guards
    against double-swapping already-correct data)."""
    import shapely

    b = gdf.total_bounds
    print(f"  total_bounds: {b}")
    if _axes_ok(b):
        return gdf
    print("  axes SWAPPED (WFS official N,E order) — swapping to (E,N)")
    gdf = gdf.copy()
    gdf["geometry"] = shapely.transform(gdf.geometry.values, lambda c: c[:, ::-1])
    b = gdf.total_bounds
    print(f"  total_bounds after swap: {b}")
    if not _axes_ok(b):
        raise SystemExit(f"axes still wrong after swap: {b}")
    return gdf


def fix_parquet_axes(path: str, *, backup: bool = True) -> str:
    """Idempotently correct an existing GeoParquet's axis order in place.

    For parquets written before the converter existed (the March local
    LPIS files predate d368dc2). Reads, bounds-gates, and — only if swapped —
    backs the original up to ``<path>.bak`` (never overwriting an existing
    backup) and rewrites with (E, N) geometry. Returns a one-word status:
    ``"ok"`` (already correct, untouched) or ``"fixed"``.
    """
    import geopandas as gpd

    gdf = gpd.read_parquet(path)
    if _axes_ok(gdf.total_bounds):
        print(f"{path}: axes already OK — untouched")
        return "ok"
    if backup:
        from pathlib import Path
        bak = Path(f"{path}.bak")
        if not bak.exists():
            gpd.read_parquet(path).to_parquet(bak, index=False)
            print(f"  backed up original → {bak}")
    gdf = _ensure_en_axes(gdf)
    gdf.to_parquet(path, index=False)
    chk = gpd.read_parquet(path)
    if len(chk) != len(gdf) or not _axes_ok(chk.total_bounds):
        raise SystemExit(f"readback mismatch after fix: {len(chk)} rows, "
                         f"bounds {chk.total_bounds}")
    print(f"{path}: FIXED ({len(chk)} parcels, axes OK)")
    return "fixed"


def convert(zip_path: str, ref_parquet: str | None, out_path: str,
            layer: str = "arslager_skiftePolygon") -> int:
    """Convert; returns the parcel count. Raises on any sanity failure.

    ``ref_parquet`` cross-checks schema + CRS against a known-good year; pass
    ``None`` to skip that check (bootstrap on a fresh machine with no parquet
    yet — the axis bounds-gate + readback assert still run)."""
    import geopandas as gpd

    print(f"reading {zip_path} (layer {layer})...")
    gdf = gpd.read_file(f"zip://{zip_path}", layer=layer)
    print(f"  {len(gdf)} parcels, crs={gdf.crs}")

    gdf = _ensure_en_axes(gdf)

    if ref_parquet:
        ref = gpd.read_parquet(ref_parquet)
        missing = set(ref.columns) - set(gdf.columns)
        if missing:
            raise SystemExit(f"schema mismatch — source saknar kolumner: {missing}")
        # pyproj equality, NOT str() — the projjson representation of 3006
        # differs textually from "EPSG:3006" while being the same CRS, and a
        # str-triggered to_crs is a numeric no-op that does NOT fix the swap.
        if gdf.crs != ref.crs:
            raise SystemExit(f"CRS mismatch per pyproj: {gdf.crs} vs {ref.crs}")
        gdf = gdf[list(ref.columns)]

    gdf.to_parquet(out_path, index=False)

    chk = gpd.read_parquet(out_path)
    if len(chk) != len(gdf) or not _axes_ok(chk.total_bounds):
        raise SystemExit(f"readback mismatch: {len(chk)} rows, "
                         f"bounds {chk.total_bounds}")
    print(f"wrote {out_path} ({len(chk)} parcels, axes OK)")
    return len(chk)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--zip", help="SJV WFS shape-zip path (fresh conversion)")
    src.add_argument("--fix-parquet", nargs="+",
                     help="existing parquet(s) to correct in place (idempotent, "
                     "backs up to .bak). For pre-converter local files.")
    p.add_argument("--ref-parquet",
                   help="known-good parquet from another year (schema + CRS ref); "
                   "required with --zip")
    p.add_argument("--out", help="output GeoParquet path; required with --zip")
    p.add_argument("--layer", default="arslager_skiftePolygon",
                   help="zip layer (the zip also holds an *_NULL layer)")
    a = p.parse_args()

    if a.fix_parquet:
        stats = {f: fix_parquet_axes(f) for f in a.fix_parquet}
        n_fixed = sum(v == "fixed" for v in stats.values())
        print(f"\n{n_fixed}/{len(stats)} parquet(s) fixed, "
              f"{len(stats) - n_fixed} already OK")
        return

    if not (a.ref_parquet and a.out):
        raise SystemExit("--zip requires --ref-parquet and --out")
    convert(a.zip, a.ref_parquet, a.out, layer=a.layer)
    print("NEXT: python scripts/preprocess_sks_lpis_spatial.py "
          "--lpis-dir <dir> för _spatial-varianten")


if __name__ == "__main__":
    main()
