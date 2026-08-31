"""scripts/crosscheck_lucas_lpis.py — LUCAS crops vs LPIS, year-matched.

An independent crop-truth cross-check: LUCAS is a field survey, LPIS
(jordbruksskiften) is the farmers' subsidy declarations. Both are the
year-specific truth for a rotating crop, so the comparison is **strictly
year-matched** — a LUCAS 2022 wheat point is compared to the 2022 LPIS parcel
at that location, never another year's. This validates both the LUCAS→unified
crop mapping and the LPIS labels the model trains on, with no model in the loop.

For each LUCAS crop point (unified class 11-21) it finds the LPIS parcel
containing it in the SAME year (point-in-polygon), maps the parcel's SJV
grödkod → unified class via the authoritative ``SJV_TO_UNIFIED``, and reports
the LUCAS×LPIS agreement + confusion. Points off any parcel (LUCAS crops on
undeclared land) are counted, not scored.

    python scripts/crosscheck_lucas_lpis.py \
        --truth data/lucas/lucas_truth_sweden.parquet \
        --lpis-dir data/lpis --out data/lucas/lucas_lpis_crosscheck.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imint.training.unified_schema import SJV_TO_UNIFIED, UNIFIED_CLASSES  # noqa: E402

CROP_CLASSES = tuple(range(11, 22))  # 11..21 in the unified schema
EPSG_3006 = 3006


def sjv_to_unified(code) -> int:
    """SJV grödkod → unified class (0 = background/unmapped)."""
    try:
        return SJV_TO_UNIFIED.get(int(code), 0)
    except (TypeError, ValueError):
        return 0


def crosscheck_year(lucas_pts, lpis_path: Path, year: int) -> pd.DataFrame:
    """Year-matched point-in-polygon of LUCAS crop points onto LPIS parcels.

    Returns a frame with one row per LUCAS crop point that YEAR, carrying the
    LUCAS unified class and the LPIS-derived unified class (NaN when the point
    is off any parcel)."""
    import geopandas as gpd
    from shapely.geometry import Point

    lp = gpd.read_parquet(lpis_path, columns=["arslager", "grdkod_mar", "geometry"])
    if lp.crs is None or lp.crs.to_epsg() != EPSG_3006:
        lp = lp.to_crs(EPSG_3006)
    # SJV/LPIS GeoParquets are written with (Y, X) coordinates — the CRS
    # declares EPSG:3006 axis order [Northing, Easting], correct per EPSG but
    # every (X, Y)-assuming consumer misses. Canonical detection + fix from
    # ae5fe17: minx > miny is the smoking gun → swap axes via an affine flip
    # [0,1,1,0,0,0] (new_x = y, new_y = x). Vectorized, unlike per-geom
    # shapely.ops.transform. (These local parquets predate the PVC rewrite.)
    minx, miny, _, _ = lp.total_bounds
    if minx > miny:
        print(f"    LPIS axes swapped (minx {minx:.0f} > miny {miny:.0f}) → flipping")
        lp = lp.set_geometry(lp.geometry.affine_transform([0, 1, 1, 0, 0, 0]))
    # Trust arslager, but the file is already a single year — assert it.
    yrs = set(pd.unique(lp["arslager"]))
    if yrs != {year}:
        raise SystemExit(f"{lpis_path}: arslager {yrs} != expected {{{year}}}")

    pts = gpd.GeoDataFrame(
        lucas_pts.copy(),
        geometry=[Point(e, n) for e, n in zip(lucas_pts.easting, lucas_pts.northing)],
        crs=EPSG_3006,
    )
    # within: each LUCAS point → the parcel it sits inside (parcels don't
    # overlap, so this is ~1:1; keep the first if a rare overlap occurs).
    joined = gpd.sjoin(pts, lp[["grdkod_mar", "geometry"]],
                       how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    joined["lpis_unified"] = joined["grdkod_mar"].map(sjv_to_unified)
    joined.loc[joined["grdkod_mar"].isna(), "lpis_unified"] = np.nan
    return pd.DataFrame({
        "point_id": joined["point_id"].to_numpy(),
        "year": year,
        "lucas_unified": joined["unified_class"].to_numpy(),
        "lpis_grodkod": joined["grdkod_mar"].to_numpy(),
        "lpis_unified": joined["lpis_unified"].to_numpy(),
    })


def summarize(df: pd.DataFrame) -> dict:
    """Agreement + confusion over LUCAS crop points that hit an LPIS parcel."""
    on = df[df["lpis_unified"].notna()].copy()
    on["lpis_unified"] = on["lpis_unified"].astype(int)
    n_total = len(df)
    n_on = len(on)
    # crop-vs-crop agreement (both in 11..21); LPIS may map a LUCAS "crop"
    # point to a non-crop unified class (e.g. 15 slåttervall, 8 öppen mark).
    agree = int((on["lucas_unified"] == on["lpis_unified"]).sum())
    per_class = {}
    for c in CROP_CLASSES:
        sub = on[on["lucas_unified"] == c]
        if len(sub) == 0:
            continue
        per_class[UNIFIED_CLASSES[c]] = {
            "n_lucas_on_parcel": int(len(sub)),
            "agree": int((sub["lpis_unified"] == c).sum()),
            "agreement": round(float((sub["lpis_unified"] == c).mean()), 4),
            "top_lpis": {UNIFIED_CLASSES.get(int(k), str(k)): int(v)
                         for k, v in sub["lpis_unified"].value_counts().head(4).items()},
        }
    return {
        "n_lucas_crop_points": n_total,
        "n_on_lpis_parcel": n_on,
        "off_parcel_frac": round(1 - n_on / n_total, 4) if n_total else None,
        "overall_agreement_on_parcel": round(agree / n_on, 4) if n_on else None,
        "per_class": per_class,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--truth", default="data/lucas/lucas_truth_sweden.parquet")
    ap.add_argument("--lpis-dir", default="data/lpis")
    ap.add_argument("--out", default="data/lucas/lucas_lpis_crosscheck.json")
    ap.add_argument("--dump", default="data/lucas/lucas_lpis_pairs.parquet")
    args = ap.parse_args()

    truth = pd.read_parquet(args.truth)
    crops = truth[truth["unified_class"].isin(CROP_CLASSES)]
    print(f"LUCAS crop points: {len(crops):,} "
          f"(years {sorted(pd.unique(crops.year).tolist())})")

    frames = []
    for year in sorted(pd.unique(crops.year).tolist()):
        lpis_path = Path(args.lpis_dir) / f"jordbruksskiften_{year}.parquet"
        yr_pts = crops[crops.year == year]
        if not lpis_path.exists():
            print(f"  year {year}: {len(yr_pts):,} pts — NO LPIS file "
                  f"({lpis_path.name}); skipped (year-match cannot be honored)")
            continue
        print(f"  year {year}: {len(yr_pts):,} LUCAS crop pts × LPIS {lpis_path.name}")
        frames.append(crosscheck_year(yr_pts, lpis_path, year))

    if not frames:
        raise SystemExit("no year had both LUCAS crops and a matching LPIS file")
    pairs = pd.concat(frames, ignore_index=True)

    result = {"per_year": {}, "overall": summarize(pairs)}
    for year, g in pairs.groupby("year"):
        result["per_year"][int(year)] = summarize(g)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    pairs.to_parquet(args.dump, index=False)

    ov = result["overall"]
    print(f"\n=== LUCAS×LPIS (year-matched) ===")
    print(f"on-parcel: {ov['n_on_lpis_parcel']:,}/{ov['n_lucas_crop_points']:,} "
          f"({100*(1-ov['off_parcel_frac']):.1f}%) · "
          f"overall agreement {ov['overall_agreement_on_parcel']}")
    print(f"{'crop':14s} {'n':>6s} {'agree':>7s}  top LPIS classes")
    for name, d in ov["per_class"].items():
        print(f"{name:14s} {d['n_lucas_on_parcel']:>6d} {d['agreement']:>7.3f}  {d['top_lpis']}")
    print(f"\nwrote {args.out} + {args.dump}")


if __name__ == "__main__":
    main()
