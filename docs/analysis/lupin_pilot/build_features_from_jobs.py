#!/usr/bin/env python3
"""Sample the batch-job rasters at the pilot points -> features_s2.npz.

Pure local step: reads the GeoTIFFs that fetch_s2_batch.py collected and
turns them into the same feature matrix schema the TESSERA run produced
(X, y, lat, lon, year, month, block, dist_lupin, dist_road), so
``probe_train.py`` runs unchanged and the two are comparable on identical
labels, folds and controls.

Points are projected into each raster's OWN CRS rather than assumed to be
SWEREF99: CDSE returns UTM (measured EPSG:32632 and :32633 for Swedish
bboxes). Reusing the SWEREF99 coordinates would sample the wrong pixels
and still return perfectly plausible reflectance — a silent failure.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default=str(HERE / "chosen_points.json"))
    ap.add_argument("--road-dist", default=str(HERE / "road_dist.json"))
    ap.add_argument("--dist-lupin", default=str(HERE / "dist_lupin.json"))
    ap.add_argument("--job-dir", default=str(HERE / "s2_jobs"))
    ap.add_argument("--out", default=str(HERE / "features_s2.npz"))
    args = ap.parse_args()

    import rasterio
    import rasterio.transform as rt
    from pyproj import Transformer
    from s2_june_composite import FEATURE_NAMES

    pts = json.loads(Path(args.points).read_text())
    road = json.loads(Path(args.road_dist).read_text())
    dlup = json.loads(Path(args.dist_lupin).read_text())

    groups = defaultdict(list)
    for p in pts:
        groups[f"{p['block']}_{p['year']}"].append(p)

    jobdir = Path(args.job_dir)
    tags = sorted(d.name for d in jobdir.iterdir() if d.is_dir())
    print(f"{len(tags)} job rasters, {len(pts)} points total", flush=True)

    projections: dict[str, object] = {}
    cols: dict[str, list] = defaultdict(list)
    n_missing_group = n_outside = n_nan = 0

    for tag in tags:
        tifs = sorted(Path(jobdir / tag).glob("*.tif")) + \
               sorted(Path(jobdir / tag).glob("*.tiff"))
        if not tifs:
            print(f"  {tag}: no raster in job dir", flush=True)
            continue
        gpts = groups.get(tag)
        if not gpts:
            n_missing_group += 1
            continue

        with rasterio.open(tifs[0]) as src:
            img = src.read().astype(np.float32)
            if src.nodata is not None:
                img[img == src.nodata] = np.nan
            crs, tr = str(src.crs), src.transform
            H, W = src.height, src.width

        if img.shape[0] != len(FEATURE_NAMES):
            print(f"  {tag}: {img.shape[0]} bands, expected "
                  f"{len(FEATURE_NAMES)} — skipping", flush=True)
            continue

        if crs not in projections:
            projections[crs] = Transformer.from_crs(4326, crs, always_xy=True)
        to_img = projections[crs]

        kept = 0
        for p in gpts:
            px, py = to_img.transform(p["lon"], p["lat"])
            row, col = rt.rowcol(tr, px, py)
            if not (0 <= row < H and 0 <= col < W):
                n_outside += 1
                continue
            vec = img[:, row, col]
            if not np.isfinite(vec).all():
                n_nan += 1
                continue
            cols["X"].append(vec.astype(np.float32))
            cols["y"].append(p["label"])
            cols["lat"].append(p["lat"]); cols["lon"].append(p["lon"])
            cols["year"].append(p["year"]); cols["month"].append(p["month"])
            cols["block"].append(p["block"])
            k = f"{p['lon']:.6f},{p['lat']:.6f}"
            cols["dist_lupin"].append(dlup.get(k, np.nan))
            cols["dist_road"].append(road.get(k, np.nan))
            kept += 1
        print(f"  {tag}: {kept}/{len(gpts)} pts ({crs})", flush=True)

    if not cols.get("X"):
        print("NO DATA: no point fell inside any raster", file=sys.stderr)
        raise SystemExit(2)

    X = np.asarray(cols.pop("X"), dtype=np.float32)
    arrs = {k: np.asarray(v) for k, v in cols.items()}
    yv = arrs["y"].astype(int)
    print(f"total: X={X.shape}, pos={int(yv.sum())}, "
          f"neg={int((yv == 0).sum())}")
    print(f"dropped: {n_outside} outside raster, {n_nan} non-finite, "
          f"{n_missing_group} job dirs with no matching points")
    np.savez_compressed(args.out, X=X, **arrs)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
