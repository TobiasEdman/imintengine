#!/usr/bin/env python3
"""How far does the TPS through the crop's GCP subset move pixels vs all 210 GCPs? (dossier b)

Compares ``inputs/target_vv.tif`` (GCP subset) with ``inputs/target_vv_allgcps.tif``
(same pixels, all GCPs) over the crop and inside the bbox, and reports the UTM
zone of each full scene's first GCP vs its GCP-bbox centroid (needs MPC access).

    python scripts/marine/s1_port/tps_subset_check.py --record <record> --out tps.json
"""
from __future__ import annotations

import argparse

import numpy as np

import _record as rec


def zone(lon: float) -> int:
    return int((lon + 180) // 6) + 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    rec.add_record_arg(ap)
    ap.add_argument("--bbox", nargs=4, type=float, default=rec.SKAGEN_BBOX)
    ap.add_argument("--out", type=rec.Path)
    a = ap.parse_args()

    import planetary_computer as pc
    import rasterio
    from pyproj import Transformer
    from pystac_client import Client
    from rasterio.transform import GCPTransformer
    from shapely.geometry import Point, box

    inputs = a.record / "inputs"
    with rasterio.open(inputs / "target_vv.tif") as s, \
            rasterio.open(inputs / "target_vv_allgcps.tif") as t:
        g_sub, g_all, w, h = s.gcps[0], t.gcps[0], s.width, s.height
    rr, cc = np.meshgrid(np.linspace(0, h - 1, 81), np.linspace(0, w - 1, 81), indexing="ij")
    utm = Transformer.from_crs(4326, 32632, always_xy=True)

    def en(gcps, tps=True):
        x, y = GCPTransformer(gcps, tps=tps).xy(rr.ravel(), cc.ravel())
        return (np.array(x), np.array(y)), utm.transform(np.array(x), np.array(y))

    (xa, ya), (ea, na) = en(g_sub)
    _, (eb, nb) = en(g_all)
    _, (ep, npoly) = en(g_all, tps=False)
    d = np.hypot(ea - eb, na - nb)
    dp = np.hypot(ea - ep, na - npoly)
    inside = np.array([box(*a.bbox).contains(Point(x, y)) for x, y in zip(xa, ya)])
    out = {"crop_m": {"max": float(d.max()), "p99": float(np.percentile(d, 99)),
                      "mean": float(d.mean())},
           "bbox_m": {"n": int(inside.sum()), "max": float(d[inside].max()),
                      "p99": float(np.percentile(d[inside], 99)), "mean": float(d[inside].mean())},
           "polynomial_all_vs_tps_subset_m": {"max": float(dp.max()), "mean": float(dp.mean())},
           "full_scenes": {}}
    col = Client.open(rec.MPC_STAC, modifier=pc.sign_inplace).get_collection("sentinel-1-grd")
    for iid in (rec.TARGET_ID, rec.HISTORICAL_ID):
        with rasterio.open(col.get_item(iid).assets["vv"].href) as s:
            g, _ = s.gcps
        lons = [p.x for p in g]
        centre = (min(lons) + max(lons)) / 2
        out["full_scenes"][iid] = {"first_gcp": [g[0].x, g[0].y], "first_gcp_zone": zone(g[0].x),
                                   "gcp_bbox_centroid_lon": centre, "centroid_zone": zone(centre)}
    rec.write_json(a.out, out)


if __name__ == "__main__":
    main()
