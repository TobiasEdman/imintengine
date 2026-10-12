#!/usr/bin/env python3
"""MPC Sentinel-1 catalogue facts behind the dossier's scene choice (sections b, c).

``search``: every ``sentinel-1-grd`` item over the bbox in a datetime range, with
its AOI cover. ``inspect``: product properties, raster size/dtype/nodata/GCPs and
TIFF tags of given items. ``footprint``: the MPC item geometry against the
CDSE OData ``GeoFootprint`` AI2 reads (free, anonymous catalogue; no PU).

    python scripts/marine/s1_port/mpc_scenes.py search 2026-06-27/2026-06-29
    python scripts/marine/s1_port/mpc_scenes.py inspect <item id> ...
    python scripts/marine/s1_port/mpc_scenes.py footprint <item id> ...
"""
from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request

import _record as rec

ODATA = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter="


def search(dt: str, bbox) -> list[dict]:
    from pystac_client import Client
    from shapely.geometry import box, shape

    aoi = box(*bbox)
    out = []
    for it in Client.open(rec.MPC_STAC).search(collections=["sentinel-1-grd"], bbox=list(bbox),
                                               datetime=dt).items():
        p = it.properties
        out.append(dict(id=it.id, dt=p.get("datetime"), orbit=p.get("sat:orbit_state"),
                        rel=p.get("sat:relative_orbit"), abs=p.get("sat:absolute_orbit"),
                        mode=p.get("sar:instrument_mode"), pol=p.get("sar:polarizations"),
                        platform=p.get("platform"),
                        aoi_cover=round(shape(it.geometry).intersection(aoi).area / aoi.area, 3)))
    return sorted(out, key=lambda o: o["dt"])


def inspect(ids: list[str]) -> dict:
    import planetary_computer as pc
    import rasterio
    from pystac_client import Client

    col = Client.open(rec.MPC_STAC, modifier=pc.sign_inplace).get_collection("sentinel-1-grd")
    keys = ["s1:product_timeliness", "s1:resolution", "s1:orbit_source", "s1:shape",
            "processing:software", "start_datetime", "end_datetime", "sat:orbit_state",
            "sat:relative_orbit", "sat:absolute_orbit"]
    out = {}
    for iid in ids:
        it = col.get_item(iid)
        entry = {"manifest": it.assets["safe-manifest"].href.split("?")[0],
                 **{k: it.properties.get(k) for k in keys}}
        for pol in rec.POLS:
            with rasterio.open(it.assets[pol].href) as src:
                gcps, gcrs = src.gcps
                entry[pol] = dict(width=src.width, height=src.height, dtype=src.dtypes[0],
                                  nodata=src.nodata, crs=str(src.crs), n_gcps=len(gcps),
                                  gcp_crs=str(gcrs), tags=dict(list(src.tags().items())[:8]))
        out[iid] = entry
    return out


def footprint(ids: list[str]) -> dict:
    from pystac_client import Client
    from shapely.geometry import shape

    col = Client.open(rec.MPC_STAC).get_collection("sentinel-1-grd")
    out = {}
    for iid in ids:
        it = col.get_item(iid)
        q = ODATA + urllib.parse.quote(f"startswith(Name,'{iid}')")
        products = json.load(urllib.request.urlopen(q, timeout=60))["value"]
        g = shape(it.geometry)
        out[iid] = [{"odata_name": p["Name"],
                     "symmetric_difference_deg2": g.symmetric_difference(
                         shape(p["GeoFootprint"])).area,
                     "odata_start": p["ContentDate"]["Start"],
                     "mpc_start": it.properties["start_datetime"]} for p in products]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("datetime_range")
    s.add_argument("--bbox", nargs=4, type=float, default=rec.SKAGEN_BBOX)
    for name in ("inspect", "footprint"):
        sub.add_parser(name).add_argument("ids", nargs="+")
    for p in sub.choices.values():
        p.add_argument("--out", type=rec.Path)
    a = ap.parse_args()
    result = {"search": lambda: search(a.datetime_range, a.bbox),
              "inspect": lambda: inspect(a.ids), "footprint": lambda: footprint(a.ids)}[a.cmd]()
    rec.write_json(a.out, result)


if __name__ == "__main__":
    main()
