#!/usr/bin/env python3
"""AI2's historical-scene rule, replicated independently on MPC (dossier section b).

rslp ``predict_pipeline.setup_dataset_with_scene_ids``: search
``[t0 - 60 d, t0 - 30 d]`` for products INTERSECTING the target footprint and
keep the one with the largest intersection area; no orbit-direction filter.
Uses pystac-client directly (not the fetch code) so it cross-checks the fetch.

    python scripts/marine/s1_port/pick_historical.py --out historical_selection.json
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import _record as rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--target-id", default=rec.TARGET_ID)
    ap.add_argument("--out", type=rec.Path)
    a = ap.parse_args()

    from pystac_client import Client
    from shapely.geometry import shape

    cat = Client.open(rec.MPC_STAC)
    tgt = cat.get_collection("sentinel-1-grd").get_item(a.target_id)
    tgeom = shape(tgt.geometry)
    t0 = datetime.fromisoformat(tgt.properties["start_datetime"].replace("Z", "+00:00"))
    lo, hi = t0 - timedelta(days=60), t0 - timedelta(days=30)
    rows = []
    for it in cat.search(collections=["sentinel-1-grd"], intersects=tgt.geometry,
                         datetime=f"{lo.isoformat()}/{hi.isoformat()}").items():
        p = it.properties
        if (p.get("sar:instrument_mode") != "IW" or "GRDH" not in it.id
                or sorted(p.get("sar:polarizations", [])) != ["VH", "VV"]):
            continue
        rows.append([tgeom.intersection(shape(it.geometry)).area / tgeom.area, it.id,
                     p["datetime"], p["sat:orbit_state"], p["sat:relative_orbit"]])
    rows.sort(reverse=True)
    rec.write_json(a.out, {"target": a.target_id, "target_start": tgt.properties["start_datetime"],
                           "hist_window": [lo.isoformat(), hi.isoformat()],
                           "candidates": rows, "best": rows[0] if rows else None})


if __name__ == "__main__":
    main()
