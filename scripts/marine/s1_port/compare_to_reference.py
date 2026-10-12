#!/usr/bin/env python3
"""Compare an AI2S1VesselAnalyzer run with AI2's reference.geojson (dossier tables b-e).

Matches mutual nearest neighbours within ``--max-m`` on the FULL sets, then
reports what lies inside ``--bbox``: a pair straddling the bbox edge is one
match, not two misses. ``run.json`` is ``{"outputs": {"regions": [...]}, ...}``
as written by run_analyzer_on.py or ``scripts/detect_vessels_s1.py --run-json``;
regions carry the AI2 point as lon/lat.

    python scripts/marine/s1_port/compare_to_reference.py run_c.json "C: production" \\
        --record <record> --bbox 10.50 57.55 11.70 58.05 --out cmp_run_c.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import _record as rec


def quantiles(values) -> dict | None:
    v = np.abs(np.asarray(values, dtype=float))
    if not len(v):
        return None
    return {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
            "max": float(v.max())}


def circular_deg(d: float) -> float:
    return abs((d + 180) % 360 - 180)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=rec.Path)
    ap.add_argument("label")
    rec.add_record_arg(ap)
    ap.add_argument("--bbox", nargs=4, type=float)
    ap.add_argument("--max-m", type=float, default=100.0)
    ap.add_argument("--out", type=rec.Path)
    a = ap.parse_args()

    from imint.utils import haversine_m

    def inside(lon: float, lat: float) -> bool:
        if not a.bbox:
            return True
        w, s, e, n = a.bbox
        return w <= lon <= e and s <= lat <= n

    ref = rec.reference_properties(a.record)
    mine = json.loads(a.run.read_text())["outputs"]["regions"]
    R = np.array([[p["longitude"], p["latitude"]] for p in ref]).reshape(-1, 2)
    M = np.array([[r["lon"], r["lat"]] for r in mine]).reshape(-1, 2)
    D = (haversine_m(M[:, None, 0], M[:, None, 1], R[None, :, 0], R[None, :, 1])
         if len(M) and len(R) else np.zeros((len(M), len(R))))
    pairs = []
    for i in range(len(M) if D.size else 0):
        j = int(np.argmin(D[i]))
        if int(np.argmin(D[:, j])) == i and D[i, j] <= a.max_m and (
                inside(*M[i]) or inside(*R[j])):
            pairs.append((i, j, float(D[i, j])))
    rows = []
    for i, j, d in pairs:
        ma, pa = mine[i]["attributes"], ref[j]["attributes"]
        rows.append(dict(dist_m=d, dscore=mine[i]["score"] - ref[j]["score"],
                         dlen=ma["length_m"] - pa["length"], dwid=ma["width_m"] - pa["width"],
                         dspd=ma["speed_knots"] - pa["speed"],
                         dhdg=circular_deg(ma["heading_deg"] - pa["heading"]),
                         same_type=ma["vessel_type"] == pa["vessel_type"], ref_len=pa["length"]))
    matched_mine, matched_ref = {i for i, _, _ in pairs}, {j for _, j, _ in pairs}
    summary = {
        "label": a.label, "bbox": a.bbox,
        "n_ref": sum(inside(p["longitude"], p["latitude"]) for p in ref),
        "n_mine": sum(inside(r["lon"], r["lat"]) for r in mine), "n_matched": len(pairs),
        "offset_m": quantiles([r["dist_m"] for r in rows]),
        "n_offset_0": int(sum(r["dist_m"] < 1e-3 for r in rows)),
        "abs_dscore": quantiles([r["dscore"] for r in rows]),
        "abs_dlength_m": quantiles([r["dlen"] for r in rows]),
        "abs_dwidth_m": quantiles([r["dwid"] for r in rows]),
        "abs_dspeed_kn": quantiles([r["dspd"] for r in rows]),
        "abs_dheading_deg": quantiles([r["dhdg"] for r in rows]),
        "rel_dlength": quantiles([r["dlen"] / r["ref_len"] for r in rows]),
        "type_agree": int(sum(r["same_type"] for r in rows)),
        "unmatched_mine": [
            dict(lon=m["lon"], lat=m["lat"], score=m["score"], point_px=m["point_px"],
                 nearest_ref_m=float(D[i].min()) if D.shape[1] else None,
                 length_m=(m.get("attributes") or {}).get("length_m"))
            for i, m in enumerate(mine) if i not in matched_mine and inside(m["lon"], m["lat"])],
        "unmatched_ref": [
            dict(lon=p["longitude"], lat=p["latitude"], col=p["col"], row=p["row"],
                 score=p["score"], nearest_mine_m=float(D[:, j].min()) if D.shape[0] else None,
                 length_m=p["attributes"]["length"])
            for j, p in enumerate(ref)
            if j not in matched_ref and inside(p["longitude"], p["latitude"])],
    }
    rec.write_json(a.out, summary)


if __name__ == "__main__":
    main()
