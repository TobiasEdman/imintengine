"""Rebuild the S1 geocoding fixtures (run by hand; the tests only read the outputs).

Inputs (the 2026-10-20 offset diagnosis, outside the repo):
    --diagnosis DIR   rows_step3.json (per detection: slice, radar row/col from the
                      chain-1 TPS inverse of the ring centre), rows_step7.json (raw
                      DMA AIS at scene time: antenna lon/lat, true heading, A/B/C/D,
                      SOG), annotation/<scene>_vv_annotation.xml
    --headers DIR     <scene>/vv_header.json as written by
                      ``imint.fetch._s1_scene_header`` (raw GCPs + incidence angles)

Outputs (this directory):
    headers/<scene>_vv_header.json.gz      four scenes (06-27 dsc slices 1+2, 06-27 asc, 06-28)
    annotation/<scene>_vv_trimmed.xml.gz   06-28 S1D and 06-27 S1C asc: adsHeader,
                                           productInformation, orbitList,
                                           imageInformation, geolocationGrid
    anchored_ships.json                    per layer, slow ships (SOG < 1 kn) with
                                           true heading and A/B/C/D: frozen radar
                                           (row, col) of the detected hull and the
                                           AIS hull centre at scene time

Hull centre = antenna + (A - B)/2 along the bow + (D - C)/2 to starboard (true
heading). The radar coordinates do not depend on the GCPs: they locate the
detection in the radar raster; only their geocoding changes with the chain.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCENES = {
    "dsc0627_s1": "S1C_IW_GRDH_1SDV_20260627T053911_20260627T053936_008287_01063E",
    "dsc0627_s2": "S1C_IW_GRDH_1SDV_20260627T053936_20260627T054001_008287_01063E",
    "asc0627": "S1C_IW_GRDH_1SDV_20260627T165257_20260627T165322_008294_010682",
    "dsc0628": "S1D_IW_GRDH_1SDV_20260628T053135_20260628T053200_003432_0060CF",
}
LAYERS = {
    "dsc0627": {"scene_time_utc": "2026-06-27T05:39:37.623515Z", "pass": "descending"},
    "asc0627": {"scene_time_utc": "2026-06-27T16:53:05.704017Z", "pass": "ascending"},
    "dsc0628": {"scene_time_utc": "2026-06-28T05:31:40.547748Z", "pass": "descending"},
}
ORACLE_SCENES = ("dsc0628", "asc0627")
_KEEP = ("adsHeader", "generalAnnotation/productInformation", "generalAnnotation/orbitList",
         "imageAnnotation/imageInformation", "geolocationGrid")


def _trim_annotation(src: Path) -> bytes:
    root = ET.parse(src).getroot()
    out = ET.Element(root.tag)
    for path in _KEEP:
        parent = out
        for part in path.split("/")[:-1]:
            parent = parent.find(part) if parent.find(part) is not None \
                else ET.SubElement(parent, part)
        parent.append(root.find(path))
    return ET.tostring(out, encoding="utf-8", xml_declaration=True)


def _hull_centre(ship: dict) -> tuple[float, float]:
    from pyproj import Geod

    h = math.radians(ship["heading"])
    fwd, stb = (ship["A"] - ship["B"]) / 2, (ship["D"] - ship["C"]) / 2
    de = fwd * math.sin(h) + stb * math.cos(h)
    dn = fwd * math.cos(h) - stb * math.sin(h)
    lon, lat, _ = Geod(ellps="WGS84").fwd(ship["ais_lon"], ship["ais_lat"],
                                          math.degrees(math.atan2(de, dn)), math.hypot(de, dn))
    return float(lon), float(lat)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--diagnosis", type=Path, required=True)
    ap.add_argument("--headers", type=Path, required=True)
    args = ap.parse_args()
    from imint import fetch

    (HERE / "headers").mkdir(exist_ok=True)
    (HERE / "annotation").mkdir(exist_ok=True)
    headers = {}
    for key, sid in SCENES.items():
        h = json.loads((args.headers / sid / "vv_header.json").read_text())
        assert h["scene_id"] == sid and len(h["gcp_incidence_deg"]) == len(h["gcps"])
        headers[key] = h
        with gzip.open(HERE / "headers" / f"{sid}_vv_header.json.gz", "wt") as f:
            json.dump(h, f)
    for key in ORACLE_SCENES:
        sid = SCENES[key]
        xml = _trim_annotation(args.diagnosis / "annotation" / f"{sid}_vv_annotation.xml")
        with gzip.GzipFile(HERE / "annotation" / f"{sid}_vv_trimmed.xml.gz", "wb", mtime=0) as f:
            f.write(xml)

    step3 = json.loads((args.diagnosis / "rows_step3.json").read_text())
    step7 = json.loads((args.diagnosis / "rows_step7.json").read_text())
    out = {}
    for layer, meta in LAYERS.items():
        by_mmsi = {r["mmsi"]: r for r in step3[layer] if r.get("mmsi")}
        ships = []
        for s in step7[layer]:
            if s["heading"] is None or not all(np.isfinite(s[k]) for k in "ABCD") \
                    or s["A"] + s["B"] <= 0 or s["sog"] >= 1.0:
                continue
            det = by_mmsi[s["mmsi"]]
            # The frozen radar coordinates must reproduce the chain-1 ring centre.
            tps = fetch._s1_gcp_tps(headers[det["slice"]]["gcps"])
            (x,), (y,) = tps.xy([det["row"]], [det["col"]], offset="ul")
            assert abs(x - det["lon"]) < 1e-7 and abs(y - det["lat"]) < 1e-7, s["mmsi"]
            hull_lon, hull_lat = _hull_centre(s)
            ships.append({
                "mmsi": s["mmsi"], "scene_id": SCENES[det["slice"]],
                "row": det["row"], "col": det["col"],
                "chain1_lon": det["lon"], "chain1_lat": det["lat"],
                "hull_lon": hull_lon, "hull_lat": hull_lat,
                "antenna_lon": s["ais_lon"], "antenna_lat": s["ais_lat"],
                "heading_deg": s["heading"], "A": s["A"], "B": s["B"], "C": s["C"], "D": s["D"],
                "sog_kn": s["sog"], "ais_gap_s": s["gap_s"],
            })
        out[layer] = {**meta, "ships": ships}
        print(layer, len(ships), "ships")
    (HERE / "anchored_ships.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
