"""Parse per-band, per-DETECTOR GPS times from MTD_DS.xml.

The previous version took median across detectors, which mixed across-track
slot positions and gave Δt~0. The CORRECT computation is:

  for a given detector (= specific across-track slot), Δt(B02 → B08)
  = GPS_TIME[bandId=7, detector=N] - GPS_TIME[bandId=1, detector=N]

We report Δt per detector, plus Δt for detector 10 (which covers our
Öckerö-skärgården AOI 57.71809°N, 11.66456°E per MTD_TL.xml
viewing-incidence-angle grids).
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SAFE_CACHE = REPO_ROOT / "demos" / "aircraft_parallax" / "cache_l1c"

BAND_ID_TO_NAME = {
    0: "B01", 1: "B02", 2: "B03", 3: "B04", 4: "B05", 5: "B06",
    6: "B07", 7: "B08", 8: "B8A", 9: "B09", 10: "B10", 11: "B11", 12: "B12",
}


def find_safe() -> Path:
    return sorted(SAFE_CACHE.glob("S2*_MSIL1C_20260111*.SAFE"))[0]


def parse_iso(text: str) -> datetime:
    txt = text.strip().rstrip("Z")
    try:
        return datetime.strptime(txt, "%Y-%m-%dT%H:%M:%S.%f")
    except ValueError:
        return datetime.strptime(txt, "%Y-%m-%dT%H:%M:%S")


def main():
    safe = find_safe()
    mtd_ds = next((safe / "DATASTRIP").rglob("MTD_DS.xml"))
    print(f"MTD_DS: {mtd_ds.relative_to(safe)}\n")

    tree = ET.parse(mtd_ds)
    root = tree.getroot()

    # Build a nested map: { band_id: { detector_id: gps_time } }
    times = {}
    for elem in root.iter():
        if elem.tag.split("}")[-1] != "Band_Time_Stamp":
            continue
        band_id = int(elem.get("bandId"))
        times.setdefault(band_id, {})
        for det in elem:
            if det.tag.split("}")[-1] != "Detector":
                continue
            det_id = int(det.get("detectorId"))
            for child in det:
                if child.tag.split("}")[-1] == "GPS_TIME":
                    times[band_id][det_id] = parse_iso(child.text)

    print(f"Bands found: {sorted(times.keys())}")
    print(f"Detectors per band: {sorted(set(d for b in times.values() for d in b))}\n")

    # Show full Δt table: rows = detectors, cols = bands B02/B03/B04/B08
    bands_of_interest = [("B02", 1), ("B03", 2), ("B04", 3), ("B08", 7)]
    print("Per-detector Δt relative to B02 (seconds):\n")
    print(f"{'detector':>9}  {'B02 GPS_TIME':<28}  " + "  ".join(f"{n:>10}" for n, _ in bands_of_interest[1:]))
    print("-" * 90)
    detectors = sorted(times[1].keys())
    for det_id in detectors:
        b02_t = times[1].get(det_id)
        if b02_t is None:
            continue
        line = f"{det_id:>9}  {b02_t.isoformat():<28}"
        for name, bid in bands_of_interest[1:]:
            t = times.get(bid, {}).get(det_id)
            if t is None:
                line += f"  {'—':>10}"
                continue
            dt = (t - b02_t).total_seconds()
            line += f"  {dt:+10.4f}"
        print(line)

    # AOI 57.71809°N, 11.66456°E (T32VPK) is covered by detector 10 (per MTD_TL.xml)
    # Show the timeline for detector 10 specifically (even detector → push-pull reverse order)
    print(f"\n=== Detector 10 (covers Öckerö-skärgården AOI in T32VPK) ===\n")
    det7 = {bid: times[bid].get(10) for bid in times}
    print(f"{'Band':<6} {'GPS_TIME (UTC)':<30} {'Δt vs B02 (s)':>16}")
    print("-" * 56)
    ref = det7.get(1)
    if ref is None:
        sys.exit("No B02 detector 10 time")
    for bid in sorted(det7):
        if det7[bid] is None:
            continue
        dt = (det7[bid] - ref).total_seconds()
        print(f"{BAND_ID_TO_NAME[bid]:<6} {det7[bid].isoformat():<30} {dt:+16.4f}")

    # Highlight the four bands of interest with detector-7 Δt
    print(f"\n=== Δt for parallax measurement (detector 10) ===")
    for name, bid in bands_of_interest:
        if det7.get(bid):
            dt = (det7[bid] - ref).total_seconds()
            print(f"  {name}: Δt = {dt:+.4f} s")

    print(f"\n=== Inter-band steps (detector 10) ===")
    pairs = [("B02→B03", 1, 2), ("B03→B04", 2, 3), ("B04→B08", 3, 7), ("B02→B08", 1, 7)]
    for label, a, b in pairs:
        if det7.get(a) and det7.get(b):
            dt = (det7[b] - det7[a]).total_seconds()
            print(f"  {label}: {dt:+.4f} s")


if __name__ == "__main__":
    main()
