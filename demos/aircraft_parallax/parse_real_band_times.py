"""Parse actual per-band GPS sensing times from MTD_DS.xml.

This is the AUTHORITATIVE source for band timing — not the hardcoded
nominal_offset_s dictionary I used earlier. Each band's Band_Time_Stamp
contains GPS_TIME entries per detector (12 detectors per VNIR band).

We compute, per band:
  - First and last GPS_TIME among the detectors covering this AOI
  - Median GPS_TIME (representative band acquisition time)

Then we report Δt(B02 → bX) for each band X.
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path


def median_dt(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        raise ValueError("empty")
    if n % 2 == 1:
        return s[n // 2]
    # Even: take mean of two middle (in seconds since epoch)
    a, b = s[n // 2 - 1], s[n // 2]
    secs = (a.timestamp() + b.timestamp()) / 2
    return datetime.fromtimestamp(secs)

REPO_ROOT = Path(__file__).resolve().parents[2]
SAFE_CACHE = REPO_ROOT / "demos" / "aircraft_parallax" / "cache_l1c"

# Sentinel-2 band ID → physical band name (per ESA S2 PDD Table 4-1)
BAND_ID_TO_NAME = {
    0: "B01", 1: "B02", 2: "B03", 3: "B04", 4: "B05", 5: "B06",
    6: "B07", 7: "B08", 8: "B8A", 9: "B09", 10: "B10", 11: "B11", 12: "B12",
}


def find_safe() -> Path:
    return sorted(SAFE_CACHE.glob("S2*_MSIL1C_20260111*.SAFE"))[0]


def main():
    safe = find_safe()
    mtd_ds = next((safe / "DATASTRIP").rglob("MTD_DS.xml"))
    print(f"MTD_DS: {mtd_ds.relative_to(safe)}\n")

    tree = ET.parse(mtd_ds)
    root = tree.getroot()

    # Build a map band_id -> list of GPS_TIME (datetime)
    band_times = {}
    for elem in root.iter():
        tag = elem.tag.split('}')[-1]   # strip namespace
        if tag == "Band_Time_Stamp":
            band_id = int(elem.get("bandId"))
            times = []
            for child in elem.iter():
                ctag = child.tag.split('}')[-1]
                if ctag == "GPS_TIME" and child.text:
                    # Strip trailing 'Z' and parse
                    txt = child.text.strip().rstrip("Z")
                    try:
                        ts = datetime.strptime(txt, "%Y-%m-%dT%H:%M:%S.%f")
                    except ValueError:
                        ts = datetime.strptime(txt, "%Y-%m-%dT%H:%M:%S")
                    times.append(ts)
            band_times[band_id] = times

    if not band_times:
        sys.exit("No Band_Time_Stamp found in MTD_DS.xml")

    print(f"Found Band_Time_Stamp for {len(band_times)} bands.\n")

    # Reference: B02 (band_id 1) median time
    ref_band_id = 1   # B02
    ref_med = median_dt(band_times[ref_band_id])
    print(f"Reference: B02 median GPS_TIME = {ref_med.isoformat()}\n")

    print(f"{'Band':<6} {'min(GPS)':<28} {'med(GPS)':<28} {'max(GPS)':<28} {'Δt vs B02 (s)':>16}")
    print("-" * 110)
    rows = []
    for band_id in sorted(band_times):
        name = BAND_ID_TO_NAME.get(band_id, f"id={band_id}")
        times = sorted(band_times[band_id])
        med = median_dt(times)
        dt = (med - ref_med).total_seconds()
        rows.append((band_id, name, dt, med))
        print(f"{name:<6} {times[0].isoformat():<28} {med.isoformat():<28} {times[-1].isoformat():<28} {dt:+16.4f}")

    # Now show only the 4 bands we care about with Δt relative to B02
    print(f"\nShort summary (Δt relative to B02):")
    interesting = ["B02", "B03", "B04", "B08"]
    sel = {name: dt for _bid, name, dt, _med in rows if name in interesting}
    for name in interesting:
        print(f"  {name}: Δt = {sel.get(name, float('nan')):+.4f} s")

    # Pixel offset prediction at given speeds
    if "B02" in sel and "B08" in sel:
        dt_b02_b08 = sel["B08"] - sel["B02"]
        print(f"\nΔt(B02→B08) ACTUAL from MTD_DS: {dt_b02_b08:+.4f} s")
        print(f"Compare to my hardcoded value:    +1.0050 s  (off by {dt_b02_b08-1.005:+.4f} s)")

    if "B02" in sel and "B03" in sel:
        d23 = sel["B03"] - sel["B02"]
        d34 = sel["B04"] - sel["B03"]
        d48 = sel["B08"] - sel["B04"]
        print(f"\nInter-band steps:")
        print(f"  B02 → B03: {d23:+.4f} s")
        print(f"  B03 → B04: {d34:+.4f} s")
        print(f"  B04 → B08: {d48:+.4f} s")
        print(f"User's claim 'jämna avstånd B02-B03-B04, B08 separat' → ", end="")
        if abs(d23 - d34) < 0.05 and abs(d48 - d34) > 0.1:
            print("CONFIRMED.")
        else:
            print("not exactly — see above values.")


if __name__ == "__main__":
    main()
