#!/usr/bin/env python3
"""Fetch Sentinel-2 TCI + Sentinel-1 GRD time series over the Strait of Hormuz.

Discovers all available scenes from CDSE STAC catalog, then fetches each
via Sentinel Hub Process API with proper radiometric calibration:
  - S2: 6-band reflectance + SCL + TCI image
  - S1: VV+VH linear σ⁰ GRD (orthorectified, SIGMA0_ELLIPSOID)

Output structure:
  leo-constellation/simulations/hormuz_data/
    s2/
      2026-02-23_spectral.tif   (6, H, W) float32 reflectance
      2026-02-23_scl.tif        (H, W) uint8
      2026-02-23_tci.jpg        TCI true-color
    s1/
      2026-02-20_vv_vh.tif      (2, H, W) float32 linear σ⁰
    dates.json                  manifest of all fetched scenes
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imint.training.cdse_s2 import fetch_s2_scene_wgs84
from imint.training.cdse_s1 import fetch_s1_scene_wgs84
from imint.training.cdse_vpp import _get_token

# ── Config ────────────────────────────────────────────────────────
BBOX = [54.0, 25.0, 58.0, 27.5]  # west, south, east, north (WGS84)
WIDTH, HEIGHT = 2048, 1280

TIME_FROM = "2026-02-01"
TIME_TO = datetime.now().strftime("%Y-%m-%d")  # today

# Output
LEO_DIR = Path(__file__).resolve().parent.parent.parent / "leo-constellation" / "simulations"
OUT_DIR = LEO_DIR / "hormuz_data"
S2_DIR = OUT_DIR / "s2"
S1_DIR = OUT_DIR / "s1"



# ── Date Generation ──────────────────────────────────────────────
def generate_dates(start: str, end: str, interval_days: int = 1) -> list[str]:
    """Generate all dates between start and end."""
    from datetime import date
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    dates = []
    d = s
    while d <= e:
        dates.append(d.isoformat())
        d += timedelta(days=interval_days)
    return dates


def s2_revisit_dates(start: str, end: str) -> list[str]:
    """Try every day — let the Process API reject dates with no data."""
    return generate_dates(start, end, interval_days=1)


def s1_revisit_dates(start: str, end: str) -> list[str]:
    """Try every day — S1 revisit varies by latitude."""
    return generate_dates(start, end, interval_days=1)


# ── Fetch Functions ──────────────────────────────────────────────
def fetch_s2_date(date: str) -> bool:
    """Fetch S2 L2A for a single date — spectral TIFF + SCL + TCI."""
    spectral_path = S2_DIR / f"{date}_spectral.tif"
    if spectral_path.exists():
        print(f"    S2 {date}: cached")
        return True

    west, south, east, north = BBOX
    result = fetch_s2_scene_wgs84(
        west, south, east, north,
        date=date,
        size_px=(HEIGHT, WIDTH),
        cloud_threshold=0.50,
        haze_threshold=0.30,  # desert/ocean has higher B02 than Scandinavia
    )
    if result is None:
        print(f"    S2 {date}: rejected (cloud/haze/nodata)")
        return False

    spectral, scl, cloud_frac = result

    # Save spectral TIFF
    import tifffile
    tifffile.imwrite(str(spectral_path), spectral)

    # Save SCL
    scl_path = S2_DIR / f"{date}_scl.tif"
    tifffile.imwrite(str(scl_path), scl)

    # Save TCI
    rgb = np.stack([spectral[2], spectral[1], spectral[0]], axis=-1)  # B04, B03, B02
    tci = (rgb * 2.5 * 255).clip(0, 255).astype(np.uint8)
    tci_path = S2_DIR / f"{date}_tci.jpg"
    Image.fromarray(tci).save(str(tci_path), quality=92)

    size_mb = spectral_path.stat().st_size / 1024 / 1024
    print(f"    S2 {date}: OK  cloud={cloud_frac:.0%}  {size_mb:.1f} MB")
    return True


def fetch_s1_date(date: str) -> bool:
    """Fetch S1 GRD for a single date — VV+VH TIFF."""
    sar_path = S1_DIR / f"{date}_vv_vh.tif"
    if sar_path.exists():
        print(f"    S1 {date}: cached")
        return True

    west, south, east, north = BBOX
    result = fetch_s1_scene_wgs84(
        west, south, east, north,
        date=date,
        size_px=(HEIGHT, WIDTH),
        output_db=False,  # linear σ⁰ — better for ship detection
        nodata_threshold=0.60,
    )
    if result is None:
        print(f"    S1 {date}: rejected (nodata)")
        return False

    sar, orbit = result

    import tifffile
    tifffile.imwrite(str(sar_path), sar)

    size_mb = sar_path.stat().st_size / 1024 / 1024
    print(f"    S1 {date}: OK  orbit={orbit}  {size_mb:.1f} MB")
    return True


# ── Main ──────────────────────────────────────────────────────────
def main():
    S2_DIR.mkdir(parents=True, exist_ok=True)
    S1_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Hormuz time series: {TIME_FROM} to {TIME_TO}")
    print(f"  Bbox: {BBOX} (WGS84)")
    print(f"  Size: {WIDTH}x{HEIGHT} px")
    print(f"  Output: {OUT_DIR}\n")

    # ── 1. Generate candidate dates ─────────────────────────
    # S2 revisit: ~3 days at 26°N (S2A+S2B+S2C combined)
    s2_dates = s2_revisit_dates(TIME_FROM, TIME_TO)
    print(f"S2 candidate dates: {len(s2_dates)} (every 3 days)")

    # S1 revisit: ~6 days at this latitude
    s1_dates = s1_revisit_dates(TIME_FROM, TIME_TO)
    print(f"S1 candidate dates: {len(s1_dates)} (every 6 days)\n")

    # ── 3. Fetch S2 scenes ────────────────────────────────────
    print(f"Fetching {len(s2_dates)} S2 scenes:")
    s2_ok = []
    for date in s2_dates:
        if fetch_s2_date(date):
            s2_ok.append(date)
        time.sleep(0.5)  # rate limit

    # ── 4. Fetch S1 scenes ────────────────────────────────────
    print(f"\nFetching {len(s1_dates)} S1 scenes:")
    s1_ok = []
    for date in s1_dates:
        if fetch_s1_date(date):
            s1_ok.append(date)
        time.sleep(0.5)

    # ── 5. Save manifest ──────────────────────────────────────
    manifest = {
        "bbox_wgs84": BBOX,
        "size_px": [WIDTH, HEIGHT],
        "time_range": [TIME_FROM, TIME_TO],
        "s2": {
            "dates_discovered": s2_dates,
            "dates_fetched": s2_ok,
            "bands": ["B02", "B03", "B04", "B8A", "B11", "B12"],
            "format": "float32 reflectance [0,1]",
            "tci_formula": "clip(reflectance × 2.5 × 255, 0, 255)",
        },
        "s1": {
            "dates_discovered": s1_dates,
            "dates_fetched": s1_ok,
            "bands": ["VV", "VH"],
            "format": "float32 linear σ⁰",
            "processing": "SIGMA0_ELLIPSOID, orthorectified, COPERNICUS_30 DEM",
        },
    }
    manifest_path = OUT_DIR / "dates.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # ── 6. Copy latest S2 TCI as globe overlay ────────────────
    if s2_ok:
        latest_tci = S2_DIR / f"{s2_ok[-1]}_tci.jpg"
        if latest_tci.exists():
            import shutil
            shutil.copy2(str(latest_tci), str(LEO_DIR / "hormuz_s2.jpg"))
            print(f"\nGlobe overlay updated: {s2_ok[-1]} → hormuz_s2.jpg")

    print(f"\n{'='*50}")
    print(f"S2: {len(s2_ok)}/{len(s2_dates)} dates fetched")
    print(f"S1: {len(s1_ok)}/{len(s1_dates)} dates fetched")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
