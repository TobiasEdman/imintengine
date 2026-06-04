#!/usr/bin/env python3
"""Probe actual SH Process SCL cloud fraction for top ranked candidates.

For each (bbox, window), runs the production rank chain and probes
``cdse_s2._prescreen_scl`` directly to see what AOI cloud fraction
the SCL band ACTUALLY reports — not just whether ``fetch_s2_scene``
returned None.

Tells us whether failures are:
  - GATE TOO STRICT  (actual SCL low, my gate too tight)
  - GENUINELY CLOUDY (actual SCL high — data is cloudy)
  - SOMETHING ELSE   (SCL fetch errors etc.)
"""
import os
import re
import sys
import time

ROOT = "/Users/tobiasedman/Developer/ImintEngine"
sys.path.insert(0, ROOT)

# Bypass load_env (which reads config/environments/dev.env) and inject
# the *root* .env values explicitly — that's what the rotate script
# updated, and it has the working creds.
for line in open(f"{ROOT}/.env"):
    m = re.match(
        r"^\s*(CDSE_CLIENT_ID|CDSE_CLIENT_SECRET)\s*=\s*(.*?)\s*$", line
    )
    if m:
        os.environ[m.group(1)] = m.group(2).strip().strip('"').strip("'")

from imint.training.cdse_s2 import _CRS_3006, _get_token, _prescreen_scl
from imint.training.optimal_fetch import era5_to_scl_gate, rank_stac_era5_candidates

# Three production-sized Swedish AOIs (5120m extent = 512 px × 10 m).
# Bboxes computed by snapping to the 10 m grid; WGS84 is bbox center
# converted via pyproj-equivalent math (good enough for STAC/ERA5).
TEST_AOIS = [
    {
        "name": "mid-Sweden farmland ~58.5°N",
        "bbox_3006": (414720, 6499720, 419840, 6504840),
        "wgs84": {"west": 11.96, "south": 58.49, "east": 12.05, "north": 58.54},
    },
    {
        "name": "central Sweden forest ~62°N",
        "bbox_3006": (519720, 6899720, 524840, 6904840),
        "wgs84": {"west": 14.40, "south": 62.20, "east": 14.50, "north": 62.25},
    },
    {
        "name": "north Sweden ~65°N",
        "bbox_3006": (619720, 7199720, 624840, 7204840),
        "wgs84": {"west": 17.50, "south": 65.00, "east": 17.62, "north": 65.05},
    },
]

WINDOWS = [
    ("growing-1 2022 (Apr-May)", "2022-04-01", "2022-05-31", False, "des"),
    ("growing-2 2022 (Jun-Jul)", "2022-06-01", "2022-07-31", False, "des"),
    ("growing-3 2022 (Aug-Sep)", "2022-08-01", "2022-09-15", False, "des"),
    ("autumn 2017 (earth-search)", "2017-08-15", "2017-10-31", True, "earth-search"),
]

SIZE_PX = 512  # production tile size — match real fetch_spectral cloud fraction


def probe_one(aoi, label, ds, de, is_autumn, backend):
    coords = aoi["wgs84"]
    b = aoi["bbox_3006"]
    try:
        ranked = rank_stac_era5_candidates(
            coords, ds, de,
            overpass_cloud_max=65.0 if is_autumn else 50.0,
            stac_backend=backend,
        )
    except Exception as e:
        print(f"    rank failed: {type(e).__name__}: {str(e)[:100]}")
        return []
    if not ranked:
        print("    (no ranked candidates)")
        return []
    print(f"    {len(ranked)} ranked. Probing top 3 with size_px={SIZE_PX}:")
    print(f"      {'date':12s}  {'ERA5%':>6}  {'gate':>6}  {'SCL_cloud':>10}  {'margin':>7}  verdict")
    try:
        token = _get_token()
    except Exception as e:
        print(f"    SH Process token failed: {e}")
        return []
    results = []
    for d, oc in ranked[:3]:
        t0 = time.time()
        try:
            _scl, cf = _prescreen_scl(
                b[0], b[1], b[2], b[3],
                d, SIZE_PX, SIZE_PX,
                token, _CRS_3006,  # OGC URI; "EPSG:3006" is silently rejected
                1.0,  # unused by _prescreen_scl
            )
        except Exception as e:
            print(f"      {d}  {oc:>6.1f}  {'-':>6}  {'ERR':>10}  -        {type(e).__name__}")
            continue
        dt = time.time() - t0
        g = era5_to_scl_gate(oc, is_autumn=is_autumn)
        margin = cf - g
        verdict = "PASS" if cf <= g else f"REJECT"
        print(
            f"      {d}  {oc:>6.1f}  {g:>6.3f}  {cf:>10.3f}  {margin:>+7.3f}  "
            f"{verdict}  ({dt:.1f}s)"
        )
        results.append((d, oc, g, cf, cf <= g))
    return results


def main():
    all_results = []
    for aoi in TEST_AOIS:
        print()
        print("=" * 92)
        print(f"AOI: {aoi['name']}")
        print("=" * 92)
        for label, ds, de, is_autumn, backend in WINDOWS:
            print(f"  WINDOW: {label}  ({ds}..{de}, backend={backend})")
            results = probe_one(aoi, label, ds, de, is_autumn, backend)
            for r in results:
                all_results.append((aoi["name"], label, *r))

    print()
    print("=" * 92)
    print("AGGREGATE")
    print("=" * 92)
    if not all_results:
        print("(no probe data)")
        return
    n = len(all_results)
    passed = sum(1 for r in all_results if r[6])
    print(f"  total probes: {n}, passed: {passed} ({100*passed/n:.0f}%)")

    # Calibration: at what gate would we pass each candidate?
    # Compare current gate vs an "if-gate-were-X" simulation
    for trial_gate in (0.10, 0.15, 0.20, 0.30, 0.50, 1.0):
        # Count how many of these candidates would pass at trial_gate
        # (without ERA5 adjustment — uniform threshold)
        would_pass = sum(1 for r in all_results if r[5] <= trial_gate)
        print(f"  if uniform gate were {trial_gate:.2f}: {would_pass}/{n} would pass")

    # Distribution of actual SCL cloud_fraction
    cfs = sorted(r[5] for r in all_results)
    print(f"\n  actual SCL cloud_fraction distribution:")
    print(f"    min={cfs[0]:.3f}  25%={cfs[len(cfs)//4]:.3f}  "
          f"median={cfs[len(cfs)//2]:.3f}  75%={cfs[3*len(cfs)//4]:.3f}  "
          f"max={cfs[-1]:.3f}")


if __name__ == "__main__":
    main()
