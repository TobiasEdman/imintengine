"""
run_des_pipeline.py -- Run the IMINT Engine pipeline with real Sentinel-2
data from DES (Digital Earth Sweden) over Malmö.

Fetches actual satellite imagery and NMD land cover data via openEO,
runs all analyzers, and saves results to outputs/des_malmo/.

Requires:
    - Valid DES authentication (see scripts/des_login.py)
    - .des_token file or stored OIDC refresh token

Usage:
    .venv/bin/python run_des_pipeline.py
"""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np

# ---- Project setup --------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from imint.job import IMINTJob
from imint.fetch import fetch_des_data
from imint.engine import run_job

# ---- Configuration --------------------------------------------------------

# Malmö centrum — ~2x2 km area covering urban, parks, and water
COORDS = {
    "west": 13.00,
    "south": 55.58,
    "east": 13.02,
    "north": 55.60,
}

DATE = "2023-07-15"        # Summer date — good vegetation contrast
DATE_WINDOW = 15           # ±15 days to find cloud-free imagery
CLOUD_THRESHOLD = 0.3      # Max 30% cloud cover

OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "des_malmo")
CONFIG_PATH = str(PROJECT_ROOT / "config" / "analyzers.yaml")


# ---- Main -----------------------------------------------------------------

def main():
    print("=" * 70)
    print("  IMINT Engine -- DES Pipeline Run (Real Data)")
    print(f"  Date: {DATE} (±{DATE_WINDOW} days)")
    print(f"  Coords: {COORDS}")
    print(f"  Cloud threshold: {CLOUD_THRESHOLD}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print("=" * 70)

    # Step 1: Fetch real Sentinel-2 data from DES
    print("\n[1] Fetching Sentinel-2 data from DES...")
    print(f"    Connecting to openeo.digitalearth.se...")

    try:
        fetch_result = fetch_des_data(
            date=DATE,
            coords=COORDS,
            cloud_threshold=CLOUD_THRESHOLD,
            include_scl=True,
            date_window=DATE_WINDOW,
        )
    except Exception as e:
        print(f"\n  ❌ DES fetch failed: {e}")
        print("  Make sure you have valid authentication:")
        print("    python scripts/des_login.py --device")
        return

    print(f"    ✅ Data fetched successfully!")
    print(f"    RGB shape: {fetch_result.rgb.shape}")
    print(f"    Bands: {list(fetch_result.bands.keys())}")
    print(f"    Cloud fraction: {fetch_result.cloud_fraction:.2%}")
    if fetch_result.geo:
        print(f"    CRS: {fetch_result.geo.crs}")
        print(f"    Bounds (EPSG:3006): {fetch_result.geo.bounds_projected}")
        print(f"    Grid shape: {fetch_result.geo.shape}")
        print(f"    Transform: {fetch_result.geo.transform}")

    # Print band statistics
    for name, arr in fetch_result.bands.items():
        print(f"    {name}: min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")

    # Check cloud cover
    if fetch_result.cloud_fraction > CLOUD_THRESHOLD:
        print(f"\n  ⚠️  Cloud fraction ({fetch_result.cloud_fraction:.1%}) exceeds "
              f"threshold ({CLOUD_THRESHOLD:.0%}).")
        print("  Continuing anyway for testing purposes...")

    # Step 2: Build IMINTJob
    print(f"\n[2] Building IMINTJob...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    job = IMINTJob(
        date=DATE,
        coords=COORDS,
        rgb=fetch_result.rgb,
        bands=fetch_result.bands,
        geo=fetch_result.geo,
        output_dir=OUTPUT_DIR,
        config_path=CONFIG_PATH,
        job_id="des-malmo-001",
    )
    print(f"    job_id: {job.job_id}")
    print(f"    geo: {job.geo.crs if job.geo else 'None'}")

    # Step 3: Run the full pipeline (NMD fetched live by NMDAnalyzer)
    print(f"\n[3] Running full IMINT Engine pipeline...")
    print(f"    (NMD will be fetched live from DES by the NMD analyzer)")

    result = run_job(job)

    # Step 4: Print results
    print("\n" + "=" * 70)
    print("  PIPELINE RESULTS")
    print("=" * 70)
    print(f"  job_id:  {result.job_id}")
    print(f"  date:    {result.date}")
    print(f"  success: {result.success}")
    if result.error:
        print(f"  error:   {result.error}")
    print(f"  summary: {result.summary_path}")

    print(f"\n  Analyzer Results ({len(result.analyzer_results)} analyzers):")
    for r in result.analyzer_results:
        print(f"\n    --- {r.analyzer} ---")
        print(f"    success: {r.success}")
        if r.error:
            print(f"    error: {r.error}")
        if r.metadata:
            print(f"    metadata: {json.dumps(r.metadata, indent=6, default=_json_default)}")
        if r.outputs:
            for k, v in r.outputs.items():
                if isinstance(v, np.ndarray):
                    print(f"    output[{k}]: ndarray shape={v.shape} dtype={v.dtype}")
                elif isinstance(v, dict):
                    try:
                        s = json.dumps(v, indent=6, default=_json_default)
                        if len(s) > 800:
                            s = s[:800] + "\n      ... (truncated)"
                        print(f"    output[{k}]: {s}")
                    except Exception:
                        print(f"    output[{k}]: <dict with {len(v)} keys>")
                elif isinstance(v, list):
                    print(f"    output[{k}]: list with {len(v)} items")
                else:
                    print(f"    output[{k}]: {v}")

    # Step 5: List output files
    print("\n" + "=" * 70)
    print("  OUTPUT FILES")
    print("=" * 70)

    total_size = 0
    file_count = 0
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            size = os.path.getsize(fpath)
            total_size += size
            file_count += 1
            rel = os.path.relpath(fpath, OUTPUT_DIR)

            ext = Path(fname).suffix.lower()
            desc = {
                ".png": "PNG image",
                ".tif": "GeoTIFF raster",
                ".geojson": "GeoJSON vector",
                ".json": "JSON report",
                ".npy": "NumPy array",
            }.get(ext, "file")

            size_str = f"{size:,} bytes" if size < 1024 else f"{size/1024:.1f} KB"
            print(f"    {rel:45s}  {size_str:>12s}  ({desc})")

    print(f"\n  Total: {file_count} output files, {total_size/1024:.1f} KB")
    print("=" * 70)

    # Print summary JSON
    if result.summary_path and os.path.exists(result.summary_path):
        print(f"\n  Summary report ({result.summary_path}):")
        with open(result.summary_path) as f:
            summary = json.load(f)
        print(json.dumps(summary, indent=2, default=_json_default))


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return f"<ndarray {obj.shape} {obj.dtype}>"
    return str(obj)


if __name__ == "__main__":
    main()
