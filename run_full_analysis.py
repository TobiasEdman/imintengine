"""
run_full_analysis.py — Run ALL IMINT Engine analyzers on DES Sentinel-2 data.

Fetches real satellite data and runs every analyzer:
  - Change Detection
  - Spectral (NDVI, NDWI, NDBI, EVI)
  - Object Detection (heatmap)
  - Prithvi Segmentation (burn_scars or sen1floods11)
  - NMD (land cover)
  - COT (cloud optical thickness)

Usage:
    DES_USER=testuser DES_PASSWORD=secretpassword \
    .venv/bin/python run_full_analysis.py \
        --west 15.392 --south 61.897 --east 15.442 --north 61.947 \
        --date 2018-07-24 --date-window 5 --no-cloud-check
"""
from __future__ import annotations

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from imint.job import IMINTJob
from imint.fetch import fetch_des_data, ensure_baseline
from imint.engine import run_job


def main():
    parser = argparse.ArgumentParser(
        description="Run ALL IMINT Engine analyzers on DES Sentinel-2 data",
    )
    parser.add_argument("--west", type=float, required=True)
    parser.add_argument("--south", type=float, required=True)
    parser.add_argument("--east", type=float, required=True)
    parser.add_argument("--north", type=float, required=True)
    parser.add_argument("--date", required=True, help="Target date ISO (e.g. 2018-07-24)")
    parser.add_argument("--date-window", type=int, default=5)
    parser.add_argument("--no-cloud-check", action="store_true")
    parser.add_argument("--task-head", default="burn_scars",
                        choices=["sen1floods11", "burn_scars"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    coords = {
        "west": args.west, "south": args.south,
        "east": args.east, "north": args.north,
    }

    cloud_threshold = 1.0 if args.no_cloud_check else 0.3

    if args.output_dir:
        output_dir = args.output_dir
    else:
        area_tag = f"{args.west:.2f}_{args.south:.2f}_{args.east:.2f}_{args.north:.2f}"
        output_dir = str(PROJECT_ROOT / "outputs" / f"full_{area_tag}_{args.date}")

    print("=" * 70)
    print("  IMINT Engine — Full Analysis (ALL Analyzers)")
    print(f"  Area:      {args.west:.4f}–{args.east:.4f}°E, {args.south:.4f}–{args.north:.4f}°N")
    print(f"  Date:      {args.date} (±{args.date_window} days)")
    print(f"  Cloud:     {cloud_threshold:.0%} threshold")
    print(f"  Task head: {args.task_head}")
    print(f"  Device:    {args.device}")
    print(f"  Output:    {output_dir}")
    print("=" * 70)

    # Step 1: Fetch data
    print("\n[1] Fetching Sentinel-2 data from DES...")
    try:
        fetch_result = fetch_des_data(
            date=args.date,
            coords=coords,
            cloud_threshold=cloud_threshold,
            include_scl=True,
            date_window=args.date_window,
        )
    except Exception as e:
        print(f"\n    DES fetch FAILED: {e}")
        print("    Check auth: python scripts/des_login.py --test")
        raise SystemExit(1)

    print(f"    ✓ Data fetched: {fetch_result.rgb.shape}")
    print(f"    Cloud fraction: {fetch_result.cloud_fraction:.1%}")
    print(f"    Bands: {list(fetch_result.bands.keys())}")
    if fetch_result.geo:
        print(f"    CRS: {fetch_result.geo.crs}")

    # Step 1.5: Ensure cloud-free baseline for change detection
    print("\n[1.5] Ensuring cloud-free baseline for change detection...")
    ensure_baseline(
        date=args.date,
        coords=coords,
        output_dir=output_dir,
        cloud_threshold=0.1,
    )

    # Step 2: Write full config
    os.makedirs(output_dir, exist_ok=True)
    config = {
        "change_detection": {"enabled": True, "threshold": 0.15, "min_region_pixels": 50},
        "spectral": {"enabled": True},
        "object_detection": {"enabled": True, "mode": "heatmap"},
        "prithvi": {
            "enabled": True,
            "mode": "segmentation",
            "task_head": args.task_head,
            "device": args.device,
        },
        "nmd": {"enabled": True},
        "cot": {"enabled": True, "device": args.device},
    }
    config_path = os.path.join(output_dir, "analyzers_full.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Step 3: Build job
    print(f"\n[2] Building IMINTJob...")
    job = IMINTJob(
        date=args.date,
        coords=coords,
        rgb=fetch_result.rgb,
        bands=fetch_result.bands,
        scl=fetch_result.scl,
        geo=fetch_result.geo,
        output_dir=output_dir,
        config_path=config_path,
        job_id=f"full-analysis-{args.date}",
    )

    # Step 4: Run all analyzers
    print(f"\n[3] Running ALL analyzers...")
    print("    (Prithvi segmentation may take a few minutes on CPU)")
    result = run_job(job)

    # Step 5: Print results
    print("\n" + "=" * 70)
    print("  FULL ANALYSIS RESULTS")
    print("=" * 70)
    print(f"  Job:     {result.job_id}")
    print(f"  Success: {result.success}")

    for r in result.analyzer_results:
        status = "✓" if r.success else "✗"
        print(f"\n  [{status}] {r.analyzer}")
        if r.error:
            print(f"      Error: {r.error}")
        if r.outputs:
            for k, v in r.outputs.items():
                if isinstance(v, np.ndarray):
                    print(f"      {k}: shape={v.shape} dtype={v.dtype}")
                elif isinstance(v, dict) and k in ("stats", "class_stats"):
                    for sk, sv in v.items():
                        if isinstance(sv, dict):
                            name = sv.get("name", sk)
                            frac = sv.get("fraction", sv.get("value", "?"))
                            if isinstance(frac, float):
                                print(f"      {k}.{sk}: {name} = {frac:.1%}")
                            else:
                                print(f"      {k}.{sk}: {name} = {frac}")
                        else:
                            if isinstance(sv, float):
                                print(f"      {k}.{sk}: {sv:.4f}")
                            else:
                                print(f"      {k}.{sk}: {sv}")
                elif isinstance(v, dict):
                    print(f"      {k}: <dict with {len(v)} keys>")
                elif isinstance(v, list):
                    print(f"      {k}: {len(v)} items")
                elif isinstance(v, (bool, int, float, str)):
                    print(f"      {k}: {v}")

    # List output files
    print("\n" + "-" * 70)
    print("  Output files:")
    total = 0
    for root, dirs, files in os.walk(output_dir):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            sz = os.path.getsize(fpath)
            total += sz
            rel = os.path.relpath(fpath, output_dir)
            print(f"    {rel:45s}  {sz/1024:.1f} KB")
    print(f"\n  Total: {total/1024:.1f} KB")
    print("=" * 70)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return f"<ndarray {obj.shape}>"
    return str(obj)


if __name__ == "__main__":
    main()
