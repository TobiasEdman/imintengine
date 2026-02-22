"""
run_local_segmentation.py -- Test Prithvi segmentation end-to-end locally

Runs the IMINT Engine with only the Prithvi analyzer enabled in segmentation
mode, using synthetic Sentinel-2 data and a pre-trained task head.

The Sen1Floods11 checkpoint (~1.3 GB) will be downloaded from HuggingFace
on first run and cached for subsequent runs.

Note: Synthetic data does not represent real flood/burn scenes, so the
segmentation output tests pipeline mechanics, not model accuracy.

Usage:
    .venv/bin/python run_local_segmentation.py
    .venv/bin/python run_local_segmentation.py --task_head burn_scars
    .venv/bin/python run_local_segmentation.py --size 128
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

# ---- Project setup --------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_local_pipeline import (
    generate_synthetic_data,
    build_geo_context,
    mock_fetch_nmd_data,
    COORDS,
    IMG_SIZE,
)
from imint.job import IMINTJob
from imint.engine import run_job


def main():
    parser = argparse.ArgumentParser(
        description="Test Prithvi segmentation locally with synthetic data",
    )
    parser.add_argument(
        "--task_head", default="sen1floods11",
        choices=["sen1floods11", "burn_scars"],
        help="Pre-trained task head to use (default: sen1floods11)",
    )
    parser.add_argument(
        "--size", type=int, default=IMG_SIZE,
        help=f"Image size in pixels, square (default: {IMG_SIZE})",
    )
    parser.add_argument(
        "--device", default=None,
        choices=["cpu", "cuda", "mps"],
        help="PyTorch device (default: auto-detect)",
    )
    args = parser.parse_args()

    date = "2026-02-22"
    output_dir = str(PROJECT_ROOT / "outputs" / f"segmentation_{args.task_head}")

    print("=" * 70)
    print(f"  Prithvi Segmentation Test")
    print(f"  Task head:  {args.task_head}")
    print(f"  Image size: {args.size}x{args.size}")
    print(f"  Device:     {args.device or 'auto'}")
    print(f"  Output dir: {output_dir}")
    print("=" * 70)

    # Generate synthetic Sentinel-2 data
    print("\n[1] Generating synthetic Sentinel-2 data...")
    rgb, bands = generate_synthetic_data(args.size, args.size)
    print(f"    RGB shape: {rgb.shape}")
    print(f"    Bands: {list(bands.keys())}")

    # Build GeoContext
    print("\n[2] Building GeoContext (EPSG:3006)...")
    geo = build_geo_context(COORDS, (args.size, args.size))

    os.makedirs(output_dir, exist_ok=True)

    # Build config: only prithvi enabled in segmentation mode
    config = {
        "change_detection": {"enabled": False},
        "spectral": {"enabled": False},
        "object_detection": {"enabled": False},
        "nmd": {"enabled": False},
        "prithvi": {
            "enabled": True,
            "mode": "segmentation",
            "task_head": args.task_head,
        },
    }
    if args.device:
        config["prithvi"]["device"] = args.device

    config_path = str(Path(output_dir) / "analyzers_seg.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    print(f"\n[3] Config written: {config_path}")

    # Build job
    job = IMINTJob(
        date=date,
        coords=COORDS,
        rgb=rgb,
        bands=bands,
        geo=geo,
        output_dir=output_dir,
        config_path=config_path,
        job_id=f"segmentation-{args.task_head}",
    )

    # Run the pipeline
    print(f"\n[4] Running segmentation pipeline with '{args.task_head}'...")
    print("    (This may take a minute on CPU; first run downloads the checkpoint)")

    with patch("imint.analyzers.nmd.fetch_nmd_data", side_effect=mock_fetch_nmd_data):
        result = run_job(job)

    # Print results
    print("\n" + "=" * 70)
    print("  SEGMENTATION RESULTS")
    print("=" * 70)

    for r in result.analyzer_results:
        if r.analyzer == "prithvi":
            print(f"  success: {r.success}")
            if r.error:
                print(f"  error: {r.error}")
                continue

            seg_mask = r.outputs.get("seg_mask")
            if seg_mask is not None:
                print(f"  seg_mask: shape={seg_mask.shape}, dtype={seg_mask.dtype}")
                print(f"  unique classes: {np.unique(seg_mask).tolist()}")

            class_stats = r.outputs.get("class_stats", {})
            if class_stats:
                print("\n  Class statistics:")
                for cls, info in sorted(class_stats.items(), key=lambda x: int(x[0])):
                    name = info.get("name", f"class_{cls}")
                    pct = info["fraction"] * 100
                    count = info["pixel_count"]
                    print(f"    {cls}: {name:20s} {pct:6.1f}%  ({count:,} pixels)")

            if r.metadata:
                print(f"\n  Metadata:")
                for k, v in r.metadata.items():
                    print(f"    {k}: {v}")

    # List output files
    print("\n  Output files:")
    total_size = 0
    for fname in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, fname)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            total_size += size
            size_str = f"{size/1024:.1f} KB" if size >= 1024 else f"{size} bytes"
            print(f"    {fname:40s}  {size_str}")

    print(f"\n  Total output: {total_size/1024:.1f} KB")
    print("=" * 70)


if __name__ == "__main__":
    main()
