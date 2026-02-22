"""
run_des_segmentation.py -- Run Prithvi segmentation on real Sentinel-2 data
from DES (Digital Earth Sweden) over Malmö.

Uses cached bands from a previous pipeline run if available, or fetches
fresh data from DES. Only runs the Prithvi analyzer in segmentation mode.

Usage:
    .venv/bin/python run_des_segmentation.py
    .venv/bin/python run_des_segmentation.py --task_head burn_scars
    .venv/bin/python run_des_segmentation.py --fetch   # Force fresh DES fetch
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import yaml

# ---- Project setup --------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from imint.job import IMINTJob, GeoContext
from imint.engine import run_job, load_bands_cache
from imint.utils import bands_to_rgb

# ---- Configuration --------------------------------------------------------

# Same area as run_des_pipeline.py — Malmö centrum
COORDS = {
    "west": 13.00,
    "south": 55.58,
    "east": 13.02,
    "north": 55.60,
}
DATE = "2023-07-15"
DATE_WINDOW = 15
CLOUD_THRESHOLD = 0.3

# Where previous DES runs saved data
DES_OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "des_malmo")


# ---- Band loading ---------------------------------------------------------

def load_cached_or_fetch(force_fetch: bool = False) -> dict:
    """Load cached bands from previous DES run, or fetch fresh from DES.

    Returns:
        Dict with keys: bands, rgb, geo, date, coords, source
    """
    bands_dir = os.path.join(DES_OUTPUT_DIR, "bands")
    prefix = f"{DATE}_"

    # Try loading from cache
    if not force_fetch and os.path.isdir(bands_dir):
        try:
            cached = load_bands_cache(bands_dir, prefix)
            print(f"    Loaded cached bands from {bands_dir}")
            print(f"    Bands: {list(cached['bands'].keys())}")

            # Reconstruct GeoContext from metadata
            geo = None
            if cached["geo_meta"]:
                from rasterio.transform import Affine
                gm = cached["geo_meta"]
                geo = GeoContext(
                    crs=gm["crs"],
                    transform=Affine(*gm["transform"][:6]),
                    bounds_projected=gm["bounds_projected"],
                    bounds_wgs84=gm["bounds_wgs84"],
                    shape=tuple(gm["shape"]),
                )

            # Reconstruct RGB from bands
            rgb = bands_to_rgb(cached["bands"])

            return {
                "bands": cached["bands"],
                "rgb": rgb,
                "geo": geo,
                "date": cached.get("date", DATE),
                "coords": cached.get("coords", COORDS),
                "source": "cache",
            }
        except (FileNotFoundError, KeyError) as e:
            print(f"    Cache incomplete ({e}), will fetch from DES...")

    # Fetch from DES
    print(f"    Fetching Sentinel-2 data from DES...")
    print(f"    Date: {DATE} (±{DATE_WINDOW} days)")
    print(f"    Coords: {COORDS}")

    from imint.fetch import fetch_des_data

    try:
        result = fetch_des_data(
            date=DATE,
            coords=COORDS,
            cloud_threshold=CLOUD_THRESHOLD,
            include_scl=True,
            date_window=DATE_WINDOW,
        )
    except Exception as e:
        print(f"\n    DES fetch failed: {e}")
        print("    Make sure you have valid authentication:")
        print("      python scripts/des_login.py --device")
        raise SystemExit(1)

    print(f"    Fetched! RGB: {result.rgb.shape}, Cloud: {result.cloud_fraction:.1%}")

    return {
        "bands": result.bands,
        "rgb": result.rgb,
        "geo": result.geo,
        "date": DATE,
        "coords": COORDS,
        "source": "des",
    }


# ---- Main -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run Prithvi segmentation on real Malmö Sentinel-2 data",
    )
    parser.add_argument(
        "--task_head", default="sen1floods11",
        choices=["sen1floods11", "burn_scars"],
        help="Pre-trained task head (default: sen1floods11)",
    )
    parser.add_argument(
        "--device", default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="PyTorch device (default: cpu)",
    )
    parser.add_argument(
        "--fetch", action="store_true",
        help="Force fresh DES fetch even if cache exists",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Override source directory for band cache (default: outputs/des_malmo)",
    )
    parser.add_argument(
        "--date", default=None,
        help="Override date for band cache prefix (default: 2023-07-15)",
    )
    args = parser.parse_args()

    if args.data_dir:
        global DES_OUTPUT_DIR
        DES_OUTPUT_DIR = args.data_dir
    if args.date:
        global DATE
        DATE = args.date

    output_dir = str(PROJECT_ROOT / "outputs" / f"des_segmentation_{args.task_head}")

    print("=" * 70)
    print(f"  Prithvi Segmentation on Real DES Data — Malmö")
    print(f"  Task head:  {args.task_head}")
    print(f"  Device:     {args.device}")
    print(f"  Output dir: {output_dir}")
    print("=" * 70)

    # Load data (cache or DES)
    print("\n[1] Loading Sentinel-2 bands...")
    data = load_cached_or_fetch(force_fetch=args.fetch)
    print(f"    Source: {data['source']}")
    print(f"    RGB shape: {data['rgb'].shape}")
    print(f"    Bands: {list(data['bands'].keys())}")
    for name, arr in data["bands"].items():
        print(f"      {name}: min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")

    # Build segmentation-only config
    os.makedirs(output_dir, exist_ok=True)
    config = {
        "change_detection": {"enabled": False},
        "spectral": {"enabled": False},
        "object_detection": {"enabled": False},
        "nmd": {"enabled": False},
        "prithvi": {
            "enabled": True,
            "mode": "segmentation",
            "task_head": args.task_head,
            "device": args.device,
        },
    }
    config_path = str(Path(output_dir) / "analyzers_seg.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    print(f"\n[2] Config: {config_path}")

    # Build job
    job = IMINTJob(
        date=data["date"],
        coords=data["coords"],
        rgb=data["rgb"],
        bands=data["bands"],
        geo=data["geo"],
        output_dir=output_dir,
        config_path=config_path,
        job_id=f"des-segmentation-{args.task_head}",
    )

    # Run segmentation
    print(f"\n[3] Running segmentation with '{args.task_head}'...")
    print("    (This may take a minute on CPU)")

    from unittest.mock import patch as mock_patch
    # Mock NMD fetch even though it's disabled (safety)
    with mock_patch("imint.analyzers.nmd.fetch_nmd_data", side_effect=lambda *a, **k: None):
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

    # List output files
    print("\n  Output files:")
    total_size = 0
    for fname in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, fname)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            total_size += size
            size_str = f"{size/1024:.1f} KB" if size >= 1024 else f"{size} bytes"
            print(f"    {fname:45s}  {size_str}")
    print(f"\n  Total output: {total_size/1024:.1f} KB")
    print("=" * 70)


if __name__ == "__main__":
    main()
