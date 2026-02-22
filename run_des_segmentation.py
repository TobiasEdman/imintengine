"""
run_des_segmentation.py -- Run Prithvi segmentation on real Sentinel-2 data
from DES (Digital Earth Sweden).

Uses cached bands from a previous pipeline run if available, or fetches
fresh data from DES. Only runs the Prithvi analyzer in segmentation mode.

Usage:
    # Malmö (default)
    .venv/bin/python run_des_segmentation.py

    # Custom area — Kubbe flood analysis
    .venv/bin/python run_des_segmentation.py \\
        --west 17.95 --south 63.51 --east 18.05 --north 63.56 \\
        --date 2025-09-08 --date-window 5

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

# ---- Defaults (Malmö centrum) --------------------------------------------

DEFAULT_COORDS = {
    "west": 13.00,
    "south": 55.58,
    "east": 13.02,
    "north": 55.60,
}
DEFAULT_DATE = "2023-07-15"
DEFAULT_DATE_WINDOW = 15
DEFAULT_CLOUD_THRESHOLD = 0.3


# ---- Band loading ---------------------------------------------------------

def load_cached_or_fetch(
    coords: dict,
    date: str,
    date_window: int,
    cloud_threshold: float,
    output_dir: str,
    force_fetch: bool = False,
) -> dict:
    """Load cached bands from previous DES run, or fetch fresh from DES.

    Returns:
        Dict with keys: bands, rgb, geo, date, coords, source
    """
    bands_dir = os.path.join(output_dir, "bands")
    prefix = f"{date}_"

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
                "date": cached.get("date", date),
                "coords": cached.get("coords", coords),
                "source": "cache",
            }
        except (FileNotFoundError, KeyError) as e:
            print(f"    Cache incomplete ({e}), will fetch from DES...")

    # Fetch from DES
    print(f"    Fetching Sentinel-2 data from DES...")
    print(f"    Date: {date} (±{date_window} days)")
    print(f"    Coords: {coords}")

    from imint.fetch import fetch_des_data

    try:
        result = fetch_des_data(
            date=date,
            coords=coords,
            cloud_threshold=cloud_threshold,
            include_scl=True,
            date_window=date_window,
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
        "date": date,
        "coords": coords,
        "source": "des",
    }


# ---- Main -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run Prithvi segmentation on Sentinel-2 data from DES",
    )
    # Area
    parser.add_argument("--west", type=float, default=DEFAULT_COORDS["west"],
                        help="West longitude (WGS84)")
    parser.add_argument("--south", type=float, default=DEFAULT_COORDS["south"],
                        help="South latitude (WGS84)")
    parser.add_argument("--east", type=float, default=DEFAULT_COORDS["east"],
                        help="East longitude (WGS84)")
    parser.add_argument("--north", type=float, default=DEFAULT_COORDS["north"],
                        help="North latitude (WGS84)")
    # Temporal
    parser.add_argument("--date", default=DEFAULT_DATE,
                        help="Target date ISO format (default: 2023-07-15)")
    parser.add_argument("--date-window", type=int, default=DEFAULT_DATE_WINDOW,
                        help="Days ± to search for cloud-free imagery (default: 15)")
    # Model
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
    # Data source
    parser.add_argument(
        "--fetch", action="store_true",
        help="Force fresh DES fetch even if cache exists",
    )
    parser.add_argument(
        "--no-cloud-check", action="store_true",
        help="Allow cloudy scenes (sets threshold to 100%%, still fetches SCL for visualization)",
    )
    parser.add_argument(
        "--cloud-threshold", type=float, default=None,
        help="Cloud fraction threshold 0.0-1.0 (default: 0.3). Overrides --no-cloud-check.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override output directory (default: auto-generated from coords)",
    )
    args = parser.parse_args()

    coords = {
        "west": args.west, "south": args.south,
        "east": args.east, "north": args.north,
    }

    # Auto-generate output dir from coordinates if not specified
    if args.output_dir:
        output_dir = args.output_dir
    else:
        area_tag = f"{args.west:.2f}_{args.south:.2f}_{args.east:.2f}_{args.north:.2f}"
        output_dir = str(PROJECT_ROOT / "outputs" / f"seg_{area_tag}_{args.date}_{args.task_head}")

    print("=" * 70)
    print(f"  Prithvi Segmentation on DES Sentinel-2 Data")
    print(f"  Area:       {args.west:.4f}–{args.east:.4f}°E, {args.south:.4f}–{args.north:.4f}°N")
    print(f"  Date:       {args.date} (±{args.date_window} days)")
    print(f"  Task head:  {args.task_head}")
    print(f"  Device:     {args.device}")
    # Resolve cloud threshold
    if args.cloud_threshold is not None:
        cloud_threshold = args.cloud_threshold
    elif args.no_cloud_check:
        cloud_threshold = 1.0  # accept everything, but still fetch SCL
    else:
        cloud_threshold = DEFAULT_CLOUD_THRESHOLD

    print(f"  Cloud thr:  {cloud_threshold:.0%}")
    print(f"  Output dir: {output_dir}")
    print("=" * 70)

    # Load data (cache or DES)
    print("\n[1] Loading Sentinel-2 bands...")
    data = load_cached_or_fetch(
        coords=coords,
        date=args.date,
        date_window=args.date_window,
        cloud_threshold=cloud_threshold,
        output_dir=output_dir,
        force_fetch=args.fetch,
    )
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
