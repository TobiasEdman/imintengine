"""
run_local_pipeline.py -- Run the IMINT Engine analysis pipeline locally
with synthetic Sentinel-2 data for the Malmo area.

Generates realistic reflectance data, builds a proper GeoContext with
EPSG:3006 snapped to the NMD 10m grid, mocks NMD fetch to avoid DES
connection requirements, and runs the full pipeline.

Usage:
    .venv/bin/python run_local_pipeline.py
"""
from __future__ import annotations

import os
import sys
import json
import math
from pathlib import Path
from datetime import datetime
from unittest.mock import patch

import numpy as np
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from rasterio.warp import transform_bounds

# ---- Project setup --------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from imint.job import IMINTJob, GeoContext
from imint.engine import run_job
from imint.fetch import NMD_GRID_SIZE, TARGET_CRS, NMDFetchResult

# ---- Configuration --------------------------------------------------------

DATE = "2026-02-21"
COORDS = {"west": 13.0, "south": 55.55, "east": 13.1, "north": 55.65}
IMG_SIZE = 64  # 64x64 pixels
OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "local_pipeline_run")
CONFIG_PATH = str(PROJECT_ROOT / "config" / "analyzers.yaml")

# ---- Step 1: Project WGS84 to EPSG:3006 and snap to NMD 10m grid ---------

def build_geo_context(coords: dict, shape: tuple) -> GeoContext:
    """Build a GeoContext by projecting WGS84 coords to EPSG:3006,
    snapping to the NMD 10m grid, and computing the affine transform."""
    w, s, e, n = transform_bounds(
        CRS.from_epsg(4326), CRS.from_epsg(3006),
        coords["west"], coords["south"], coords["east"], coords["north"],
    )

    grid = NMD_GRID_SIZE
    proj = {
        "west": math.floor(w / grid) * grid,
        "south": math.floor(s / grid) * grid,
        "east": math.ceil(e / grid) * grid,
        "north": math.ceil(n / grid) * grid,
    }

    h, w_px = shape
    transform = from_bounds(
        proj["west"], proj["south"], proj["east"], proj["north"],
        w_px, h,
    )

    return GeoContext(
        crs=TARGET_CRS,
        transform=transform,
        bounds_projected=proj,
        bounds_wgs84=coords,
        shape=shape,
    )


# ---- Step 2: Generate synthetic Sentinel-2 reflectance data --------------

def generate_synthetic_data(h: int, w: int, seed: int = 42) -> tuple:
    """Generate synthetic but realistic Sentinel-2 reflectance data.

    Returns:
        rgb: (H, W, 3) float32 in [0, 1]
        bands: dict with B02, B03, B04, B08, B8A, B11, B12 as float32 [0, 1]

    The synthetic scene has:
        - Top-left quadrant: vegetation (high NIR, moderate red)
        - Top-right quadrant: water (low reflectance, slightly higher blue)
        - Bottom-left quadrant: urban/built-up (moderate, uniform reflectance)
        - Bottom-right quadrant: bare soil (high red, moderate NIR)
    """
    rng = np.random.default_rng(seed)

    # Base reflectance arrays per band
    b02 = np.zeros((h, w), dtype=np.float32)  # Blue
    b03 = np.zeros((h, w), dtype=np.float32)  # Green
    b04 = np.zeros((h, w), dtype=np.float32)  # Red
    b08 = np.zeros((h, w), dtype=np.float32)  # NIR (wide, 10m)
    b8a = np.zeros((h, w), dtype=np.float32)  # NIR (narrow, 20m)
    b11 = np.zeros((h, w), dtype=np.float32)  # SWIR1
    b12 = np.zeros((h, w), dtype=np.float32)  # SWIR2

    mid_h, mid_w = h // 2, w // 2

    # -- Vegetation (top-left): high NIR, low red, moderate green
    b02[:mid_h, :mid_w] = 0.03 + rng.normal(0, 0.005, (mid_h, mid_w))
    b03[:mid_h, :mid_w] = 0.06 + rng.normal(0, 0.008, (mid_h, mid_w))
    b04[:mid_h, :mid_w] = 0.04 + rng.normal(0, 0.006, (mid_h, mid_w))
    b08[:mid_h, :mid_w] = 0.35 + rng.normal(0, 0.03, (mid_h, mid_w))
    b8a[:mid_h, :mid_w] = 0.33 + rng.normal(0, 0.03, (mid_h, mid_w))
    b11[:mid_h, :mid_w] = 0.15 + rng.normal(0, 0.015, (mid_h, mid_w))
    b12[:mid_h, :mid_w] = 0.08 + rng.normal(0, 0.01, (mid_h, mid_w))

    # -- Water (top-right): very low reflectance, slightly higher blue
    b02[:mid_h, mid_w:] = 0.05 + rng.normal(0, 0.003, (mid_h, w - mid_w))
    b03[:mid_h, mid_w:] = 0.03 + rng.normal(0, 0.002, (mid_h, w - mid_w))
    b04[:mid_h, mid_w:] = 0.02 + rng.normal(0, 0.002, (mid_h, w - mid_w))
    b08[:mid_h, mid_w:] = 0.01 + rng.normal(0, 0.002, (mid_h, w - mid_w))
    b8a[:mid_h, mid_w:] = 0.01 + rng.normal(0, 0.002, (mid_h, w - mid_w))
    b11[:mid_h, mid_w:] = 0.005 + rng.normal(0, 0.001, (mid_h, w - mid_w))
    b12[:mid_h, mid_w:] = 0.003 + rng.normal(0, 0.001, (mid_h, w - mid_w))

    # -- Urban/built-up (bottom-left): moderate, uniform, high SWIR
    b02[mid_h:, :mid_w] = 0.10 + rng.normal(0, 0.01, (h - mid_h, mid_w))
    b03[mid_h:, :mid_w] = 0.10 + rng.normal(0, 0.01, (h - mid_h, mid_w))
    b04[mid_h:, :mid_w] = 0.12 + rng.normal(0, 0.01, (h - mid_h, mid_w))
    b08[mid_h:, :mid_w] = 0.15 + rng.normal(0, 0.01, (h - mid_h, mid_w))
    b8a[mid_h:, :mid_w] = 0.14 + rng.normal(0, 0.01, (h - mid_h, mid_w))
    b11[mid_h:, :mid_w] = 0.25 + rng.normal(0, 0.02, (h - mid_h, mid_w))
    b12[mid_h:, :mid_w] = 0.22 + rng.normal(0, 0.02, (h - mid_h, mid_w))

    # -- Bare soil (bottom-right): high red, moderate NIR
    b02[mid_h:, mid_w:] = 0.08 + rng.normal(0, 0.01, (h - mid_h, w - mid_w))
    b03[mid_h:, mid_w:] = 0.10 + rng.normal(0, 0.01, (h - mid_h, w - mid_w))
    b04[mid_h:, mid_w:] = 0.15 + rng.normal(0, 0.015, (h - mid_h, w - mid_w))
    b08[mid_h:, mid_w:] = 0.20 + rng.normal(0, 0.015, (h - mid_h, w - mid_w))
    b8a[mid_h:, mid_w:] = 0.19 + rng.normal(0, 0.015, (h - mid_h, w - mid_w))
    b11[mid_h:, mid_w:] = 0.22 + rng.normal(0, 0.02, (h - mid_h, w - mid_w))
    b12[mid_h:, mid_w:] = 0.20 + rng.normal(0, 0.02, (h - mid_h, w - mid_w))

    # Clip to valid reflectance range
    for arr in [b02, b03, b04, b08, b8a, b11, b12]:
        np.clip(arr, 0.0, 1.0, out=arr)

    bands = {"B02": b02, "B03": b03, "B04": b04, "B08": b08, "B8A": b8a, "B11": b11, "B12": b12}

    # Build RGB from bands with percentile stretch
    r, g, b = b04.copy(), b03.copy(), b02.copy()
    rgb = np.stack([r, g, b], axis=-1)
    p2, p98 = np.percentile(rgb, [2, 98])
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1).astype(np.float32)

    return rgb, bands


# ---- Step 3: Generate a synthetic NMD raster ------------------------------

def generate_synthetic_nmd(h: int, w: int) -> np.ndarray:
    """Generate a synthetic NMD raster matching the scene layout.

    NMD class codes used:
        111 = Forest (pine)
        61  = Water (lakes)
        51  = Developed land (buildings)
        3   = Cropland
        41  = Open land (bare)
    """
    nmd = np.full((h, w), 3, dtype=np.uint8)  # Default: cropland
    mid_h, mid_w = h // 2, w // 2

    nmd[:mid_h, :mid_w] = 111   # Forest (pine) - matches vegetation quadrant
    nmd[:mid_h, mid_w:] = 61    # Water - matches water quadrant
    nmd[mid_h:, :mid_w] = 51    # Developed - matches urban quadrant
    nmd[mid_h:, mid_w:] = 3     # Cropland/bare soil - matches bare soil quadrant

    # Add a small patch of open land in the bare soil area
    nmd[mid_h + 5:mid_h + 15, mid_w + 5:mid_w + 15] = 41

    return nmd


# ---- Step 4: Mock NMD fetch -----------------------------------------------

def mock_fetch_nmd_data(coords, target_shape=None, token=None, cache_dir=None):
    """Mock NMD fetch that returns synthetic data instead of calling DES."""
    h, w = target_shape if target_shape else (IMG_SIZE, IMG_SIZE)
    nmd_raster = generate_synthetic_nmd(h, w)
    return NMDFetchResult(
        nmd_raster=nmd_raster,
        crs=CRS.from_epsg(3006),
        transform=None,
        from_cache=False,
    )


# ---- Step 5: Set up a baseline for change detection ----------------------

def setup_baseline(output_dir: str, coords: dict, rgb: np.ndarray, date: str):
    """Create a slightly different baseline image for change detection.

    The baseline is the same image with some modifications so the
    change detector finds actual changes.
    """
    month = int(date.split("-")[1])
    if month in (3, 4, 5):
        season = "spring"
    elif month in (6, 7, 8):
        season = "summer"
    elif month in (9, 10, 11):
        season = "autumn"
    else:
        season = "winter"

    area_key = f"{coords['west']}_{coords['south']}_{coords['east']}_{coords['north']}"
    baseline_dir = os.path.join(output_dir, "..", "baselines")
    os.makedirs(baseline_dir, exist_ok=True)
    baseline_path = os.path.join(baseline_dir, f"{season}_{area_key}.npy")

    # Create a baseline that differs from current image in specific areas
    baseline = rgb.copy()
    # Simulate vegetation growth in the top-left (brighter green)
    baseline[5:20, 5:20, 1] = np.clip(baseline[5:20, 5:20, 1] + 0.3, 0, 1)
    # Simulate construction in bottom-left (brighter)
    baseline[40:55, 5:20] = np.clip(baseline[40:55, 5:20] + 0.25, 0, 1)

    np.save(baseline_path, baseline)
    print(f"  Baseline saved: {baseline_path}")
    return baseline_path


# ---- Main -----------------------------------------------------------------

def main():
    print("=" * 70)
    print("  IMINT Engine -- Local Pipeline Run")
    print(f"  Date: {DATE}")
    print(f"  Coords: {COORDS}")
    print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print("=" * 70)

    # Generate data
    print("\n[1] Generating synthetic Sentinel-2 data...")
    rgb, bands = generate_synthetic_data(IMG_SIZE, IMG_SIZE)
    print(f"    RGB shape: {rgb.shape}, dtype: {rgb.dtype}")
    print(f"    Bands: {list(bands.keys())}")
    for name, arr in bands.items():
        print(f"      {name}: shape={arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")

    # Build GeoContext
    print("\n[2] Building GeoContext (EPSG:3006, NMD 10m grid)...")
    geo = build_geo_context(COORDS, (IMG_SIZE, IMG_SIZE))
    print(f"    CRS: {geo.crs}")
    print(f"    Bounds (projected): {geo.bounds_projected}")
    print(f"    Bounds (WGS84): {geo.bounds_wgs84}")
    print(f"    Transform: {geo.transform}")
    print(f"    Shape: {geo.shape}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Set up baseline for change detection
    print("\n[3] Setting up change detection baseline...")
    setup_baseline(OUTPUT_DIR, COORDS, rgb, DATE)

    # Build the IMINTJob
    print("\n[4] Building IMINTJob...")
    job = IMINTJob(
        date=DATE,
        coords=COORDS,
        rgb=rgb,
        bands=bands,
        geo=geo,
        output_dir=OUTPUT_DIR,
        config_path=CONFIG_PATH,
        job_id="local-pipeline-001",
    )
    print(f"    job_id: {job.job_id}")
    print(f"    date: {job.date}")
    print(f"    config: {job.config_path}")

    # Run the pipeline with NMD fetch mocked
    print("\n[5] Running full IMINT Engine pipeline...")
    print("    (NMD fetch is mocked to avoid DES connection)")

    with patch("imint.analyzers.nmd.fetch_nmd_data", side_effect=mock_fetch_nmd_data):
        result = run_job(job)

    # Print results
    print("\n" + "=" * 70)
    print("  PIPELINE RESULTS")
    print("=" * 70)
    print(f"  job_id:  {result.job_id}")
    print(f"  date:    {result.date}")
    print(f"  success: {result.success}")
    print(f"  summary: {result.summary_path}")

    print(f"\n  Analyzer Results ({len(result.analyzer_results)} analyzers):")
    for r in result.analyzer_results:
        print(f"\n    --- {r.analyzer} ---")
        print(f"    success: {r.success}")
        if r.error:
            print(f"    error: {r.error}")
        if r.metadata:
            print(f"    metadata: {json.dumps(r.metadata, indent=6, default=str)}")
        if r.outputs:
            for k, v in r.outputs.items():
                if isinstance(v, np.ndarray):
                    print(f"    output[{k}]: ndarray shape={v.shape} dtype={v.dtype}")
                elif isinstance(v, dict):
                    # Print dicts but truncate large nested structures
                    try:
                        s = json.dumps(v, indent=6, default=str)
                        if len(s) > 500:
                            s = s[:500] + "\n      ... (truncated)"
                        print(f"    output[{k}]: {s}")
                    except Exception:
                        print(f"    output[{k}]: <dict with {len(v)} keys>")
                else:
                    print(f"    output[{k}]: {v}")

    # List all output files
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

            # Describe file type
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

    # Also check baselines directory
    baseline_dir = os.path.join(OUTPUT_DIR, "..", "baselines")
    if os.path.exists(baseline_dir):
        print(f"\n  Baseline files (in {os.path.relpath(baseline_dir, PROJECT_ROOT)}):")
        for fname in sorted(os.listdir(baseline_dir)):
            fpath = os.path.join(baseline_dir, fname)
            if os.path.isfile(fpath):
                size = os.path.getsize(fpath)
                size_str = f"{size:,} bytes" if size < 1024 else f"{size/1024:.1f} KB"
                print(f"    {fname:45s}  {size_str:>12s}")

    print(f"\n  Total: {file_count} output files, {total_size/1024:.1f} KB")
    print("=" * 70)

    # Read and display the summary JSON
    if result.summary_path and os.path.exists(result.summary_path):
        print(f"\n  Summary report ({result.summary_path}):")
        with open(result.summary_path) as f:
            summary = json.load(f)
        print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
