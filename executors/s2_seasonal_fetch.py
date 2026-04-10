"""
executors/s2_seasonal_fetch.py — ColonyOS executor for S2 L2A via Process API.

Fetches all 4 seasonal frames for a single training tile using the CDSE
Sentinel Hub Process API (1-stage: 6 spectral + SCL in one HTTP call)
and saves the result as a compressed .npz file to CFS.

This is the Process API replacement for `seasonal_fetch.py` (openEO).
Key differences:
  - Uses Sentinel Hub Process API instead of openEO batch jobs
  - 1-stage fetch: spectral + SCL in one HTTP POST, cloud check local
  - Only needs CDSE credentials (no DES for spectral data)
  - STAC discovery still uses DES STAC API (same as before)
  - NMD labels are fetched via openEO/DES but cached as .npy

The ColonyOS job spec (config/s2_seasonal_fetch_job.json) invokes:
    python executors/s2_seasonal_fetch.py

Environment variables are set by ColonyOS from the job spec:
    EASTING, NORTHING            — tile center in EPSG:3006
    WEST_WGS84, SOUTH_WGS84,
    EAST_WGS84, NORTH_WGS84     — bounding box in WGS84
    YEARS                        — comma-separated, e.g. "2019,2018"
    SEASONAL_WINDOWS             — e.g. "4-5,6-7,8-9,1-2"
    SEASONAL_CLOUD_THRESHOLD     — max cloud fraction, e.g. "0.10"
    B02_HAZE_THRESHOLD           — max mean B02 reflectance (default: 0.06)
    TILES_DIR                    — output directory (default: /cfs/tiles)
    NUM_CLASSES                  — NMD class count (default: 19)
    TILE_SIZE_PX                 — tile size in pixels (default: 256)
    CDSE_CLIENT_ID               — CDSE OAuth client ID
    CDSE_CLIENT_SECRET           — CDSE OAuth client secret
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np
from pathlib import Path

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imint.training.cdse_s2 import fetch_s2_seasonal_tile
from imint.fetch import fetch_nmd_data, FetchError
from imint.training.class_schema import nmd_raster_to_lulc


def _parse_windows(s: str) -> list[tuple[int, int]]:
    """Parse "4-5,6-7,8-9,1-2" → [(4,5), (6,7), (8,9), (1,2)]."""
    windows = []
    for part in s.split(","):
        start, end = part.strip().split("-")
        windows.append((int(start), int(end)))
    return windows


class S2SeasonalFetchExecutor:
    """ColonyOS executor: S2 L2A fetch via Sentinel Hub Process API."""

    def execute(self):
        # ── 1. Parse environment ─────────────────────────────────────────
        easting = int(os.environ["EASTING"])
        northing = int(os.environ["NORTHING"])
        cell_key = f"{easting}_{northing}"

        coords_wgs84 = {
            "west": float(os.environ["WEST_WGS84"]),
            "south": float(os.environ["SOUTH_WGS84"]),
            "east": float(os.environ["EAST_WGS84"]),
            "north": float(os.environ["NORTH_WGS84"]),
        }

        years = os.environ.get("YEARS", "2019,2018").split(",")
        windows = _parse_windows(
            os.environ.get("SEASONAL_WINDOWS", "4-5,6-7,8-9,1-2")
        )
        cloud_threshold = float(
            os.environ.get("SEASONAL_CLOUD_THRESHOLD", "0.10")
        )
        haze_threshold = float(
            os.environ.get("B02_HAZE_THRESHOLD", "0.06")
        )
        tiles_dir = Path(os.environ.get("TILES_DIR", "/cfs/tiles"))
        num_classes = int(os.environ.get("NUM_CLASSES", "19"))
        tile_size_px = int(os.environ.get("TILE_SIZE_PX", "256"))

        tile_path = tiles_dir / f"tile_{cell_key}.npz"

        print(f"[S2Fetch] Tile {cell_key} via Sentinel Hub Process API")
        print(f"  Windows: {windows}, Years: {years}")
        print(f"  Cloud threshold: {cloud_threshold:.0%}, "
              f"Haze threshold: {haze_threshold}")
        print(f"  Tile size: {tile_size_px}px")
        t_start = time.monotonic()

        # ── 2. Skip if already done ──────────────────────────────────────
        if tile_path.exists():
            print(f"[S2Fetch] Tile already exists: {tile_path}")
            return

        # ── 3. Fetch seasonal frames via Process API ─────────────────────
        result = fetch_s2_seasonal_tile(
            easting=easting,
            northing=northing,
            coords_wgs84=coords_wgs84,
            windows=windows,
            years=years,
            size_px=tile_size_px,
            cloud_threshold=cloud_threshold,
            haze_threshold=haze_threshold,
        )

        if result is None:
            raise RuntimeError(
                f"Tile {cell_key}: no valid frames from Process API"
            )

        n_valid = int(result["temporal_mask"].sum())
        n_frames = len(windows)
        image = result["spectral"]
        ref_shape = (image.shape[-2], image.shape[-1])  # (H, W)

        print(f"  Spectral: {n_valid}/{n_frames} frames, "
              f"shape={image.shape}")

        # ── 4. Fetch NMD labels ──────────────────────────────────────────
        try:
            nmd_result = fetch_nmd_data(
                coords=coords_wgs84,
                target_shape=ref_shape,
            )
            labels = nmd_raster_to_lulc(
                nmd_result.nmd_raster,
                num_classes=num_classes,
            )
            print(f"  NMD labels: shape={labels.shape}, "
                  f"classes={np.unique(labels).tolist()}")
        except (FetchError, Exception) as e:
            print(f"  Warning: NMD fetch failed: {e}")
            print(f"  Using zero labels (will need NMD re-fetch later)")
            labels = np.zeros(ref_shape, dtype=np.uint8)

        # ── 5. Save tile (atomic write) ──────────────────────────────────
        tiles_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tile_path.with_suffix(".tmp.npz")

        save_data = dict(result)  # copy from fetch_s2_seasonal_tile
        save_data["label"] = labels

        np.savez_compressed(tmp_path, **save_data)
        tmp_path.rename(tile_path)

        elapsed = time.monotonic() - t_start
        print(f"[S2Fetch] Tile {cell_key} saved "
              f"({n_valid}/{n_frames} frames, {elapsed:.1f}s)")


if __name__ == "__main__":
    executor = S2SeasonalFetchExecutor()
    try:
        executor.execute()
    except Exception as e:
        print(f"[S2Fetch] FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
