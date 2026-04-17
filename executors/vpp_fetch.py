"""
executors/vpp_fetch.py — ColonyOS executor for VPP phenology enrichment.

Adds HR-VPP vegetation phenology channels to existing training tiles.
Designed to run as a ColonyOS container job, reading tiles from CFS
and writing enriched tiles back.

Each job processes a single tile:
    1. Read existing .npz from CFS
    2. Check if VPP bands already present → skip
    3. Fetch VPP from Sentinel Hub Process API
    4. Write updated .npz back to CFS (atomic)

The ColonyOS job spec (config/vpp_fetch_job.json) invokes:
    python executors/vpp_fetch.py

Environment variables are set by ColonyOS from the job spec:
    EASTING, NORTHING        — tile center in EPSG:3006
    TILES_DIR                — tiles directory (default: /cfs/tiles)
    VPP_YEAR                 — product year (default: 2021)
    PATCH_SIZE_M             — tile spatial extent in meters (default: 2560)
    CDSE_CLIENT_ID           — CDSE OAuth client ID
    CDSE_CLIENT_SECRET       — CDSE OAuth client secret
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np
from pathlib import Path

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imint.training.cdse_vpp import fetch_vpp_tiles

# VPP band names as stored in the .npz tile
_VPP_CHANNEL_NAMES = ["vpp_sosd", "vpp_eosd", "vpp_length", "vpp_maxv", "vpp_minv"]


class VPPFetchExecutor:
    """ColonyOS executor that enriches a tile with VPP phenology data."""

    def execute(self):
        # ── 1. Parse environment ──────────────────────────────────────────
        easting = int(os.environ["EASTING"])
        northing = int(os.environ["NORTHING"])
        cell_key = f"{easting}_{northing}"

        tiles_dir = Path(os.environ.get("TILES_DIR", "/cfs/tiles"))
        vpp_year = int(os.environ.get("VPP_YEAR", "2021"))
        patch_size_m = int(os.environ.get("PATCH_SIZE_M", "2560"))
        half_m = patch_size_m // 2

        tile_path = tiles_dir / f"tile_{cell_key}.npz"

        print(f"[VPPFetch] Tile {cell_key} (year={vpp_year})")
        t_start = time.monotonic()

        # ── 2. Check tile exists ──────────────────────────────────────────
        if not tile_path.exists():
            print(f"[VPPFetch] Tile not found: {tile_path}")
            print(f"[VPPFetch] Skipping — tile must be created first via seasonal fetch")
            return

        # ── 3. Check if VPP already present ───────────────────────────────
        with np.load(tile_path, allow_pickle=True) as data:
            existing_keys = list(data.keys())

        already_has = all(ch in existing_keys for ch in _VPP_CHANNEL_NAMES)
        if already_has:
            print(f"[VPPFetch] Tile already has VPP channels, skipping")
            return

        # ── 4. Compute bbox from tile center ──────────────────────────────
        west = easting - half_m
        south = northing - half_m
        east = easting + half_m
        north = northing + half_m

        # Determine pixel size from existing tile
        with np.load(tile_path, allow_pickle=True) as data:
            existing = dict(data)
            img = existing.get("spectral", existing.get("image"))
            if img is not None and img.ndim >= 2:
                h_px, w_px = img.shape[-2], img.shape[-1]
            else:
                h_px, w_px = 256, 256

        # ── 5. Fetch VPP data ─────────────────────────────────────────────
        print(f"  Fetching VPP (year={vpp_year}, bbox={west},{south},{east},{north})...")

        vpp = fetch_vpp_tiles(
            west, south, east, north,
            size_px=(h_px, w_px),
            year=vpp_year,
        )

        # ── 6. Add VPP bands to tile ──────────────────────────────────────
        for raw_name, arr in vpp.items():
            channel_name = f"vpp_{raw_name}"
            existing[channel_name] = arr
            nz_pct = float((arr > 0).mean() * 100)
            print(f"  {channel_name}: mean={arr.mean():.2f}, "
                  f"max={arr.max():.2f}, nonzero={nz_pct:.0f}%")

        # ── 7. Atomic rewrite ─────────────────────────────────────────────
        tmp_path = tile_path.with_suffix(".vpp_tmp.npz")
        np.savez_compressed(tmp_path, **existing)
        tmp_path.rename(tile_path)

        elapsed = time.monotonic() - t_start
        print(f"[VPPFetch] Tile {cell_key} enriched with VPP ({elapsed:.1f}s)")


if __name__ == "__main__":
    executor = VPPFetchExecutor()
    try:
        executor.execute()
    except Exception as e:
        print(f"[VPPFetch] FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
