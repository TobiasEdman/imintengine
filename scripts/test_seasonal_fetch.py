#!/usr/bin/env python3
"""
Quick integration test for seasonal/multitemporal Sentinel-2 fetching.

Tests STAC discovery for all 4 seasons, then fetches ONE image at a time
from DES until we have a clear scene for each season.

Uses the standard small test bbox.

Usage:
    python scripts/test_seasonal_fetch.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def main():
    from imint.fetch import (
        fetch_seasonal_dates, fetch_seasonal_image,
        _connect, _fetch_scl, _to_nmd_grid,
    )

    # ── Standard small test bbox (southern Sweden) ──────────────────
    coords = {
        "west": 13.0, "south": 55.5,
        "east": 13.1, "north": 55.6,
    }
    years = ["2019", "2018"]
    seasonal_windows = [
        (4, 5),   # spring
        (6, 7),   # summer
        (8, 9),   # autumn
        (1, 2),   # winter
    ]
    season_names = ["spring(4-5)", "summer(6-7)", "autumn(8-9)", "winter(1-2)"]
    prithvi_bands = ["B02", "B03", "B04", "B8A", "B11", "B12"]
    cloud_threshold = 0.10

    print("=" * 60)
    print("  Seasonal Fetch Test")
    print(f"  Coords: {coords}")
    print("=" * 60)

    # ── Phase 1: STAC discovery (fast, no DES needed) ───────────────
    print("\n── Phase 1: STAC Discovery ──")
    t0 = time.time()
    season_candidates = fetch_seasonal_dates(
        coords, seasonal_windows, years, scene_cloud_max=50.0,
    )
    print(f"  Time: {time.time() - t0:.1f}s\n")

    for name, cands in zip(season_names, season_candidates):
        n = len(cands)
        if cands:
            top3 = ", ".join(f"{d}({c:.0f}%)" for d, c in cands[:3])
            print(f"  {name}: {n} dates | top: {top3}")
        else:
            print(f"  {name}: no candidates")

    # ── Phase 2: SCL pre-screen + fetch, one image at a time ────────
    print("\n── Phase 2: Fetch one image at a time per season ──")

    conn = _connect()
    projected = _to_nmd_grid(coords)

    frames = []
    frame_dates = []
    frame_mask = []

    for win_idx, (name, cands) in enumerate(
        zip(season_names, season_candidates)
    ):
        print(f"\n  {name}:")
        if not cands:
            print(f"    No candidates, skipping")
            frames.append(None)
            frame_dates.append("")
            frame_mask.append(0)
            continue

        found = False
        for date, scene_cloud in cands[:5]:  # Try up to 5 candidates
            # SCL pre-screen (one date at a time)
            print(f"    Trying {date} (scene cloud {scene_cloud:.0f}%)...")
            from datetime import datetime, timedelta
            dt = datetime.strptime(date, "%Y-%m-%d")
            temporal = [date, (dt + timedelta(days=1)).strftime("%Y-%m-%d")]

            t0 = time.time()
            try:
                _scl, aoi_cloud, _crs, _tr = _fetch_scl(
                    conn, projected, temporal, date_window=0,
                )
                scl_time = time.time() - t0
                print(f"    SCL: cloud={aoi_cloud:.1%} ({scl_time:.1f}s)", end="")
            except Exception as e:
                print(f"    SCL failed: {e}")
                time.sleep(2)
                continue

            if aoi_cloud > cloud_threshold:
                print(f" > {cloud_threshold:.0%}, skip")
                time.sleep(1)
                continue

            print(f" ✓ clear!")
            time.sleep(1)

            # Full spectral fetch
            print(f"    Fetching 6 bands...", end="", flush=True)
            t0 = time.time()
            result = fetch_seasonal_image(date, coords, prithvi_bands)
            fetch_time = time.time() - t0

            if result is not None:
                img, dt_str = result
                print(f" shape={img.shape} ({fetch_time:.1f}s) ✓")
                frames.append(img)
                frame_dates.append(dt_str)
                frame_mask.append(1)
                found = True
                break
            else:
                print(f" FAILED ({fetch_time:.1f}s)")

            time.sleep(2)

        if not found:
            frames.append(None)
            frame_dates.append("")
            frame_mask.append(0)

    # ── Phase 3: Summary ────────────────────────────────────────────
    n_valid = sum(frame_mask)
    print(f"\n{'='*60}")
    print(f"  Results: {n_valid}/4 seasonal frames")
    for name, d, m in zip(season_names, frame_dates, frame_mask):
        status = f"✓ {d}" if m else "✗ missing"
        print(f"    {name}: {status}")

    if n_valid > 0:
        valid = [f for f in frames if f is not None]
        shapes = [f.shape for f in valid]
        print(f"\n  Shapes: {shapes}")
        print(f"  All same shape: {len(set(shapes)) == 1}")

        # Stack test
        ref = valid[0].shape[1:]
        stacked = []
        for f in frames:
            if f is not None:
                stacked.append(f)
            else:
                stacked.append(np.zeros((6,) + ref, dtype=np.float32))
        mt_image = np.concatenate(stacked, axis=0)
        print(f"  Stacked: {mt_image.shape} ({mt_image.nbytes / 1024**2:.1f} MB)")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
