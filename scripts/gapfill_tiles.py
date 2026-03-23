#!/usr/bin/env python3
"""
scripts/gapfill_tiles.py — Fill missing frames in multitemporal tiles

Scans existing .npz tiles, finds those with missing temporal frames
(temporal_mask has zeros), and tries to fetch the missing frames from
additional years (e.g. 2020, 2021).

This is a second-pass operation: run after the initial prepare_lulc_data.py
fetch completes with some tiles having <4/4 frames.

Usage:
    # Dry run — report how many tiles need gap-filling
    python scripts/gapfill_tiles.py --data-dir data/lulc_seasonal_vpp --dry-run

    # Fill missing frames using 2020 and 2021 data
    python scripts/gapfill_tiles.py --data-dir data/lulc_seasonal_vpp \
        --years 2020 2021

    # Also try 2017 as fallback
    python scripts/gapfill_tiles.py --data-dir data/lulc_seasonal_vpp \
        --years 2020 2021 2017
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def scan_incomplete_tiles(tiles_dir: Path) -> list[dict]:
    """Scan tiles directory for tiles with missing frames.

    Returns list of dicts with tile info for gap-filling.
    """
    incomplete = []
    for npz_path in sorted(tiles_dir.glob("tile_*.npz")):
        try:
            data = np.load(npz_path, allow_pickle=True)
            mask = data["temporal_mask"]
            n_frames = int(data["num_frames"])

            if mask.sum() < n_frames:
                missing_indices = [i for i, m in enumerate(mask) if m == 0]
                incomplete.append({
                    "path": npz_path,
                    "name": npz_path.stem,
                    "easting": float(data["easting"]),
                    "northing": float(data["northing"]),
                    "lat": float(data["lat"]),
                    "lon": float(data["lon"]),
                    "n_frames": n_frames,
                    "n_valid": int(mask.sum()),
                    "missing": missing_indices,
                    "temporal_mask": mask.copy(),
                    "dates": list(data["dates"]),
                    "doy": data["doy"].copy(),
                })
        except Exception as e:
            print(f"  ⚠ Error reading {npz_path.name}: {e}")

    return incomplete


def _tile_bbox(easting: float, northing: float, grid_half: float = 5000.0):
    """Compute WGS84 and snapped EPSG:3006 bboxes for a tile center.

    Uses the same coordinate flow as prepare_data.py:
      1. _sweref99_to_wgs84 (center ± grid_half) → WGS84 bbox
      2. _to_nmd_grid → snapped EPSG:3006 bbox

    Returns (coords_wgs84, projected_3006) dicts.
    """
    from imint.training.sampler import _sweref99_to_wgs84
    from imint.fetch import _to_nmd_grid

    # Same corner logic as grid_to_wgs84: sw=(east_3006, south_3006)
    sw = _sweref99_to_wgs84(easting + grid_half, northing - grid_half)
    ne = _sweref99_to_wgs84(easting - grid_half, northing + grid_half)

    coords_wgs84 = {
        "west": min(sw[1], ne[1]),
        "south": min(sw[0], ne[0]),
        "east": max(sw[1], ne[1]),
        "north": max(sw[0], ne[0]),
    }
    projected = _to_nmd_grid(coords_wgs84)
    return coords_wgs84, projected


def get_vpp_windows(easting: float, northing: float,
                    n_frames: int = 4) -> list[tuple[int, int]]:
    """Re-derive VPP-guided DOY windows for a tile location."""
    from imint.training.cdse_vpp import fetch_vpp_tiles
    from imint.training.vpp_windows import compute_growing_season_windows

    _wgs84, projected = _tile_bbox(easting, northing)

    try:
        vpp = fetch_vpp_tiles(
            west=projected["west"], south=projected["south"],
            east=projected["east"], north=projected["north"],
            size_px=64,
        )
        windows = compute_growing_season_windows(
            vpp["sosd"], vpp["eosd"],
            num_frames=n_frames,
        )
        return windows
    except Exception:
        from imint.training.vpp_windows import _FALLBACK_DOY_WINDOWS
        return _FALLBACK_DOY_WINDOWS[:n_frames]


def gapfill_tile(
    tile_info: dict,
    years: list[str],
    cloud_threshold: float = 0.10,
    haze_threshold: float = 0.12,
) -> bool:
    """Try to fill missing frames in a single tile.

    Returns True if any frames were filled.
    """
    from datetime import datetime
    from imint.fetch import fetch_seasonal_dates_doy
    from imint.training.cdse_s2 import fetch_s2_scene

    npz_path = tile_info["path"]
    easting = tile_info["easting"]
    northing = tile_info["northing"]
    missing = tile_info["missing"]
    n_frames = tile_info["n_frames"]

    # Load the full tile
    data = dict(np.load(npz_path, allow_pickle=True))
    image = data["image"]
    n_bands = int(data["num_bands"])
    h, w = image.shape[1], image.shape[2]

    temporal_mask = data["temporal_mask"].copy()
    dates = list(data["dates"])
    doy_vals = data["doy"].copy()

    # Get aligned bboxes (same flow as prepare_data.py)
    coords, projected = _tile_bbox(easting, northing)

    # Get VPP windows for this tile
    windows = get_vpp_windows(easting, northing, n_frames)

    # S2 fetch uses the snapped EPSG:3006 bbox (from _to_nmd_grid)
    sh_west = projected["west"]
    sh_south = projected["south"]
    sh_east = projected["east"]
    sh_north = projected["north"]

    # Only fetch windows for missing frames
    missing_windows = [windows[i] for i in missing]

    # STAC discovery uses WGS84 coords
    from imint.fetch import fetch_seasonal_dates_doy
    season_candidates = fetch_seasonal_dates_doy(
        coords, missing_windows, years, scene_cloud_max=50.0,
    )

    filled_any = False

    for slot, (frame_idx, (doy_start, doy_end)) in enumerate(
        zip(missing, missing_windows)
    ):
        win_label = f"doy{doy_start}-{doy_end}"
        candidates_all = season_candidates[slot]

        if not candidates_all:
            print(f"    {win_label}: no candidates in {','.join(years)}")
            continue

        # Try top 3 candidates
        found = False
        for cand_date, scene_cc in candidates_all[:3]:
            try:
                result = fetch_s2_scene(
                    sh_west, sh_south, sh_east, sh_north,
                    date=cand_date,
                    size_px=(h, w),
                    cloud_threshold=cloud_threshold,
                    haze_threshold=haze_threshold,
                )
            except Exception:
                result = None

            if result is not None:
                frame_img, _scl, cloud_frac = result
                # Insert into the image array
                band_start = frame_idx * n_bands
                band_end = (frame_idx + 1) * n_bands

                # Handle shape mismatch
                if frame_img.shape[1:] != (h, w):
                    padded = np.zeros((n_bands, h, w), dtype=np.float32)
                    fh = min(frame_img.shape[1], h)
                    fw = min(frame_img.shape[2], w)
                    padded[:, :fh, :fw] = frame_img[:, :fh, :fw]
                    frame_img = padded

                image[band_start:band_end] = frame_img
                temporal_mask[frame_idx] = 1
                dates[frame_idx] = cand_date
                dt = datetime.strptime(cand_date, "%Y-%m-%d")
                doy_vals[frame_idx] = dt.timetuple().tm_yday

                print(f"    {win_label}: {cand_date} ✓ (cloud={cloud_frac:.0%})")
                found = True
                filled_any = True
                break

            time.sleep(1.5)  # Rate limit

        if not found:
            print(f"    {win_label}: no clear date (tried {min(3, len(candidates_all))} "
                  f"from {','.join(years)})")

    if filled_any:
        # Save updated tile
        data["image"] = image
        data["temporal_mask"] = temporal_mask
        data["dates"] = np.array(dates)
        data["doy"] = doy_vals

        # Atomic write
        tmp_path = npz_path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp_path, **data)
        tmp_path.rename(npz_path)

    return filled_any


def main():
    parser = argparse.ArgumentParser(
        description="Gap-fill missing temporal frames using additional years",
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Training data directory (e.g. data/lulc_seasonal_vpp)",
    )
    parser.add_argument(
        "--years", nargs="+", default=["2020", "2021"],
        help="Years to try for gap-filling (default: 2020 2021)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only report incomplete tiles, don't fetch",
    )
    parser.add_argument(
        "--cloud-threshold", type=float, default=0.10,
        help="Max cloud fraction (default: 0.10)",
    )
    parser.add_argument(
        "--max-tiles", type=int, default=None,
        help="Max tiles to process (for testing)",
    )
    args = parser.parse_args()

    tiles_dir = Path(args.data_dir) / "tiles"
    if not tiles_dir.exists():
        print(f"Error: {tiles_dir} not found")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Gap-fill: scanning {tiles_dir}")
    print(f"Additional years: {args.years}")
    print(f"{'='*60}\n")

    # Scan for incomplete tiles
    incomplete = scan_incomplete_tiles(tiles_dir)

    # Stats
    total = len(list(tiles_dir.glob("tile_*.npz")))
    by_valid = {}
    for t in incomplete:
        key = f"{t['n_valid']}/{t['n_frames']}"
        by_valid[key] = by_valid.get(key, 0) + 1

    print(f"Total tiles: {total}")
    print(f"Incomplete: {len(incomplete)}")
    for k, v in sorted(by_valid.items()):
        print(f"  {k} frames: {v} tiles")
    print()

    if args.dry_run:
        print("Dry run — not fetching.")
        return

    # Gap-fill
    n_process = len(incomplete)
    if args.max_tiles:
        n_process = min(n_process, args.max_tiles)

    filled_count = 0
    frames_filled = 0
    t_start = time.time()

    for i, tile_info in enumerate(incomplete[:n_process]):
        n_missing = len(tile_info["missing"])
        print(f"\n[{i+1}/{n_process}] {tile_info['name']} "
              f"({tile_info['n_valid']}/{tile_info['n_frames']}, "
              f"{n_missing} missing)")

        old_valid = tile_info["n_valid"]
        try:
            success = gapfill_tile(
                tile_info,
                years=args.years,
                cloud_threshold=args.cloud_threshold,
            )
            if success:
                # Re-read to check new frame count
                data = np.load(tile_info["path"], allow_pickle=True)
                new_valid = int(data["temporal_mask"].sum())
                n_filled = new_valid - old_valid
                frames_filled += n_filled
                filled_count += 1
                print(f"  → {old_valid}/{tile_info['n_frames']} → "
                      f"{new_valid}/{tile_info['n_frames']} "
                      f"(+{n_filled} frames)")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Gap-fill complete:")
    print(f"  Processed: {n_process} tiles in {elapsed:.0f}s")
    print(f"  Tiles improved: {filled_count}")
    print(f"  Frames filled: {frames_filled}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
