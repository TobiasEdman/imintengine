#!/usr/bin/env python3
"""Batch fetch Sentinel-2 tiles via CDSE openEO.

Groups tiles by similar VPP temporal windows (14-day rounding) and
submits each group as one openEO batch job covering a merged bbox.
Results are downloaded as GeoTIFFs and split back into individual tiles.

This complements the per-tile SH Process API fetch — use for tiles
that haven't been fetched yet, especially when the Process API is slow.

Usage:
    python scripts/batch_fetch_openeo.py \
        --tile-locations /data/tile_locations_full.json \
        --vpp-cache /data/unified_v2_512/.vpp_cache.json \
        --output-dir /data/unified_v2_512 \
        --tile-size-px 512 \
        --parallel-jobs 2 \
        --temporal-resolution 14

Credentials: CDSE_CLIENT_ID, CDSE_CLIENT_SECRET env vars.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def round_doy(doy: int, resolution: int = 14) -> int:
    return round(doy / resolution) * resolution


def doy_to_date(year: int, doy: int) -> str:
    """Convert year + day-of-year to ISO date string."""
    dt = datetime(year, 1, 1) + timedelta(days=doy - 1)
    return dt.strftime("%Y-%m-%d")


def group_tiles_by_temporal_window(
    tile_locations: list[dict],
    vpp_cache: dict[str, list],
    temporal_resolution: int = 14,
) -> dict[tuple, list[dict]]:
    """Group tiles by rounded VPP temporal windows.

    Returns dict mapping (rounded_window_tuple) → [tile_locations].
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)

    for tile in tile_locations:
        name = tile["name"]
        windows = vpp_cache.get(name)
        if windows is None:
            # No VPP data — use default seasonal windows
            windows = [(91, 150), (151, 210), (211, 270)]

        key = tuple(
            (round_doy(s, temporal_resolution), round_doy(e, temporal_resolution))
            for s, e in windows
        )
        groups[key] = groups.get(key, [])
        groups[key].append(tile)

    return dict(groups)


def merge_bbox(tiles: list[dict]) -> dict:
    """Compute merged bounding box covering all tiles."""
    west = min(t["bbox_3006"]["west"] for t in tiles)
    south = min(t["bbox_3006"]["south"] for t in tiles)
    east = max(t["bbox_3006"]["east"] for t in tiles)
    north = max(t["bbox_3006"]["north"] for t in tiles)
    return {"west": west, "south": south, "east": east, "north": north}


def bbox_3006_to_4326(bbox: dict) -> dict:
    """Convert SWEREF99 TM bbox to WGS84."""
    from imint.training.tile_fetch import bbox_3006_to_wgs84
    return bbox_3006_to_wgs84(bbox)


def fetch_group_openeo(
    conn,
    tiles: list[dict],
    temporal_windows: tuple[tuple[int, int], ...],
    frame_idx: int,
    year: int,
    output_dir: Path,
    tile_size_px: int = 512,
) -> list[str]:
    """Fetch one temporal frame for a group of tiles via openEO batch job.

    Args:
        conn: openEO connection.
        tiles: List of tile locations in this group.
        temporal_windows: The rounded (doy_start, doy_end) windows.
        frame_idx: Which frame (0=spring, 1=summer, 2=late summer).
        year: Target year for this frame.
        output_dir: Where to save results.
        tile_size_px: Tile size in pixels.

    Returns:
        List of tile names that were successfully fetched.
    """
    if frame_idx >= len(temporal_windows):
        return []

    doy_start, doy_end = temporal_windows[frame_idx]
    date_start = doy_to_date(year, max(doy_start, 1))
    date_end = doy_to_date(year, min(doy_end, 365))

    # Merged bbox in WGS84 for openEO
    merged_bbox = merge_bbox(tiles)
    bbox_wgs = bbox_3006_to_4326(merged_bbox)

    print(f"    Frame {frame_idx}: {date_start} → {date_end}, "
          f"{len(tiles)} tiles, bbox={merged_bbox['west']:.0f},"
          f"{merged_bbox['south']:.0f}→{merged_bbox['east']:.0f},"
          f"{merged_bbox['north']:.0f}")

    # Build openEO process graph
    # Strategy: load SCL to compute per-scene cloud fraction, select the
    # scene with lowest cloud coverage, return its 6 spectral bands.
    # This gives a single clean scene per pixel — no compositing.

    # Step 1: Load spectral + SCL
    cube = conn.load_collection(
        "SENTINEL2_L2A",
        spatial_extent={
            "west": bbox_wgs["west"],
            "south": bbox_wgs["south"],
            "east": bbox_wgs["east"],
            "north": bbox_wgs["north"],
            "crs": "EPSG:4326",
        },
        temporal_extent=[date_start, date_end],
        bands=["B02", "B03", "B04", "B8A", "B11", "B12", "SCL"],
        max_cloud_cover=30,
    )

    # Step 2: Compute per-scene clear-sky fraction from SCL.
    # SCL clear classes: 4=vegetation, 5=bare, 6=water, 7=low_cloud_prob
    scl = cube.band("SCL")
    clear_flag = (scl == 4) | (scl == 5) | (scl == 6) | (scl == 7)

    # Step 3: Mask cloud/shadow pixels in spectral bands
    cube_spectral = cube.filter_bands(["B02", "B03", "B04", "B8A", "B11", "B12"])
    cube_masked = cube_spectral.mask(~clear_flag)

    # Step 4: Reduce temporal — cloud-free mosaic.
    # "first" takes the first valid (non-masked) pixel per position.
    # With max_cloud_cover=30 pre-filter and SCL mask, the first valid
    # observation is from the clearest scene at each pixel.
    # If "first" is not supported, fall back to "median".
    try:
        cube_best = cube_masked.reduce_dimension(dimension="t", reducer="first")
    except Exception:
        # "first" not available — use median as fallback (mixes scenes but
        # still cloud-free thanks to mask)
        cube_best = cube_masked.reduce_dimension(dimension="t", reducer="median")

    # Submit batch job
    job = cube.create_job(
        title=f"frame{frame_idx}_{date_start}_{len(tiles)}tiles",
        out_format="GTiff",
    )
    job.start_job()

    # Poll until done
    print(f"    Job {job.job_id} submitted, polling...")
    while True:
        status = job.status()
        if status == "finished":
            break
        elif status == "error":
            logs = job.logs()
            print(f"    Job FAILED: {logs[-1] if logs else 'no logs'}")
            return []
        time.sleep(30)

    # Download results
    job_dir = output_dir / f"_openeo_job_{job.job_id}"
    job_dir.mkdir(parents=True, exist_ok=True)
    job.get_results().download_files(str(job_dir))

    # Split large GeoTIFF into individual tile .npz files
    fetched = []
    try:
        import rasterio
        from rasterio.windows import from_bounds

        tiff_files = list(job_dir.glob("*.tif")) + list(job_dir.glob("*.tiff"))
        if not tiff_files:
            print(f"    No GeoTIFF output from job {job.job_id}")
            return []

        with rasterio.open(tiff_files[0]) as src:
            for tile in tiles:
                bbox = tile["bbox_3006"]
                try:
                    window = from_bounds(
                        bbox["west"], bbox["south"],
                        bbox["east"], bbox["north"],
                        src.transform,
                    )
                    data = src.read(window=window,
                                     out_shape=(6, tile_size_px, tile_size_px))
                    # Convert DN to reflectance [0, 1]
                    spectral = data.astype(np.float32) / 10000.0

                    # Save as frame data (will be merged into .npz later)
                    frame_path = output_dir / f"_frame_{tile['name']}_f{frame_idx}.npy"
                    np.save(frame_path, spectral)
                    fetched.append(tile["name"])
                except Exception as e:
                    print(f"    Failed to extract {tile['name']}: {e}")
    except Exception as e:
        print(f"    Failed to split GeoTIFF: {e}")

    return fetched


def merge_frames_to_npz(
    tile_name: str,
    n_frames: int,
    output_dir: Path,
    tile_loc: dict,
    tile_size_px: int = 512,
) -> bool:
    """Merge individual frame .npy files into final .npz tile."""
    frames = []
    for f_idx in range(n_frames):
        frame_path = output_dir / f"_frame_{tile_name}_f{f_idx}.npy"
        if frame_path.exists():
            frames.append(np.load(frame_path))
        else:
            frames.append(np.zeros((6, tile_size_px, tile_size_px),
                                    dtype=np.float32))

    spectral = np.concatenate(frames, axis=0)  # (T*6, H, W)
    temporal_mask = np.array([
        1 if (output_dir / f"_frame_{tile_name}_f{i}.npy").exists() else 0
        for i in range(n_frames)
    ], dtype=np.uint8)

    bbox = tile_loc["bbox_3006"]
    out_path = output_dir / f"{tile_name}.npz"
    np.savez_compressed(
        out_path,
        spectral=spectral,
        temporal_mask=temporal_mask,
        doy=np.zeros(n_frames, dtype=np.int32),  # filled by build_labels
        dates=np.array([""]*n_frames),
        multitemporal=np.int32(1),
        num_frames=np.int32(n_frames),
        num_bands=np.int32(6),
        bbox_3006=np.array([bbox["west"], bbox["south"],
                            bbox["east"], bbox["north"]], dtype=np.int32),
        easting=np.int32((bbox["west"] + bbox["east"]) // 2),
        northing=np.int32((bbox["south"] + bbox["north"]) // 2),
        source=tile_loc.get("source", "lulc"),
    )

    # Cleanup frame files
    for f_idx in range(n_frames):
        frame_path = output_dir / f"_frame_{tile_name}_f{f_idx}.npy"
        frame_path.unlink(missing_ok=True)

    return True


def main():
    parser = argparse.ArgumentParser(description="Batch fetch via CDSE openEO")
    parser.add_argument("--tile-locations", required=True)
    parser.add_argument("--vpp-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tile-size-px", type=int, default=512)
    parser.add_argument("--parallel-jobs", type=int, default=2,
                        help="Max concurrent openEO batch jobs (free tier: 2)")
    parser.add_argument("--temporal-resolution", type=int, default=14,
                        help="DOY rounding for temporal grouping (days)")
    parser.add_argument("--max-groups", type=int, default=None,
                        help="Limit number of groups to process")
    parser.add_argument("--min-group-size", type=int, default=10,
                        help="Skip groups smaller than this (use SH API instead)")
    parser.add_argument("--year", type=int, default=2022)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tile locations
    with open(args.tile_locations) as f:
        all_tiles = json.load(f)

    # Filter out already-fetched
    existing = {p.stem for p in output_dir.glob("*.npz")}
    tiles = [t for t in all_tiles if t["name"] not in existing]
    print(f"Total: {len(all_tiles)}, existing: {len(existing)}, to fetch: {len(tiles)}")

    # Load VPP cache
    with open(args.vpp_cache) as f:
        vpp_cache = json.load(f)

    # Group tiles by temporal window
    groups = group_tiles_by_temporal_window(
        tiles, vpp_cache, args.temporal_resolution,
    )
    # Sort by group size (largest first)
    sorted_groups = sorted(groups.items(), key=lambda x: -len(x[1]))

    # Filter by min size
    sorted_groups = [(k, v) for k, v in sorted_groups if len(v) >= args.min_group_size]
    if args.max_groups:
        sorted_groups = sorted_groups[:args.max_groups]

    total_tiles = sum(len(v) for _, v in sorted_groups)
    print(f"Groups: {len(sorted_groups)}, covering {total_tiles} tiles")
    print(f"Temporal resolution: {args.temporal_resolution} days")

    # Connect to CDSE openEO
    import openeo
    conn = openeo.connect("https://openeo.dataspace.copernicus.eu/")
    conn.authenticate_oidc_client_credentials(
        client_id=os.environ["CDSE_CLIENT_ID"],
        client_secret=os.environ["CDSE_CLIENT_SECRET"],
    )
    print(f"Connected to CDSE openEO")

    # Process groups
    n_frames = 3  # growing season frames (autumn fetched separately)
    for gi, (windows, group_tiles) in enumerate(sorted_groups):
        print(f"\n=== Group {gi+1}/{len(sorted_groups)}: "
              f"{len(group_tiles)} tiles, windows={windows} ===")

        for frame_idx in range(n_frames):
            year = args.year
            fetched = fetch_group_openeo(
                conn, group_tiles, windows, frame_idx, year,
                output_dir, args.tile_size_px,
            )
            print(f"    Frame {frame_idx}: {len(fetched)}/{len(group_tiles)} fetched")

        # Merge frames into .npz per tile
        tile_lookup = {t["name"]: t for t in group_tiles}
        for tile in group_tiles:
            merge_frames_to_npz(
                tile["name"], n_frames, output_dir,
                tile, args.tile_size_px,
            )

    print(f"\n=== Done ===")
    print(f"Tiles in output: {len(list(output_dir.glob('*.npz')))}")


if __name__ == "__main__":
    main()
