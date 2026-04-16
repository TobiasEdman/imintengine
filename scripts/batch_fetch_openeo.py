#!/usr/bin/env python3
"""Two-stage batch fetch via CDSE openEO.

Stage 1 (screen): Per tile, fetch SCL band → count cloud pixels per
scene server-side → return {date: cloud_frac} as JSON. Tiles batched
via MultiBackendJobManager (2 concurrent on free tier). Output:
best_dates.json with {tile_id: {frame_idx: {date, cloud_frac}}}.

Stage 2 (fetch): Group tiles by their best date. Per date-group,
fetch 6 spectral bands in one openEO job covering the merged bbox.
Split GeoTIFF back into individual tiles. No pixel compositing.

Usage:
    python scripts/batch_fetch_openeo.py stage1 \
        --tile-locations /data/tile_locations_full.json \
        --vpp-cache /data/unified_v2_512/.vpp_cache.json \
        --output-dir /data/unified_v2_512 --year 2022

    python scripts/batch_fetch_openeo.py stage2 \
        --best-dates /data/unified_v2_512/best_dates.json \
        --tile-locations /data/tile_locations_full.json \
        --output-dir /data/unified_v2_512 --tile-size-px 512
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SCL_CLOUD_CLASSES = {3, 8, 9, 10}


def doy_to_date(year: int, doy: int) -> str:
    return (datetime(year, 1, 1) + timedelta(days=max(doy - 1, 0))).strftime("%Y-%m-%d")


def connect_cdse():
    import openeo
    conn = openeo.connect("https://openeo.dataspace.copernicus.eu/")
    conn.authenticate_oidc_client_credentials(
        client_id=os.environ["CDSE_CLIENT_ID"],
        client_secret=os.environ["CDSE_CLIENT_SECRET"],
    )
    return conn


def _normalize_bbox(tile: dict) -> dict:
    """Extract bbox_3006 dict from tile, handling both formats."""
    if "bbox_3006" in tile and isinstance(tile["bbox_3006"], dict):
        return tile["bbox_3006"]
    if "bbox" in tile:
        b = tile["bbox"]
        if isinstance(b, (list, tuple)) and len(b) == 4:
            return {"west": b[0], "south": b[1], "east": b[2], "north": b[3]}
    raise KeyError(f"No bbox in tile: {tile.get('name', '?')}")


def bbox_to_wgs84(bbox_3006: dict) -> dict:
    from imint.training.tile_fetch import bbox_3006_to_wgs84
    return bbox_3006_to_wgs84(bbox_3006)


def merge_bbox(tiles: list[dict]) -> dict:
    bboxes = [_normalize_bbox(t) for t in tiles]
    return {
        "west": min(b["west"] for b in bboxes),
        "south": min(b["south"] for b in bboxes),
        "east": max(b["east"] for b in bboxes),
        "north": max(b["north"] for b in bboxes),
    }


def wait_for_job(job, poll_interval=20):
    while True:
        status = job.status()
        if status == "finished":
            return True
        elif status in ("error", "canceled"):
            try:
                logs = job.logs()
                print(f"    FAILED: {logs[-1] if logs else 'no logs'}")
            except Exception:
                print(f"    FAILED (no logs)")
            return False
        time.sleep(poll_interval)


# ── Stage 1: SCL Screening (per tile, batched) ──────────────────────────────

def screen_tile_scl(conn, tile: dict, frame_windows: list, year: int) -> dict:
    """Screen one tile: per frame, find the date with lowest cloud fraction.

    Runs server-side: loads SCL, counts cloud pixels, reduces to scalar
    per scene. Downloads tiny JSON result, not raster data.

    Returns: {frame_idx: {date, cloud_frac}} or empty dict on failure.
    """
    bbox_wgs = bbox_to_wgs84(_normalize_bbox(tile))
    spatial = {
        "west": bbox_wgs["west"], "south": bbox_wgs["south"],
        "east": bbox_wgs["east"], "north": bbox_wgs["north"],
        "crs": "EPSG:4326",
    }

    results = {}
    for frame_idx, (doy_start, doy_end) in enumerate(frame_windows):
        date_start = doy_to_date(year, max(doy_start, 1))
        date_end = doy_to_date(year, min(doy_end, 365))

        try:
            from shapely.geometry import box, mapping

            # Load SCL for this tile's bbox and temporal window
            scl = conn.load_collection(
                "SENTINEL2_L2A",
                spatial_extent=spatial,
                temporal_extent=[date_start, date_end],
                bands=["SCL"],
                max_cloud_cover=50,
            )

            # Server-side cloud fraction: SCL cloud classes → boolean → spatial mean
            cloud_flag = (scl.band("SCL") == 3) | (scl.band("SCL") == 8) | \
                         (scl.band("SCL") == 9) | (scl.band("SCL") == 10)

            # aggregate_spatial collapses x/y → scalar per scene per geometry
            geom = mapping(box(
                bbox_wgs["west"], bbox_wgs["south"],
                bbox_wgs["east"], bbox_wgs["north"],
            ))
            cloud_frac_ts = cloud_flag.aggregate_spatial(
                geometries=geom, reducer="mean",
            )

            # Download as JSON timeseries (tiny — one float per scene date)
            ts_json = cloud_frac_ts.execute()

            # CDSE openEO aggregate_spatial returns:
            # {"2022-06-05T00:00:00Z": [[0.0003]], "2022-06-15T00:00:00Z": [[0.65]], ...}
            # Keys = ISO timestamps, values = [[cloud_frac]] (double-nested list)
            best_date = None
            best_frac = 1.0

            if isinstance(ts_json, dict):
                for date_key, val in ts_json.items():
                    if date_key == "data":
                        continue  # skip metadata keys
                    date_str = str(date_key)[:10]
                    # Extract float from [[value]] or [value] or value
                    frac = None
                    if isinstance(val, (int, float)):
                        frac = float(val)
                    elif isinstance(val, list):
                        flat = val
                        while isinstance(flat, list) and flat:
                            flat = flat[0]
                        if isinstance(flat, (int, float)):
                            frac = float(flat)
                    if frac is not None and frac < best_frac:
                        best_frac = frac
                        best_date = date_str

            if best_date:
                results[str(frame_idx)] = {
                    "date": best_date,
                    "cloud_frac": round(best_frac, 4),
                }

        except Exception as e:
            print(f"    Screen failed for {tile['name']} frame {frame_idx}: {e}")

    return results


def run_stage1(args):
    """Screen all tiles for best dates via per-tile SCL analysis."""
    with open(args.tile_locations) as f:
        all_tiles = json.load(f)
    with open(args.vpp_cache) as f:
        vpp_cache = json.load(f)

    output_dir = Path(args.output_dir)
    existing = {p.stem for p in output_dir.glob("*.npz")}

    best_dates_path = output_dir / "best_dates.json"
    best_dates = {}
    if best_dates_path.exists():
        with open(best_dates_path) as f:
            best_dates = json.load(f)

    # Filter: skip fetched tiles and already-screened tiles
    tiles = [
        t for t in all_tiles
        if t["name"] not in existing and t["name"] not in best_dates
    ]
    print(f"Total: {len(all_tiles)}, existing: {len(existing)}, "
          f"already screened: {len(best_dates)}, to screen: {len(tiles)}")

    conn = connect_cdse()

    for i, tile in enumerate(tiles):
        name = tile["name"]
        windows = vpp_cache.get(name)
        if not windows:
            windows = [[91, 150], [151, 210], [211, 270]]

        print(f"[{i+1}/{len(tiles)}] {name}...", end=" ", flush=True)
        result = screen_tile_scl(conn, tile, windows, args.year)

        if result:
            best_dates[name] = result
            dates = [f"f{k}={v['date']}({v['cloud_frac']:.0%})" for k, v in result.items()]
            print(f"OK: {', '.join(dates)}")
        else:
            print("no clear scenes")

        # Save every 10 tiles (resumable)
        if (i + 1) % 10 == 0:
            with open(best_dates_path, "w") as f:
                json.dump(best_dates, f, indent=2)

    # Final save
    with open(best_dates_path, "w") as f:
        json.dump(best_dates, f, indent=2)
    print(f"\n=== Stage 1 done: {len(best_dates)} tiles screened ===")


# ── Stage 2: Spectral Fetch (grouped by date) ───────────────────────────────

def run_stage2(args):
    """Fetch spectral for each tile using best dates from stage 1."""
    with open(args.best_dates) as f:
        best_dates = json.load(f)
    with open(args.tile_locations) as f:
        all_tiles = json.load(f)

    tile_lookup = {t["name"]: t for t in all_tiles}
    output_dir = Path(args.output_dir)
    existing = {p.stem for p in output_dir.glob("*.npz")}

    # Group by (date, frame_idx)
    by_date = defaultdict(list)
    for name, frames in best_dates.items():
        if name in existing or name not in tile_lookup:
            continue
        for frame_idx, info in frames.items():
            by_date[(info["date"], int(frame_idx))].append(tile_lookup[name])

    total_groups = len(by_date)
    total_tiles = len({t["name"] for group in by_date.values() for t in group})
    print(f"Tiles to fetch: {total_tiles}")
    print(f"Date groups: {total_groups}")

    conn = connect_cdse()

    for gi, ((date_str, frame_idx), group_tiles) in enumerate(sorted(by_date.items())):
        print(f"\n[{gi+1}/{total_groups}] {date_str} frame {frame_idx}: "
              f"{len(group_tiles)} tiles")

        merged = merge_bbox(group_tiles)
        bbox_wgs = bbox_to_wgs84(merged)

        # Single-date spectral fetch
        cube = conn.load_collection(
            "SENTINEL2_L2A",
            spatial_extent={
                "west": bbox_wgs["west"], "south": bbox_wgs["south"],
                "east": bbox_wgs["east"], "north": bbox_wgs["north"],
                "crs": "EPSG:4326",
            },
            temporal_extent=[date_str, date_str],
            bands=["B02", "B03", "B04", "B8A", "B11", "B12"],
        )
        cube = cube.reduce_dimension(dimension="t", reducer="first")

        job = cube.create_job(
            title=f"spec_{date_str}_f{frame_idx}_{len(group_tiles)}t",
            out_format="GTiff",
        )
        job.start_job()
        print(f"  Job {job.job_id}...", end=" ", flush=True)

        if not wait_for_job(job):
            continue

        # Download and split
        job_dir = output_dir / f"_spectral_{job.job_id}"
        job_dir.mkdir(parents=True, exist_ok=True)
        job.get_results().download_files(str(job_dir))

        n_ok = _split_to_frames(job_dir, group_tiles, frame_idx,
                                 output_dir, args.tile_size_px)
        print(f"  {n_ok}/{len(group_tiles)} tiles extracted")

    # Merge frames → .npz
    print(f"\n=== Merging frames ===")
    n_frames = max(
        int(fi) for frames in best_dates.values() for fi in frames
    ) + 1
    merged_count = 0
    for name in best_dates:
        if name in existing or name not in tile_lookup:
            continue
        if _merge_to_npz(name, n_frames, output_dir, tile_lookup[name], args.tile_size_px):
            merged_count += 1
    print(f"Merged {merged_count} tiles. Total: {len(list(output_dir.glob('*.npz')))}")


def _split_to_frames(job_dir, tiles, frame_idx, output_dir, tile_size_px):
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.enums import Resampling

    tifs = list(job_dir.glob("*.tif")) + list(job_dir.glob("*.tiff"))
    if not tifs:
        return 0

    count = 0
    with rasterio.open(tifs[0]) as src:
        for tile in tiles:
            bbox = tile["bbox_3006"]
            try:
                window = from_bounds(
                    bbox["west"], bbox["south"], bbox["east"], bbox["north"],
                    src.transform,
                )
                data = src.read(window=window,
                                out_shape=(6, tile_size_px, tile_size_px),
                                resampling=Resampling.nearest)
                spectral = data.astype(np.float32) / 10000.0
                np.save(output_dir / f"_frame_{tile['name']}_f{frame_idx}.npy", spectral)
                count += 1
            except Exception:
                pass
    return count


def _merge_to_npz(name, n_frames, output_dir, tile_loc, tile_size_px):
    frames, mask = [], []
    for fi in range(n_frames):
        fp = output_dir / f"_frame_{name}_f{fi}.npy"
        if fp.exists():
            frames.append(np.load(fp))
            mask.append(1)
        else:
            frames.append(np.zeros((6, tile_size_px, tile_size_px), dtype=np.float32))
            mask.append(0)

    if sum(mask) == 0:
        return False

    bbox = _normalize_bbox(tile_loc)
    np.savez_compressed(
        output_dir / f"{name}.npz",
        spectral=np.concatenate(frames, axis=0),
        temporal_mask=np.array(mask, dtype=np.uint8),
        doy=np.zeros(n_frames, dtype=np.int32),
        dates=np.array([""] * n_frames),
        multitemporal=np.int32(1),
        num_frames=np.int32(n_frames),
        num_bands=np.int32(6),
        bbox_3006=np.array([bbox["west"], bbox["south"],
                            bbox["east"], bbox["north"]], dtype=np.int32),
        easting=np.int32((bbox["west"] + bbox["east"]) // 2),
        northing=np.int32((bbox["south"] + bbox["north"]) // 2),
        source=tile_loc.get("source", "lulc"),
    )
    for fi in range(n_frames):
        (output_dir / f"_frame_{name}_f{fi}.npy").unlink(missing_ok=True)
    return True


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)

    s1 = sub.add_parser("stage1", help="SCL cloud screening per tile")
    s1.add_argument("--tile-locations", required=True)
    s1.add_argument("--vpp-cache", required=True)
    s1.add_argument("--output-dir", required=True)
    s1.add_argument("--year", type=int, default=2022)

    s2 = sub.add_parser("stage2", help="Fetch spectral for best dates")
    s2.add_argument("--best-dates", required=True)
    s2.add_argument("--tile-locations", required=True)
    s2.add_argument("--output-dir", required=True)
    s2.add_argument("--tile-size-px", type=int, default=512)

    args = parser.parse_args()
    if args.stage == "stage1":
        run_stage1(args)
    else:
        run_stage2(args)


if __name__ == "__main__":
    main()
