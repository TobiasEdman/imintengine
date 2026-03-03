#!/usr/bin/env python3
"""
scripts/test_batch_fetch.py — Compare sequential vs batch-job spectral fetch.

Tests whether openEO batch jobs reduce wall-clock time for seasonal tile
fetching.  Runs three strategies on a single tile:

  A) SEQUENTIAL (current) — 4 × fetch_seasonal_image() sync downloads
  B) MERGED SYNC         — build one multi-date datacube, 1 × download()
  C) BATCH JOB           — create_job() → start_and_wait() → download_files()

Usage:
    # Test DES (default):
    python scripts/test_batch_fetch.py

    # Test CDSE:
    python scripts/test_batch_fetch.py --source copernicus

    # Skip STAC/SCL — use hard-coded test dates:
    python scripts/test_batch_fetch.py --skip-discovery

    # Only run specific strategies:
    python scripts/test_batch_fetch.py --strategies sequential batch
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
import time
import tarfile
import tempfile
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imint.fetch import (
    _connect, _connect_cdse, _to_nmd_grid,
    _fetch_scl_batch, _snap_to_target_grid,
    fetch_seasonal_dates, fetch_seasonal_image,
    COLLECTION, BANDS_10M, BANDS_20M_SPECTRAL, BANDS_60M,
    CDSE_COLLECTION, CDSE_BANDS_10M, CDSE_BANDS_20M_SPECTRAL, CDSE_BANDS_60M,
    NMD_GRID_SIZE,
)
from imint.utils import dn_to_reflectance


# ── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_COORDS = {
    "west": 16.0, "south": 58.4,
    "east": 16.05, "north": 58.45,
}
DEFAULT_WINDOWS = [(4, 5), (6, 7), (8, 9), (1, 2)]
DEFAULT_YEARS = ["2019", "2018"]
PRITHVI_BANDS_DES = ["b02", "b03", "b04", "b8a", "b11", "b12"]
PRITHVI_BANDS_CDSE = ["B02", "B03", "B04", "B8A", "B11", "B12"]

# Hard-coded known-good dates for the default tile (skip STAC/SCL)
KNOWN_DATES = {
    "des": ["2019-04-27", "2019-07-11", "2019-08-20", "2019-01-17"],
    "copernicus": ["2019-04-27", "2019-07-11", "2019-08-20", "2019-01-17"],
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def _get_constants(source: str):
    """Return (collection, bands_10m, bands_20m, bands_60m, dn_source) for source."""
    if source == "copernicus":
        return (CDSE_COLLECTION, CDSE_BANDS_10M, CDSE_BANDS_20M_SPECTRAL,
                CDSE_BANDS_60M, "copernicus")
    return (COLLECTION, BANDS_10M, BANDS_20M_SPECTRAL, BANDS_60M, "des")


def _build_single_date_cube(conn, collection, bands_10m, bands_20m, bands_60m,
                             projected_coords, date_str):
    """Build a merged spectral datacube for a single date."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    temporal = [date_str, (dt + timedelta(days=1)).strftime("%Y-%m-%d")]

    cube_10m = conn.load_collection(
        collection_id=collection,
        spatial_extent=projected_coords,
        temporal_extent=temporal,
        bands=bands_10m,
    )
    cube_20m = conn.load_collection(
        collection_id=collection,
        spatial_extent=projected_coords,
        temporal_extent=temporal,
        bands=bands_20m,
    )
    cube_20m = cube_20m.resample_cube_spatial(target=cube_10m, method="bilinear")

    cube_60m = conn.load_collection(
        collection_id=collection,
        spatial_extent=projected_coords,
        temporal_extent=temporal,
        bands=bands_60m,
    )
    cube_60m = cube_60m.resample_cube_spatial(target=cube_10m, method="bilinear")

    return cube_10m.merge_cubes(cube_20m).merge_cubes(cube_60m)


def _parse_multidate_result(data: bytes, dates: list[str], dn_source: str,
                            projected_coords: dict):
    """Parse multi-date GeoTIFF / tar.gz result into per-date arrays.

    Returns:
        dict mapping date_str → (n_bands, H, W) float32 reflectance array.
    """
    import rasterio

    _DATE_RE = re.compile(r"(\d{4})[-_](\d{2})[-_](\d{2})")
    results: dict[str, np.ndarray] = {}
    target_bounds = {k: v for k, v in projected_coords.items() if k != "crs"}

    if isinstance(data, bytes) and data[:2] == b"\x1f\x8b":
        # tar.gz archive — one GeoTIFF per date
        tf = tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")
        for member in tf.getmembers():
            if not member.name.lower().endswith((".tif", ".tiff")):
                continue
            m = _DATE_RE.search(member.name)
            if not m:
                continue
            date_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
            if date_str not in dates:
                continue

            f = tf.extractfile(member)
            if f is None:
                continue

            with rasterio.open(io.BytesIO(f.read())) as src:
                raw = src.read()
                crs = src.crs
                transform = src.transform

            # Snap to NMD grid
            raw, transform = _snap_to_target_grid(
                raw, transform, crs, target_bounds, pixel_size=NMD_GRID_SIZE,
            )

            # DN → reflectance
            n_bands = raw.shape[0]
            refl = np.stack([
                dn_to_reflectance(raw[i], source=dn_source) for i in range(n_bands)
            ], axis=0).astype(np.float32)
            results[date_str] = refl

        tf.close()

    elif isinstance(data, bytes):
        # Single-date — plain GeoTIFF (happens when temporal range has 1 date)
        with rasterio.open(io.BytesIO(data)) as src:
            raw = src.read()
            crs = src.crs
            transform = src.transform

        raw, transform = _snap_to_target_grid(
            raw, transform, crs, target_bounds, pixel_size=NMD_GRID_SIZE,
        )

        n_bands = raw.shape[0]
        refl = np.stack([
            dn_to_reflectance(raw[i], source=dn_source) for i in range(n_bands)
        ], axis=0).astype(np.float32)

        # Map to first matching date
        if dates:
            results[dates[0]] = refl

    return results


# ── Strategy A: Sequential (current approach) ──────────────────────────────

def run_sequential(source: str, coords: dict, dates: list[str]) -> dict:
    """Fetch each date individually via fetch_seasonal_image()."""
    prithvi = PRITHVI_BANDS_CDSE if source == "copernicus" else PRITHVI_BANDS_DES
    results = {}

    for date_str in dates:
        print(f"    {date_str}...", end=" ", flush=True)
        t0 = time.monotonic()
        try:
            img_result = fetch_seasonal_image(date_str, coords, source=source)
            elapsed = time.monotonic() - t0
            if img_result is not None:
                img, dt = img_result
                results[date_str] = img
                print(f"OK ({elapsed:.1f}s, shape={img.shape})")
            else:
                print(f"NONE ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.monotonic() - t0
            print(f"FAIL ({elapsed:.1f}s): {e}")

    return results


# ── Strategy B: Merged sync download ───────────────────────────────────────

def run_merged_sync(source: str, coords: dict, dates: list[str],
                    conn) -> dict:
    """Build one multi-date datacube, download in one sync call."""
    collection, bands_10m, bands_20m, bands_60m, dn_source = _get_constants(source)
    projected_coords = _to_nmd_grid(coords)

    print(f"    Building merged cube for {len(dates)} dates...")

    # Build per-date cubes with narrow 1-day temporal extents
    cubes = []
    for date_str in dates:
        cube = _build_single_date_cube(
            conn, collection, bands_10m, bands_20m, bands_60m,
            projected_coords, date_str,
        )
        cubes.append(cube)
        print(f"      Added {date_str}")

    # Merge all date cubes into one
    merged = cubes[0]
    for c in cubes[1:]:
        merged = merged.merge_cubes(c)

    print(f"    Downloading merged cube ({len(dates)} dates × "
          f"{len(bands_10m)+len(bands_20m)+len(bands_60m)} bands)...",
          flush=True)

    t0 = time.monotonic()
    try:
        data = merged.download(format="gtiff")
        elapsed = time.monotonic() - t0
        print(f"    Download OK ({elapsed:.1f}s, {len(data)/1024:.0f} KB)")
    except Exception as e:
        elapsed = time.monotonic() - t0
        print(f"    Download FAILED ({elapsed:.1f}s): {e}")
        return {}

    # Parse multi-date result
    results = _parse_multidate_result(data, dates, dn_source, projected_coords)
    print(f"    Parsed {len(results)}/{len(dates)} dates")

    return results


# ── Strategy C: Batch job ──────────────────────────────────────────────────

def run_batch_job(source: str, coords: dict, dates: list[str],
                  conn) -> dict:
    """Submit as batch job, wait for completion, download results."""
    collection, bands_10m, bands_20m, bands_60m, dn_source = _get_constants(source)
    projected_coords = _to_nmd_grid(coords)

    print(f"    Building batch job datacube for {len(dates)} dates...")

    # Build per-date cubes
    cubes = []
    for date_str in dates:
        cube = _build_single_date_cube(
            conn, collection, bands_10m, bands_20m, bands_60m,
            projected_coords, date_str,
        )
        cubes.append(cube)
        print(f"      Added {date_str}")

    # Merge all
    merged = cubes[0]
    for c in cubes[1:]:
        merged = merged.merge_cubes(c)

    # Save result format
    merged = merged.save_result(format="GTiff")

    # Create and start batch job
    print("    Creating batch job...", flush=True)
    t0 = time.monotonic()
    try:
        job = merged.create_job(title=f"seasonal_batch_test_{source}")
        print(f"    Job created: {job.job_id}")

        print("    Starting job and waiting for completion...", flush=True)
        job.start_and_wait(
            max_poll_interval=10,
            print=lambda msg: print(f"      {msg}"),
        )
        elapsed_processing = time.monotonic() - t0
        print(f"    Job finished in {elapsed_processing:.1f}s")

    except Exception as e:
        elapsed = time.monotonic() - t0
        print(f"    Batch job FAILED ({elapsed:.1f}s): {e}")
        return {}

    # Download results
    print("    Downloading results...", flush=True)
    t_dl = time.monotonic()
    try:
        result_obj = job.get_results()
        # Download to temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            result_files = result_obj.download_files(tmpdir)
            elapsed_dl = time.monotonic() - t_dl
            total_elapsed = time.monotonic() - t0

            # List downloaded files
            downloaded = list(Path(tmpdir).glob("*"))
            print(f"    Downloaded {len(downloaded)} files "
                  f"({elapsed_dl:.1f}s download, {total_elapsed:.1f}s total)")

            for fp in downloaded:
                print(f"      {fp.name} ({fp.stat().st_size / 1024:.0f} KB)")

            # Parse results — try each file
            results: dict[str, np.ndarray] = {}
            target_bounds = {k: v for k, v in projected_coords.items()
                            if k != "crs"}

            for fp in downloaded:
                if fp.suffix.lower() in (".tif", ".tiff"):
                    # Single GeoTIFF per date or multi-band
                    import rasterio
                    with rasterio.open(str(fp)) as src:
                        raw = src.read()
                        crs = src.crs
                        transform = src.transform

                    raw, transform = _snap_to_target_grid(
                        raw, transform, crs, target_bounds,
                        pixel_size=NMD_GRID_SIZE,
                    )

                    n_bands = raw.shape[0]
                    refl = np.stack([
                        dn_to_reflectance(raw[i], source=dn_source)
                        for i in range(n_bands)
                    ], axis=0).astype(np.float32)

                    # Try to extract date from filename
                    date_re = re.compile(r"(\d{4})[-_](\d{2})[-_](\d{2})")
                    m = date_re.search(fp.name)
                    if m:
                        date_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                        results[date_str] = refl
                    else:
                        # Can't determine date — assign to first unmatched
                        for d in dates:
                            if d not in results:
                                results[d] = refl
                                break

                elif fp.suffix.lower() in (".gz", ".tar"):
                    # tar.gz bundle
                    file_data = fp.read_bytes()
                    batch_results = _parse_multidate_result(
                        file_data, dates, dn_source, projected_coords,
                    )
                    results.update(batch_results)

            print(f"    Parsed {len(results)}/{len(dates)} dates")
            return results

    except Exception as e:
        elapsed = time.monotonic() - t0
        print(f"    Result download FAILED ({elapsed:.1f}s): {e}")
        return {}


# ── Date discovery ──────────────────────────────────────────────────────────

def discover_dates(source: str, coords: dict, conn,
                   windows: list[tuple[int, int]],
                   years: list[str]) -> list[str]:
    """Find one good date per seasonal window via STAC + SCL screening."""
    projected = _to_nmd_grid(coords)

    print("  STAC date discovery...")
    season_candidates = fetch_seasonal_dates(
        coords, windows, years, scene_cloud_max=50.0,
    )

    good_dates = []
    for win_idx, (m_start, m_end) in enumerate(windows):
        candidates = season_candidates[win_idx]
        win_label = f"m{m_start}-{m_end}"

        if not candidates:
            print(f"    {win_label}: no STAC candidates")
            continue

        # SCL batch pre-screen
        cand_strs = [d for d, _ in candidates[:8]]
        good_date = None

        if len(cand_strs) > 1:
            try:
                batch_results = _fetch_scl_batch(
                    conn, projected, cand_strs, source=source,
                )
                for cand_date, aoi_cloud in batch_results:
                    if aoi_cloud <= 0.10:
                        good_date = cand_date
                        break
            except Exception as e:
                print(f"    {win_label}: SCL batch failed: {e}")

        # Fallback: take least cloudy STAC candidate
        if good_date is None and candidates:
            good_date = candidates[0][0]

        if good_date:
            good_dates.append(good_date)
            print(f"    {win_label}: {good_date}")

    return good_dates


# ── Comparison ──────────────────────────────────────────────────────────────

def compare_results(results_a: dict, results_b: dict, label_a: str,
                    label_b: str):
    """Compare spectral arrays from two strategies."""
    common = set(results_a.keys()) & set(results_b.keys())
    if not common:
        print("  No common dates to compare")
        return

    for date_str in sorted(common):
        a = results_a[date_str]
        b = results_b[date_str]

        if a.shape != b.shape:
            print(f"  {date_str}: shape mismatch {a.shape} vs {b.shape}")
            continue

        diff = np.abs(a - b)
        print(f"  {date_str}: mean_abs_diff={diff.mean():.6f}, "
              f"max_abs_diff={diff.max():.6f}, "
              f"shapes={a.shape}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare sequential vs batch-job spectral fetch"
    )
    parser.add_argument("--source", default="des",
                        choices=["des", "copernicus"],
                        help="Backend to test (default: des)")
    parser.add_argument("--skip-discovery", action="store_true",
                        help="Use hard-coded dates (skip STAC/SCL)")
    parser.add_argument("--strategies", nargs="+",
                        default=["sequential", "merged", "batch"],
                        choices=["sequential", "merged", "batch"],
                        help="Which strategies to run")

    # Tile coordinates
    parser.add_argument("--west", type=float, default=DEFAULT_COORDS["west"])
    parser.add_argument("--south", type=float, default=DEFAULT_COORDS["south"])
    parser.add_argument("--east", type=float, default=DEFAULT_COORDS["east"])
    parser.add_argument("--north", type=float, default=DEFAULT_COORDS["north"])

    parser.add_argument("--years", default="2019,2018")
    parser.add_argument("--windows", default="4-5,6-7,8-9,1-2")

    args = parser.parse_args()

    coords = {
        "west": args.west, "south": args.south,
        "east": args.east, "north": args.north,
    }
    years = args.years.split(",")
    windows = [(int(w.split("-")[0]), int(w.split("-")[1]))
               for w in args.windows.split(",")]

    print("=" * 65)
    print("  Batch Fetch Benchmark")
    print(f"  Source: {args.source.upper()}")
    print(f"  Tile:  ({coords['west']}, {coords['south']}) → "
          f"({coords['east']}, {coords['north']})")
    print(f"  Strategies: {', '.join(args.strategies)}")
    print("=" * 65)

    # ── Connect ─────────────────────────────────────────────────────────
    print(f"\n  Connecting to {args.source.upper()}...", flush=True)
    t0 = time.monotonic()
    if args.source == "copernicus":
        conn = _connect_cdse()
    else:
        conn = _connect()
    print(f"  Connected ({time.monotonic() - t0:.1f}s)")

    # ── Discover dates ──────────────────────────────────────────────────
    if args.skip_discovery:
        dates = KNOWN_DATES.get(args.source, KNOWN_DATES["des"])
        print(f"\n  Using hard-coded dates: {dates}")
    else:
        print()
        dates = discover_dates(args.source, coords, conn, windows, years)
        print(f"\n  Selected {len(dates)} dates: {dates}")

    if not dates:
        print("\n  No dates found — cannot benchmark")
        sys.exit(1)

    # ── Run strategies ──────────────────────────────────────────────────
    all_results = {}
    timings = {}

    # A) Sequential
    if "sequential" in args.strategies:
        print(f"\n{'─' * 65}")
        print("  Strategy A: SEQUENTIAL (current — 1 download per date)")
        print(f"{'─' * 65}")
        t0 = time.monotonic()
        all_results["sequential"] = run_sequential(args.source, coords, dates)
        timings["sequential"] = time.monotonic() - t0
        print(f"  Total: {timings['sequential']:.1f}s, "
              f"{len(all_results['sequential'])}/{len(dates)} dates OK")

    # B) Merged sync
    if "merged" in args.strategies:
        print(f"\n{'─' * 65}")
        print("  Strategy B: MERGED SYNC (1 merged multi-date download)")
        print(f"{'─' * 65}")
        t0 = time.monotonic()
        all_results["merged"] = run_merged_sync(
            args.source, coords, dates, conn,
        )
        timings["merged"] = time.monotonic() - t0
        print(f"  Total: {timings['merged']:.1f}s, "
              f"{len(all_results['merged'])}/{len(dates)} dates OK")

    # C) Batch job
    if "batch" in args.strategies:
        print(f"\n{'─' * 65}")
        print("  Strategy C: BATCH JOB (create_job → start_and_wait)")
        print(f"{'─' * 65}")
        t0 = time.monotonic()
        all_results["batch"] = run_batch_job(
            args.source, coords, dates, conn,
        )
        timings["batch"] = time.monotonic() - t0
        print(f"  Total: {timings['batch']:.1f}s, "
              f"{len(all_results['batch'])}/{len(dates)} dates OK")

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 65}")
    print(f"  Source: {args.source.upper()}")
    print(f"  Dates:  {len(dates)}")
    print()

    for strategy, elapsed in sorted(timings.items(), key=lambda x: x[1]):
        n_ok = len(all_results.get(strategy, {}))
        per_date = elapsed / max(n_ok, 1)
        print(f"  {strategy:12s}: {elapsed:6.1f}s total, "
              f"{per_date:5.1f}s/date, {n_ok}/{len(dates)} dates OK")

    # Best vs worst
    if len(timings) >= 2:
        sorted_t = sorted(timings.items(), key=lambda x: x[1])
        fastest = sorted_t[0]
        slowest = sorted_t[-1]
        speedup = slowest[1] / max(fastest[1], 0.1)
        print(f"\n  Speedup: {fastest[0]} is {speedup:.1f}× faster "
              f"than {slowest[0]}")

    # ── Cross-compare reflectance ───────────────────────────────────────
    strategies_with_results = [s for s in all_results if all_results[s]]
    if len(strategies_with_results) >= 2:
        print(f"\n{'─' * 65}")
        print("  REFLECTANCE COMPARISON")
        print(f"{'─' * 65}")
        for i, s1 in enumerate(strategies_with_results):
            for s2 in strategies_with_results[i + 1:]:
                print(f"\n  {s1} vs {s2}:")
                compare_results(
                    all_results[s1], all_results[s2], s1, s2,
                )

    print()


if __name__ == "__main__":
    main()
