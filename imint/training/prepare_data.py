"""
imint/training/prepare_data.py — Fetch and cache LULC training data

Orchestrates bulk fetching of Sentinel-2 + NMD tile pairs with progress
tracking for resumability.

Data fetch pipeline:
    STAC discovery:   DES STAC (explorer.digitalearth.se)
    Spectral + SCL:   CDSE Sentinel Hub Process API (primary, fast HTTP)
                      openEO (fallback if HTTP fails)
    NMD labels:       DES openEO
    Aux channels:     SKS WMS, Copernicus DEM, HR-VPP

NMD pre-filter and STAC/spectral fetch run in parallel: a background
thread filters cells via NMD while worker threads fetch spectral data
for cells that have already been approved.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import socket
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ── Global socket timeout — prevents openEO from hanging indefinitely ──────
_SOCKET_TIMEOUT = 90  # seconds; applies to all socket operations
socket.setdefaulttimeout(_SOCKET_TIMEOUT)

_MAX_RETRIES = 2      # retries per cell on transient/timeout errors
_SCL_CANDIDATES = 3   # max STAC dates to pre-screen with SCL before giving up

# ── Adaptive concurrency defaults ────────────────────────────────────────
_INITIAL_WORKERS = 3
_MIN_WORKERS = 1
_MAX_WORKERS = 3
_ADAPT_WINDOW = 10          # requests to consider for adaptation
_LATENCY_HIGH_S = 60.0      # p90 above this → scale down
_LATENCY_LOW_S = 30.0       # p90 below this → scale up
_ERROR_RATE_HIGH = 0.30     # >30 % errors → scale down

from .config import TrainingConfig
from .class_schema import nmd_raster_to_lulc
from .sampler import (
    generate_grid, grid_to_wgs84, split_by_latitude,
    densify_grid, generate_densification_regions,
    filter_land_cells, filter_sea_cells_swedish_waters,
)
from .scb_tatort import generate_scb_densification_regions
from .skg_sumpskog import generate_sumpskog_densification_regions
from .skg_height import fetch_height_tile
from .skg_grunddata import fetch_volume_tile, fetch_basal_area_tile, fetch_diameter_tile
from .copernicus_dem import fetch_dem_tile

# Sentinel value to signal that the NMD producer is done
_NMD_DONE = None

# Adaptive concurrency controller — adjusts fetch parallelism to DES health
class _AdaptiveWorkerPool:
    """Tracks DES response latency/errors and adjusts active worker count.

    Workers check ``may_proceed()`` before starting a DES request.
    If the pool wants fewer active workers it blocks extras on a semaphore
    until conditions improve.
    """

    def __init__(
        self,
        initial: int = _INITIAL_WORKERS,
        lo: int = _MIN_WORKERS,
        hi: int = _MAX_WORKERS,
    ):
        self._lock = threading.Lock()
        self._active = initial
        self._lo = lo
        self._hi = hi
        self._sem = threading.Semaphore(initial)
        self._latencies: list[float] = []   # last N request durations (seconds)
        self._errors: list[bool] = []        # last N: True = error
        self._last_adjust = time.monotonic()

    @property
    def active(self) -> int:
        with self._lock:
            return self._active

    def acquire(self) -> None:
        """Block until a worker slot is available."""
        self._sem.acquire()

    def release(self) -> None:
        self._sem.release()

    def record(self, latency_s: float, error: bool) -> None:
        """Record a DES request outcome and maybe adjust concurrency."""
        with self._lock:
            self._latencies.append(latency_s)
            self._errors.append(error)
            # Keep only the window
            if len(self._latencies) > _ADAPT_WINDOW:
                self._latencies = self._latencies[-_ADAPT_WINDOW:]
                self._errors = self._errors[-_ADAPT_WINDOW:]
            # Only adapt after a full window and at most every 30 s
            if (len(self._latencies) < _ADAPT_WINDOW
                    or time.monotonic() - self._last_adjust < 30):
                return
            self._maybe_adapt()

    def _maybe_adapt(self) -> None:
        """Adjust concurrency based on recent DES behaviour (called with lock)."""
        lats = sorted(self._latencies)
        p90 = lats[int(len(lats) * 0.9)]
        err_rate = sum(self._errors) / len(self._errors)

        old = self._active

        if err_rate > _ERROR_RATE_HIGH or p90 > _LATENCY_HIGH_S:
            # Scale down
            new = max(self._lo, self._active - 1)
        elif err_rate == 0 and p90 < _LATENCY_LOW_S:
            # Scale up
            new = min(self._hi, self._active + 1)
        else:
            new = self._active

        if new != old:
            diff = new - old
            self._active = new
            self._last_adjust = time.monotonic()
            # Reset window after adjustment so we evaluate fresh data
            self._latencies.clear()
            self._errors.clear()
            if diff > 0:
                for _ in range(diff):
                    self._sem.release()
                print(f"    ⚡ DES adaptive: {old} → {new} workers "
                      f"(p90={p90:.0f}s, err={err_rate:.0%})")
            else:
                # Draining: next -diff acquire() calls will block
                for _ in range(-diff):
                    # Non-blocking drain; if can't drain now the
                    # natural acquire/release flow will catch up.
                    self._sem.acquire(blocking=False)
                print(f"    ⚠ DES adaptive: {old} → {new} workers "
                      f"(p90={p90:.0f}s, err={err_rate:.0%})")

# Static count used for NMD_DONE sentinels — start with max possible
_FETCH_WORKERS = _MAX_WORKERS


def _record_des_call(
    stats: dict, call_type: str, *, cell_key=None, date=None,
    latency_s=0.0, band_count=0, pixel_dims=None,
    response_bytes=0, success=True, error=None,
    retry=0, from_cache=False,
):
    """Record a single DES API call to the stats buffer (thread-safe)."""
    record = {
        "call_type": call_type, "cell_key": cell_key, "date": date,
        "band_count": band_count, "latency_s": round(latency_s, 2),
        "response_bytes": response_bytes,
        "pixel_dims": list(pixel_dims) if pixel_dims else None,
        "success": success,
        "error": type(error).__name__ if error else None,
        "retry": retry, "from_cache": from_cache,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    with stats["lock"]:
        s = stats["summary"][call_type]
        s["count"] += 1
        s["latency_sum"] += latency_s
        s["bytes_sum"] += response_bytes
        if not success:
            s["errors"] += 1
        stats["buffer"].append(record)
        if len(stats["buffer"]) >= 20:
            with open(stats["log_path"], "a") as f:
                for r in stats["buffer"]:
                    f.write(json.dumps(r) + "\n")
            stats["buffer"].clear()


def _flush_des_stats(stats: dict):
    """Flush remaining buffered stats to disk."""
    with stats["lock"]:
        if stats["buffer"]:
            with open(stats["log_path"], "a") as f:
                for r in stats["buffer"]:
                    f.write(json.dumps(r) + "\n")
            stats["buffer"].clear()


def prepare_training_data(config: TrainingConfig) -> None:
    """Main entry point for training data preparation.

    NMD pre-filter and spectral data fetch run concurrently:
    a producer thread filters cells via NMD and pushes approved cells
    into a queue; consumer threads pick them up for STAC + DES fetch.
    """
    from ..fetch import (
        fetch_des_data, fetch_copernicus_data, fetch_nmd_data,
        _connect, _connect_cdse,
        _stac_available_dates, _fetch_scl, _fetch_scl_batch,
        _to_nmd_grid, FetchError,
        fetch_seasonal_dates, fetch_seasonal_image,
        fetch_sentinel2_data,
    )

    data_dir = Path(config.data_dir)
    tiles_dir = data_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Generate grid ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  LULC Training Data Preparation")
    print(f"  Grid spacing: {config.grid_spacing_m}m")
    print(f"  Years: {config.years}")
    if config.enable_multitemporal:
        print(f"  Mode: MULTITEMPORAL ({config.num_temporal_frames} frames)")
        for i, (ms, me) in enumerate(config.seasonal_windows[:config.num_temporal_frames]):
            print(f"    Frame {i}: months {ms}-{me}")
    else:
        print(f"  Mode: Single-date (growing season months "
              f"{config.growing_season[0]}-{config.growing_season[1]})")
    _aux = []
    if config.enable_height_channel:
        _aux.append("height")
    if config.enable_volume_channel:
        _aux.append("volume")
    if config.enable_basal_area_channel:
        _aux.append("basal_area")
    if config.enable_diameter_channel:
        _aux.append("diameter")
    if config.enable_dem_channel:
        _aux.append("dem")
    if _aux:
        print(f"  Aux channels: {', '.join(_aux)}")
    print(f"  Fetch sources: {', '.join(s.upper() for s in config.fetch_sources)}")
    print(f"{'='*60}")

    patch_size_m = config.fetch_pixels * 10  # 10m resolution
    cache_dir = data_dir / "cache"

    # Generate base grid (disable built-in land filter if we need sea cells)
    if config.enable_sea_densification:
        cells = generate_grid(
            spacing_m=config.grid_spacing_m,
            patch_size_m=patch_size_m,
            land_filter=False,
        )
        cells, sea_cells = filter_land_cells(cells, return_sea_cells=True)
        coastal_cells = filter_sea_cells_swedish_waters(
            sea_cells,
            max_distance_m=config.max_sea_distance_m,
            cache_dir=cache_dir,
        )
        for cell in coastal_cells:
            cell.skip_land_filter = True
        cells = cells + coastal_cells
        print(f"  Sea densification: +{len(coastal_cells)} coastal water cells")
    else:
        cells = generate_grid(
            spacing_m=config.grid_spacing_m,
            patch_size_m=patch_size_m,
        )
    base_count = len(cells)

    # Optional: densify grid in predefined rare-class areas
    if config.enable_grid_densification:
        regions = generate_densification_regions()
        cells = densify_grid(
            cells,
            densification_regions=regions,
            densify_spacing_m=config.densify_spacing_m,
            patch_size_m=patch_size_m,
        )
        print(f"\n  Grid: {base_count} base + "
              f"{len(cells) - base_count} densified = {len(cells)} locations")
    else:
        print(f"\n  Grid: {len(cells)} candidate locations")

    # Optional: SCB tätort urban densification
    if config.enable_scb_densification:
        scb_regions = generate_scb_densification_regions(
            cache_dir=cache_dir,
            min_population=config.scb_min_population,
            patch_size_m=patch_size_m,
        )
        pre_scb = len(cells)
        cells = densify_grid(
            cells,
            densification_regions=scb_regions,
            densify_spacing_m=config.scb_densify_spacing_m,
            patch_size_m=patch_size_m,
        )
        print(f"  SCB urban: +{len(cells) - pre_scb} cells "
              f"({len(scb_regions)} tätort regions)")

    # Optional: Skogsstyrelsen sumpskog densification
    if config.enable_sumpskog_densification:
        skg_regions = generate_sumpskog_densification_regions(
            cache_dir=cache_dir,
            min_density_pct=config.sumpskog_min_density_pct,
        )
        pre_skg = len(cells)
        cells = densify_grid(
            cells,
            densification_regions=skg_regions,
            densify_spacing_m=config.sumpskog_densify_spacing_m,
            patch_size_m=patch_size_m,
        )
        print(f"  Sumpskog: +{len(cells) - pre_skg} cells "
              f"({len(skg_regions)} wetland regions)")

    cells = grid_to_wgs84(cells)

    # ── Step 2: Split by latitude ─────────────────────────────────────
    train_cells, val_cells, test_cells = split_by_latitude(
        cells,
        val_lat_min=config.val_latitude_min,
        val_lat_max=config.val_latitude_max,
        test_lat_min=config.test_latitude_min,
    )
    print(f"  Split: train={len(train_cells)}, val={len(val_cells)}, test={len(test_cells)}")

    # ── Step 3: Load progress ─────────────────────────────────────────
    progress_path = data_dir / "progress.json"
    progress = _load_progress(progress_path)
    completed = set(progress.get("completed", []))
    failed = set(progress.get("failed", []))
    print(f"  Progress: {len(completed)} completed, {len(failed)} failed")

    # ── Step 4: Verify data source connections ───────────────────────
    # Multitemporal mode uses CDSE Sentinel Hub Process API (HTTP) for
    # spectral data — no openEO needed.  DES STAC is used for date
    # discovery (plain HTTP, no connection required).
    # Single-temporal mode still uses openEO.
    sources = config.fetch_sources
    connections = {}
    if not config.enable_multitemporal:
        for src in sources:
            if src == "copernicus":
                print("\n  Verifying CDSE connection...")
                connections["copernicus"] = _connect_cdse()
                print("  CDSE connection OK")
            else:
                print(f"\n  Verifying DES connection...")
                connections["des"] = _connect()
                print("  DES connection OK")
    else:
        print("\n  Multitemporal mode: using CDSE Sentinel Hub HTTP")
        print("  Date discovery: DES STAC (explorer.digitalearth.se)")
        # No openEO connections needed
        for src in sources:
            connections[src] = None  # Placeholder for worker distribution

    # Worker distribution: more workers to faster source (CDSE)
    workers_per_source = {}
    n_sources = len(sources)
    if n_sources == 1:
        workers_per_source[sources[0]] = _MAX_WORKERS
    elif n_sources >= 2:
        if "copernicus" in sources:
            workers_per_source["copernicus"] = max(2, _MAX_WORKERS // 2 + 1)
            for s in sources:
                if s != "copernicus":
                    workers_per_source[s] = max(
                        1, _MAX_WORKERS - workers_per_source["copernicus"],
                    )
        else:
            for s in sources:
                workers_per_source[s] = max(1, _MAX_WORKERS // n_sources)

    # Backward compat: single conn for code that still uses it directly
    conn = connections.get("des") or connections.get("copernicus")

    # ── Build cell list with split labels ─────────────────────────────
    all_cells_with_split = []
    for split_name, split_cells in [("train", train_cells),
                                     ("val", val_cells),
                                     ("test", test_cells)]:
        for cell in split_cells:
            all_cells_with_split.append((split_name, cell))

    total_cells = len(all_cells_with_split)

    # ── Thread-safe shared state ──────────────────────────────────────
    lock = threading.Lock()
    class_counts = {}
    tile_histograms = {}
    tile_names_by_split = {"train": [], "val": [], "test": []}
    tile_idx_box = [len(completed)]  # mutable counter in a list
    tiles_at_session_start = len(completed)
    from collections import deque
    _tile_timestamps = deque()  # timestamps of recent tile completions

    # ── Restore class distribution from existing tiles on disk ───────
    if completed and tiles_dir.exists():
        print(f"\n  Restoring class distribution from {len(completed)} existing tiles...")
        restored = 0
        for npz_path in tiles_dir.glob("tile_*.npz"):
            tile_name = npz_path.name
            cell_key = tile_name.replace("tile_", "").replace(".npz", "")
            if cell_key not in completed:
                continue
            try:
                with np.load(npz_path) as td:
                    labels = td["label"]
                tile_hist = {}
                for cls_idx in range(config.num_classes + 1):
                    count = int((labels == cls_idx).sum())
                    if count > 0:
                        tile_hist[cls_idx] = count
                for cls_idx, count in tile_hist.items():
                    class_counts[cls_idx] = class_counts.get(cls_idx, 0) + count
                tile_histograms[tile_name] = tile_hist
                restored += 1
            except Exception:
                pass  # skip corrupt tiles
        print(f"  Restored class counts from {restored} tiles")

    # ── DES API call statistics ────────────────────────────────────────
    des_stats = {
        "lock": threading.Lock(),
        "log_path": data_dir / "des_api_stats.jsonl",
        "buffer": [],
        "summary": {
            "stac_search":   {"count": 0, "latency_sum": 0.0, "bytes_sum": 0, "errors": 0},
            "scl_batch":     {"count": 0, "latency_sum": 0.0, "bytes_sum": 0, "errors": 0},
            "scl_prescreen": {"count": 0, "latency_sum": 0.0, "bytes_sum": 0, "errors": 0},
            "full_spectral": {"count": 0, "latency_sum": 0.0, "bytes_sum": 0, "errors": 0},
            "nmd_fetch":     {"count": 0, "latency_sum": 0.0, "bytes_sum": 0, "errors": 0},
        },
    }

    # Queue for NMD-approved cells → spectral fetch workers
    approved_q = queue.Queue(maxsize=200)

    # Adaptive concurrency pools — one per source for independent tracking
    pools = {}
    for src in sources:
        n = workers_per_source[src]
        pools[src] = _AdaptiveWorkerPool(
            initial=n, lo=_MIN_WORKERS, hi=n,
        )
    # Backward compat: single pool reference for code that uses it
    pool = pools.get("des") or pools.get("copernicus")

    # ── NMD pre-filter log ────────────────────────────────────────────
    nmd_log_path = data_dir / "nmd_prefilter_log.json"
    nmd_log = {
        "status": "running",
        "total_cells": total_cells,
        "processed": 0,
        "land_kept": 0,
        "water_skipped": 0,
        "failed": 0,
        "already_done": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": None,
    }
    _write_prepare_log(nmd_log_path, nmd_log)

    # ── Prepare log for dashboard ─────────────────────────────────────
    prep_log_path = data_dir / "prepare_log.json"
    # Restore cumulative elapsed time from previous runs
    prev_elapsed = 0.0
    if prep_log_path.exists():
        try:
            with open(prep_log_path) as f:
                prev_log = json.load(f)
            prev_elapsed = prev_log.get("elapsed_s", 0.0)
        except Exception:
            pass
    prep_log = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": None,
        "grid_cells": total_cells,
        "completed": len(completed),
        "failed": len(failed),
        "current_split": "",
        "tiles_saved": tile_idx_box[0],
        "latest_tile": "",
        "latest_date": "",
        "latest_cloud": 0.0,
        "class_counts": {str(k): v for k, v in class_counts.items()},
        "elapsed_s": prev_elapsed,
        "recent_previews": [],
    }

    # Restore recent previews from existing tiles on disk
    if completed and tiles_dir.exists():
        existing_previews = sorted(
            tiles_dir.glob("preview_*.png"),
            key=lambda p: p.stat().st_mtime,
        )
        if existing_previews:
            prep_log["recent_previews"] = [
                p.name for p in existing_previews[-3:]
            ]
    t_start = time.time()
    _write_prepare_log(prep_log_path, prep_log)

    # ── NMD producer thread ───────────────────────────────────────────
    def _nmd_producer():
        water_skipped = 0
        nmd_failed = 0
        land_kept = 0

        for ci, (split_name, cell) in enumerate(all_cells_with_split):
            cell_key = f"{cell.easting}_{cell.northing}"

            # Already processed in a previous run
            with lock:
                already = cell_key in completed or cell_key in failed
            if already:
                nmd_log["already_done"] = nmd_log.get("already_done", 0) + 1
                nmd_log["processed"] = ci + 1
                # Restore counters for previously-processed cells
                with lock:
                    if cell_key in completed:
                        land_kept += 1
                        tile_name = f"tile_{cell_key}.npz"
                        if (tiles_dir / tile_name).exists():
                            tile_names_by_split[split_name].append(tile_name)
                    elif cell_key in failed:
                        water_skipped += 1  # most failed cells are water/empty
                nmd_log["land_kept"] = land_kept
                nmd_log["water_skipped"] = water_skipped
                if (ci + 1) % 50 == 0:
                    nmd_log["updated_at"] = datetime.now(timezone.utc).isoformat()
                    _write_prepare_log(nmd_log_path, nmd_log)
                continue

            coords_nmd = {
                "west": cell.west_wgs84, "east": cell.east_wgs84,
                "south": cell.south_wgs84, "north": cell.north_wgs84,
            }
            nmd_ok = False
            for attempt in range(_MAX_RETRIES + 1):
                try:
                    t0_nmd = time.monotonic()
                    nmd_result = fetch_nmd_data(coords=coords_nmd)
                    _record_des_call(
                        des_stats, "nmd_fetch", cell_key=cell_key,
                        latency_s=time.monotonic() - t0_nmd,
                        band_count=1, success=True, retry=attempt,
                        from_cache=nmd_result.from_cache,
                    )
                    # Polite pause if we had to hit DES (not cache)
                    if not nmd_result.from_cache:
                        time.sleep(1)
                    labels = nmd_raster_to_lulc(
                        nmd_result.nmd_raster, num_classes=config.num_classes,
                    )
                    # Skip tiles that are entirely nodata (background)
                    # — unless it's a sea-densified cell (known water)
                    valid_frac = float(np.mean(labels > 0))
                    is_sea_cell = getattr(cell, "skip_land_filter", False)
                    if valid_frac < 0.01 and not is_sea_cell:
                        water_skipped += 1
                        with lock:
                            failed.add(cell_key)
                        print(f"    NMD {cell_key}: SKIP nodata — "
                              f"valid={valid_frac:.1%}")
                    else:
                        land_kept += 1
                        # Push approved cell to fetch queue
                        approved_q.put((split_name, cell, cell_key))
                    nmd_ok = True
                    break
                except Exception as e:
                    _record_des_call(
                        des_stats, "nmd_fetch", cell_key=cell_key,
                        latency_s=time.monotonic() - t0_nmd,
                        band_count=1, success=False, error=e, retry=attempt,
                    )
                    if attempt < _MAX_RETRIES:
                        wait = 5 * (attempt + 1)
                        print(f"    NMD {cell_key}: retry {attempt+1} "
                              f"after {type(e).__name__}, wait {wait}s")
                        time.sleep(wait)
                    else:
                        nmd_failed += 1
                        with lock:
                            failed.add(cell_key)
                        err_msg = str(e)
                        if len(err_msg) > 80:
                            err_msg = err_msg[:77] + "..."
                        print(f"    NMD fail {cell_key}: "
                              f"{type(e).__name__}: {err_msg}")

            # Update NMD log
            nmd_log["processed"] = ci + 1
            nmd_log["land_kept"] = land_kept
            nmd_log["water_skipped"] = water_skipped
            nmd_log["failed"] = nmd_failed
            nmd_log["updated_at"] = datetime.now(timezone.utc).isoformat()
            if (ci + 1) % 5 == 0 or ci + 1 == total_cells:
                _write_prepare_log(nmd_log_path, nmd_log)

            if (ci + 1) % 50 == 0:
                print(f"    NMD pre-filter: {ci+1}/{total_cells} "
                      f"(kept {land_kept}, water={water_skipped}, "
                      f"fail={nmd_failed})")

        # Signal done
        nmd_log["status"] = "completed"
        nmd_log["processed"] = total_cells
        nmd_log["land_kept"] = land_kept
        nmd_log["water_skipped"] = water_skipped
        nmd_log["failed"] = nmd_failed
        nmd_log["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_prepare_log(nmd_log_path, nmd_log)

        with lock:
            _save_progress(progress_path, completed, failed)

        print(f"  NMD pre-filter done: {land_kept} land, "
              f"{water_skipped} water, {nmd_failed} failed")

        # Signal all fetch workers to stop
        n_fetch_workers = sum(workers_per_source.values())
        for _ in range(n_fetch_workers):
            approved_q.put(_NMD_DONE)

    # ── Spectral fetch worker ─────────────────────────────────────────
    def _fetch_worker():
        while True:
            item = approved_q.get()
            if item is _NMD_DONE:
                break

            split_name, cell, cell_key = item
            coords = {
                "west": cell.west_wgs84, "east": cell.east_wgs84,
                "south": cell.south_wgs84, "north": cell.north_wgs84,
            }
            tile_name = f"tile_{cell_key}.npz"
            success = False

            # Alternate starting year & month per cell for temporal
            # variation.  hash(cell_key) is deterministic so restarts
            # give the same assignment regardless of processing order.
            cell_hash = int(hashlib.md5(cell_key.encode()).hexdigest(), 16)
            years_offset = cell_hash % len(config.years)
            years_order = config.years[years_offset:] + config.years[:years_offset]

            for year in years_order:
                try:
                    from datetime import timedelta as _td

                    m_start = config.growing_season[0]
                    m_end = config.growing_season[1]
                    months = list(range(m_end, m_start - 1, -1))
                    # Rotate month order per cell for within-season variation
                    month_offset = (cell_hash // len(config.years)) % len(months)
                    months = months[month_offset:] + months[:month_offset]

                    projected = _to_nmd_grid(coords)
                    good_date = None
                    good_cloud = None

                    for month in months:
                        if good_date:
                            break
                        date_start = f"{year}-{month:02d}-01"
                        date_end = f"{year}-{month:02d}-28"
                        if month in (1,3,5,7,8,10,12):
                            date_end = f"{year}-{month:02d}-31"
                        elif month in (4,6,9,11):
                            date_end = f"{year}-{month:02d}-30"

                        t0_stac = time.monotonic()
                        stac_ok = True
                        try:
                            dates = _stac_available_dates(
                                coords, date_start, date_end,
                                scene_cloud_max=50,
                            )
                        except Exception as e:
                            stac_ok = False
                            _record_des_call(
                                des_stats, "stac_search", cell_key=cell_key,
                                date=f"{date_start}/{date_end}",
                                latency_s=time.monotonic() - t0_stac,
                                success=False, error=e,
                            )
                            raise
                        _record_des_call(
                            des_stats, "stac_search", cell_key=cell_key,
                            date=f"{date_start}/{date_end}",
                            latency_s=time.monotonic() - t0_stac,
                            success=True,
                        )
                        if not dates:
                            continue

                        # SCL pre-screen — try batch (1 call), fall back to per-date
                        top_dates = dates[:_SCL_CANDIDATES]
                        cand_strs = [d for d, _ in top_dates]
                        batch_ok = False

                        if len(cand_strs) > 1:
                            pool.acquire()
                            t0_batch = time.monotonic()
                            batch_err = False
                            try:
                                batch_results = _fetch_scl_batch(
                                    conn, projected, cand_strs,
                                )
                                batch_ok = True
                                for cand_date, aoi_cloud in batch_results:
                                    if aoi_cloud <= config.cloud_threshold:
                                        good_date = cand_date
                                        good_cloud = aoi_cloud
                                        break
                                    else:
                                        print(f"    SCL {cell_key} {cand_date}: "
                                              f"cloud={aoi_cloud:.1%} > "
                                              f"{config.cloud_threshold:.0%}, skip")
                            except Exception as e:
                                batch_err = True
                                err_msg = str(e)
                                if len(err_msg) > 80:
                                    err_msg = err_msg[:77] + "..."
                                print(f"    SCL batch {cell_key}: "
                                      f"{type(e).__name__}: {err_msg}")
                            finally:
                                batch_lat = time.monotonic() - t0_batch
                                pool.record(batch_lat, batch_err)
                                pool.release()
                                _record_des_call(
                                    des_stats, "scl_batch",
                                    cell_key=cell_key,
                                    date="/".join(cand_strs),
                                    latency_s=batch_lat,
                                    band_count=len(cand_strs),
                                    success=batch_ok,
                                )
                            time.sleep(1)

                        # Fall back to per-date if batch failed or only 1 candidate
                        if not batch_ok and good_date is None:
                            for cand_date, scene_cloud in top_dates:
                                try:
                                    cand_dt = datetime.strptime(cand_date, "%Y-%m-%d")
                                    temporal = [
                                        cand_date,
                                        (cand_dt + _td(days=1)).strftime("%Y-%m-%d"),
                                    ]
                                    pool.acquire()
                                    t0 = time.monotonic()
                                    scl_err = False
                                    try:
                                        _scl, aoi_cloud, _crs, _tr = _fetch_scl(
                                            conn, projected, temporal, date_window=0,
                                        )
                                    except Exception:
                                        scl_err = True
                                        raise
                                    finally:
                                        scl_lat = time.monotonic() - t0
                                        pool.record(scl_lat, scl_err)
                                        pool.release()
                                        _record_des_call(
                                            des_stats, "scl_prescreen",
                                            cell_key=cell_key, date=cand_date,
                                            latency_s=scl_lat, band_count=1,
                                            success=not scl_err,
                                        )
                                    time.sleep(1)

                                    if aoi_cloud <= config.cloud_threshold:
                                        good_date = cand_date
                                        good_cloud = aoi_cloud
                                        break
                                    else:
                                        print(f"    SCL {cell_key} {cand_date}: "
                                              f"cloud={aoi_cloud:.1%} > "
                                              f"{config.cloud_threshold:.0%}, skip")
                                except Exception as e:
                                    err_msg = str(e)
                                    if len(err_msg) > 80:
                                        err_msg = err_msg[:77] + "..."
                                    print(f"    SCL {cell_key} {cand_date}: "
                                          f"{type(e).__name__}: {err_msg}")
                                    time.sleep(1)

                    if good_date is None:
                        with lock:
                            n_done = len(completed) + len(failed)
                        print(f"    \u2717 [{n_done}/{total_cells}] {cell_key} "
                              f"{year}: no clear date found")
                        continue

                    # ── Full spectral fetch (SCL already verified) ─────
                    pool.acquire()
                    t0_spec = time.monotonic()
                    spec_err = False
                    try:
                        result = fetch_des_data(
                            date=good_date,
                            coords=coords,
                            cloud_threshold=1.0,    # already screened
                            include_scl=False,      # skip redundant SCL
                        )
                    except Exception:
                        spec_err = True
                        raise
                    finally:
                        spec_lat = time.monotonic() - t0_spec
                        pool.record(spec_lat, spec_err)
                        pool.release()
                        _record_des_call(
                            des_stats, "full_spectral",
                            cell_key=cell_key, date=good_date,
                            latency_s=spec_lat, band_count=11,
                            pixel_dims=(result.bands["B02"].shape
                                        if not spec_err else None),
                            success=not spec_err,
                        )

                    # Save RGB preview
                    preview_fname = f"preview_{cell_key}_{good_date}.png"
                    try:
                        from PIL import Image as _PILImage
                        rgb_u8 = (result.rgb * 255).clip(0, 255).astype(np.uint8)
                        _PILImage.fromarray(rgb_u8).save(
                            tiles_dir / preview_fname)
                    except Exception:
                        pass

                    # Check Prithvi bands
                    missing = [b for b in config.prithvi_bands
                               if b not in result.bands]
                    if missing:
                        continue

                    # Stack bands
                    image = np.stack(
                        [result.bands[b] for b in config.prithvi_bands],
                        axis=0,
                    ).astype(np.float32)

                    # ── Nodata quality gate (band misalignment) ────────
                    # Sentinel-2 orbit edges can have 10m bands (B02-B04)
                    # with much less coverage than 20m bands (B8A, B11).
                    nodata_frac = float((image[0] == 0).mean())
                    if nodata_frac > 0.10:
                        print(f"    nodata {cell_key} {good_date}: "
                              f"B02 zeros={nodata_frac:.0%}, reject")
                        continue

                    # ── B02 haze quality gate (thin cloud detection) ───
                    # B02 (blue) is first Prithvi band; high mean on
                    # SCL-clear pixels indicates residual haze/thin cloud
                    b02_idx = config.prithvi_bands.index("B02") \
                        if "B02" in config.prithvi_bands else 0
                    b02_mean = float(image[b02_idx].mean())
                    if b02_mean > config.b02_haze_threshold:
                        print(f"    haze {cell_key} {good_date}: "
                              f"B02={b02_mean:.4f} > "
                              f"{config.b02_haze_threshold}, reject")
                        continue

                    # Fetch NMD (cached) with target_shape
                    nmd_result = fetch_nmd_data(
                        coords=coords,
                        target_shape=image.shape[1:],
                    )
                    labels = nmd_raster_to_lulc(
                        nmd_result.nmd_raster,
                        num_classes=config.num_classes,
                    )

                    # ── Optional: height channel ──────────────────────
                    save_kwargs = dict(
                        image=image,
                        label=labels,
                        easting=cell.easting,
                        northing=cell.northing,
                        date=good_date,
                        lat=cell.center_lat,
                        lon=cell.center_lon,
                    )
                    _spatial = (image.shape[-2], image.shape[-1])
                    _aux_cache = (lambda sub: cache_dir / sub
                                  if config.aux_cache_enabled
                                  else None)
                    if config.enable_height_channel:
                        try:
                            height = fetch_height_tile(
                                cell.west_3006, cell.south_3006,
                                cell.east_3006, cell.north_3006,
                                size_px=_spatial,
                                cache_dir=_aux_cache("height"),
                            )
                            save_kwargs["height"] = height
                        except Exception as e:
                            print(f"    height {cell_key}: "
                                  f"{type(e).__name__}: {e}")
                    if config.enable_volume_channel:
                        try:
                            volume = fetch_volume_tile(
                                cell.west_3006, cell.south_3006,
                                cell.east_3006, cell.north_3006,
                                size_px=_spatial,
                                cache_dir=_aux_cache("volume"),
                            )
                            save_kwargs["volume"] = volume
                        except Exception as e:
                            print(f"    volume {cell_key}: "
                                  f"{type(e).__name__}: {e}")
                    if config.enable_basal_area_channel:
                        try:
                            basal_area = fetch_basal_area_tile(
                                cell.west_3006, cell.south_3006,
                                cell.east_3006, cell.north_3006,
                                size_px=_spatial,
                                cache_dir=_aux_cache("basal_area"),
                            )
                            save_kwargs["basal_area"] = basal_area
                        except Exception as e:
                            print(f"    basal_area {cell_key}: "
                                  f"{type(e).__name__}: {e}")
                    if config.enable_diameter_channel:
                        try:
                            diameter = fetch_diameter_tile(
                                cell.west_3006, cell.south_3006,
                                cell.east_3006, cell.north_3006,
                                size_px=_spatial,
                                cache_dir=_aux_cache("diameter"),
                            )
                            save_kwargs["diameter"] = diameter
                        except Exception as e:
                            print(f"    diameter {cell_key}: "
                                  f"{type(e).__name__}: {e}")
                    if config.enable_dem_channel:
                        try:
                            dem = fetch_dem_tile(
                                cell.west_3006, cell.south_3006,
                                cell.east_3006, cell.north_3006,
                                size_px=_spatial,
                                cache_dir=_aux_cache("dem"),
                            )
                            save_kwargs["dem"] = dem
                        except Exception as e:
                            print(f"    dem {cell_key}: "
                                  f"{type(e).__name__}: {e}")

                    # Save tile
                    np.savez_compressed(
                        tiles_dir / tile_name, **save_kwargs,
                    )

                    # Compute class histogram
                    tile_hist = {}
                    for cls_idx in range(config.num_classes + 1):
                        count = int((labels == cls_idx).sum())
                        if count > 0:
                            tile_hist[cls_idx] = count

                    # Update shared state
                    with lock:
                        for cls_idx, count in tile_hist.items():
                            class_counts[cls_idx] = class_counts.get(cls_idx, 0) + count
                        tile_histograms[tile_name] = tile_hist
                        tile_names_by_split[split_name].append(tile_name)
                        completed.add(cell_key)
                        tile_idx_box[0] += 1
                        cur_idx = tile_idx_box[0]

                        prep_log["completed"] = len(completed)
                        prep_log["failed"] = len(failed)
                        prep_log["tiles_saved"] = cur_idx
                        prep_log["latest_tile"] = tile_name
                        prep_log["latest_date"] = good_date
                        prep_log["latest_cloud"] = round(good_cloud, 4)
                        prep_log["class_counts"] = {
                            str(k): v for k, v in class_counts.items()
                        }
                        prep_log["elapsed_s"] = round(prev_elapsed + time.time() - t_start, 1)
                        # Rolling 5-minute rate for ETA
                        now = time.time()
                        _tile_timestamps.append(now)
                        # Trim to last 5 minutes
                        cutoff = now - 300
                        while _tile_timestamps and _tile_timestamps[0] < cutoff:
                            _tile_timestamps.popleft()
                        window = now - _tile_timestamps[0] if len(_tile_timestamps) > 1 else (now - t_start)
                        n_in_window = len(_tile_timestamps)
                        prep_log["session_rate"] = round(n_in_window / max(window, 1) * 3600, 1)
                        prep_log["active_workers"] = pool.active
                        prep_log["updated_at"] = datetime.now(timezone.utc).isoformat()

                        # Track recent preview for dashboard
                        recent = prep_log.get("recent_previews", [])
                        recent.append(preview_fname)
                        prep_log["recent_previews"] = recent[-3:]

                        # DES API call summary for dashboard
                        with des_stats["lock"]:
                            prep_log["des_api_summary"] = {
                                k: {
                                    **v,
                                    "avg_latency_s": round(
                                        v["latency_sum"] / max(v["count"], 1), 1
                                    ),
                                }
                                for k, v in des_stats["summary"].items()
                            }

                        _write_prepare_log(prep_log_path, prep_log)

                        if cur_idx % 50 == 0:
                            _save_progress(progress_path, completed, failed)

                    rate = prep_log.get("session_rate", 0)
                    print(f"    \u2713 [{cur_idx}/{total_cells}] {tile_name} "
                          f"({good_date}, cloud={good_cloud:.1%}) "
                          f"[{rate:.0f}/h, {pool.active}w]")
                    success = True
                    time.sleep(1)  # polite pause between DES requests
                    break  # No need to try other years

                except Exception as e:
                    err_msg = str(e)
                    if len(err_msg) > 80:
                        err_msg = err_msg[:77] + "..."
                    print(f"    {cell_key} {year}: {type(e).__name__}: {err_msg}")
                    time.sleep(3)  # back off after errors
                    continue

            if not success:
                with lock:
                    failed.add(cell_key)
                    prep_log["failed"] = len(failed)
                    prep_log["elapsed_s"] = round(prev_elapsed + time.time() - t_start, 1)
                    prep_log["updated_at"] = datetime.now(timezone.utc).isoformat()
                    _write_prepare_log(prep_log_path, prep_log)

    # ── Multitemporal spectral fetch worker ─────────────────────────
    def _fetch_worker_multitemporal(worker_conn=None, worker_pool=None,
                                     worker_source="des"):
        """Fetch T seasonal frames per cell and save as multitemporal tile.

        For each cell:
        1. Query DES STAC for each seasonal window across all years
        2. Try top candidates via CDSE Sentinel Hub Process API (HTTP):
           - Fetches spectral + SCL in one call
           - Checks cloud, nodata, haze locally
           - Falls back to openEO if HTTP fails
        3. Stack into (T*6, H, W) and save .npz with dates array

        Falls back gracefully: if a season has no clear image,
        that frame is filled with zeros and flagged in the mask.
        """
        # openEO conn/pool kept for fallback if CDSE HTTP fails
        _conn = worker_conn if worker_conn is not None else conn
        _pool = worker_pool if worker_pool is not None else pool
        _source = worker_source  # "des" or "copernicus" — used for openEO fallback

        n_frames = config.num_temporal_frames
        default_windows = config.seasonal_windows[:n_frames]
        prithvi_bands = config.prithvi_bands
        n_bands = len(prithvi_bands)
        use_vpp_guided = getattr(config, 'enable_vpp_guided_windows', False)

        while True:
            item = approved_q.get()
            if item is _NMD_DONE:
                break

            split_name, cell, cell_key = item
            coords = {
                "west": cell.west_wgs84, "east": cell.east_wgs84,
                "south": cell.south_wgs84, "north": cell.north_wgs84,
            }
            tile_name = f"tile_{cell_key}.npz"
            success = False

            # Deterministic year rotation per cell
            cell_hash = int(hashlib.md5(cell_key.encode()).hexdigest(), 16)
            years_offset = cell_hash % len(config.years)
            years_order = config.years[years_offset:] + config.years[:years_offset]

            try:
                projected = _to_nmd_grid(coords)

                # ── VPP-guided per-tile windows ─────────────────────────
                doy_windows = None
                windows = default_windows
                if use_vpp_guided:
                    try:
                        from .cdse_vpp import fetch_vpp_tiles
                        from .vpp_windows import (
                            compute_growing_season_windows,
                        )
                        vpp = fetch_vpp_tiles(
                            west=projected["west"],
                            south=projected["south"],
                            east=projected["east"],
                            north=projected["north"],
                            size_px=64,
                        )
                        doy_windows = compute_growing_season_windows(
                            vpp["sosd"], vpp["eosd"],
                            num_frames=n_frames,
                        )
                    except Exception as e:
                        print(f"    {cell_key} VPP failed ({e}), "
                              f"using default windows")

                # ── Phase 1: STAC discovery per season ──────────────────
                if doy_windows is not None:
                    from ..fetch import fetch_seasonal_dates_doy
                    season_candidates = fetch_seasonal_dates_doy(
                        coords, doy_windows, years_order,
                        scene_cloud_max=50.0,
                    )
                    active_windows = doy_windows
                else:
                    season_candidates = fetch_seasonal_dates(
                        coords, windows, years_order,
                        scene_cloud_max=50.0,
                    )
                    active_windows = windows

                # ── Phase 2: Spectral + SCL fetch per season ────────────
                # CDSE Sentinel Hub Process API: single HTTP POST per
                # candidate — fetches spectral + SCL, checks cloud/nodata/
                # haze locally.  No openEO fallback (same CDSE platform,
                # same rate limits).
                from .cdse_s2 import fetch_s2_scene

                frames = []       # list of (6, H, W) arrays
                frame_dates = []  # list of date strings
                frame_mask = []   # 1 = valid, 0 = padded

                # Compute EPSG:3006 bbox for Sentinel Hub
                sh_west = projected["west"]
                sh_south = projected["south"]
                sh_east = projected["east"]
                sh_north = projected["north"]
                sh_size = config.fetch_pixels

                for win_idx, (w_start, w_end) in enumerate(active_windows):
                    candidates = season_candidates[win_idx]
                    if doy_windows is not None:
                        win_label = f"doy{w_start}-{w_end}"
                    else:
                        win_label = f"m{w_start}-{w_end}"

                    if not candidates:
                        print(f"    {cell_key} {win_label}: no STAC candidates")
                        frames.append(None)
                        frame_dates.append("")
                        frame_mask.append(0)
                        continue

                    # Try top candidates — each is one fast HTTP call
                    # (spectral + SCL + cloud/nodata/haze gates in one shot)
                    top_cands = candidates[:_SCL_CANDIDATES]
                    found = False

                    for cand_date, scene_cc in top_cands:
                        t0 = time.monotonic()
                        try:
                            result = fetch_s2_scene(
                                sh_west, sh_south, sh_east, sh_north,
                                date=cand_date,
                                size_px=sh_size,
                                cloud_threshold=config.seasonal_cloud_threshold,
                                haze_threshold=config.b02_haze_threshold,
                            )
                        except Exception:
                            result = None
                        spec_lat = time.monotonic() - t0
                        _record_des_call(
                            des_stats, "full_spectral",
                            cell_key=cell_key, date=cand_date,
                            latency_s=spec_lat, band_count=n_bands,
                            success=result is not None,
                        )

                        if result is not None:
                            frame_img, _scl, cloud_frac = result
                            frames.append(frame_img)
                            frame_dates.append(cand_date)
                            frame_mask.append(1)
                            print(f"    {cell_key} {win_label}: "
                                  f"{cand_date} ✓ (cloud={cloud_frac:.0%})")
                            found = True
                            break
                        time.sleep(1.5)  # Rate limit: ~1 req/s per worker

                    if not found:
                        print(f"    {cell_key} {win_label}: "
                              f"no clear date (tried {len(top_cands)})")
                        frames.append(None)
                        frame_dates.append("")
                        frame_mask.append(0)

                # ── Check minimum frames ────────────────────────────────
                n_valid = sum(frame_mask)
                if config.seasonal_require_all and n_valid < n_frames:
                    with lock:
                        n_done = len(completed) + len(failed)
                    print(f"    ✗ [{n_done}/{total_cells}] {cell_key}: "
                          f"only {n_valid}/{n_frames} seasonal frames")
                    continue
                if n_valid == 0:
                    with lock:
                        n_done = len(completed) + len(failed)
                    print(f"    ✗ [{n_done}/{total_cells}] {cell_key}: "
                          f"no valid frames")
                    continue

                # ── Stack frames into (T*6, H, W) ──────────────────────
                # Use the shape from the first valid frame
                ref_shape = None
                for f in frames:
                    if f is not None:
                        ref_shape = f.shape[1:]  # (H, W)
                        break

                stacked_frames = []
                for f in frames:
                    if f is not None:
                        # Ensure consistent shape (may differ by ±1 pixel)
                        if f.shape[1:] != ref_shape:
                            padded = np.zeros(
                                (n_bands, ref_shape[0], ref_shape[1]),
                                dtype=np.float32,
                            )
                            h = min(f.shape[1], ref_shape[0])
                            w = min(f.shape[2], ref_shape[1])
                            padded[:, :h, :w] = f[:, :h, :w]
                            stacked_frames.append(padded)
                        else:
                            stacked_frames.append(f)
                    else:
                        # Zero-padded frame for missing season
                        stacked_frames.append(
                            np.zeros((n_bands,) + ref_shape, dtype=np.float32)
                        )

                # (T*6, H, W) — interleaved: [t0_b02, t0_b03, ..., t1_b02, ...]
                image = np.concatenate(stacked_frames, axis=0)

                # ── Fetch NMD labels (same as single-date) ──────────────
                nmd_result = fetch_nmd_data(
                    coords=coords,
                    target_shape=ref_shape,
                )
                labels = nmd_raster_to_lulc(
                    nmd_result.nmd_raster,
                    num_classes=config.num_classes,
                )

                # Sea-densified cells with no NMD coverage: assign water
                is_sea_cell = getattr(cell, "skip_land_filter", False)
                if is_sea_cell and float(np.mean(labels > 0)) < 0.01:
                    water_class = (
                        10 if config.num_classes == 10
                        else 19  # 19-class: water_sea
                    )
                    labels = np.full_like(labels, water_class)

                # ── Compute day-of-year for temporal position encoding ──
                doy_values = []
                for d_str in frame_dates:
                    if d_str:
                        dt = datetime.strptime(d_str, "%Y-%m-%d")
                        doy_values.append(dt.timetuple().tm_yday)
                    else:
                        doy_values.append(0)

                # ── Save tile ───────────────────────────────────────────
                save_kwargs = dict(
                    image=image,                         # (T*6, H, W)
                    label=labels,                        # (H, W)
                    easting=cell.easting,
                    northing=cell.northing,
                    dates=np.array(frame_dates),          # (T,) strings
                    doy=np.array(doy_values, dtype=np.int32),  # (T,)
                    temporal_mask=np.array(frame_mask, dtype=np.uint8),
                    num_frames=n_frames,
                    num_bands=n_bands,
                    lat=cell.center_lat,
                    lon=cell.center_lon,
                    multitemporal=True,
                )
                _spatial = (ref_shape[0], ref_shape[1])
                _aux_cache = (lambda sub: cache_dir / sub
                              if config.aux_cache_enabled
                              else None)
                if config.enable_height_channel:
                    try:
                        height = fetch_height_tile(
                            cell.west_3006, cell.south_3006,
                            cell.east_3006, cell.north_3006,
                            size_px=_spatial,
                            cache_dir=_aux_cache("height"),
                        )
                        save_kwargs["height"] = height
                    except Exception as e:
                        print(f"    height {cell_key}: "
                              f"{type(e).__name__}: {e}")
                if config.enable_volume_channel:
                    try:
                        volume = fetch_volume_tile(
                            cell.west_3006, cell.south_3006,
                            cell.east_3006, cell.north_3006,
                            size_px=_spatial,
                            cache_dir=_aux_cache("volume"),
                        )
                        save_kwargs["volume"] = volume
                    except Exception as e:
                        print(f"    volume {cell_key}: "
                              f"{type(e).__name__}: {e}")
                if config.enable_basal_area_channel:
                    try:
                        basal_area = fetch_basal_area_tile(
                            cell.west_3006, cell.south_3006,
                            cell.east_3006, cell.north_3006,
                            size_px=_spatial,
                            cache_dir=_aux_cache("basal_area"),
                        )
                        save_kwargs["basal_area"] = basal_area
                    except Exception as e:
                        print(f"    basal_area {cell_key}: "
                              f"{type(e).__name__}: {e}")
                if config.enable_diameter_channel:
                    try:
                        diameter = fetch_diameter_tile(
                            cell.west_3006, cell.south_3006,
                            cell.east_3006, cell.north_3006,
                            size_px=_spatial,
                            cache_dir=_aux_cache("diameter"),
                        )
                        save_kwargs["diameter"] = diameter
                    except Exception as e:
                        print(f"    diameter {cell_key}: "
                              f"{type(e).__name__}: {e}")
                if config.enable_dem_channel:
                    try:
                        dem = fetch_dem_tile(
                            cell.west_3006, cell.south_3006,
                            cell.east_3006, cell.north_3006,
                            size_px=_spatial,
                            cache_dir=_aux_cache("dem"),
                        )
                        save_kwargs["dem"] = dem
                    except Exception as e:
                        print(f"    dem {cell_key}: "
                              f"{type(e).__name__}: {e}")

                np.savez_compressed(
                    tiles_dir / tile_name, **save_kwargs,
                )

                # Compute class histogram
                tile_hist = {}
                for cls_idx in range(config.num_classes + 1):
                    count = int((labels == cls_idx).sum())
                    if count > 0:
                        tile_hist[cls_idx] = count

                # Update shared state
                primary_date = next(
                    (d for d in frame_dates if d), "unknown"
                )
                with lock:
                    for cls_idx, count in tile_hist.items():
                        class_counts[cls_idx] = class_counts.get(cls_idx, 0) + count
                    tile_histograms[tile_name] = tile_hist
                    tile_names_by_split[split_name].append(tile_name)
                    completed.add(cell_key)
                    tile_idx_box[0] += 1
                    cur_idx = tile_idx_box[0]

                    prep_log["completed"] = len(completed)
                    prep_log["failed"] = len(failed)
                    prep_log["tiles_saved"] = cur_idx
                    prep_log["latest_tile"] = tile_name
                    prep_log["latest_date"] = primary_date
                    prep_log["class_counts"] = {
                        str(k): v for k, v in class_counts.items()
                    }
                    prep_log["elapsed_s"] = round(
                        prev_elapsed + time.time() - t_start, 1)
                    now = time.time()
                    _tile_timestamps.append(now)
                    cutoff = now - 300
                    while _tile_timestamps and _tile_timestamps[0] < cutoff:
                        _tile_timestamps.popleft()
                    window = (now - _tile_timestamps[0]
                              if len(_tile_timestamps) > 1
                              else (now - t_start))
                    n_in_window = len(_tile_timestamps)
                    prep_log["session_rate"] = round(
                        n_in_window / max(window, 1) * 3600, 1)
                    prep_log["active_workers"] = pool.active
                    prep_log["updated_at"] = datetime.now(
                        timezone.utc).isoformat()
                    _write_prepare_log(prep_log_path, prep_log)

                    if cur_idx % 50 == 0:
                        _save_progress(progress_path, completed, failed)

                rate = prep_log.get("session_rate", 0)
                frames_str = "/".join(
                    f"{'✓' if m else '✗'}" for m in frame_mask
                )
                print(f"    ✓ [{cur_idx}/{total_cells}] {tile_name} "
                      f"[{frames_str}] {n_valid}/{n_frames} frames "
                      f"[{rate:.0f}/h, {pool.active}w]")
                success = True

            except Exception as e:
                err_msg = str(e)
                if len(err_msg) > 80:
                    err_msg = err_msg[:77] + "..."
                print(f"    {cell_key}: {type(e).__name__}: {err_msg}")
                time.sleep(3)

            if not success:
                with lock:
                    failed.add(cell_key)
                    prep_log["failed"] = len(failed)
                    prep_log["elapsed_s"] = round(
                        prev_elapsed + time.time() - t_start, 1)
                    prep_log["updated_at"] = datetime.now(
                        timezone.utc).isoformat()
                    _write_prepare_log(prep_log_path, prep_log)

    # ── System metrics background thread ─────────────────────────────
    _metrics_stop = threading.Event()

    def _system_metrics_writer():
        """Write system-wide metrics (CPU, RAM, network delta since start)."""
        try:
            import psutil
        except ImportError:
            return
        psutil.cpu_percent()  # prime (always returns 0 on first call)
        net_baseline = psutil.net_io_counters()
        metrics_path = data_dir / "system_metrics.json"
        while not _metrics_stop.is_set():
            try:
                vm = psutil.virtual_memory()
                net = psutil.net_io_counters()
                metrics = {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": round(vm.percent, 1),
                    "memory_used_gb": round(vm.used / (1024**3), 2),
                    "memory_total_gb": round(vm.total / (1024**3), 1),
                    "device": "cpu",
                    "gpu_percent": None,
                    "gpu_memory_used_gb": None,
                    "gpu_memory_total_gb": None,
                    "net_sent_mb": round(
                        (net.bytes_sent - net_baseline.bytes_sent) / (1024**2), 1),
                    "net_recv_mb": round(
                        (net.bytes_recv - net_baseline.bytes_recv) / (1024**2), 1),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                tmp = metrics_path.with_suffix(".json.tmp")
                with open(tmp, "w") as f:
                    json.dump(metrics, f, indent=2)
                tmp.rename(metrics_path)
            except Exception:
                pass
            _metrics_stop.wait(5)

    metrics_thread = threading.Thread(target=_system_metrics_writer, daemon=True)
    metrics_thread.start()

    # ── Launch parallel pipeline ──────────────────────────────────────
    total_workers = sum(workers_per_source.values())
    source_desc = ", ".join(
        f"{n}×{s.upper()}" for s, n in workers_per_source.items()
    )
    print(f"\n  Starting parallel pipeline: NMD filter + "
          f"{total_workers} fetch workers ({source_desc})...")

    nmd_thread = threading.Thread(target=_nmd_producer, daemon=True)
    nmd_thread.start()

    # Spawn workers per source — each gets its own connection + pool
    fetch_threads = []
    for src in sources:
        n_workers = workers_per_source[src]
        src_conn = connections[src]
        src_pool = pools[src]
        for _ in range(n_workers):
            if config.enable_multitemporal:
                t = threading.Thread(
                    target=_fetch_worker_multitemporal,
                    kwargs=dict(
                        worker_conn=src_conn,
                        worker_pool=src_pool,
                        worker_source=src,
                    ),
                    daemon=True,
                )
            else:
                t = threading.Thread(target=_fetch_worker, daemon=True)
            t.start()
            fetch_threads.append(t)

    # Wait for NMD producer to finish
    nmd_thread.join()
    # Wait for all fetch workers to drain the queue and finish
    for t in fetch_threads:
        t.join()

    print(f"\n  All workers done.")

    # ── Step 6: Save progress, splits, and stats ──────────────────────
    _flush_des_stats(des_stats)
    _save_progress(progress_path, completed, failed)

    for split_name, names in tile_names_by_split.items():
        split_path = data_dir / f"split_{split_name}.txt"
        with open(split_path, "w") as f:
            f.write("\n".join(sorted(names)) + "\n")
        print(f"  {split_name}: {len(names)} tiles \u2192 {split_path}")

    # Compute tile weights for rare-class oversampling
    tile_weights = _compute_tile_weights(
        tile_histograms, class_counts,
        num_classes=config.num_classes,
        rare_threshold=config.rare_class_threshold,
        max_weight=config.max_tile_weight,
        ignore_index=config.ignore_index,
    )

    # Save class statistics
    stats = {
        "class_counts": class_counts,
        "tile_weights": tile_weights,
        "rare_classes": _find_rare_classes(
            class_counts, config.num_classes,
            config.rare_class_threshold, config.ignore_index,
        ),
        "total_tiles": tile_idx_box[0],
        "num_classes": config.num_classes,
    }
    stats_path = data_dir / "class_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Class stats \u2192 {stats_path}")

    # Report rare classes
    if stats["rare_classes"]:
        print(f"  Rare classes (<{config.rare_class_threshold:.0%} of pixels):")
        for cls_idx in stats["rare_classes"]:
            total_px = sum(class_counts.values())
            frac = class_counts.get(cls_idx, 0) / max(total_px, 1)
            print(f"    class {cls_idx:2d}: {frac:.4%}")
        n_boosted = sum(1 for w in tile_weights.values() if w > 1.01)
        print(f"  Tiles with boosted weight: {n_boosted}/{len(tile_weights)}")

    # Final prepare log update
    prep_log["status"] = "completed"
    prep_log["completed"] = len(completed)
    prep_log["failed"] = len(failed)
    prep_log["tiles_saved"] = tile_idx_box[0]
    prep_log["class_counts"] = {str(k): v for k, v in class_counts.items()}
    prep_log["elapsed_s"] = round(prev_elapsed + time.time() - t_start, 1)
    prep_log["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_prepare_log(prep_log_path, prep_log)

    # Stop system metrics writer
    _metrics_stop.set()

    print(f"\n  Done! {tile_idx_box[0]} tiles saved to {tiles_dir}")


def _write_prepare_log(path: Path, log: dict) -> None:
    """Atomic write of prepare_log.json for dashboard polling."""
    try:
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(log, f, indent=2)
        tmp.rename(path)
    except Exception:
        pass


def _load_progress(path: Path) -> dict:
    """Load progress checkpoint, merging with actual tile files on disk.

    Always cross-checks progress.json against tiles/ directory so that
    tiles saved after the last checkpoint are not re-fetched.
    """
    saved_completed: set[str] = set()
    saved_failed: set[str] = set()

    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            saved_completed = set(data.get("completed", []))
            saved_failed = set(data.get("failed", []))
        except (json.JSONDecodeError, KeyError):
            pass

    # Cross-check against actual tile files on disk
    tiles_dir = path.parent / "tiles"
    disk_keys: set[str] = set()
    if tiles_dir.exists():
        for npz in tiles_dir.glob("tile_*.npz"):
            key = npz.stem.replace("tile_", "")
            disk_keys.add(key)

    # Merge: tiles on disk are definitely completed
    merged = saved_completed | disk_keys
    if len(merged) > len(saved_completed):
        added = len(merged) - len(saved_completed)
        print(f"  Recovered {added} tiles from disk "
              f"({len(saved_completed)} in checkpoint, "
              f"{len(disk_keys)} on disk)")

    if merged:
        progress = {
            "completed": sorted(merged),
            "failed": sorted(saved_failed - disk_keys),
        }
        # Persist the merged state
        try:
            tmp = path.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump(progress, f)
            tmp.rename(path)
        except Exception:
            pass
        return progress

    return {"completed": [], "failed": []}


def _save_progress(path: Path, completed: set, failed: set) -> None:
    """Save progress checkpoint (atomic write to prevent corruption)."""
    try:
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump({
                "completed": sorted(completed),
                "failed": sorted(failed),
            }, f)
        tmp.rename(path)
    except Exception:
        pass


def _find_rare_classes(
    class_counts: dict[int, int],
    num_classes: int,
    threshold: float,
    ignore_index: int = 0,
) -> list[int]:
    """Identify classes whose pixel fraction is below threshold.

    Args:
        class_counts: Dict mapping class index to total pixel count.
        num_classes: Number of classes.
        threshold: Fraction below which a class is considered rare.
        ignore_index: Background class to exclude.

    Returns:
        List of rare class indices.
    """
    total = sum(v for k, v in class_counts.items() if k != ignore_index)
    if total == 0:
        return []

    rare = []
    for cls_idx in range(num_classes + 1):
        if cls_idx == ignore_index:
            continue
        frac = class_counts.get(cls_idx, 0) / total
        if frac < threshold:
            rare.append(cls_idx)
    return rare


def _compute_tile_weights(
    tile_histograms: dict[str, dict[int, int]],
    class_counts: dict[int, int],
    num_classes: int,
    rare_threshold: float = 0.02,
    max_weight: float = 5.0,
    ignore_index: int = 0,
) -> dict[str, float]:
    """Compute per-tile sampling weights based on rare-class content.

    Tiles containing pixels from rare classes get higher weights so
    they are sampled more frequently during training.

    Algorithm:
        1. Find rare classes (global pixel fraction < threshold)
        2. For each tile, compute rarity_score = fraction of pixels
           belonging to rare classes
        3. weight = 1.0 + (max_weight - 1.0) * rarity_score

    Args:
        tile_histograms: Per-tile class pixel counts.
        class_counts: Global class pixel counts.
        num_classes: Number of classes.
        rare_threshold: Classes below this fraction are "rare".
        max_weight: Maximum tile weight.
        ignore_index: Background class to exclude.

    Returns:
        Dict mapping tile name to sampling weight.
    """
    rare_classes = set(_find_rare_classes(
        class_counts, num_classes, rare_threshold, ignore_index,
    ))

    if not rare_classes:
        # No rare classes → uniform weights
        return {name: 1.0 for name in tile_histograms}

    weights = {}
    for tile_name, hist in tile_histograms.items():
        total_px = sum(v for k, v in hist.items() if k != ignore_index)
        if total_px == 0:
            weights[tile_name] = 1.0
            continue

        rare_px = sum(v for k, v in hist.items() if k in rare_classes)
        rarity_score = rare_px / total_px

        w = 1.0 + (max_weight - 1.0) * rarity_score
        weights[tile_name] = round(w, 3)

    return weights
