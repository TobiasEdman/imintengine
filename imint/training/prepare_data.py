"""
imint/training/prepare_data.py — Fetch and cache LULC training data

Orchestrates bulk fetching of Sentinel-2 + NMD tile pairs from DES,
with progress tracking for resumability.

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
_MAX_WORKERS = 4
_ADAPT_WINDOW = 10          # requests to consider for adaptation
_LATENCY_HIGH_S = 60.0      # p90 above this → scale down
_LATENCY_LOW_S = 30.0       # p90 below this → scale up
_ERROR_RATE_HIGH = 0.30     # >30 % errors → scale down

from .config import TrainingConfig
from .class_schema import nmd_raster_to_lulc
from .sampler import (
    generate_grid, grid_to_wgs84, split_by_latitude,
    densify_grid, generate_densification_regions,
)

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
        fetch_des_data, fetch_nmd_data,
        _connect, _stac_available_dates, _fetch_scl, _fetch_scl_batch,
        _to_nmd_grid, FetchError,
    )

    data_dir = Path(config.data_dir)
    tiles_dir = data_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Generate grid ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  LULC Training Data Preparation")
    print(f"  Grid spacing: {config.grid_spacing_m}m")
    print(f"  Years: {config.years}")
    print(f"  Growing season: months {config.growing_season[0]}-{config.growing_season[1]}")
    print(f"{'='*60}")

    patch_size_m = config.fetch_pixels * 10  # 10m resolution
    cells = generate_grid(
        spacing_m=config.grid_spacing_m,
        patch_size_m=patch_size_m,
    )
    base_count = len(cells)

    # Optional: densify grid in rare-class areas
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

    # ── Step 4: Verify DES connection ─────────────────────────────────
    print("\n  Verifying DES connection...")
    conn = _connect()
    print("  DES connection OK")

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
            "scl_prescreen": {"count": 0, "latency_sum": 0.0, "bytes_sum": 0, "errors": 0},
            "full_spectral": {"count": 0, "latency_sum": 0.0, "bytes_sum": 0, "errors": 0},
            "nmd_fetch":     {"count": 0, "latency_sum": 0.0, "bytes_sum": 0, "errors": 0},
        },
    }

    # Queue for NMD-approved cells → spectral fetch workers
    approved_q = queue.Queue(maxsize=200)

    # Adaptive concurrency pool
    pool = _AdaptiveWorkerPool(
        initial=_INITIAL_WORKERS, lo=_MIN_WORKERS, hi=_MAX_WORKERS,
    )

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
        "elapsed_s": 0.0,
        "recent_previews": [],
    }
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
                    land_frac = float(np.mean(
                        (labels > 0) & (labels != 18) & (labels != 19)))
                    if land_frac < 0.05:
                        water_skipped += 1
                        with lock:
                            failed.add(cell_key)
                        print(f"    NMD {cell_key}: SKIP water — "
                              f"land_frac={land_frac:.1%}")
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
        for _ in range(_FETCH_WORKERS):
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

                        # SCL pre-screen: batch all candidates in one DES call
                        cand_dates = [d for d, _ in dates[:_SCL_CANDIDATES]]
                        all_stac = [d for d, _ in dates]  # all dates from STAC for this month
                        scl_results = []
                        try:
                            pool.acquire()
                            t0 = time.monotonic()
                            scl_err = False
                            try:
                                scl_results = _fetch_scl_batch(
                                    conn, projected, cand_dates,
                                    all_stac_dates=all_stac,
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
                                    cell_key=cell_key,
                                    date=f"{cand_dates[0]}..{cand_dates[-1]}",
                                    latency_s=scl_lat,
                                    band_count=len(all_stac),
                                    success=not scl_err,
                                )
                            time.sleep(1)
                        except (ValueError, Exception) as e:
                            # Batch failed (band mismatch or DES error) — fall back
                            # to per-date fetching
                            err_msg = str(e)
                            if len(err_msg) > 80:
                                err_msg = err_msg[:77] + "..."
                            print(f"    SCL batch {cell_key}: "
                                  f"{type(e).__name__}: {err_msg}, "
                                  f"falling back to per-date")
                            time.sleep(1)
                            scl_results = []
                            for cand_date, scene_cloud in dates[:_SCL_CANDIDATES]:
                                try:
                                    cand_dt = datetime.strptime(cand_date, "%Y-%m-%d")
                                    temporal = [
                                        cand_date,
                                        (cand_dt + _td(days=1)).strftime("%Y-%m-%d"),
                                    ]
                                    pool.acquire()
                                    t0 = time.monotonic()
                                    scl_err2 = False
                                    try:
                                        _scl, aoi_cloud, _crs, _tr = _fetch_scl(
                                            conn, projected, temporal, date_window=0,
                                        )
                                    except Exception:
                                        scl_err2 = True
                                        raise
                                    finally:
                                        scl_lat2 = time.monotonic() - t0
                                        pool.record(scl_lat2, scl_err2)
                                        pool.release()
                                        _record_des_call(
                                            des_stats, "scl_prescreen",
                                            cell_key=cell_key, date=cand_date,
                                            latency_s=scl_lat2, band_count=1,
                                            success=not scl_err2,
                                        )
                                    time.sleep(1)
                                    scl_results.append((cand_date, aoi_cloud))
                                except Exception as e2:
                                    err_msg2 = str(e2)
                                    if len(err_msg2) > 80:
                                        err_msg2 = err_msg2[:77] + "..."
                                    print(f"    SCL {cell_key} {cand_date}: "
                                          f"{type(e2).__name__}: {err_msg2}")
                                    time.sleep(1)

                        # Find first date below cloud threshold
                        for cand_date_str, aoi_cloud in scl_results:
                            if aoi_cloud <= config.cloud_threshold:
                                good_date = cand_date_str
                                good_cloud = aoi_cloud
                                break
                            else:
                                print(f"    SCL {cell_key} {cand_date_str}: "
                                      f"cloud={aoi_cloud:.1%} > "
                                      f"{config.cloud_threshold:.0%}, skip")

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

                    # Save tile
                    np.savez_compressed(
                        tiles_dir / tile_name,
                        image=image,
                        label=labels,
                        easting=cell.easting,
                        northing=cell.northing,
                        date=good_date,
                        lat=cell.center_lat,
                        lon=cell.center_lon,
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
                        prep_log["elapsed_s"] = round(time.time() - t_start, 1)
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

                    elapsed = time.time() - t_start
                    rate = cur_idx / max(elapsed, 1) * 3600
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
                    prep_log["elapsed_s"] = round(time.time() - t_start, 1)
                    prep_log["updated_at"] = datetime.now(timezone.utc).isoformat()
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
    print(f"\n  Starting parallel pipeline: NMD filter + "
          f"{_INITIAL_WORKERS} fetch workers (adaptive {_MIN_WORKERS}-{_MAX_WORKERS})...")

    nmd_thread = threading.Thread(target=_nmd_producer, daemon=True)
    nmd_thread.start()

    # Spawn MAX threads — the adaptive semaphore controls how many
    # are actually active at any time.
    fetch_threads = []
    for _ in range(_MAX_WORKERS):
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
    prep_log["elapsed_s"] = round(time.time() - t_start, 1)
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
