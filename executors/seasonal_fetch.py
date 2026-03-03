"""
executors/seasonal_fetch.py — ColonyOS executor for seasonal tile fetching.

Fetches all 4 seasonal frames for a single training tile and saves
the result as a compressed .npz file to the ColonyOS filesystem (CFS).

The ColonyOS job spec (config/seasonal_fetch_job.json) invokes:
    python executors/seasonal_fetch.py

Environment variables are set by ColonyOS from the job spec:
    EASTING, NORTHING            — tile center in EPSG:3006
    WEST_WGS84, SOUTH_WGS84,
    EAST_WGS84, NORTH_WGS84     — bounding box in WGS84
    FETCH_SOURCE                 — "des" or "copernicus" (default: "copernicus")
    YEARS                        — comma-separated, e.g. "2019,2018"
    SEASONAL_WINDOWS             — e.g. "4-5,6-7,8-9,1-2"
    SEASONAL_CLOUD_THRESHOLD     — max cloud fraction, e.g. "0.10"
    TILES_DIR                    — output directory (default: /cfs/tiles)
    NUM_CLASSES                  — NMD class count (default: 10)
    B02_HAZE_THRESHOLD           — max mean B02 reflectance (default: 0.06)
    DES_TOKEN                    — DES auth token (for source=des)
    CDSE_CLIENT_ID               — CDSE OAuth client ID (for source=copernicus)
    CDSE_CLIENT_SECRET           — CDSE OAuth client secret
"""
from __future__ import annotations

import os
import sys
import time
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imint.fetch import (
    _connect, _connect_cdse, _to_nmd_grid,
    _fetch_scl, _fetch_scl_batch,
    fetch_seasonal_dates, fetch_seasonal_image,
    fetch_nmd_data, FetchError,
)
from imint.training.class_schema import nmd_raster_to_lulc

# ── Defaults ─────────────────────────────────────────────────────────────
_SCL_CANDIDATES = 3       # max STAC dates to pre-screen per season
_PRITHVI_BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]


def _parse_windows(s: str) -> list[tuple[int, int]]:
    """Parse "4-5,6-7,8-9,1-2" → [(4,5), (6,7), (8,9), (1,2)]."""
    windows = []
    for part in s.split(","):
        start, end = part.strip().split("-")
        windows.append((int(start), int(end)))
    return windows


class SeasonalFetchExecutor:
    """ColonyOS executor that fetches seasonal Sentinel-2 data for one tile."""

    def execute(self):
        # ── 1. Parse environment ─────────────────────────────────────────
        easting = int(os.environ["EASTING"])
        northing = int(os.environ["NORTHING"])
        cell_key = f"{easting}_{northing}"

        coords = {
            "west": float(os.environ["WEST_WGS84"]),
            "south": float(os.environ["SOUTH_WGS84"]),
            "east": float(os.environ["EAST_WGS84"]),
            "north": float(os.environ["NORTH_WGS84"]),
        }

        source = os.environ.get("FETCH_SOURCE", "copernicus")
        years = os.environ.get("YEARS", "2019,2018").split(",")
        windows = _parse_windows(
            os.environ.get("SEASONAL_WINDOWS", "4-5,6-7,8-9,1-2")
        )
        cloud_threshold = float(
            os.environ.get("SEASONAL_CLOUD_THRESHOLD", "0.10")
        )
        tiles_dir = Path(os.environ.get("TILES_DIR", "/cfs/tiles"))
        num_classes = int(os.environ.get("NUM_CLASSES", "10"))
        b02_haze_threshold = float(
            os.environ.get("B02_HAZE_THRESHOLD", "0.06")
        )

        n_frames = len(windows)
        n_bands = len(_PRITHVI_BANDS)
        tile_path = tiles_dir / f"tile_{cell_key}.npz"

        print(f"[SeasonalFetch] Tile {cell_key} via {source.upper()}")
        print(f"  Windows: {windows}, Years: {years}")
        print(f"  Cloud threshold: {cloud_threshold:.0%}")
        t_start = time.monotonic()

        # ── 2. Skip if already done ──────────────────────────────────────
        if tile_path.exists():
            print(f"[SeasonalFetch] Tile already exists: {tile_path}")
            return

        # ── 3. Connect to backend ────────────────────────────────────────
        if source == "copernicus":
            conn = _connect_cdse()
        else:
            conn = _connect()
        print(f"  {source.upper()} connection OK")

        # ── 4. STAC discovery per season ─────────────────────────────────
        # Deterministic year rotation per cell
        cell_hash = int(hashlib.md5(cell_key.encode()).hexdigest(), 16)
        years_offset = cell_hash % len(years)
        years_order = years[years_offset:] + years[:years_offset]

        season_candidates = fetch_seasonal_dates(
            coords, windows, years_order,
            scene_cloud_max=50.0,
        )

        # ── 5. SCL pre-screen + spectral fetch per season ────────────────
        projected = _to_nmd_grid(coords)
        frames = []       # list of (6, H, W) arrays or None
        frame_dates = []   # list of date strings
        frame_mask = []    # 1 = valid, 0 = padded

        for win_idx, (m_start, m_end) in enumerate(windows):
            candidates = season_candidates[win_idx]
            win_label = f"m{m_start}-{m_end}"

            if not candidates:
                print(f"  {win_label}: no STAC candidates")
                frames.append(None)
                frame_dates.append("")
                frame_mask.append(0)
                continue

            # Try top candidates with SCL pre-screening
            top_cands = candidates[:_SCL_CANDIDATES]
            cand_strs = [d for d, _ in top_cands]
            good_date = None

            # Batch SCL pre-screen
            if len(cand_strs) > 1:
                try:
                    batch_results = _fetch_scl_batch(
                        conn, projected, cand_strs,
                        source=source,
                    )
                    for cand_date, aoi_cloud in batch_results:
                        if aoi_cloud <= cloud_threshold:
                            good_date = cand_date
                            break
                except Exception as e:
                    print(f"  {win_label}: batch SCL failed: {e}")

            # Per-date fallback
            if good_date is None:
                from datetime import timedelta as _td
                for cand_date, _ in top_cands:
                    try:
                        cand_dt = datetime.strptime(cand_date, "%Y-%m-%d")
                        temporal = [
                            cand_date,
                            (cand_dt + _td(days=1)).strftime("%Y-%m-%d"),
                        ]
                        _scl, aoi_cloud, _crs, _tr = _fetch_scl(
                            conn, projected, temporal,
                            date_window=0, source=source,
                        )
                        if aoi_cloud <= cloud_threshold:
                            good_date = cand_date
                            break
                    except Exception:
                        time.sleep(1)

            if good_date is None:
                print(f"  {win_label}: no clear date (tried {len(cand_strs)})")
                frames.append(None)
                frame_dates.append("")
                frame_mask.append(0)
                continue

            # Full spectral fetch
            try:
                img_result = fetch_seasonal_image(
                    good_date, coords, _PRITHVI_BANDS,
                    source=source,
                )
            except Exception:
                img_result = None

            if img_result is not None:
                frame_img, frame_dt = img_result

                # Quality gates
                nodata_frac = float((frame_img[0] == 0).mean())
                if nodata_frac > 0.10:
                    print(f"  {win_label} {good_date}: "
                          f"nodata={nodata_frac:.0%}, skip")
                    frames.append(None)
                    frame_dates.append("")
                    frame_mask.append(0)
                    continue

                b02_idx = _PRITHVI_BANDS.index("B02")
                b02_mean = float(frame_img[b02_idx].mean())
                if b02_mean > b02_haze_threshold:
                    print(f"  {win_label} {good_date}: "
                          f"haze B02={b02_mean:.4f}, skip")
                    frames.append(None)
                    frame_dates.append("")
                    frame_mask.append(0)
                    continue

                frames.append(frame_img)
                frame_dates.append(frame_dt)
                frame_mask.append(1)
                print(f"  {win_label}: {frame_dt} OK")
            else:
                frames.append(None)
                frame_dates.append("")
                frame_mask.append(0)

        # ── 6. Check minimum frames ──────────────────────────────────────
        n_valid = sum(frame_mask)
        if n_valid == 0:
            raise RuntimeError(f"Tile {cell_key}: no valid frames")

        # ── 7. Stack frames into (T*6, H, W) ────────────────────────────
        ref_shape = None
        for f in frames:
            if f is not None:
                ref_shape = f.shape[1:]  # (H, W)
                break

        stacked_frames = []
        for f in frames:
            if f is not None:
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
                stacked_frames.append(
                    np.zeros((n_bands,) + ref_shape, dtype=np.float32)
                )

        image = np.concatenate(stacked_frames, axis=0)  # (T*6, H, W)

        # ── 8. Fetch NMD labels ──────────────────────────────────────────
        nmd_result = fetch_nmd_data(
            coords=coords,
            target_shape=ref_shape,
        )
        labels = nmd_raster_to_lulc(
            nmd_result.nmd_raster,
            num_classes=num_classes,
        )

        # Day-of-year calculation
        doy_values = []
        for d in frame_dates:
            if d:
                doy_values.append(
                    datetime.strptime(d, "%Y-%m-%d").timetuple().tm_yday
                )
            else:
                doy_values.append(0)

        # ── 9. Save tile ─────────────────────────────────────────────────
        tiles_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tile_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp_path,
            image=image,
            label=labels,
            easting=easting,
            northing=northing,
            dates=np.array(frame_dates),
            doy=np.array(doy_values, dtype=np.int32),
            temporal_mask=np.array(frame_mask, dtype=np.int32),
            num_frames=n_frames,
            num_bands=n_bands,
            multitemporal=True,
            source=source,
        )
        # Atomic rename
        tmp_path.rename(tile_path)

        elapsed = time.monotonic() - t_start
        print(f"[SeasonalFetch] Tile {cell_key} saved "
              f"({n_valid}/{n_frames} frames, {elapsed:.1f}s)")


if __name__ == "__main__":
    executor = SeasonalFetchExecutor()
    try:
        executor.execute()
    except Exception as e:
        print(f"[SeasonalFetch] FAILED: {e}")
        sys.exit(1)
