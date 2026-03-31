"""
imint/training/tile_fetch.py — Shared tile fetching primitives

Core building blocks for fetching Sentinel-2 tiles with seasonal windows.
Used by both fetch_lucas_tiles.py and fetch_unified_tiles.py.
"""
from __future__ import annotations

import calendar
import os
from datetime import datetime

import numpy as np

TILE_SIZE_M = 2560   # 256 pixels × 10m
TILE_SIZE_PX = 256
N_BANDS = 6
PRITHVI_BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]


def point_to_bbox_3006(lat: float, lon: float) -> dict:
    """Convert WGS84 point → 2560m × 2560m EPSG:3006 bounding box."""
    from rasterio.crs import CRS
    from rasterio.warp import transform

    xs, ys = transform(
        CRS.from_epsg(4326), CRS.from_epsg(3006), [lon], [lat],
    )
    cx, cy = xs[0], ys[0]
    half = TILE_SIZE_M / 2
    return {
        "west": int(cx - half),
        "south": int(cy - half),
        "east": int(cx + half),
        "north": int(cy + half),
    }


def bbox_3006_to_wgs84(bbox: dict) -> dict:
    """Convert EPSG:3006 bbox → WGS84 approximate bbox for STAC queries."""
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds

    w, s, e, n = transform_bounds(
        CRS.from_epsg(3006), CRS.from_epsg(4326),
        bbox["west"], bbox["south"], bbox["east"], bbox["north"],
    )
    return {"west": w, "south": s, "east": e, "north": n}


def _fetch_single_scene(
    bbox_3006: dict,
    coords_wgs84: dict,
    date_start: str,
    date_end: str,
    *,
    scene_cloud_max: float = 30.0,
    max_candidates: int = 3,
) -> tuple[np.ndarray | None, str]:
    """Fetch best S2 scene within a date range. STAC → CDSE → DES fallback.

    Returns:
        (scene, date_str). scene is (6, H, W) float32 or None.
    """
    from imint.fetch import _stac_available_dates
    from imint.training.cdse_s2 import fetch_s2_scene

    candidates = []
    try:
        dates = _stac_available_dates(
            coords_wgs84, date_start, date_end,
            scene_cloud_max=scene_cloud_max,
        )
        candidates.extend(dates)
    except Exception:
        pass

    # If STAC found nothing (e.g. pre-2018 data not in DES catalog),
    # generate synthetic date candidates and try CDSE directly
    if not candidates:
        from datetime import datetime as _dt, timedelta as _td
        d0 = _dt.strptime(date_start, "%Y-%m-%d")
        d1 = _dt.strptime(date_end, "%Y-%m-%d")
        step = max(1, (d1 - d0).days // 6)
        for i in range(0, (d1 - d0).days + 1, step):
            candidates.append(((d0 + _td(days=i)).strftime("%Y-%m-%d"), 50.0))

    candidates.sort(key=lambda x: x[1])

    # Primary: CDSE Sentinel Hub HTTP
    for date_str, _cloud in candidates[:max_candidates]:
        try:
            result = fetch_s2_scene(
                bbox_3006["west"], bbox_3006["south"],
                bbox_3006["east"], bbox_3006["north"],
                date=date_str,
                size_px=TILE_SIZE_PX,
                cloud_threshold=0.15,
                haze_threshold=0.08,
            )
            if result is not None:
                return result[0], date_str
        except Exception:
            continue

    # Fallback: DES openEO
    if candidates:
        try:
            from imint.fetch import fetch_seasonal_image
            result = fetch_seasonal_image(
                date=candidates[0][0],
                coords=coords_wgs84,
                prithvi_bands=PRITHVI_BANDS,
                source="des",
            )
            if result is not None:
                return result[0], candidates[0][0]
        except Exception:
            pass

    return None, ""


def _get_vpp_doy_windows(bbox_3006: dict, num_growing_frames: int = 3) -> list[tuple[int, int]] | None:
    """Get VPP-guided growing season DOY windows for a tile.

    Returns list of (doy_start, doy_end) tuples, or None if VPP fails.
    """
    try:
        from imint.training.cdse_vpp import fetch_vpp_tiles
        from imint.training.vpp_windows import compute_growing_season_windows

        vpp = fetch_vpp_tiles(
            west=bbox_3006["west"],
            south=bbox_3006["south"],
            east=bbox_3006["east"],
            north=bbox_3006["north"],
            size_px=64,
        )
        return compute_growing_season_windows(
            vpp["sosd"], vpp["eosd"],
            num_frames=num_growing_frames,
        )
    except Exception:
        return None


def doy_to_date_range(year: int, doy_start: int, doy_end: int) -> tuple[str, str]:
    """Convert DOY range to ISO date strings for a given year."""
    from datetime import timedelta
    base = datetime(year, 1, 1)
    d_start = base + timedelta(days=doy_start - 1)
    d_end = base + timedelta(days=min(doy_end - 1, 364))
    return d_start.strftime("%Y-%m-%d"), d_end.strftime("%Y-%m-%d")


def fetch_4frame_scenes(
    bbox_3006: dict,
    coords_wgs84: dict,
    years: list[str],
    *,
    scene_cloud_max: float = 30.0,
    max_candidates: int = 3,
) -> list[tuple[np.ndarray | None, str]]:
    """Fetch 4-frame tile: 1 autumn (year-1) + 3 VPP-guided growing season.

    Frame layout:
        0: Autumn (Sep-Oct from previous year)
        1-3: Growing season (VPP-guided DOY windows)

    Args:
        bbox_3006: Tile bbox in EPSG:3006.
        coords_wgs84: WGS84 bbox for STAC queries.
        years: Growing season years to search, e.g. ["2022", "2023"].

    Returns:
        List of 4 (scene, date_str) tuples.
    """
    # Get VPP-guided growing season windows (3 frames)
    vpp_windows = _get_vpp_doy_windows(bbox_3006, num_growing_frames=3)
    # vpp_windows is None if VPP fails — will be handled below

    results: list[tuple[np.ndarray | None, str]] = []

    # --- Frame 0: Autumn (Sep-Oct from year-1) ---
    autumn_scene, autumn_date = None, ""
    for year in years:
        prev_year = str(int(year) - 1)
        s, a = _fetch_single_scene(
            bbox_3006, coords_wgs84,
            f"{prev_year}-09-01", f"{prev_year}-10-31",
            scene_cloud_max=scene_cloud_max,
            max_candidates=max_candidates,
        )
        if s is not None:
            autumn_scene, autumn_date = s, a
            break
    results.append((autumn_scene, autumn_date))

    # --- Frames 1-3: VPP-guided growing season ---
    if vpp_windows and len(vpp_windows) >= 3:
        for doy_start, doy_end in vpp_windows[:3]:
            best_scene, best_date = None, ""
            for year in years:
                ds, de = doy_to_date_range(int(year), doy_start, doy_end)
                s, d = _fetch_single_scene(
                    bbox_3006, coords_wgs84, ds, de,
                    scene_cloud_max=scene_cloud_max,
                    max_candidates=max_candidates,
                )
                if s is not None:
                    best_scene, best_date = s, d
                    break
            results.append((best_scene, best_date))
    else:
        # VPP unavailable — should not happen, but handle gracefully
        # Leave frames 1-3 as None (will be zero-padded by stack_frames)
        for _ in range(3):
            results.append((None, ""))

    return results


def fetch_aux_channels(bbox_3006: dict) -> dict[str, np.ndarray]:
    """Fetch auxiliary channels (VPP phenology + DEM) for a tile.

    Returns dict of channel_name → (H, W) float32. Missing channels skipped.
    """
    aux = {}

    # VPP phenology (5 bands)
    try:
        from imint.training.cdse_vpp import fetch_vpp_tiles
        vpp = fetch_vpp_tiles(
            west=bbox_3006["west"],
            south=bbox_3006["south"],
            east=bbox_3006["east"],
            north=bbox_3006["north"],
            size_px=TILE_SIZE_PX,
        )
        for band in ["sosd", "eosd", "length", "maxv", "minv"]:
            if band in vpp and vpp[band] is not None:
                aux[f"vpp_{band}"] = vpp[band].astype(np.float32)
    except Exception:
        pass

    # DEM (Copernicus GLO-30)
    try:
        from imint.training.copernicus_dem import fetch_dem_tile
        dem = fetch_dem_tile(
            west=bbox_3006["west"],
            south=bbox_3006["south"],
            east=bbox_3006["east"],
            north=bbox_3006["north"],
            size_px=TILE_SIZE_PX,
        )
        if dem is not None:
            aux["dem"] = dem.astype(np.float32)
    except Exception:
        pass

    return aux


_NMD_SRC = None  # lazy-opened rasterio handle


def fetch_nmd_label_local(
    bbox_3006: dict,
    nmd_raster: str = "data/nmd/nmd2018bas_ogeneraliserad_v1_1.tif",
    num_classes: int = 10,
) -> np.ndarray | None:
    """Read NMD label from local GeoTIFF raster (no internet).

    Falls back to openEO if local raster is unavailable.
    Returns (H, W) uint8 or None.
    """
    global _NMD_SRC

    try:
        import rasterio
        from rasterio.windows import from_bounds

        if _NMD_SRC is None:
            if not os.path.exists(nmd_raster):
                return _fetch_nmd_label_remote(bbox_3006)
            _NMD_SRC = rasterio.open(nmd_raster)

        w, s, e, n = bbox_3006["west"], bbox_3006["south"], bbox_3006["east"], bbox_3006["north"]

        # Bounds check
        b = _NMD_SRC.bounds
        if w < b.left or e > b.right or s < b.bottom or n > b.top:
            return None

        window = from_bounds(w, s, e, n, _NMD_SRC.transform)
        nmd_raw = _NMD_SRC.read(1, window=window)

        # Resize to tile size if needed
        if nmd_raw.shape != (TILE_SIZE_PX, TILE_SIZE_PX):
            from scipy.ndimage import zoom
            zy = TILE_SIZE_PX / nmd_raw.shape[0]
            zx = TILE_SIZE_PX / nmd_raw.shape[1]
            nmd_raw = zoom(nmd_raw, (zy, zx), order=0)

        from imint.training.class_schema import nmd_raster_to_lulc
        return nmd_raster_to_lulc(nmd_raw, num_classes=num_classes).astype(np.uint8)
    except Exception:
        return _fetch_nmd_label_remote(bbox_3006)


def _fetch_nmd_label_remote(bbox_3006: dict) -> np.ndarray | None:
    """Fallback: fetch NMD via openEO (requires internet)."""
    try:
        coords_wgs84 = bbox_3006_to_wgs84(bbox_3006)
        from imint.fetch import fetch_nmd_data
        result = fetch_nmd_data(
            coords_wgs84, target_shape=(TILE_SIZE_PX, TILE_SIZE_PX),
        )
        if result and result.nmd_raster is not None:
            return result.nmd_raster
    except Exception:
        pass
    return None


def stack_frames(
    scene_results: list[tuple[np.ndarray | None, str]],
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Stack seasonal scenes into multitemporal array.

    Pads missing frames with zeros, then copies nearest valid frame.

    Returns:
        image: (T*6, H, W) float32
        temporal_mask: (T,) uint8
        doy: (T,) int32
        dates: list of date strings
    """
    frames = []
    valid = []
    dates = []
    doys = []

    for scene, date_str in scene_results:
        if scene is not None:
            # Ensure correct size
            if scene.shape[1] != TILE_SIZE_PX or scene.shape[2] != TILE_SIZE_PX:
                padded = np.zeros((N_BANDS, TILE_SIZE_PX, TILE_SIZE_PX), dtype=np.float32)
                h = min(scene.shape[1], TILE_SIZE_PX)
                w = min(scene.shape[2], TILE_SIZE_PX)
                padded[:, :h, :w] = scene[:, :h, :w]
                frames.append(padded)
            else:
                frames.append(scene)
            valid.append(True)
        else:
            frames.append(np.zeros((N_BANDS, TILE_SIZE_PX, TILE_SIZE_PX), dtype=np.float32))
            valid.append(False)

        dates.append(date_str)
        if date_str:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            doys.append(dt.timetuple().tm_yday)
        else:
            doys.append(0)

    # Pad/trim to num_frames
    while len(frames) < num_frames:
        frames.append(np.zeros((N_BANDS, TILE_SIZE_PX, TILE_SIZE_PX), dtype=np.float32))
        valid.append(False)
        dates.append("")
        doys.append(0)

    image = np.concatenate(frames[:num_frames], axis=0)
    temporal_mask = np.array(valid[:num_frames], dtype=np.uint8)
    doy_arr = np.array(doys[:num_frames], dtype=np.int32)

    # Replace zero-padded frames with nearest valid
    valid_indices = [i for i in range(num_frames) if valid[i]]
    for t in range(num_frames):
        if not valid[t] and valid_indices:
            nearest = min(valid_indices, key=lambda i: abs(i - t))
            src = nearest * N_BANDS
            dst = t * N_BANDS
            image[dst:dst + N_BANDS] = image[src:src + N_BANDS]

    return image, temporal_mask, doy_arr, dates[:num_frames]
