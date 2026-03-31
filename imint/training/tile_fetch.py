"""
imint/training/tile_fetch.py — Shared tile fetching primitives

Core building blocks for fetching Sentinel-2 tiles with seasonal windows.
Used by both fetch_lucas_tiles.py and fetch_unified_tiles.py.
"""
from __future__ import annotations

import calendar
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


def fetch_seasonal_scenes(
    bbox_3006: dict,
    coords_wgs84: dict,
    seasonal_windows: list[tuple[int, int]],
    years: list[str],
    *,
    scene_cloud_max: float = 30.0,
    max_candidates: int = 3,
) -> list[tuple[np.ndarray | None, str]]:
    """Fetch N seasonal S2 scenes using STAC + CDSE primary + DES fallback.

    Args:
        bbox_3006: Tile bbox in EPSG:3006 (west, south, east, north keys).
        coords_wgs84: Approximate WGS84 bbox for STAC queries.
        seasonal_windows: List of (start_month, end_month) per frame.
        years: S2 search years, e.g. ["2022", "2023"].
        scene_cloud_max: Max scene cloud % for STAC pre-filter.
        max_candidates: Try this many candidate dates per window.

    Returns:
        List of (scene, date_str) tuples. scene is (6, H, W) float32
        or None if no scene found. date_str is ISO date or "".
    """
    from imint.fetch import _stac_available_dates
    from imint.training.cdse_s2 import fetch_s2_scene

    results = []
    for month_start, month_end in seasonal_windows:
        candidates = []
        for year in years:
            last_day = calendar.monthrange(int(year), month_end)[1]
            date_start = f"{year}-{month_start:02d}-01"
            date_end = f"{year}-{month_end:02d}-{last_day:02d}"

            try:
                dates = _stac_available_dates(
                    coords_wgs84, date_start, date_end,
                    scene_cloud_max=scene_cloud_max,
                )
                candidates.extend(dates)
            except Exception:
                pass

        candidates.sort(key=lambda x: x[1])  # sort by cloud ascending

        scene = None
        scene_date = ""

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
                    scene = result[0]  # (6, H, W) float32
                    scene_date = date_str
                    break
            except Exception:
                continue

        # Fallback: DES openEO
        if scene is None and candidates:
            try:
                from imint.fetch import fetch_seasonal_image
                result = fetch_seasonal_image(
                    date=candidates[0][0],
                    coords=coords_wgs84,
                    prithvi_bands=PRITHVI_BANDS,
                    source="des",
                )
                if result is not None:
                    scene = result[0]
                    scene_date = candidates[0][0]
            except Exception:
                pass

        results.append((scene, scene_date))

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


def fetch_nmd_label(coords_wgs84: dict) -> np.ndarray | None:
    """Fetch NMD land cover label for a tile. Returns (H, W) uint8 or None."""
    try:
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
