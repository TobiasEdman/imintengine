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

import threading
import time as _time

TILE_SIZE_M = 2560   # 256 pixels × 10m
TILE_SIZE_PX = 256
N_BANDS = 6
PRITHVI_BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]


class AdaptiveSemaphore:
    """Semaphore that adjusts concurrency based on success/failure rates.

    Starts at ``initial`` permits, increases by 1 (up to ``max_permits``)
    after ``ramp_up_after`` consecutive successes, decreases by 1 (down to
    ``min_permits``) on any failure or timeout.
    """

    def __init__(
        self,
        initial: int = 3,
        min_permits: int = 1,
        max_permits: int = 8,
        ramp_up_after: int = 10,
        name: str = "",
    ):
        self._lock = threading.Lock()
        self._sem = threading.Semaphore(initial)
        self._permits = initial
        self._min = min_permits
        self._max = max_permits
        self._ramp_up_after = ramp_up_after
        self._consecutive_ok = 0
        self._name = name

    @property
    def permits(self) -> int:
        return self._permits

    def acquire(self, timeout: float | None = None) -> bool:
        return self._sem.acquire(timeout=timeout)

    def release(self) -> None:
        self._sem.release()

    def report_success(self) -> None:
        with self._lock:
            self._consecutive_ok += 1
            if self._consecutive_ok >= self._ramp_up_after and self._permits < self._max:
                self._permits += 1
                self._consecutive_ok = 0
                self._sem.release()  # add a permit
                print(f"    [{self._name}] ↑ concurrency → {self._permits}")

    def report_failure(self) -> None:
        with self._lock:
            self._consecutive_ok = 0
            if self._permits > self._min:
                self._permits -= 1
                # consume a permit (don't release — effectively reduces slots)
                self._sem.acquire(timeout=0)
                print(f"    [{self._name}] ↓ concurrency → {self._permits}")


_DES_SEMAPHORE = AdaptiveSemaphore(
    initial=2, min_permits=1, max_permits=8,
    ramp_up_after=10, name="DES",
)
# CDSE allows 300 req/min. Start aggressive, back off only on real errors.
_CDSE_SEMAPHORE = AdaptiveSemaphore(
    initial=20, min_permits=3, max_permits=50,
    ramp_up_after=20, name="CDSE",
)


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
    cloud_threshold: float = 0.15,
    haze_threshold: float = 0.08,
) -> tuple[np.ndarray | None, str]:
    """Fetch best S2 scene within a date range. STAC → CDSE → DES fallback.

    Returns:
        (scene, date_str). scene is (6, H, W) float32 or None.
    """
    from imint.fetch import _stac_available_dates
    from imint.training.cdse_s2 import fetch_s2_scene

    candidates = []
    # DES STAC only indexes S2 from 2018 onwards — skip for pre-2018 ranges
    if date_end >= "2018-01-01":
        try:
            dates = _stac_available_dates(
                coords_wgs84, date_start, date_end,
                scene_cloud_max=scene_cloud_max,
            )
            candidates.extend(dates)
        except Exception:
            pass

    # If STAC found nothing (pre-2018, or simply no clear scenes indexed),
    # generate synthetic date candidates and try CDSE directly
    if not candidates:
        from datetime import datetime as _dt, timedelta as _td
        d0 = _dt.strptime(date_start, "%Y-%m-%d")
        d1 = _dt.strptime(date_end, "%Y-%m-%d")
        # Pre-2018: probe every 3 days (S2 revisit ~5 days with single satellite)
        # Post-2018: every 5 days is fine (STAC already found candidates above)
        step = 3 if date_end < "2018-01-01" else max(1, (d1 - d0).days // 6)
        for i in range(0, (d1 - d0).days + 1, step):
            candidates.append(((d0 + _td(days=i)).strftime("%Y-%m-%d"), 50.0))

    candidates.sort(key=lambda x: x[1])

    # DES-primary with 3 workers, CDSE as single-worker fallback.
    # DES has more capacity and doesn't rate-limit as aggressively.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from imint.fetch import fetch_seasonal_image

    def _cdse_try(date_str: str) -> tuple[np.ndarray | None, str]:
        _CDSE_SEMAPHORE.acquire()
        try:
            result = fetch_s2_scene(
                bbox_3006["west"], bbox_3006["south"],
                bbox_3006["east"], bbox_3006["north"],
                date=date_str,
                size_px=TILE_SIZE_PX,
                cloud_threshold=cloud_threshold,
                haze_threshold=haze_threshold,
            )
            # Always report success — the API responded.
            # result=None means cloud/haze rejection, not rate-limiting.
            _CDSE_SEMAPHORE.report_success()
            if result is not None:
                return result[0], date_str
        except Exception:
            # Network error, timeout, 5xx → real failure
            _CDSE_SEMAPHORE.report_failure()
        finally:
            _CDSE_SEMAPHORE.release()
        return None, date_str

    def _des_try(date_str: str) -> tuple[np.ndarray | None, str]:
        _DES_SEMAPHORE.acquire()
        try:
            result = fetch_seasonal_image(
                date=date_str,
                coords=coords_wgs84,
                prithvi_bands=PRITHVI_BANDS,
                source="des",
            )
            _DES_SEMAPHORE.report_success()
            if result is not None:
                return result[0], date_str
        except Exception:
            _DES_SEMAPHORE.report_failure()
        finally:
            _DES_SEMAPHORE.release()
        return None, date_str

    top_dates = [d for d, _ in candidates[:max_candidates]]

    # 3 DES + 1 CDSE in parallel per tile. First successful result wins.
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = []
        # DES: submit up to 3 candidates
        for d in top_dates[:3]:
            futures.append(pool.submit(_des_try, d))
        # CDSE: submit best candidate (1 worker)
        if top_dates:
            futures.append(pool.submit(_cdse_try, top_dates[0]))

        for f in as_completed(futures):
            scene, date_str = f.result()
            if scene is not None:
                for pending in futures:
                    pending.cancel()
                return scene, date_str

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
    vpp_windows: list[tuple[int, int]] | None = ...,  # sentinel: fetch on demand
) -> list[tuple[np.ndarray | None, str]]:
    """Fetch 4-frame tile: 1 autumn (year-1) + 3 VPP-guided growing season.

    Frame layout:
        0: Autumn (Sep-Oct from previous year)
        1-3: Growing season (VPP-guided DOY windows)

    Args:
        bbox_3006: Tile bbox in EPSG:3006.
        coords_wgs84: WGS84 bbox for STAC queries.
        years: Growing season years to search, e.g. ["2022", "2023"].
        vpp_windows: Pre-fetched list of (doy_start, doy_end) tuples.
            Pass None explicitly to skip VPP (use fixed seasonal dates).
            Omit (default sentinel) to fetch VPP on demand for this tile.

    Returns:
        List of 4 (scene, date_str) tuples.
    """
    # Get VPP-guided growing season windows (3 frames)
    if vpp_windows is ...:  # sentinel → fetch on demand for this tile
        vpp_windows = _get_vpp_doy_windows(bbox_3006, num_growing_frames=3)
    # vpp_windows is None if VPP was skipped or failed — handled below

    results: list[tuple[np.ndarray | None, str]] = []

    # --- Frame 0: Autumn (Sep-Oct from year-1) ---
    # Two-pass cloud filtering, same pattern as growing season but more permissive:
    #   scene_cloud_max: full S2 swath STAC filter (up to 2× scene_cloud_max, max 60%)
    #   cloud_threshold: tile spectral cutout acceptance (0.20 vs 0.15 for crops)
    # DES STAC skipped automatically for pre-2018 in _fetch_single_scene.
    autumn_scene_cloud_max = min(scene_cloud_max * 2.0, 60.0)
    autumn_scene, autumn_date = None, ""
    for year in years:
        prev_year = str(int(year) - 1)
        s, a = _fetch_single_scene(
            bbox_3006, coords_wgs84,
            f"{prev_year}-08-15", f"{prev_year}-10-31",
            scene_cloud_max=autumn_scene_cloud_max,
            max_candidates=max(max_candidates, 16),
            cloud_threshold=0.30,
            haze_threshold=0.12,
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
    size_px: int | None = None,
) -> np.ndarray | None:
    """Read NMD 19-class sequential label from local GeoTIFF raster.

    Args:
        bbox_3006: Tile bounding box in SWEREF99 TM.
        nmd_raster: Path to NMD GeoTIFF.
        size_px: Output resolution in pixels. Derived from bbox extent
            at 10m/px if None.

    Returns (H, W) uint8 with indices 0-19, or None if unavailable.
    Falls back to openEO remote fetch if the local raster is missing.
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

        b = _NMD_SRC.bounds
        if w < b.left or e > b.right or s < b.bottom or n > b.top:
            return None

        window = from_bounds(w, s, e, n, _NMD_SRC.transform)
        # Read directly into the target shape so rasterio handles the
        # sub-pixel alignment via its own nearest-neighbour resampling.
        # Never use scipy.ndimage.zoom here — it distributes dropped rows
        # unevenly and creates visible seams when the window is fractional.
        from rasterio.enums import Resampling
        # Derive output size from bbox extent at 10m/px if not specified
        out_px = size_px or round((bbox_3006["east"] - bbox_3006["west"]) / 10)
        nmd_raw = _NMD_SRC.read(
            1,
            window=window,
            out_shape=(out_px, out_px),
            resampling=Resampling.nearest,
        )

        from imint.training.class_schema import nmd_raster_to_lulc
        return nmd_raster_to_lulc(nmd_raw).astype(np.uint8)
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


def fetch_background_frame(
    bbox_3006: dict,
    *,
    primary_year: int = 2016,
    fallback_year: int = 2015,
    doy_start: int = 152,   # Jun  1
    doy_end: int = 228,     # Aug 16
    scene_cloud_max: float = 40.0,
    cloud_threshold: float = 0.15,
    haze_threshold: float = 0.10,
    nodata_threshold: float = 0.20,
    max_candidates: int = 5,
) -> dict | None:
    """Fetch the best summer background frame for a tile.

    Uses the CDSE Catalog STAC to enumerate **all** available S2A
    acquisitions in a fixed DOY window (Jun 1–Aug 16), then fetches
    each candidate via the Sentinel Hub Process API and scores it on
    tile-level quality (valid-pixel fraction, cloud fraction).

    The primary year is tried first; if no qualifying scene is found,
    the fallback year is used.  VPP is deliberately bypassed (VPP only
    covers 2017+).

    Args:
        bbox_3006: Tile bbox as ``{"west", "south", "east", "north"}``.
        primary_year: First year to search (default 2016).
        fallback_year: Second year if primary fails (default 2015).
        doy_start: Start DOY of summer window (default 152 = Jun 1).
        doy_end: End DOY of summer window (default 228 = Aug 16).
        scene_cloud_max: Scene-level cloud ceiling for catalog filter.
        cloud_threshold: Max tile-level cloud fraction (0–1).
        haze_threshold: Max B02 mean reflectance for haze gate (0–1).
        nodata_threshold: Max nodata (zero-pixel) fraction (0–1).
        max_candidates: Max catalog candidates to try per year.

    Returns:
        Dict with keys::

            frame_2016         (6, 256, 256) float32 spectral
            frame_2016_date    "YYYY-MM-DD"
            frame_2016_doy     int32
            frame_2016_cloud_pct float32 (tile-level, 0–1)
            frame_2016_year    int32 (actual year used; may be 2015)
            has_frame_2016     int32 (1 = success, 0 = failure)

        Returns ``None`` only if both years completely fail (e.g.
        network unreachable).  On a per-tile failure the caller should
        store ``has_frame_2016 = 0`` and a zero-filled ``frame_2016``.
    """
    from imint.training.cdse_s2 import fetch_s2_scene, cdse_catalog_search

    coords_wgs84 = bbox_3006_to_wgs84(bbox_3006)
    bbox_4326 = (
        coords_wgs84["west"], coords_wgs84["south"],
        coords_wgs84["east"], coords_wgs84["north"],
    )
    w, s, e, n = (
        bbox_3006["west"], bbox_3006["south"],
        bbox_3006["east"], bbox_3006["north"],
    )

    for year in [primary_year, fallback_year]:
        date_start, date_end = doy_to_date_range(year, doy_start, doy_end)

        # Query CDSE Catalog STAC — returns all acquisitions sorted by cloud %
        candidates = cdse_catalog_search(
            bbox_4326, date_start, date_end,
            max_cloud=scene_cloud_max,
        )

        if not candidates:
            print(f"    [BG {year}] no catalog results ({date_start}..{date_end})")
            continue

        print(
            f"    [BG {year}] {len(candidates)} catalog scenes, "
            f"best cloud={candidates[0][1]:.0f}%"
        )

        for cand_date, _scene_cloud in candidates[:max_candidates]:
            try:
                result = fetch_s2_scene(
                    w, s, e, n,
                    date=cand_date,
                    size_px=TILE_SIZE_PX,
                    cloud_threshold=cloud_threshold,
                    haze_threshold=haze_threshold,
                    nodata_threshold=nodata_threshold,
                )
            except Exception:
                continue

            if result is None:
                continue

            spectral, _scl, cloud_frac = result

            valid_pct = float((spectral[0] > 0).mean())
            if valid_pct < 0.80:
                print(
                    f"    [BG {year}] {cand_date}: rejected "
                    f"(valid_pct={valid_pct:.0%})"
                )
                continue

            doy_val = datetime.strptime(cand_date, "%Y-%m-%d").timetuple().tm_yday
            print(
                f"    [BG {year}] {cand_date} OK "
                f"(cloud={cloud_frac:.0%}, valid={valid_pct:.0%})"
            )
            return {
                "frame_2016": spectral,
                "frame_2016_date": np.bytes_(cand_date),
                "frame_2016_doy": np.int32(doy_val),
                "frame_2016_cloud_pct": np.float32(cloud_frac),
                "frame_2016_year": np.int32(year),
                "has_frame_2016": np.int32(1),
            }

        print(
            f"    [BG {year}]: no qualifying scene "
            f"(tried {min(len(candidates), max_candidates)} of {len(candidates)})"
        )

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
