"""
imint/training/tile_fetch.py — Shared tile fetching primitives

Core building blocks for fetching Sentinel-2 tiles with seasonal windows.
Used by both fetch_lucas_tiles.py and fetch_unified_tiles.py.
"""
from __future__ import annotations

import calendar
import os
import threading
import time as _time
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from imint.training.tile_config import TileConfig

N_BANDS = 6
PRITHVI_BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]

# NOTE: Tile size (size_px, size_m, half_m) is NOT stored as a module-level
# constant. It is a runtime parameter carried by TileConfig and threaded
# explicitly through every function that needs it. See imint.training.tile_config.
# Callers MUST pass TileConfig to every fetcher below; the previous mutable-
# module-constant pattern caused a hard-to-diagnose GSD corruption bug where
# `from tile_fetch import TILE_SIZE_M` snapshotted 2560 even after a CLI
# override attempted to mutate _tf.TILE_SIZE_M = 5120.


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
        self._total_success = 0
        self._total_failure = 0

    @property
    def permits(self) -> int:
        return self._permits

    @property
    def stats(self) -> str:
        return f"{self._name}: ok={self._total_success} fail={self._total_failure} permits={self._permits}"

    def acquire(self, timeout: float | None = None) -> bool:
        return self._sem.acquire(timeout=timeout)

    def release(self) -> None:
        self._sem.release()

    def report_success(self) -> None:
        with self._lock:
            self._total_success += 1
            self._consecutive_ok += 1
            if self._consecutive_ok >= self._ramp_up_after and self._permits < self._max:
                self._permits += 1
                self._consecutive_ok = 0
                self._sem.release()  # add a permit
                print(f"    [{self._name}] ↑ concurrency → {self._permits}")

    def report_failure(self) -> None:
        with self._lock:
            self._total_failure += 1
            self._consecutive_ok = 0
            if self._permits > self._min:
                self._permits -= 1
                # consume a permit (don't release — effectively reduces slots)
                self._sem.acquire(timeout=0)
                print(f"    [{self._name}] ↓ concurrency → {self._permits}")


# DES openEO: raised 2026-05-26 to 6 concurrent slots after CDSE openEO
# became the primary source (single-flight) and DES needed to absorb the
# parallel-worker load. Race-bug fix (commit bbea8af) means a DES hang
# no longer blocks tile completion — workers time out at 180 s and
# threads are abandoned via shutdown(wait=False, cancel_futures=True).
_DES_SEMAPHORE = AdaptiveSemaphore(
    initial=6, min_permits=2, max_permits=6,
    ramp_up_after=10, name="DES",
)
# CDSE SH Process API allows 300 req/min but each 512px request takes
# ~20s. 10 concurrent = ~30 req/min, well within quota.
_CDSE_SEMAPHORE = AdaptiveSemaphore(
    initial=10, min_permits=3, max_permits=20,
    ramp_up_after=20, name="CDSE",
)
# CDSE openEO enforces a HARD per-account ceiling of 1 concurrent
# connection (verified 2026-05-26: synchronous fetches over that limit
# return `[429] max connections reached: 1` at preflight, before any
# process graph runs). Adaptive ramp-up would just bounce us repeatedly
# into 429-spam, so we lock the semaphore at single-flight. Throughput
# tradeoff: ~60-120 frames/h via this source alone — acceptable because
# (a) the SH PU pool is exhausted and (b) DES openEO can race in
# parallel as opportunistic secondary.
_CDSE_OPENEO_SEMAPHORE = AdaptiveSemaphore(
    initial=1, min_permits=1, max_permits=1,
    ramp_up_after=10, name="CDSE-OPENEO",
)


def point_to_bbox_3006(lat: float, lon: float, tile: "TileConfig") -> dict:
    """Convert WGS84 point → tile-sized EPSG:3006 bounding box."""
    from rasterio.crs import CRS
    from rasterio.warp import transform

    xs, ys = transform(
        CRS.from_epsg(4326), CRS.from_epsg(3006), [lon], [lat],
    )
    cx, cy = xs[0], ys[0]
    return tile.bbox_from_center(cx, cy)


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
    tile: "TileConfig",
    *,
    scene_cloud_max: float = 30.0,
    max_aoi_cloud: float = 0.10,
    max_candidates: int = 3,
    cloud_threshold: float = 0.15,
    haze_threshold: float = 0.08,
    sources: tuple[str, ...] = ("cdse", "des"),
    prefetched_dates: list[str] | None = None,
    collect_extra: dict | None = None,
) -> tuple[np.ndarray | None, str]:
    """Fetch the best S2 scene within ``[date_start, date_end]`` via the
    unified per-slot fetch path.

    Routes the chosen date through
    :func:`imint.training.fetch_spectral.fetch_spectral`. The first live
    non-sen2cor token in ``sources`` is the primary backend (no backend race).
    ``l1c_sen2cor``, when present in ``sources``, is the deliberate LAST-RESORT
    fallthrough — tried (and visibly logged) only after the primary yields
    nothing for every ranked candidate: the PU-free, pre-2018-capable sen2cor
    path that backfills slots the openEO backends cannot reach. The old silent
    race-pool (``except Exception: pass``) stays retired — this fallthrough is
    single, ordered, and logged, not a race.

    Candidate dates come from (in priority order):

      1. ``prefetched_dates`` filtered to the window — used directly
         when the caller has already run a tile-wide ERA5 ∩ SCL ranker
         (e.g. a caller that pre-ranked dates tile-wide).
      2. ``optimal_fetch_dates(mode="era5_then_scl")`` for ≥ 2018.
      3. Synthetic every-N-days fallback for pre-2018 (no ERA5/STAC).

    Candidates are sorted by ``(cloud, distance-to-window-center)`` so
    a tie on cloud picks the more representative mid-window snapshot
    — avoids the edge-of-window bias for fixed-snapshot training
    (e.g. DOY 201 instead of representative DOY 220 for a 201–244
    window).

    Args:
        prefetched_dates: Pre-ranked candidate dates from the caller.
        cloud_threshold: AOI cloud-fraction ceiling (0–1) forwarded to
            ``fetch_spectral`` as the per-candidate acceptance gate.
        max_aoi_cloud: Ceiling for the per-window ERA5+SCL ranker
            when ``prefetched_dates`` is not supplied.
        scene_cloud_max, haze_threshold: Retained for signature
            stability; not consumed by the unified flow. Slated for
            removal once all call sites stop passing them.

    Returns:
        ``(scene, date_str)`` — ``(None, "")`` on failure or when no
        ``sources`` token resolves to a healthy backend.
    """
    from imint.training.fetch_spectral import (
        fetch_spectral, SUPPORTED_BACKENDS, DES_L2A_FLOOR)
    from imint.training.openeo_tile_graph import is_source_dead

    # 1) Build candidate list.
    candidates: list[tuple[str, float]] = []
    if prefetched_dates is not None:
        candidates = [
            (d, 0.0) for d in prefetched_dates
            if date_start <= d <= date_end
        ]
    elif date_end >= DES_L2A_FLOOR:
        try:
            from imint.training.optimal_fetch import optimal_fetch_dates
            plan = optimal_fetch_dates(
                coords_wgs84, date_start, date_end,
                mode="era5_then_scl",
                max_aoi_cloud=max_aoi_cloud,
            )
            candidates = [(d, 0.0) for d in plan.dates]
        except Exception:
            pass

    # Pre-2018 fallback: ERA5/STAC may both be unavailable. Synthesize
    # candidate dates every N days; the per-tile SCL gate inside
    # ``fetch_spectral`` still rejects cloudy acquisitions.
    if not candidates:
        from datetime import datetime as _dt, timedelta as _td
        d0 = _dt.strptime(date_start, "%Y-%m-%d")
        d1 = _dt.strptime(date_end, "%Y-%m-%d")
        step = 3 if date_end < DES_L2A_FLOOR else max(1, (d1 - d0).days // 6)
        for i in range(0, (d1 - d0).days + 1, step):
            candidates.append(((d0 + _td(days=i)).strftime("%Y-%m-%d"), 50.0))

    if not candidates:
        return None, ""

    # 2) Sort by (cloud, distance-to-center).
    from datetime import datetime as _dt2
    _ds = _dt2.strptime(date_start, "%Y-%m-%d")
    _de = _dt2.strptime(date_end, "%Y-%m-%d")
    _center = _ds + (_de - _ds) / 2

    def _dist_to_center(item):
        d_str, cloud = item
        try:
            d = _dt2.strptime(d_str, "%Y-%m-%d")
            return (cloud, abs((d - _center).days))
        except Exception:
            return (cloud, 9999)

    candidates.sort(key=_dist_to_center)
    top_dates = [d for d, _ in candidates[:max_candidates]]

    # 3) Backend order. The first live non-sen2cor token is the primary backend
    #    (the existing single-backend rule — no race). ``l1c_sen2cor``, when
    #    requested in ``sources``, is appended as a deliberate, LOGGED
    #    last-resort fallthrough: a scoped reversal of the
    #    no-cross-backend-fallback rule for the PU-free, pre-2018-capable
    #    sen2cor path ONLY (never a general race over the other backends).
    backends: list[str] = []
    for src in sources:
        if src == "l1c_sen2cor":
            continue
        if src in SUPPORTED_BACKENDS and not is_source_dead(src):
            backends.append(src)
            break
    if "l1c_sen2cor" in sources:
        backends.append("l1c_sen2cor")
    if not backends:
        return None, ""

    # 4) Try backends in order; within a backend the first accepted candidate
    #    wins. The per-candidate ``is_source_dead`` re-check covers cdse-openeo's
    #    402 PaymentRequired flipping the flag mid-call. Per-candidate temp extras
    #    dict: only the ACCEPTED candidate's b08/rededge/b01/b09 are copied out;
    #    a rejected (None) candidate may have left stale extras behind.
    for bi, backend in enumerate(backends):
        if is_source_dead(backend):
            continue
        if bi > 0:
            print(
                f"    [fetch_single_scene] {date_start[:7]}..{date_end[:7]}: "
                f"primary empty → last-resort fallthrough to {backend}",
                flush=True,
            )
        for d in top_dates:
            if is_source_dead(backend):
                break
            cand_extra: dict | None = {} if collect_extra is not None else None
            scene = fetch_spectral(
                bbox_3006, coords_wgs84, d,
                backend=backend,
                size_px=tile.size_px,
                cloud_threshold=cloud_threshold,
                collect_extra=cand_extra,
            )
            if scene is not None:
                if collect_extra is not None and cand_extra:
                    collect_extra.update(cand_extra)
                return scene, d
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


def _best_date_in_window(
    coords_wgs84: dict,
    date_start: str,
    date_end: str,
    *,
    mode: str,
    scl_backend: str,
) -> str | None:
    """The :func:`optimal_fetch_dates` clean date nearest the window midpoint, or None.

    ``mode`` is ``"era5_then_scl"`` for ≥2018 slots (full ERA5→STAC→SCL screen) or
    ``"era5_then_stac"`` for pre-2018 (DES SCL has no L2A index before 2018, so the
    SCL stage is skipped and STAC existence is the gate). Midpoint-nearest keeps a
    VPP growing-season pick phenologically centred within its window.
    """
    from datetime import date as _date

    from imint.training.optimal_fetch import optimal_fetch_dates

    plan = optimal_fetch_dates(
        coords_wgs84, date_start, date_end, mode=mode, scl_backend=scl_backend)
    if not plan.dates:
        return None
    mid = (_date.fromisoformat(date_start).toordinal()
           + _date.fromisoformat(date_end).toordinal()) // 2
    return min(plan.dates,
               key=lambda d: abs(_date.fromisoformat(d).toordinal() - mid))


def select_slot_dates(
    coords_wgs84: dict,
    *,
    tile_year: int,
    vpp_windows: list[tuple[int, int]] | None,
    background_year: int = 2016,
    background_fallback_year: int = 2015,
    scl_backend: str = "des",
) -> dict[int, str]:
    """One ERA5/SCL-clean date per temporal slot for the 5-slot through-entry fetch.

    Canonical 5-slot layout (matches ``fetch_tile_spectral(n_frames=5)``)::

        0       autumn, tile_year-1 (Aug 15 - Oct 31)    era5_then_scl
        1..3    VPP growing-season windows, tile_year    era5_then_scl
        4       2016 summer background (Jun 1 - Aug 16)   era5_then_stac (pre-2018:
                DES SCL is pre-2018-blind; falls back to background_fallback_year)

    Each window is screened with the repo-sanctioned :func:`optimal_fetch_dates`;
    the clean date nearest the window midpoint is taken. Slots with no clean
    candidate are omitted — the entry zero-fills them and ``temporal_mask`` records
    the gap. For FRESH fetches only; refetch reuses the tile's stored dates instead
    (never re-selects, so the spectral year can't drift off the labels).
    """
    dates: dict[int, str] = {}

    # Slot 0 — autumn from the previous year.
    d0 = _best_date_in_window(
        coords_wgs84, f"{tile_year - 1}-08-15", f"{tile_year - 1}-10-31",
        mode="era5_then_scl", scl_backend=scl_backend)
    if d0:
        dates[0] = d0

    # Slots 1-3 — VPP-guided growing season (current year).
    for slot, (doy_start, doy_end) in enumerate(vpp_windows or [], start=1):
        if slot > 3:
            break
        ds, de = doy_to_date_range(tile_year, doy_start, doy_end)
        di = _best_date_in_window(
            coords_wgs84, ds, de, mode="era5_then_scl", scl_backend=scl_backend)
        if di:
            dates[slot] = di

    # Slot 4 — 2016 summer background. ERA5 + STAC-existence (pre-2018, no SCL).
    for year in (background_year, background_fallback_year):
        dbg = _best_date_in_window(
            coords_wgs84, f"{year}-06-01", f"{year}-08-16",
            mode="era5_then_stac", scl_backend=scl_backend)
        if dbg:
            dates[4] = dbg
            break

    return dates


def fetch_aux_channels(bbox_3006: dict, tile: "TileConfig") -> dict[str, np.ndarray]:
    """Fetch auxiliary channels for a tile: VPP phenology, DEM, SKG forestry
    (height/volume/basal_area/diameter) and SLU markfukt.

    Returns dict of channel_name → (H, W) float32. Each source is independently
    skip-on-fail, so a single unavailable webservice never aborts the tile.
    """
    tile.assert_bbox_matches(bbox_3006)
    aux = {}

    # VPP phenology (5 bands) — goes through CDSE semaphore
    _CDSE_SEMAPHORE.acquire()
    try:
        from imint.training.cdse_vpp import fetch_vpp_tiles
        vpp = fetch_vpp_tiles(
            west=bbox_3006["west"],
            south=bbox_3006["south"],
            east=bbox_3006["east"],
            north=bbox_3006["north"],
            size_px=tile.size_px,
        )
        for band in ["sosd", "eosd", "length", "maxv", "minv"]:
            if band in vpp and vpp[band] is not None:
                aux[f"vpp_{band}"] = vpp[band].astype(np.float32)
        _CDSE_SEMAPHORE.report_success()
    except Exception:
        _CDSE_SEMAPHORE.report_failure()
    finally:
        _CDSE_SEMAPHORE.release()

    # DEM (Copernicus GLO-30) — separate API, no CDSE semaphore
    try:
        from imint.training.copernicus_dem import fetch_dem_tile
        dem = fetch_dem_tile(
            west=bbox_3006["west"],
            south=bbox_3006["south"],
            east=bbox_3006["east"],
            north=bbox_3006["north"],
            size_px=tile.size_px,
        )
        if dem is not None:
            aux["dem"] = dem.astype(np.float32)
    except Exception:
        pass

    # SKG forestry — Skogsstyrelsen ImageServer (height) + Skogliga grunddata
    # (volume/basal_area/diameter). Free webservice, no CDSE semaphore.
    w, s, e, n = bbox_3006["west"], bbox_3006["south"], bbox_3006["east"], bbox_3006["north"]
    try:
        from imint.training.skg_grunddata import (
            fetch_basal_area_tile,
            fetch_diameter_tile,
            fetch_tree_height_tile,
            fetch_volume_tile,
        )

        for name, fetch_fn in (
            ("height", fetch_tree_height_tile),
            ("volume", fetch_volume_tile),
            ("basal_area", fetch_basal_area_tile),
            ("diameter", fetch_diameter_tile),
        ):
            try:
                arr = fetch_fn(w, s, e, n, size_px=tile.size_px)
                if arr is not None:
                    aux[name] = arr.astype(np.float32)
            except Exception:
                pass
    except Exception:
        pass

    # SLU Markfuktighetskarta — raw codes → float32: 0=nodata (NaN),
    # 1-100=moisture/100, 101=water (1.01). Mirrors prefetch_aux.
    try:
        from imint.training.slu_markfukt import fetch_markfukt_tile

        raw = fetch_markfukt_tile(w, s, e, n, size_px=tile.size_px)
        if raw is not None:
            mf = raw.astype(np.float32)
            mf[(raw == 0) | (raw > 101)] = np.nan   # nodata (incl. 255)
            valid = (raw >= 1) & (raw <= 100)
            mf[valid] = mf[valid] / 100.0
            mf[raw == 101] = 1.01
            aux["markfukt"] = mf
    except Exception:
        pass

    return aux


# Per-thread rasterio handles for the NMD raster.
#
# Rasterio's DatasetReader.read() is NOT thread-safe: two threads
# calling read() with different windows on the same handle can race
# and return data from the wrong location. We saw this in the multi-
# threaded build-labels pipeline (2026-04) — saved nmd_label_raw
# disagreed with a fresh single-threaded read by 33%, with sharp
# class-jump seams at unpredictable column positions.
#
# Fix: keep one rasterio handle per thread via threading.local(),
# keyed by the file path so a thread that touches multiple NMD files
# (rare, but possible) doesn't end up with the wrong handle.
import threading as _threading

_NMD_TLS = _threading.local()


def _get_nmd_handle(path: str):
    """Return a per-thread rasterio handle for ``path``.

    The handle is opened lazily on first call from each thread and
    cached for that thread's lifetime. Different threads each get
    their own handle so rasterio reads can't race.
    """
    cache = getattr(_NMD_TLS, "by_path", None)
    if cache is None:
        cache = {}
        _NMD_TLS.by_path = cache
    src = cache.get(path)
    if src is None:
        import rasterio
        src = rasterio.open(path)
        cache[path] = src
    return src


def fetch_nmd_label_local(
    bbox_3006: dict,
    tile: "TileConfig",
    nmd_raster: str = "data/nmd/nmd2018bas_ogeneraliserad_v1_1.tif",
) -> np.ndarray | None:
    """Read NMD 19-class sequential label from local GeoTIFF raster.

    Args:
        bbox_3006: Tile bounding box in SWEREF99 TM.
        tile: Tile geometry — used to size the output raster.
        nmd_raster: Path to NMD GeoTIFF.

    Returns (H, W) uint8 with indices 0-19, or None if unavailable.
    Falls back to openEO remote fetch if the local raster is missing.

    Thread-safe: each calling thread gets its own rasterio handle.
    """
    tile.assert_bbox_matches(bbox_3006)

    if not os.path.exists(nmd_raster):
        return _fetch_nmd_label_remote(bbox_3006, tile)

    w = bbox_3006["west"]; s = bbox_3006["south"]
    e = bbox_3006["east"]; n = bbox_3006["north"]

    try:
        src = _get_nmd_handle(nmd_raster)
    except Exception:
        return _fetch_nmd_label_remote(bbox_3006, tile)

    b = src.bounds
    if w < b.left or e > b.right or s < b.bottom or n > b.top:
        return None

    # Lattice-aligned bbox → integer window → pixel-exact native read.
    # native_window raises loudly when the bbox is off the NMD lattice; that
    # is a caller bug, so it must propagate rather than fall through to the
    # openEO target_shape resample below — which would silently re-hide it.
    window = tile.native_window(src.transform, w, s, e, n)

    try:
        nmd_raw = src.read(1, window=window)
    except Exception:
        return _fetch_nmd_label_remote(bbox_3006, tile)

    from imint.training.class_schema import nmd_raster_to_lulc
    return nmd_raster_to_lulc(nmd_raw).astype(np.uint8)


def _fetch_nmd_label_remote(bbox_3006: dict, tile: "TileConfig") -> np.ndarray | None:
    """Fallback: fetch NMD via openEO (requires internet)."""
    try:
        coords_wgs84 = bbox_3006_to_wgs84(bbox_3006)
        from imint.fetch import fetch_nmd_data
        result = fetch_nmd_data(
            coords_wgs84, target_shape=(tile.size_px, tile.size_px),
        )
        if result and result.nmd_raster is not None:
            return result.nmd_raster
    except Exception:
        pass
    return None


def fetch_background_frame(
    bbox_3006: dict,
    tile: "TileConfig",
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
                    size_px=tile.size_px,
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
    tile: "TileConfig",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Stack seasonal scenes into multitemporal array.

    Pads missing frames with zeros, then copies nearest valid frame.

    Returns:
        image: (T*6, H, W) float32
        temporal_mask: (T,) uint8
        doy: (T,) int32
        dates: list of date strings
    """
    size_px = tile.size_px
    frames = []
    valid = []
    dates = []
    doys = []

    for scene, date_str in scene_results:
        if scene is not None:
            # Ensure correct size
            if scene.shape[1] != size_px or scene.shape[2] != size_px:
                padded = np.zeros((N_BANDS, size_px, size_px), dtype=np.float32)
                h = min(scene.shape[1], size_px)
                w = min(scene.shape[2], size_px)
                padded[:, :h, :w] = scene[:, :h, :w]
                frames.append(padded)
            else:
                frames.append(scene)
            valid.append(True)
        else:
            frames.append(np.zeros((N_BANDS, size_px, size_px), dtype=np.float32))
            valid.append(False)

        dates.append(date_str)
        if date_str:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            doys.append(dt.timetuple().tm_yday)
        else:
            doys.append(0)

    # Pad/trim to num_frames
    while len(frames) < num_frames:
        frames.append(np.zeros((N_BANDS, size_px, size_px), dtype=np.float32))
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
