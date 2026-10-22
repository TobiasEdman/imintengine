"""Sentinel-1 GRD over CDSE STAC — σ⁰ path RETIRED (task 0113, 2026-10-22).

This module used to download whole IW-GRDH products from CDSE and compute
σ⁰ = DN² / LUT² on a window warped over the product's own GCPs. Those tie
points sit at ellipsoid height, so the sea is expected 41–56 m toward the
satellite (vessel-chain geometry, not measured on the tiles), and the calibration LUT was sampled at warped-grid offsets rather
than radar line/pixel. The path is retired, not fixed: no model was trained
on its tiles, and the v4 training tiles come from ``pc_s1_rtc`` (RTC γ⁰).

What remains:
    * STAC search with WAF backoff, orbit probing
      (:func:`probe_orbits_with_items`) and the season-composite loop
      (:func:`fetch_s1_season_composite`, :func:`_select_spread_scenes`,
      :func:`_nodata_median`) — ``pc_s1_rtc`` reuses the scene spread rule.
    * :func:`_read_calibrated_scene` refuses, so the composite cannot produce
      a σ⁰ tile; ``fetch_s1_scene``, the GRD product download/cache
      (``$S1_CACHE_DIR``) and ``clear_cache`` are gone.

See ``docs/training/s1_fetch.md`` for the history of this path.
"""
from __future__ import annotations

import os
import threading
import warnings
from typing import Any

import numpy as np

from imint.data import s1_shared

# ── Module constants ─────────────────────────────────────────────────────

_STAC_ROOT = "https://stac.dataspace.copernicus.eu/v1"
# Collection ID is lowercase ``sentinel-1-grd`` per CDSE STAC v1
# (verified 2026-05-02 via direct GET against
# /v1/collections/sentinel-1-grd which returned the collection metadata).
# The pre-2026-05-02 value "SENTINEL-1-GRD" returned 0 items on every
# search — the v1 endpoint silently treats unknown collections as empty
# rather than 404, which is why our cascade always returned None and
# fell through to MPC. Probe-cdse-collections-pphd2 only listed CCM +
# CLMS collections, suggesting Sentinel collections are paginated
# beyond the default page size — but the per-collection GET works.
_STAC_COLLECTION = "sentinel-1-grd"

# CDSE STAC's Cloudflare WAF rate-limits by source IP. All tile-workers in a
# pod share one IP, so a burst of parallel searches (orbit probe + 2
# composites × N workers) trips a 429 storm that the per-call backoff then
# serialises into multi-minute sleeps. Serialise searches process-wide with a
# minimum spacing so we stay under the WAF's rate rather than bouncing off it.
# Tunable via $S1_STAC_MIN_INTERVAL_S (default 1.0s ≈ ≤1 search/s/IP).
_STAC_MIN_INTERVAL_S = float(os.environ.get("S1_STAC_MIN_INTERVAL_S", "1.0"))
_stac_rate_lock = threading.Lock()
_stac_last_search_t = [0.0]  # mutable cell, guarded by _stac_rate_lock


def _stac_rate_limit() -> None:
    """Block until at least ``_STAC_MIN_INTERVAL_S`` has passed since the last
    search, then stamp the clock. Process-global (all workers, one IP)."""
    import time

    if _STAC_MIN_INTERVAL_S <= 0:
        return
    with _stac_rate_lock:
        wait = _STAC_MIN_INTERVAL_S - (time.monotonic() - _stac_last_search_t[0])
        if wait > 0:
            time.sleep(wait)
        _stac_last_search_t[0] = time.monotonic()


# ── Public API ───────────────────────────────────────────────────────────

def _import_stac_client():
    """Return the pystac-client ``Client`` class, or raise a clear ImportError.

    Kept in one place so the STAC-search functions stay testable: unit tests
    monkeypatch ``_stac_search_with_backoff`` (which is what actually touches
    the client), and the mocked path never reaches a real search — but the
    functions still resolve ``Client`` up front, so this must raise only when
    pystac-client is genuinely needed and absent.
    """
    try:
        from pystac_client import Client
        return Client
    except ImportError as e:
        raise ImportError(
            "cdse_s1_stac requires pystac-client. "
            "Install with: pip install pystac-client"
        ) from e


def fetch_s1_season_composite(
    west: float,
    south: float,
    east: float,
    north: float,
    *,
    doy_window: tuple[int, int],
    year: int,
    orbit_direction: str,
    crs: str = "http://www.opengis.net/def/crs/EPSG/0/3006",
    size_px: int | tuple[int, int] = 256,
    max_scenes: int = 3,
    output_db: bool = True,
    nodata_threshold: float = 0.10,
    items: list[Any] | None = None,
) -> tuple[np.ndarray, list[str], str] | None:
    """Per-orbit **median** VV/VH season composite over ≤``max_scenes`` scenes.

    Searches CDSE STAC for every IW-GRDH Sentinel-1 scene inside the
    ``doy_window`` of ``year``, keeps only ``orbit_direction`` (never mixes
    ASC/DESC — their look geometry differs and a mixed median corrupts σ⁰),
    reads each surviving scene through :func:`_read_calibrated_scene` (retired,
    task 0113: it raises), rejects scenes with >``nodata_threshold``
    zero fraction, and returns the **per-pixel median** VV/VH over up to
    ``max_scenes`` scenes spread across the window (not clustered).

    Args:
        west, south, east, north: Tile bbox in ``crs``.
        doy_window: ``(doy_start, doy_end)`` growing-season window (inclusive).
        year: Calendar year the window belongs to (label year, or 2016).
        orbit_direction: ``"ASCENDING"`` / ``"DESCENDING"`` — the tile's
            dominant orbit; used for BOTH composites so they stay comparable.
        crs: OGC CRS URI for the bbox. Default EPSG:3006.
        size_px: Output H×W.
        max_scenes: Cap on scenes contributing to the median (speckle sweet
            spot ≈ 3; the plan's CDSE budget assumes ≤3).
        output_db: Return ``10·log10(σ⁰)`` per band (default True — dB is what
            the SAR encoders' normalizers expect).
        nodata_threshold: Reject a scene when its VV zero fraction exceeds
            this (swath-edge / partial coverage).
        items: Pre-fetched STAC items for this window (e.g. from
            :func:`probe_orbit_availability`) — skips the redundant STAC
            search, halving WAF-rate-limited search traffic. Filtered to
            ``orbit_direction`` internally. ``None`` runs the search.

    Returns:
        ``(sar, contributing_dates, orbit)`` on success:

            * ``sar``: ``(2, H, W)`` float32 median composite, [VV, VH].
            * ``contributing_dates``: ISO ``YYYY-MM-DD`` of each scene used.
            * ``orbit``: the resolved orbit direction (== ``orbit_direction``).

        ``None`` when no same-orbit scene survived the window + nodata gate.
    """
    from rasterio.warp import transform_bounds

    from .vpp_windows import doy_to_date_str

    h_px, w_px = (size_px, size_px) if isinstance(size_px, int) else size_px
    s1_shared.assert_bbox_size_match(west, south, east, north, h_px, w_px)
    epsg = s1_shared.crs_uri_to_epsg(crs)

    bbox_4326 = transform_bounds(
        f"EPSG:{epsg}", "EPSG:4326", west, south, east, north, densify_pts=21,
    )

    doy_start, doy_end = doy_window
    date_start = doy_to_date_str(year, max(1, doy_start))
    date_end = doy_to_date_str(year, min(365, doy_end))
    dt_from = f"{date_start}T00:00:00Z"
    # Inclusive of the end day.
    dt_to = f"{date_end}T23:59:59Z"

    if items is None:
        items = _stac_search_with_backoff(
            None, bbox_4326, dt_from, dt_to, f"{date_start}..{date_end}",
        )
    if not items:
        return None

    # ONE orbit direction only — this is the correctness risk (mixed-orbit
    # median corrupts backscatter geometry). filter_iw_grdh already drops the
    # other orbit; assert it below when reading.
    items = s1_shared.filter_iw_grdh(items, orbit_direction)
    if not items:
        return None

    ordered = _select_spread_scenes(items, bbox_4326, max_scenes)

    vv_stack: list[np.ndarray] = []
    vh_stack: list[np.ndarray] = []
    used_dates: list[str] = []
    for item in ordered:
        obs_orbit = s1_shared.orbit_from_item(item)
        if obs_orbit and obs_orbit.upper() != orbit_direction.upper():
            # Defensive: filter_iw_grdh should have removed these already.
            continue
        sar = _read_calibrated_scene(item, west, south, east, north, epsg,
                                     h_px, w_px, output_db=output_db)
        if sar is None:
            continue
        # nodata gate on the raw (pre-dB) zero mask: the σ⁰ reader preserved
        # DN==0 as 0.0, and dB maps 0 → 0.0 too, so a zero VV pixel is nodata
        # in both spaces.
        if float((sar[0] == 0).mean()) > nodata_threshold:
            continue
        vv_stack.append(sar[0])
        vh_stack.append(sar[1])
        dt = s1_shared.item_datetime(item)
        used_dates.append(dt.strftime("%Y-%m-%d") if dt else "")
        if len(vv_stack) >= max_scenes:
            break

    if not vv_stack:
        return None

    # Per-pixel median, ignoring nodata (0) so a swath-edge pixel present in
    # only some scenes is not dragged toward 0. A pixel that is nodata in
    # EVERY contributing scene stays 0 (genuine gap).
    vv = _nodata_median(np.stack(vv_stack, axis=0))
    vh = _nodata_median(np.stack(vh_stack, axis=0))
    composite = np.stack([vv, vh], axis=0).astype(np.float32)
    return composite, used_dates, orbit_direction.upper()


def _select_spread_scenes(
    items: list[Any],
    bbox_4326: tuple[float, float, float, float],
    max_scenes: int,
) -> list[Any]:
    """Order candidate scenes so a ≤``max_scenes`` prefix is spread in time.

    Scenes are sorted by acquisition date; then a subset of size
    ``2*max_scenes`` (to leave slack for nodata rejections) is picked at
    evenly-spaced temporal positions across the window — median of scenes
    spread across the season suppresses speckle better than a temporal
    cluster. Items with the same date are ordered by bbox overlap so the
    best-covering scene is tried first.
    """
    dated = [(s1_shared.item_datetime(it), it) for it in items]
    dated.sort(
        key=lambda t: (
            t[0].timestamp() if t[0] is not None else 0.0,
            -s1_shared.item_overlap_area(t[1], bbox_4326),
        )
    )
    ordered = [it for _, it in dated]
    n = len(ordered)
    take = min(n, max(max_scenes * 2, max_scenes))
    if n <= take:
        return ordered
    # Evenly-spaced indices across the sorted-by-date list.
    idxs = np.linspace(0, n - 1, take).round().astype(int)
    seen: set[int] = set()
    spread = []
    for i in idxs:
        if i not in seen:
            seen.add(int(i))
            spread.append(ordered[int(i)])
    return spread


def _read_calibrated_scene(
    item: Any,
    west: float, south: float, east: float, north: float,
    epsg: int, h_px: int, w_px: int,
    *,
    output_db: bool,
) -> np.ndarray | None:
    """Retired (task 0113): always raises.

    It window-read a GCP-referenced GRD (ellipsoid-height tie points) and
    sampled the calibration LUT on the warped grid — see the module
    docstring. Kept by name only so the composite loop above fails loudly.
    """
    raise RuntimeError(f"{item!r}: {s1_shared.GCP_SIGMA0_RETIRED}")


def _nodata_median(stack: np.ndarray) -> np.ndarray:
    """Per-pixel median over a ``(N, H, W)`` stack, treating 0 as nodata.

    A pixel present (nonzero) in ≥1 scene → median of its nonzero values.
    A pixel that is 0 in every scene → 0 (genuine nodata, preserved).
    """
    masked = np.where(stack == 0, np.nan, stack)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        # An all-nodata pixel yields an "All-NaN slice" warning; that pixel is
        # scrubbed to 0 below (genuine gap), so the warning is expected noise.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(masked, axis=0)
    return np.nan_to_num(med, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def probe_orbits_with_items(
    west: float, south: float, east: float, north: float,
    *,
    windows: list[tuple[tuple[int, int], int]],
    crs: str = "http://www.opengis.net/def/crs/EPSG/0/3006",
) -> tuple[str | None, dict[int, list[Any]]]:
    """Dominant orbit across windows + the raw items per window index.

    Runs ONE STAC search per window and returns:
      * the orbit direction with the most valid IW-GRDH passes across all
        windows (chosen ONCE, reused for both composites so their
        backscatter is comparable — never ASC-vs-DESC across the pair), or
        ``None`` if no scene was found in any window;
      * ``{window_index: [items]}`` so the caller can feed the already-
        fetched items straight into :func:`fetch_s1_season_composite`
        (``items=`` kwarg), avoiding a second search per window against the
        WAF-rate-limited endpoint.
    """
    from rasterio.warp import transform_bounds

    from .vpp_windows import doy_to_date_str

    epsg = s1_shared.crs_uri_to_epsg(crs)
    bbox_4326 = transform_bounds(
        f"EPSG:{epsg}", "EPSG:4326", west, south, east, north, densify_pts=21,
    )

    counts: dict[str, int] = {"ASCENDING": 0, "DESCENDING": 0}
    items_by_window: dict[int, list[Any]] = {}
    for wi, ((doy_start, doy_end), year) in enumerate(windows):
        date_start = doy_to_date_str(year, max(1, doy_start))
        date_end = doy_to_date_str(year, min(365, doy_end))
        items = _stac_search_with_backoff(
            None, bbox_4326,
            f"{date_start}T00:00:00Z", f"{date_end}T23:59:59Z",
            f"{date_start}..{date_end}",
        )
        items = list(items) if items else []
        items_by_window[wi] = items
        for it in s1_shared.filter_iw_grdh(items, None):
            orbit = s1_shared.orbit_from_item(it)
            if orbit in counts:
                counts[orbit] += 1

    if counts["ASCENDING"] == 0 and counts["DESCENDING"] == 0:
        return None, items_by_window
    return max(counts, key=counts.get), items_by_window


def probe_orbit_availability(
    west: float, south: float, east: float, north: float,
    *,
    windows: list[tuple[tuple[int, int], int]],
    crs: str = "http://www.opengis.net/def/crs/EPSG/0/3006",
) -> str | None:
    """Dominant orbit across ``windows`` (thin wrapper over
    :func:`probe_orbits_with_items` that drops the per-window items)."""
    orbit, _ = probe_orbits_with_items(
        west, south, east, north, windows=windows, crs=crs,
    )
    return orbit


def _stac_search_with_backoff(
    Client: Any | None,
    bbox_4326: tuple[float, float, float, float],
    dt_from: str,
    dt_to: str,
    date: str,
    *,
    max_attempts: int = 5,
    base_delay_s: float = 2.0,
) -> list[Any] | None:
    """Run a STAC search with WAF/429-aware exponential backoff.

    The CDSE STAC frontend rate-limits aggressive callers via Cloudflare-
    style WAF rules and returns ``HTTP 429`` (or a JSON body with
    ``"status":429``). Returning ``None`` on the first 429 is what made
    the cascade façade fall through to MPC immediately at scale —
    instead, retry locally with backoff so CDSE remains the primary path
    when it's just being throttled.

    Returns:
        List of STAC items (possibly empty) on success; ``None`` if
        every attempt failed (the caller should then fall through to
        the next backend).
    """
    import time
    if Client is None:
        Client = _import_stac_client()
    delay = base_delay_s
    for attempt in range(max_attempts):
        try:
            _stac_rate_limit()
            client = Client.open(_STAC_ROOT)
            search = client.search(
                collections=[_STAC_COLLECTION],
                bbox=list(bbox_4326),
                datetime=f"{dt_from}/{dt_to}",
                limit=50,
            )
            return list(search.items())
        except Exception as e:
            msg = str(e)
            is_rate_limit = "429" in msg or "Rate limit" in msg or "WAF" in msg
            if is_rate_limit and attempt < max_attempts - 1:
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
                continue
            print(f"    [STAC S1] {date}: search failed: {e}")
            return None
    return None
