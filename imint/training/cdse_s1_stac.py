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
      a σ⁰ tile; ``fetch_s1_scene`` is gone.
    * The product download cache helpers, now without a caller.

See ``docs/training/s1_fetch.md`` for the history of this path.
"""
from __future__ import annotations

import os
import threading
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from typing import Any

import numpy as np

# Reuse CDSE OAuth token machinery — STAC search is anonymous, but asset
# downloads go through the authenticated zipper endpoint.
from imint.data.cdse_vpp import _get_token
from imint.data import s1_shared

# CDSE Keycloak token endpoint (password grant). The zipper/OData download
# nodes reject the client_credentials service-account token with 401 (noted
# in k8s/enrich-s1-job.yaml 2026-08-04); the resource-owner password grant
# against the public client is the account CDSE authorises for downloads.
_CDSE_KEYCLOAK_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
_pw_token: str | None = None
_pw_token_expires: float = 0.0
_pw_token_lock = threading.Lock()


def _get_download_token() -> str:
    """Token for authenticated COG/OData downloads.

    Prefers the resource-owner **password** grant (``CDSE_USER`` +
    ``CDSE_PASSWORD``, or the ``user``/``password`` keys of the
    ``cdse-credentials`` secret) — that is the account the download nodes
    authorise. Falls back to the client-credentials token from
    :func:`cdse_vpp._get_token` when no password is configured (older setups
    where the service account did have download rights).
    """
    import json
    import time
    import urllib.parse

    user = os.environ.get("CDSE_USER") or os.environ.get("CDSE_USERNAME")
    password = os.environ.get("CDSE_PASSWORD")
    if not (user and password):
        # No password grant available → keep the historical behaviour.
        return _get_token()

    global _pw_token, _pw_token_expires
    with _pw_token_lock:
        if _pw_token and time.time() < _pw_token_expires - 60:
            return _pw_token

    data = urllib.parse.urlencode({
        "grant_type": "password",
        "client_id": "cdse-public",
        "username": user,
        "password": password,
    }).encode()
    req = urllib.request.Request(
        _CDSE_KEYCLOAK_TOKEN_URL, data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
        td = json.loads(resp.read())
    with _pw_token_lock:
        _pw_token = td["access_token"]
        _pw_token_expires = time.time() + td.get("expires_in", 300)
    return _pw_token

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

_PRODUCT_CACHE_DIR = Path(os.environ.get("S1_CACHE_DIR", "/data/s1_cache"))

_DOWNLOAD_TIMEOUT_S = 300  # 2 GB @ reasonable throughput
_REQUEST_TIMEOUT_S = 60
_MAX_RETRIES = 3
_RETRY_DELAY_S = 2.0

# Guard concurrent downloads of the same product so N parallel tile-workers
# don't all race to pull the same 2 GB file.
_product_locks: dict[str, threading.Lock] = {}
_product_locks_guard = threading.Lock()

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
    σ⁰-calibrates each surviving scene (reusing the existing ``s1_shared``
    calibration + product cache), rejects scenes with >``nodata_threshold``
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
        # nodata gate on the raw (pre-dB) zero mask: compute_sigma0 preserves
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
    """Retired (task 0113): always raises, before any product download.

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


def clear_cache(product_id: str | None = None) -> int:
    """Delete cached product(s) from disk.

    Args:
        product_id: Remove only this product's cache dir. ``None`` wipes
            the whole ``S1_CACHE_DIR``.

    Returns:
        Number of product directories removed.
    """
    import shutil

    if not _PRODUCT_CACHE_DIR.exists():
        return 0

    if product_id is not None:
        target = _PRODUCT_CACHE_DIR / product_id
        if target.exists():
            shutil.rmtree(target)
            with _lut_cache_guard:
                _lut_cache.pop(product_id, None)
            return 1
        return 0

    n = 0
    for child in _PRODUCT_CACHE_DIR.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
            n += 1
    return n


# ── Internal: product cache & downloads ──────────────────────────────────

def _product_lock(product_id: str) -> threading.Lock:
    """Get / create the per-product download lock."""
    with _product_locks_guard:
        lock = _product_locks.get(product_id)
        if lock is None:
            lock = threading.Lock()
            _product_locks[product_id] = lock
        return lock


def _prune_cache_to_cap() -> None:
    """Evict oldest product dirs until the cache fits ``S1_CACHE_MAX_GB``.

    The cache shares the training PVC: uncapped it grew to 317 GB and filled
    the 1.6 T volume mid-run (2026-08-17), failing every download with
    ENOSPC. LRU by directory mtime — cache HITS refresh mtime in
    ``_ensure_product_cached``, so recently used products survive. Dirs
    younger than 30 min are never evicted (may be mid-download / in use by a
    concurrent worker). Prunes to 90 % of cap to avoid thrashing.
    """
    import shutil
    import time

    cap_gb = float(os.environ.get("S1_CACHE_MAX_GB", "150"))
    cap = cap_gb * 1024**3
    root = _PRODUCT_CACHE_DIR
    if not root.is_dir():
        return
    dirs = []
    total = 0
    now = time.time()
    for d in root.iterdir():
        if not d.is_dir():
            continue
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        dirs.append((d.stat().st_mtime, size, d))
        total += size
    if total <= cap:
        return
    dirs.sort()  # oldest mtime first
    target = cap * 0.9
    for mtime, size, d in dirs:
        if total <= target:
            break
        if now - mtime < 1800:  # in-use safety window
            continue
        shutil.rmtree(d, ignore_errors=True)
        total -= size
        print(f"  [S1 cache] evicted {d.name} ({size / 1024**3:.1f} GB) "
              f"— cache {total / 1024**3:.0f}/{cap_gb:.0f} GB")


def _ensure_product_cached(item: Any) -> tuple[Path, Path, Path, Path]:
    """Download VV/VH COGs + calibration XMLs if missing. Return their paths.

    Returns:
        (vv_cog, vh_cog, vv_calibration_xml, vh_calibration_xml)
    """
    product_id = item.id
    prod_dir = _PRODUCT_CACHE_DIR / product_id

    with _product_lock(product_id):
        prod_dir.mkdir(parents=True, exist_ok=True)

        vv_cog = prod_dir / "measurement_vv.tiff"
        vh_cog = prod_dir / "measurement_vh.tiff"
        vv_cal = prod_dir / "calibration_vv.xml"
        vh_cal = prod_dir / "calibration_vh.xml"

        if all(p.exists() and p.stat().st_size > 0 for p in (vv_cog, vh_cog, vv_cal, vh_cal)):
            os.utime(prod_dir)  # refresh LRU recency on cache hit
            return vv_cog, vh_cog, vv_cal, vh_cal

        _prune_cache_to_cap()  # make room BEFORE adding a new product

        vv_cog_url, vh_cog_url = s1_shared.pick_measurement_urls(item)
        vv_cal_url, vh_cal_url = s1_shared.pick_calibration_urls(item)

        token = _get_download_token()

        if not vv_cog.exists() or vv_cog.stat().st_size == 0:
            _download(vv_cog_url, vv_cog, token)
        if not vh_cog.exists() or vh_cog.stat().st_size == 0:
            _download(vh_cog_url, vh_cog, token)
        if not vv_cal.exists() or vv_cal.stat().st_size == 0:
            _download(vv_cal_url, vv_cal, token)
        if not vh_cal.exists() or vh_cal.stat().st_size == 0:
            _download(vh_cal_url, vh_cal, token)

        return vv_cog, vh_cog, vv_cal, vh_cal


def _download(url: str, dest: Path, token: str) -> None:
    """Stream a URL to disk via an atomic tmp → rename pattern."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}"},
    )
    last_err: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=_DOWNLOAD_TIMEOUT_S) as resp, \
                 open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(1 << 20)  # 1 MB
                    if not chunk:
                        break
                    f.write(chunk)
            tmp.rename(dest)
            return
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            last_err = e
            if isinstance(e, urllib.error.HTTPError) and e.code == 401:
                # Token expired mid-download — refresh once.
                token = _get_download_token()
                req = urllib.request.Request(
                    url,
                    headers={"Authorization": f"Bearer {token}"},
                )
            if attempt < _MAX_RETRIES:
                import time
                time.sleep(_RETRY_DELAY_S * (attempt + 1))
                continue
            raise
        finally:
            if tmp.exists() and not dest.exists():
                # Left-over partial on failure; clean up so next retry is clean.
                try:
                    tmp.unlink()
                except OSError:
                    pass
    if last_err is not None:
        raise last_err
