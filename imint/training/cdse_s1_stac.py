"""Sentinel-1 GRD fetching via CDSE STAC + direct COG reads + local σ⁰ calibration.

Why this module exists
----------------------
The original ``imint.training.cdse_s1`` fetcher uses the Sentinel Hub Process API
(``https://sh.dataspace.copernicus.eu/api/v1/process``) which bills Processing
Units (PUs) per tile per date. For this project's scale
(≈ 8260 tiles × 4 temporal frames × up to 7 date retries) the monthly free-tier
10k-PU budget is exhausted well before the full dataset is covered, and we ran
out mid-enrichment.

CDSE exposes several other free quotas on the same account; the one that fits
a one-time bulk backfill is the OData/STAC catalog with direct COG access:

    * 12 TB / month bandwidth
    * 50 000 HTTP requests / month against object storage
    * No PU accounting — products are streamed directly from the Copernicus
      S3-backed object store as Cloud-Optimized GeoTIFFs.

A Sentinel-1 GRD IW product is ~2 GB uncompressed; Sweden covered by our
4-frame-per-tile schedule needs on the order of 100–300 unique products total.
Downloading each product **once** and then serving many tile-sized window reads
from the local cache fits easily inside both quotas.

Pipeline
--------
1.  Search ``SENTINEL-1-GRD`` via ``https://stac.dataspace.copernicus.eu/v1``
    for items whose temporal extent contains ``date`` and whose footprint
    intersects the requested bbox. Prefer IW / GRDH.
2.  Keep a module-level cache keyed by product id. If the product's VV / VH
    measurement COGs and calibration XMLs are already in
    ``$S1_CACHE_DIR/<product_id>/``, skip the download step.
3.  Otherwise stream the assets (VV & VH GeoTIFFs + ``calibration-*.xml``) to
    the cache dir with ``urllib.request`` — one download per product, shared
    across every tile that needs it.
4.  Open the VV COG with ``rasterio``, reproject the request bbox into the
    product CRS, build ``rasterio.windows.from_bounds``, and read the window at
    the requested ``size_px`` with bilinear resampling. Same for VH.
5.  Parse the matching ``calibration-*.xml`` (``<sigmaNought>`` LUT),
    bilinearly interpolate the LUT onto the window's pixel/line grid, and
    compute ``σ0 = DN² / LUT²``. This is sigma-nought on the WGS84 ellipsoid —
    no terrain correction, no SNAP, pure Python.

Returned shape and semantics match ``cdse_s1.fetch_s1_scene`` exactly, so
``scripts/enrich_tiles_s1.py`` can switch backends with a single import line.

See also: ``docs/training/s1_fetch.md`` for the why/how of this path.
"""
from __future__ import annotations

import os
import threading
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

# Reuse CDSE OAuth token machinery — STAC search is anonymous, but asset
# downloads go through the authenticated zipper endpoint.
from .cdse_vpp import _get_token
from . import s1_shared

# ── Module constants ─────────────────────────────────────────────────────

_STAC_ROOT = "https://stac.dataspace.copernicus.eu/v1"
_STAC_COLLECTION = "SENTINEL-1-GRD"

_PRODUCT_CACHE_DIR = Path(os.environ.get("S1_CACHE_DIR", "/data/s1_cache"))

_DOWNLOAD_TIMEOUT_S = 300  # 2 GB @ reasonable throughput
_REQUEST_TIMEOUT_S = 60
_MAX_RETRIES = 3
_RETRY_DELAY_S = 2.0

# Guard concurrent downloads of the same product so N parallel tile-workers
# don't all race to pull the same 2 GB file.
_product_locks: dict[str, threading.Lock] = {}
_product_locks_guard = threading.Lock()


# ── Public API ───────────────────────────────────────────────────────────

def fetch_s1_scene(
    west: float,
    south: float,
    east: float,
    north: float,
    date: str,
    *,
    crs: str = "http://www.opengis.net/def/crs/EPSG/0/3006",
    size_px: int | tuple[int, int] = 256,
    orbit_direction: str | None = None,
    output_db: bool = False,
    nodata_threshold: float | None = 0.10,
) -> tuple[np.ndarray, str] | None:
    """Fetch a Sentinel-1 GRD σ⁰ window via STAC + direct COG reads.

    Drop-in replacement for ``imint.training.cdse_s1.fetch_s1_scene`` with
    identical return contract — see the module docstring for the why.

    Args:
        west, south, east, north: Bounding box in ``crs``.
        date: ISO date string (``YYYY-MM-DD``). Items whose time range
            contains this date are considered. (GRD products are effectively
            instantaneous, so this is a point match in practice.)
        crs: OGC-style CRS URI for the input bbox. Default EPSG:3006.
        size_px: Output H×W — int for square, or (H, W) tuple.
        orbit_direction: ``"ASCENDING"`` / ``"DESCENDING"``. ``None`` = any.
        output_db: If True, return ``10·log10(σ⁰)``. Default: linear σ⁰.
        nodata_threshold: Reject tiles where the VV zero fraction exceeds
            this. ``None`` disables the check.

    Returns:
        ``(sar, orbit_direction)`` on success:

            * ``sar``: ``(2, H, W)`` float32, channels = [VV, VH].

        ``None`` when no item matched, any asset could not be downloaded,
        or the nodata threshold was exceeded.
    """
    # Inside-the-function import: the module should load even on machines
    # without pystac-client, and the existing ``fetch_s1_scene`` path keeps
    # working regardless. Only the STAC path requires the dep.
    try:
        from pystac_client import Client
    except ImportError as e:
        raise ImportError(
            "cdse_s1_stac requires pystac-client. "
            "Install with: pip install pystac-client"
        ) from e

    import rasterio  # noqa: F401 — eager fail if missing
    from rasterio.warp import transform_bounds

    h_px, w_px = (size_px, size_px) if isinstance(size_px, int) else size_px

    s1_shared.assert_bbox_size_match(west, south, east, north, h_px, w_px)
    epsg = s1_shared.crs_uri_to_epsg(crs)

    # ── 1. STAC search ───────────────────────────────────────────────────
    bbox_4326 = transform_bounds(
        f"EPSG:{epsg}", "EPSG:4326",
        west, south, east, north,
        densify_pts=21,
    )

    dt = datetime.strptime(date, "%Y-%m-%d")
    # GRD items carry a short (~25 s) acquisition window — a one-day span
    # around the requested date is safe without broadening to ±N days here,
    # since enrich_tiles_s1.py already tries adjacent dates explicitly.
    dt_from = dt.strftime("%Y-%m-%dT00:00:00Z")
    dt_to = (dt + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")

    items = _stac_search_with_backoff(
        Client, bbox_4326, dt_from, dt_to, date,
    )
    if items is None:
        return None
    if not items:
        return None

    items = s1_shared.filter_iw_grdh(items, orbit_direction)
    if not items:
        return None

    # Prefer the item with the largest footprint intersection with the tile
    # bbox — minimizes the chance that the window read hits the product edge.
    item = s1_shared.pick_best_item(items, bbox_4326)

    # ── 2/3. Cache hit or download ───────────────────────────────────────
    product_id = item.id
    try:
        vv_path, vh_path, vv_cal_path, vh_cal_path = _ensure_product_cached(item)
    except Exception as e:
        print(f"    [STAC S1] {product_id}: download failed: {e}")
        return None

    # ── 4. Window read (VV + VH) ─────────────────────────────────────────
    try:
        vv_dn, vv_win = s1_shared.read_window(
            vv_path, west, south, east, north, epsg, h_px, w_px,
        )
        vh_dn, vh_win = s1_shared.read_window(
            vh_path, west, south, east, north, epsg, h_px, w_px,
        )
    except Exception as e:
        print(f"    [STAC S1] {product_id}: window read failed: {e}")
        return None

    # ── 5. Calibration ───────────────────────────────────────────────────
    try:
        vv_lut = s1_shared.interp_lut(vv_cal_path, vv_win, product_id, "vv", h_px, w_px)
        vh_lut = s1_shared.interp_lut(vh_cal_path, vh_win, product_id, "vh", h_px, w_px)
    except Exception as e:
        print(f"    [STAC S1] {product_id}: calibration failed: {e}")
        return None

    sar = s1_shared.compute_sigma0(vv_dn, vh_dn, vv_lut, vh_lut, output_db=output_db)

    if nodata_threshold is not None:
        if float((sar[0] == 0).mean()) > nodata_threshold:
            return None

    orbit = s1_shared.orbit_from_item(item) or (orbit_direction or "UNKNOWN")
    return sar, orbit


def _stac_search_with_backoff(
    Client: Any,
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
    delay = base_delay_s
    for attempt in range(max_attempts):
        try:
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
            return vv_cog, vh_cog, vv_cal, vh_cal

        vv_cog_url, vh_cog_url = s1_shared.pick_measurement_urls(item)
        vv_cal_url, vh_cal_url = s1_shared.pick_calibration_urls(item)

        token = _get_token()

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
                token = _get_token()
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


# ── Smoke test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # A small patch over southern Sweden — 2560 m extent in EPSG:3006,
    # matches the 256 px × 10 m GSD convention of the existing pipeline.
    # Centre is near Lund. Pick a date with known Sentinel-1 coverage over
    # Sweden (orbit 95, descending, every 6 days).
    west, south, east, north = 380000, 6170000, 382560, 6172560
    date = "2023-06-15"

    print(f"[smoke] fetching S1 {date} bbox=({west},{south},{east},{north}) "
          f"cache={_PRODUCT_CACHE_DIR}")
    result = fetch_s1_scene(
        west, south, east, north, date,
        size_px=256,
        orbit_direction=None,
        output_db=False,
        nodata_threshold=None,  # disable for smoke
    )
    if result is None:
        print("[smoke] no scene returned — check date coverage or cache dir permissions")
        raise SystemExit(1)

    sar, orbit = result
    vv, vh = sar[0], sar[1]
    print(f"[smoke] shape={sar.shape} dtype={sar.dtype} orbit={orbit}")
    print(f"[smoke] VV  min={vv.min():.4g} max={vv.max():.4g} mean={vv.mean():.4g} "
          f"nonzero_frac={(vv > 0).mean():.3f}")
    print(f"[smoke] VH  min={vh.min():.4g} max={vh.max():.4g} mean={vh.mean():.4g} "
          f"nonzero_frac={(vh > 0).mean():.3f}")
