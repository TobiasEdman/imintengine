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
import re
import threading
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

# Reuse CDSE OAuth token machinery — STAC search is anonymous, but asset
# downloads go through the authenticated zipper endpoint.
from .cdse_vpp import _get_token

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

# Small in-memory cache of parsed calibration LUTs (pid → {'vv': LUT, 'vh': LUT}).
# The LUT itself is already cheap (~1 MB per product) but parsing the XML on
# every window read is wasteful when one product serves many tiles.
_lut_cache: dict[str, dict[str, dict[str, np.ndarray]]] = {}
_lut_cache_guard = threading.Lock()


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
    from rasterio.windows import from_bounds as window_from_bounds
    from rasterio.enums import Resampling

    h_px, w_px = (size_px, size_px) if isinstance(size_px, int) else size_px

    # Mirror the defensive bbox/size consistency check from the SH fetcher —
    # 10 m native GSD means a 256 px tile must be exactly a 2560 m extent,
    # otherwise the window read silently resamples to the wrong GSD.
    expected_m = w_px * 10
    if abs((east - west) - expected_m) > 1 or abs((north - south) - expected_m) > 1:
        raise ValueError(
            f"fetch_s1_scene (stac): bbox/size_px mismatch. "
            f"bbox ew={east - west}m ns={north - south}m size_px={w_px} "
            f"→ expected {expected_m}m extent."
        )

    epsg = _crs_uri_to_epsg(crs)

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

    try:
        client = Client.open(_STAC_ROOT)
        search = client.search(
            collections=[_STAC_COLLECTION],
            bbox=list(bbox_4326),
            datetime=f"{dt_from}/{dt_to}",
            limit=50,
        )
        items = list(search.items())
    except Exception as e:
        print(f"    [STAC S1] {date}: search failed: {e}")
        return None

    items = _filter_iw_grdh(items, orbit_direction)
    if not items:
        return None

    # Prefer the item with the largest footprint intersection with the tile
    # bbox — minimizes the chance that the window read hits the product edge.
    item = _pick_best_item(items, bbox_4326)

    # ── 2/3. Cache hit or download ───────────────────────────────────────
    product_id = item.id
    try:
        vv_path, vh_path, vv_cal_path, vh_cal_path = _ensure_product_cached(item)
    except Exception as e:
        print(f"    [STAC S1] {product_id}: download failed: {e}")
        return None

    # ── 4. Window read (VV + VH) ─────────────────────────────────────────
    try:
        vv_dn, vv_win = _read_window(
            vv_path, west, south, east, north, epsg, h_px, w_px, Resampling.bilinear,
        )
        vh_dn, vh_win = _read_window(
            vh_path, west, south, east, north, epsg, h_px, w_px, Resampling.bilinear,
        )
    except Exception as e:
        print(f"    [STAC S1] {product_id}: window read failed: {e}")
        return None

    # ── 5. Calibration ───────────────────────────────────────────────────
    try:
        vv_lut = _interp_lut(vv_cal_path, vv_win, product_id, "vv", h_px, w_px)
        vh_lut = _interp_lut(vh_cal_path, vh_win, product_id, "vh", h_px, w_px)
    except Exception as e:
        print(f"    [STAC S1] {product_id}: calibration failed: {e}")
        return None

    # σ⁰ = DN² / LUT²  — standard linear sigma-nought on the ellipsoid.
    with np.errstate(divide="ignore", invalid="ignore"):
        vv_sigma0 = (vv_dn.astype(np.float32) ** 2) / (vv_lut ** 2)
        vh_sigma0 = (vh_dn.astype(np.float32) ** 2) / (vh_lut ** 2)

    # Pixels with DN==0 are genuine nodata — preserve as 0 (not NaN) so the
    # downstream nodata-fraction check behaves the same as the SH path.
    vv_sigma0 = np.where(vv_dn == 0, 0.0, vv_sigma0).astype(np.float32)
    vh_sigma0 = np.where(vh_dn == 0, 0.0, vh_sigma0).astype(np.float32)

    if output_db:
        vv_sigma0 = _to_db(vv_sigma0)
        vh_sigma0 = _to_db(vh_sigma0)

    sar = np.stack([vv_sigma0, vh_sigma0], axis=0)

    if nodata_threshold is not None:
        if float((sar[0] == 0).mean()) > nodata_threshold:
            return None

    orbit = _orbit_from_item(item) or (orbit_direction or "UNKNOWN")
    return sar, orbit


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
    with _lut_cache_guard:
        _lut_cache.clear()
    return n


# ── Internal: STAC helpers ───────────────────────────────────────────────

def _crs_uri_to_epsg(crs: str) -> int:
    """Extract EPSG code from an OGC CRS URI (or raw ``EPSG:XXXX``)."""
    if crs.upper().startswith("EPSG:"):
        return int(crs.split(":", 1)[1])
    m = re.search(r"/EPSG/\d+/(\d+)", crs)
    if m:
        return int(m.group(1))
    raise ValueError(f"Unrecognised CRS URI: {crs}")


def _filter_iw_grdh(items: list[Any], orbit_direction: str | None) -> list[Any]:
    """Keep only IW GRDH items matching the requested orbit direction."""
    out = []
    for it in items:
        p = it.properties or {}

        # Mode filter — STAC uses various property names across providers.
        mode = (
            p.get("sar:instrument_mode")
            or p.get("instrumentMode")
            or p.get("sentinel1:mode")
            or ""
        )
        if mode and mode.upper() != "IW":
            continue

        # Product-type filter (GRDH). Absence → don't reject, since not
        # every provider populates it; the collection is already GRD-only.
        ptype = (
            p.get("sar:product_type")
            or p.get("productType")
            or p.get("sentinel1:product_type")
            or ""
        )
        if ptype and "GRDH" not in ptype.upper() and "GRD" not in ptype.upper():
            continue

        if orbit_direction:
            obs = _orbit_from_item(it)
            if obs and obs.upper() != orbit_direction.upper():
                continue

        out.append(it)
    return out


def _orbit_from_item(item: Any) -> str | None:
    """Extract orbit direction from a STAC item (best-effort across schemas)."""
    p = item.properties or {}
    for key in (
        "sat:orbit_state",
        "sar:orbit_state",
        "orbitDirection",
        "sentinel1:orbit_direction",
    ):
        v = p.get(key)
        if v:
            return str(v).upper()
    return None


def _pick_best_item(items: list[Any], bbox_4326: tuple[float, float, float, float]) -> Any:
    """Pick the item whose bbox overlaps the request bbox most."""
    def overlap_area(it: Any) -> float:
        b = it.bbox
        if not b:
            return 0.0
        w = max(0.0, min(b[2], bbox_4326[2]) - max(b[0], bbox_4326[0]))
        h = max(0.0, min(b[3], bbox_4326[3]) - max(b[1], bbox_4326[1]))
        return w * h

    return max(items, key=overlap_area)


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

        vv_cog_url, vh_cog_url = _pick_measurement_urls(item)
        vv_cal_url, vh_cal_url = _pick_calibration_urls(item)

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


def _pick_measurement_urls(item: Any) -> tuple[str, str]:
    """Extract VV and VH measurement COG URLs from item.assets."""
    vv_url = vh_url = None
    for name, asset in (item.assets or {}).items():
        href = asset.href
        lname = name.lower()
        lhref = href.lower()
        if "measurement" not in lname and "measurement" not in lhref:
            continue
        if "vv" in lname or "-vv-" in lhref:
            vv_url = href
        elif "vh" in lname or "-vh-" in lhref:
            vh_url = href
    if vv_url is None or vh_url is None:
        raise RuntimeError(
            f"Item {item.id} missing VV or VH measurement asset "
            f"(assets: {list((item.assets or {}).keys())})"
        )
    return vv_url, vh_url


def _pick_calibration_urls(item: Any) -> tuple[str, str]:
    """Extract VV and VH calibration XML URLs from item.assets."""
    vv_url = vh_url = None
    for name, asset in (item.assets or {}).items():
        href = asset.href
        lname = name.lower()
        lhref = href.lower()
        if "calibration" not in lname and "calibration" not in lhref:
            continue
        # Exclude noise files which live alongside calibration.
        if "noise" in lname or "noise" in lhref:
            continue
        if "vv" in lname or "-vv-" in lhref:
            vv_url = href
        elif "vh" in lname or "-vh-" in lhref:
            vh_url = href
    if vv_url is None or vh_url is None:
        raise RuntimeError(
            f"Item {item.id} missing VV or VH calibration asset "
            f"(assets: {list((item.assets or {}).keys())})"
        )
    return vv_url, vh_url


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


# ── Internal: window read + calibration ──────────────────────────────────

def _read_window(
    cog_path: Path,
    west: float, south: float, east: float, north: float,
    src_epsg: int,
    h_px: int, w_px: int,
    resampling: Any,
) -> tuple[np.ndarray, Any]:
    """Read an H×W window from the COG at the requested bbox.

    Returns:
        (dn_array, window) — ``dn_array`` is ``(H, W)`` float32 raw DN values
        (no calibration applied), ``window`` is the ``rasterio.Window`` used
        to read it, needed to interpolate the calibration LUT onto the same
        pixel/line grid.
    """
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds as window_from_bounds

    with rasterio.open(cog_path) as ds:
        dst_bounds = transform_bounds(
            f"EPSG:{src_epsg}", ds.crs,
            west, south, east, north,
            densify_pts=21,
        )
        window = window_from_bounds(*dst_bounds, transform=ds.transform)
        dn = ds.read(
            1,
            window=window,
            out_shape=(h_px, w_px),
            resampling=resampling,
            boundless=True,
            fill_value=0,
        ).astype(np.float32)
    return dn, window


def _interp_lut(
    cal_xml_path: Path,
    window: Any,
    product_id: str,
    pol: str,
    h_px: int, w_px: int,
) -> np.ndarray:
    """Interpolate the σ⁰ calibration LUT onto the window's pixel grid.

    The XML holds a sparse 2-D grid of ``sigmaNought`` values indexed by
    (line, pixel) in full-product coordinates. We:
        1. Parse the grid into flat ``pixels``, ``lines``, ``values`` arrays.
        2. Map the window's (out_H, out_W) output grid to full-product
           (line, pixel) coordinates.
        3. Bilinear-interpolate via ``scipy.interpolate.RegularGridInterpolator``.
    """
    grid = _load_cal_grid(cal_xml_path, product_id, pol)
    lines_ax = grid["lines"]
    pixels_ax = grid["pixels"]
    values = grid["values"]  # (len(lines_ax), len(pixels_ax))

    # Map the window → full-product (line, pixel) at the *output* grid.
    # rasterio.Window gives the source pixel/line extents; we sample (h_px, w_px)
    # uniformly across that extent (bilinear in COG space, nearest-enough in LUT
    # space — the LUT is smooth across thousands of pixels).
    col_off = float(window.col_off)
    row_off = float(window.row_off)
    width = float(window.width)
    height = float(window.height)

    out_cols = np.linspace(col_off, col_off + width, w_px, endpoint=False) + (width / w_px) / 2.0
    out_rows = np.linspace(row_off, row_off + height, h_px, endpoint=False) + (height / h_px) / 2.0

    # Clamp to LUT axis bounds — the RGI will extrapolate at the edges otherwise.
    out_cols = np.clip(out_cols, pixels_ax[0], pixels_ax[-1])
    out_rows = np.clip(out_rows, lines_ax[0], lines_ax[-1])

    try:
        from scipy.interpolate import RegularGridInterpolator
    except ImportError as e:
        raise ImportError(
            "cdse_s1_stac requires scipy for LUT interpolation."
        ) from e

    rgi = RegularGridInterpolator(
        (lines_ax, pixels_ax), values,
        method="linear", bounds_error=False, fill_value=None,
    )

    rr, cc = np.meshgrid(out_rows, out_cols, indexing="ij")
    lut = rgi(np.stack([rr.ravel(), cc.ravel()], axis=-1))
    return lut.reshape(h_px, w_px).astype(np.float32)


def _load_cal_grid(
    cal_xml_path: Path,
    product_id: str,
    pol: str,
) -> dict[str, np.ndarray]:
    """Parse and cache the sigmaNought LUT grid from a calibration XML."""
    with _lut_cache_guard:
        per_prod = _lut_cache.get(product_id)
        if per_prod is not None and pol in per_prod:
            return per_prod[pol]

    tree = ET.parse(cal_xml_path)
    root = tree.getroot()

    # The SAFE schema is un-namespaced for these fields; iterate all
    # calibrationVector elements in document order.
    vectors = list(root.iter("calibrationVector"))
    if not vectors:
        raise RuntimeError(f"{cal_xml_path}: no <calibrationVector> elements")

    # Build a (N_lines, N_pixels) matrix. All vectors share the same pixel
    # grid inside a product — guarded with an explicit check.
    first_pixels = _parse_int_array(vectors[0].find("pixel").text)
    n_pix = len(first_pixels)

    lines = np.empty(len(vectors), dtype=np.float64)
    values = np.empty((len(vectors), n_pix), dtype=np.float64)

    for i, v in enumerate(vectors):
        line_el = v.find("line")
        pix_el = v.find("pixel")
        sig_el = v.find("sigmaNought")
        if line_el is None or pix_el is None or sig_el is None:
            raise RuntimeError(f"{cal_xml_path}: malformed calibrationVector {i}")
        pix_arr = _parse_int_array(pix_el.text)
        sig_arr = _parse_float_array(sig_el.text)
        if len(pix_arr) != n_pix or len(sig_arr) != n_pix:
            raise RuntimeError(
                f"{cal_xml_path}: calibrationVector {i} width mismatch "
                f"({len(pix_arr)}/{len(sig_arr)} vs expected {n_pix})"
            )
        lines[i] = float(line_el.text)
        values[i, :] = sig_arr

    entry = {
        "lines": lines,
        "pixels": np.asarray(first_pixels, dtype=np.float64),
        "values": values,
    }

    with _lut_cache_guard:
        per_prod = _lut_cache.setdefault(product_id, {})
        per_prod[pol] = entry
    return entry


def _parse_int_array(text: str) -> list[int]:
    return [int(x) for x in text.split()]


def _parse_float_array(text: str) -> list[float]:
    return [float(x) for x in text.split()]


def _to_db(x: np.ndarray) -> np.ndarray:
    """Convert linear σ⁰ to dB, replacing non-finite with 0 (nodata)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 10.0 * np.log10(np.maximum(x, 1e-10))
    out = np.where(np.isfinite(out) & (x > 0), out, 0.0)
    return out.astype(np.float32)


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
