"""Shared helpers for Sentinel-1 STAC items and projected window reads.

Users:
    * ``imint.training.pc_s1_rtc`` — RTC γ⁰ season composites (window read)
    * ``imint.training.cdse_s1_stac`` — STAC search / orbit helpers
    * ``imint.fetch`` (vessel chain) — item filtering, footprints, asset URLs

The σ⁰ path over GCP-referenced GRD (``mpc_s1`` / ``aws_s1`` /
``cdse_s1_stac`` ``fetch_s1_scene``) is retired (task 0113): its tie points
sit at ellipsoid height and the calibration lookup was sampled on the warped
grid. :func:`read_window` refuses such sources.

What lives here
---------------
    * STAC item filtering / orbit / footprint helpers
    * Asset URL extraction (VV/VH measurement + calibration)
    * Projected COG window read at requested EPSG and size
    * Calibration LUT XML parsing, σ⁰ assembly + dB conversion (no caller
      since task 0113)
    * CRS-URI parsing

What does NOT live here
-----------------------
    * STAC client construction (each backend points at a different root)
    * Asset download (auth differs: Bearer / anonymous / signed URL)
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np


# ── CRS / STAC item filtering ─────────────────────────────────────────────

def crs_uri_to_epsg(crs: str) -> int:
    """Extract EPSG code from an OGC CRS URI (or raw ``EPSG:XXXX``)."""
    if crs.upper().startswith("EPSG:"):
        return int(crs.split(":", 1)[1])
    m = re.search(r"/EPSG/\d+/(\d+)", crs)
    if m:
        return int(m.group(1))
    raise ValueError(f"Unrecognised CRS URI: {crs}")


def filter_iw_grdh(items: list[Any], orbit_direction: str | None) -> list[Any]:
    """Keep only IW GRDH items matching the requested orbit direction.

    Property names differ across STAC providers — accept all known
    variants and reject only when an explicit value contradicts the
    filter.
    """
    out = []
    for it in items:
        p = it.properties or {}

        mode = (
            p.get("sar:instrument_mode")
            or p.get("instrumentMode")
            or p.get("sentinel1:mode")
            or ""
        )
        if mode and mode.upper() != "IW":
            continue

        ptype = (
            p.get("sar:product_type")
            or p.get("productType")
            or p.get("sentinel1:product_type")
            or ""
        )
        if ptype and "GRDH" not in ptype.upper() and "GRD" not in ptype.upper():
            continue

        if orbit_direction:
            obs = orbit_from_item(it)
            if obs and obs.upper() != orbit_direction.upper():
                continue

        out.append(it)
    return out


def orbit_from_item(item: Any) -> str | None:
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


def pick_best_item(items: list[Any], bbox_4326: tuple[float, float, float, float]) -> Any:
    """Pick the item whose bbox overlaps the request bbox most."""
    return max(items, key=lambda it: item_overlap_area(it, bbox_4326))


def item_overlap_area(item: Any, bbox_4326: tuple[float, float, float, float]) -> float:
    """Bbox-overlap area (deg²) between a STAC item and the request bbox."""
    b = item.bbox
    if not b:
        return 0.0
    w = max(0.0, min(b[2], bbox_4326[2]) - max(b[0], bbox_4326[0]))
    h = max(0.0, min(b[3], bbox_4326[3]) - max(b[1], bbox_4326[1]))
    return w * h


def footprint_intersection_area(item: Any, geometry: Any) -> float:
    """Intersection area (deg²) of an item's footprint polygon with a shapely geometry.

    Unlike :func:`item_overlap_area` this uses the swath footprint, not its
    bbox: S1 footprints are rotated quadrilaterals whose bboxes overlap far
    more than the swaths do. Areas are in WGS84 degrees², as AI2's
    historical-scene selection computes them.
    """
    from shapely.geometry import shape

    if not item.geometry:
        return 0.0
    return float(shape(item.geometry).intersection(geometry).area)


def item_datetime(item: Any):
    """Best-effort acquisition ``datetime`` for a STAC item (None on failure).

    Prefers ``item.datetime`` (pystac parses it), falls back to the
    ``datetime`` / ``start_datetime`` property strings. Used to spread the
    season-composite scenes across the window and to group items by orbit.
    """
    from datetime import datetime as _dt

    dt = getattr(item, "datetime", None)
    if dt is not None:
        return dt
    p = item.properties or {}
    for key in ("datetime", "start_datetime", "end_datetime"):
        v = p.get(key)
        if v:
            try:
                return _dt.fromisoformat(str(v).replace("Z", "+00:00"))
            except ValueError:
                continue
    return None


# ── Asset URL extraction ──────────────────────────────────────────────────

def _download_href(asset: Any) -> str:
    """Return a fetchable URL for an asset.

    CDSE STAC gives measurement assets an ``s3://eodata/...`` primary href
    (which urllib can't fetch — "unknown url type: s3", 2026-08-04) plus the
    real HTTPS mirror in ``extra_fields["alternate"]["https"]["href"]`` (the
    zipper OData Nodes() path). Prefer that HTTPS alternate when the primary
    is s3://; otherwise return href unchanged (MPC/AWS give https/blob URLs).
    """
    href = asset.href
    if href.lower().startswith("s3://"):
        alt = (getattr(asset, "extra_fields", {}) or {}).get("alternate", {})
        https = (alt.get("https") or {}).get("href")
        if https:
            return https
    return href


def pick_measurement_urls(item: Any) -> tuple[str, str]:
    """Extract VV and VH measurement COG URLs from item.assets.

    Different providers name the assets differently (``vv`` / ``measurement-vv``
    / ``vv-grd`` / etc.) — match on either the asset key or the href path
    containing the polarisation token. The returned URL is the fetchable one
    (HTTPS alternate on CDSE; see ``_download_href``).
    """
    vv_url = vh_url = None
    for name, asset in (item.assets or {}).items():
        href = asset.href
        lname = name.lower()
        lhref = href.lower()
        # Reject calibration / noise / annotation assets — only keep the
        # raw measurement TIFFs.
        if "calibration" in lname or "calibration" in lhref:
            continue
        if "noise" in lname or "noise" in lhref:
            continue
        if "annotation" in lname or "annotation" in lhref:
            continue
        # Require a measurement signal — either explicit "measurement" tag
        # or a .tiff/.tif extension on a polarised asset.
        if (
            "measurement" not in lname
            and "measurement" not in lhref
            and not (lhref.endswith(".tiff") or lhref.endswith(".tif"))
        ):
            continue
        if "vv" in lname or "-vv-" in lhref or "_vv_" in lhref or "vv.tiff" in lhref:
            vv_url = _download_href(asset)
        elif "vh" in lname or "-vh-" in lhref or "_vh_" in lhref or "vh.tiff" in lhref:
            vh_url = _download_href(asset)
    if vv_url is None or vh_url is None:
        raise RuntimeError(
            f"Item {item.id} missing VV or VH measurement asset "
            f"(assets: {list((item.assets or {}).keys())})"
        )
    return vv_url, vh_url


def pick_calibration_urls(item: Any) -> tuple[str, str]:
    """Extract VV and VH calibration XML URLs from item.assets."""
    vv_url = vh_url = None
    for name, asset in (item.assets or {}).items():
        href = asset.href
        lname = name.lower()
        lhref = href.lower()
        if "calibration" not in lname and "calibration" not in lhref:
            continue
        if "noise" in lname or "noise" in lhref:
            continue
        if "vv" in lname or "-vv-" in lhref or "_vv_" in lhref or "vv.xml" in lhref:
            vv_url = _download_href(asset)
        elif "vh" in lname or "-vh-" in lhref or "_vh_" in lhref or "vh.xml" in lhref:
            vh_url = _download_href(asset)
    if vv_url is None or vh_url is None:
        raise RuntimeError(
            f"Item {item.id} missing VV or VH calibration asset "
            f"(assets: {list((item.assets or {}).keys())})"
        )
    return vv_url, vh_url


# ── Window read + calibration ─────────────────────────────────────────────

# Message raised for a GCP-referenced source (shared so callers/tests can match it).
GCP_SIGMA0_RETIRED = (
    "σ⁰ from GCP-referenced GRD is retired: its tie points sit at ellipsoid "
    "height (sea expected 41–56 m toward the satellite) and the calibration "
    "lookup is misaligned; use imint.training.pc_s1_rtc (RTC) instead "
    "(ImintEngine task 0113)"
)


def read_window(
    cog_path_or_uri: str | Path,
    west: float, south: float, east: float, north: float,
    src_epsg: int,
    h_px: int, w_px: int,
    resampling: Any | None = None,
) -> tuple[np.ndarray, Any]:
    """Read an H×W window from a projected COG at the requested bbox.

    Accepts both local Paths and remote URIs (s3://, https://, /vsicurl/…)
    — rasterio's GDAL bindings handle the protocol selection.

    Only sources with a projected ``ds.crs`` (e.g. the RTC COGs of
    ``imint.training.pc_s1_rtc``) are read. A GCP-referenced GRD raises
    ``RuntimeError`` (:data:`GCP_SIGMA0_RETIRED`): warping it over its own
    ellipsoid-height tie points displaced the sea toward the satellite.

    Returns:
        (dn_array, window) — ``dn_array`` is ``(H, W)`` float32 raw values
        (no calibration applied), ``window`` is the rasterio ``Window``
        used to read it.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds as window_from_bounds

    if resampling is None:
        resampling = Resampling.bilinear

    with rasterio.open(str(cog_path_or_uri)) as ds:
        if ds.crs is None:
            _, gcp_crs = ds.gcps
            if gcp_crs is not None:
                raise RuntimeError(f"{cog_path_or_uri}: {GCP_SIGMA0_RETIRED}")
            raise RuntimeError(
                f"{cog_path_or_uri}: no projected CRS and no GCP CRS "
                "— cannot georeference the window read")
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


# Calibration LUT cache — shared across all backends, keyed by the canonical
# product id. Values are dicts ``{"vv": grid, "vh": grid}``.
_lut_cache: dict[str, dict[str, dict[str, np.ndarray]]] = {}
import threading as _threading
_lut_cache_guard = _threading.Lock()


def load_cal_grid(
    cal_xml_source: str | Path | bytes,
    product_id: str,
    pol: str,
) -> dict[str, np.ndarray]:
    """Parse and cache the sigmaNought LUT grid from a calibration XML.

    Accepts a path, an HTTP/S3 URL (read into memory first by the caller
    and passed as bytes), or raw bytes.
    """
    with _lut_cache_guard:
        per_prod = _lut_cache.get(product_id)
        if per_prod is not None and pol in per_prod:
            return per_prod[pol]

    if isinstance(cal_xml_source, bytes):
        root = ET.fromstring(cal_xml_source)
    else:
        tree = ET.parse(cal_xml_source)
        root = tree.getroot()

    vectors = list(root.iter("calibrationVector"))
    if not vectors:
        raise RuntimeError(f"{cal_xml_source}: no <calibrationVector> elements")

    first_pixels = _parse_int_array(vectors[0].find("pixel").text)
    n_pix = len(first_pixels)

    lines = np.empty(len(vectors), dtype=np.float64)
    values = np.empty((len(vectors), n_pix), dtype=np.float64)

    for i, v in enumerate(vectors):
        line_el = v.find("line")
        pix_el = v.find("pixel")
        sig_el = v.find("sigmaNought")
        if line_el is None or pix_el is None or sig_el is None:
            raise RuntimeError(f"{cal_xml_source}: malformed calibrationVector {i}")
        pix_arr = _parse_int_array(pix_el.text)
        sig_arr = _parse_float_array(sig_el.text)
        if len(pix_arr) != n_pix or len(sig_arr) != n_pix:
            raise RuntimeError(
                f"{cal_xml_source}: calibrationVector {i} width mismatch "
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


def compute_sigma0(
    vv_dn: np.ndarray, vh_dn: np.ndarray,
    vv_lut: np.ndarray, vh_lut: np.ndarray,
    *,
    output_db: bool = False,
) -> np.ndarray:
    """Assemble σ⁰ = DN² / LUT² per polarisation; return ``(2, H, W)``.

    DN==0 pixels are preserved as 0 (genuine nodata). With ``output_db``,
    apply ``10·log10`` conversion.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        vv_sigma0 = (vv_dn.astype(np.float32) ** 2) / (vv_lut ** 2)
        vh_sigma0 = (vh_dn.astype(np.float32) ** 2) / (vh_lut ** 2)

    vv_sigma0 = np.where(vv_dn == 0, 0.0, vv_sigma0).astype(np.float32)
    vh_sigma0 = np.where(vh_dn == 0, 0.0, vh_sigma0).astype(np.float32)

    if output_db:
        vv_sigma0 = to_db(vv_sigma0)
        vh_sigma0 = to_db(vh_sigma0)

    return np.stack([vv_sigma0, vh_sigma0], axis=0)


def to_db(x: np.ndarray) -> np.ndarray:
    """Convert linear σ⁰ to dB, replacing non-finite with 0 (nodata)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 10.0 * np.log10(np.maximum(x, 1e-10))
    out = np.where(np.isfinite(out) & (x > 0), out, 0.0)
    return out.astype(np.float32)


def _parse_int_array(text: str) -> list[int]:
    return [int(x) for x in text.split()]


def _parse_float_array(text: str) -> list[float]:
    return [float(x) for x in text.split()]


# ── Bbox/size sanity check ────────────────────────────────────────────────

def assert_bbox_size_match(
    west: float, south: float, east: float, north: float,
    h_px: int, w_px: int,
    *,
    expected_gsd_m: float = 10.0,
) -> None:
    """Reject a fetch request whose bbox extent is inconsistent with size_px.

    A 256 px tile at 10 m GSD must be a 2560 m extent — otherwise the
    window read silently resamples to the wrong GSD.
    """
    expected_w_m = w_px * expected_gsd_m
    expected_h_m = h_px * expected_gsd_m
    if abs((east - west) - expected_w_m) > 1 or abs((north - south) - expected_h_m) > 1:
        raise ValueError(
            f"S1 fetch bbox/size_px mismatch. "
            f"bbox ew={east - west}m ns={north - south}m size=({h_px},{w_px}) "
            f"@ gsd={expected_gsd_m}m → expected {expected_w_m}×{expected_h_m}m extent."
        )
