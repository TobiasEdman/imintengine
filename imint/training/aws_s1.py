"""Sentinel-1 GRD fetching via Element84 Earth-Search STAC + AWS S3.

Last-resort fallback in the S1 fetch chain. Uses Element84's free
Earth-Search STAC API as the catalogue and the underlying ``s3://sentinel-s1-l1c``
bucket for asset reads.

⚠ Requester-pays note
---------------------
``s3://sentinel-s1-l1c`` is operated by ESA on AWS as a **requester-pays**
bucket. The consumer (us) pays the egress (~$0.09/GB out of eu-central-1).
A typical 4-frame fetch over Sweden touches on the order of 100–300 unique
2 GB GRD products, so a full backfill against this backend is ~$20–50.

This module sets ``AWS_REQUEST_PAYER=requester`` automatically so reads
succeed; the caller must have AWS credentials configured (env vars or
``~/.aws/credentials``). Anonymous access is **not** supported by the
upstream bucket. If credentials are absent, the module raises a clear
``RuntimeError`` rather than silently failing.

Layout
------
1. STAC search at ``https://earth-search.aws.element84.com/v1`` for
   collection ``sentinel-1-grd``.
2. Each item's ``vv`` / ``vh`` assets carry public S3 hrefs that can be
   read directly by rasterio's ``/vsis3/`` driver (with requester-pays
   header).
3. Calibration XMLs are HTTPS-public on the same bucket — fetched once
   into memory.
"""
from __future__ import annotations

import os
import urllib.request
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from . import s1_shared

# ── Module constants ─────────────────────────────────────────────────────

_STAC_ROOT = "https://earth-search.aws.element84.com/v1"
_STAC_COLLECTION = "sentinel-1-grd"
_S3_REGION = "eu-central-1"

_REQUEST_TIMEOUT_S = 60


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
    """Fetch a Sentinel-1 GRD σ⁰ window via Element84 STAC + AWS S3.

    See module docstring for the requester-pays implications. Args /
    return identical to :func:`cdse_s1_stac.fetch_s1_scene`.

    Raises:
        RuntimeError: When AWS credentials are not configured. The
            requester-pays bucket needs them; we surface this immediately
            rather than letting rasterio fail with a misleading 403.
    """
    try:
        from pystac_client import Client
    except ImportError as e:
        raise ImportError(
            "aws_s1 requires pystac-client. Install: pip install pystac-client"
        ) from e

    _check_aws_credentials()
    _set_gdal_aws_env()

    h_px, w_px = (size_px, size_px) if isinstance(size_px, int) else size_px

    s1_shared.assert_bbox_size_match(west, south, east, north, h_px, w_px)
    epsg = s1_shared.crs_uri_to_epsg(crs)

    from rasterio.warp import transform_bounds
    bbox_4326 = transform_bounds(
        f"EPSG:{epsg}", "EPSG:4326",
        west, south, east, north,
        densify_pts=21,
    )

    # ── 1. STAC search ───────────────────────────────────────────────────
    dt = datetime.strptime(date, "%Y-%m-%d")
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
        print(f"    [AWS S1] {date}: search failed: {e}")
        return None

    items = s1_shared.filter_iw_grdh(items, orbit_direction)
    if not items:
        return None

    item = s1_shared.pick_best_item(items, bbox_4326)
    product_id = item.id

    # ── 2. Asset selection ───────────────────────────────────────────────
    try:
        vv_url, vh_url = s1_shared.pick_measurement_urls(item)
        vv_cal_url, vh_cal_url = s1_shared.pick_calibration_urls(item)
    except RuntimeError as e:
        print(f"    [AWS S1] {product_id}: asset selection failed: {e}")
        return None

    # ── 3. Window read via /vsis3/ ───────────────────────────────────────
    try:
        vv_dn, vv_win = s1_shared.read_window(
            _to_vsi(vv_url), west, south, east, north, epsg, h_px, w_px,
        )
        vh_dn, vh_win = s1_shared.read_window(
            _to_vsi(vh_url), west, south, east, north, epsg, h_px, w_px,
        )
    except Exception as e:
        print(f"    [AWS S1] {product_id}: window read failed: {e}")
        return None

    # ── 4. Calibration ───────────────────────────────────────────────────
    try:
        vv_cal_bytes = _http_get(_to_https(vv_cal_url))
        vh_cal_bytes = _http_get(_to_https(vh_cal_url))
    except Exception as e:
        print(f"    [AWS S1] {product_id}: calibration download failed: {e}")
        return None

    try:
        vv_lut = s1_shared.interp_lut(
            vv_cal_bytes, vv_win, product_id, "vv", h_px, w_px,
        )
        vh_lut = s1_shared.interp_lut(
            vh_cal_bytes, vh_win, product_id, "vh", h_px, w_px,
        )
    except Exception as e:
        print(f"    [AWS S1] {product_id}: calibration parse failed: {e}")
        return None

    sar = s1_shared.compute_sigma0(vv_dn, vh_dn, vv_lut, vh_lut, output_db=output_db)

    if nodata_threshold is not None:
        if float((sar[0] == 0).mean()) > nodata_threshold:
            return None

    orbit = s1_shared.orbit_from_item(item) or (orbit_direction or "UNKNOWN")
    return sar, orbit


# ── Helpers ──────────────────────────────────────────────────────────────

def _check_aws_credentials() -> None:
    """Fail early if AWS credentials aren't configured.

    The bucket is requester-pays and rejects unsigned requests with a
    misleading 403 — surface the real problem to the caller.
    """
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        return
    home = os.environ.get("HOME", "")
    if home and os.path.exists(os.path.join(home, ".aws", "credentials")):
        return
    raise RuntimeError(
        "aws_s1 requires AWS credentials (s3://sentinel-s1-l1c is "
        "requester-pays). Set AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY "
        "or configure ~/.aws/credentials."
    )


def _set_gdal_aws_env() -> None:
    """Configure GDAL for the requester-pays + region requirements."""
    os.environ.setdefault("AWS_REQUEST_PAYER", "requester")
    os.environ.setdefault("AWS_REGION", _S3_REGION)
    # GDAL needs the explicit region for /vsis3/ when the bucket isn't in us-east-1.
    os.environ.setdefault("AWS_DEFAULT_REGION", _S3_REGION)


def _to_vsi(s3_or_https_url: str) -> str:
    """Convert a presented asset URL to a rasterio-friendly /vsis3/ path."""
    if s3_or_https_url.startswith("s3://"):
        return f"/vsis3/{s3_or_https_url[5:]}"
    if s3_or_https_url.startswith("https://"):
        # Element84 sometimes presents HTTPS rather than s3://; let GDAL's
        # /vsicurl/ handle it (no requester-pays signal but the bucket may
        # accept anonymous reads on HTTPS for some endpoints).
        return f"/vsicurl/{s3_or_https_url}"
    return s3_or_https_url


def _to_https(s3_or_https_url: str) -> str:
    """Convert s3:// to a public HTTPS URL on the same bucket."""
    if s3_or_https_url.startswith("s3://"):
        bucket, _, key = s3_or_https_url[5:].partition("/")
        return f"https://{bucket}.s3.{_S3_REGION}.amazonaws.com/{key}"
    return s3_or_https_url


def _http_get(url: str) -> bytes:
    """Tiny synchronous HTTPS GET for calibration XMLs."""
    with urllib.request.urlopen(url, timeout=_REQUEST_TIMEOUT_S) as resp:
        return resp.read()
