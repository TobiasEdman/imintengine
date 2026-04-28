"""Sentinel-1 GRD fetching via Microsoft Planetary Computer STAC.

Free, anonymous fallback for the CDSE STAC backend — uses Microsoft's
publicly-signed Azure URLs which carry no PU billing and no requester-pays.

Layout
------
1. STAC search at ``https://planetarycomputer.microsoft.com/api/stac/v1``
   for collection ``sentinel-1-grd``. Search is anonymous.
2. Each item's asset hrefs are unsigned Azure blob URLs that 401 on direct
   GET. ``planetary_computer.sign()`` upgrades them with a SAS token good
   for ~1 hour.
3. VV/VH measurement TIFFs are read via ``/vsicurl/<signed_url>`` so
   rasterio streams the COG window without staging the full 2 GB to disk.
4. Calibration XMLs are tiny (~1 MB) — fetched once into memory via HTTPS
   and parsed with ``s1_shared.load_cal_grid``.

Drop-in replacement for :func:`cdse_s1_stac.fetch_s1_scene`. Same return
contract — ``(sar, orbit_direction)`` or ``None``.
"""
from __future__ import annotations

import urllib.request
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from . import s1_shared

# ── Module constants ─────────────────────────────────────────────────────

_STAC_ROOT = "https://planetarycomputer.microsoft.com/api/stac/v1"
_STAC_COLLECTION = "sentinel-1-grd"

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
    """Fetch a Sentinel-1 GRD σ⁰ window via Planetary Computer STAC.

    See module docstring for the why. Args / return identical to
    :func:`cdse_s1_stac.fetch_s1_scene`.
    """
    try:
        from pystac_client import Client
    except ImportError as e:
        raise ImportError(
            "mpc_s1 requires pystac-client. Install: pip install pystac-client"
        ) from e

    try:
        import planetary_computer as pc
    except ImportError as e:
        raise ImportError(
            "mpc_s1 requires planetary-computer. "
            "Install: pip install planetary-computer"
        ) from e

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
        print(f"    [MPC S1] {date}: search failed: {e}")
        return None

    items = s1_shared.filter_iw_grdh(items, orbit_direction)
    if not items:
        return None

    item = s1_shared.pick_best_item(items, bbox_4326)
    product_id = item.id

    # ── 2. Sign asset URLs ───────────────────────────────────────────────
    try:
        signed = pc.sign(item)
    except Exception as e:
        print(f"    [MPC S1] {product_id}: signing failed: {e}")
        return None

    try:
        vv_url, vh_url = s1_shared.pick_measurement_urls(signed)
        vv_cal_url, vh_cal_url = s1_shared.pick_calibration_urls(signed)
    except RuntimeError as e:
        print(f"    [MPC S1] {product_id}: asset selection failed: {e}")
        return None

    # ── 3. Window read via /vsicurl/ — streams from Azure ────────────────
    try:
        vv_dn, vv_win = s1_shared.read_window(
            f"/vsicurl/{vv_url}", west, south, east, north, epsg, h_px, w_px,
        )
        vh_dn, vh_win = s1_shared.read_window(
            f"/vsicurl/{vh_url}", west, south, east, north, epsg, h_px, w_px,
        )
    except Exception as e:
        print(f"    [MPC S1] {product_id}: window read failed: {e}")
        return None

    # ── 4. Calibration — fetch XMLs into memory ──────────────────────────
    try:
        vv_cal_bytes = _http_get(vv_cal_url)
        vh_cal_bytes = _http_get(vh_cal_url)
    except Exception as e:
        print(f"    [MPC S1] {product_id}: calibration download failed: {e}")
        return None

    try:
        vv_lut = s1_shared.interp_lut(
            vv_cal_bytes, vv_win, product_id, "vv", h_px, w_px,
        )
        vh_lut = s1_shared.interp_lut(
            vh_cal_bytes, vh_win, product_id, "vh", h_px, w_px,
        )
    except Exception as e:
        print(f"    [MPC S1] {product_id}: calibration parse failed: {e}")
        return None

    sar = s1_shared.compute_sigma0(vv_dn, vh_dn, vv_lut, vh_lut, output_db=output_db)

    if nodata_threshold is not None:
        if float((sar[0] == 0).mean()) > nodata_threshold:
            return None

    orbit = s1_shared.orbit_from_item(item) or (orbit_direction or "UNKNOWN")
    return sar, orbit


# ── Helpers ──────────────────────────────────────────────────────────────

def _http_get(url: str) -> bytes:
    """Tiny synchronous HTTPS GET — used for calibration XML fetch."""
    with urllib.request.urlopen(url, timeout=_REQUEST_TIMEOUT_S) as resp:
        return resp.read()


# ── Smoke test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    west, south, east, north = 380000, 6170000, 382560, 6172560
    date = "2023-06-15"
    print(f"[smoke] fetching S1 from MPC for {date} bbox=({west},{south},{east},{north})")
    result = fetch_s1_scene(
        west, south, east, north, date,
        size_px=256,
        orbit_direction=None,
        output_db=False,
        nodata_threshold=None,
    )
    if result is None:
        print("[smoke] no scene returned")
        raise SystemExit(1)
    sar, orbit = result
    vv, vh = sar[0], sar[1]
    print(f"[smoke] shape={sar.shape} dtype={sar.dtype} orbit={orbit}")
    print(f"[smoke] VV  min={vv.min():.4g} max={vv.max():.4g} mean={vv.mean():.4g}")
    print(f"[smoke] VH  min={vh.min():.4g} max={vh.max():.4g} mean={vh.mean():.4g}")
