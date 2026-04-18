"""Sentinel-1 SAR data fetching via CDSE Sentinel Hub Process API.

Fetches Sentinel-1 GRD (Ground Range Detected) IW (Interferometric Wide)
data for VV and VH polarizations. Uses the same CDSE Sentinel Hub HTTP
approach as cdse_s2.py — single HTTP POST per tile.

Data products:
    - VV: co-polarization backscatter (dB or linear)
    - VH: cross-polarization backscatter (dB or linear)
    - VV/VH ratio: useful for land cover discrimination

Output format:
    (2, H, W) float32 — VV and VH in linear backscatter coefficient (σ⁰)
    Values typically range 0.0–1.0 (linear) or -25 to 0 dB.

Usage:
    from imint.training.cdse_s1 import fetch_s1_scene

    result = fetch_s1_scene(west, south, east, north, date="2022-06-15")
    if result is not None:
        sar, orbit_dir = result
        # sar.shape == (2, 256, 256), dtype float32 (linear σ⁰)
        # orbit_dir == "ASCENDING" or "DESCENDING"
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime

import numpy as np

from .cdse_vpp import _get_token, _token_lock, _SH_PROCESS_URL

# ── Constants ────────────────────────────────────────────────────────────

_S1_BANDS = ["VV", "VH"]
_REQUEST_TIMEOUT_S = 60
_MAX_RETRIES = 3
_RETRY_DELAY_S = 2.0


# ── Public API ───────────────────────────────────────────────────────────

_CRS_3006 = "http://www.opengis.net/def/crs/EPSG/0/3006"
_CRS_4326 = "http://www.opengis.net/def/crs/EPSG/0/4326"


def fetch_s1_scene(
    west: float,
    south: float,
    east: float,
    north: float,
    date: str,
    *,
    crs: str = _CRS_3006,
    size_px: int | tuple[int, int] = 256,
    orbit_direction: str | None = None,
    output_db: bool = False,
    nodata_threshold: float | None = 0.10,
) -> tuple[np.ndarray, str] | None:
    """Fetch a single Sentinel-1 GRD scene via Sentinel Hub Process API.

    Args:
        west, south, east, north: Bounding box coordinates.
        date: ISO date string.
        crs: Coordinate reference system URI. Default EPSG:3006.
        size_px: Output size — int for square or (H, W) tuple.
        orbit_direction: "ASCENDING" or "DESCENDING" (None = any).
        output_db: If True, output in dB scale. Default False (linear σ⁰).
        nodata_threshold: Max nodata fraction (0–1). None = skip check.

    Returns:
        (sar, orbit_direction) on success, None on rejection.
            sar: (2, H, W) float32 — VV, VH
    """
    h_px, w_px = (size_px, size_px) if isinstance(size_px, int) else size_px

    # Defense: guarantee bbox / size consistency — same pattern as fetch_s2_scene.
    # S1 is acquired in linear SAR backscatter but Sentinel Hub still computes
    # GSD as (east-west)/width_px when rendering. Mismatched bbox silently
    # produces an up/downsampled raster with misregistered pixels.
    expected_m = w_px * 10
    if abs((east - west) - expected_m) > 1 or abs((north - south) - expected_m) > 1:
        raise ValueError(
            f"fetch_s1_scene: bbox/size_px mismatch. "
            f"bbox ew={east - west}m ns={north - south}m size_px={w_px} "
            f"→ expected {expected_m}m extent."
        )

    token = _get_token()

    try:
        tiff_bytes = _fetch_s1_tiff(
            west, south, east, north, w_px, h_px,
            date=date, token=token,
            orbit_direction=orbit_direction, output_db=output_db, crs=crs,
        )
    except Exception as e:
        print(f"    [SH S1] {date}: {e}")
        return None

    bands = _parse_multiband_tiff(tiff_bytes, h_px, w_px, 2)
    if bands is None or len(bands) < 2:
        return None

    sar = np.stack(bands, axis=0)  # (2, H, W) float32

    if nodata_threshold is not None:
        if float((sar[0] == 0).mean()) > nodata_threshold:
            return None

    return sar, (orbit_direction or "UNKNOWN")


def fetch_s1_scene_wgs84(
    west: float, south: float, east: float, north: float, date: str, **kwargs,
) -> tuple[np.ndarray, str] | None:
    """Convenience wrapper: fetch_s1_scene with WGS84 defaults.

    Defaults: crs=EPSG:4326, nodata=None (disabled).
    """
    kwargs.setdefault("crs", _CRS_4326)
    kwargs.setdefault("nodata_threshold", None)
    return fetch_s1_scene(west, south, east, north, date, **kwargs)


def fetch_s1_pair(
    west: float,
    south: float,
    east: float,
    north: float,
    date_1: str,
    date_2: str,
    *,
    size_px: int = 256,
    orbit_direction: str = "DESCENDING",
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fetch a pair of Sentinel-1 scenes for InSAR differential analysis.

    Args:
        date_1, date_2: ISO date strings for the two acquisitions.
        orbit_direction: Must be same for both (default: DESCENDING).

    Returns:
        Tuple of (sar_1, sar_2), each (2, H, W) float32, or None.
    """
    result_1 = fetch_s1_scene(
        west, south, east, north, date_1,
        size_px=size_px, orbit_direction=orbit_direction,
    )
    if result_1 is None:
        return None

    result_2 = fetch_s1_scene(
        west, south, east, north, date_2,
        size_px=size_px, orbit_direction=orbit_direction,
    )
    if result_2 is None:
        return None

    return result_1[0], result_2[0]


# ── Internal helpers ─────────────────────────────────────────────────────

def _build_s1_evalscript(output_db: bool = False) -> str:
    """Build Sentinel Hub evalscript for 2-band S1 GRD fetch.

    Returns VV and VH as either linear σ⁰ or dB backscatter.
    """
    if output_db:
        conversion = """
    // Linear to dB: 10 * log10(σ⁰)
    var vv_db = 10 * Math.log10(Math.max(sample.VV, 1e-10));
    var vh_db = 10 * Math.log10(Math.max(sample.VH, 1e-10));
    return [vv_db, vh_db];"""
    else:
        conversion = """
    return [sample.VV, sample.VH];"""

    return f"""//VERSION=3
function setup() {{
  return {{
    input: [{{
      bands: ["VV", "VH"],
      units: "LINEAR_POWER"
    }}],
    output: {{
      bands: 2,
      sampleType: "FLOAT32"
    }}
  }};
}}

function evaluatePixel(sample) {{{conversion}
}}
"""


def _fetch_s1_tiff(
    west: float, south: float,
    east: float, north: float,
    width_px: int, height_px: int,
    *,
    date: str,
    token: str,
    orbit_direction: str | None = None,
    output_db: bool = False,
    crs: str = "http://www.opengis.net/def/crs/EPSG/0/3006",
) -> bytes:
    """Fetch S1 GRD data as multi-band GeoTIFF from Sentinel Hub Process API."""
    import time

    time_from = f"{date}T00:00:00Z"
    time_to = f"{date}T23:59:59Z"

    data_filter = {
        "timeRange": {
            "from": time_from,
            "to": time_to,
        },
        "mosaickingOrder": "mostRecent",
    }
    if orbit_direction:
        data_filter["acquisitionMode"] = "IW"
        data_filter["polarization"] = "DV"
        data_filter["orbitDirection"] = orbit_direction

    request_body = {
        "input": {
            "bounds": {
                "bbox": [west, south, east, north],
                "properties": {
                    "crs": crs
                },
            },
            "data": [{
                "type": "sentinel-1-grd",
                "dataFilter": data_filter,
                "processing": {
                    "backCoeff": "SIGMA0_ELLIPSOID",
                    "orthorectify": True,
                    "demInstance": "COPERNICUS_30",
                },
            }],
        },
        "output": {
            "width": width_px,
            "height": height_px,
            "responses": [{
                "identifier": "default",
                "format": {
                    "type": "image/tiff",
                },
            }],
        },
        "evalscript": _build_s1_evalscript(output_db),
    }

    body_bytes = json.dumps(request_body).encode()

    for attempt in range(_MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                _SH_PROCESS_URL,
                data=body_bytes,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                    "Accept": "image/tiff",
                },
            )
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
                return resp.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY_S * (attempt + 1))
                continue
            raise

    raise RuntimeError("Unreachable — all retries exhausted")


def _parse_multiband_tiff(
    tiff_bytes: bytes,
    h_px: int,
    w_px: int,
    n_bands: int,
) -> list[np.ndarray] | None:
    """Parse multi-band TIFF bytes into list of numpy arrays."""
    import io

    try:
        import rasterio
        with rasterio.open(io.BytesIO(tiff_bytes)) as src:
            bands = []
            for i in range(1, n_bands + 1):
                bands.append(src.read(i).astype(np.float32))
            return bands
    except Exception:
        return None
