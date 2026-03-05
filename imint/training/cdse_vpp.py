"""Copernicus HR-VPP vegetation phenology fetching.

Downloads High-Resolution Vegetation Phenology and Productivity (HR-VPP)
parameters from the Copernicus Data Space Ecosystem via the Sentinel Hub
Process API.  Returns per-band NumPy arrays aligned to the NMD 10 m grid
used by the training pipeline.

HR-VPP is derived from Sentinel-2 time series and provides 10 m resolution
phenology metrics per growing season.  We fetch Season 1 (primary growing
season) which covers the main vegetation cycle in Scandinavia.

Parameters fetched:
    SOSD  — Start Of Season Day (day of year)
    EOSD  — End Of Season Day (day of year)
    LENGTH — Season Length (days)
    MAXV  — Maximum Plant Phenology Index (PPI, unitless 0–2, stored as
             INT16 with scale 0.0001)
    MINV  — Minimum PPI value (same scale as MAXV)

Access:
    Sentinel Hub Process API  (CDSE)
    BYOC Collection (Season 1): 67c73156-095d-4f53-8a09-9ddf3848fbb6
    Endpoint: https://sh.dataspace.copernicus.eu/api/v1/process
    Auth: CDSE OAuth2 client_credentials grant

License: Copernicus Open Access

Typical usage::

    from imint.training.cdse_vpp import fetch_vpp_tiles
    vpp = fetch_vpp_tiles(west, south, east, north, size_px=256)
    # vpp["sosd"].shape == (256, 256), dtype float32

"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from threading import Lock

import numpy as np

# ── Sentinel Hub Process API config ──────────────────────────────────────
_SH_PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
_SH_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
    "/protocol/openid-connect/token"
)

# HR-VPP VPP Season 1 — Bring Your Own COG collection
_VPP_COLLECTION_ID = "67c73156-095d-4f53-8a09-9ddf3848fbb6"

# Bands to fetch — phenology metrics that discriminate vegetation types
_VPP_BANDS = ["SOSD", "EOSD", "LENGTH", "MAXV", "MINV"]

# INT16 bands that need scaling by 0.0001 to get PPI values
_PPI_BANDS = {"MAXV", "MINV"}

# Day-of-year bands — keep as integers (cast to float for consistency)
_DOY_BANDS = {"SOSD", "EOSD", "LENGTH"}

_REQUEST_TIMEOUT_S = 60
_MAX_RETRIES = 3
_RETRY_DELAY_S = 2.0

# ── Token cache (thread-safe) ──────────────────────────────────────────
_token_lock = Lock()
_cached_token: str | None = None
_token_expires: float = 0.0


# ── Public API ───────────────────────────────────────────────────────────

def fetch_vpp_tiles(
    west: float,
    south: float,
    east: float,
    north: float,
    *,
    size_px: int | tuple[int, int] = 256,
    cache_dir: Path | None = None,
    year: int = 2021,
) -> dict[str, np.ndarray]:
    """Fetch HR-VPP Season 1 phenology for a tile.

    Args:
        west, south, east, north: Bounding box in EPSG:3006 (meters).
        size_px: Output size — int for square or (H, W) tuple.
        cache_dir: Optional cache directory for .npy files.
        year: VPP product year.  Season 1 data is available from ~2017.
              Default 2021 gives a recent, complete dataset for Sweden.

    Returns:
        Dict mapping band names (lowercase) to (H, W) float32 arrays:
            sosd, eosd, length, maxv, minv
        PPI bands (maxv, minv) are scaled to real values (0–2).
        Day bands (sosd, eosd, length) are in days.
        NoData pixels are 0.
    """
    if isinstance(size_px, int):
        h_px, w_px = size_px, size_px
    else:
        h_px, w_px = size_px

    # Check cache
    if cache_dir is not None:
        cache_key = f"vpp_{int(west)}_{int(south)}_{int(east)}_{int(north)}.npz"
        cache_path = cache_dir / cache_key
        if cache_path.exists():
            try:
                cached = np.load(cache_path)
                # Verify all bands present and correct shape
                if all(
                    b.lower() in cached
                    and cached[b.lower()].shape == (h_px, w_px)
                    for b in _VPP_BANDS
                ):
                    return {b.lower(): cached[b.lower()] for b in _VPP_BANDS}
            except Exception:
                pass  # Re-fetch on cache corruption

    # Convert bbox from EPSG:3006 to WGS84
    lon_min, lat_min, lon_max, lat_max = _bbox_3006_to_4326(
        west, south, east, north
    )

    # Get auth token
    token = _get_token()

    # Fetch all VPP bands in a single request
    tiff_bytes = _fetch_vpp_tiff(
        lon_min, lat_min, lon_max, lat_max,
        w_px, h_px,
        token=token,
        year=year,
    )

    # Parse multi-band TIFF → per-band arrays
    bands = _parse_multiband_tiff(tiff_bytes, h_px, w_px, len(_VPP_BANDS))

    # Build result dict with proper scaling
    result: dict[str, np.ndarray] = {}
    for i, band_name in enumerate(_VPP_BANDS):
        arr = bands[i].astype(np.float32)

        # Scale PPI bands: INT16 with factor 0.0001
        if band_name in _PPI_BANDS:
            arr = arr * 0.0001
            arr = np.clip(arr, 0.0, None)  # Clamp negatives (nodata)

        # Day bands: keep as float, clamp nodata to 0
        if band_name in _DOY_BANDS:
            arr = np.clip(arr, 0.0, None)

        result[band_name.lower()] = arr

    # Cache
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache_dir / cache_key.replace(".npz", "_tmp.npz")
        np.savez_compressed(tmp, **result)
        Path(str(tmp)).rename(cache_path)

    return result


def fetch_vpp_band(
    west: float,
    south: float,
    east: float,
    north: float,
    *,
    band: str = "maxv",
    size_px: int | tuple[int, int] = 256,
    cache_dir: Path | None = None,
    year: int = 2021,
) -> np.ndarray:
    """Fetch a single VPP band.  Convenience wrapper around fetch_vpp_tiles.

    Returns (H, W) float32 array.
    """
    all_bands = fetch_vpp_tiles(
        west, south, east, north,
        size_px=size_px,
        cache_dir=cache_dir,
        year=year,
    )
    key = band.lower()
    if key not in all_bands:
        raise ValueError(
            f"Unknown VPP band '{band}'. "
            f"Available: {list(all_bands.keys())}"
        )
    return all_bands[key]


# ── Internal helpers ─────────────────────────────────────────────────────

def _bbox_3006_to_4326(
    west: float, south: float, east: float, north: float,
) -> tuple[float, float, float, float]:
    """Convert EPSG:3006 bbox to EPSG:4326 (lon_min, lat_min, lon_max, lat_max)."""
    from rasterio.warp import transform_bounds
    from rasterio.crs import CRS

    bounds = transform_bounds(
        CRS.from_epsg(3006), CRS.from_epsg(4326),
        west, south, east, north,
    )
    return bounds  # (lon_min, lat_min, lon_max, lat_max)


def _get_credentials() -> tuple[str, str]:
    """Get CDSE client_id and client_secret.

    Priority:
        1. CDSE_CLIENT_ID + CDSE_CLIENT_SECRET env vars
        2. .cdse_credentials file (lines 3+4)

    Returns:
        (client_id, client_secret)
    """
    # 1. Environment variables
    client_id = os.environ.get("CDSE_CLIENT_ID")
    client_secret = os.environ.get("CDSE_CLIENT_SECRET")
    if client_id and client_secret:
        return client_id, client_secret

    # 2. Credentials file
    cred_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        ))),
        ".cdse_credentials",
    )
    if os.path.isfile(cred_path):
        with open(cred_path) as f:
            lines = [line.strip() for line in f.readlines()]
        if len(lines) >= 4 and lines[2] and lines[3]:
            return lines[2], lines[3]

    raise RuntimeError(
        "CDSE credentials not found.\n"
        "Set CDSE_CLIENT_ID + CDSE_CLIENT_SECRET env vars, or add\n"
        "client_id on line 3 and client_secret on line 4 of .cdse_credentials"
    )


def _get_token() -> str:
    """Get a valid CDSE OAuth2 access token (cached, thread-safe)."""
    global _cached_token, _token_expires

    with _token_lock:
        # Return cached token if still valid (with 60s margin)
        if _cached_token and time.time() < _token_expires - 60:
            return _cached_token

    client_id, client_secret = _get_credentials()

    data = urllib.parse.urlencode({
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }).encode()

    req = urllib.request.Request(
        _SH_TOKEN_URL,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = urllib.request.urlopen(req, timeout=30)
            token_data = json.loads(resp.read())
            with _token_lock:
                _cached_token = token_data["access_token"]
                _token_expires = time.time() + token_data.get("expires_in", 300)
            return _cached_token
        except Exception as e:
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY_S * (attempt + 1))
                continue
            raise RuntimeError(f"CDSE token fetch failed: {e}") from e


def _build_evalscript() -> str:
    """Build Sentinel Hub evalscript for multi-band VPP fetch."""
    bands_input = ", ".join(f'"{b}"' for b in _VPP_BANDS)
    bands_output = ", ".join(
        f"sample.{b}" for b in _VPP_BANDS
    )
    return f"""//VERSION=3
function setup() {{
  return {{
    input: [{{
      type: "BYOC",
      bands: [{bands_input}]
    }}],
    output: {{
      bands: {len(_VPP_BANDS)},
      sampleType: "FLOAT32"
    }}
  }};
}}

function evaluatePixel(sample) {{
  return [{bands_output}];
}}
"""


def _fetch_vpp_tiff(
    lon_min: float, lat_min: float,
    lon_max: float, lat_max: float,
    width_px: int, height_px: int,
    *,
    token: str,
    year: int = 2021,
) -> bytes:
    """Fetch VPP data as multi-band GeoTIFF from Sentinel Hub Process API."""

    request_body = {
        "input": {
            "bounds": {
                "bbox": [lon_min, lat_min, lon_max, lat_max],
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/EPSG/0/4326"
                },
            },
            "data": [{
                "type": "byoc-" + _VPP_COLLECTION_ID,
                "dataFilter": {
                    "timeRange": {
                        "from": f"{year}-01-01T00:00:00Z",
                        "to": f"{year}-12-31T23:59:59Z",
                    },
                    "mosaickingOrder": "mostRecent",
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
        "evalscript": _build_evalscript(),
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
            resp = urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S)
            data = resp.read()

            # Check we got TIFF (not JSON error)
            if data[:4] in (b"II*\x00", b"MM\x00*"):
                return data

            # Try to parse as JSON error
            try:
                err = json.loads(data)
                msg = err.get("error", {}).get("message", str(err)[:300])
                raise RuntimeError(f"Sentinel Hub error: {msg}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise RuntimeError(
                    f"Unexpected response ({len(data)} bytes, "
                    f"starts with {data[:20]!r})"
                )

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:500]

            # Token expired — refresh and retry
            if e.code == 401:
                global _cached_token, _token_expires
                with _token_lock:
                    _cached_token = None
                    _token_expires = 0.0
                token = _get_token()
                continue

            # Rate limited
            if e.code == 429:
                retry_after = int(e.headers.get("Retry-After", 10))
                time.sleep(retry_after)
                continue

            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY_S * (attempt + 1))
                continue
            raise RuntimeError(
                f"Sentinel Hub Process API error (HTTP {e.code}): {body}"
            ) from e

        except Exception as e:
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY_S * (attempt + 1))
                continue
            raise RuntimeError(
                f"VPP fetch failed after {_MAX_RETRIES + 1} attempts: {e}"
            ) from e

    # Should not reach here, but just in case
    raise RuntimeError("VPP fetch failed: exhausted all retries")


def _parse_multiband_tiff(
    tiff_bytes: bytes,
    expected_h: int,
    expected_w: int,
    expected_bands: int,
) -> list[np.ndarray]:
    """Parse a multi-band GeoTIFF from bytes into per-band numpy arrays.

    Returns list of (H, W) float32 arrays, one per band.
    """
    try:
        import rasterio
        from rasterio.io import MemoryFile

        with MemoryFile(tiff_bytes) as memfile:
            with memfile.open() as ds:
                n_bands = min(ds.count, expected_bands)
                result = []
                for band_idx in range(1, n_bands + 1):
                    data = ds.read(band_idx)

                    # Handle nodata
                    nodata = ds.nodata
                    if nodata is not None:
                        data = np.where(data == nodata, 0, data)

                    # Resize if needed
                    if data.shape != (expected_h, expected_w):
                        from scipy.ndimage import zoom
                        zy = expected_h / data.shape[0]
                        zx = expected_w / data.shape[1]
                        data = zoom(data, (zy, zx), order=0)  # nearest for int

                    result.append(data.astype(np.float32))

                # Pad with zeros if fewer bands returned
                while len(result) < expected_bands:
                    result.append(
                        np.zeros((expected_h, expected_w), dtype=np.float32)
                    )

                return result

    except ImportError:
        raise ImportError(
            "rasterio is required for TIFF parsing. "
            "Install with: pip install rasterio"
        )
