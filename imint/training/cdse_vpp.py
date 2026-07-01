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

Source routing ($VPP_SOURCE):
    CDSE Sentinel Hub and the WEkEO COG cache serve the SAME HR-VPP
    product, so routing is by cost/coverage/availability, not quality.
      * ``auto`` (default): cache-first + circuit breaker.
          1. WEkEO cache (imint.training.wekeo_vpp, $VPP_WEKEO_DIR, default
             /data/vpp_wekeo) when it covers the tile — free, local.
          2. CDSE for coverage gaps, but only while the shared SH-Process
             PU pool isn't marked dead (imint.training.openeo_tile_graph
             credit guard, key "cdse"); first PU exhaustion trips it so no
             later tile wastes a doomed call.
          3. WEkEO best-effort otherwise.
      * ``cdse``: force the metered Process API.
      * ``wekeo``: force the cache (skip CDSE) — a blunt override; rarely
        needed since ``auto`` already prefers the cache when it has data.

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

    # Source selection (see module docstring). VPP_SOURCE:
    #   wekeo → force the local COG cache (skip CDSE entirely)
    #   cdse  → force the metered SH Process API
    #   auto/unset (default) → cache-first + circuit breaker (best practice):
    #     CDSE and WEkEO serve the SAME HR-VPP product, so route by cost,
    #     not quality — prefer the free local cache when it covers the tile,
    #     hit the metered CDSE only for coverage gaps, and skip CDSE entirely
    #     once the shared SH-Process PU pool is marked dead this session.
    src = os.environ.get("VPP_SOURCE", "").strip().lower()
    if src == "wekeo":
        result = _read_wekeo_vpp(west, south, east, north, (h_px, w_px), year)
        # A partial-coverage read returns an all-zero array instead of None
        # (WEkEO reader fills nodata with 0). Without _has_sufficient_coverage
        # the wekeo-forced path silently accepts that as a hit and writes
        # all-zero VPP into the tile — 130/390 orphan-512 tiles landed with
        # all-zero vpp_{sosd,eosd,length,maxv,minv} 2026-07-01 (concentrated
        # on dates[0] years 2019 + 2020, the years the WEkEO cache had 3/44
        # and 0/44 MGRS respectively). Same >=5% non-zero-sosd floor the
        # auto-router already applies. Miss → raise loud, don't zero-fill.
        if result is None or not _has_sufficient_coverage(result):
            raise RuntimeError(
                f"VPP_SOURCE=wekeo but no covering WEkEO cache at "
                f"{os.environ.get('VPP_WEKEO_DIR', '/data/vpp_wekeo')} "
                f"for year={year} bbox=({west:.0f},{south:.0f},"
                f"{east:.0f},{north:.0f})"
            )
    elif src == "cdse":
        result = _fetch_cdse_vpp(west, south, east, north, h_px, w_px, year)
    else:
        result = _auto_fetch_vpp(west, south, east, north, h_px, w_px, year)

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


# Shared SH-Process credit-guard key. CDSE VPP (SH Process API) draws from
# the SAME PU pool as the CDSE spectral path (imint.training.cdse_s2 marks
# "cdse"), so one exhaustion must trip the guard for both.
_PU_POOL = "cdse"


# ── Internal helpers ─────────────────────────────────────────────────────

def _fetch_cdse_vpp(
    west: float, south: float, east: float, north: float,
    h_px: int, w_px: int, year: int,
) -> dict[str, np.ndarray]:
    """Fetch VPP from the metered CDSE Sentinel Hub Process API.

    Raises RuntimeError on failure (quota/outage/parse).
    """
    lon_min, lat_min, lon_max, lat_max = _bbox_3006_to_4326(
        west, south, east, north
    )
    token = _get_token()
    tiff_bytes = _fetch_vpp_tiff(
        lon_min, lat_min, lon_max, lat_max, w_px, h_px,
        token=token, year=year,
    )
    bands = _parse_multiband_tiff(tiff_bytes, h_px, w_px, len(_VPP_BANDS))
    result: dict[str, np.ndarray] = {}
    for i, band_name in enumerate(_VPP_BANDS):
        arr = bands[i].astype(np.float32)
        if band_name in _PPI_BANDS:
            arr = np.clip(arr * 0.0001, 0.0, None)  # INT16 scale + clamp nodata
        if band_name in _DOY_BANDS:
            arr = np.clip(arr, 0.0, None)
        result[band_name.lower()] = arr
    return result


def _read_wekeo_vpp(
    west: float, south: float, east: float, north: float,
    size_px: tuple[int, int], year: int,
) -> dict[str, np.ndarray] | None:
    """Read VPP from the prefetched WEkEO COG cache.

    Returns the band dict, or ``None`` if no cache is present (no
    ``index.json``) — callers decide whether that's a hard error or a
    cue to try CDSE. The COGs are populated by scripts/prefetch_vpp_wekeo.py.
    """
    cog_dir = Path(os.environ.get("VPP_WEKEO_DIR", "/data/vpp_wekeo"))
    if not (cog_dir / "index.json").exists():
        return None
    from imint.training.wekeo_vpp import fetch_vpp_tiles_local
    return fetch_vpp_tiles_local(
        west, south, east, north,
        size_px=size_px, vpp_cog_dir=cog_dir, year=year,
    )


def _has_sufficient_coverage(result: dict[str, np.ndarray] | None) -> bool:
    """True if the WEkEO read actually covers the tile.

    WEkEO returns all-zeros for areas outside the cached COGs' footprint
    (no exception), so "got a dict" != "covered". Require the SOSD band to
    have >5% valid (non-zero) pixels — the same floor compute_growing_season
    uses, so a "covered" result is one that will actually yield a window.
    """
    if not result or "sosd" not in result:
        return False
    sosd = np.asarray(result["sosd"])
    if sosd.size == 0:
        return False
    return float(np.count_nonzero(sosd)) >= max(10, sosd.size * 0.05)


def _auto_fetch_vpp(
    west: float, south: float, east: float, north: float,
    h_px: int, w_px: int, year: int,
) -> dict[str, np.ndarray]:
    """Cache-first + circuit-breaker VPP routing (the default).

    1. WEkEO cache — if it covers the tile, use it (free, local, equivalent).
    2. Coverage gap — CDSE, but only if the shared SH-Process PU pool isn't
       already marked dead this session; on PU exhaustion, mark it dead so
       no later tile wastes a doomed call.
    3. Last resort — return the (partial/empty) WEkEO read so windowing
       degrades to fallback windows rather than aborting the tile; raise
       only if there is no WEkEO cache at all and CDSE is unavailable.
    """
    from imint.training.openeo_tile_graph import (
        is_source_dead, mark_source_dead, _is_payment_required_error,
    )

    wk = _read_wekeo_vpp(west, south, east, north, (h_px, w_px), year)
    if _has_sufficient_coverage(wk):
        return wk  # cache-first: free + covers the tile

    if not is_source_dead(_PU_POOL):
        try:
            return _fetch_cdse_vpp(west, south, east, north, h_px, w_px, year)
        except RuntimeError as exc:
            if _is_payment_required_error(exc):
                mark_source_dead(
                    _PU_POOL, f"VPP SH-Process PU exhausted: {str(exc)[:160]}")
            else:
                print(f"    [cdse_vpp] CDSE error, using WEkEO best-effort: "
                      f"{str(exc)[:160]}", flush=True)

    if wk is not None:
        return wk  # best-effort (may be partial → fallback windows)
    raise RuntimeError(
        "VPP unavailable: CDSE PU exhausted/failed and no WEkEO cache at "
        f"{os.environ.get('VPP_WEKEO_DIR', '/data/vpp_wekeo')}"
    )


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
