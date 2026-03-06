"""Sentinel-2 L2A data fetching via CDSE Sentinel Hub Process API.

Single-stage fetch: downloads 6 spectral bands + SCL in one HTTP POST,
checks cloud fraction locally, and discards cloudy scenes.  This replaces
the two-stage openEO approach (SCL pre-screen → spectral fetch) with a
simpler, faster pipeline.

Typical per-tile workflow:
    1. STAC discovery (DES STAC, same as before)
    2. For each seasonal window, try top candidate dates:
       a. Fetch 7 bands (B02..B12 + SCL) via Sentinel Hub Process API
       b. Check SCL cloud fraction locally (same logic as openEO path)
       c. Check quality gates (nodata %, B02 haze)
       d. Accept or try next candidate
    3. Stack frames → (T*6, H, W) multitemporal tile

Access:
    Sentinel Hub Process API (CDSE)
    Collection: sentinel-2-l2a (built-in, no BYOC)
    Endpoint: https://sh.dataspace.copernicus.eu/api/v1/process
    Auth: CDSE OAuth2 client_credentials (same as VPP)

License: Copernicus Open Access

Typical usage::

    from imint.training.cdse_s2 import fetch_s2_scene
    result = fetch_s2_scene(west, south, east, north, date="2019-07-15")
    if result is not None:
        spectral, scl, cloud_frac = result
        # spectral.shape == (6, 256, 256), dtype float32, reflectance [0,1]
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from threading import Lock

import numpy as np

# Reuse CDSE token management from VPP module
from .cdse_vpp import _get_token, _token_lock, _SH_PROCESS_URL

# ── Constants ────────────────────────────────────────────────────────────

_PRITHVI_BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]
_ALL_BANDS = _PRITHVI_BANDS + ["SCL"]

# SCL cloud + shadow classes (same as imint/fetch.py)
_SCL_CLOUD_CLASSES = frozenset({3, 8, 9, 10})

_REQUEST_TIMEOUT_S = 60
_MAX_RETRIES = 3
_RETRY_DELAY_S = 2.0


# ── Public API ───────────────────────────────────────────────────────────

def fetch_s2_scene(
    west: float,
    south: float,
    east: float,
    north: float,
    date: str,
    *,
    size_px: int | tuple[int, int] = 256,
    cloud_threshold: float = 0.10,
    haze_threshold: float = 0.06,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Fetch a single Sentinel-2 L2A scene via Sentinel Hub Process API.

    Downloads 6 spectral bands + SCL in one HTTP POST, then checks
    cloud fraction and haze locally.

    Args:
        west, south, east, north: Bounding box in EPSG:3006 (meters).
        date: ISO date string, e.g. "2019-07-15".
        size_px: Output size — int for square or (H, W) tuple.
        cloud_threshold: Max cloud+shadow fraction (0–1).  Scenes above
            this are rejected (returns None).
        haze_threshold: Max mean B02 reflectance for haze gate.
            Scenes above this are rejected (returns None).

    Returns:
        Tuple of (spectral, scl, cloud_fraction) on success:
            spectral: (6, H, W) float32 reflectance [0, 1]
            scl: (H, W) uint8 Scene Classification Layer
            cloud_fraction: float cloud+shadow fraction
        Returns None if scene is too cloudy, too hazy, or fetch fails.
    """
    if isinstance(size_px, int):
        h_px, w_px = size_px, size_px
    else:
        h_px, w_px = size_px

    token = _get_token()

    # Fetch 7-band TIFF (6 spectral + SCL)
    try:
        tiff_bytes = _fetch_s2_tiff(
            west, south, east, north,
            w_px, h_px,
            date=date,
            token=token,
        )
    except Exception as e:
        return None

    # Parse multi-band TIFF
    bands = _parse_multiband_tiff(tiff_bytes, h_px, w_px, len(_ALL_BANDS))
    if bands is None or len(bands) < len(_ALL_BANDS):
        return None

    # Split spectral (0-5) and SCL (6)
    spectral = np.stack(bands[:6], axis=0)  # (6, H, W) float32 reflectance
    scl = bands[6].astype(np.uint8)         # (H, W) SCL class values

    # Cloud fraction check
    cloud_fraction = _check_cloud_fraction(scl)
    if cloud_fraction > cloud_threshold:
        return None

    # Nodata check (band 0 = B02)
    nodata_frac = float((spectral[0] == 0).mean())
    if nodata_frac > 0.10:
        return None

    # Haze check (mean B02 reflectance)
    b02_mean = float(spectral[0].mean())
    if b02_mean > haze_threshold:
        return None

    return spectral, scl, cloud_fraction


def fetch_s2_seasonal_tile(
    easting: int,
    northing: int,
    coords_wgs84: dict,
    windows: list[tuple[int, int]],
    years: list[str],
    *,
    size_px: int = 256,
    cloud_threshold: float = 0.10,
    haze_threshold: float = 0.06,
    scene_cloud_max: float = 50.0,
    max_candidates: int = 3,
) -> dict | None:
    """Fetch a complete multitemporal tile via Sentinel Hub Process API.

    Full pipeline: STAC discovery → Process API fetch → quality gates →
    frame assembly.  Returns a dict ready to save as .npz.

    Args:
        easting, northing: Tile center in EPSG:3006.
        coords_wgs84: WGS84 bounding box dict for STAC queries.
        windows: Seasonal windows, e.g. [(4,5), (6,7), (8,9), (1,2)].
        years: Years to search, e.g. ["2019", "2018"].
        size_px: Tile size in pixels (square).
        cloud_threshold: Max AOI cloud fraction for acceptance.
        haze_threshold: Max B02 mean reflectance for haze gate.
        scene_cloud_max: STAC pre-filter for scene-level cloud %.
        max_candidates: Max STAC candidates to try per season.

    Returns:
        Dict with keys ready for np.savez_compressed():
            image: (T*6, H, W) float32 reflectance [0, 1]
            dates, doy, temporal_mask, num_frames, num_bands, etc.
        Returns None if no valid frames could be fetched.
    """
    # Lazy import to avoid circular dependency
    from ..fetch import fetch_seasonal_dates

    half_m = size_px * 10 // 2  # 256 px × 10 m = 2560 m → half = 1280 m
    west = easting - half_m
    south = northing - half_m
    east = easting + half_m
    north = northing + half_m

    cell_key = f"{easting}_{northing}"
    n_frames = len(windows)
    n_bands = len(_PRITHVI_BANDS)

    # ── STAC discovery (same DES STAC as openEO path) ────────────────
    # Deterministic year rotation using cell hash
    cell_hash = int(hashlib.md5(cell_key.encode()).hexdigest(), 16)
    years_offset = cell_hash % len(years)
    years_order = years[years_offset:] + years[:years_offset]

    season_candidates = fetch_seasonal_dates(
        coords_wgs84, windows, years_order,
        scene_cloud_max=scene_cloud_max,
    )

    # ── Per-season fetch ─────────────────────────────────────────────
    frames: list[np.ndarray | None] = []
    frame_dates: list[str] = []
    frame_mask: list[int] = []

    for win_idx, (m_start, m_end) in enumerate(windows):
        candidates = season_candidates[win_idx]
        win_label = f"m{m_start}-{m_end}"

        if not candidates:
            print(f"  {win_label}: no STAC candidates")
            frames.append(None)
            frame_dates.append("")
            frame_mask.append(0)
            continue

        # Try top candidates — each is one fast HTTP call
        found = False
        for cand_date, scene_cc in candidates[:max_candidates]:
            result = fetch_s2_scene(
                west, south, east, north,
                date=cand_date,
                size_px=size_px,
                cloud_threshold=cloud_threshold,
                haze_threshold=haze_threshold,
            )

            if result is not None:
                spectral, scl, cloud_frac = result
                frames.append(spectral)
                frame_dates.append(cand_date)
                frame_mask.append(1)
                print(f"  {win_label}: {cand_date} OK "
                      f"(cloud={cloud_frac:.0%}, B02={spectral[0].mean():.4f})")
                found = True
                break
            else:
                print(f"  {win_label}: {cand_date} rejected")

        if not found:
            print(f"  {win_label}: no clear date (tried {min(len(candidates), max_candidates)})")
            frames.append(None)
            frame_dates.append("")
            frame_mask.append(0)

    # ── Check minimum frames ─────────────────────────────────────────
    n_valid = sum(frame_mask)
    if n_valid == 0:
        return None

    # ── Stack frames into (T*6, H, W) ────────────────────────────────
    ref_shape = None
    for f in frames:
        if f is not None:
            ref_shape = f.shape[1:]  # (H, W)
            break

    stacked = []
    for f in frames:
        if f is not None:
            if f.shape[1:] != ref_shape:
                padded = np.zeros(
                    (n_bands, ref_shape[0], ref_shape[1]),
                    dtype=np.float32,
                )
                h = min(f.shape[1], ref_shape[0])
                w = min(f.shape[2], ref_shape[1])
                padded[:, :h, :w] = f[:, :h, :w]
                stacked.append(padded)
            else:
                stacked.append(f)
        else:
            stacked.append(
                np.zeros((n_bands,) + ref_shape, dtype=np.float32)
            )

    image = np.concatenate(stacked, axis=0)  # (T*6, H, W)

    # ── Day-of-year ──────────────────────────────────────────────────
    doy_values = []
    for d in frame_dates:
        if d:
            doy_values.append(
                datetime.strptime(d, "%Y-%m-%d").timetuple().tm_yday
            )
        else:
            doy_values.append(0)

    return {
        "image": image,
        "dates": np.array(frame_dates),
        "doy": np.array(doy_values, dtype=np.int32),
        "temporal_mask": np.array(frame_mask, dtype=np.int32),
        "num_frames": n_frames,
        "num_bands": n_bands,
        "multitemporal": True,
        "source": "sentinel_hub",
        "easting": easting,
        "northing": northing,
    }


# ── Internal helpers ─────────────────────────────────────────────────────

def _check_cloud_fraction(scl: np.ndarray) -> float:
    """Compute cloud+shadow fraction from SCL array.

    Same logic as ``check_cloud_fraction`` in ``imint/fetch.py``:
    classes 3 (shadow), 8 (cloud medium), 9 (cloud high), 10 (cirrus).
    """
    cloud_mask = np.isin(scl, list(_SCL_CLOUD_CLASSES))
    return float(cloud_mask.sum()) / max(scl.size, 1)


def _build_evalscript() -> str:
    """Build Sentinel Hub evalscript for 7-band S2 L2A fetch.

    Returns 6 spectral bands as reflectance [0,1] and SCL as class value.
    Sentinel Hub auto-handles processing baseline offsets, so DN/10000
    gives correct BOA reflectance for all baselines.
    """
    return """//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B02", "B03", "B04", "B8A", "B11", "B12", "SCL"],
      units: "DN"
    }],
    output: {
      bands: 7,
      sampleType: "FLOAT32"
    }
  };
}

function evaluatePixel(sample) {
  // Spectral bands: DN → reflectance [0, 1]
  // Sentinel Hub normalizes baseline offsets, so DN/10000 is correct.
  // SCL: pass through as integer class value (0-11)
  return [
    sample.B02 / 10000,
    sample.B03 / 10000,
    sample.B04 / 10000,
    sample.B8A / 10000,
    sample.B11 / 10000,
    sample.B12 / 10000,
    sample.SCL
  ];
}
"""


def _fetch_s2_tiff(
    west: float, south: float,
    east: float, north: float,
    width_px: int, height_px: int,
    *,
    date: str,
    token: str,
) -> bytes:
    """Fetch S2 L2A data as multi-band GeoTIFF from Sentinel Hub Process API.

    Uses EPSG:3006 bbox directly — output is already aligned to the
    NMD 10 m grid, no reprojection needed.
    """
    # Time range: single day
    time_from = f"{date}T00:00:00Z"
    time_to = f"{date}T23:59:59Z"

    request_body = {
        "input": {
            "bounds": {
                "bbox": [west, south, east, north],
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/EPSG/0/3006"
                },
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": time_from,
                        "to": time_to,
                    },
                    "mosaickingOrder": "leastCC",
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

            # Verify TIFF magic bytes
            if data[:4] in (b"II*\x00", b"MM\x00*"):
                return data

            # Try JSON error
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

            # Token expired
            if e.code == 401:
                from .cdse_vpp import _cached_token, _token_expires
                with _token_lock:
                    import imint.training.cdse_vpp as _vpp_mod
                    _vpp_mod._cached_token = None
                    _vpp_mod._token_expires = 0.0
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
                f"Sentinel Hub S2 error (HTTP {e.code}): {body}"
            ) from e

        except Exception as e:
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY_S * (attempt + 1))
                continue
            raise RuntimeError(
                f"S2 fetch failed after {_MAX_RETRIES + 1} attempts: {e}"
            ) from e

    raise RuntimeError("S2 fetch failed: exhausted all retries")


def _parse_multiband_tiff(
    tiff_bytes: bytes,
    expected_h: int,
    expected_w: int,
    expected_bands: int,
) -> list[np.ndarray] | None:
    """Parse a multi-band GeoTIFF into per-band numpy arrays.

    Returns list of (H, W) float32 arrays, or None on failure.
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

                    nodata = ds.nodata
                    if nodata is not None:
                        data = np.where(data == nodata, 0, data)

                    if data.shape != (expected_h, expected_w):
                        from scipy.ndimage import zoom
                        zy = expected_h / data.shape[0]
                        zx = expected_w / data.shape[1]
                        # Nearest for SCL (last band), bilinear for spectral
                        order = 0 if band_idx == n_bands else 1
                        data = zoom(data, (zy, zx), order=order)

                    result.append(data.astype(np.float32))

                # Pad with zeros if fewer bands returned
                while len(result) < expected_bands:
                    result.append(
                        np.zeros((expected_h, expected_w), dtype=np.float32)
                    )

                return result

    except Exception:
        return None
