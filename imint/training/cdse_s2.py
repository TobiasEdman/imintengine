"""Sentinel-2 L2A data fetching via CDSE Sentinel Hub Process API.

Primary HTTP-based fetcher for Sentinel-2 spectral and SCL data from the
Copernicus Data Space Ecosystem.  Uses the Sentinel Hub Process API (direct
HTTP POST) instead of openEO, providing lower latency and fewer dependencies.

**OpenEO is the fallback** — if the HTTP fetch fails (network error, rate
limit, token issue), the pipeline falls back to the openEO-based fetcher
(``fetch_copernicus_data`` / ``fetch_des_data`` in ``imint/fetch.py``).
See ``fetch_s2_scene_with_fallback()`` for the combined approach.

Fetch strategy:
    1. STAC discovery via DES STAC (``explorer.digitalearth.se``)
    2. For each seasonal window, try top candidate dates:
       a. Fetch 7 bands (B02..B12 + SCL) via Sentinel Hub Process API
       b. Check SCL cloud fraction locally (same logic as openEO path)
       c. Check quality gates (nodata %, B02 haze)
       d. Accept or try next candidate
    3. If SH Process API fails → fall back to openEO for that tile
    4. Stack frames → (T*6, H, W) multitemporal tile

Access:
    STAC discovery (date/cloud search):
    DES STAC: https://explorer.digitalearth.se/stac/search

    Data fetch (primary):
    Sentinel Hub Process API (CDSE)
    Collection: sentinel-2-l2a (built-in, no BYOC)
    Endpoint: https://sh.dataspace.copernicus.eu/api/v1/process
    Auth: CDSE OAuth2 client_credentials (same as VPP)

    Data fetch (fallback — used if HTTP fails):
    openEO via ``imint/fetch.py``
    CDSE openEO: https://openeo.dataspace.copernicus.eu/
    DES openEO:  https://openeo.digitalearth.se

License: Copernicus Open Access

Typical usage::

    from imint.training.cdse_s2 import fetch_s2_scene, fetch_s2_scene_with_fallback

    # HTTP only (fast, no openEO dependency)
    result = fetch_s2_scene(west, south, east, north, date="2019-07-15")
    if result is not None:
        spectral, scl, cloud_frac = result
        # spectral.shape == (6, 256, 256), dtype float32, reflectance [0,1]

    # HTTP primary + openEO fallback
    result = fetch_s2_scene_with_fallback(
        west, south, east, north, date="2019-07-15",
        coords_wgs84={"west": 18.0, "south": 59.0, "east": 18.1, "north": 59.1},
    )
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

_CRS_3006 = "http://www.opengis.net/def/crs/EPSG/0/3006"
_CRS_4326 = "http://www.opengis.net/def/crs/EPSG/0/4326"


def _prescreen_scl(
    west: float, south: float, east: float, north: float,
    date: str, h_px: int, w_px: int,
    token: str, crs: str,
    cloud_threshold: float,
) -> tuple[np.ndarray | None, float]:
    """Fetch only SCL band for cloud pre-screening (6× less data).

    Returns (scl, cloud_fraction) or (None, 1.0) on failure.
    """
    try:
        tiff_bytes = _fetch_s2_tiff(
            west, south, east, north, w_px, h_px,
            date=date, token=token, crs=crs,
            evalscript=_build_scl_only_evalscript(),
        )
    except Exception:
        return None, 1.0

    bands = _parse_multiband_tiff(tiff_bytes, h_px, w_px, 1)
    if bands is None or len(bands) < 1:
        return None, 1.0

    scl = bands[0].astype(np.uint8)
    cloud_fraction = _check_cloud_fraction(scl)
    return scl, cloud_fraction


# Tiles >= this size use two-stage fetch (SCL pre-screen + spectral)
_TWO_STAGE_THRESHOLD = 384  # 512px tiles benefit; 256px overhead not worth it


def fetch_s2_scene(
    west: float,
    south: float,
    east: float,
    north: float,
    date: str,
    *,
    crs: str = _CRS_3006,
    size_px: int | tuple[int, int] = 256,
    cloud_threshold: float = 0.10,
    haze_threshold: float = 0.06,
    nodata_threshold: float | None = 0.05,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Fetch a single Sentinel-2 L2A scene via Sentinel Hub Process API.

    For tiles >= 384px, uses two-stage fetch: SCL-only pre-screen (cheap)
    followed by full spectral (expensive) only if cloud gate passes.
    Saves ~6× bandwidth on rejected cloudy scenes.

    Returns:
        (spectral, scl, cloud_fraction) on success, None on rejection.
    """
    h_px, w_px = (size_px, size_px) if isinstance(size_px, int) else size_px
    token = _get_token()

    # Two-stage for large tiles: pre-screen with SCL only
    use_two_stage = max(h_px, w_px) >= _TWO_STAGE_THRESHOLD
    if use_two_stage:
        scl, cloud_fraction = _prescreen_scl(
            west, south, east, north, date, h_px, w_px,
            token, crs, cloud_threshold,
        )
        if scl is None or cloud_fraction > cloud_threshold:
            return None

    # Fetch full spectral + SCL
    try:
        tiff_bytes = _fetch_s2_tiff(
            west, south, east, north, w_px, h_px,
            date=date, token=token, crs=crs,
        )
    except Exception as e:
        print(f"    [SH HTTP] {date}: {e}")
        return None

    bands = _parse_multiband_tiff(tiff_bytes, h_px, w_px, len(_ALL_BANDS))
    if bands is None or len(bands) < len(_ALL_BANDS):
        return None

    spectral = np.stack(bands[:6], axis=0)  # (6, H, W) float32
    scl = bands[6].astype(np.uint8)         # (H, W) uint8

    if not use_two_stage:
        cloud_fraction = _check_cloud_fraction(scl)
        if cloud_fraction > cloud_threshold:
            return None

    if nodata_threshold is not None:
        nodata_frac = float((spectral[0] == 0).mean())
        if nodata_frac > nodata_threshold:
            return None

    valid = spectral[0] > 0
    if valid.any():
        if float(spectral[0][valid].mean()) > haze_threshold:
            return None

    return spectral, scl, cloud_fraction


def fetch_s2_scene_wgs84(
    west: float, south: float, east: float, north: float, date: str, **kwargs,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Convenience wrapper: fetch_s2_scene with WGS84 defaults.

    Defaults: crs=EPSG:4326, cloud<0.30, haze<0.10, nodata=None (disabled).
    """
    kwargs.setdefault("crs", _CRS_4326)
    kwargs.setdefault("cloud_threshold", 0.30)
    kwargs.setdefault("haze_threshold", 0.10)
    kwargs.setdefault("nodata_threshold", None)
    return fetch_s2_scene(west, south, east, north, date, **kwargs)

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
        "spectral": image,
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


def fetch_s2_scene_with_fallback(
    west: float,
    south: float,
    east: float,
    north: float,
    date: str,
    *,
    coords_wgs84: dict | None = None,
    size_px: int | tuple[int, int] = 256,
    cloud_threshold: float = 0.10,
    haze_threshold: float = 0.06,
    source: str = "copernicus",
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Fetch S2 scene via HTTP, falling back to openEO if HTTP fails.

    Primary: Sentinel Hub Process API (fast, single HTTP POST)
    Fallback: openEO (``fetch_copernicus_data`` or ``fetch_des_data``)

    Args:
        west, south, east, north: EPSG:3006 bbox (snapped to NMD grid).
        date: ISO date string.
        coords_wgs84: WGS84 bbox dict for openEO fallback. If None,
            openEO fallback is skipped.
        size_px: Output size — int for square or (H, W) tuple.
        cloud_threshold: Max cloud+shadow fraction (0–1).
        haze_threshold: Max mean B02 reflectance for haze gate.
        source: openEO fallback source ("copernicus" or "des").

    Returns:
        Tuple of (spectral, scl, cloud_fraction) on success, or None.
    """
    # ── Primary: Sentinel Hub HTTP ────────────────────────────────────
    result = fetch_s2_scene(
        west, south, east, north, date,
        size_px=size_px,
        cloud_threshold=cloud_threshold,
        haze_threshold=haze_threshold,
    )
    if result is not None:
        return result

    # ── Fallback: openEO ──────────────────────────────────────────────
    if coords_wgs84 is None:
        return None

    try:
        from ..fetch import fetch_sentinel2_data, check_cloud_fraction
        print(f"    [openEO fallback] Trying {source}...")
        fetch_result = fetch_sentinel2_data(
            source=source,
            date=date,
            coords=coords_wgs84,
            cloud_threshold=1.0,    # check manually below
            include_scl=True,
        )
        # Extract Prithvi bands
        missing = [b for b in _PRITHVI_BANDS if b not in fetch_result.bands]
        if missing:
            return None

        spectral = np.stack(
            [fetch_result.bands[b] for b in _PRITHVI_BANDS], axis=0,
        ).astype(np.float32)

        scl = fetch_result.scl
        if scl is None:
            scl = np.zeros(spectral.shape[1:], dtype=np.uint8)
        cloud_frac = check_cloud_fraction(scl)

        if cloud_frac > cloud_threshold:
            return None

        nodata_frac = float((spectral[0] == 0).mean())
        if nodata_frac > 0.10:
            return None

        b02_mean = float(spectral[0].mean())
        if b02_mean > haze_threshold:
            return None

        return spectral, scl, cloud_frac
    except Exception as e:
        print(f"    [openEO fallback] Also failed: {e}")
        return None


# ── Internal helpers ─────────────────────────────────────────────────────

def _check_cloud_fraction(scl: np.ndarray) -> float:
    """Compute cloud+shadow fraction from SCL array.

    Same logic as ``check_cloud_fraction`` in ``imint/fetch.py``:
    classes 3 (shadow), 8 (cloud medium), 9 (cloud high), 10 (cirrus).
    """
    cloud_mask = np.isin(scl, list(_SCL_CLOUD_CLASSES))
    return float(cloud_mask.sum()) / max(scl.size, 1)


def _build_evalscript() -> str:
    """Build Sentinel Hub evalscript for 7-band S2 L2A fetch."""
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


def _build_scl_only_evalscript() -> str:
    """Evalscript that fetches only the SCL band for cloud pre-screening.

    At 512px this transfers ~250 KB instead of ~1.5 MB (6× less).
    """
    return """//VERSION=3
function setup() {
  return {
    input: [{ bands: ["SCL"], units: "DN" }],
    output: { bands: 1, sampleType: "UINT8" }
  };
}
function evaluatePixel(sample) {
  return [sample.SCL];
}
"""


def _fetch_s2_tiff(
    west: float, south: float,
    east: float, north: float,
    width_px: int, height_px: int,
    *,
    date: str,
    token: str,
    crs: str = "http://www.opengis.net/def/crs/EPSG/0/3006",
    evalscript: str | None = None,
) -> bytes:
    """Fetch S2 L2A data as multi-band GeoTIFF from Sentinel Hub Process API.

    Args:
        crs: Coordinate reference system URI. Default EPSG:3006 (Sweden).
        evalscript: Custom evalscript. Defaults to 7-band spectral+SCL.
    """
    # Time range: single day
    time_from = f"{date}T00:00:00Z"
    time_to = f"{date}T23:59:59Z"

    request_body = {
        "input": {
            "bounds": {
                "bbox": [west, south, east, north],
                "properties": {
                    "crs": crs
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
        "evalscript": evalscript or _build_evalscript(),
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

            # Server overload — raise immediately, no point retrying
            if e.code in (502, 503, 504):
                raise RuntimeError(
                    f"Sentinel Hub S2 error (HTTP {e.code}): server overload"
                ) from e

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


def cdse_catalog_search(
    bbox_4326: tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    *,
    max_cloud: float = 40.0,
    max_results: int = 100,
) -> list[tuple[str, float]]:
    """Search CDSE OData catalog for Sentinel-2 L2A scenes.

    Uses the CDSE OData v1 API with ``$expand=Attributes`` to retrieve
    cloud cover.  Requires CDSE client credentials (env vars
    ``CDSE_CLIENT_ID`` / ``CDSE_CLIENT_SECRET``) for attribute expansion.

    Note: The CDSE STAC endpoint no longer indexes Sentinel-2 and the
    OpenSearch/Resto API now requires auth.  OData is the only reliable
    unauthenticated entry-point for catalog queries, but cloud cover
    only appears when ``$expand=Attributes`` is combined with an auth
    Bearer token.

    Args:
        bbox_4326: ``(west, south, east, north)`` in WGS84.
        date_start: ISO date string, e.g. ``"2016-06-01"``.
        date_end: ISO date string, e.g. ``"2016-08-16"``.
        max_cloud: Scene-level cloud cover ceiling (0–100 %).
        max_results: Max items per OData page.

    Returns:
        List of ``(date_str, cloud_pct)`` tuples sorted by
        ``cloud_pct`` ascending (clearest first).  ``date_str``
        is ``"YYYY-MM-DD"``; ``cloud_pct`` is 0–100.
        Returns an empty list on network or auth failure.
    """
    west, south, east, north = bbox_4326

    # ── Auth token (needed for $expand=Attributes → cloud cover) ──
    token: str | None = None
    try:
        token = _get_cdse_token()
    except Exception:
        pass  # fall through — will proceed without cloud filter

    # ── OData filter ──────────────────────────────────────────────
    wkt = (
        f"POLYGON(({west} {south},{east} {south},"
        f"{east} {north},{west} {north},{west} {south}))"
    )
    filt_parts = [
        "Collection/Name eq 'SENTINEL-2'",
        f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')",
        f"ContentDate/Start gt {date_start}T00:00:00.000Z",
        f"ContentDate/Start lt {date_end}T23:59:59.000Z",
        "contains(Name,'L2A')",
    ]
    filt = " and ".join(filt_parts)

    base = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    url = (
        base
        + "?$top=" + str(max_results)
        + "&$expand=Attributes"
        + "&$filter=" + urllib.parse.quote(filt)
    )

    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        req = urllib.request.Request(url, headers=headers)
        resp = urllib.request.urlopen(req, timeout=30)
        payload = json.loads(resp.read())
    except Exception:
        return []

    best: dict[str, float] = {}   # date → lowest cloud

    for item in payload.get("value", []):
        # Cloud cover is in the expanded Attributes list
        attrs = {a["Name"]: a.get("Value") for a in item.get("Attributes", [])}
        cloud_raw = attrs.get("cloudCover")
        if cloud_raw is None:
            cloud = 50.0   # unknown — treat as 50 %
        else:
            cloud = float(cloud_raw)

        if cloud > max_cloud:
            continue

        # Date from ContentDate.Start ("2016-07-24T10:30:32.000Z")
        raw_dt = (item.get("ContentDate") or {}).get("Start", "")
        if not raw_dt:
            continue
        date_str = raw_dt[:10]  # "YYYY-MM-DD"

        # Keep only the lowest-cloud entry per date
        if date_str not in best or cloud < best[date_str]:
            best[date_str] = cloud

    return sorted(best.items(), key=lambda x: x[1])


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
