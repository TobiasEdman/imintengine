"""Skogsstyrelsen Skogliga grunddata fetching.

Downloads forestry variable rasters from Skogsstyrelsen's ArcGIS
ImageServer (Skogliga grunddata 3.1) and returns them as NumPy arrays
aligned to the NMD 10 m grid used by the training pipeline.

The mosaic contains 10 bands with these forestry variables:

    Band  Variable              Unit         Typical range
    ────  ────────────────────  ───────────  ────────────────
    0     Volym (volume)        m³sk/ha      0 – 700
    1     Medelhöjd (Hgv)       dm           0 – 350
    2     Grundyta (Gy)         m²/ha        0 – 60
    3     Medeldiameter (Dgv)   cm           0 – 60
    4     Biomassa              ton/ha       0 – 400
    5     Trädhöjd (laser)      dm           0 – 350
    6     (reserved)            —            —
    7     Scanningsdatum        code         —
    8     NMD produktivitet     class        0/1/2
    9     Omdrev                version      1/2

Access goes through Skogsstyrelsen's public kartportal proxy which
exposes the ImageServer ``exportImage`` endpoint without requiring a
Geodatasamverkan user account.

Endpoint URL is loaded from ``.skg_endpoints`` config file or the
``SKG_GRUNDDATA_URL`` environment variable.

License: CC0 (public domain)

Typical usage::

    from imint.training.skg_grunddata import fetch_volume_tile
    volume = fetch_volume_tile(west, south, east, north, size_px=256)
    # volume.shape == (256, 256), dtype float32, unit: m³sk/ha
"""

from __future__ import annotations

import configparser
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

# ── Band indices ──────────────────────────────────────────────────────────
BAND_VOLUME = 0          # Virkesförråd, m³sk/ha
BAND_MEAN_HEIGHT = 1     # Grundytevägd medelhöjd (Hgv), dm
BAND_BASAL_AREA = 2      # Grundyta (Gy), m²/ha
BAND_MEAN_DIAMETER = 3   # Grundytevägd medeldiameter (Dgv), cm
BAND_BIOMASS = 4         # Biomassa, ton torrsubstans/ha
BAND_TREE_HEIGHT = 5     # Trädhöjd (laser), dm
BAND_SCAN_DATE = 7       # Scanningsdatum (kodad)
BAND_NMD_PROD = 8        # NMD produktivitet: 0=annan, 1=produktiv, 2=improduktiv
BAND_OMDREV = 9          # Omdrev (version)

# ── ArcGIS ImageServer endpoint ────────────────────────────────────────
# Loaded from .skg_endpoints config file or SKG_GRUNDDATA_URL env var.
_IMAGESERVER_URL: str | None = None


def _get_grunddata_url() -> str:
    """Resolve the Skogliga grunddata endpoint URL."""
    global _IMAGESERVER_URL
    if _IMAGESERVER_URL is not None:
        return _IMAGESERVER_URL

    url = os.environ.get("SKG_GRUNDDATA_URL")
    if url:
        _IMAGESERVER_URL = url
        return url

    for candidate in [
        Path(__file__).resolve().parents[2] / ".skg_endpoints",
        Path.home() / ".skg_endpoints",
    ]:
        if candidate.exists():
            cfg = configparser.ConfigParser()
            cfg.read(candidate)
            url = cfg.get("grunddata", "url", fallback=None)
            if url:
                _IMAGESERVER_URL = url
                return url

    raise RuntimeError(
        "Skogliga grunddata URL not configured. "
        "Set SKG_GRUNDDATA_URL env var or create .skg_endpoints file."
    )

_REQUEST_TIMEOUT_S = 60
_MAX_RETRIES = 2
_RETRY_DELAY_S = 3.0


# ── Public API ───────────────────────────────────────────────────────────

def fetch_volume_tile(
    west: float,
    south: float,
    east: float,
    north: float,
    *,
    size_px: int | tuple[int, int] = 256,
    cache_dir: Path | None = None,
) -> np.ndarray:
    """Fetch a timber volume tile (m³sk/ha) from Skogliga grunddata.

    Args:
        west, south, east, north: Bounding box in EPSG:3006 (meters).
        size_px: Output size — int for square or (H, W) tuple.
        cache_dir: Optional .npy cache directory.

    Returns:
        (H, W) float32 array, unit m³sk/ha.  NoData pixels are 0.
    """
    return fetch_grunddata_tile(
        west, south, east, north,
        band=BAND_VOLUME,
        size_px=size_px,
        cache_dir=cache_dir,
        cache_prefix="volume",
    )


def fetch_basal_area_tile(
    west: float,
    south: float,
    east: float,
    north: float,
    *,
    size_px: int | tuple[int, int] = 256,
    cache_dir: Path | None = None,
) -> np.ndarray:
    """Fetch a basal area tile (m²/ha) from Skogliga grunddata.

    Grundyta (Gy) — the cross-sectional area of all tree stems at
    breast height (1.3 m) per hectare.  Typical range 0–60 m²/ha.

    Args:
        west, south, east, north: Bounding box in EPSG:3006 (meters).
        size_px: Output size — int for square or (H, W) tuple.
        cache_dir: Optional .npy cache directory.

    Returns:
        (H, W) float32 array, unit m²/ha.  NoData pixels are 0.
    """
    return fetch_grunddata_tile(
        west, south, east, north,
        band=BAND_BASAL_AREA,
        size_px=size_px,
        cache_dir=cache_dir,
        cache_prefix="basal_area",
    )


def fetch_diameter_tile(
    west: float,
    south: float,
    east: float,
    north: float,
    *,
    size_px: int | tuple[int, int] = 256,
    cache_dir: Path | None = None,
) -> np.ndarray:
    """Fetch a mean diameter tile (cm) from Skogliga grunddata.

    Grundytevägd medeldiameter (Dgv) — basal-area-weighted mean stem
    diameter at breast height.  Typical range 0–60 cm.

    Args:
        west, south, east, north: Bounding box in EPSG:3006 (meters).
        size_px: Output size — int for square or (H, W) tuple.
        cache_dir: Optional .npy cache directory.

    Returns:
        (H, W) float32 array, unit cm.  NoData pixels are 0.
    """
    return fetch_grunddata_tile(
        west, south, east, north,
        band=BAND_MEAN_DIAMETER,
        size_px=size_px,
        cache_dir=cache_dir,
        cache_prefix="diameter",
    )


def fetch_grunddata_tile(
    west: float,
    south: float,
    east: float,
    north: float,
    *,
    band: int = BAND_VOLUME,
    size_px: int | tuple[int, int] = 256,
    cache_dir: Path | None = None,
    cache_prefix: str = "grunddata",
) -> np.ndarray:
    """Fetch a single band from the Skogliga grunddata mosaic.

    The ImageServer returns all 10 bands as a multi-band GeoTIFF.
    We extract the requested band and return it as a 2D array.

    Args:
        west, south, east, north: Bounding box in EPSG:3006 (meters).
        band: Band index (0-based).  Use the ``BAND_*`` constants.
        size_px: Output size — int for square or (H, W) tuple.
        cache_dir: Optional .npy cache directory.
        cache_prefix: Prefix for cache filenames.

    Returns:
        (H, W) float32 array.  NoData pixels are 0.
    """
    if isinstance(size_px, int):
        h_px, w_px = size_px, size_px
    else:
        h_px, w_px = size_px

    # Check cache
    if cache_dir is not None:
        cache_key = (f"{cache_prefix}_{int(west)}_{int(south)}"
                     f"_{int(east)}_{int(north)}.npy")
        cache_path = cache_dir / cache_key
        if cache_path.exists():
            cached = np.load(cache_path)
            if cached.shape == (h_px, w_px):
                return cached

    # Download multi-band TIFF
    tiff_bytes = _download_tile(west, south, east, north, w_px, h_px)

    # Parse and extract requested band
    data = _parse_tiff_band(tiff_bytes, band, h_px, w_px)

    # Convert to float and clamp negatives
    data = data.astype(np.float32)
    data = np.clip(data, 0.0, None)

    # Cache
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache_dir / cache_key.replace(".npy", "_tmp.npy")
        np.save(tmp, data)
        Path(str(tmp)).rename(cache_path)

    return data


# ── Internal helpers ─────────────────────────────────────────────────────

def _download_tile(
    west: float, south: float, east: float, north: float,
    width_px: int, height_px: int,
) -> bytes:
    """Download a multi-band GeoTIFF from the Skogliga grunddata service."""
    params = {
        "bbox": f"{west},{south},{east},{north}",
        "bboxSR": "3006",
        "imageSR": "3006",
        "size": f"{width_px},{height_px}",
        "format": "tiff",
        "pixelType": "S16",
        "noDataInterpretation": "esriNoDataMatchAny",
        "interpolation": "RSP_BilinearInterpolation",
        "renderingRule": '{"rasterFunction":"none"}',
        "f": "image",
    }

    url = _get_grunddata_url() + "?" + urllib.parse.urlencode(params)

    for attempt in range(_MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "ImintEngine/1.0")
            resp = urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S)
            data = resp.read()

            if data[:4] in (b"II*\x00", b"MM\x00*"):
                return data

            try:
                import json
                err = json.loads(data)
                msg = err.get("error", {}).get("message", str(err)[:200])
                raise RuntimeError(f"ImageServer error: {msg}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise RuntimeError(
                    f"Unexpected response ({len(data)} bytes, "
                    f"starts with {data[:20]!r})"
                )

        except Exception as e:
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY_S * (attempt + 1))
                continue
            raise RuntimeError(
                f"Skogliga grunddata fetch failed after "
                f"{_MAX_RETRIES + 1} attempts: {e}"
            ) from e


def _parse_tiff_band(
    tiff_bytes: bytes,
    band: int,
    expected_h: int,
    expected_w: int,
) -> np.ndarray:
    """Parse a multi-band GeoTIFF and extract a single band."""
    try:
        import rasterio
        from rasterio.io import MemoryFile

        with MemoryFile(tiff_bytes) as memfile:
            with memfile.open() as ds:
                n_bands = ds.count
                if band >= n_bands:
                    raise ValueError(
                        f"Requested band {band} but TIFF has {n_bands} bands"
                    )

                # rasterio bands are 1-indexed
                data = ds.read(band + 1)

                # Handle nodata
                nodata = ds.nodata
                if nodata is not None:
                    data = np.where(data == nodata, 0, data)

                # Resize if needed
                if data.shape != (expected_h, expected_w):
                    from scipy.ndimage import zoom
                    zy = expected_h / data.shape[0]
                    zx = expected_w / data.shape[1]
                    data = zoom(data, (zy, zx), order=1)

                return data.astype(np.float32)

    except ImportError:
        raise ImportError(
            "rasterio is required for TIFF parsing. "
            "Install with: pip install rasterio"
        )
