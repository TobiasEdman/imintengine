"""Skogsstyrelsen tree height data fetching.

Downloads tree height rasters from Skogsstyrelsen's ArcGIS ImageServer
(Trädhöjd 3.1) and returns them as NumPy arrays aligned to the NMD 10 m
grid used by the training pipeline.

The height data is "Skogliga grunddata — Trädhöjd", derived from
airborne laser scanning (ALS) and aerial image matching, with ~2 m
native resolution.  Values are stored in **decimeters** (125 = 12.5 m).
We resample to 10 m to match the Sentinel-2 / NMD grid.

Access goes through Skogsstyrelsen's public kartportal proxy which
exposes the ImageServer ``exportImage`` endpoint without requiring a
Geodatasamverkan user account.

Data source:
    URL_REMOVED_SEE_SKG_ENDPOINTS
    (proxies geodata.[REDACTED] Trädhöjd 3.1 ImageServer)

License: CC0 (public domain)

Typical usage in the training pipeline::

    from imint.training.skg_height import fetch_height_tile
    height = fetch_height_tile(west, south, east, north, size_px=256)
    # height.shape == (256, 256), dtype float32, unit: meters
"""

from __future__ import annotations

import io
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

# ── ArcGIS ImageServer endpoint (via kartportal proxy) ─────────────────
# The kartportal at kartor.skogsstyrelsen.se proxies the authenticated
# ImageServer exportImage endpoint, making it publicly accessible.
_IMAGESERVER_URL = (
    "URL_REMOVED_SEE_SKG_ENDPOINTS"
)

# MosaicRule selects the "THF" product (Trädhöjd Flygbild — aerial image
# matching + laser ground hits).  Without this the server may return
# composites from different processing rounds.
_MOSAIC_RULE = '{"mosaicMethod":"esriMosaicNone","where":"ProductName = \'THF\'"}'

_REQUEST_TIMEOUT_S = 60
_MAX_RETRIES = 2
_RETRY_DELAY_S = 3.0


# ── Public API ───────────────────────────────────────────────────────────

def fetch_height_tile(
    west: float,
    south: float,
    east: float,
    north: float,
    *,
    size_px: int | tuple[int, int] = 256,
    cache_dir: Path | None = None,
) -> np.ndarray:
    """Fetch a tree height tile from Skogsstyrelsen ImageServer.

    The output resolution is determined by bbox extent / pixel count,
    so a 2560 m bbox with size_px=256 gives exactly 10 m pixels aligned
    to the same EPSG:3006 grid as Sentinel-2 and NMD.

    Args:
        west, south, east, north: Bounding box in EPSG:3006 (meters).
        size_px: Output raster size in pixels.  Either a single int for
            square tiles (e.g. 256 → 256×256) or a ``(height, width)``
            tuple for non-square tiles matching the Sentinel-2 image
            shape.
        cache_dir: If provided, cache downloaded tiles as .npy files
            for fast resumption.

    Returns:
        NumPy array of shape ``(H, W)``, dtype float32, with tree height
        in **meters** (converted from decimeters).  NoData pixels are 0.
    """
    if isinstance(size_px, int):
        h_px, w_px = size_px, size_px
    else:
        h_px, w_px = size_px

    # Check cache first
    if cache_dir is not None:
        cache_key = f"height_{int(west)}_{int(south)}_{int(east)}_{int(north)}.npy"
        cache_path = cache_dir / cache_key
        if cache_path.exists():
            cached = np.load(cache_path)
            # Resize if cached shape doesn't match (e.g. square cache
            # but non-square tile requested now)
            if cached.shape == (h_px, w_px):
                return cached

    # Build request — ask server for exact (width, height) we need
    tiff_bytes = _download_tile(west, south, east, north, w_px, h_px)

    # Parse TIFF → numpy
    height = _parse_tiff(tiff_bytes, h_px, w_px)

    # Convert decimeters → meters
    height = height.astype(np.float32) / 10.0

    # Clamp negatives / nodata to zero
    height = np.clip(height, 0.0, None)

    # Cache if requested
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        # np.save appends .npy if missing, so use .npy suffix for tmp too
        tmp = cache_dir / cache_key.replace(".npy", "_tmp.npy")
        np.save(tmp, height)
        Path(str(tmp)).rename(cache_path)

    return height


def fetch_height_for_coords(
    coords: dict,
    *,
    size_px: int = 256,
    cache_dir: Path | None = None,
) -> np.ndarray:
    """Fetch tree height for a WGS84 bounding box.

    Converts WGS84 coords to EPSG:3006, snaps to NMD grid, and fetches.

    Args:
        coords: Dict with west/south/east/north in WGS84 (decimal degrees).
        size_px: Output size in pixels.
        cache_dir: Optional cache directory.

    Returns:
        Height array (size_px, size_px) in meters.
    """
    projected = _to_nmd_grid_bounds(coords)
    return fetch_height_tile(
        projected["west"], projected["south"],
        projected["east"], projected["north"],
        size_px=size_px,
        cache_dir=cache_dir,
    )


# ── Internal helpers ─────────────────────────────────────────────────────

def _to_nmd_grid_bounds(coords: dict) -> dict:
    """Convert WGS84 bbox to EPSG:3006, snapped to 10 m NMD grid.

    Reuses the same logic as fetch.py _to_nmd_grid for consistency.
    """
    from rasterio.warp import transform_bounds
    from rasterio.crs import CRS

    src_crs = CRS.from_epsg(4326)
    dst_crs = CRS.from_epsg(3006)

    w, s, e, n = transform_bounds(
        src_crs, dst_crs,
        coords["west"], coords["south"],
        coords["east"], coords["north"],
    )

    # Snap to 10 m grid (same as NMD_GRID_SIZE in fetch.py)
    grid = 10
    w = int(w // grid) * grid
    s = int(s // grid) * grid
    e = int(-(-e // grid)) * grid  # ceil
    n = int(-(-n // grid)) * grid  # ceil

    return {"west": w, "south": s, "east": e, "north": n}


def _download_tile(
    west: float, south: float, east: float, north: float,
    width_px: int, height_px: int | None = None,
) -> bytes:
    """Download a GeoTIFF tile from ArcGIS ImageServer exportImage."""
    if height_px is None:
        height_px = width_px
    params = {
        "bbox": f"{west},{south},{east},{north}",
        "bboxSR": "3006",
        "imageSR": "3006",
        "size": f"{width_px},{height_px}",
        "format": "tiff",
        "pixelType": "S16",          # signed 16-bit (height in dm)
        "noDataInterpretation": "esriNoDataMatchAny",
        "interpolation": "RSP_BilinearInterpolation",
        "mosaicRule": _MOSAIC_RULE,
        "f": "image",
    }

    url = _IMAGESERVER_URL + "?" + urllib.parse.urlencode(params)

    for attempt in range(_MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "ImintEngine/1.0")
            resp = urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S)
            data = resp.read()

            # Check we got actual image data (not JSON error)
            if data[:4] in (b"II*\x00", b"MM\x00*"):
                return data

            # If response looks like JSON error, decode and raise
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
                f"Skogsstyrelsen height fetch failed after "
                f"{_MAX_RETRIES + 1} attempts: {e}"
            ) from e


def _parse_tiff(
    tiff_bytes: bytes,
    expected_h: int,
    expected_w: int | None = None,
) -> np.ndarray:
    """Parse a GeoTIFF from bytes into a 2D numpy array.

    Uses rasterio (already a dependency) for reliable TIFF parsing.
    """
    if expected_w is None:
        expected_w = expected_h

    try:
        import rasterio
        from rasterio.io import MemoryFile

        with MemoryFile(tiff_bytes) as memfile:
            with memfile.open() as ds:
                data = ds.read(1)  # Single band

                # Handle nodata
                nodata = ds.nodata
                if nodata is not None:
                    data = np.where(data == nodata, 0, data)

                # Resize if needed (shouldn't be, but safety check)
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
