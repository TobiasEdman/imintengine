"""SLU Markfuktighetskarta fetching — Pirinen 2023 lager #4.

Downloads soil moisture rasters from Skogsstyrelsen's ArcGIS ImageServer
(SLU Markfuktighetskarta v1, William Lidberg & Anneli Ågren, SLU Umeå)
and returns them as NumPy arrays aligned to the NMD 10 m grid.

Native resolution: ~2 m (1 m DEM-derived), resampled to 10 m by the
ImageServer when we ask for `size=500x500` over a 5 km bbox.

Encoding (uint8):
    0        no data / outside Sweden
    1–100    soil moisture probability (low → high), %
    101      saturated water / lake fill

The fetcher snaps the input bbox to the 10 m EPSG:3006 grid (same as
``skg_height.py``) before requesting from the proxy.

Access goes through Skogsstyrelsen's public kartportal proxy that
forwards to the underlying ImageServer without requiring auth — same
mechanism documented in ``docs/conversation_log/13_auxiliary_channels_fusion.md``.

Endpoint URL is loaded from ``.skg_endpoints`` config (``[markfukt_slu]``
section) or the ``SKG_MARKFUKT_URL`` environment variable. A public
default is provided as fallback since SLU Markfuktighetskarta is open
data (CC0 / open license).

License: Open (Skogsstyrelsen + SLU)

Typical usage::

    from imint.training.slu_markfukt import fetch_markfukt_tile
    sm = fetch_markfukt_tile(west, south, east, north, size_px=1000)
    # sm.shape == (1000, 1000), dtype uint8, 0 = nodata, 1–101 = moisture
"""

from __future__ import annotations

import configparser
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from .skg_height import _to_nmd_grid_bounds


# ── ArcGIS ImageServer endpoint ──────────────────────────────────────────
#
# SLU Markfuktighetskarta is open data; the proxy is publicly reachable.
# A default URL is hardcoded; can be overridden via env var or config file.

_DEFAULT_URL = "https://kartor.skogsstyrelsen.se/kartor/markfuktighetSLUArcGIS93Rest"

_REQUEST_TIMEOUT_S = 60
_MAX_RETRIES = 2
_RETRY_DELAY_S = 3.0


def _get_markfukt_url() -> str:
    """Resolve the ImageServer URL.

    Priority: env var → ``.skg_endpoints`` → hardcoded public default.
    """
    url = os.environ.get("SKG_MARKFUKT_URL")
    if url:
        return url

    for cfg_path in (
        Path(__file__).resolve().parents[2] / ".skg_endpoints",
        Path.home() / ".skg_endpoints",
    ):
        if cfg_path.exists():
            cp = configparser.ConfigParser()
            cp.read(cfg_path)
            if cp.has_option("markfukt_slu", "url"):
                return cp.get("markfukt_slu", "url")

    return _DEFAULT_URL


# ── Public API ───────────────────────────────────────────────────────────


def fetch_markfukt_tile(
    west: float,
    south: float,
    east: float,
    north: float,
    *,
    size_px: int | tuple[int, int] = 1000,
    cache_dir: Path | None = None,
) -> np.ndarray:
    """Fetch an SLU Markfuktighet tile from Skogsstyrelsen ImageServer.

    Args:
        west, south, east, north: Bounding box in EPSG:3006 (meters).
        size_px: Output raster size in pixels.  Either a single int for
            square tiles or a ``(height, width)`` tuple.
        cache_dir: If provided, cache downloaded tiles as ``.npy`` files.

    Returns:
        NumPy array of shape ``(H, W)``, dtype uint8.
        Values: 0 = nodata, 1–101 = soil moisture probability (%).
    """
    h_px, w_px = (size_px, size_px) if isinstance(size_px, int) else size_px

    if cache_dir is not None:
        cache_key = (
            f"markfukt_{int(west)}_{int(south)}_{int(east)}_{int(north)}.npy"
        )
        cache_path = cache_dir / cache_key
        if cache_path.exists():
            cached = np.load(cache_path)
            if cached.shape == (h_px, w_px):
                return cached

    tiff_bytes = _download_tile(west, south, east, north, w_px, h_px)
    arr = _parse_tiff(tiff_bytes, h_px, w_px)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache_dir / cache_key.replace(".npy", "_tmp.npy")
        np.save(tmp, arr)
        Path(str(tmp)).rename(cache_path)

    return arr


def fetch_markfukt_for_coords(
    coords: dict,
    *,
    size_px: int = 1000,
    cache_dir: Path | None = None,
) -> np.ndarray:
    """Fetch SLU Markfuktighet for a WGS84 bbox (snapped to NMD 10 m grid)."""
    projected = _to_nmd_grid_bounds(coords)
    return fetch_markfukt_tile(
        projected["west"], projected["south"],
        projected["east"], projected["north"],
        size_px=size_px,
        cache_dir=cache_dir,
    )


# ── Internal helpers ─────────────────────────────────────────────────────


def _download_tile(
    west: float, south: float, east: float, north: float,
    width_px: int, height_px: int,
) -> bytes:
    """Download a uint8 GeoTIFF from the ImageServer exportImage endpoint."""
    params = {
        "bbox": f"{west},{south},{east},{north}",
        "bboxSR": "3006",
        "imageSR": "3006",
        "size": f"{width_px},{height_px}",
        "format": "tiff",
        "pixelType": "U8",
        "noDataInterpretation": "esriNoDataMatchAny",
        "interpolation": "RSP_NearestNeighbor",
        "f": "image",
    }
    url = _get_markfukt_url() + "?" + urllib.parse.urlencode(params)

    last_err: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "ImintEngine/1.0")
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
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
            last_err = e
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY_S * (attempt + 1))
                continue
            raise RuntimeError(
                f"SLU Markfuktighet fetch failed after "
                f"{_MAX_RETRIES + 1} attempts: {e}"
            ) from e

    # Unreachable, satisfies type-checker
    raise RuntimeError(f"Unreachable: {last_err}")


def _parse_tiff(
    tiff_bytes: bytes,
    expected_h: int,
    expected_w: int,
) -> np.ndarray:
    """Parse a uint8 GeoTIFF from bytes into a 2D numpy array."""
    import rasterio
    from rasterio.io import MemoryFile

    with MemoryFile(tiff_bytes) as memfile:
        with memfile.open() as ds:
            data = ds.read(1)

            if data.shape != (expected_h, expected_w):
                from scipy.ndimage import zoom
                zy = expected_h / data.shape[0]
                zx = expected_w / data.shape[1]
                data = zoom(data, (zy, zx), order=0)  # nearest for class-like

            return data.astype(np.uint8)
