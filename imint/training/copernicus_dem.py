"""Copernicus DEM GLO-30 fetching.

Downloads terrain elevation tiles from the Copernicus DEM GLO-30
dataset hosted on AWS S3 (public, no authentication required) and
returns them as NumPy arrays aligned to the NMD 10 m grid.

The Copernicus DEM has ~30 m native resolution (1 arcsecond) and
is stored as Cloud-Optimized GeoTIFFs (COG) in 1°×1° tiles on S3.

We use rasterio with /vsicurl/ to fetch only the window we need
from each COG without downloading the entire tile. The result is
reprojected from EPSG:4326 to EPSG:3006 and resampled to 10 m
with bilinear interpolation.

Data source:
    https://copernicus-dem-30m.s3.amazonaws.com/
    (Copernicus DEM GLO-30, ESA/Airbus, free and open)

License: Open (Copernicus License)

Typical usage::

    from imint.training.copernicus_dem import fetch_dem_tile
    dem = fetch_dem_tile(west, south, east, north, size_px=256)
    # dem.shape == (256, 256), dtype float32, unit: meters above EGM2008

"""
from __future__ import annotations

import math
import time
import urllib.request
from pathlib import Path

import numpy as np

# ── S3 COG base URL ──────────────────────────────────────────────────────
_S3_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"
_REQUEST_TIMEOUT_S = 60
_MAX_RETRIES = 2
_RETRY_DELAY_S = 3.0


# ── Public API ───────────────────────────────────────────────────────────

def fetch_dem_tile(
    west: float,
    south: float,
    east: float,
    north: float,
    *,
    size_px: int | tuple[int, int] = 256,
    cache_dir: Path | None = None,
) -> np.ndarray:
    """Fetch terrain elevation tile from Copernicus DEM GLO-30.

    Args:
        west, south, east, north: Bounding box in EPSG:3006 (meters).
        size_px: Output size — int for square or (H, W) tuple.
        cache_dir: Optional .npy cache directory.

    Returns:
        (H, W) float32 array with elevation in meters above
        EGM2008 geoid.  NoData pixels are 0.
    """
    if isinstance(size_px, int):
        h_px, w_px = size_px, size_px
    else:
        h_px, w_px = size_px

    # Check cache
    if cache_dir is not None:
        cache_key = f"dem_{int(west)}_{int(south)}_{int(east)}_{int(north)}.npy"
        cache_path = cache_dir / cache_key
        if cache_path.exists():
            cached = np.load(cache_path)
            if cached.shape == (h_px, w_px):
                return cached

    # Reproject bbox from EPSG:3006 → EPSG:4326
    lon_min, lat_min, lon_max, lat_max = _bbox_3006_to_4326(
        west, south, east, north
    )

    # Determine which 1°×1° COG tiles we need
    cog_urls = _get_cog_urls(lat_min, lat_max, lon_min, lon_max)

    # Fetch and mosaic the DEM, reproject to EPSG:3006
    dem = _fetch_and_reproject(
        cog_urls, west, south, east, north, h_px, w_px,
    )

    # Clamp nodata/negative to 0 (some coastal tiles may have <0)
    dem = np.clip(dem, 0.0, None)

    # Cache
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache_dir / cache_key.replace(".npy", "_tmp.npy")
        np.save(tmp, dem)
        Path(str(tmp)).rename(cache_path)

    return dem


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


def _get_cog_urls(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
) -> list[str]:
    """Get Copernicus DEM COG URLs covering a WGS84 bounding box."""
    urls = []
    for lat in range(math.floor(lat_min), math.ceil(lat_max)):
        for lon in range(math.floor(lon_min), math.ceil(lon_max)):
            lat_prefix = "N" if lat >= 0 else "S"
            lon_prefix = "E" if lon >= 0 else "W"
            lat_str = f"{lat_prefix}{abs(lat):02d}"
            lon_str = f"{lon_prefix}{abs(lon):03d}"
            tile_name = (
                f"Copernicus_DSM_COG_10_{lat_str}_00_{lon_str}_00_DEM"
            )
            url = f"{_S3_BASE}/{tile_name}/{tile_name}.tif"
            urls.append(url)
    return urls


def _fetch_and_reproject(
    cog_urls: list[str],
    west: float, south: float, east: float, north: float,
    h_px: int, w_px: int,
) -> np.ndarray:
    """Fetch DEM from COG URLs, mosaic and reproject to EPSG:3006.

    Uses rasterio's WarpedVRT for efficient on-the-fly reprojection
    without creating intermediate files.
    """
    try:
        import rasterio
        from rasterio.crs import CRS
        from rasterio.io import MemoryFile
        from rasterio.merge import merge
        from rasterio.vrt import WarpedVRT
        from rasterio.warp import Resampling
        from rasterio.transform import from_bounds
    except ImportError:
        raise ImportError(
            "rasterio is required for DEM fetching. "
            "Install with: pip install rasterio"
        )

    dst_crs = CRS.from_epsg(3006)
    dst_transform = from_bounds(west, south, east, north, w_px, h_px)

    datasets = []
    for url in cog_urls:
        for attempt in range(_MAX_RETRIES + 1):
            try:
                src = rasterio.open(url)
                # Create a WarpedVRT that reprojects on-the-fly
                vrt = WarpedVRT(
                    src,
                    crs=dst_crs,
                    resampling=Resampling.bilinear,
                )
                datasets.append(vrt)
                break
            except Exception as e:
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY_S * (attempt + 1))
                    continue
                # Skip tiles that can't be opened (e.g. ocean-only)
                break

    if not datasets:
        # No DEM data available for this area (e.g. far offshore)
        return np.zeros((h_px, w_px), dtype=np.float32)

    # Merge and read at target resolution
    if len(datasets) == 1:
        # Single tile — read directly with target transform
        from rasterio.windows import from_bounds as win_from_bounds
        vrt = datasets[0]
        window = win_from_bounds(
            west, south, east, north, vrt.transform,
        )
        data = vrt.read(
            1, window=window,
            out_shape=(h_px, w_px),
            resampling=Resampling.bilinear,
        )
    else:
        # Multiple tiles — merge then read
        mosaic, mosaic_transform = merge(
            datasets,
            bounds=(west, south, east, north),
            res=(
                (east - west) / w_px,
                (north - south) / h_px,
            ),
            resampling=Resampling.bilinear,
        )
        data = mosaic[0]  # First band

    # Clean up
    for ds in datasets:
        try:
            ds.close()
        except Exception:
            pass

    # Ensure correct output shape
    if data.shape != (h_px, w_px):
        from scipy.ndimage import zoom
        zy = h_px / data.shape[0]
        zx = w_px / data.shape[1]
        data = zoom(data, (zy, zx), order=1)

    # Handle nodata
    data = np.where(np.isnan(data), 0, data)
    return data.astype(np.float32)
