"""
imint/fetch.py — DES/openEO data fetching and cloud detection

Fetches Sentinel-2 L2A data from Digital Earth Sweden (DES) via openEO,
handles multi-resolution bands, converts DN to reflectance, and checks
cloud cover using the SCL (Scene Classification Layer) band.

Also provides NMD (Nationellt Marktäckedata) fetching for LULC analysis.
NMD is a static 10m land cover dataset from Naturvårdsverket.

Usage:
    from imint.fetch import fetch_des_data, fetch_nmd_data, FetchError

    result = fetch_des_data(date="2022-06-15", coords={...})
    if result.cloud_fraction < 0.3:
        # Use result.rgb and result.bands for analysis
        ...

    nmd = fetch_nmd_data(coords={...}, target_shape=(H, W))
    # nmd.nmd_raster is uint8 with NMD class codes
"""
from __future__ import annotations

import hashlib
import io
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .utils import dn_to_reflectance, des_to_imint_bands, bands_to_rgb
from .job import GeoContext

# ── Constants ────────────────────────────────────────────────────────────────

OPENEO_URL = "https://openeo.digitalearth.se"
COLLECTION = "s2_msi_l2a"
NMD_COLLECTION = "NMD_2018_Basskikt_v1_1"
NMD_BAND = "Basskikt"
# NMD is a static dataset — temporal extent on DES is a single ingestion
# timestamp (2024-08-28), NOT the data vintage year. We omit the temporal
# filter entirely to avoid the DES server returning an empty cube and
# crashing with 500 Internal Error.
NMD_TEMPORAL = None

# NMD grid size in meters (EPSG:3006). All bounding boxes are snapped to this
# grid so that Sentinel-2 and NMD rasters are pixel-aligned.
NMD_GRID_SIZE = 10
TARGET_CRS = "EPSG:3006"

# DES band groupings by native resolution
BANDS_10M = ["b02", "b03", "b04", "b08"]
BANDS_20M_SPECTRAL = ["b05", "b06", "b07", "b8a", "b11", "b12"]
BANDS_60M = ["b09"]
BANDS_20M_CATEGORICAL = ["scl"]

# SCL cloud classes (Sentinel-2 L2A Scene Classification)
# 8 = cloud_medium_probability, 9 = cloud_high_probability, 10 = thin_cirrus
SCL_CLOUD_CLASSES = frozenset({8, 9, 10})

# Default token file location (project root)
TOKEN_PATH_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".des_token"
)


# ── Grid alignment ───────────────────────────────────────────────────────────

def _to_nmd_grid(coords: dict) -> dict:
    """Convert WGS84 bounding box to EPSG:3006 snapped to the NMD 10m grid.

    Steps:
        1. Project WGS84 (EPSG:4326) → SWEREF99 TM (EPSG:3006)
        2. Snap to NMD 10m grid boundaries (floor for west/south, ceil for east/north)
        3. Return projected bbox with ``"crs": "EPSG:3006"``

    This ensures all rasters fetched with the same WGS84 input coords land on
    identical 10m pixel grids, making Sentinel-2 and NMD pixel-aligned.

    Args:
        coords: WGS84 bounding box ``{"west": lon, "south": lat, "east": lon, "north": lat}``.

    Returns:
        EPSG:3006 bounding box dict with ``"crs"`` key, snapped to 10m boundaries.
        Example: ``{"west": 373820, "south": 6157630, "east": 380450, "north": 6168950, "crs": "EPSG:3006"}``
    """
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds

    w, s, e, n = transform_bounds(
        CRS.from_epsg(4326), CRS.from_epsg(3006),
        coords["west"], coords["south"], coords["east"], coords["north"],
    )

    grid = NMD_GRID_SIZE
    return {
        "west": math.floor(w / grid) * grid,
        "south": math.floor(s / grid) * grid,
        "east": math.ceil(e / grid) * grid,
        "north": math.ceil(n / grid) * grid,
        "crs": TARGET_CRS,
    }


# ── Exceptions ───────────────────────────────────────────────────────────────

class FetchError(Exception):
    """Raised when DES data fetching fails."""
    pass


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class FetchResult:
    """Result of a DES data fetch.

    Attributes:
        bands: Band dict with uppercase keys, reflectance [0, 1].
               Example: {"B02": arr, "B03": arr, "B04": arr, "B08": arr, "B11": arr}
        scl: Scene Classification Layer array (uint8, values 0-11), or None.
        cloud_fraction: Fraction of pixels classified as cloud by SCL (0.0-1.0).
        rgb: RGB composite (H, W, 3) float32 [0, 1], percentile-stretched.
        geo: GeoContext with CRS, transform, and projected bounds.
        crs: Coordinate reference system from the GeoTIFF (e.g. "EPSG:3006").
             Deprecated — use geo.crs instead.
        transform: Affine transform from the GeoTIFF.
                   Deprecated — use geo.transform instead.
    """
    bands: dict[str, np.ndarray]
    scl: np.ndarray | None
    cloud_fraction: float
    rgb: np.ndarray
    geo: GeoContext | None = None
    crs: Any | None = None
    transform: Any | None = None


# ── Cloud detection ──────────────────────────────────────────────────────────

def check_cloud_fraction(scl: np.ndarray) -> float:
    """Compute cloud fraction from a Scene Classification Layer array.

    SCL classes counted as cloud:
        8 = cloud_medium_probability
        9 = cloud_high_probability
        10 = thin_cirrus

    Args:
        scl: 2D array (H, W) with SCL class values (0-11).

    Returns:
        Fraction of pixels that are cloud (0.0 to 1.0).
    """
    cloud_mask = np.isin(scl, list(SCL_CLOUD_CLASSES))
    return float(cloud_mask.sum()) / max(scl.size, 1)


# ── DES connection ───────────────────────────────────────────────────────────

def _connect(token: str | None = None, token_path: str | None = None):
    """Connect and authenticate to DES.

    Authentication priority:
        1. Explicit token argument
        2. DES_TOKEN environment variable
        3. Stored refresh token (from ``des_login.py --device``, auto-renews)
        4. Token file at token_path (default: .des_token in project root)

    Returns:
        Authenticated openeo.Connection.

    Raises:
        FetchError: If authentication fails.
    """
    try:
        import openeo
    except ImportError:
        raise ImportError(
            "openeo is required for DES data fetching. "
            "Install with: pip install openeo"
        )

    try:
        conn = openeo.connect(OPENEO_URL)
    except Exception as e:
        raise FetchError(f"Failed to connect to {OPENEO_URL}: {e}")

    # 1. Explicit token
    if token:
        conn.authenticate_oidc_access_token(access_token=token, provider_id="egi")
        return conn

    # 2. Environment variable (used in Docker/ColonyOS)
    env_token = os.environ.get("DES_TOKEN")
    if env_token:
        conn.authenticate_oidc_access_token(access_token=env_token, provider_id="egi")
        return conn

    # 3. Stored refresh token (from des_login.py --device)
    #    This auto-renews expired access tokens — best for local dev.
    try:
        conn.authenticate_oidc_refresh_token(
            provider_id="egi",
            store_refresh_token=True,
        )
        return conn
    except Exception:
        pass  # No stored refresh token, try next method

    # 4. Token file (short-lived access token from Web Editor)
    resolved_path = token_path or TOKEN_PATH_DEFAULT
    if os.path.exists(resolved_path):
        with open(resolved_path) as f:
            file_token = f.read().strip()
        if file_token:
            conn.authenticate_oidc_access_token(
                access_token=file_token, provider_id="egi"
            )
            return conn

    raise FetchError(
        "No valid DES authentication found. Run:\n"
        "  python scripts/des_login.py --device   (recommended, persistent)\n"
        "  python scripts/des_login.py --token YOUR_TOKEN  (short-lived)"
    )


# ── Main fetch function ─────────────────────────────────────────────────────

def _fetch_scl(conn, projected_coords: dict, temporal: list, date_window: int):
    """Fetch only the SCL band from DES and compute cloud fraction.

    This is a lightweight request (~20x smaller than full spectral fetch)
    used to pre-screen scenes before downloading expensive spectral bands.

    Returns:
        Tuple of (scl_array, cloud_fraction, crs, transform).
    """
    import rasterio

    # Load SCL at native 20m, resample to 10m grid
    cube_ref = conn.load_collection(
        collection_id=COLLECTION,
        spatial_extent=projected_coords,
        temporal_extent=temporal,
        bands=["b02"],  # lightweight reference for grid alignment
    )
    cube_scl = conn.load_collection(
        collection_id=COLLECTION,
        spatial_extent=projected_coords,
        temporal_extent=temporal,
        bands=BANDS_20M_CATEGORICAL,
    )
    cube_scl = cube_scl.resample_cube_spatial(target=cube_ref, method="near")

    if date_window > 0:
        cube_scl = cube_scl.reduce_dimension(dimension="t", reducer="last")

    data = cube_scl.download(format="gtiff")
    if not data:
        raise FetchError("DES returned empty SCL data")

    with rasterio.open(io.BytesIO(data)) as src:
        raw = src.read()
        crs = src.crs
        transform = src.transform

    scl = raw[0].astype(np.uint8)
    cloud_fraction = check_cloud_fraction(scl)
    return scl, cloud_fraction, crs, transform


def fetch_des_data(
    date: str,
    coords: dict,
    cloud_threshold: float = 0.1,
    token: str | None = None,
    include_scl: bool = True,
    date_window: int = 0,
) -> FetchResult:
    """Fetch Sentinel-2 L2A data from DES via openEO.

    Uses a two-stage fetch strategy:
        1. Fetch SCL (Scene Classification Layer) first — lightweight
        2. Check cloud fraction against threshold
        3. Only fetch full spectral bands if cloud cover is acceptable

    This avoids downloading large spectral cubes for cloudy scenes.

    Args:
        date: ISO date string, e.g. "2022-06-15".
        coords: Bounding box dict with keys: west, south, east, north.
        cloud_threshold: Maximum cloud fraction (0.0–1.0). If SCL shows
                         more cloud than this, spectral bands are NOT fetched
                         and a FetchError is raised. Default: 0.1 (10%).
        token: Optional DES access token. Falls back to env/file/OIDC.
        include_scl: If True, pre-screen with SCL before fetching spectral
                     bands. If False, skip cloud check and fetch everything.
        date_window: Days before/after date to search for imagery.
                     0 = single day, 5 = ±5 days window. DES will pick
                     the most recent available acquisition.

    Returns:
        FetchResult with bands, rgb, cloud_fraction, etc.

    Raises:
        FetchError: If data fetching fails or cloud cover exceeds threshold.
        ImportError: If openeo is not installed.
    """
    import rasterio
    from datetime import datetime, timedelta

    conn = _connect(token=token)

    # Project WGS84 coords to EPSG:3006 snapped to NMD 10m grid
    projected_coords = _to_nmd_grid(coords)

    # Temporal extent: date ± window
    dt = datetime.strptime(date, "%Y-%m-%d")
    start = (dt - timedelta(days=date_window)).strftime("%Y-%m-%d")
    end = (dt + timedelta(days=max(date_window, 1))).strftime("%Y-%m-%d")
    temporal = [start, end]

    # ── Stage 1: Fetch SCL and check cloud cover ─────────────────────────
    scl = None
    cloud_fraction = 0.0

    if include_scl:
        try:
            print(f"    [SCL] Fetching cloud mask...")
            scl, cloud_fraction, scl_crs, scl_transform = _fetch_scl(
                conn, projected_coords, temporal, date_window
            )
            print(f"    [SCL] Cloud fraction: {cloud_fraction:.1%}")

            if cloud_fraction > cloud_threshold:
                raise FetchError(
                    f"Scene too cloudy: {cloud_fraction:.1%} cloud "
                    f"(threshold: {cloud_threshold:.0%}). "
                    f"Spectral bands not downloaded. "
                    f"Try a different date or wider date_window."
                )
        except FetchError:
            raise
        except Exception as e:
            raise FetchError(f"SCL fetch failed for {date}: {e}")

    # ── Stage 2: Fetch spectral bands (scene is clear enough) ────────────
    try:
        print(f"    [Spectral] Fetching bands (scene passed cloud check)...")

        # Load 10m bands (EPSG:3006, snapped to NMD grid)
        cube_10m = conn.load_collection(
            collection_id=COLLECTION,
            spatial_extent=projected_coords,
            temporal_extent=temporal,
            bands=BANDS_10M,
        )

        # Load 20m spectral bands, resample to 10m with bilinear
        cube_20m = conn.load_collection(
            collection_id=COLLECTION,
            spatial_extent=projected_coords,
            temporal_extent=temporal,
            bands=BANDS_20M_SPECTRAL,
        )
        cube_20m = cube_20m.resample_cube_spatial(
            target=cube_10m, method="bilinear"
        )

        # Load 60m bands (B09), resample to 10m
        cube_60m = conn.load_collection(
            collection_id=COLLECTION,
            spatial_extent=projected_coords,
            temporal_extent=temporal,
            bands=BANDS_60M,
        )
        cube_60m = cube_60m.resample_cube_spatial(
            target=cube_10m, method="bilinear"
        )

        # Merge spectral bands
        cube = cube_10m.merge_cubes(cube_20m).merge_cubes(cube_60m)

        # If searching a date window, reduce temporal axis to get
        # the most recent pixel values (last available observation)
        if date_window > 0:
            cube = cube.reduce_dimension(
                dimension="t", reducer="last"
            )

        # Download as GeoTIFF
        data = cube.download(format="gtiff")

        if not data:
            raise FetchError(f"DES returned empty data for {date}")

    except FetchError:
        raise
    except Exception as e:
        raise FetchError(f"DES fetch failed for {date}: {e}")

    # Parse GeoTIFF
    try:
        with rasterio.open(io.BytesIO(data)) as src:
            raw = src.read()  # (n_bands, H, W)
            crs = src.crs
            transform = src.transform
    except Exception as e:
        raise FetchError(f"Failed to parse GeoTIFF for {date}: {e}")

    # Split into individual bands (order follows merge order)
    # 10m: b02=0, b03=1, b04=2, b08=3
    # 20m spectral: b05=4, b06=5, b07=6, b8a=7, b11=8, b12=9
    # 60m: b09=10
    spectral_names = BANDS_10M + BANDS_20M_SPECTRAL + BANDS_60M

    # Convert spectral bands: DN → reflectance
    des_bands = {}
    for i, band_name in enumerate(spectral_names):
        des_bands[band_name] = dn_to_reflectance(raw[i], source="des")

    # Map lowercase → uppercase band names
    imint_bands = des_to_imint_bands(des_bands)

    # Create RGB composite (mask clouds from stretch if SCL available)
    rgb = bands_to_rgb(imint_bands, scl=scl)

    # Build GeoContext — links this raster to the NMD 10m grid
    geo = GeoContext(
        crs=str(crs),
        transform=transform,
        bounds_projected={k: v for k, v in projected_coords.items() if k != "crs"},
        bounds_wgs84=coords,
        shape=rgb.shape[:2],
    )

    return FetchResult(
        bands=imint_bands,
        scl=scl,
        cloud_fraction=cloud_fraction,
        rgb=rgb,
        geo=geo,
        crs=crs,
        transform=transform,
    )


# ── NMD (Nationellt Marktäckedata) ────────────────────────────────────────────

# Default cache directory (project root/.nmd_cache)
NMD_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".nmd_cache"
)


@dataclass
class NMDFetchResult:
    """Result of an NMD data fetch.

    Attributes:
        nmd_raster: 2D uint8 array (H, W) with NMD class codes.
        crs: Coordinate reference system from the GeoTIFF.
        transform: Affine transform from the GeoTIFF.
        from_cache: True if loaded from local cache.
    """
    nmd_raster: np.ndarray
    crs: Any | None = None
    transform: Any | None = None
    from_cache: bool = False


def _nmd_cache_key(coords: dict) -> str:
    """Generate a deterministic cache key from bounding box coordinates.

    Args:
        coords: Bounding box dict with keys: west, south, east, north.

    Returns:
        Hex string suitable for use as a filename.
    """
    key_str = f"{coords['west']:.6f}_{coords['south']:.6f}_{coords['east']:.6f}_{coords['north']:.6f}"
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def _resample_nearest(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Resample a 2D categorical array to target_shape using nearest-neighbor.

    Uses scipy.ndimage.zoom with order=0 (nearest) to preserve category values.

    Args:
        arr: Input 2D array (H, W).
        target_shape: Desired (H, W).

    Returns:
        Resampled array with same dtype as input.
    """
    if arr.shape == target_shape:
        return arr

    from scipy.ndimage import zoom

    zoom_factors = (target_shape[0] / arr.shape[0], target_shape[1] / arr.shape[1])
    return zoom(arr, zoom_factors, order=0, mode="nearest").astype(arr.dtype)


def fetch_nmd_data(
    coords: dict,
    target_shape: tuple | None = None,
    token: str | None = None,
    cache_dir: str | None = None,
) -> NMDFetchResult:
    """Fetch NMD (Nationellt Marktäckedata) from DES via openEO.

    NMD is a static 10m land cover dataset — safe to cache locally.
    Coords are projected to EPSG:3006 and snapped to the NMD 10m grid
    before fetching, ensuring pixel-perfect alignment with Sentinel-2.

    The cache is keyed by the snapped EPSG:3006 coordinates for stability.

    Args:
        coords: WGS84 bounding box dict with keys: west, south, east, north.
        target_shape: Optional (H, W) to resample the NMD raster to.
                      Uses nearest-neighbor to preserve category values.
        token: Optional DES access token. Falls back to env/file/OIDC.
        cache_dir: Directory for the NMD cache. Defaults to .nmd_cache/.

    Returns:
        NMDFetchResult with nmd_raster (uint8 class codes).

    Raises:
        FetchError: If NMD data fetching fails.
    """
    import rasterio

    # Project and snap to NMD grid — same grid as Sentinel-2
    projected_coords = _to_nmd_grid(coords)

    resolved_cache = cache_dir or NMD_CACHE_DIR
    # Cache key based on snapped EPSG:3006 coords (deterministic, no float drift)
    cache_key = _nmd_cache_key(projected_coords)
    cache_path = os.path.join(resolved_cache, f"nmd_{cache_key}.npy")

    # Check cache first — NMD is static, safe to cache indefinitely
    if os.path.exists(cache_path):
        nmd_raster = np.load(cache_path)
        result_raster = nmd_raster
        if target_shape and nmd_raster.shape != target_shape:
            result_raster = _resample_nearest(nmd_raster, target_shape)
        return NMDFetchResult(
            nmd_raster=result_raster,
            from_cache=True,
        )

    # Fetch from DES using projected, snapped coords
    conn = _connect(token=token)

    try:
        load_kwargs = dict(
            collection_id=NMD_COLLECTION,
            spatial_extent=projected_coords,
            bands=[NMD_BAND],
        )
        if NMD_TEMPORAL is not None:
            load_kwargs["temporal_extent"] = NMD_TEMPORAL

        cube = conn.load_collection(**load_kwargs)

        # Reduce temporal dimension if present (NMD has one time step
        # but openEO may still return a temporal axis)
        try:
            cube = cube.reduce_dimension(dimension="t", reducer="first")
        except Exception:
            pass  # No temporal dimension — that's fine for static NMD

        data = cube.download(format="gtiff")

        if not data:
            raise FetchError("DES returned empty NMD data")

    except FetchError:
        raise
    except Exception as e:
        raise FetchError(f"NMD fetch failed: {e}")

    # Parse GeoTIFF
    try:
        with rasterio.open(io.BytesIO(data)) as src:
            raw = src.read(1)  # Single band
            crs = src.crs
            transform = src.transform
    except Exception as e:
        raise FetchError(f"Failed to parse NMD GeoTIFF: {e}")

    nmd_raster = raw.astype(np.uint8)

    # Cache the raw (unresampled) raster
    os.makedirs(resolved_cache, exist_ok=True)
    np.save(cache_path, nmd_raster)

    # Resample to target if needed
    result_raster = nmd_raster
    if target_shape and nmd_raster.shape != target_shape:
        result_raster = _resample_nearest(nmd_raster, target_shape)

    return NMDFetchResult(
        nmd_raster=result_raster,
        crs=crs,
        transform=transform,
        from_cache=False,
    )
