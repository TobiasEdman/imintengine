"""
imint/fetch.py — DES/openEO data fetching and cloud detection

Fetches Sentinel-2 L2A data from Digital Earth Sweden (DES) via openEO,
handles multi-resolution bands, converts DN to reflectance, and checks
cloud cover using the SCL (Scene Classification Layer) band.

Usage:
    from imint.fetch import fetch_des_data, FetchError

    result = fetch_des_data(date="2022-06-15", coords={...})
    if result.cloud_fraction < 0.3:
        # Use result.rgb and result.bands for analysis
        ...
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from .utils import dn_to_reflectance, des_to_imint_bands, bands_to_rgb

# ── Constants ────────────────────────────────────────────────────────────────

OPENEO_URL = "https://openeo.digitalearth.se"
COLLECTION = "s2_msi_l2a"

# DES band groupings by native resolution
BANDS_10M = ["b02", "b03", "b04", "b08"]
BANDS_20M_SPECTRAL = ["b11"]
BANDS_20M_CATEGORICAL = ["scl"]

# SCL cloud classes (Sentinel-2 L2A Scene Classification)
# 8 = cloud_medium_probability, 9 = cloud_high_probability, 10 = thin_cirrus
SCL_CLOUD_CLASSES = frozenset({8, 9, 10})

# Default token file location (project root)
TOKEN_PATH_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".des_token"
)


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
        crs: Coordinate reference system from the GeoTIFF (e.g. "EPSG:3006").
        transform: Affine transform from the GeoTIFF.
    """
    bands: dict[str, np.ndarray]
    scl: np.ndarray | None
    cloud_fraction: float
    rgb: np.ndarray
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
        3. Token file at token_path (default: .des_token in project root)
        4. Cached OIDC session (interactive device flow)

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

    # 3. Token file
    resolved_path = token_path or TOKEN_PATH_DEFAULT
    if os.path.exists(resolved_path):
        with open(resolved_path) as f:
            file_token = f.read().strip()
        if file_token:
            conn.authenticate_oidc_access_token(
                access_token=file_token, provider_id="egi"
            )
            return conn

    # 4. Cached OIDC (interactive)
    try:
        conn.authenticate_oidc(provider_id="egi")
        return conn
    except Exception as e:
        raise FetchError(
            f"No valid DES authentication found. "
            f"Set DES_TOKEN env var or run: python scripts/des_login.py --token YOUR_TOKEN. "
            f"Error: {e}"
        )


# ── Main fetch function ─────────────────────────────────────────────────────

def fetch_des_data(
    date: str,
    coords: dict,
    cloud_threshold: float = 0.3,
    token: str | None = None,
    include_scl: bool = True,
) -> FetchResult:
    """Fetch Sentinel-2 L2A data from DES via openEO.

    Loads 10m bands (B02, B03, B04, B08) and 20m bands (B11), resamples
    to a common 10m grid, converts DN to reflectance, and optionally
    checks cloud cover using the SCL band.

    Args:
        date: ISO date string, e.g. "2022-06-15".
        coords: Bounding box dict with keys: west, south, east, north.
        cloud_threshold: Cloud fraction threshold (not used for filtering,
                         just returned in the result for the caller to decide).
        token: Optional DES access token. Falls back to env/file/OIDC.
        include_scl: If True, also fetch the SCL band for cloud detection.

    Returns:
        FetchResult with bands, rgb, cloud_fraction, etc.

    Raises:
        FetchError: If data fetching fails.
        ImportError: If openeo is not installed.
    """
    import rasterio
    from datetime import datetime, timedelta

    conn = _connect(token=token)

    # Temporal extent: single day
    dt = datetime.strptime(date, "%Y-%m-%d")
    temporal = [date, (dt + timedelta(days=1)).strftime("%Y-%m-%d")]

    try:
        # Load 10m bands
        cube_10m = conn.load_collection(
            collection_id=COLLECTION,
            spatial_extent=coords,
            temporal_extent=temporal,
            bands=BANDS_10M,
        )

        # Load 20m spectral bands, resample to 10m with bilinear
        cube_20m = conn.load_collection(
            collection_id=COLLECTION,
            spatial_extent=coords,
            temporal_extent=temporal,
            bands=BANDS_20M_SPECTRAL,
        )
        cube_20m = cube_20m.resample_cube_spatial(
            target=cube_10m, method="bilinear"
        )

        # Merge spectral bands
        cube = cube_10m.merge_cubes(cube_20m)

        # Optionally load SCL for cloud detection
        cube_scl = None
        if include_scl:
            cube_scl = conn.load_collection(
                collection_id=COLLECTION,
                spatial_extent=coords,
                temporal_extent=temporal,
                bands=BANDS_20M_CATEGORICAL,
            )
            cube_scl = cube_scl.resample_cube_spatial(
                target=cube_10m, method="near"
            )
            cube = cube.merge_cubes(cube_scl)

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
    # 20m spectral: b11=4
    # 20m categorical: scl=5 (if included)
    n_spectral = len(BANDS_10M) + len(BANDS_20M_SPECTRAL)
    spectral_names = BANDS_10M + BANDS_20M_SPECTRAL

    # Convert spectral bands: DN → reflectance
    des_bands = {}
    for i, band_name in enumerate(spectral_names):
        des_bands[band_name] = dn_to_reflectance(raw[i], source="des")

    # Map lowercase → uppercase band names
    imint_bands = des_to_imint_bands(des_bands)

    # Extract SCL and compute cloud fraction
    scl = None
    cloud_fraction = 0.0
    if include_scl and raw.shape[0] > n_spectral:
        scl = raw[n_spectral].astype(np.uint8)
        cloud_fraction = check_cloud_fraction(scl)

    # Create RGB composite
    rgb = bands_to_rgb(imint_bands)

    return FetchResult(
        bands=imint_bands,
        scl=scl,
        cloud_fraction=cloud_fraction,
        rgb=rgb,
        crs=crs,
        transform=transform,
    )
