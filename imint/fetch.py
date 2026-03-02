"""
imint/fetch.py — DES/openEO data fetching and cloud detection

Fetches Sentinel-2 L2A data from Digital Earth Sweden (DES) via openEO,
handles multi-resolution bands, converts DN to reflectance, and checks
cloud cover using the SCL (Scene Classification Layer) band.

Also provides NMD (Nationellt Marktäckedata) fetching for LULC analysis.
NMD is a static 10m land cover dataset from Naturvårdsverket.

Sjökort (nautical chart) data can be fetched from Sjöfartsverket via
SLU GET (Geodata Extraction Tool). This provides S-57 vector data for
Swedish coastal and inland waters.

Usage:
    from imint.fetch import fetch_des_data, fetch_nmd_data, FetchError
    from imint.fetch import fetch_sjokort_data
    from imint.fetch import fetch_grazing_timeseries

    result = fetch_des_data(date="2022-06-15", coords={...})
    if result.cloud_fraction < 0.3:
        # Use result.rgb and result.bands for analysis
        ...

    nmd = fetch_nmd_data(coords={...}, target_shape=(H, W))
    # nmd.nmd_raster is uint8 with NMD class codes

    sjokort = fetch_sjokort_data(coords={...}, email="user@org.se", session=shibboleth_session)
    # sjokort.s57_paths contains downloaded S-57 (.000) files

    # Or with pre-downloaded S-57 data and rendering:
    sjokort = fetch_sjokort_data(
        coords={...}, s57_dir="path/to/s57/",
        render=True, output_path="sjokort.png",
    )
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
BANDS_60M_ALL = ["b01", "b09"]  # Includes B01 (coastal aerosol) for grazing model
BANDS_20M_CATEGORICAL = ["scl"]

# 12-band order expected by pib-ml-grazing CNN-LSTM model (all S2 except B10)
GRAZING_BAND_ORDER = [
    "b01", "b02", "b03", "b04", "b05", "b06",
    "b07", "b08", "b8a", "b09", "b11", "b12",
]
# Reorder indices: download order (10m+20m+60m_all) → GRAZING_BAND_ORDER
# Download: b02(0) b03(1) b04(2) b08(3) b05(4) b06(5) b07(6) b8a(7) b11(8) b12(9) b01(10) b09(11)
# Target:   b01(0) b02(1) b03(2) b04(3) b05(4) b06(5) b07(6) b08(7) b8a(8) b09(9) b11(10) b12(11)
_GRAZING_REORDER = [10, 0, 1, 2, 4, 5, 6, 3, 7, 11, 8, 9]

# SCL cloud + shadow classes (Sentinel-2 L2A Scene Classification)
# 3 = cloud_shadow, 8 = cloud_medium_probability,
# 9 = cloud_high_probability, 10 = thin_cirrus
SCL_CLOUD_CLASSES = frozenset({3, 8, 9, 10})

# ── LPIS (Jordbruksverket) ────────────────────────────────────────────────
LPIS_WFS_URL = "http://epub.sjv.se/inspire/inspire/wfs"
LPIS_LAYER = "inspire:senaste_arslager_block"
LPIS_PASTURE_AGOSLAG = "Bete"

# Default token file location (project root)
TOKEN_PATH_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".des_token"
)


# ── Grid alignment ───────────────────────────────────────────────────────────

def _snap_to_target_grid(
    raw: np.ndarray,
    src_transform,
    src_crs,
    target_bounds: dict,
    pixel_size: int = 10,
) -> tuple[np.ndarray, "Affine"]:
    """Reproject/snap a raster to the exact target pixel grid.

    The DES/openEO server may return Sentinel-2 data on a pixel grid that
    is offset from the requested bounds due to the native tile alignment
    and reprojection pipeline. This function ensures the output is
    pixel-aligned to our canonical NMD 10m grid.

    Uses a Fourier phase-shift for sub-pixel corrections (sinc
    interpolation) when the offset is fractional, and simple array
    slicing for integer-pixel offsets. Falls back to rasterio warp
    for CRS mismatches.

    Args:
        raw:            (n_bands, H, W) array from rasterio.
        src_transform:  Affine transform of the downloaded GeoTIFF.
        src_crs:        CRS of the downloaded GeoTIFF.
        target_bounds:  Dict with west/south/east/north in EPSG:3006.
        pixel_size:     Grid cell size (default 10m).

    Returns:
        (aligned_raw, target_transform) — the raster snapped to the
        target grid, and the corresponding Affine transform.
    """
    from rasterio.transform import from_origin
    from rasterio.crs import CRS

    target_w = target_bounds["west"]
    target_n = target_bounds["north"]
    target_width = int((target_bounds["east"] - target_bounds["west"]) / pixel_size)
    target_height = int((target_bounds["north"] - target_bounds["south"]) / pixel_size)
    target_transform = from_origin(target_w, target_n, pixel_size, pixel_size)

    # Check if the source transform already matches the target
    src_x0 = src_transform.c
    src_y0 = src_transform.f
    dx_m = src_x0 - target_w
    dy_m = target_n - src_y0

    # Convert offset to pixels
    dx_px = dx_m / pixel_size
    dy_px = dy_m / pixel_size

    # If essentially zero offset and same dimensions, no correction needed
    if (abs(dx_px) < 0.01 and abs(dy_px) < 0.01
            and raw.shape[1] == target_height and raw.shape[2] == target_width):
        return raw, target_transform

    # Check CRS match
    if isinstance(src_crs, CRS):
        src_crs_obj = src_crs
    elif isinstance(src_crs, str):
        src_crs_obj = CRS.from_user_input(src_crs)
    else:
        src_crs_obj = CRS(src_crs)
    target_crs = CRS.from_epsg(3006)
    if src_crs_obj != target_crs:
        # Full reprojection needed (different CRS)
        from rasterio.warp import reproject, Resampling
        n_bands = raw.shape[0]
        aligned = np.zeros((n_bands, target_height, target_width), dtype=raw.dtype)
        for b in range(n_bands):
            reproject(
                source=raw[b],
                destination=aligned[b],
                src_transform=src_transform,
                src_crs=src_crs_obj,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear,
            )
        print(f"    [grid] Reprojected from {src_crs_obj} to target grid "
              f"(offset was {dx_px:.2f}, {dy_px:.2f} px)")
        return aligned, target_transform

    # Same CRS — handle pixel offset (integer + sub-pixel)
    int_dx = int(round(dx_px))
    int_dy = int(round(dy_px))
    frac_dx = dx_px - int_dx
    frac_dy = dy_px - int_dy

    n_bands, src_h, src_w = raw.shape
    aligned = np.zeros((n_bands, target_height, target_width), dtype=raw.dtype)

    # Compute overlap region
    # Source pixel (sr, sc) maps to target pixel (sr + int_dy, sc + int_dx)
    tgt_r0 = max(0, int_dy)
    tgt_c0 = max(0, int_dx)
    src_r0 = max(0, -int_dy)
    src_c0 = max(0, -int_dx)
    copy_h = min(target_height - tgt_r0, src_h - src_r0)
    copy_w = min(target_width - tgt_c0, src_w - src_c0)

    if copy_h > 0 and copy_w > 0:
        aligned[:, tgt_r0:tgt_r0 + copy_h, tgt_c0:tgt_c0 + copy_w] = \
            raw[:, src_r0:src_r0 + copy_h, src_c0:src_c0 + copy_w]

    # Apply sub-pixel correction if needed (uses shared coregistration module)
    if abs(frac_dx) > 0.01 or abs(frac_dy) > 0.01:
        from .coregistration import subpixel_shift
        for b in range(n_bands):
            aligned[b] = subpixel_shift(
                aligned[b].astype(np.float64), frac_dy, frac_dx
            ).astype(aligned.dtype)
        print(f"    [grid] Applied sub-pixel correction: "
              f"dy={frac_dy:+.3f} dx={frac_dx:+.3f} px "
              f"({frac_dy*pixel_size:+.1f}m / {frac_dx*pixel_size:+.1f}m)")

    if int_dx != 0 or int_dy != 0:
        print(f"    [grid] Snapped to target grid: "
              f"integer offset {int_dy},{int_dx} px "
              f"({int_dy*pixel_size}m, {int_dx*pixel_size}m)")

    return aligned, target_transform


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


@dataclass
class GrazingTimeseriesResult:
    """Result of a grazing timeseries fetch for one polygon.

    Attributes:
        data: (T, 12, H, W) float32 reflectance array.
              Band order: B01, B02, B03, B04, B05, B06, B07, B08,
                          B8A, B09, B11, B12.
        dates: List of ISO date strings for the T cloud-free timesteps.
        cloud_fractions: Per-date cloud fraction within the polygon.
        polygon_id: Identifier for this polygon (index or name).
        geo: GeoContext with CRS, transform, and projected bounds.
        shape_hw: Spatial dimensions (H, W) of each timestep.
    """
    data: np.ndarray             # (T, 12, H, W) float32
    dates: list[str]
    cloud_fractions: list[float]
    polygon_id: str | int
    geo: GeoContext
    shape_hw: tuple[int, int]


# ── Cloud detection ──────────────────────────────────────────────────────────

def check_cloud_fraction(scl: np.ndarray) -> float:
    """Compute cloud + shadow fraction from a Scene Classification Layer array.

    SCL classes counted as cloud/shadow:
        3 = cloud_shadow
        8 = cloud_medium_probability
        9 = cloud_high_probability
        10 = thin_cirrus

    Args:
        scl: 2D array (H, W) with SCL class values (0-11).

    Returns:
        Fraction of pixels that are cloud or shadow (0.0 to 1.0).
    """
    cloud_mask = np.isin(scl, list(SCL_CLOUD_CLASSES))
    return float(cloud_mask.sum()) / max(scl.size, 1)


def _polygon_cloud_fraction(
    scl: np.ndarray,
    polygon_mask: np.ndarray,
) -> float:
    """Compute cloud fraction ONLY within a polygon mask.

    Unlike :func:`check_cloud_fraction` which checks the full bounding
    box, this function restricts the count to pixels inside *polygon_mask*.

    Args:
        scl: 2D array (H, W) with SCL class values (0-11).
        polygon_mask: 2D boolean array (H, W), True inside polygon.

    Returns:
        Fraction of polygon pixels that are cloud/shadow (0.0 to 1.0).
        Returns 1.0 if polygon_mask has zero True pixels.
    """
    n_polygon = int(polygon_mask.sum())
    if n_polygon == 0:
        return 1.0
    cloud_mask = np.isin(scl, list(SCL_CLOUD_CLASSES))
    cloud_in_polygon = int((cloud_mask & polygon_mask).sum())
    return cloud_in_polygon / n_polygon


def _rasterize_polygon(
    polygon_geom,
    projected_bounds: dict,
    pixel_size: int = 10,
) -> np.ndarray:
    """Rasterize a shapely polygon to a boolean mask on the NMD 10m grid.

    Args:
        polygon_geom: Shapely geometry in EPSG:3006.
        projected_bounds: Dict with west/south/east/north in EPSG:3006.
        pixel_size: Grid cell size (default 10m).

    Returns:
        Boolean array (H, W) where True = inside polygon.
    """
    from rasterio.transform import from_origin
    from rasterio.features import rasterize as rio_rasterize

    width = int((projected_bounds["east"] - projected_bounds["west"]) / pixel_size)
    height = int((projected_bounds["north"] - projected_bounds["south"]) / pixel_size)
    transform = from_origin(
        projected_bounds["west"], projected_bounds["north"],
        pixel_size, pixel_size,
    )
    mask = rio_rasterize(
        [(polygon_geom, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    return mask.astype(bool)


def _polygon_to_projected_bbox(
    polygon_geom,
    src_crs: str = "EPSG:4326",
    buffer_m: float = 50.0,
) -> dict:
    """Convert a polygon geometry to a projected EPSG:3006 bbox on the NMD grid.

    Computes the bounding box of the polygon, adds a buffer, and snaps
    to the 10m NMD grid.

    Args:
        polygon_geom: Shapely geometry (in *src_crs*).
        src_crs: CRS of the input geometry.  Default: EPSG:4326.
        buffer_m: Buffer in metres around the polygon bbox (default 50m =
                  5 pixels at 10m, giving the model context around edges).

    Returns:
        EPSG:3006 bbox dict snapped to NMD grid, with ``"crs"`` key.
    """
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds

    minx, miny, maxx, maxy = polygon_geom.bounds

    if src_crs != TARGET_CRS:
        w, s, e, n = transform_bounds(
            CRS.from_user_input(src_crs), CRS.from_epsg(3006),
            minx, miny, maxx, maxy,
        )
    else:
        w, s, e, n = minx, miny, maxx, maxy

    # Add buffer
    w -= buffer_m
    s -= buffer_m
    e += buffer_m
    n += buffer_m

    # Snap to NMD grid
    grid = NMD_GRID_SIZE
    return {
        "west": math.floor(w / grid) * grid,
        "south": math.floor(s / grid) * grid,
        "east": math.ceil(e / grid) * grid,
        "north": math.ceil(n / grid) * grid,
        "crs": TARGET_CRS,
    }


# ── DES connection ───────────────────────────────────────────────────────────

def _connect(token: str | None = None, token_path: str | None = None):
    """Connect and authenticate to DES.

    Authentication priority:
        1. Explicit token argument
        2. DES_TOKEN environment variable
        3. DES_USER + DES_PASSWORD environment variables (Basic Auth)
        4. Stored refresh token (from ``des_login.py --device``, auto-renews)
        5. Token file at token_path (default: .des_token in project root)

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

    # Basic Auth — DES community tutorial credentials
    des_user = os.environ.get("DES_USER", "testuser")
    des_password = os.environ.get("DES_PASSWORD", "secretpassword")
    conn.authenticate_basic(username=des_user, password=des_password)
    return conn


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


def _fetch_scl_batch(
    conn,
    projected_coords: dict,
    candidate_dates: list[str] | None = None,
    *,
    temporal: list[str] | None = None,
) -> list[tuple[str, float]]:
    """Fetch SCL for multiple dates in one DES call, return per-date cloud fractions.

    DES returns multi-date downloads as a ``.tar.gz`` archive containing
    one GeoTIFF per date (filename pattern ``out_YYYY_MM_DDT...tif``).
    This function extracts each TIF, parses the date from the filename,
    and computes ``check_cloud_fraction`` locally.

    Supports two modes:

    * **candidate_dates** — temporal extent is derived from the list;
      only these dates are returned.
    * **temporal** — explicit ``[start, end)`` range; ALL dates found
      in the archive are returned (sorted by date).  This is the
      "skip STAC" mode where DES discovers available dates directly.

    Args:
        conn: openEO connection.
        projected_coords: EPSG:3006 bbox dict with ``"crs"`` key.
        candidate_dates: Optional list of date strings to filter to.
        temporal: Optional ``[start, end)`` ISO date pair.  If given,
            *candidate_dates* is ignored and all dates are returned.

    Returns:
        List of ``(date_str, cloud_fraction)`` sorted by date or in
        *candidate_dates* order.

    Raises:
        FetchError: If DES returns empty/unparseable data.
    """
    import re
    import gzip
    import tarfile
    import rasterio
    from datetime import datetime as _dt, timedelta as _td

    # Determine temporal extent
    if temporal is not None:
        _temporal = temporal
        cand_set = None  # accept all dates
    elif candidate_dates:
        sorted_cands = sorted(candidate_dates)
        dt_end = _dt.strptime(sorted_cands[-1], "%Y-%m-%d") + _td(days=1)
        _temporal = [sorted_cands[0], dt_end.strftime("%Y-%m-%d")]
        cand_set = set(candidate_dates)
    else:
        return []

    # Load SCL, resample to 10m grid
    cube_ref = conn.load_collection(
        collection_id=COLLECTION,
        spatial_extent=projected_coords,
        temporal_extent=_temporal,
        bands=["b02"],
    )
    cube_scl = conn.load_collection(
        collection_id=COLLECTION,
        spatial_extent=projected_coords,
        temporal_extent=_temporal,
        bands=BANDS_20M_CATEGORICAL,
    )
    cube_scl = cube_scl.resample_cube_spatial(target=cube_ref, method="near")

    data = cube_scl.download(format="gtiff")
    if not data:
        raise FetchError("DES returned empty SCL batch data")

    # DES returns tar.gz for multi-date results
    cloud_by_date: dict[str, float] = {}
    _DATE_RE = re.compile(r"out_(\d{4})_(\d{2})_(\d{2})T")

    if isinstance(data, bytes) and data[:2] == b"\x1f\x8b":
        # tar.gz archive — extract TIFs, parse dates from filenames
        tf = tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")
        for member in tf.getmembers():
            if not member.name.lower().endswith((".tif", ".tiff")):
                continue
            m = _DATE_RE.search(member.name)
            if not m:
                continue
            date_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
            if cand_set is not None and date_str not in cand_set:
                continue  # extra date in range, skip
            f = tf.extractfile(member)
            if f is None:
                continue
            with rasterio.open(io.BytesIO(f.read())) as src:
                scl = src.read()[0].astype(np.uint8)
            cloud_by_date[date_str] = check_cloud_fraction(scl)
        tf.close()
    else:
        raise FetchError(
            f"SCL batch: unexpected format (magic={data[:4].hex()})"
        )

    if not cloud_by_date:
        raise FetchError("SCL batch: no matching dates in archive")

    # Return in candidate_dates order, or sorted by date
    if candidate_dates is not None and cand_set is not None:
        return [(d, cloud_by_date[d]) for d in candidate_dates if d in cloud_by_date]
    return sorted(cloud_by_date.items())


def fetch_des_data(
    date: str,
    coords: dict,
    cloud_threshold: float = 0.1,
    token: str | None = None,
    include_scl: bool = True,
    date_window: int = 0,
) -> FetchResult:
    """Fetch Sentinel-2 L2A data from DES via openEO.

    Fetches all spectral bands + SCL in a single request. Cloud fraction
    is computed *after* download from the SCL band and compared against
    the threshold.  If the scene exceeds the threshold a ``FetchError``
    is raised.

    Args:
        date: ISO date string, e.g. "2022-06-15".
        coords: Bounding box dict with keys: west, south, east, north.
        cloud_threshold: Maximum cloud fraction (0.0–1.0). Set to 1.0
                         to accept any cloud cover. Default: 0.1 (10%).
        token: Optional DES access token. Falls back to env/file/OIDC.
        include_scl: If True, fetch SCL band for cloud stats and
                     visualization. If False, skip SCL entirely.
        date_window: Days before/after date to search for imagery.
                     0 = single day, 5 = ±5 days window.

    Returns:
        FetchResult with bands, rgb, cloud_fraction, etc.

    Raises:
        FetchError: If data fetching fails or cloud cover exceeds threshold.
        ImportError: If openeo is not installed.
    """
    import rasterio
    from datetime import datetime, timedelta

    # STAC-guided date selection when using a date window
    if date_window > 0:
        date = _stac_best_date(coords, date, date_window)
        date_window = 0  # fetch exact date

    conn = _connect(token=token)

    # Project WGS84 coords to EPSG:3006 snapped to NMD 10m grid
    projected_coords = _to_nmd_grid(coords)

    # Temporal extent: single date (STAC already selected the best)
    dt = datetime.strptime(date, "%Y-%m-%d")
    start = dt.strftime("%Y-%m-%d")
    end = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
    temporal = [start, end]

    # ── Fetch all bands in a single request ──────────────────────────────
    try:
        print(f"    Fetching spectral bands{' + SCL' if include_scl else ''}...")

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

        # Optionally include SCL (resampled to 10m with nearest-neighbor)
        if include_scl:
            cube_scl = conn.load_collection(
                collection_id=COLLECTION,
                spatial_extent=projected_coords,
                temporal_extent=temporal,
                bands=BANDS_20M_CATEGORICAL,
            )
            cube_scl = cube_scl.resample_cube_spatial(
                target=cube_10m, method="near"
            )
            cube = cube.merge_cubes(cube_scl)

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

    # Snap to target NMD grid — corrects pixel-level and sub-pixel offsets
    # that arise from Sentinel-2 tile alignment during server-side
    # reprojection.  Ensures all dates produce identical pixel grids.
    target_bounds = {k: v for k, v in projected_coords.items() if k != "crs"}
    raw, transform = _snap_to_target_grid(
        raw, transform, crs, target_bounds, pixel_size=NMD_GRID_SIZE,
    )
    crs = rasterio.crs.CRS.from_epsg(3006)

    # Split into individual bands (order follows merge order)
    # 10m: b02=0, b03=1, b04=2, b08=3
    # 20m spectral: b05=4, b06=5, b07=6, b8a=7, b11=8, b12=9
    # 60m: b09=10
    # SCL: 11 (if included)
    spectral_names = BANDS_10M + BANDS_20M_SPECTRAL + BANDS_60M
    n_spectral = len(spectral_names)

    # Convert spectral bands: DN → reflectance
    des_bands = {}
    for i, band_name in enumerate(spectral_names):
        des_bands[band_name] = dn_to_reflectance(raw[i], source="des")

    # Extract SCL if present (last band)
    scl = None
    cloud_fraction = 0.0
    if include_scl and raw.shape[0] > n_spectral:
        scl = raw[n_spectral].astype(np.uint8)
        cloud_fraction = check_cloud_fraction(scl)
        print(f"    Cloud fraction: {cloud_fraction:.1%}")

        if cloud_fraction > cloud_threshold:
            raise FetchError(
                f"Scene too cloudy: {cloud_fraction:.1%} cloud "
                f"(threshold: {cloud_threshold:.0%}). "
                f"Try a different date or wider date_window."
            )

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


# ── Seasonal / multitemporal fetching ────────────────────────────────────────


def fetch_seasonal_dates(
    coords: dict,
    seasonal_windows: list[tuple[int, int]],
    years: list[str],
    scene_cloud_max: float = 50.0,
) -> list[list[tuple[str, float]]]:
    """Find best available Sentinel-2 dates for each seasonal window.

    Queries STAC for each window across the given years and returns
    candidate dates sorted by cloud cover (ascending).

    Args:
        coords: WGS84 bounding box dict.
        seasonal_windows: List of (start_month, end_month) tuples defining
            each seasonal window, e.g. [(4,5), (6,7), (8,9), (1,2)].
        years: Years to search, e.g. ["2019", "2018"].
        scene_cloud_max: Max scene cloud % for STAC pre-filter.

    Returns:
        List of lists (one per window), each containing
        ``[(date_str, scene_cloud_pct), ...]`` sorted by cloud ascending.
        Empty list if no data found for that window.
    """
    results = []
    for m_start, m_end in seasonal_windows:
        window_candidates = []
        for year in years:
            date_start = f"{year}-{m_start:02d}-01"
            # End of the last month in the window
            if m_end in (1, 3, 5, 7, 8, 10, 12):
                date_end = f"{year}-{m_end:02d}-31"
            elif m_end in (4, 6, 9, 11):
                date_end = f"{year}-{m_end:02d}-30"
            else:  # February
                date_end = f"{year}-{m_end:02d}-28"

            try:
                dates = _stac_available_dates(
                    coords, date_start, date_end,
                    scene_cloud_max=scene_cloud_max,
                )
                window_candidates.extend(dates)
            except Exception:
                pass  # STAC timeout — skip this year/window

        # Sort all candidates across years by cloud ascending
        window_candidates.sort(key=lambda x: x[1])
        results.append(window_candidates)

    return results


def fetch_seasonal_image(
    date: str,
    coords: dict,
    prithvi_bands: list[str] | None = None,
) -> tuple[np.ndarray, str] | None:
    """Fetch a single Sentinel-2 scene for seasonal tile assembly.

    Uses the standard ``fetch_des_data`` pipeline (including NMD grid
    snapping and sub-pixel correction) but skips SCL and cloud threshold
    since STAC + SCL pre-screening has already been done.

    Args:
        date: ISO date string (already cloud-screened).
        coords: WGS84 bounding box dict.
        prithvi_bands: Band names for stacking, e.g. ["B02", ..., "B12"].
                       Defaults to the Prithvi 6-band set.

    Returns:
        Tuple of ``(image_array, date_str)`` where image_array is
        ``(6, H, W)`` float32 reflectance [0,1], or ``None`` if fetch
        fails (caller handles missing seasons).
    """
    if prithvi_bands is None:
        prithvi_bands = ["B02", "B03", "B04", "B8A", "B11", "B12"]

    try:
        result = fetch_des_data(
            date=date,
            coords=coords,
            cloud_threshold=1.0,    # already screened
            include_scl=False,      # skip redundant SCL
        )
    except FetchError:
        return None

    # Check all Prithvi bands present
    missing = [b for b in prithvi_bands if b not in result.bands]
    if missing:
        return None

    image = np.stack(
        [result.bands[b] for b in prithvi_bands], axis=0,
    ).astype(np.float32)

    return image, date


# ── Multi-temporal vessel heatmap ────────────────────────────────────────────


def _fetch_tci_bands(
    conn,
    projected_coords: dict,
    temporal: list,
) -> tuple[dict, np.ndarray, "GeoContext"]:
    """Fetch B02+B03+B04+SCL for vessel detection (lightweight, single date).

    Downloads only the three 10 m TCI bands and the SCL layer — roughly
    3× smaller than a full spectral fetch.  The TCI bands are needed by
    ``MarineVesselAnalyzer`` (L1C-TCI formula) and the SCL is used for
    water-mask filtering of false positives.

    Args:
        conn: Authenticated openEO connection.
        projected_coords: EPSG:3006 bbox dict with ``"crs"`` key.
        temporal: Two-element list ``[start_date, end_date]``.

    Returns:
        Tuple of ``(bands_dict, scl_array, geo)`` where *bands_dict*
        has uppercase keys ``{"B02", "B03", "B04"}`` as float32
        reflectance, *scl_array* is uint8, and *geo* is a GeoContext.
    """
    import rasterio
    from .utils import dn_to_reflectance, des_to_imint_bands

    tci_bands = ["b02", "b03", "b04"]

    cube_tci = conn.load_collection(
        collection_id=COLLECTION,
        spatial_extent=projected_coords,
        temporal_extent=temporal,
        bands=tci_bands,
    )
    cube_scl = conn.load_collection(
        collection_id=COLLECTION,
        spatial_extent=projected_coords,
        temporal_extent=temporal,
        bands=BANDS_20M_CATEGORICAL,
    )
    cube_scl = cube_scl.resample_cube_spatial(target=cube_tci, method="near")
    cube = cube_tci.merge_cubes(cube_scl)

    data = cube.download(format="gtiff")
    if not data:
        raise FetchError("DES returned empty TCI data")

    with rasterio.open(io.BytesIO(data)) as src:
        raw = src.read()  # (4, H, W): b02, b03, b04, scl
        crs = src.crs
        transform = src.transform

    # TCI bands → reflectance
    bands = {}
    for i, name in enumerate(tci_bands):
        bands[name] = dn_to_reflectance(raw[i], source="des")
    imint_bands = des_to_imint_bands(bands)

    # SCL (last band)
    scl = raw[len(tci_bands)].astype(np.uint8)

    # Strip "crs" key for bounds_projected
    proj_bounds = {k: v for k, v in projected_coords.items() if k != "crs"}

    geo = GeoContext(
        crs=str(crs),
        transform=transform,
        bounds_projected=proj_bounds,
        bounds_wgs84=None,  # not needed for vessel detection
        shape=(raw.shape[1], raw.shape[2]),
    )

    return imint_bands, scl, geo


def _fetch_ai2_bands(
    conn,
    projected_coords: dict,
    temporal: list,
) -> tuple[dict, np.ndarray, "GeoContext"]:
    """Fetch all 9 bands needed by the AI2 vessel detector, plus SCL.

    Downloads B02, B03, B04, B08 (10 m) and B05, B06, B07, B11, B12 (20 m)
    plus SCL.  The 20 m bands and SCL are resampled to 10 m to match the
    high-resolution bands.

    The returned bands dict contains **raw DN values** (not reflectance),
    as the AI2 model handles its own normalization (divide by 3000/8160).

    Args:
        conn: Authenticated openEO connection.
        projected_coords: EPSG:3006 bbox dict with ``"crs"`` key.
        temporal: Two-element list ``[start_date, end_date]``.

    Returns:
        Tuple of ``(bands_dict, scl_array, geo)`` where *bands_dict*
        has uppercase keys (B02, B03, B04, B05, B06, B07, B08, B11, B12)
        as float32 DN values, *scl_array* is uint8, and *geo* is a GeoContext.
    """
    import rasterio

    bands_10m = ["b02", "b03", "b04", "b08"]
    bands_20m = ["b05", "b06", "b07", "b11", "b12"]

    # Load 10 m bands (reference grid)
    cube_10m = conn.load_collection(
        collection_id=COLLECTION,
        spatial_extent=projected_coords,
        temporal_extent=temporal,
        bands=bands_10m,
    )

    # Load 20 m spectral bands and resample to 10 m grid
    cube_20m = conn.load_collection(
        collection_id=COLLECTION,
        spatial_extent=projected_coords,
        temporal_extent=temporal,
        bands=bands_20m,
    )
    cube_20m = cube_20m.resample_cube_spatial(target=cube_10m, method="bilinear")

    # Load SCL (20 m categorical) and resample with nearest-neighbour
    cube_scl = conn.load_collection(
        collection_id=COLLECTION,
        spatial_extent=projected_coords,
        temporal_extent=temporal,
        bands=BANDS_20M_CATEGORICAL,
    )
    cube_scl = cube_scl.resample_cube_spatial(target=cube_10m, method="near")

    # Merge all cubes and download
    cube = cube_10m.merge_cubes(cube_20m).merge_cubes(cube_scl)
    data = cube.download(format="gtiff")
    if not data:
        raise FetchError("DES returned empty AI2 band data")

    with rasterio.open(io.BytesIO(data)) as src:
        raw = src.read()  # (10, H, W): b02,b03,b04,b08, b05,b06,b07,b11,b12, scl
        crs = src.crs
        transform = src.transform

    all_band_names = bands_10m + bands_20m  # 9 bands
    bands = {}
    for i, name in enumerate(all_band_names):
        uname = name.upper()
        bands[uname] = raw[i].astype(np.float32)

    # SCL is the last band
    scl = raw[len(all_band_names)].astype(np.uint8)

    proj_bounds = {k: v for k, v in projected_coords.items() if k != "crs"}
    geo = GeoContext(
        crs=str(crs),
        transform=transform,
        bounds_projected=proj_bounds,
        bounds_wgs84=None,
        shape=(raw.shape[1], raw.shape[2]),
    )

    return bands, scl, geo


def _fetch_tci_scl_batch(
    conn,
    projected_coords: dict,
    temporal: list[str],
) -> list[tuple[str, dict, np.ndarray, "GeoContext"]]:
    """Fetch B02+B03+B04+SCL for ALL dates in a temporal range in one call.

    Returns per-date tuples so the caller can screen cloud and run
    vessel detection without additional DES calls.

    Args:
        conn: Authenticated openEO connection.
        projected_coords: EPSG:3006 bbox dict with ``"crs"`` key.
        temporal: ``[start, end)`` ISO date pair for the range.

    Returns:
        List of ``(date_str, bands_dict, scl, geo)`` sorted by date.
        *bands_dict* has uppercase keys ``{"B02", "B03", "B04"}`` as
        float32 reflectance.
    """
    import re
    import tarfile
    import rasterio
    from .utils import dn_to_reflectance, des_to_imint_bands

    tci_band_names = ["b02", "b03", "b04"]

    cube_tci = conn.load_collection(
        collection_id=COLLECTION,
        spatial_extent=projected_coords,
        temporal_extent=temporal,
        bands=tci_band_names,
    )
    cube_scl = conn.load_collection(
        collection_id=COLLECTION,
        spatial_extent=projected_coords,
        temporal_extent=temporal,
        bands=BANDS_20M_CATEGORICAL,
    )
    cube_scl = cube_scl.resample_cube_spatial(target=cube_tci, method="near")
    cube = cube_tci.merge_cubes(cube_scl)

    data = cube.download(format="gtiff")
    if not data:
        raise FetchError("DES returned empty TCI+SCL batch data")

    _DATE_RE = re.compile(r"out_(\d{4})_(\d{2})_(\d{2})T")
    results: dict[str, tuple[dict, np.ndarray, "GeoContext"]] = {}

    if isinstance(data, bytes) and data[:2] == b"\x1f\x8b":
        tf = tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")
        for member in tf.getmembers():
            if not member.name.lower().endswith((".tif", ".tiff")):
                continue
            m = _DATE_RE.search(member.name)
            if not m:
                continue
            date_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
            f = tf.extractfile(member)
            if f is None:
                continue
            try:
                with rasterio.open(io.BytesIO(f.read())) as src:
                    raw = src.read()  # (4, H, W): b02, b03, b04, scl
                    crs = src.crs
                    transform = src.transform
            except Exception:
                continue

            bands = {}
            for i, name in enumerate(tci_band_names):
                bands[name] = dn_to_reflectance(raw[i], source="des")
            imint_bands = des_to_imint_bands(bands)
            scl = raw[len(tci_band_names)].astype(np.uint8)

            proj_bounds = {k: v for k, v in projected_coords.items() if k != "crs"}
            geo = GeoContext(
                crs=str(crs),
                transform=transform,
                bounds_projected=proj_bounds,
                bounds_wgs84=None,
                shape=(raw.shape[1], raw.shape[2]),
            )
            results[date_str] = (imint_bands, scl, geo)
        tf.close()
    elif isinstance(data, bytes):
        # Single-date result (plain GeoTIFF, not tar.gz)
        with rasterio.open(io.BytesIO(data)) as src:
            raw = src.read()
            crs = src.crs
            transform = src.transform
        bands = {}
        for i, name in enumerate(tci_band_names):
            bands[name] = dn_to_reflectance(raw[i], source="des")
        imint_bands = des_to_imint_bands(bands)
        scl = raw[len(tci_band_names)].astype(np.uint8)
        proj_bounds = {k: v for k, v in projected_coords.items() if k != "crs"}
        geo = GeoContext(
            crs=str(crs), transform=transform,
            bounds_projected=proj_bounds, bounds_wgs84=None,
            shape=(raw.shape[1], raw.shape[2]),
        )
        # Use start date as fallback
        results[temporal[0]] = (imint_bands, scl, geo)

    return [(d, *results[d]) for d in sorted(results)]


def fetch_vessel_heatmap(
    coords: dict,
    date_start: str,
    date_end: str,
    output_dir: str | Path,
    *,
    cloud_threshold: float = 0.3,
    scene_cloud_max: float = 50.0,
    gaussian_sigma: float = 5.0,
    prefix: str = "",
    analyzer_type: str = "yolo",
    predict_attributes: bool = False,
) -> dict:
    """Fetch all cloud-free Sentinel-2 images in a date range, run vessel
    detection on each, and aggregate into a heatmap.

    Pipeline (single DES call per month — no STAC):

    1. For each calendar month, batch-fetch spectral bands + SCL for ALL
       available dates in one openEO call.
    2. Screen per-date AOI cloud fraction from the downloaded SCL.
    3. Run the selected vessel detector on each cloud-free date (already
       in memory — no extra fetch needed).
    4. Accumulate detection centroids into a (H, W) grid and smooth
       with a Gaussian kernel.
    5. Save the heatmap as a colormapped RGBA PNG.

    Args:
        coords: WGS84 bounding box ``{west, south, east, north}``.
        date_start: Start of search period (ISO date, e.g. ``"2025-07-01"``).
        date_end: End of search period (ISO date, e.g. ``"2025-07-31"``).
        output_dir: Directory for output files.
        cloud_threshold: Maximum AOI cloud fraction (0.0–1.0). Dates with
            higher cloud coverage are skipped. Default: 0.3 (30%).
        scene_cloud_max: Unused (kept for CLI compatibility).
        gaussian_sigma: Standard deviation (in pixels, 1 px = 10 m) for
            Gaussian smoothing of the raw detection grid. Default: 5.0.
        prefix: Date prefix for output filenames (e.g. ``"2025-07-10_"``).
        analyzer_type: ``"yolo"`` for YOLO11s or ``"ai2"`` for the Allen AI
            rslearn Swin V2 + Faster R-CNN model. Default: ``"yolo"``.
        predict_attributes: If ``True``, run the AI2 attribute model on
            each detection to predict speed, heading, length, width, and
            vessel type. Requires fetching all 9 bands (forces per-date
            mode). Default: ``False``.

    Returns:
        Dict with keys ``heatmap``, ``dates_used``, ``dates_skipped``,
        ``total_detections``, ``per_date``, ``heatmap_path``.
    """
    from datetime import datetime as _dt, timedelta as _td
    from scipy.ndimage import gaussian_filter
    from .exporters.export import save_vessel_heatmap_png, save_ai2_vessel_overlay
    from .analyzers.ai2_vessels import (
        _load_attribute_model, _predict_attributes,
        _bands_dict_to_9ch, _DEFAULT_ATTR_CKPT, BAND_ORDER,
    )
    from .utils import bands_to_rgb
    import json as _json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    projected_coords = _to_nmd_grid(coords)
    conn = _connect()

    # Select analyzer
    _use_ai2 = analyzer_type.lower() == "ai2"
    _want_attrs = predict_attributes or _use_ai2  # AI2 always predicts attributes
    if _use_ai2:
        from .analyzers.ai2_vessels import AI2VesselAnalyzer
        analyzer = AI2VesselAnalyzer(config={"water_filter": True})
        print("[vessel-heatmap] Using AI2 rslearn detector (Swin V2 B + Faster R-CNN)")
    else:
        from .analyzers.marine_vessels import MarineVesselAnalyzer
        analyzer = MarineVesselAnalyzer(config={
            "confidence": 0.286,
            "water_filter": True,
        })
        print("[vessel-heatmap] Using YOLO11s detector")
        if _want_attrs:
            print("[vessel-heatmap] + AI2 attribute prediction (speed/heading/type)")

    # Lazy-loaded attribute model (shared across all dates)
    _attr_model = [None]  # mutable container for closure access

    def _ensure_attr_model():
        if _attr_model[0] is None:
            attr_ckpt = str(_DEFAULT_ATTR_CKPT)
            if not Path(attr_ckpt).exists():
                print(f"    [attr] Checkpoint not found: {attr_ckpt}")
                return None
            _attr_model[0] = _load_attribute_model(attr_ckpt)
        return _attr_model[0]

    # Generate 2-week windows.  DES limits synchronous jobs to ~20
    # time steps; with 4 bands (B02+B03+B04+SCL) per date and ~3
    # Sentinel-2 passes per 2-week window, each call stays well
    # within the limit.
    _CHUNK_DAYS = 14
    _d_start = _dt.strptime(date_start, "%Y-%m-%d")
    _d_end = _dt.strptime(date_end, "%Y-%m-%d")

    _chunk_ranges: list[tuple[str, str]] = []
    _cur = _d_start
    while _cur <= _d_end:
        _c_end = min(_cur + _td(days=_CHUNK_DAYS), _d_end + _td(days=1))
        _chunk_ranges.append((_cur.strftime("%Y-%m-%d"), _c_end.strftime("%Y-%m-%d")))
        _cur = _c_end

    # ── Fetch + detect ──────────────────────────────────────────────
    # For large areas (> ~2000 px on any side), batch mode times out
    # on DES synchronous — go straight to per-date fetching.
    _px_w = (projected_coords["east"] - projected_coords["west"]) / 10
    _px_h = (projected_coords["north"] - projected_coords["south"]) / 10
    _use_single_date = max(_px_w, _px_h) > 2000
    if _use_single_date:
        print(f"    Large area ({_px_w:.0f}×{_px_h:.0f} px) — using per-date mode")
    heatmap = None
    per_date = []
    dates_used = []
    dates_skipped = []
    total_detections = 0
    # Track best (lowest cloud) date for overlay snapshot
    _best_snapshot = {"cloud": 1.0, "rgb": None, "regions": None, "date": None}

    def _detect_and_accumulate(date, bands, scl):
        """Run vessel detection on one date and accumulate into heatmap."""
        nonlocal heatmap, total_detections
        cloud = check_cloud_fraction(scl)
        if cloud > cloud_threshold:
            dates_skipped.append(date)
            print(f"    {date}  cloud={cloud:.1%}  ✗")
            return
        dates_used.append(date)
        rgb = bands_to_rgb(bands, scl=scl)
        result = analyzer.run(rgb, bands=bands, scl=scl)
        regions = result.outputs.get("regions", []) if result.success else []

        # ── Attribute prediction (hybrid YOLO + AI2 attributes) ──
        if _want_attrs and regions and not _use_ai2:
            # AI2 analyzer already predicts attributes internally;
            # this branch adds attributes to YOLO detections.
            attr_model = _ensure_attr_model()
            if attr_model is not None:
                try:
                    image_9ch = _bands_dict_to_9ch(bands)
                    # Convert regions to the format _predict_attributes expects
                    det_for_attr = []
                    for r in regions:
                        bb = r["bbox"]
                        det_for_attr.append({
                            "x_min": bb["x_min"], "y_min": bb["y_min"],
                            "x_max": bb["x_max"], "y_max": bb["y_max"],
                        })
                    attrs = _predict_attributes(attr_model, image_9ch, det_for_attr)
                    for r, attr in zip(regions, attrs):
                        r["attributes"] = attr
                    n_attr = sum(1 for a in attrs if a.get("vessel_type"))
                    print(f"      + attributes predicted for {n_attr}/{len(regions)} detections")
                except Exception as e:
                    print(f"      ⚠ attribute prediction failed: {e}")

        n = len(regions)
        total_detections += n
        per_date.append({"date": date, "cloud": cloud, "vessels": n})
        print(f"    {date}  cloud={cloud:.1%}  ✓  vessels={n}")
        if heatmap is None:
            heatmap = np.zeros(rgb.shape[:2], dtype=np.float32)
        for r in regions:
            bb = r["bbox"]
            cy = (bb["y_min"] + bb["y_max"]) // 2
            cx = (bb["x_min"] + bb["x_max"]) // 2
            if 0 <= cy < heatmap.shape[0] and 0 <= cx < heatmap.shape[1]:
                heatmap[cy, cx] += 1.0
        # Keep the clearest date with detections for overlay snapshot
        if n > 0 and cloud < _best_snapshot["cloud"]:
            _best_snapshot["cloud"] = cloud
            _best_snapshot["rgb"] = rgb
            _best_snapshot["regions"] = regions
            _best_snapshot["date"] = date

    _band_label = "9 bands" if _want_attrs else "TCI+SCL"
    print(f"\n[vessel-heatmap] Fetching {_band_label} from DES ({len(_chunk_ranges)} chunks) ...")
    # Attribute prediction needs 9 bands — the batch TCI+SCL fetcher only
    # provides 4, so we always use per-date mode when attributes are wanted.
    if not _use_single_date and _want_attrs:
        _use_single_date = True
        print("    Attribute prediction needs 9 bands — using per-date mode")
    if not _use_single_date:
        _use_single_date = False  # set True on first batch timeout
    for _c_start, _c_end in _chunk_ranges:
        print(f"\n  ── {_c_start} – {_c_end} ──")

        if not _use_single_date:
            try:
                batch = _fetch_tci_scl_batch(conn, projected_coords, [_c_start, _c_end])
                print(f"    {len(batch)} dates returned")
                for date, bands, scl, geo in batch:
                    _detect_and_accumulate(date, bands, scl)
                continue
            except (FetchError, Exception) as _batch_err:
                if "504" in str(_batch_err) or "Gateway" in str(_batch_err):
                    print(f"    Batch too large — switching to per-date mode")
                    _use_single_date = True
                elif "no matching dates" in str(_batch_err):
                    print(f"    (no data)")
                    continue
                else:
                    print(f"    Batch failed: {_batch_err} — switching to per-date mode")
                    _use_single_date = True

        # Per-date fallback for large areas
        _c_dt = _dt.strptime(_c_start, "%Y-%m-%d")
        _e_dt = _dt.strptime(_c_end, "%Y-%m-%d")
        while _c_dt < _e_dt:
            date = _c_dt.strftime("%Y-%m-%d")
            temporal = [date, (_c_dt + _td(days=1)).strftime("%Y-%m-%d")]
            try:
                if _want_attrs:
                    bands, scl, geo = _fetch_ai2_bands(conn, projected_coords, temporal)
                else:
                    bands, scl, geo = _fetch_tci_bands(conn, projected_coords, temporal)
                _detect_and_accumulate(date, bands, scl)
            except Exception as e:
                if "empty" not in str(e).lower():
                    print(f"    {date}  ✗ {e}")
            _c_dt += _td(days=1)

    print(f"\n    {len(dates_used)} dates used, {len(dates_skipped)} skipped")

    # ── Step 5: Smooth and save ──────────────────────────────────────
    heatmap_path = None
    if heatmap is not None and heatmap.max() > 0:
        heatmap = gaussian_filter(heatmap, sigma=gaussian_sigma)
        # Cache the raw smoothed array for quick re-export
        np.save(str(output_dir / f"{prefix}vessel_heatmap.npy"), heatmap)
        heatmap_path = output_dir / f"{prefix}vessel_heatmap_clean.png"
        save_vessel_heatmap_png(heatmap, str(heatmap_path))

    # ── Save vessel overlay with attributes (AI2 or YOLO+attributes) ──
    ai2_overlay_path = None
    if _want_attrs and _best_snapshot["regions"]:
        # Check that at least some detections have attributes
        has_attrs = any(r.get("attributes") for r in _best_snapshot["regions"])
        if has_attrs:
            ai2_overlay_path = output_dir / f"{prefix}ai2_vessels_clean.png"
            save_ai2_vessel_overlay(
                _best_snapshot["rgb"],
                _best_snapshot["regions"],
                str(ai2_overlay_path),
            )
            _src = "AI2" if _use_ai2 else "YOLO+attributes"
            print(f"[vessel-heatmap] {_src} overlay ({_best_snapshot['date']}) → {ai2_overlay_path}")

    # Save summary JSON
    summary = {
        "date_start": date_start,
        "date_end": date_end,
        "dates_used": dates_used,
        "dates_skipped": dates_skipped,
        "total_detections": total_detections,
        "per_date": per_date,
        "cloud_threshold": cloud_threshold,
        "gaussian_sigma": gaussian_sigma,
        "analyzer": analyzer_type,
    }
    summary_path = output_dir / f"{prefix}vessel_heatmap_summary.json"
    with open(summary_path, "w") as f:
        _json.dump(summary, f, indent=2)
    print(f"\n[vessel-heatmap] Summary → {summary_path}")

    return {
        "heatmap": heatmap,
        "dates_used": dates_used,
        "dates_skipped": dates_skipped,
        "total_detections": total_detections,
        "per_date": per_date,
        "heatmap_path": heatmap_path,
    }


# ── Grazing timeseries fetching ──────────────────────────────────────────────

def _process_grazing_tif(
    tif_bytes: bytes,
    date_str: str,
    projected_bbox: dict,
    polygon_mask: np.ndarray,
    cloud_threshold: float,
    clean_dates: list,
    clean_cloud_fracs: list,
    clean_bands_stack: list,
    clean_transforms: list,
) -> None:
    """Parse one GeoTIFF from a grazing timeseries fetch.

    Performs grid-snapping, polygon-level cloud check, band reordering,
    and DN→reflectance conversion. Appends to the clean_* lists if the
    timestep passes cloud filtering.

    Band order in the downloaded GeoTIFF (from merge order):
        10m:  b02(0) b03(1) b04(2) b08(3)
        20m:  b05(4) b06(5) b07(6) b8a(7) b11(8) b12(9)
        60m:  b01(10) b09(11)
        cat:  scl(12)
    """
    import rasterio

    with rasterio.open(io.BytesIO(tif_bytes)) as src:
        raw = src.read()  # (13, H, W)
        crs = src.crs
        transform = src.transform

    if raw.shape[0] < 13:
        print(f"    {date_str}  only {raw.shape[0]} bands (expected 13), skip")
        return

    # Grid-align to NMD grid
    target_bounds = {k: v for k, v in projected_bbox.items() if k != "crs"}
    raw, transform = _snap_to_target_grid(
        raw, transform, crs, target_bounds, pixel_size=NMD_GRID_SIZE,
    )

    # Extract SCL (band index 12) and check polygon-level cloud
    scl = raw[12].astype(np.uint8)

    # Ensure mask matches raster dimensions
    if polygon_mask.shape != scl.shape:
        # Re-rasterize if grid snapping changed dimensions
        from rasterio.transform import from_origin
        from rasterio.features import rasterize as rio_rasterize
        _mask = np.zeros(scl.shape, dtype=bool)
        print(f"    {date_str}  mask shape mismatch {polygon_mask.shape} vs "
              f"{scl.shape}, using full-bbox cloud check")
        cloud_frac = check_cloud_fraction(scl)
    else:
        cloud_frac = _polygon_cloud_fraction(scl, polygon_mask)

    if cloud_frac > cloud_threshold:
        print(f"    {date_str}  cloud={cloud_frac:.1%}  (skip)")
        return

    # Reorder spectral bands: download order → GRAZING_BAND_ORDER
    spectral = raw[_GRAZING_REORDER]  # (12, H, W)

    # Convert DN to reflectance
    spectral_ref = np.stack([
        dn_to_reflectance(spectral[i], source="des")
        for i in range(12)
    ])  # (12, H, W) float32

    clean_dates.append(date_str)
    clean_cloud_fracs.append(cloud_frac)
    clean_bands_stack.append(spectral_ref)
    clean_transforms.append(transform)
    print(f"    {date_str}  cloud={cloud_frac:.1%}  OK")


# ── LPIS (Jordbruksverket) polygon fetching ──────────────────────────────

def _lpis_bbox_string(bbox, bbox_crs: str | None = None) -> tuple[str, str]:
    """Convert bbox to WFS bbox parameter string + CRS.

    Auto-detects CRS from coordinate magnitude if *bbox_crs* is None:
    values with ``|west| < 180`` are treated as WGS84, otherwise EPSG:3006.

    Args:
        bbox: dict ``{"west","south","east","north"}`` **or** tuple
            ``(west, south, east, north)``.
        bbox_crs: Explicit CRS (e.g. ``"EPSG:4326"``).  Auto-detected if
            *None*.

    Returns:
        ``(bbox_str, crs_str)`` — e.g. ``("13.4,55.9,13.5,56.0", "EPSG:4326")``.
    """
    if isinstance(bbox, dict):
        w, s, e, n = bbox["west"], bbox["south"], bbox["east"], bbox["north"]
    elif isinstance(bbox, (tuple, list)) and len(bbox) == 4:
        w, s, e, n = bbox
    else:
        raise ValueError(
            "bbox must be dict with west/south/east/north or 4-element tuple"
        )

    if bbox_crs is None:
        bbox_crs = "EPSG:4326" if abs(w) < 180 and abs(e) < 180 else "EPSG:3006"

    return f"{w},{s},{e},{n},{bbox_crs}", bbox_crs


def fetch_lpis_polygons(
    bbox,
    *,
    agoslag: str | list[str] | None = "Bete",
    bbox_crs: str | None = None,
    max_features: int = 5000,
    timeout: int = 60,
    cache_dir: str | None = None,
):
    """Fetch LPIS field boundary polygons from Jordbruksverket WFS.

    Downloads agricultural block polygons (jordbruksblock) for a given
    bounding box.  By default only pasture blocks (ägoslag = "Bete") are
    returned — set *agoslag=None* for all land-use types.

    The data is open (CC BY 4.0) and requires no authentication.

    Args:
        bbox: Bounding box — dict ``{"west","south","east","north"}`` or
            tuple ``(west, south, east, north)``.  Accepts both WGS84 and
            EPSG:3006 coordinates (auto-detected from magnitude).
        agoslag: Land-use filter.  ``"Bete"`` for pasture only, ``None``
            for all types, or a list like ``["Bete", "Åker"]``.
        bbox_crs: Explicit CRS for *bbox*.  Auto-detected if *None*.
        max_features: Maximum number of features to request.  Default 5000.
        timeout: HTTP timeout in seconds.
        cache_dir: Directory for caching WFS responses.  If provided,
            subsequent calls with the same bbox + filter skip the HTTP
            request.

    Returns:
        ``geopandas.GeoDataFrame`` with columns ``geometry``, ``blockid``,
        ``agoslag``, ``kategori``, ``areal``, ``region_kod``, ``arslager``.
        CRS is EPSG:3006.

    Raises:
        FetchError: If the WFS request fails.
    """
    import geopandas as gpd
    import hashlib
    import json
    import urllib.request
    import urllib.parse

    bbox_str, _crs = _lpis_bbox_string(bbox, bbox_crs)

    # ── Check cache ───────────────────────────────────────────────
    cache_path = None
    if cache_dir:
        _cache = Path(cache_dir)
        _cache.mkdir(parents=True, exist_ok=True)
        _key = f"{bbox_str}|{agoslag}|{max_features}"
        _hash = hashlib.md5(_key.encode()).hexdigest()[:12]
        cache_path = _cache / f"lpis_{_hash}.json"
        if cache_path.exists():
            print(f"  LPIS cache hit: {cache_path}")
            gdf = gpd.read_file(str(cache_path))
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:3006")
            return gdf

    # ── Build WFS URL ─────────────────────────────────────────────
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": LPIS_LAYER,
        "outputFormat": "application/json",
        "srsName": "EPSG:3006",
        "bbox": bbox_str,
        "count": str(max_features),
    }
    url = f"{LPIS_WFS_URL}?{urllib.parse.urlencode(params)}"

    print(f"  Fetching LPIS polygons from Jordbruksverket WFS …")
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            data = json.loads(raw)
    except Exception as e:
        raise FetchError(f"LPIS WFS request failed: {e}") from e

    features = data.get("features", [])
    if not features:
        print("  LPIS: no features found in bbox")
        # Return empty GeoDataFrame with correct schema
        gdf = gpd.GeoDataFrame(
            columns=["geometry", "blockid", "agoslag", "kategori",
                      "areal", "region_kod", "arslager"],
            geometry="geometry",
            crs="EPSG:3006",
        )
        return gdf

    # ── Parse to GeoDataFrame ─────────────────────────────────────
    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:3006")
    print(f"  LPIS: {len(gdf)} block fetched")

    # ── Filter by ägoslag ─────────────────────────────────────────
    if agoslag is not None and "agoslag" in gdf.columns:
        if isinstance(agoslag, str):
            agoslag = [agoslag]
        gdf = gdf[gdf["agoslag"].isin(agoslag)].copy()
        print(f"  LPIS: {len(gdf)} block after ägoslag filter ({agoslag})")

    # ── Cache ─────────────────────────────────────────────────────
    if cache_path is not None and len(gdf) > 0:
        tmp = cache_path.with_suffix(".json.tmp")
        gdf.to_file(str(tmp), driver="GeoJSON")
        tmp.rename(cache_path)
        print(f"  LPIS: cached to {cache_path}")

    return gdf


def fetch_grazing_lpis(
    bbox,
    year: int,
    *,
    agoslag: str | list[str] | None = "Bete",
    bbox_crs: str | None = None,
    cloud_threshold: float = 0.01,
    scene_cloud_max: float = 50.0,
    buffer_m: float = 50.0,
    chunk_days: int = 5,
    token: str | None = None,
    max_features: int = 5000,
    lpis_cache_dir: str | None = None,
) -> list:
    """Fetch LPIS pasture polygons and their Sentinel-2 timeseries.

    Convenience wrapper that combines :func:`fetch_lpis_polygons` with
    :func:`fetch_grazing_timeseries` in a single call.

    Args:
        bbox: Bounding box for LPIS polygon search (WGS84 or EPSG:3006).
        year: Year to fetch Sentinel-2 data (growing season Apr–Oct).
        agoslag: Land-use filter for LPIS (default ``"Bete"``).
        bbox_crs: CRS of *bbox* (auto-detected if *None*).
        cloud_threshold: Max polygon-level cloud fraction.
        scene_cloud_max: STAC pre-filter for scene cloud %.
        buffer_m: Buffer around polygon bbox in metres.
        chunk_days: Temporal chunk size for DES calls.
        token: DES access token.
        max_features: Max LPIS polygons to fetch.
        lpis_cache_dir: Cache directory for WFS responses.

    Returns:
        List of :class:`GrazingTimeseriesResult`, one per LPIS polygon.

    Raises:
        FetchError: If LPIS returns no polygons or DES calls fail.
    """
    lpis_gdf = fetch_lpis_polygons(
        bbox,
        agoslag=agoslag,
        bbox_crs=bbox_crs,
        max_features=max_features,
        cache_dir=lpis_cache_dir,
    )

    if len(lpis_gdf) == 0:
        raise FetchError(
            f"No LPIS polygons found for bbox={bbox} with ägoslag={agoslag}"
        )

    print(f"  Fetching grazing timeseries for {len(lpis_gdf)} LPIS polygons …")
    return fetch_grazing_timeseries(
        lpis_gdf,
        year,
        cloud_threshold=cloud_threshold,
        scene_cloud_max=scene_cloud_max,
        buffer_m=buffer_m,
        chunk_days=chunk_days,
        token=token,
        polygon_id_col="blockid",
        polygon_crs="EPSG:3006",
    )


def fetch_grazing_timeseries(
    polygons,
    year: int,
    *,
    date_start: str | None = None,
    date_end: str | None = None,
    cloud_threshold: float = 0.01,
    scene_cloud_max: float = 50.0,
    buffer_m: float = 50.0,
    chunk_days: int = 5,
    token: str | None = None,
    polygon_id_col: str | None = None,
    polygon_crs: str = "EPSG:4326",
) -> list[GrazingTimeseriesResult]:
    """Fetch multi-date 12-band Sentinel-2 timeseries for grazing polygons.

    Designed for the pib-ml-grazing CNN-LSTM model. For each polygon:

    1. Compute projected bbox from polygon geometry (+buffer)
    2. Discover available dates via STAC (April 1 – October 21)
    3. Fetch all 12 bands + SCL in 14-day chunks via openEO
    4. Filter by polygon-level cloud fraction (default 1%)
    5. Co-register all timesteps (integer + sub-pixel alignment)
    6. Stack cloud-free dates into (T, 12, H, W) array

    Band order in output: B01, B02, B03, B04, B05, B06, B07, B08,
                          B8A, B09, B11, B12

    Values are BOA reflectance: (DN − 1000) / 10000, clipped to [0, 1].

    Args:
        polygons: One of:
            - shapely Polygon/MultiPolygon geometry (single polygon)
            - list of shapely geometries
            - GeoDataFrame with polygon geometries
        year: Year to fetch (e.g. 2023). Growing season Apr 1 – Oct 21.
        date_start: Override start date (ISO). Default: ``"{year}-04-01"``.
        date_end: Override end date (ISO). Default: ``"{year}-10-21"``.
        cloud_threshold: Max cloud fraction within polygon to keep a
            timestep. Default: 0.01 (1%).
        scene_cloud_max: STAC pre-filter for scene-level cloud %.
            Default: 50%.
        buffer_m: Buffer in metres around polygon bbox. Default: 50m.
        chunk_days: Days per temporal chunk for DES calls. Default: 14.
        token: Optional DES access token.
        polygon_id_col: Column name in GeoDataFrame to use as polygon ID.
            If None, uses integer index.
        polygon_crs: CRS of input polygon geometries. Default: EPSG:4326.
            Ignored if *polygons* is a GeoDataFrame (uses ``.crs``).

    Returns:
        List of :class:`GrazingTimeseriesResult`, one per polygon.

    Raises:
        FetchError: If DES calls fail.
        ValueError: If no valid polygons provided.
    """
    import re
    import tarfile
    from datetime import datetime as _dt, timedelta as _td
    from rasterio.transform import from_origin
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds as _tb

    from .coregistration import coregister_to_reference

    # ── Normalise polygon input ──────────────────────────────────────
    if hasattr(polygons, "geometry") and hasattr(polygons, "crs"):
        # GeoDataFrame
        _crs = str(polygons.crs) if polygons.crs else polygon_crs
        geom_list = []
        for idx, row in polygons.iterrows():
            pid = (
                row[polygon_id_col]
                if polygon_id_col and polygon_id_col in row.index
                else idx
            )
            geom_list.append((pid, row.geometry))
    elif hasattr(polygons, "geom_type"):
        # Single shapely geometry
        _crs = polygon_crs
        geom_list = [(0, polygons)]
    elif isinstance(polygons, list):
        _crs = polygon_crs
        geom_list = [(i, g) for i, g in enumerate(polygons)]
    else:
        raise ValueError(
            "polygons must be a shapely geometry, list of geometries, "
            "or GeoDataFrame"
        )
    if not geom_list:
        raise ValueError("No valid polygons provided")

    # ── Temporal range ───────────────────────────────────────────────
    _start = date_start or f"{year}-04-01"
    _end = date_end or f"{year}-10-21"

    # ── Connect to DES ───────────────────────────────────────────────
    conn = _connect(token=token)
    results: list[GrazingTimeseriesResult] = []

    _DATE_RE = re.compile(r"out_(\d{4})_(\d{2})_(\d{2})T")

    for poly_idx, (pid, geom) in enumerate(geom_list):
        print(f"\n[grazing] Polygon {poly_idx + 1}/{len(geom_list)} "
              f"(id={pid})")

        # ── Project polygon bbox to EPSG:3006 ────────────────────────
        projected_bbox = _polygon_to_projected_bbox(
            geom, src_crs=_crs, buffer_m=buffer_m,
        )
        proj_bounds = {k: v for k, v in projected_bbox.items()
                       if k != "crs"}

        # ── Project polygon geometry for rasterization ───────────────
        if _crs != TARGET_CRS:
            import pyproj
            from shapely.ops import transform as shapely_transform
            _proj = pyproj.Transformer.from_crs(
                _crs, TARGET_CRS, always_xy=True,
            ).transform
            geom_3006 = shapely_transform(_proj, geom)
        else:
            geom_3006 = geom

        polygon_mask = _rasterize_polygon(
            geom_3006, proj_bounds, pixel_size=NMD_GRID_SIZE,
        )

        # ── STAC date discovery ──────────────────────────────────────
        # Need WGS84 bbox for STAC
        w84 = _tb(
            CRS.from_epsg(3006), CRS.from_epsg(4326),
            proj_bounds["west"], proj_bounds["south"],
            proj_bounds["east"], proj_bounds["north"],
        )
        wgs84_coords = {
            "west": w84[0], "south": w84[1],
            "east": w84[2], "north": w84[3],
        }
        stac_dates = _stac_available_dates(
            wgs84_coords, _start, _end, scene_cloud_max,
        )
        print(f"  [STAC] {len(stac_dates)} dates with scene cloud "
              f"<= {scene_cloud_max:.0f}%")

        if not stac_dates:
            H, W = polygon_mask.shape
            results.append(GrazingTimeseriesResult(
                data=np.empty((0, 12, H, W), dtype=np.float32),
                dates=[], cloud_fractions=[], polygon_id=pid,
                geo=GeoContext(
                    crs=TARGET_CRS,
                    transform=from_origin(
                        proj_bounds["west"], proj_bounds["north"],
                        NMD_GRID_SIZE, NMD_GRID_SIZE),
                    bounds_projected=proj_bounds,
                    bounds_wgs84=wgs84_coords,
                    shape=(H, W),
                ),
                shape_hw=(H, W),
            ))
            continue

        # ── Generate 14-day chunk windows ────────────────────────────
        _d_start = _dt.strptime(_start, "%Y-%m-%d")
        _d_end = _dt.strptime(_end, "%Y-%m-%d")
        _chunk_ranges: list[tuple[str, str]] = []
        _cur = _d_start
        while _cur <= _d_end:
            _c_end = min(_cur + _td(days=chunk_days),
                         _d_end + _td(days=1))
            _chunk_ranges.append((
                _cur.strftime("%Y-%m-%d"),
                _c_end.strftime("%Y-%m-%d"),
            ))
            _cur = _c_end

        # ── Fetch per chunk ──────────────────────────────────────────
        clean_dates: list[str] = []
        clean_cloud_fracs: list[float] = []
        clean_bands_stack: list[np.ndarray] = []
        clean_transforms: list = []

        for c_start, c_end in _chunk_ranges:
            print(f"  -- {c_start} to {c_end} --")
            try:
                # 10m bands
                cube_10m = conn.load_collection(
                    collection_id=COLLECTION,
                    spatial_extent=projected_bbox,
                    temporal_extent=[c_start, c_end],
                    bands=BANDS_10M,
                )
                # 20m spectral → bilinear resample to 10m
                cube_20m = conn.load_collection(
                    collection_id=COLLECTION,
                    spatial_extent=projected_bbox,
                    temporal_extent=[c_start, c_end],
                    bands=BANDS_20M_SPECTRAL,
                )
                cube_20m = cube_20m.resample_cube_spatial(
                    target=cube_10m, method="bilinear",
                )
                # 60m bands (B01 + B09) → bilinear resample to 10m
                cube_60m = conn.load_collection(
                    collection_id=COLLECTION,
                    spatial_extent=projected_bbox,
                    temporal_extent=[c_start, c_end],
                    bands=BANDS_60M_ALL,
                )
                cube_60m = cube_60m.resample_cube_spatial(
                    target=cube_10m, method="bilinear",
                )
                # SCL → nearest resample to 10m
                cube_scl = conn.load_collection(
                    collection_id=COLLECTION,
                    spatial_extent=projected_bbox,
                    temporal_extent=[c_start, c_end],
                    bands=BANDS_20M_CATEGORICAL,
                )
                cube_scl = cube_scl.resample_cube_spatial(
                    target=cube_10m, method="near",
                )
                # Merge all and download
                cube = (cube_10m.merge_cubes(cube_20m)
                        .merge_cubes(cube_60m)
                        .merge_cubes(cube_scl))
                data = cube.download(format="gtiff")

                if not data:
                    print("    (no data)")
                    continue

                # Parse response — multi-date returns tar.gz
                if isinstance(data, bytes) and data[:2] == b"\x1f\x8b":
                    tf = tarfile.open(
                        fileobj=io.BytesIO(data), mode="r:gz",
                    )
                    for member in tf.getmembers():
                        if not member.name.lower().endswith(
                            (".tif", ".tiff")
                        ):
                            continue
                        m = _DATE_RE.search(member.name)
                        if not m:
                            continue
                        d_str = (f"{m.group(1)}-{m.group(2)}-"
                                 f"{m.group(3)}")
                        f = tf.extractfile(member)
                        if f is None:
                            continue
                        _process_grazing_tif(
                            f.read(), d_str, projected_bbox,
                            polygon_mask, cloud_threshold,
                            clean_dates, clean_cloud_fracs,
                            clean_bands_stack, clean_transforms,
                        )
                    tf.close()
                elif isinstance(data, bytes):
                    _process_grazing_tif(
                        data, c_start, projected_bbox,
                        polygon_mask, cloud_threshold,
                        clean_dates, clean_cloud_fracs,
                        clean_bands_stack, clean_transforms,
                    )
            except Exception as e:
                print(f"    Chunk {c_start}–{c_end} failed: {e}")
                continue

        # ── Co-registration ──────────────────────────────────────────
        # First cloud-free timestep is the reference; align all others.
        if len(clean_bands_stack) >= 2:
            ref_bands = clean_bands_stack[0]   # (12, H, W)
            ref_tf = clean_transforms[0]
            ref_tf_list = list(ref_tf)[:6] if hasattr(ref_tf, '__iter__') else None

            aligned_stack = [ref_bands]
            for i in range(1, len(clean_bands_stack)):
                cur_bands = clean_bands_stack[i]
                cur_tf = clean_transforms[i]
                cur_tf_list = list(cur_tf)[:6] if hasattr(cur_tf, '__iter__') else None

                # Transpose (12, H, W) → (H, W, 12) for coregister
                cur_hwc = np.transpose(cur_bands, (1, 2, 0))
                ref_hwc = np.transpose(ref_bands, (1, 2, 0))

                try:
                    aligned_cur, _aligned_ref, meta = coregister_to_reference(
                        target=cur_hwc,
                        reference=ref_hwc,
                        target_transform=cur_tf_list,
                        reference_transform=ref_tf_list,
                        subpixel=True,
                        reference_band=3,  # B04 (red) in GRAZING order
                    )
                    # Transpose back (H, W, 12) → (12, H, W)
                    aligned_stack.append(
                        np.transpose(aligned_cur, (2, 0, 1))
                    )
                    sub = meta.get("subpixel_offset", (0, 0))
                    intg = meta.get("integer_offset", (0, 0))
                    if abs(intg[0]) + abs(intg[1]) > 0 or abs(sub[0]) + abs(sub[1]) > 0.05:
                        print(f"    [coreg] {clean_dates[i]}: "
                              f"int=({intg[0]},{intg[1]}) "
                              f"sub=({sub[0]:.3f},{sub[1]:.3f})")
                except Exception as e:
                    print(f"    [coreg] {clean_dates[i]} failed: {e}, "
                          f"using grid-snapped version")
                    aligned_stack.append(cur_bands)

            # After co-registration, all arrays must have the same shape.
            # Crop to the minimum common shape.
            min_h = min(a.shape[1] for a in aligned_stack)
            min_w = min(a.shape[2] for a in aligned_stack)
            clean_bands_stack = [
                a[:, :min_h, :min_w] for a in aligned_stack
            ]

        # ── Stack result ─────────────────────────────────────────────
        if clean_bands_stack:
            # Sort by date
            order = sorted(range(len(clean_dates)),
                           key=lambda i: clean_dates[i])
            data_array = np.stack(
                [clean_bands_stack[i] for i in order], axis=0,
            )  # (T, 12, H, W)
            sorted_dates = [clean_dates[i] for i in order]
            sorted_fracs = [clean_cloud_fracs[i] for i in order]
        else:
            H, W = polygon_mask.shape
            data_array = np.empty((0, 12, H, W), dtype=np.float32)
            sorted_dates = []
            sorted_fracs = []

        H_out, W_out = data_array.shape[2], data_array.shape[3] if data_array.size else polygon_mask.shape
        geo = GeoContext(
            crs=TARGET_CRS,
            transform=from_origin(
                proj_bounds["west"], proj_bounds["north"],
                NMD_GRID_SIZE, NMD_GRID_SIZE,
            ),
            bounds_projected=proj_bounds,
            bounds_wgs84=wgs84_coords,
            shape=(H_out, W_out),
        )

        result = GrazingTimeseriesResult(
            data=data_array,
            dates=sorted_dates,
            cloud_fractions=sorted_fracs,
            polygon_id=pid,
            geo=geo,
            shape_hw=(H_out, W_out),
        )
        results.append(result)
        print(f"  [grazing] Polygon {pid}: {len(sorted_dates)} clean "
              f"dates out of {len(stac_dates)} available, "
              f"shape={data_array.shape}")

    return results


# ── Cloud-free baseline fetching ─────────────────────────────────────────────

def fetch_cloud_free_baseline(
    date: str,
    coords: dict,
    search_start_days: int = 30,
    search_end_days: int = 90,
    scan_interval_days: int = 7,
    cloud_threshold: float = 0.1,
    token: str | None = None,
) -> FetchResult:
    """Fetch a cloud-free Sentinel-2 image for use as a change detection baseline.

    Three-phase STAC-guided approach:
    1. STAC discovery — find all available dates in the search window
    2. SCL screening — compute AOI-specific cloud fraction per date
    3. COT ranking — run COT on top 5 to pick best visibility

    Args:
        date: ISO date string of the analysis (e.g. "2024-07-15").
        coords: WGS84 bounding box dict.
        search_start_days: Start of search window (days before date). Default: 30.
        search_end_days: End of search window (days before date). Default: 90.
        scan_interval_days: Ignored (kept for API compatibility).
        cloud_threshold: Maximum acceptable cloud fraction. Default: 0.1.
        token: Optional DES access token.

    Returns:
        FetchResult for the image with best visibility.

    Raises:
        FetchError: If no candidate below cloud_threshold is found.
    """
    from datetime import datetime, timedelta

    dt = datetime.strptime(date, "%Y-%m-%d")
    start = (dt - timedelta(days=search_end_days)).strftime("%Y-%m-%d")
    end = (dt - timedelta(days=search_start_days)).strftime("%Y-%m-%d")

    # Phase 1: STAC discovery
    print(f"    [STAC] Querying available dates {start} to {end}...")
    stac_dates = _stac_available_dates(coords, start, end, scene_cloud_max=80.0)
    print(f"    [STAC] {len(stac_dates)} dates with scene cloud <= 80%")

    if not stac_dates:
        raise FetchError(
            f"No Sentinel-2 data available in baseline window "
            f"({search_start_days}-{search_end_days} days back). "
            f"STAC returned no results."
        )

    # Phase 2: SCL screening per STAC date
    conn = _connect(token=token)
    projected_coords = _to_nmd_grid(coords)

    scl_results: list[tuple[str, float, float]] = []  # (date, aoi_cloud, scene_cloud)

    for i, (candidate_date, scene_cloud) in enumerate(stac_dates, 1):
        candidate_dt = datetime.strptime(candidate_date, "%Y-%m-%d")
        temporal = [
            candidate_date,
            (candidate_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
        ]
        try:
            _scl, aoi_cloud, _crs, _transform = _fetch_scl(
                conn, projected_coords, temporal, date_window=0
            )
            print(f"    [{i}/{len(stac_dates)}] {candidate_date}  "
                  f"scene={scene_cloud:.1f}%  AOI(SCL)={aoi_cloud:.1%}")
            scl_results.append((candidate_date, aoi_cloud, scene_cloud))
        except Exception:
            print(f"    [{i}/{len(stac_dates)}] {candidate_date}  -> no data")
            continue

    if not scl_results:
        raise FetchError("All SCL fetches failed — no baseline candidates")

    # Sort by AOI cloud fraction
    scl_results.sort(key=lambda x: x[1])

    # Phase 3: COT on top 5 for visibility ranking
    top_n = min(5, len(scl_results))
    top_dates = scl_results[:top_n]

    # Only run COT if we have multiple good candidates to choose from
    best_date = top_dates[0][0]
    best_cloud = top_dates[0][1]

    if top_n >= 2 and top_dates[0][1] < cloud_threshold:
        try:
            from imint.analyzers.cot import COTAnalyzer
            import rasterio

            analyzer = COTAnalyzer(config={"device": "cpu"})
            print(f"    [COT] Ranking top {top_n} by visibility...")

            cot_scores: list[tuple[str, float, float]] = []  # (date, clear_frac, cot_mean)

            for candidate_date, aoi_cloud, _scene_cloud in top_dates:
                candidate_dt = datetime.strptime(candidate_date, "%Y-%m-%d")
                temporal = [
                    candidate_date,
                    (candidate_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
                ]
                try:
                    # Full spectral fetch for COT
                    cube_10m = conn.load_collection(
                        collection_id=COLLECTION,
                        spatial_extent=projected_coords,
                        temporal_extent=temporal,
                        bands=BANDS_10M,
                    )
                    cube_20m = conn.load_collection(
                        collection_id=COLLECTION,
                        spatial_extent=projected_coords,
                        temporal_extent=temporal,
                        bands=BANDS_20M_SPECTRAL,
                    )
                    cube_20m = cube_20m.resample_cube_spatial(
                        target=cube_10m, method="bilinear"
                    )
                    cube_60m = conn.load_collection(
                        collection_id=COLLECTION,
                        spatial_extent=projected_coords,
                        temporal_extent=temporal,
                        bands=BANDS_60M,
                    )
                    cube_60m = cube_60m.resample_cube_spatial(
                        target=cube_10m, method="bilinear"
                    )
                    cube = cube_10m.merge_cubes(cube_20m).merge_cubes(cube_60m)
                    data = cube.download(format="gtiff")

                    with rasterio.open(io.BytesIO(data)) as src:
                        raw = src.read()

                    spectral_names = BANDS_10M + BANDS_20M_SPECTRAL + BANDS_60M
                    bands = des_to_imint_bands(
                        {n.lower(): dn_to_reflectance(raw[j], source="des")
                         for j, n in enumerate(spectral_names)}
                    )

                    rgb_dummy = np.zeros((*raw.shape[1:], 3), dtype=np.float32)
                    result = analyzer.analyze(
                        rgb=rgb_dummy, bands=bands, date=candidate_date,
                        output_dir="/tmp",
                    )
                    if result.success:
                        cf = result.outputs["stats"]["clear_fraction"]
                        cm = result.outputs["stats"]["cot_mean"]
                        print(f"      {candidate_date}  clear={cf:.1%}  COT={cm:.6f}")
                        cot_scores.append((candidate_date, cf, cm))
                except Exception as e:
                    print(f"      {candidate_date}  COT failed: {e}")

            if cot_scores:
                # Best visibility: highest clear_fraction, then lowest cot_mean
                cot_scores.sort(key=lambda x: (-x[1], x[2]))
                best_date = cot_scores[0][0]
                print(f"    [COT] Best visibility: {best_date} "
                      f"(clear={cot_scores[0][1]:.1%}, COT={cot_scores[0][2]:.6f})")

        except ImportError:
            print("    [COT] COT analyzer not available, using SCL ranking")

    # Check threshold
    if best_cloud > cloud_threshold and best_date == top_dates[0][0]:
        raise FetchError(
            f"No cloud-free baseline found. "
            f"Best AOI cloud: {best_cloud:.1%} (threshold: {cloud_threshold:.0%})"
        )

    # Phase 4: Full fetch for the best date
    print(f"    Selected baseline: {best_date}")
    return fetch_des_data(
        date=best_date,
        coords=coords,
        cloud_threshold=1.0,  # already verified
        token=token,
        include_scl=True,
        date_window=0,
    )


STAC_SEARCH_URL = "https://explorer.digitalearth.se/stac/search"


def _stac_best_date(coords: dict, date: str, window: int) -> str:
    """Find best available date via STAC within date ± window.

    Queries STAC for all Sentinel-2 L2A dates in the window and returns
    the one with lowest scene-level cloud cover.

    Args:
        coords: WGS84 bounding box dict.
        date: Target ISO date string.
        window: Days before/after date to search.

    Returns:
        ISO date string of the best available date.

    Raises:
        FetchError: If no data available in the window.
    """
    from datetime import datetime, timedelta

    dt = datetime.strptime(date, "%Y-%m-%d")
    start = (dt - timedelta(days=window)).strftime("%Y-%m-%d")
    end = (dt + timedelta(days=window)).strftime("%Y-%m-%d")

    stac_dates = _stac_available_dates(coords, start, end, scene_cloud_max=100.0)
    if not stac_dates:
        raise FetchError(
            f"No Sentinel-2 data available for {date} ±{window} days "
            f"(STAC query returned no results)"
        )

    best_date, best_cloud = stac_dates[0]  # already sorted by cloud asc
    print(f"    [STAC] {len(stac_dates)} dates in {start}/{end}, "
          f"best: {best_date} (scene cloud {best_cloud:.1f}%)")
    return best_date


@dataclass
class BaselineCandidate:
    """A candidate baseline image with cloud statistics and RGB thumbnail."""
    date: str
    cloud_fraction: float        # SCL-based, within AOI
    scene_cloud_fraction: float  # Scene-level from STAC metadata
    rgb_thumbnail: np.ndarray    # (H, W, 3) float32 [0, 1]
    shape: tuple                 # (H, W)
    cot_stats: dict | None = None  # COT analysis: clear_fraction, cot_mean, etc.


def _stac_available_dates(
    coords: dict,
    date_start: str,
    date_end: str,
    scene_cloud_max: float = 80.0,
) -> list[tuple[str, float]]:
    """Query STAC API for available Sentinel-2 L2A dates within bbox and time range.

    Returns deduplicated dates sorted by scene cloud cover ascending.
    Multiple tiles/orbits on the same date are merged (lowest cloud kept).

    Args:
        coords: WGS84 bounding box dict.
        date_start: ISO start date (inclusive).
        date_end: ISO end date (inclusive).
        scene_cloud_max: Discard dates with scene cloud > this %.

    Returns:
        List of (date_str, scene_cloud_pct) sorted by cloud ascending.
    """
    import urllib.request
    import urllib.parse
    import json

    params = urllib.parse.urlencode({
        "collections": "s2_msi_l2a",
        "bbox": f"{coords['west']},{coords['south']},{coords['east']},{coords['north']}",
        "datetime": f"{date_start}/{date_end}",
        "limit": 200,
    })
    url = f"{STAC_SEARCH_URL}?{params}"

    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    # Deduplicate: keep lowest cloud per date
    date_cloud: dict[str, float] = {}
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        dt_str = (props.get("datetime") or props.get("start_datetime") or "")[:10]
        if not dt_str:
            continue
        cc = props.get("eo:cloud_cover", props.get("cloud_cover"))
        if cc is None:
            cc = 100.0
        if dt_str not in date_cloud or cc < date_cloud[dt_str]:
            date_cloud[dt_str] = cc

    # Filter and sort by cloud ascending
    result = [
        (d, c) for d, c in date_cloud.items()
        if c <= scene_cloud_max
    ]
    result.sort(key=lambda x: x[1])
    return result


def scan_baseline_candidates(
    date: str,
    coords: dict,
    search_start_days: int = 30,
    search_end_days: int = 90,
    scene_cloud_max: float = 80.0,
    token: str | None = None,
) -> list[BaselineCandidate]:
    """Scan all available dates via STAC API, then fetch RGB+SCL for AOI cloud %.

    Two-phase approach:
    1. Query STAC for all dates with data (fast HTTP call)
    2. Fetch RGB+SCL only for dates with scene cloud < threshold

    Args:
        date: ISO date string of the analysis (e.g. "2018-07-24").
        coords: WGS84 bounding box dict.
        search_start_days: Start of search window (days before date).
        search_end_days: End of search window (days before date).
        scene_cloud_max: Skip dates with scene cloud above this %.
        token: Optional DES access token.

    Returns:
        List of BaselineCandidate sorted by cloud_fraction ascending.
    """
    from datetime import datetime, timedelta

    dt = datetime.strptime(date, "%Y-%m-%d")
    start = (dt - timedelta(days=search_end_days)).strftime("%Y-%m-%d")
    end = (dt - timedelta(days=search_start_days)).strftime("%Y-%m-%d")

    # Phase 1: STAC discovery
    print(f"  [STAC] Querying available dates {start} to {end}...")
    stac_dates = _stac_available_dates(coords, start, end, scene_cloud_max)
    print(f"  [STAC] {len(stac_dates)} dates with scene cloud <= {scene_cloud_max:.0f}%")

    if not stac_dates:
        return []

    # Phase 2: Fetch RGB+SCL per date
    conn = _connect(token=token)
    projected_coords = _to_nmd_grid(coords)

    results = []
    for i, (candidate_date, scene_cloud) in enumerate(stac_dates, 1):
        candidate_dt = datetime.strptime(candidate_date, "%Y-%m-%d")
        temporal = [
            candidate_date,
            (candidate_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
        ]

        try:
            rgb, aoi_cloud, shape = _fetch_rgb_and_scl(
                conn, projected_coords, temporal
            )
            print(f"    [{i}/{len(stac_dates)}] {candidate_date}  "
                  f"scene={scene_cloud:.1f}%  AOI(SCL)={aoi_cloud:.1%}  "
                  f"[{shape[1]}x{shape[0]}]")

            results.append(BaselineCandidate(
                date=candidate_date,
                cloud_fraction=aoi_cloud,
                scene_cloud_fraction=scene_cloud,
                rgb_thumbnail=rgb,
                shape=shape,
            ))

        except Exception as e:
            print(f"    [{i}/{len(stac_dates)}] {candidate_date}  "
                  f"scene={scene_cloud:.1f}%  -> failed ({e})")
            continue

    results.sort(key=lambda c: c.cloud_fraction)
    return results


def _fetch_rgb_and_scl(conn, projected_coords: dict, temporal: list):
    """Fetch B02/B03/B04 + SCL for a single date. Lightweight RGB thumbnail.

    Returns:
        Tuple of (rgb_array, cloud_fraction, shape).
    """
    import rasterio

    cube_rgb = conn.load_collection(
        collection_id=COLLECTION,
        spatial_extent=projected_coords,
        temporal_extent=temporal,
        bands=["b02", "b03", "b04"],
    )

    cube_scl = conn.load_collection(
        collection_id=COLLECTION,
        spatial_extent=projected_coords,
        temporal_extent=temporal,
        bands=BANDS_20M_CATEGORICAL,
    )
    cube_scl = cube_scl.resample_cube_spatial(target=cube_rgb, method="near")

    cube = cube_rgb.merge_cubes(cube_scl)
    data = cube.download(format="gtiff")

    if not data:
        raise FetchError("DES returned empty data")

    with rasterio.open(io.BytesIO(data)) as src:
        raw = src.read()  # (4, H, W): b02, b03, b04, scl

    b02 = dn_to_reflectance(raw[0], source="des")
    b03 = dn_to_reflectance(raw[1], source="des")
    b04 = dn_to_reflectance(raw[2], source="des")
    scl = raw[3].astype(np.uint8)

    cloud_frac = check_cloud_fraction(scl)

    band_dict = {"B02": b02, "B03": b03, "B04": b04}
    rgb = bands_to_rgb(band_dict, scl=scl)
    shape = rgb.shape[:2]

    return rgb, cloud_frac, shape


def run_cot_on_candidates(
    candidates: list[BaselineCandidate],
    coords: dict,
    top_n: int = 5,
    token: str | None = None,
) -> list[BaselineCandidate]:
    """Fetch full bands for top N candidates and run COT analysis.

    Fetches all 11 spectral bands needed by COT, runs the analyzer,
    and populates the ``cot_stats`` field. Returns candidates re-sorted
    by best visibility (highest clear_fraction, then lowest cot_mean).

    Args:
        candidates: Pre-sorted by cloud_fraction (from scan_baseline_candidates).
        coords: WGS84 bounding box dict.
        top_n: How many top candidates to run COT on.
        token: Optional DES access token.

    Returns:
        The top_n candidates with cot_stats populated, sorted by visibility.
    """
    from datetime import datetime, timedelta
    from imint.analyzers.cot import COTAnalyzer
    import rasterio

    top = candidates[:top_n]
    if not top:
        return []

    conn = _connect(token=token)
    projected_coords = _to_nmd_grid(coords)

    analyzer = COTAnalyzer(config={"device": "cpu"})
    print(f"\n  [COT] Running visibility analysis on top {len(top)} candidates...")

    for i, c in enumerate(top, 1):
        dt = datetime.strptime(c.date, "%Y-%m-%d")
        temporal = [c.date, (dt + timedelta(days=1)).strftime("%Y-%m-%d")]

        try:
            # Full spectral fetch (all 11 bands)
            cube_10m = conn.load_collection(
                collection_id=COLLECTION,
                spatial_extent=projected_coords,
                temporal_extent=temporal,
                bands=BANDS_10M,
            )
            cube_20m = conn.load_collection(
                collection_id=COLLECTION,
                spatial_extent=projected_coords,
                temporal_extent=temporal,
                bands=BANDS_20M_SPECTRAL,
            )
            cube_20m = cube_20m.resample_cube_spatial(
                target=cube_10m, method="bilinear"
            )
            cube_60m = conn.load_collection(
                collection_id=COLLECTION,
                spatial_extent=projected_coords,
                temporal_extent=temporal,
                bands=BANDS_60M,
            )
            cube_60m = cube_60m.resample_cube_spatial(
                target=cube_10m, method="bilinear"
            )
            cube = cube_10m.merge_cubes(cube_20m).merge_cubes(cube_60m)
            data = cube.download(format="gtiff")

            with rasterio.open(io.BytesIO(data)) as src:
                raw = src.read()

            spectral_names = BANDS_10M + BANDS_20M_SPECTRAL + BANDS_60M
            bands = {}
            for j, name in enumerate(spectral_names):
                bands[name.upper()] = dn_to_reflectance(raw[j], source="des")
            # B8A uppercase fix
            if "B8A" not in bands and "b8a".upper() in bands:
                pass  # already uppercase
            bands = des_to_imint_bands(
                {n.lower(): dn_to_reflectance(raw[j], source="des")
                 for j, n in enumerate(spectral_names)}
            )

            # Run COT
            rgb_dummy = c.rgb_thumbnail
            result = analyzer.analyze(
                rgb=rgb_dummy, bands=bands, date=c.date, output_dir="/tmp",
            )

            if result.success:
                c.cot_stats = result.outputs["stats"]
                cf = c.cot_stats["clear_fraction"]
                cm = c.cot_stats["cot_mean"]
                print(f"    [{i}/{len(top)}] {c.date}  "
                      f"clear={cf:.1%}  COT_mean={cm:.6f}")
            else:
                print(f"    [{i}/{len(top)}] {c.date}  COT failed: {result.error}")

        except Exception as e:
            print(f"    [{i}/{len(top)}] {c.date}  COT fetch failed: {e}")

    # Sort by visibility: highest clear_fraction, then lowest cot_mean
    def _visibility_key(c):
        if c.cot_stats is None:
            return (0.0, 1.0)
        return (c.cot_stats["clear_fraction"], -c.cot_stats["cot_mean"])

    top.sort(key=_visibility_key, reverse=True)
    return top


def _baseline_area_key(coords: dict) -> str:
    """Stable string key from bbox (matches change_detection._area_key)."""
    return f"{coords['west']}_{coords['south']}_{coords['east']}_{coords['north']}"


def ensure_baseline(
    date: str,
    coords: dict,
    output_dir: str,
    search_start_days: int = 30,
    search_end_days: int = 90,
    cloud_threshold: float = 0.1,
    token: str | None = None,
) -> str | None:
    """Ensure a cloud-free baseline exists for this area.

    Checks if a baseline .npy file already exists at the expected path
    (matching the area naming used by ChangeDetectionAnalyzer).
    If not, fetches a cloud-free image from DES and saves it.

    Args:
        date: ISO date string of the analysis.
        coords: WGS84 bounding box dict.
        output_dir: The job's output directory (baselines stored at ../baselines/).
        search_start_days: Start of search window in days before date.
        search_end_days: End of search window in days before date.
        cloud_threshold: Maximum acceptable cloud fraction for baseline.
        token: Optional DES access token.

    Returns:
        Path to the baseline .npy file (existing or newly created),
        or None if fetching failed (logged as warning, not raised).
    """
    area = _baseline_area_key(coords)
    baseline_dir = os.path.join(output_dir, "..", "baselines")
    baseline_path = os.path.join(baseline_dir, f"{area}.npy")

    if os.path.exists(baseline_path):
        print(f"  [baseline] Using existing baseline: {baseline_path}")
        return baseline_path

    try:
        print(f"  [baseline] No baseline found for area {area}")
        print(f"  [baseline] Scanning {search_start_days}-{search_end_days} days back...")
        result = fetch_cloud_free_baseline(
            date=date,
            coords=coords,
            search_start_days=search_start_days,
            search_end_days=search_end_days,
            cloud_threshold=cloud_threshold,
            token=token,
        )

        os.makedirs(baseline_dir, exist_ok=True)
        np.save(baseline_path, result.rgb)
        if result.scl is not None:
            scl_path = baseline_path.replace(".npy", "_scl.npy")
            np.save(scl_path, result.scl)

        # Save multispectral bands stack for dNBR / change detection
        _CHANGE_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
        if result.bands and all(b in result.bands for b in _CHANGE_BANDS):
            bands_stack = np.stack(
                [result.bands[b] for b in _CHANGE_BANDS], axis=-1
            ).astype(np.float32)
            bands_path = baseline_path.replace(".npy", "_bands.npy")
            np.save(bands_path, bands_stack)
            print(f"  [baseline] Saved multispectral bands: {bands_path}")

        # Save geospatial transform so change detection can align grids
        if result.geo and result.geo.transform:
            import json as _json
            geo_path = baseline_path.replace(".npy", "_geo.json")
            with open(geo_path, "w") as _f:
                _json.dump({
                    "transform": list(result.geo.transform)[:6],
                    "crs": str(result.geo.crs),
                    "shape": list(result.rgb.shape[:2]),
                }, _f, indent=2)
            print(f"  [baseline] Saved geo transform: {geo_path}")

        print(f"  [baseline] Saved cloud-free baseline: {baseline_path}")
        return baseline_path

    except FetchError as e:
        print(f"  [baseline] WARNING: Could not fetch baseline: {e}")
        print(f"  [baseline] Change detection will create baseline from current image.")
        return None


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


# ── Sjökort (Nautical chart from SLU GET) ─────────────────────────────────────

# SLU GET API for ordering S-57 nautical chart data from Sjöfartsverket.
# Requires a Shibboleth-authenticated session (browser cookies).
# Chart data is in S-57 (.000) vector format, delivered in EPSG:4326.
# Max area per order: ~2.5 km² (workload 2 = "orange" in SLU GET UI).
# For larger areas the bbox must be tiled into ≤2.5 km² pieces.

SLU_GET_BASE_URL = "https://maps.slu.se/get"
SLU_GET_DOWNLOAD_URL = "https://maps.slu.se/get/done"
SLU_GET_JOB_ID = "sjokort_vektor"
SLU_GET_MAX_AREA_M2 = 2_500_000  # ~2.5 km²
SLU_GET_DEFAULT_MARGIN_M = 1000  # 1 km margin around bbox

# Default cache directory (project root/.sjokort_cache)
SJOKORT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".sjokort_cache"
)

# S-57 layer rendering config — IHO S-52 inspired colour scheme
# Depth ranges (DRVAL1/DRVAL2 midpoint) → hex colour
_S57_DEPTH_COLOURS = [
    (0,   3,   "#B0DEF5"),   # Very shallow — bright cyan-blue
    (3,   6,   "#C2E6F8"),   # Shallow
    (6,   10,  "#D0ECF9"),   # Moderate shallow
    (10,  20,  "#DDF2FB"),   # Moderate
    (20,  30,  "#E6F5FC"),   # Deeper
    (30,  50,  "#EDF8FD"),   # Deep
    (50,  999, "#EDF8FD"),   # Very deep — same as BG (no visible border)
]
_S57_LAND_COLOUR = "#F5E6C8"       # Warm tan/sand
_S57_COASTLINE_COLOUR = "#2C2C2C"  # Near-black
_S57_DEPCNT_COLOUR = "#7CADC8"     # Blue-grey contour lines
_S57_FAIRWAY_COLOUR = "#C9ABE0"    # Light purple
_S57_BG_COLOUR = "#EDF8FD"         # Default sea — matches DEPARE 30-50 m
_S57_BUILDING_COLOUR = "#C8AD8A"   # Darker tan for buildings
_S57_ROAD_COLOUR = "#999999"       # Grey for roads
_S57_SLCONS_COLOUR = "#555555"     # Dark grey for shoreline constructions
_S57_BRIDGE_COLOUR = "#444444"     # Dark for bridges
_S57_NAVLNE_COLOUR = "#AA5599"     # Magenta for navigation lines
_S57_SOUNDG_COLOUR = "#3366AA"     # Blue for sounding labels
_S57_OBSTRN_COLOUR = "#CC3333"     # Red for obstructions
_S57_WRECKS_COLOUR = "#333333"     # Dark for wrecks

# S-57 COLOUR attribute mapping (IHO S-57 Appendix A, Chapter 2)
_S57_COLOUR_MAP = {
    "1": "#FFFFFF",   # white
    "2": "#000000",   # black
    "3": "#FF0000",   # red
    "4": "#00AA00",   # green
    "5": "#0000FF",   # blue
    "6": "#FFCC00",   # yellow
    "7": "#888888",   # grey
    "8": "#8B4513",   # brown
    "9": "#FFBF00",   # amber
    "10": "#8B008B",  # violet
    "11": "#FF8800",  # orange
    "12": "#FF00FF",  # magenta
    "13": "#FFB6C1",  # pink
}

# Rendering colours for light sectors (semi-transparent fills)
_SECTOR_FILL = {
    "3": (1.0, 0.0, 0.0, 0.18),   # red
    "4": (0.0, 0.7, 0.0, 0.18),   # green
    "1": (1.0, 1.0, 0.6, 0.12),   # white (pale yellow tint)
    "6": (1.0, 0.8, 0.0, 0.15),   # yellow
}
_SECTOR_EDGE = {
    "3": "#CC0000",
    "4": "#006600",
    "1": "#AA8800",
    "6": "#CC8800",
}


def _load_svg_symbol(svg_path: str, flip_y: bool = True):
    """Parse an OpenSeaMap SVG file into a normalised matplotlib Path.

    Extracts all ``<path d="...">`` elements, combines them, centres on
    the origin, and scales to the range *-1 … 1*.  Y is flipped by
    default because SVG has *y-down* while matplotlib markers use *y-up*.

    Returns ``None`` if the file cannot be read or contains no paths.
    """
    import re
    import numpy as np
    from svgpath2mpl import parse_path
    from matplotlib.path import Path as MPath

    try:
        with open(svg_path) as fh:
            svg = fh.read()
    except OSError:
        return None

    paths_data = re.findall(r'<path[^>]*\bd="([^"]+)"', svg)
    if not paths_data:
        return None

    all_verts: list[np.ndarray] = []
    all_codes: list[int] = []
    for d in paths_data:
        try:
            p = parse_path(d)
            all_verts.append(p.vertices.copy())
            all_codes.extend(list(p.codes))
        except Exception:
            continue

    if not all_verts:
        return None

    verts = np.vstack(all_verts)
    xmin, ymin = verts.min(axis=0)
    xmax, ymax = verts.max(axis=0)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    scale = max(xmax - xmin, ymax - ymin) / 2
    if scale < 1e-6:
        scale = 1.0
    verts = (verts - [cx, cy]) / scale
    if flip_y:
        verts[:, 1] *= -1

    return MPath(verts, all_codes)


def _int1_symbols() -> dict:
    """Load INT1/S-52 nautical chart symbols from OpenSeaMap SVGs.

    Returns a dict mapping symbol name → ``matplotlib.path.Path`` usable
    as a ``marker`` in ``ax.scatter()`` / ``ax.plot()``.

    Symbols are loaded from ``data/symbols/*.svg`` (OpenSeaMap renderer
    project, archived 2021).  Each SVG is parsed with *svgpath2mpl* and
    normalised to the coordinate range -1 … 1.

    If an SVG file is missing a simple hand-coded fallback is used so
    the renderer never crashes.
    """
    import os
    import math
    from matplotlib.path import Path as MPath

    MOVETO = MPath.MOVETO
    LINETO = MPath.LINETO
    CLOSEPOLY = MPath.CLOSEPOLY

    symbols: dict[str, MPath] = {}

    # ── Locate SVG directory ────────────────────────────────────────
    svg_dir = os.path.join(os.path.dirname(__file__), "..", "data", "symbols")
    svg_dir = os.path.normpath(svg_dir)

    def _svg(name: str):
        return _load_svg_symbol(os.path.join(svg_dir, f"{name}.svg"))

    # ── Buoys (pillar / spar pole symbols) ─────────────────────────
    symbols["buoy_can"]      = _svg("Pillar")     # port / lateral
    symbols["buoy_cone"]     = _svg("Spar")       # starboard / lateral
    symbols["buoy_cardinal"] = _svg("Pillar")     # cardinal
    symbols["buoy_special"]  = _svg("Float")      # special purpose

    # ── Beacon ──────────────────────────────────────────────────────
    symbols["beacon"] = _svg("Beacon")

    # ── Light ───────────────────────────────────────────────────────
    symbols["light"] = _svg("Light")

    # ── Wreck ───────────────────────────────────────────────────────
    symbols["wreck"] = _svg("WreckD")

    # ── Rock ────────────────────────────────────────────────────────
    symbols["rock"] = _svg("Rock")

    # ── Obstruction ─────────────────────────────────────────────────
    # Reuse rock cross — good + symbol
    symbols["obstruction"] = _svg("Rock")

    # ── Landmark ────────────────────────────────────────────────────
    symbols["landmark"] = _svg("Tower") or _svg("Church")

    # ── Mooring / bollard ───────────────────────────────────────────
    symbols["mooring"] = _svg("Bollard")

    # ── Harbour ─────────────────────────────────────────────────────
    symbols["harbour"] = _svg("Harbour")

    # ── Topmarks ────────────────────────────────────────────────────
    symbols["topmark"]       = _svg("Top_Can")
    symbols["topmark_can"]   = _svg("Top_Can")
    symbols["topmark_cone"]  = _svg("Top_Cone")
    symbols["topmark_x"]     = _svg("Top_X")
    symbols["topmark_north"] = _svg("Top_North")
    symbols["topmark_south"] = _svg("Top_South")

    # ── Fallbacks for any missing SVGs ──────────────────────────────
    # Simple cross for rock / obstruction
    _cross = MPath(
        [(0, -0.8), (0, 0.8), (0, 0),
         (-0.8, 0), (0.8, 0), (0, 0)],
        [MOVETO, LINETO, MOVETO,
         MOVETO, LINETO, MOVETO],
    )
    # Simple triangle for beacon / generic
    _tri = MPath(
        [(0, -1), (0, 0.1), (-0.5, 0.1), (0, 0.9),
         (0.5, 0.1), (-0.5, 0.1)],
        [MOVETO, LINETO, MOVETO, LINETO, LINETO, CLOSEPOLY],
    )
    # Simple filled circle for buoys
    import numpy as _np
    _circ_pts = [(0.8 * math.cos(t), 0.8 * math.sin(t))
                 for t in _np.linspace(0, 2 * math.pi, 20)]
    _circ_codes = [MOVETO] + [LINETO] * 18 + [CLOSEPOLY]
    _circ = MPath(_circ_pts, _circ_codes)

    _fallbacks = {
        "buoy_can": _circ, "buoy_cone": _circ,
        "buoy_cardinal": _circ, "buoy_special": _circ,
        "beacon": _tri, "light": _tri,
        "wreck": _cross, "rock": _cross,
        "obstruction": _cross, "landmark": _tri,
        "mooring": _circ, "harbour": _cross,
        "topmark": _tri, "topmark_can": _tri,
        "topmark_cone": _tri, "topmark_x": _cross,
        "topmark_north": _tri, "topmark_south": _tri,
    }
    for key, fallback in _fallbacks.items():
        if symbols.get(key) is None:
            symbols[key] = fallback

    return symbols


@dataclass
class SjokortTile:
    """A single sjökort order tile with SWEREF99 TM bounds.

    Attributes:
        north: Max Y in EPSG:3006.
        south: Min Y in EPSG:3006.
        east:  Max X in EPSG:3006.
        west:  Min X in EPSG:3006.
        uuid:  Server-assigned UUID after successful order (None before).
        area_m2: Tile area in square meters.
    """
    north: int
    south: int
    east: int
    west: int
    uuid: str | None = None
    area_m2: int = 0

    def __post_init__(self):
        self.area_m2 = (self.east - self.west) * (self.north - self.south)


@dataclass
class SjokortFetchResult:
    """Result of a sjökort fetch operation.

    Attributes:
        s57_paths: Deduplicated list of unique S-57 file paths (one per ENC cell).
        tiles:     List of SjokortTile objects with order metadata.
        bbox_wgs84: Original WGS84 bounding box (without margin).
        bbox_sweref: NMD-aligned SWEREF99 TM bounding box (without margin).
        bbox_sweref_padded: SWEREF99 TM bbox including margin (used for ordering).
        rendered_png: Path to rendered PNG if ``render=True`` was used, else None.
        from_cache: True if loaded from local cache.
    """
    s57_paths: list[Path]
    tiles: list[SjokortTile]
    bbox_wgs84: dict
    bbox_sweref: dict
    bbox_sweref_padded: dict | None = None
    rendered_png: Path | None = None
    from_cache: bool = False


def _sjokort_cache_key(coords: dict) -> str:
    """Generate a deterministic cache key from SWEREF99 TM coordinates."""
    key_str = f"sjokort_{coords['west']}_{coords['south']}_{coords['east']}_{coords['north']}"
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def _tile_sjokort_bbox(projected_coords: dict, max_area: int = SLU_GET_MAX_AREA_M2) -> list[SjokortTile]:
    """Split a SWEREF99 TM bbox into tiles that fit under the SLU GET area limit.

    Tiles are aligned to the NMD 10m grid. The number of columns and rows is
    chosen to keep each tile under ``max_area`` while using the fewest splits.

    Args:
        projected_coords: SWEREF99 TM bbox with keys west, south, east, north.
        max_area: Maximum tile area in m² (default ~2.5 km²).

    Returns:
        List of SjokortTile objects covering the full bbox.
    """
    W = projected_coords["west"]
    S = projected_coords["south"]
    E = projected_coords["east"]
    N = projected_coords["north"]
    total_w = E - W
    total_h = N - S
    total_area = total_w * total_h

    if total_area <= max_area:
        return [SjokortTile(north=N, south=S, east=E, west=W)]

    # Find minimum number of columns and rows
    grid = NMD_GRID_SIZE
    n_cols = 1
    n_rows = 1
    while True:
        tile_w = total_w / n_cols
        tile_h = total_h / n_rows
        if tile_w * tile_h <= max_area:
            break
        # Split along the longer dimension
        if tile_w >= tile_h:
            n_cols += 1
        else:
            n_rows += 1

    # Calculate tile dimensions snapped to NMD grid
    col_width = math.floor(total_w / n_cols / grid) * grid
    row_height = math.floor(total_h / n_rows / grid) * grid

    tiles = []
    for r in range(n_rows):
        for c in range(n_cols):
            tile_w = W + c * col_width
            tile_e = W + (c + 1) * col_width if c < n_cols - 1 else E
            tile_s = S + r * row_height
            tile_n = S + (r + 1) * row_height if r < n_rows - 1 else N
            tiles.append(SjokortTile(north=tile_n, south=tile_s, east=tile_e, west=tile_w))

    return tiles


def _order_sjokort_tile(
    tile: SjokortTile,
    email: str,
    session: Any,
) -> SjokortTile:
    """Submit a single sjökort order via the SLU GET API.

    Args:
        tile: SjokortTile with SWEREF99 TM bounds.
        email: E-mail address for the download link.
        session: ``requests.Session`` with Shibboleth cookies.

    Returns:
        The same tile with ``uuid`` populated on success.

    Raises:
        FetchError: If the order submission fails.
    """
    url = (
        f"{SLU_GET_BASE_URL}/api/job/{SLU_GET_JOB_ID}"
        f"/{tile.north}/{tile.south}/{tile.east}/{tile.west}/{email}"
    )
    try:
        resp = session.post(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        uuid = data.get("Uuid")
        if not uuid:
            raise FetchError(f"SLU GET order failed — no Uuid in response: {data}")
        tile.uuid = uuid
        return tile
    except FetchError:
        raise
    except Exception as e:
        raise FetchError(f"SLU GET order request failed: {e}")


def _download_sjokort_zip(
    uuid: str,
    dest_dir: Path,
    session: Any | None = None,
    max_retries: int = 20,
    retry_delay: float = 30.0,
) -> Path:
    """Download a processed sjökort ZIP from SLU GET.

    Polls the download URL until the ZIP is ready (server returns HTML
    while processing, ZIP when done).  Uses the same Shibboleth session
    as the ordering step (maps.slu.se requires it).

    Args:
        uuid: Order UUID from the API response.
        dest_dir: Directory to save the ZIP file.
        session: ``requests.Session`` with Shibboleth cookies (recommended).
                 Falls back to a plain ``requests.get`` if None.
        max_retries: Maximum number of download attempts.
        retry_delay: Seconds between retries.

    Returns:
        Path to the downloaded ZIP file.

    Raises:
        FetchError: If download fails after all retries.
    """
    import time
    import requests

    url = f"{SLU_GET_DOWNLOAD_URL}/{uuid}.zip"
    zip_path = dest_dir / f"{uuid}.zip"
    getter = session if session is not None else requests

    for attempt in range(max_retries):
        try:
            resp = getter.get(url, timeout=60)
            resp.raise_for_status()
            # SLU returns HTML while processing, ZIP when ready
            content_type = resp.headers.get("Content-Type", "")
            if "html" in content_type or resp.content[:4] != b"PK\x03\x04":
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise FetchError(
                    f"Sjökort ZIP not ready after {max_retries * retry_delay:.0f}s "
                    f"for UUID {uuid}"
                )
            zip_path.write_bytes(resp.content)
            return zip_path
        except FetchError:
            raise
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            raise FetchError(f"Failed to download sjökort ZIP {uuid}: {e}")

    raise FetchError(f"Sjökort download exhausted retries for {uuid}")


def _extract_s57_from_zip(zip_path: Path, dest_dir: Path) -> list[Path]:
    """Extract S-57 (.000) files from a sjökort ZIP archive.

    Args:
        zip_path: Path to the ZIP file.
        dest_dir: Directory to extract S-57 files into.

    Returns:
        List of extracted S-57 file paths.
    """
    import zipfile

    s57_paths = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith(".000"):
                extracted = dest_dir / Path(name).name
                extracted.write_bytes(zf.read(name))
                s57_paths.append(extracted)

    return s57_paths


def _pad_sweref_bbox(coords: dict, margin_m: int) -> dict:
    """Expand a SWEREF99 TM bbox by ``margin_m`` metres on each side, snapped to NMD grid.

    Args:
        coords: SWEREF99 TM bbox with keys west, south, east, north.
        margin_m: Margin in metres to add on each side.

    Returns:
        Expanded bbox dict (same format, re-snapped to NMD 10m grid).
    """
    grid = NMD_GRID_SIZE
    return {
        "west": math.floor((coords["west"] - margin_m) / grid) * grid,
        "south": math.floor((coords["south"] - margin_m) / grid) * grid,
        "east": math.ceil((coords["east"] + margin_m) / grid) * grid,
        "north": math.ceil((coords["north"] + margin_m) / grid) * grid,
        "crs": coords.get("crs", TARGET_CRS),
    }


def _deduplicate_s57(s57_paths: list[Path]) -> list[Path]:
    """Remove duplicate S-57 files across tiles (same ENC cell name).

    SLU GET delivers full ENC cells for every overlapping tile order. This
    function keeps only one copy per unique filename (stem), preferring the
    first occurrence.

    Args:
        s57_paths: All extracted S-57 file paths (may contain duplicates).

    Returns:
        Deduplicated list of paths (one per unique ENC cell).
    """
    seen: dict[str, Path] = {}
    for p in s57_paths:
        if p.stem not in seen:
            seen[p.stem] = p
    return sorted(seen.values())


def _depth_colour(drval1: float | None, drval2: float | None) -> str:
    """Map S-57 DEPARE depth range to an IHO-inspired colour hex string."""
    mid = ((drval1 or 0) + (drval2 or 0)) / 2.0
    for lo, hi, col in _S57_DEPTH_COLOURS:
        if lo <= mid < hi:
            return col
    return _S57_BG_COLOUR


def _best_scale_prefix(s57_paths: list, bbox_wgs84: dict) -> str | None:
    """Auto-select the most detailed ENC scale whose data covers the AOI.

    Swedish ENC cells use prefixes that encode the navigational purpose:

        SE2 — Overview        (1:1 000 000)
        SE3 — General         (1:350 000)
        SE4 — Harbour/Approach (1:90 000)
        SE5 — Berthing        (1:22 000)

    The function groups the supplied S-57 files by prefix, then tests
    each group starting from the most detailed (highest number).  For
    each scale the DEPARE (depth area) and LNDARE (land area) polygons
    are unioned and intersected with the AOI.  If the coverage ratio
    is ≥ 95 % the scale is selected.

    Actual geometry coverage is checked — not the cell metadata extent
    (M_COVR) — so scales with gaps in depth data (e.g. open water
    outside a harbour cell) are correctly rejected.

    Returns:
        The chosen prefix string (e.g. ``"SE4"``) or ``None`` if no
        single scale covers the AOI sufficiently.
    """
    import fiona
    import geopandas as gpd
    from collections import defaultdict
    from shapely.geometry import box as _box
    from shapely.ops import unary_union

    W = bbox_wgs84["west"]
    S = bbox_wgs84["south"]
    E = bbox_wgs84["east"]
    N = bbox_wgs84["north"]
    aoi = _box(W, S, E, N)
    aoi_area = aoi.area

    # Gruppera filer per prefix (SE2, SE3, SE4, SE5, …)
    groups: dict[str, list] = defaultdict(list)
    for p in s57_paths:
        stem = Path(p).stem.upper()
        prefix = stem[:3]           # "SE4", "SE3", etc.
        groups[prefix].append(p)

    # Testa varje grupp — högsta nummer (mest detaljerad) först
    for prefix in sorted(groups.keys(), reverse=True):
        paths = groups[prefix]
        dep_polys = []
        lnd_polys = []
        for p in paths:
            try:
                lyrs = fiona.listlayers(str(p))
                if "DEPARE" in lyrs:
                    gdf = gpd.read_file(str(p), layer="DEPARE",
                                        bbox=(W, S, E, N))
                    if not gdf.empty:
                        dep_polys.extend(
                            g for g in gdf.geometry if g and g.is_valid
                        )
                if "LNDARE" in lyrs:
                    gdf = gpd.read_file(str(p), layer="LNDARE",
                                        bbox=(W, S, E, N))
                    if not gdf.empty:
                        lnd_polys.extend(
                            g for g in gdf.geometry if g and g.is_valid
                        )
            except Exception:
                continue

        if not dep_polys and not lnd_polys:
            continue

        # Union av DEPARE + LNDARE klippt mot AOI
        all_polys = dep_polys + lnd_polys
        coverage = unary_union(all_polys).intersection(aoi)
        ratio = coverage.area / aoi_area if aoi_area > 0 else 0

        if ratio >= 0.95:
            return prefix

    return None          # Ingen skala täcker ≥95 %


def render_sjokort_png(
    s57_paths: list[Path],
    bbox_wgs84: dict | None,
    output_path: Path,
    img_w: int = 0,
    img_h: int = 0,
    *,
    scale_prefix: str | None = None,
) -> Path:
    """Render S-57 nautical chart data to a PNG image.

    Reads key layers from ENC cell files and renders them with IHO S-52
    inspired colours, clipped to the given WGS84 bounding box.

    **CRS handling — critical for pixel alignment:**

    S-57 ENC data is natively in WGS84 (EPSG:4326), but the Sentinel-2
    RGB image is in SWEREF99 TM (EPSG:3006).  If the sjökort were rendered
    directly in WGS84, its pixel grid would not align with the RGB because
    the two coordinate systems distort space differently (geographic degrees
    vs projected metres).

    To guarantee exact pixel alignment the function:

    1. Reads S-57 features using the WGS84 bbox (native CRS of the ENC
       files) and clips them to the AOI.
    2. Reprojects every GeoDataFrame to the target CRS read from
       ``bands_meta.json`` (normally ``EPSG:3006``).  Fiona-based point
       features (lights, buoys, labels) are transformed via ``pyproj``.
    3. Sets the matplotlib axes to the **projected bounds** from the
       metadata (``bounds_projected``) so that the rendered image covers
       exactly the same metric rectangle as the RGB.

    If ``bands_meta.json`` is unavailable, projected bounds are computed
    on-the-fly from the WGS84 bbox using ``pyproj``.

    If *bbox_wgs84* is ``None``, or *img_w*/*img_h* are 0, the
    function reads ``bands_meta.json`` from the same output directory
    (using the date prefix from *output_path*) and uses its
    ``bounds_wgs84`` and ``shape`` so that the chart matches the
    Sentinel-2 RGB pixel grid exactly.

    Rendering order (bottom to top):
        1. Sea background
        2. DEPARE — depth areas (polygon fills, colour by depth)
        3. FAIRWY — fairway areas (semi-transparent)
        4. DRGARE — dredged areas
        5. LNDARE — land areas (warm tan fill)
        6. BUISGL — buildings on land
        7. DEPCNT — depth contour lines
        8. SLCONS — shoreline constructions (piers, breakwaters)
        9. COALNE — coastline
        10. ROADWY — roads
        11. BRIDGE — bridges
        12. NAVLNE — navigation lines
        13. SOUNDG — depth soundings (point text labels)
        14. OBSTRN — obstructions (markers)
        15. WRECKS — wrecks (markers)

    Args:
        s57_paths: Deduplicated list of S-57 (.000) file paths.
        bbox_wgs84: WGS84 bounding box ``{west, south, east, north}``,
            or ``None`` to read from bands_meta.json.
        output_path: Where to save the output PNG.
        img_w: Target image width in pixels (0 = read from meta).
        img_h: Target image height in pixels (0 = read from meta).
        scale_prefix: Optional ENC scale prefix filter (e.g. "SE5") —
            only files whose stem starts with this prefix are rendered.

    Returns:
        Path to the rendered PNG file.

    Raises:
        FetchError: If rendering fails (e.g. no data within bbox).
    """
    import warnings

    try:
        import geopandas as gpd
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, Wedge
        from shapely.geometry import box, MultiPoint
        from PIL import Image
        import math
    except ImportError as e:
        raise FetchError(
            f"Sjökort rendering requires geopandas, matplotlib and Pillow: {e}"
        )

    warnings.filterwarnings("ignore", message=".*Skipping field.*")
    warnings.filterwarnings("ignore", category=UserWarning)

    # ── Read bands_meta.json for rendering parameters ────────────
    # Always read meta for projected bounds (CRS alignment).  Also use
    # it as fallback for bbox / image dimensions when not supplied.
    import json as _json
    _out_dir = Path(output_path).parent
    _meta_candidates = sorted(_out_dir.glob("bands/*bands_meta.json"))
    _bm: dict = {}
    _geo: dict = {}
    if _meta_candidates:
        with open(_meta_candidates[0]) as _mf:
            _bm = _json.load(_mf)
        _geo = _bm.get("geo", {})
    elif bbox_wgs84 is None or img_w == 0 or img_h == 0:
        raise FetchError(
            "bbox_wgs84 is None and no bands_meta.json found in "
            f"{_out_dir}/bands/"
        )

    if bbox_wgs84 is None:
        bbox_wgs84 = (
            _bm.get("bounds_wgs84")
            or _geo.get("bounds_wgs84")
            or _bm.get("coords")
        )
    if img_w == 0 or img_h == 0:
        _shape = _bm.get("shape") or _geo.get("shape") or [573, 324]
        img_h, img_w = _shape[0], _shape[1]

    # Projected CRS + bounds — the RGB image is in EPSG:3006 (SWEREF99 TM).
    # The sjökort MUST render in the same CRS so pixels align perfectly.
    target_crs = _geo.get("crs", "EPSG:3006")
    _proj_bounds = _geo.get("bounds_projected")

    # ── Filter paths by scale prefix ──────────────────────────────
    all_s57_paths = list(s57_paths)
    if scale_prefix:
        s57_paths = [
            p for p in s57_paths
            if Path(p).stem.upper().startswith(scale_prefix.upper())
        ]
        # Fallback cells: coarser scales for gap-fill (e.g. SE3 behind SE4)
        _fallback_paths = [p for p in all_s57_paths if p not in s57_paths]
    else:
        _fallback_paths = []
    if not s57_paths:
        raise FetchError(
            f"No S-57 files match scale_prefix={scale_prefix!r}"
        )

    # WGS84 bbox — used ONLY as a spatial filter when reading S-57 data
    # (the native CRS of ENC files).  NOT used for clipping — see below.
    W = bbox_wgs84["west"]
    S_ = bbox_wgs84["south"]
    E = bbox_wgs84["east"]
    N = bbox_wgs84["north"]

    # Projected rendering bounds (e.g. EPSG:3006 / SWEREF99 TM).
    # These define the matplotlib axes so the output pixel grid matches
    # the Sentinel-2 RGB exactly.  Prefer the snapped bounds stored in
    # bands_meta.json; fall back to a pyproj transform if unavailable.
    if _proj_bounds:
        pW = _proj_bounds["west"]
        pS = _proj_bounds["south"]
        pE = _proj_bounds["east"]
        pN = _proj_bounds["north"]
    else:
        from pyproj import Transformer as _Tr
        _t = _Tr.from_crs("EPSG:4326", target_crs, always_xy=True)
        pW, pS = _t.transform(W, S_)
        pE, pN = _t.transform(E, N)

    # Clip box in the PROJECTED CRS.  Clipping must happen after
    # reprojection — clipping in WGS84 then reprojecting creates curved
    # edges that leave thin gaps at the image border.
    clip_box = box(pW, pS, pE, pN)

    # ── Helper: read & clip one layer ─────────────────────────────
    def _read_layer(
        path: Path,
        layer: str,
        geom_types: list[str] | None = None,
    ) -> gpd.GeoDataFrame:
        import fiona
        try:
            if layer not in fiona.listlayers(str(path)):
                return gpd.GeoDataFrame()
            # Read with WGS84 bbox as spatial filter (S-57 native CRS).
            gdf = gpd.read_file(str(path), layer=layer,
                                bbox=(W, S_, E, N))
            if gdf.empty:
                return gdf
            gdf = gdf[gdf.geometry.notna()]
            if geom_types and not gdf.empty:
                gdf = gdf[gdf.geometry.geom_type.isin(geom_types)]
            # Reproject WGS84 → target CRS (e.g. EPSG:3006 SWEREF99 TM)
            # BEFORE clipping.  Clipping in WGS84 then reprojecting
            # produces curved edges that don't align with the projected
            # axis limits, leaving thin gaps at the image border.
            if not gdf.empty:
                if gdf.crs is None:
                    gdf = gdf.set_crs("EPSG:4326")
                gdf = gdf.to_crs(target_crs)
            # Clip in the projected CRS — straight edges match the axes.
            if not gdf.empty:
                gdf = gpd.clip(gdf, clip_box)
            return gdf
        except Exception:
            return gpd.GeoDataFrame()

    # ── Helper: accumulate a layer from all files ─────────────────
    def _collect(layer: str, geom_types: list[str] | None = None):
        parts = []
        for f in s57_paths:
            gdf = _read_layer(f, layer, geom_types)
            if not gdf.empty:
                parts.append(gdf)
        if parts:
            return pd.concat(parts, ignore_index=True)
        return gpd.GeoDataFrame()

    # ── Collect all layers ────────────────────────────────────────
    poly_t = ["Polygon", "MultiPolygon"]
    line_t = ["LineString", "MultiLineString"]
    point_t = ["Point", "MultiPoint"]

    # ── Polygon layers (bottom) ───────────────────────────────────
    dep_gdf = _collect("DEPARE", poly_t)     # Depth areas
    fwy_gdf = _collect("FAIRWY", poly_t)     # Fairways
    drg_gdf = _collect("DRGARE", poly_t)     # Dredged areas
    swp_gdf = _collect("SWPARE", poly_t)     # Swept areas
    res_gdf = _collect("RESARE", poly_t)     # Restricted areas
    ctn_gdf = _collect("CTNARE", poly_t)     # Caution areas
    ach_gdf = _collect("ACHARE", poly_t)     # Anchorage areas
    mar_gdf = _collect("MARCUL", poly_t)     # Marine cultivation
    tes_gdf = _collect("TESARE", poly_t)     # Territorial sea
    lnd_gdf = _collect("LNDARE", poly_t)     # Land areas
    bua_gdf = _collect("BUAARE", poly_t)     # Built-up areas
    bui_gdf = _collect("BUISGL", poly_t)     # Buildings

    # ── Line layers (middle) ──────────────────────────────────────
    dpc_gdf = _collect("DEPCNT", line_t)     # Depth contours
    slc_gdf = _collect("SLCONS", line_t)     # Shoreline constructions
    coa_gdf = _collect("COALNE", line_t)     # Coastline
    rwy_gdf = _collect("ROADWY", line_t)     # Roads
    brg_gdf = _collect("BRIDGE", line_t)     # Bridges
    nav_gdf = _collect("NAVLNE", line_t)     # Navigation lines
    rec_gdf = _collect("RECTRC", line_t)     # Recommended tracks
    fer_gdf = _collect("FERYRT", line_t)     # Ferry routes
    cbs_gdf = _collect("CBLSUB", line_t)     # Submarine cables
    cbo_gdf = _collect("CBLOHD", line_t)     # Overhead cables
    pip_gdf = _collect("PIPSOL", line_t)     # Pipelines
    elv_gdf = _collect("LNDELV", line_t)     # Land elevation contours
    rdl_gdf = _collect("RDOCAL", line_t)     # Radio calling lines

    # ── Point layers (top) ────────────────────────────────────────
    snd_gdf = _collect("SOUNDG", point_t)    # Soundings (depth values)
    uwr_gdf = _collect("UWTROC", point_t)    # Underwater rocks
    obs_gdf = _collect("OBSTRN", point_t)    # Obstructions
    wrk_gdf = _collect("WRECKS", point_t)    # Wrecks
    # ── Fiona-based readers for layers with List-type attributes ──
    import fiona as _fiona
    _LITCHR_MAP = {1: "F", 2: "Fl", 3: "LFl", 4: "Q", 5: "VQ",
                   6: "UQ", 7: "Oc", 8: "Iso", 9: "Mo"}
    _COL_LETTER = {"1": "W", "3": "R", "4": "G", "6": "Y"}

    def _fiona_points(layer):
        """Read point features via fiona (handles List[str] COLOUR)."""
        results = []
        for fp in s57_paths:
            try:
                if layer not in _fiona.listlayers(str(fp)):
                    continue
                with _fiona.open(str(fp), layer=layer) as src:
                    for feat in src:
                        geom = feat.get("geometry")
                        if not geom or geom["type"] != "Point":
                            continue
                        x, y = geom["coordinates"][:2]
                        if not (W <= x <= E and S_ <= y <= N):
                            continue
                        p = feat["properties"]
                        col_list = p.get("COLOUR") or []
                        results.append({"x": x, "y": y, "props": p,
                                        "colour": col_list})
            except Exception:
                pass
        return results

    # Lights (with sectors/characteristics)
    lit_features = []
    for item in _fiona_points("LIGHTS"):
        p = item["props"]
        col_code = item["colour"][0] if item["colour"] else "1"
        lit_features.append({
            "x": item["x"], "y": item["y"],
            "sectr1": p.get("SECTR1"), "sectr2": p.get("SECTR2"),
            "colour": col_code,
            "valnmr": p.get("VALNMR"), "litchr": p.get("LITCHR"),
            "sigper": p.get("SIGPER"), "siggrp": p.get("SIGGRP"),
            "objnam": p.get("OBJNAM"),
        })

    # Buoys with colour (BOYLAT, BOYSPP, BOYCAR)
    buoy_features = []
    for layer in ("BOYLAT", "BOYSPP", "BOYCAR"):
        for item in _fiona_points(layer):
            col_code = item["colour"][0] if item["colour"] else "0"
            buoy_features.append({
                "x": item["x"], "y": item["y"],
                "colour": col_code, "layer": layer,
            })

    # Place names (LNDRGN, SEAARE)
    place_labels = []
    for layer in ("LNDRGN", "SEAARE"):
        for item in _fiona_points(layer):
            nm = item["props"].get("OBJNAM", "")
            if nm:
                place_labels.append({
                    "x": item["x"], "y": item["y"],
                    "name": nm, "type": layer,
                })
    bcn_gdf = _collect("BCNSPP", point_t)    # Beacons
    blt_gdf = _collect("BOYLAT", point_t)    # Lateral buoys
    bsp_gdf = _collect("BOYSPP", point_t)    # Special purpose buoys
    bca_gdf = _collect("BOYCAR", point_t)    # Cardinal buoys
    top_gdf = _collect("TOPMAR", point_t)    # Topmarks
    lmk_gdf = _collect("LNDMRK", point_t)   # Landmarks
    lrg_gdf = _collect("LNDRGN", point_t)    # Land region
    sea_gdf = _collect("SEAARE", point_t)    # Sea area names
    sba_gdf = _collect("SBDARE", point_t)    # Seabed area
    mor_gdf = _collect("MORFAC", point_t)    # Mooring facility
    ber_gdf = _collect("BERTHS", point_t)    # Berths
    hbr_gdf = _collect("HRBFAC", point_t)    # Harbour facility
    sil_gdf = _collect("SILTNK", point_t)    # Silo/tank
    pil_gdf = _collect("PILPNT", point_t)    # Pile/post
    abr_gdf = _collect("ACHBRT", point_t)    # Anchor berth

    # ── Reproject fiona-based point data (WGS84 → target CRS) ──
    # Lights, buoys and place labels are read via fiona (not geopandas)
    # to handle List-type attributes.  Their (x, y) coordinates are
    # still in WGS84 and must be transformed to the projected CRS so
    # they land on the correct pixel when plotted on the projected axes.
    from pyproj import Transformer as _Transformer
    _wgs_to_proj = _Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    for lf in lit_features:
        lf["x"], lf["y"] = _wgs_to_proj.transform(lf["x"], lf["y"])
    for bf in buoy_features:
        bf["x"], bf["y"] = _wgs_to_proj.transform(bf["x"], bf["y"])
    for pl in place_labels:
        pl["x"], pl["y"] = _wgs_to_proj.transform(pl["x"], pl["y"])

    # ── Figure setup ──────────────────────────────────────────────
    # Axes use projected bounds (pW, pS, pE, pN) so the pixel grid
    # matches the Sentinel-2 RGB image exactly.  aspect="auto" lets
    # matplotlib stretch the projected rectangle to fill img_w × img_h.
    dpi = 200
    fig_w = img_w / dpi
    fig_h = img_h / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # axes fills entire figure — no margins
    ax.set_xlim(pW, pE)
    ax.set_ylim(pS, pN)
    ax.set_aspect("auto")
    ax.axis("off")
    fig.patch.set_facecolor(_S57_BG_COLOUR)
    ax.set_facecolor(_S57_BG_COLOUR)

    # ── Plot helpers ──────────────────────────────────────────────
    def _plot_poly(gdf, colour, edgecolour="none", alpha=1.0, lw=0):
        if gdf.empty:
            return
        gpd.GeoDataFrame(gdf, geometry="geometry").plot(
            ax=ax, color=colour, edgecolor=edgecolour,
            alpha=alpha, linewidth=lw,
        )

    def _plot_line(gdf, colour, lw=0.4, alpha=1.0, ls="-"):
        if gdf.empty:
            return
        gpd.GeoDataFrame(gdf, geometry="geometry").plot(
            ax=ax, color=colour, linewidth=lw, alpha=alpha, linestyle=ls,
        )

    def _plot_points(gdf, colour, marker="o", size=4, lw=0.4,
                     alpha=0.9, zorder=10, edgecolor=None):
        if gdf.empty:
            return
        xs = [g.x for g in gdf.geometry
              if g and g.geom_type == "Point"]
        ys = [g.y for g in gdf.geometry
              if g and g.geom_type == "Point"]
        if xs:
            kw = dict(c=colour, s=size, marker=marker,
                      linewidths=lw, zorder=zorder, alpha=alpha)
            if edgecolor:
                kw["edgecolors"] = edgecolor
            ax.scatter(xs, ys, **kw)

    # Load INT1 symbol markers
    int1 = _int1_symbols()
    ms = max(4, img_w / 100)  # base marker size (compact for clean chart look)

    # ══════════════════════════════════════════════════════════════
    # RENDER ORDER — bottom to top
    # ══════════════════════════════════════════════════════════════

    # ── 0. Fallback base layer (coarser ENC cells) ──────────────
    # Render DEPARE + LNDARE + COALNE from fallback cells first so
    # that gaps in the primary (harbour-scale) data are filled with
    # coarser but complete coverage.
    if _fallback_paths:
        for fp in _fallback_paths:
            _fb_dep = _read_layer(fp, "DEPARE", poly_t)
            if not _fb_dep.empty:
                _fb_dep = gpd.GeoDataFrame(_fb_dep, geometry="geometry")
                _fb_dep["_colour"] = _fb_dep.apply(
                    lambda r: _depth_colour(r.get("DRVAL1"), r.get("DRVAL2")),
                    axis=1,
                )
                for colour in _fb_dep["_colour"].unique():
                    subset = _fb_dep[_fb_dep["_colour"] == colour]
                    subset.plot(ax=ax, color=colour, edgecolor="none")
            _fb_lnd = _read_layer(fp, "LNDARE", poly_t)
            if not _fb_lnd.empty:
                _plot_poly(_fb_lnd, _S57_LAND_COLOUR)
            _fb_coa = _read_layer(fp, "COALNE", line_t)
            if not _fb_coa.empty:
                gpd.GeoDataFrame(_fb_coa, geometry="geometry").plot(
                    ax=ax, color="#000000", linewidth=0.15, alpha=0.5,
                )

    # ── 1. Depth areas (DEPARE) ───────────────────────────────────
    if not dep_gdf.empty:
        dep_gdf = gpd.GeoDataFrame(dep_gdf, geometry="geometry")
        dep_gdf["_colour"] = dep_gdf.apply(
            lambda r: _depth_colour(r.get("DRVAL1"), r.get("DRVAL2")),
            axis=1,
        )
        for colour in dep_gdf["_colour"].unique():
            subset = dep_gdf[dep_gdf["_colour"] == colour]
            subset.plot(ax=ax, color=colour, edgecolor="none")

    # ── 2. Swept areas (SWPARE) ───────────────────────────────────
    _plot_poly(swp_gdf, "#D0E8F0", edgecolour="#90B8D0", alpha=0.4, lw=0.3)

    # ── 3. Fairways (FAIRWY) ─────────────────────────────────────
    _plot_poly(fwy_gdf, _S57_FAIRWAY_COLOUR, alpha=0.35,
               edgecolour=_S57_FAIRWAY_COLOUR, lw=0.3)

    # ── 4. Dredged areas (DRGARE) ────────────────────────────────
    _plot_poly(drg_gdf, "#C5E0F0", edgecolour="#8BB8D9", alpha=0.5, lw=0.3)

    # ── 5. Restricted areas (RESARE) ─────────────────────────────
    _plot_poly(res_gdf, "none", edgecolour="#CC4444", alpha=0.5, lw=0.6)

    # ── 6. Caution areas (CTNARE) ────────────────────────────────
    _plot_poly(ctn_gdf, "#FFEE88", edgecolour="#CCAA00", alpha=0.15, lw=0.4)

    # ── 7. Anchorage areas (ACHARE) ──────────────────────────────
    _plot_poly(ach_gdf, "#E0D0F0", edgecolour="#9977BB", alpha=0.2, lw=0.4)

    # ── 8. Marine cultivation (MARCUL) ───────────────────────────
    _plot_poly(mar_gdf, "#A8D8A8", edgecolour="#66AA66", alpha=0.3, lw=0.4)

    # ── 9. Territorial sea (TESARE) ──────────────────────────────
    _plot_poly(tes_gdf, "none", edgecolour="#666699", alpha=0.3, lw=0.3)

    # ── 10. Land (LNDARE) ────────────────────────────────────────
    _plot_poly(lnd_gdf, _S57_LAND_COLOUR)

    # ── 11. Built-up areas (BUAARE) ──────────────────────────────
    _plot_poly(bua_gdf, "#E8D8B0", edgecolour="#B8A880", alpha=0.7, lw=0.2)

    # ── 12. Buildings (BUISGL) ───────────────────────────────────
    _plot_poly(bui_gdf, _S57_BUILDING_COLOUR, edgecolour="#8A7A5A", lw=0.2)

    # ── 13. Land elevation contours (LNDELV) ─────────────────────
    _plot_line(elv_gdf, "#C0A880", lw=0.2, alpha=0.4)

    # ── 14. Depth contours (DEPCNT) ──────────────────────────────
    _plot_line(dpc_gdf, _S57_DEPCNT_COLOUR, lw=0.25, alpha=0.6)

    # ── 15. Submarine cables (CBLSUB) ────────────────────────────
    _plot_line(cbs_gdf, "#CC44CC", lw=0.3, alpha=0.5, ls="--")

    # ── 16. Overhead cables (CBLOHD) ─────────────────────────────
    _plot_line(cbo_gdf, "#CC2222", lw=0.3, alpha=0.5, ls="--")

    # ── 17. Pipelines (PIPSOL) ───────────────────────────────────
    _plot_line(pip_gdf, "#44AA44", lw=0.3, alpha=0.5, ls="--")

    # ── 18. Shoreline constructions (SLCONS) ─────────────────────
    _plot_line(slc_gdf, _S57_SLCONS_COLOUR, lw=0.3)

    # ── 19. Coastline (COALNE) ───────────────────────────────────
    _plot_line(coa_gdf, _S57_COASTLINE_COLOUR, lw=0.25)

    # ── 20. Roads (ROADWY) ───────────────────────────────────────
    _plot_line(rwy_gdf, _S57_ROAD_COLOUR, lw=0.3, alpha=0.7)

    # ── 21. Bridges (BRIDGE) ─────────────────────────────────────
    _plot_line(brg_gdf, _S57_BRIDGE_COLOUR, lw=0.8)

    # ── 22. Navigation lines (NAVLNE) ────────────────────────────
    _plot_line(nav_gdf, _S57_NAVLNE_COLOUR, lw=0.4, alpha=0.6, ls="--")

    # ── 23. Recommended tracks (RECTRC) ──────────────────────────
    _plot_line(rec_gdf, "#228833", lw=0.5, alpha=0.6)

    # ── 24. Ferry routes (FERYRT) ────────────────────────────────
    _plot_line(fer_gdf, "#8844AA", lw=0.5, alpha=0.6, ls="-.")

    # ── 25. Radio calling lines (RDOCAL) ─────────────────────────
    _plot_line(rdl_gdf, "#996633", lw=0.3, alpha=0.4, ls=":")

    # ── 26. Soundings — depth labels (SOUNDG) ────────────────────
    if not snd_gdf.empty:
        pts = []
        for _, row in snd_gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            if geom.geom_type == "MultiPoint":
                for pt in geom.geoms:
                    pts.append(pt)
            else:
                pts.append(geom)
        snd_fs = max(2.0, min(3.5, img_w / 200))
        for pt in pts:
            if pt.has_z:
                depth = abs(pt.z)
            else:
                continue
            depth_str = f"{depth:.1f}" if depth % 1 else f"{int(depth)}"
            ax.text(pt.x, pt.y, depth_str,
                    fontsize=snd_fs, color=_S57_SOUNDG_COLOUR,
                    ha="center", va="center", zorder=16)

    # ── 27. Underwater rocks (UWTROC) — INT1 K.14 cross ──────────
    _plot_points(uwr_gdf, "#333333", marker="+", size=ms * 0.6,
                 lw=0.4, alpha=0.7)

    # ── 28. Seabed area type (SBDARE) ────────────────────────────
    _plot_points(sba_gdf, "#8899AA", marker=".", size=ms * 0.3,
                 alpha=0.3, zorder=5)

    # ── 29. Obstructions (OBSTRN) — INT1 K.40 cross-in-circle ───
    _plot_points(obs_gdf, _S57_OBSTRN_COLOUR, marker=int1["obstruction"],
                 size=ms * 0.9, lw=0.3, alpha=0.8)

    # ── 30. Wrecks (WRECKS) — INT1 K.20 hull outline ────────────
    _plot_points(wrk_gdf, _S57_WRECKS_COLOUR, marker=int1["wreck"],
                 size=ms * 1.0, lw=0.3, alpha=0.85)

    # ── 31. Lights (LIGHTS) — sector arc outlines + symbol + labels ─
    if lit_features:
        from collections import defaultdict
        light_groups = defaultdict(list)
        for lf in lit_features:
            key = (round(lf["x"], 1), round(lf["y"], 1))
            light_groups[key].append(lf)

        # Arc radius in projected metres.  Because we render in a
        # metric CRS (SWEREF99 TM) the radius is simply a fraction of
        # the north–south extent — no cos(lat) correction is needed
        # (unlike WGS84 where longitude degrees are shorter than latitude).
        sector_r = (pN - pS) * 0.035
        sector_r_lg = (pN - pS) * 0.055

        _SECTOR_LINE = {"3": "#CC0000", "4": "#006600", "1": "#888888", "6": "#CC8800"}

        def _arc_points(cx, cy, r, brg1, brg2, n_pts=40):
            """Return (xs, ys) for a circular arc in projected (metric) coords."""
            if brg2 <= brg1:
                brg2 += 360.0
            xs, ys = [], []
            for i in range(n_pts + 1):
                brg = math.radians(brg1 + (brg2 - brg1) * i / n_pts)
                xs.append(cx + r * math.sin(brg))
                ys.append(cy + r * math.cos(brg))
            return xs, ys

        label_fs = max(3.0, min(5.0, img_w / 130))

        for (lx, ly), feats in light_groups.items():
            has_sectors = any(
                f["sectr1"] is not None and f["sectr2"] is not None
                for f in feats
            )
            if has_sectors:
                # Draw arc outline per sector
                for f in feats:
                    if f["sectr1"] is None or f["sectr2"] is None:
                        continue
                    col_code = f["colour"]
                    r = sector_r if not (f.get("valnmr") and f["valnmr"] >= 6) else sector_r_lg
                    colour = _SECTOR_LINE.get(col_code, "#888888")
                    xs, ys = _arc_points(lx, ly, r, f["sectr1"], f["sectr2"])
                    ax.plot(xs, ys, color=colour, linewidth=0.8,
                            alpha=0.85, zorder=14, solid_capstyle="round")
                    # Tick lines at sector boundaries
                    for bx, by in [(xs[0], ys[0]), (xs[-1], ys[-1])]:
                        ax.plot([lx, bx], [ly, by], color=colour,
                                linewidth=0.3, alpha=0.5, zorder=13)

                # Light symbol at center
                ax.scatter([lx], [ly], c="#000000", s=ms * 2.0,
                           marker=int1["light"], zorder=16,
                           edgecolors="none", linewidths=0)

                # Light characteristic label
                # Collect unique colours across all sectors
                all_cols = sorted(set(
                    _COL_LETTER.get(f["colour"], "") for f in feats
                    if f["sectr1"] is not None
                ))
                col_str = "".join(all_cols)
                f0 = feats[0]
                chr_str = _LITCHR_MAP.get(f0.get("litchr"), "")
                per_str = f"{f0['sigper']:.0f}s" if f0.get("sigper") else ""
                rng_str = f"{f0['valnmr']:.0f}M" if f0.get("valnmr") else ""
                nm = f0.get("objnam") or ""
                parts = [p for p in [chr_str, col_str, per_str, rng_str] if p]
                label = " ".join(parts)
                if nm:
                    # Name on one line, characteristic below
                    ax.text(lx, ly - (pN - pS) * 0.018, nm,
                            fontsize=label_fs, fontweight="bold",
                            color="#000000", ha="center", va="top", zorder=17)
                    ax.text(lx, ly - (pN - pS) * 0.035, label,
                            fontsize=label_fs * 0.85,
                            color="#000000", ha="center", va="top", zorder=17)
                elif label:
                    ax.text(lx, ly - (pN - pS) * 0.018, label,
                            fontsize=label_fs * 0.85,
                            color="#000000", ha="center", va="top", zorder=17)
            else:
                # All-round light — star symbol + label
                f0 = feats[0]
                col_code = f0.get("colour", "1")
                ax.scatter([lx], [ly], c="#000000", s=ms * 1.5,
                           marker=int1["light"], zorder=15,
                           edgecolors="none", linewidths=0)
                # Label: "name\nchr col rngM"
                nm = f0.get("objnam") or ""
                chr_str = _LITCHR_MAP.get(f0.get("litchr"), "")
                col_l = _COL_LETTER.get(col_code, "")
                rng = f"{f0['valnmr']:.0f}M" if f0.get("valnmr") else ""
                parts = [p for p in [chr_str, col_l, rng] if p]
                label = " ".join(parts)
                if nm or label:
                    txt = f"{nm}\n{label}" if nm and label else (nm or label)
                    ax.text(lx, ly - (pN - pS) * 0.015, txt,
                            fontsize=label_fs * 0.85,
                            color="#000000", ha="center", va="top", zorder=17)

    # ── 32. Beacons (BCNSPP) — INT1 Q.100 filled black ───────────
    _plot_points(bcn_gdf, "#000000", marker=int1["beacon"],
                 size=ms * 1.5, lw=0.3, alpha=0.9, zorder=14,
                 edgecolor="#000000")

    # ── 33–35. Buoys — colour-correct via fiona data ─────────────
    _BUOY_COLOURS = {"3": "#CC0000", "4": "#00AA00", "6": "#DDAA00",
                     "2": "#000000", "1": "#FFFFFF"}
    _BUOY_MARKERS = {
        "BOYLAT": {
            "1": int1["buoy_can"],    # cat 1 = port = can
            "2": int1["buoy_cone"],   # cat 2 = starboard = cone
        },
        "BOYSPP": int1["buoy_special"],
        "BOYCAR": int1["buoy_cardinal"],
    }
    buoy_label_fs = max(2.5, min(4.0, img_w / 150))
    for bf in buoy_features:
        col_hex = _BUOY_COLOURS.get(bf["colour"], "#888888")
        col_letter = _COL_LETTER.get(bf["colour"], "")
        layer = bf["layer"]
        if layer == "BOYLAT":
            # Determine can vs cone based on colour (port=red=can, stbd=green=cone)
            mkr = int1["buoy_can"] if bf["colour"] == "3" else int1["buoy_cone"]
        elif layer == "BOYSPP":
            mkr = int1["buoy_special"]
        else:
            mkr = int1["buoy_cardinal"]

        edge = "#880000" if bf["colour"] == "3" else (
               "#006600" if bf["colour"] == "4" else "#666600")
        ax.scatter([bf["x"]], [bf["y"]], c=col_hex, s=ms * 1.0,
                   marker=mkr, zorder=13, edgecolors=edge,
                   linewidths=0.3, alpha=0.9)
        # Colour letter label
        if col_letter:
            ax.text(bf["x"], bf["y"] - (pN - pS) * 0.008, col_letter,
                    fontsize=buoy_label_fs, fontweight="bold",
                    color=col_hex, ha="center", va="top", zorder=14)

    # ── 36. Topmarks (TOPMAR) — INT1 Q.9 day mark ───────────────
    _plot_points(top_gdf, "#DD4444", marker=int1["topmark"],
                 size=ms * 0.7, lw=0.3, alpha=0.75, zorder=12)

    # ── 37. Landmarks (LNDMRK) — INT1 E.10 tower/spire ──────────
    _plot_points(lmk_gdf, "#885522", marker=int1["landmark"],
                 size=ms * 0.8, lw=0.3, alpha=0.75, zorder=11)

    # ── 38. Mooring facilities (MORFAC) — bollard symbol ─────────
    _plot_points(mor_gdf, "#4466AA", marker=int1["mooring"],
                 size=ms * 0.7, lw=0.3, alpha=0.75, zorder=11)

    # ── 39. Harbour facilities (HRBFAC) — INT1 F.10 anchor ──────
    _plot_points(hbr_gdf, "#664499", marker=int1["harbour"],
                 size=ms * 1.0, lw=0.3, alpha=0.75, zorder=11)

    # ── 40. Berths (BERTHS) ──────────────────────────────────────
    _plot_points(ber_gdf, "#4488AA", marker="s", size=ms * 0.5,
                 lw=0.2, alpha=0.6, zorder=11)

    # ── 41. Silo/tank (SILTNK) ───────────────────────────────────
    _plot_points(sil_gdf, "#998877", marker="o", size=ms * 0.4,
                 lw=0.2, alpha=0.5, zorder=8)

    # ── 42. Pile/post (PILPNT) ───────────────────────────────────
    _plot_points(pil_gdf, "#666666", marker="|", size=ms * 0.5,
                 lw=0.2, alpha=0.6, zorder=9)

    # ── 43. Anchor berth (ACHBRT) ────────────────────────────────
    _plot_points(abr_gdf, "#7755AA", marker="P", size=ms * 0.7,
                 lw=0.2, alpha=0.6, zorder=10)

    # ── 44. Place name labels (LNDRGN + SEAARE) — disabled ────────
    # Names omitted to keep the chart clean at small sizes.

    # ── Save ──────────────────────────────────────────────────────
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path, dpi=dpi, pad_inches=0,
        facecolor=_S57_BG_COLOUR,
    )
    plt.close(fig)

    # Resize to exact target pixel dimensions
    img = Image.open(output_path)
    if (img.width, img.height) != (img_w, img_h):
        img = img.resize((img_w, img_h), Image.LANCZOS)
        img.save(output_path)

    return output_path


def fetch_sjokort_data(
    coords: dict,
    email: str | None = None,
    session: Any | None = None,
    cookies: dict | None = None,
    cache_dir: str | None = None,
    max_retries: int = 20,
    retry_delay: float = 30.0,
    margin_m: int = SLU_GET_DEFAULT_MARGIN_M,
    render: bool = False,
    output_path: str | Path | None = None,
    img_w: int = 324,
    img_h: int = 573,
    s57_dir: str | Path | None = None,
    scale_prefix: str | None = None,
) -> SjokortFetchResult:
    """Fetch sjökort (nautical chart) data from SLU GET for a given bounding box.

    The area is projected to SWEREF99 TM and snapped to the NMD 10m grid,
    expanded by ``margin_m`` metres on each side (for cropping flexibility),
    then split into tiles ≤ 2.5 km² for ordering. Each tile is submitted via
    the SLU GET API, downloaded, and extracted. Duplicate ENC cells across
    tiles are deduplicated automatically.

    Optionally renders the S-57 data to a PNG image via ``render=True``.

    **Authentication note:**
    SLU GET (maps.slu.se) uses Shibboleth SSO with httpOnly cookies that
    cannot be extracted from a browser via JavaScript. There are three
    ways to supply authentication:

    1. *Pre-extracted S-57 directory* — set ``s57_dir`` to a directory
       containing .000 files. The order/download steps are skipped entirely.
       This is the recommended approach when data has been downloaded via
       a browser session manually or through automation.

    2. *requests.Session with Shibboleth cookies* — pass a ``session``
       obtained via Selenium/Playwright that has navigated through the
       Shibboleth login flow, or manually copy cookies into a session.

    3. *Cookie dict* — pass ``cookies`` as a dict; a new session is created.

    S-57 files are cached locally by NMD-aligned (padded) bounding box.

    Args:
        coords: WGS84 bounding box dict with keys: west, south, east, north.
        email: E-mail address registered with SLU GET. Required unless
               ``s57_dir`` is provided or cache exists.
        session: Optional ``requests.Session`` with Shibboleth cookies.
                 If not provided, one is created from ``cookies``.
        cookies: Optional dict of Shibboleth cookies (used if session is None).
        cache_dir: Directory for the sjökort cache. Defaults to .sjokort_cache/.
        max_retries: Max download poll attempts per tile (default 20).
        retry_delay: Seconds between download retries (default 30).
        margin_m: Margin in metres to expand the bbox before ordering
                  (default 1000). Set to 0 to order exactly the input bbox.
        render: If True, render the S-57 data to a PNG at ``output_path``.
        output_path: Destination for the rendered PNG. Required when
                     ``render=True``. Accepts str or Path.
        img_w: Rendered image width in pixels (default 324).
        img_h: Rendered image height in pixels (default 573).
        s57_dir: Path to a directory of pre-extracted S-57 (.000) files.
                 When set, the order and download steps are skipped — the
                 function only deduplicates and optionally renders.
        scale_prefix: Optional ENC scale prefix filter (e.g. "SE5") —
                      only ENC cells whose filename starts with this prefix
                      are used for rendering. Useful to avoid mixing scales.

    Returns:
        SjokortFetchResult with S-57 file paths, tile metadata, and
        optionally the path to the rendered PNG.

    Raises:
        FetchError: If ordering, downloading, or rendering fails.
        ValueError: If ``render=True`` but ``output_path`` is not set,
                    or if no ``email`` / ``s57_dir`` / cache is available.
    """
    import logging
    import requests as req_lib

    log = logging.getLogger(__name__)

    if render and output_path is None:
        raise ValueError("output_path is required when render=True")
    if output_path is not None:
        output_path = Path(output_path)

    # Project and snap to NMD grid
    projected_coords = _to_nmd_grid(coords)
    log.info(
        "Sjökort bbox SWEREF99 TM: W=%d S=%d E=%d N=%d",
        projected_coords["west"],
        projected_coords["south"],
        projected_coords["east"],
        projected_coords["north"],
    )

    # Expand by margin for ordering (padded bbox)
    if margin_m > 0:
        padded_coords = _pad_sweref_bbox(projected_coords, margin_m)
        log.info(
            "Sjökort padded bbox (+%dm): W=%d S=%d E=%d N=%d",
            margin_m,
            padded_coords["west"],
            padded_coords["south"],
            padded_coords["east"],
            padded_coords["north"],
        )
    else:
        padded_coords = projected_coords

    resolved_cache = Path(cache_dir or SJOKORT_CACHE_DIR)

    # ── Mode 1: Pre-extracted S-57 directory ──────────────────────────
    if s57_dir is not None:
        s57_dir = Path(s57_dir)
        raw_files = sorted(s57_dir.glob("**/*.000"))
        if not raw_files:
            raise FetchError(f"No S-57 (.000) files found in {s57_dir}")
        unique_files = _deduplicate_s57(raw_files)
        log.info(
            "Sjökort from s57_dir: %d raw → %d unique S-57 files",
            len(raw_files), len(unique_files),
        )
        result = SjokortFetchResult(
            s57_paths=unique_files,
            tiles=[],
            bbox_wgs84=coords,
            bbox_sweref=projected_coords,
            bbox_sweref_padded=padded_coords if margin_m > 0 else None,
            from_cache=False,
        )
        if render:
            # Auto-select the most detailed ENC scale that covers the AOI
            if scale_prefix is None:
                scale_prefix = _best_scale_prefix(unique_files, coords)
            result.rendered_png = render_sjokort_png(
                unique_files, coords, output_path, img_w, img_h,
                scale_prefix=scale_prefix,
            )
            log.info("Sjökort rendered to %s", result.rendered_png)
        return result

    # ── Mode 2: Cache lookup (keyed on padded bbox) ───────────────────
    cache_key = _sjokort_cache_key(padded_coords)
    cache_s57_dir = resolved_cache / cache_key

    if cache_s57_dir.exists():
        cached_files = sorted(cache_s57_dir.glob("*.000"))
        if cached_files:
            unique_files = _deduplicate_s57(cached_files)
            log.info("Sjökort loaded from cache: %d S-57 files", len(unique_files))
            result = SjokortFetchResult(
                s57_paths=unique_files,
                tiles=[],
                bbox_wgs84=coords,
                bbox_sweref=projected_coords,
                bbox_sweref_padded=padded_coords if margin_m > 0 else None,
                from_cache=True,
            )
            if render:
                # Auto-select the most detailed ENC scale that covers the AOI
                if scale_prefix is None:
                    scale_prefix = _best_scale_prefix(unique_files, coords)
                result.rendered_png = render_sjokort_png(
                    unique_files, coords, output_path, img_w, img_h,
                    scale_prefix=scale_prefix,
                )
                log.info("Sjökort rendered to %s", result.rendered_png)
            return result

    # ── Mode 3: Full order + download via SLU GET ─────────────────────
    if email is None:
        raise ValueError(
            "email is required for SLU GET ordering. Provide email, "
            "or use s57_dir= for pre-downloaded data, "
            "or ensure data is cached."
        )

    # Create session if needed
    if session is None:
        session = req_lib.Session()
        if cookies:
            session.cookies.update(cookies)

    # Tile the padded area
    tiles = _tile_sjokort_bbox(padded_coords)
    log.info(
        "Sjökort area %.2f km² → %d tile(s)",
        sum(t.area_m2 for t in tiles) / 1e6,
        len(tiles),
    )

    # Submit orders
    for i, tile in enumerate(tiles):
        _order_sjokort_tile(tile, email, session)
        log.info(
            "  Tile %d/%d ordered: UUID=%s (%.2f km²)",
            i + 1, len(tiles), tile.uuid, tile.area_m2 / 1e6,
        )

    # Download and extract
    cache_s57_dir.mkdir(parents=True, exist_ok=True)
    zip_dir = resolved_cache / "zips"
    zip_dir.mkdir(parents=True, exist_ok=True)

    all_s57: list[Path] = []
    for i, tile in enumerate(tiles):
        log.info("  Downloading tile %d/%d (UUID=%s)...", i + 1, len(tiles), tile.uuid)
        zip_path = _download_sjokort_zip(
            tile.uuid, zip_dir, session=session,
            max_retries=max_retries, retry_delay=retry_delay,
        )
        s57_files = _extract_s57_from_zip(zip_path, cache_s57_dir)
        all_s57.extend(s57_files)
        log.info("  Extracted %d S-57 files from tile %d", len(s57_files), i + 1)

    # Deduplicate across tiles
    unique_files = _deduplicate_s57(all_s57)
    log.info(
        "Sjökort fetch complete: %d raw → %d unique S-57 files",
        len(all_s57), len(unique_files),
    )

    result = SjokortFetchResult(
        s57_paths=unique_files,
        tiles=tiles,
        bbox_wgs84=coords,
        bbox_sweref=projected_coords,
        bbox_sweref_padded=padded_coords if margin_m > 0 else None,
        from_cache=False,
    )

    if render:
        # Auto-select the most detailed ENC scale that covers the AOI
        if scale_prefix is None:
            scale_prefix = _best_scale_prefix(unique_files, coords)
        result.rendered_png = render_sjokort_png(
            unique_files, coords, output_path, img_w, img_h,
            scale_prefix=scale_prefix,
        )
        log.info("Sjökort rendered to %s", result.rendered_png)

    return result
