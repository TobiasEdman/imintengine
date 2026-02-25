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

# SCL cloud + shadow classes (Sentinel-2 L2A Scene Classification)
# 3 = cloud_shadow, 8 = cloud_medium_probability,
# 9 = cloud_high_probability, 10 = thin_cirrus
SCL_CLOUD_CLASSES = frozenset({3, 8, 9, 10})

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

    # 1. Explicit token
    if token:
        conn.authenticate_oidc_access_token(access_token=token, provider_id="egi")
        return conn

    # 2. Environment variable (used in Docker/ColonyOS)
    env_token = os.environ.get("DES_TOKEN")
    if env_token:
        conn.authenticate_oidc_access_token(access_token=env_token, provider_id="egi")
        return conn

    # 3. Basic Auth (DES_USER + DES_PASSWORD env vars)
    #    Compatible with DES community tutorial pattern.
    des_user = os.environ.get("DES_USER")
    des_password = os.environ.get("DES_PASSWORD")
    if des_user and des_password:
        conn.authenticate_basic(username=des_user, password=des_password)
        return conn

    # 4. Stored refresh token (from des_login.py --device)
    #    This auto-renews expired access tokens — best for local dev.
    try:
        conn.authenticate_oidc_refresh_token(
            provider_id="egi",
            store_refresh_token=True,
        )
        return conn
    except Exception:
        pass  # No stored refresh token, try next method

    # 5. Token file (short-lived access token from Web Editor)
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
        "  python scripts/des_login.py --basic     (tutorial-style basic auth)\n"
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


def _fetch_scl_batch(
    conn,
    projected_coords: dict,
    candidate_dates: list[str],
) -> list[tuple[str, float]]:
    """Fetch SCL for multiple dates in one DES call, return per-date cloud fractions.

    DES returns multi-date downloads as a ``.tar.gz`` archive containing
    one GeoTIFF per date (filename pattern ``out_YYYY_MM_DDT...tif``).
    This function extracts each TIF, parses the date from the filename,
    and computes ``check_cloud_fraction`` locally.

    One batch call replaces N separate openEO calls, saving connection,
    auth, and graph-compilation overhead.

    The temporal extent spans from the earliest to the latest candidate
    date. DES may return extra dates within that range; only dates that
    appear in *candidate_dates* are returned.

    Args:
        conn: openEO connection.
        projected_coords: EPSG:3006 bbox dict with ``"crs"`` key.
        candidate_dates: Date strings ordered by STAC cloud score
            (best first), e.g. ``["2018-08-11", "2018-08-13"]``.

    Returns:
        List of ``(date_str, cloud_fraction)`` in *candidate_dates*
        order (only dates found in the archive).

    Raises:
        FetchError: If DES returns empty/unparseable data.
    """
    import re
    import gzip
    import tarfile
    import rasterio
    from datetime import datetime as _dt, timedelta as _td

    if not candidate_dates:
        return []

    # Build temporal extent spanning all candidates
    sorted_cands = sorted(candidate_dates)
    dt_end = _dt.strptime(sorted_cands[-1], "%Y-%m-%d") + _td(days=1)
    temporal = [sorted_cands[0], dt_end.strftime("%Y-%m-%d")]

    # Load SCL, resample to 10m grid
    cube_ref = conn.load_collection(
        collection_id=COLLECTION,
        spatial_extent=projected_coords,
        temporal_extent=temporal,
        bands=["b02"],
    )
    cube_scl = conn.load_collection(
        collection_id=COLLECTION,
        spatial_extent=projected_coords,
        temporal_extent=temporal,
        bands=BANDS_20M_CATEGORICAL,
    )
    cube_scl = cube_scl.resample_cube_spatial(target=cube_ref, method="near")

    data = cube_scl.download(format="gtiff")
    if not data:
        raise FetchError("DES returned empty SCL batch data")

    # DES returns tar.gz for multi-date results
    cand_set = set(candidate_dates)
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
            if date_str not in cand_set:
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

    # Return in candidate_dates order (preserves STAC ranking)
    return [(d, cloud_by_date[d]) for d in candidate_dates if d in cloud_by_date]


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
