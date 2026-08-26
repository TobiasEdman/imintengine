"""
imint/training/era5_aux.py — ERA5 weather auxiliary data via ECMWF Polytope

Fetches ERA5 reanalysis weather variables for crop training points using
ECMWF's Polytope feature extraction API. Weather context improves crop
classification — especially distinguishing visually similar crops
(wheat vs barley vs oats) that have different phenological responses
to temperature and moisture.

Variables fetched per point (growing season Apr-Sep):
  - 2m temperature (mean, min, max)
  - Total precipitation (cumulative)
  - Volumetric soil water layer 1 (0-7cm, mean)
  - Surface solar radiation downwards (cumulative)
  - Growing degree days (GDD, base 5°C)

Output format:
  Per point: dict with monthly aggregates + growing season summary
  Saved alongside .npz tiles as auxiliary channels

Requires:
    pip install polytope-client   # ECMWF Polytope client
    # OR: pip install cdsapi       # CDS API as fallback
    ECMWF API key in ~/.ecmwfapirc or ~/.cdsapirc

Usage:
    from imint.training.era5_aux import fetch_era5_for_points

    points = [{"lat": 59.33, "lon": 18.07, "year": 2022}, ...]
    weather = fetch_era5_for_points(points)
    # weather[0] = {"t2m_mean": 15.2, "tp_sum": 342.1, ...}
"""
from __future__ import annotations

import os
import json
import hashlib
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import numpy as np


# ERA5 parameters
ERA5_PARAMS = {
    "t2m": {
        "code": "167",
        "name": "2m_temperature",
        "unit": "K",
        "agg": "mean",
        "description": "2-metre temperature",
    },
    "tp": {
        "code": "228",
        "name": "total_precipitation",
        "unit": "m",
        "agg": "sum",
        "description": "Total precipitation",
    },
    "swvl1": {
        "code": "39",
        "name": "volumetric_soil_water_layer_1",
        "unit": "m³/m³",
        "agg": "mean",
        "description": "Soil moisture 0-7cm",
    },
    "ssrd": {
        "code": "169",
        "name": "surface_solar_radiation_downwards",
        "unit": "J/m²",
        "agg": "sum",
        "description": "Solar radiation (cumulative)",
    },
}

# Growing season months for Sweden
GROWING_SEASON = [4, 5, 6, 7, 8, 9]  # April-September


def check_polytope_available() -> bool:
    """Check if polytope-client is installed."""
    try:
        from polytope_client import Client  # noqa: F401
        return True
    except ImportError:
        return False


def check_cdsapi_available() -> bool:
    """Check if CDS API is installed (fallback)."""
    try:
        import cdsapi  # noqa: F401
        return True
    except ImportError:
        return False


def fetch_era5_for_points(
    points: list[dict],
    *,
    params: list[str] | None = None,
    use_polytope: bool = True,
    cache_dir: str | None = None,
) -> list[dict]:
    """Fetch ERA5 weather data for a list of crop training points.

    Each point must have 'lat', 'lon', and 'year' keys.

    Returns growing season (Apr-Sep) aggregates per point:
      - t2m_mean: mean 2m temperature (°C)
      - t2m_min: min monthly mean temperature (°C)
      - t2m_max: max monthly mean temperature (°C)
      - tp_sum: total precipitation (mm)
      - swvl1_mean: mean soil moisture (m³/m³)
      - ssrd_sum: total solar radiation (MJ/m²)
      - gdd: growing degree days (base 5°C)
      - monthly: dict of per-month values

    Args:
        points: List of dicts with lat, lon, year.
        params: ERA5 parameters to fetch (default: all).
        use_polytope: Try Polytope first, fallback to CDS API.
        cache_dir: Cache directory for API responses.

    Returns:
        List of weather dicts, one per input point.
    """
    if params is None:
        params = list(ERA5_PARAMS.keys())

    if use_polytope and check_polytope_available():
        return _fetch_via_polytope(points, params, cache_dir)
    elif check_cdsapi_available():
        return _fetch_via_cdsapi(points, params, cache_dir)
    else:
        print(
            "WARNING: Neither polytope-client nor cdsapi installed. "
            "Returning empty weather data. Install with:\n"
            "  pip install polytope-client   # preferred\n"
            "  pip install cdsapi            # fallback"
        )
        return [_empty_weather() for _ in points]


def _fetch_via_polytope(
    points: list[dict],
    params: list[str],
    cache_dir: str | None,
) -> list[dict]:
    """Fetch ERA5 data using ECMWF Polytope feature extraction.

    Polytope extracts timeseries directly from ECMWF's FDB store
    without downloading full fields — much faster for point queries.
    """
    from polytope_client import Client

    client = Client(address="polytope.ecmwf.int")

    results = []
    # Group points by year to batch requests
    by_year: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    for i, p in enumerate(points):
        by_year[p["year"]].append((i, p))

    # Initialize results
    results = [_empty_weather() for _ in points]

    for year, year_points in by_year.items():
        # Build coordinate lists
        lons = [p["lon"] for _, p in year_points]
        lats = [p["lat"] for _, p in year_points]

        for param_key in params:
            param_info = ERA5_PARAMS[param_key]

            request = {
                "class": "ea",
                "stream": "oper",
                "type": "an",
                "expver": "1",
                "levtype": "sfc",
                "param": param_info["code"],
                "date": f"{year}-04-01/to/{year}-09-30",
                "time": "12:00:00",
                "feature": {
                    "type": "timeseries",
                    "points": [[lon, lat] for lon, lat in zip(lons, lats)],
                },
            }

            try:
                # Check cache
                if cache_dir:
                    cache_path = os.path.join(
                        cache_dir,
                        f"era5_{param_key}_{year}_{len(year_points)}pts.npy",
                    )
                    if os.path.exists(cache_path):
                        data = np.load(cache_path, allow_pickle=True).item()
                        _merge_param_data(results, year_points, param_key, data)
                        continue

                result = client.retrieve("era5", request)

                # Parse Polytope response → per-point monthly values
                data = _parse_polytope_response(result, param_key, year)

                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)
                    np.save(cache_path, data)

                _merge_param_data(results, year_points, param_key, data)

            except Exception as e:
                print(f"  ERA5 Polytope error ({param_key}, {year}): {e}")
                continue

    # Compute derived fields
    for r in results:
        _compute_derived(r)

    return results


def _fetch_via_cdsapi(
    points: list[dict],
    params: list[str],
    cache_dir: str | None,
) -> list[dict]:
    """Fallback: Fetch ERA5 data using CDS API (slower, downloads full fields).

    KNOWN GOTCHA — if you hit this fallback path and the GRIB→xarray decode
    in _extract_grib_points() raises "did not find a match in any of xarray's
    currently installed IO backends", the cause is almost certainly a missing
    ``cftime`` package (transitive xarray dependency that pip doesn't pull
    on its own). Same symptom that bit the DES metafilter colleague —
    `pip install cftime` and re-run. We pinned cftime in requirements.txt
    for exactly this reason; this comment exists in case someone bypasses
    requirements.txt during a debugging session.
    """
    import cdsapi

    client = cdsapi.Client()

    results = [_empty_weather() for _ in points]
    by_year: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    for i, p in enumerate(points):
        by_year[p["year"]].append((i, p))

    for year, year_points in by_year.items():
        for param_key in params:
            param_info = ERA5_PARAMS[param_key]

            try:
                # CDS API: download monthly means for growing season
                target = os.path.join(
                    cache_dir or "/tmp",
                    f"era5_{param_key}_{year}.grib",
                )

                if not os.path.exists(target):
                    client.retrieve(
                        "reanalysis-era5-single-levels-monthly-means",
                        {
                            "product_type": "monthly_averaged_reanalysis",
                            "variable": param_info["name"],
                            "year": str(year),
                            "month": [f"{m:02d}" for m in GROWING_SEASON],
                            "time": "00:00",
                            "area": [70, 10, 55, 25],  # Sweden bbox (N,W,S,E)
                            "format": "grib",
                        },
                        target,
                    )

                # Extract point values from GRIB
                _extract_grib_points(
                    target, results, year_points, param_key,
                )

            except Exception as e:
                print(f"  ERA5 CDS error ({param_key}, {year}): {e}")
                continue

    for r in results:
        _compute_derived(r)

    return results


def _empty_weather() -> dict:
    """Return empty weather dict structure."""
    return {
        "t2m_mean": None,
        "t2m_min": None,
        "t2m_max": None,
        "tp_sum": None,
        "swvl1_mean": None,
        "ssrd_sum": None,
        "gdd": None,
        "monthly": {},
    }


def _parse_polytope_response(
    response: Any,
    param_key: str,
    year: int,
) -> dict:
    """Parse Polytope timeseries response into per-point monthly values."""
    # Polytope returns JSON with timeseries per point
    # Structure varies by response format — handle common cases
    data = {}
    try:
        if hasattr(response, "json"):
            resp_data = response.json()
        elif isinstance(response, dict):
            resp_data = response
        elif isinstance(response, bytes):
            import json
            resp_data = json.loads(response)
        else:
            resp_data = {"points": []}

        for i, point_data in enumerate(resp_data.get("points", [])):
            values = point_data.get("values", [])
            dates = point_data.get("dates", [])

            monthly: dict[int, list[float]] = defaultdict(list)
            for val, date_str in zip(values, dates):
                month = int(date_str.split("-")[1])
                if month in GROWING_SEASON:
                    monthly[month].append(float(val))

            data[i] = {
                month: np.mean(vals) if vals else None
                for month, vals in monthly.items()
            }
    except Exception:
        pass

    return data


def _merge_param_data(
    results: list[dict],
    year_points: list[tuple[int, dict]],
    param_key: str,
    data: dict,
) -> None:
    """Merge parameter data into results list."""
    for local_idx, (global_idx, _) in enumerate(year_points):
        if local_idx in data:
            monthly = data[local_idx]
            if param_key not in results[global_idx]["monthly"]:
                results[global_idx]["monthly"][param_key] = {}
            results[global_idx]["monthly"][param_key] = monthly


def _extract_grib_points(
    grib_path: str,
    results: list[dict],
    year_points: list[tuple[int, dict]],
    param_key: str,
) -> None:
    """Extract point values from a GRIB file."""
    try:
        import eccodes
    except ImportError:
        try:
            import cfgrib
            import xarray as xr
            ds = xr.open_dataset(grib_path, engine="cfgrib")
            param_info = ERA5_PARAMS[param_key]
            var_name = list(ds.data_vars)[0]

            for local_idx, (global_idx, pt) in enumerate(year_points):
                try:
                    point_data = ds[var_name].sel(
                        latitude=pt["lat"],
                        longitude=pt["lon"],
                        method="nearest",
                    )
                    monthly = {}
                    for month_idx, month in enumerate(GROWING_SEASON):
                        if month_idx < len(point_data):
                            monthly[month] = float(point_data.values[month_idx])

                    if param_key not in results[global_idx]["monthly"]:
                        results[global_idx]["monthly"][param_key] = {}
                    results[global_idx]["monthly"][param_key] = monthly
                except Exception:
                    continue
        except ImportError:
            print("  WARNING: Neither eccodes nor cfgrib available for GRIB parsing")


def _compute_derived(weather: dict) -> None:
    """Compute derived weather fields from monthly data."""
    monthly = weather.get("monthly", {})

    # Temperature
    t2m_monthly = monthly.get("t2m", {})
    if t2m_monthly:
        vals_k = [v for v in t2m_monthly.values() if v is not None]
        if vals_k:
            vals_c = [v - 273.15 for v in vals_k]  # K → °C
            weather["t2m_mean"] = round(np.mean(vals_c), 1)
            weather["t2m_min"] = round(min(vals_c), 1)
            weather["t2m_max"] = round(max(vals_c), 1)

            # Growing degree days (GDD, base 5°C)
            # Approximate: sum of (monthly_mean - 5) × 30 days
            gdd = sum(max(t - 5, 0) * 30 for t in vals_c)
            weather["gdd"] = round(gdd, 0)

    # Precipitation
    tp_monthly = monthly.get("tp", {})
    if tp_monthly:
        vals = [v for v in tp_monthly.values() if v is not None]
        if vals:
            # ERA5 tp is in meters, convert to mm
            weather["tp_sum"] = round(sum(vals) * 1000, 1)

    # Soil moisture
    swvl1_monthly = monthly.get("swvl1", {})
    if swvl1_monthly:
        vals = [v for v in swvl1_monthly.values() if v is not None]
        if vals:
            weather["swvl1_mean"] = round(np.mean(vals), 4)

    # Solar radiation
    ssrd_monthly = monthly.get("ssrd", {})
    if ssrd_monthly:
        vals = [v for v in ssrd_monthly.values() if v is not None]
        if vals:
            # ERA5 ssrd is in J/m², convert to MJ/m²
            weather["ssrd_sum"] = round(sum(vals) / 1e6, 1)


def weather_to_aux_channels(
    weather: dict,
    tile_shape: tuple[int, int] = (256, 256),
) -> dict[str, np.ndarray]:
    """Convert weather dict to spatial auxiliary channels for CropDataset.

    Weather values are uniform across the tile (same for all pixels)
    since ERA5 resolution (~30km) >> tile size (~2.5km).

    Returns:
        Dict of channel_name → (H, W) float32 arrays.
    """
    h, w = tile_shape
    channels = {}

    for key in ["t2m_mean", "tp_sum", "swvl1_mean", "ssrd_sum", "gdd"]:
        val = weather.get(key)
        if val is not None:
            channels[f"era5_{key}"] = np.full((h, w), val, dtype=np.float32)
        else:
            channels[f"era5_{key}"] = np.zeros((h, w), dtype=np.float32)

    return channels


# Normalisation constants for ERA5 auxiliary channels (empirical, Sweden)
ERA5_AUX_NORM = {
    "era5_t2m_mean": (12.0, 5.0),    # mean °C, std °C
    "era5_tp_sum": (350.0, 100.0),    # mean mm, std mm
    "era5_swvl1_mean": (0.25, 0.08),  # mean m³/m³, std
    "era5_ssrd_sum": (3500.0, 500.0), # mean MJ/m², std
    "era5_gdd": (1200.0, 400.0),      # mean GDD, std
}

ERA5_LAND_GRID_DEGREES = Decimal("0.1")
ERA5_ATMOSPHERE_GRID_DEGREES = Decimal("0.25")
# Open-Meteo serializes selected grid coordinates as float32-like JSON
# values (for example 59.300003 for the requested 59.3).  This tolerance is
# deliberately much smaller than half of the finest consumed ERA5 grid
# spacing (0.05 degrees), so it absorbs representation noise without accepting
# a neighbouring cell.
ERA5_API_CELL_COORD_ATOL_DEGREES = 1e-4


def era5_api_cell_coords_match(
    actual_lat: float,
    actual_lon: float,
    expected_lat: float,
    expected_lon: float,
) -> bool:
    """Return whether an API-selected cell matches its canonical grid cell.

    This is only for coordinates returned by Open-Meteo.  Request and source
    coordinates are identities under our control and must be checked more
    strictly by their callers.
    """
    try:
        actual = np.asarray((actual_lat, actual_lon), dtype=np.float64)
        expected = np.asarray((expected_lat, expected_lon), dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        np.isfinite(actual).all()
        and np.isfinite(expected).all()
        and np.allclose(
            actual,
            expected,
            rtol=0.0,
            atol=ERA5_API_CELL_COORD_ATOL_DEGREES,
        )
    )


def _snap_grid(value: float, resolution: Decimal) -> float:
    """Snap a coordinate with an explicit half-up rule.

    Python's built-in ``round`` uses ties-to-even, which can put points that
    lie exactly between two reanalysis cells into a different group across
    implementations.  The smoke cohort and the API request must share this
    one deterministic rule.
    """
    decimal_value = Decimal(str(float(value)))
    index = (decimal_value / resolution).quantize(
        Decimal("1"), rounding=ROUND_HALF_UP,
    )
    return float(index * resolution)


def era5_grid_context(lat: float, lon: float) -> dict[str, Any]:
    """Return the exact requested and model-specific nearest grid cells."""
    request_lat = _snap_grid(lat, ERA5_LAND_GRID_DEGREES)
    request_lon = _snap_grid(lon, ERA5_LAND_GRID_DEGREES)
    atmosphere_lat = _snap_grid(
        request_lat, ERA5_ATMOSPHERE_GRID_DEGREES,
    )
    atmosphere_lon = _snap_grid(
        request_lon, ERA5_ATMOSPHERE_GRID_DEGREES,
    )
    return {
        "request_lat": request_lat,
        "request_lon": request_lon,
        "land_cell": {"lat": request_lat, "lon": request_lon},
        "atmosphere_cell": {
            "lat": atmosphere_lat,
            "lon": atmosphere_lon,
        },
    }


def era5_atmosphere_cell_id(lat: float, lon: float) -> str:
    """ID used to group the cohort on the coarsest consumed ERA5 grid."""
    cell = era5_grid_context(lat, lon)["atmosphere_cell"]
    return format_era5_cell_id(cell["lat"], cell["lon"])


def format_era5_cell_id(lat: float, lon: float) -> str:
    """Format already-selected response coordinates without re-snapping."""
    return f"{float(lat):+07.2f},{float(lon):+07.2f}"


ERA5_REQUEST_SCHEMA = {
    "version": 5,
    "period": ["04-01", "09-30"],
    "timezone": "GMT",
    "elevation": "nan",
    "cell_selection": "nearest",
    "land_grid_degrees": float(ERA5_LAND_GRID_DEGREES),
    "atmosphere_grid_degrees": float(ERA5_ATMOSPHERE_GRID_DEGREES),
    "land_model": "era5_land",
    "land_daily": ["temperature_2m_mean"],
    "land_hourly": ["soil_moisture_0_to_7cm"],
    "atmosphere_model": "era5",
    "atmosphere_hourly": ["precipitation", "shortwave_radiation"],
}


def _request_with_retry(params: dict) -> dict:
    import requests

    last_error = None
    for attempt in range(4):
        try:
            response = requests.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params=params,
                timeout=120,
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError("ERA5 request failed after 4 attempts") from last_error


def _land_series_all_finite(land: dict) -> bool:
    """True if ERA5-Land returned usable (finite) values for both its series.

    ERA5-Land is a land-only product: over sea and large lakes it returns a
    correctly-sized series of NaNs rather than an error, and the request's
    ``cell_selection="nearest"`` snaps to the nearest GRID cell, not the
    nearest LAND cell. Size alone therefore cannot distinguish coverage from
    absence — which is why the caller must test finiteness before accepting.
    """
    for values in (
        land.get("daily", {}).get("temperature_2m_mean"),
        land.get("hourly", {}).get("soil_moisture_0_to_7cm"),
    ):
        arr = np.asarray(values if values is not None else [], dtype=np.float64)
        if arr.size == 0 or not np.isfinite(arr).all():
            return False
    return True


def _validate_era5_payload(
    payload: dict,
    *,
    year: int,
    expected_grid: dict[str, Any] | None = None,
) -> None:
    if payload.get("schema") != ERA5_REQUEST_SCHEMA:
        raise ValueError("ERA5 cache schema mismatch")
    request_grid = payload.get("request")
    if not isinstance(request_grid, dict):
        raise ValueError("ERA5 cache is missing its requested grid")
    if expected_grid is not None and request_grid != expected_grid:
        raise ValueError("ERA5 cache request grid mismatch")
    land = payload.get("land", {})
    atmosphere = payload.get("atmosphere", {})
    for model_name, response, expected_cell in (
        ("era5_land", land, request_grid["land_cell"]),
        ("era5", atmosphere, request_grid["atmosphere_cell"]),
    ):
        try:
            actual_lat = float(response["latitude"])
            actual_lon = float(response["longitude"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{model_name} response is missing its selected grid cell"
            ) from exc
        if not era5_api_cell_coords_match(
            actual_lat,
            actual_lon,
            expected_cell["lat"],
            expected_cell["lon"],
        ):
            raise ValueError(
                f"{model_name} selected unexpected cell "
                f"({actual_lat}, {actual_lon}); expected "
                f"({expected_cell['lat']}, {expected_cell['lon']})"
            )
    value_series = (
        land.get("daily", {}).get("temperature_2m_mean"),
        land.get("hourly", {}).get("soil_moisture_0_to_7cm"),
        atmosphere.get("hourly", {}).get("precipitation"),
        atmosphere.get("hourly", {}).get("shortwave_radiation"),
    )
    expected = (183, 4392, 4392, 4392)
    for values, count in zip(value_series, expected):
        arr = np.asarray(values if values is not None else [], dtype=np.float64)
        if arr.size != count or not np.isfinite(arr).all():
            # Report WHICH term failed. Testing size-or-finiteness while
            # printing only size produced "expected 183 finite values, got 183"
            # on 2026-08-26 and cost a debugging round-trip: the length was
            # right and the values were NaN, which the message could not say.
            n_bad = int((~np.isfinite(arr)).sum()) if arr.size else 0
            raise ValueError(
                f"Incomplete ERA5 series: expected {count} finite values, "
                f"got size={arr.size} with {n_bad} non-finite"
            )
    start = datetime(year, 4, 1)
    expected_daily = [
        (start + timedelta(days=index)).strftime("%Y-%m-%d")
        for index in range(183)
    ]
    expected_hourly = [
        (start + timedelta(hours=index)).strftime("%Y-%m-%dT%H:%M")
        for index in range(4392)
    ]
    time_series = (
        (land.get("daily", {}).get("time"), expected_daily),
        (land.get("hourly", {}).get("time"), expected_hourly),
        (atmosphere.get("hourly", {}).get("time"), expected_hourly),
    )
    for values, expected_values in time_series:
        if values != expected_values:
            raise ValueError("ERA5 timestamps are incomplete, duplicated, or unordered")


def fetch_era5_land_growing_season(
    lat: float,
    lon: float,
    year: int,
    *,
    cache_dir: str | Path,
    cutoff_date: str | None = None,
) -> dict[str, float]:
    """Fetch one ERA5-Land Apr-Sep summary and persist the raw response.

    Open-Meteo is used only as a transport for the explicitly selected
    ``era5_land`` reanalysis. Coordinates are snapped to its native 0.1-degree
    grid, making the cache reusable by nearby 5.12 km training tiles.
    """
    grid = era5_grid_context(lat, lon)
    lat = grid["request_lat"]
    lon = grid["request_lon"]
    cutoff_date = cutoff_date or f"{year}-09-30"
    if not (f"{year}-04-01" <= cutoff_date <= f"{year}-09-30"):
        raise ValueError(f"cutoff_date outside growing season: {cutoff_date}")
    cache_dir = Path(cache_dir)
    schema_hash = hashlib.sha256(
        json.dumps(ERA5_REQUEST_SCHEMA, sort_keys=True).encode()
    ).hexdigest()[:12]
    cache_path = cache_dir / (
        f"era5_aux_{schema_hash}_{lat:+06.1f}_{lon:+06.1f}_{year}.json"
    )
    payload = None
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text())
            _validate_era5_payload(
                payload, year=year, expected_grid=grid,
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            payload = None
    if payload is None:
        common = {
            "latitude": lat,
            "longitude": lon,
            "start_date": f"{year}-04-01",
            "end_date": f"{year}-09-30",
            "timezone": "GMT",
            "elevation": "nan",
            "cell_selection": "nearest",
        }
        land = _request_with_retry(
            {**common,
                "daily": "temperature_2m_mean",
                "hourly": "soil_moisture_0_to_7cm",
                "models": "era5_land",
            }
        )
        atmosphere = _request_with_retry(
            {**common,
                "hourly": "precipitation,shortwave_radiation",
                "models": "era5",
            }
        )
        # ERA5-Land has no ocean coverage, so a cell over sea or a large lake
        # comes back full-length and all-NaN. Fall back to the plain ERA5
        # reanalysis, which does cover water. Record WHICH model supplied the
        # values: this deliberately mixes two products, and a mix that is not
        # recorded is indistinguishable from one that never happened.
        land_model = "era5_land"
        if not _land_series_all_finite(land):
            land = _request_with_retry(
                {**common,
                    "daily": "temperature_2m_mean",
                    "hourly": "soil_moisture_0_to_7cm",
                    "models": "era5",
                }
            )
            land_model = "era5"
        payload = {
            "schema": ERA5_REQUEST_SCHEMA,
            "request": grid,
            "land": land,
            "land_model": land_model,
            "atmosphere": atmosphere,
        }
        _validate_era5_payload(payload, year=year, expected_grid=grid)
        cache_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, cache_path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    daily = payload["land"]["daily"]
    daily_mask = np.asarray(payload["land"]["daily"]["time"]) <= cutoff_date
    hourly_mask = np.asarray(payload["land"]["hourly"]["time"]) < f"{cutoff_date}T23:59"
    temps = np.asarray(daily["temperature_2m_mean"], dtype=np.float64)[daily_mask]
    hourly = payload["atmosphere"]["hourly"]
    precip = np.asarray(hourly["precipitation"], dtype=np.float64)[hourly_mask]
    # Hourly W/m2 mean over one hour -> 0.0036 MJ/m2.
    radiation = (np.asarray(hourly["shortwave_radiation"], dtype=np.float64)[hourly_mask]
                 * 0.0036)
    moisture = np.asarray(
        payload["land"]["hourly"]["soil_moisture_0_to_7cm"], dtype=np.float64,
    )[hourly_mask]
    return {
        "era5_t2m_mean": float(np.nanmean(temps)),
        "era5_tp_sum": float(np.nansum(precip)),
        "era5_swvl1_mean": float(np.nanmean(moisture)),
        "era5_ssrd_sum": float(np.nansum(radiation)),  # API unit: MJ/m2
        "era5_gdd": float(np.nansum(np.maximum(temps - 5.0, 0.0))),
        "era5_request_lat": float(grid["request_lat"]),
        "era5_request_lon": float(grid["request_lon"]),
        "era5_land_cell_lat": float(payload["land"]["latitude"]),
        "era5_land_cell_lon": float(payload["land"]["longitude"]),
        "era5_atmosphere_cell_lat": float(payload["atmosphere"]["latitude"]),
        "era5_atmosphere_cell_lon": float(payload["atmosphere"]["longitude"]),
    }
