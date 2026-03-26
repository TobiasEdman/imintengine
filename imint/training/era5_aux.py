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
from collections import defaultdict
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
    """Fallback: Fetch ERA5 data using CDS API (slower, downloads full fields)."""
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
