"""
imint/training/optimal_fetch.py — Optimal Sentinel-2 candidate-day selection.

Chains the cheapest filter first so that costly downstream calls only run
on the few surviving days:

    Stage 1  ERA5 atmosphere prefilter  (free, AOI-aware ~30 km grid)
    Stage 2  SCL-stack screen           (1 openEO call, server-side aggregation)
    Stage 3  Spectral fetch             (caller's job; only the selected dates)

Stage 1 mirrors the metafilter pattern. Stage 2 mirrors
`scripts/batch_fetch_openeo.py:screen_tile_scl()` (which targets CDSE)
but routes through the same DES openEO endpoint our training pipeline
uses.

Usage
-----

    from imint.training.optimal_fetch import optimal_fetch_dates

    dates = optimal_fetch_dates(
        bbox_wgs84={"west": ..., "south": ..., "east": ..., "north": ...},
        date_start="2022-06-01",
        date_end="2022-08-31",
        mode="era5_then_scl",
        max_aoi_cloud=0.10,          # SCL post-filter
        atmosphere_rules=DEFAULT_ATMOSPHERE_RULES,
    )
    # `dates` is sorted ISO date strings; the caller fetches spectral via
    # fetch_seasonal_image() — only on the selected dates.

`mode` options
--------------

    "stac_only"            — STAC eo:cloud_cover ≤ scene_cloud_max
    "atmosphere"           — ERA5 prefilter only
    "scl_only"             — SCL-stack screen only
    "stac_then_scl"        — current Imint default (STAC → AOI-SCL post-filter)
    "era5_then_scl"        — RECOMMENDED. Cheapest filter first, AOI-aware end-to-end.
    "era5_then_stac"       — light variant, no openEO SCL call

The module returns *dates*, not arrays. The cost model that justifies the
recommendation:

    * ERA5 prefilter            ~free, disk-cached per (~0.25° cell, window)
    * STAC search               ~0.5 s
    * SCL-stack screen          ~60 s, one openEO call covering the whole period
    * Spectral fetch            ~13 s/scene amortised at 6 workers (DES openEO)

Spectral dominates total wall-clock at scale, so eliminating candidates
*before* spectral fetch is the load-bearing optimization.
"""
from __future__ import annotations

import json
import os
import random
import tempfile
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ATMOSPHERE_RULES = {
    "precip_today_max_mm":  0.5,
    "precip_prev2d_max_mm": 3.0,
    "t2m_mean_min_c":       10.0,
}

DEFAULT_SCL_CLOUD_THRESHOLD = 0.10   # production AOI-SCL default
DEFAULT_STAC_CLOUD_MAX      = 30.0


# ── Rate-limit-resilient HTTP ──────────────────────────────────────────────

def _is_rate_limited(exc: Exception) -> bool:
    """True if *exc* is an HTTP 429 / WAF rate-limit rejection.

    Covers both ``requests.HTTPError`` (carries ``.response.status_code``)
    and ``pystac_client.APIError`` (carries the WAF body as its message),
    so one predicate guards the Open-Meteo and CDSE-STAC calls alike.
    """
    resp = getattr(exc, "response", None)
    if resp is not None and getattr(resp, "status_code", None) == 429:
        return True
    msg = str(exc)
    return "429" in msg or "Rate limit exceeded" in msg


def retry_on_rate_limit(fn, *, attempts: int = 5, base_delay: float = 2.0):
    """Run *fn*, retrying on HTTP 429 with exponential backoff + jitter.

    Open-Meteo and the CDSE STAC are both fronted by WAF burst limiters
    that reject the cold-start thundering herd when many worker threads
    fire at once. Without backoff a throttled call raises and its tile is
    silently dropped from scene selection; a few retries absorb the burst.
    Non-rate-limit errors propagate immediately.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if attempt == attempts - 1 or not _is_rate_limited(exc):
                raise
            time.sleep(base_delay * 2 ** attempt + random.uniform(0.0, 1.0))


# ── Stage 1: ERA5 atmosphere prefilter ─────────────────────────────────────

_ERA5_GRID_DEG = 0.25   # Open-Meteo ERA5 archive native grid spacing
_ERA5_CACHE_DIR = Path(
    os.environ.get("IMINT_ERA5_CACHE")
    or Path.home() / ".cache" / "imint" / "era5"
)


def _read_json_cache(path: Path) -> Any | None:
    """Cached JSON payload, or None if absent / unreadable."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _write_json_cache(path: Path, payload: Any) -> None:
    """Atomically write *payload* as JSON — safe under concurrent workers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _request_era5_daily(
    lat: float, lon: float, date_start: str, date_end: str,
) -> list[dict]:
    """One Open-Meteo archive request → list of daily weather dicts."""
    import requests

    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude":   f"{lat:.4f}",
            "longitude":  f"{lon:.4f}",
            "start_date": date_start,
            "end_date":   date_end,
            "daily":      "temperature_2m_mean,precipitation_sum",
            "timezone":   "Europe/Stockholm",
        },
        timeout=60,
    )
    r.raise_for_status()
    daily = r.json()["daily"]
    out: list[dict] = []
    for d, t, p in zip(
        daily["time"], daily["temperature_2m_mean"], daily["precipitation_sum"]
    ):
        if t is None or p is None:
            continue
        out.append({"date": d, "t2m_mean": float(t), "precip_mm": float(p)})
    return out


def _era5_daily_open_meteo(
    bbox_wgs84: dict,
    date_start: str,
    date_end: str,
) -> list[dict]:
    """Daily ERA5 reanalysis via Open-Meteo Historical Archive (no auth).

    The bbox centroid is snapped to the ~0.25° ERA5 grid, so tiles sharing
    a grid cell resolve to one disk-cached response — the network is hit at
    most once per (cell, date-window). On HTTP 429 the call retries with
    exponential backoff. Cache dir: ``IMINT_ERA5_CACHE`` or
    ``~/.cache/imint/era5``.
    """
    cx = (bbox_wgs84["west"] + bbox_wgs84["east"]) / 2
    cy = (bbox_wgs84["south"] + bbox_wgs84["north"]) / 2
    lat = round(cy / _ERA5_GRID_DEG) * _ERA5_GRID_DEG
    lon = round(cx / _ERA5_GRID_DEG) * _ERA5_GRID_DEG

    cache_path = _ERA5_CACHE_DIR / (
        f"era5_{lat:+07.2f}_{lon:+07.2f}_{date_start}_{date_end}.json"
    )
    cached = _read_json_cache(cache_path)
    if cached is not None:
        return cached

    daily = retry_on_rate_limit(
        lambda: _request_era5_daily(lat, lon, date_start, date_end)
    )
    _write_json_cache(cache_path, daily)
    return daily


def era5_prefilter_dates(
    bbox_wgs84: dict,
    date_start: str,
    date_end: str,
    *,
    rules: dict | None = None,
) -> list[str]:
    """Stage 1: Return ISO dates whose weather passes the atmosphere rules.

    Free, AOI-aware (ERA5 grid ~30 km), and the cheapest filter to apply
    before any satellite API call.
    """
    rules = rules or DEFAULT_ATMOSPHERE_RULES
    daily = _era5_daily_open_meteo(bbox_wgs84, date_start, date_end)
    by_date = {w["date"]: w for w in daily}

    keep: list[str] = []
    for w in daily:
        d = date.fromisoformat(w["date"])
        prev_sum = 0.0
        for off in (1, 2):
            p = by_date.get((d - timedelta(days=off)).isoformat())
            if p is not None:
                prev_sum += p["precip_mm"]
        if (
            w["precip_mm"] <= rules["precip_today_max_mm"]
            and prev_sum     <= rules["precip_prev2d_max_mm"]
            and w["t2m_mean"] >= rules["t2m_mean_min_c"]
        ):
            keep.append(w["date"])
    return keep


# ── Stage 2: SCL-stack screen via DES openEO ───────────────────────────────

# Sen2Cor SCL classes counted as cloud / cloud-shadow
_SCL_CLOUD_CLASSES = (3, 8, 9, 10)


def _connect_des_openeo():
    """Authenticated DES openEO connection. Mirrors imint.fetch._des_connect()."""
    import os
    import openeo

    conn = openeo.connect("https://openeo.digitalearth.se/")
    user = os.environ.get("DES_USER")
    pw = os.environ.get("DES_PASSWORD")
    if not user or not pw:
        from imint.config.env import load_env
        load_env()
        user = os.environ.get("DES_USER")
        pw = os.environ.get("DES_PASSWORD")
    if not user or not pw:
        raise RuntimeError("DES_USER / DES_PASSWORD must be set for SCL screen.")
    conn.authenticate_basic(username=user, password=pw)
    return conn


def _connect_cdse_openeo():
    """Authenticated CDSE openEO connection.

    Uses the same CDSE_CLIENT_ID / CDSE_CLIENT_SECRET pair already in the
    cdse-credentials k8s secret. CDSE openEO draws from the monthly
    credit pool — separate from the DES per-session concurrency cap and
    from the SH Process API PU quota — so SCL screening here doesn't
    contend with spectral fetches on either of the other two backends.
    """
    from imint.fetch import _connect_cdse
    return _connect_cdse()


_DES_MAX_TIMESTEPS_PER_CALL = 19  # DES openEO save_result caps at 20 → use 19

_SCL_BACKEND_DEFAULTS = {
    "des":  {"collection": "s2_msi_l2a",   "band": "scl"},
    "cdse": {"collection": "SENTINEL2_L2A", "band": "SCL"},
}


def _scl_chunk(
    conn: Any,
    bbox_wgs84: dict,
    chunk_start: str,
    chunk_end: str,
    *,
    backend: str = "des",
) -> dict[str, float]:
    """Fetch SCL stack for a small period (≤ 19 timesteps) and aggregate locally.

    Two payload formats are handled, picked by backend:

    - **DES** (``backend="des"``): GeoTIFF output. The openEO server
      returns a multi-file zip where each file is a single timestep;
      we read each .tif via rasterio and extract the date from the
      filename / tags. (Historical path.)

    - **CDSE** (``backend="cdse"``): NetCDF output. CDSE's GeoTIFF
      output collapses the time dimension (single composite raster
      with no timestamps), so we use NetCDF instead — its native
      time-dim is preserved and dates come back as np.datetime64 in
      the ``t`` coordinate.

    Args:
        backend: "des" (GeoTIFF + zip-of-tifs) or "cdse" (NetCDF).
    """
    import os as _os
    import tempfile

    cfg = _SCL_BACKEND_DEFAULTS.get(backend)
    if cfg is None:
        raise ValueError(f"Unknown SCL backend: {backend!r}")

    scl_cube = conn.load_collection(
        cfg["collection"],
        spatial_extent={
            "west":  bbox_wgs84["west"],  "south": bbox_wgs84["south"],
            "east":  bbox_wgs84["east"],  "north": bbox_wgs84["north"],
            "crs":   "EPSG:4326",
        },
        temporal_extent=[chunk_start, chunk_end],
        bands=[cfg["band"]],
    )

    if backend == "cdse":
        return _read_scl_netcdf(scl_cube, cfg["band"])
    return _read_scl_geotiff(scl_cube)


def _read_scl_netcdf(scl_cube: Any, band_name: str) -> dict[str, float]:
    """Download as NetCDF and aggregate per timestep. Used for CDSE backend.

    NetCDF preserves the time dimension (CDSE GTiff doesn't), so we get
    one cloud-fraction per actual S2 overpass date in the requested window.
    """
    import tempfile
    import xarray as xr

    out: dict[str, float] = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = f"{tmpdir}/payload.nc"
        scl_cube.save_result(format="netCDF").download(tmp_path)
        ds = xr.open_dataset(tmp_path)
        if band_name not in ds.data_vars:
            # CDSE sometimes returns the variable under "SCL" regardless of
            # case in the request. Fall back to a case-insensitive lookup.
            candidates = [v for v in ds.data_vars if v.upper() == band_name.upper()]
            if not candidates:
                raise RuntimeError(
                    f"NetCDF payload missing band {band_name!r}; "
                    f"present vars={list(ds.data_vars)}"
                )
            band_name = candidates[0]
        scl_var = ds[band_name]
        if "t" not in scl_var.dims:
            # Single-timestep payload — extract the temporal_extent start
            # as fallback date.
            raise RuntimeError(
                f"NetCDF SCL variable missing 't' dimension: dims={scl_var.dims}"
            )
        t_values = ds.coords["t"].values
        for ti in range(int(ds.sizes["t"])):
            ts = t_values[ti]
            d_str = str(ts)[:10]  # 'YYYY-MM-DD' from numpy.datetime64
            scl_slice = scl_var.isel(t=ti).values
            cloud_frac = float(np.isin(scl_slice, _SCL_CLOUD_CLASSES).mean())
            prev = out.get(d_str)
            if prev is None or cloud_frac < prev:
                out[d_str] = cloud_frac
    return out


def _read_scl_geotiff(scl_cube: Any) -> dict[str, float]:
    """Download as GeoTIFF (zip-of-tifs / single tif / tar.gz) and aggregate.

    Used for DES backend. DES returns one file per timestep in a zip,
    where the date appears in the filename and / or band tags. This is
    the path that's been in production since the original ERA5→SCL
    benchmark.
    """
    import os as _os
    import re
    import tarfile
    import tempfile
    import gzip
    import zipfile
    import rasterio

    out: dict[str, float] = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = f"{tmpdir}/payload"
        scl_cube.save_result(format="GTiff").download(tmp_path)

        with open(tmp_path, "rb") as f:
            magic = f.read(4)

        tif_paths: list[str] = []
        if magic[:2] == b"PK":
            with zipfile.ZipFile(tmp_path) as zf:
                zf.extractall(tmpdir)
            tif_paths = [
                _os.path.join(tmpdir, n) for n in _os.listdir(tmpdir)
                if n.endswith(".tif") or n.endswith(".tiff")
            ]
        elif magic[:3] == b"\x1f\x8b\x08":
            try:
                with tarfile.open(tmp_path, mode="r:gz") as tf:
                    tf.extractall(tmpdir)
                tif_paths = []
                for root, _, files in _os.walk(tmpdir):
                    for fn in files:
                        if fn.endswith(".tif") or fn.endswith(".tiff"):
                            tif_paths.append(_os.path.join(root, fn))
            except tarfile.ReadError:
                ungz = f"{tmpdir}/scl.tif"
                with gzip.open(tmp_path, "rb") as gf, open(ungz, "wb") as wf:
                    wf.write(gf.read())
                tif_paths = [ungz]
        elif magic[:4] in (b"II*\x00", b"MM\x00*"):
            tif_paths = [tmp_path]
        else:
            raise RuntimeError(
                f"Unknown SCL payload format (magic={magic!r})"
            )

        for tif in tif_paths:
            with rasterio.open(tif) as src:
                bands = src.read()
                if bands.ndim == 2:
                    bands = bands[np.newaxis]
                tags = src.tags()
                file_date = None
                m = re.search(r"(\d{4}-\d{2}-\d{2})",
                              tags.get("timestamp", "") + " " + tif)
                if m:
                    file_date = m.group(1)
                for bi in range(bands.shape[0]):
                    btag = src.tags(bi + 1).get("description", "") + \
                           " " + src.tags(bi + 1).get("timestamp", "")
                    m2 = re.search(r"(\d{4}-\d{2}-\d{2})", btag)
                    d_str = m2.group(1) if m2 else file_date
                    if not d_str:
                        continue
                    cloud_frac = float(
                        np.isin(bands[bi], _SCL_CLOUD_CLASSES).mean()
                    )
                    prev = out.get(d_str)
                    if prev is None or cloud_frac < prev:
                        out[d_str] = cloud_frac
    return out


def scl_stack_screen(
    bbox_wgs84: dict,
    date_start: str,
    date_end: str,
    *,
    conn: Any | None = None,
    chunk_days: int = 19,
    backend: str = "des",
) -> dict[str, float]:
    """Stage 2: openEO-driven AOI cloud-fraction per scene date.

    Mirrors `batch_fetch_openeo.py:screen_tile_scl()` (which targets CDSE
    via server-side `aggregate_spatial`). DES openEO currently has two
    blockers on that path:
      - `aggregate_spatial` errors on geopandas dtypes (server bug).
      - `save_result` caps NetCDF timesteps at 20 per call.

    We work around both by chunking the period into ≤ 19-timestep windows,
    pulling each chunk's SCL raster, and computing the AOI cloud fraction
    locally. For a typical training-tile AOI the per-chunk download is
    < 1 MB; the whole season is a handful of openEO calls (3–6), still
    far cheaper than per-scene SH HTTP screening for hundreds of scenes
    spread across 1000+ tiles.

    Args:
        backend: "des" (default, the long-standing path) or "cdse" to
            route SCL chunks through the Copernicus Data Space Ecosystem
            openEO endpoint. Useful when DES is throttled and CDSE
            monthly credits are available — keeps the DES per-session
            cap free for spectral fetches downstream.
    """
    from datetime import date as _date, timedelta

    if conn is None:
        conn = _connect_cdse_openeo() if backend == "cdse" else _connect_des_openeo()

    d0 = _date.fromisoformat(date_start)
    d1 = _date.fromisoformat(date_end)
    total_days = (d1 - d0).days + 1
    # Re-balance chunks so the trailing chunk isn't a sliver (e.g. 78d / 19d
    # leaves a 2-day tail that always returns NoDataAvailable). Pick the
    # smallest chunk size in [10, chunk_days] that yields an even-ish split
    # — minimises noisy per-tile log lines without changing the ≤19 cap.
    if total_days > chunk_days:
        n_chunks = (total_days + chunk_days - 1) // chunk_days
        effective_chunk = (total_days + n_chunks - 1) // n_chunks
    else:
        effective_chunk = chunk_days
    out: dict[str, float] = {}
    cur = d0
    while cur <= d1:
        cend = min(cur + timedelta(days=effective_chunk - 1), d1)
        # NoDataAvailable on a chunk with no S2 overpass is benign — the
        # backend has nothing to return. Suppress the log line for that.
        try:
            chunk = _scl_chunk(
                conn, bbox_wgs84, cur.isoformat(), cend.isoformat(),
                backend=backend,
            )
            for k, v in chunk.items():
                prev = out.get(k)
                if prev is None or v < prev:
                    out[k] = v
        except Exception as e:
            msg = str(e)
            if "NoDataAvailable" not in msg:
                # Don't lose the whole stack to one bad chunk.
                print(f"  scl_stack chunk {cur}..{cend} failed: {e}")
        cur = cend + timedelta(days=1)
    return out


# ── STAC granule pre-filter (used by stac_* modes) ─────────────────────────

def stac_filter_dates(
    bbox_wgs84: dict,
    date_start: str,
    date_end: str,
    *,
    scene_cloud_max: float = DEFAULT_STAC_CLOUD_MAX,
) -> list[str]:
    """Anonymous earth-search STAC search; return dates with eo:cloud_cover ≤ N %."""
    import requests

    body = {
        "collections": ["sentinel-2-l2a"],
        "bbox": [bbox_wgs84["west"], bbox_wgs84["south"],
                 bbox_wgs84["east"], bbox_wgs84["north"]],
        "datetime": f"{date_start}T00:00:00Z/{date_end}T23:59:59Z",
        "limit": 500,
    }
    r = requests.post(
        "https://earth-search.aws.element84.com/v1/search",
        json=body, timeout=60,
    )
    r.raise_for_status()

    by_date: dict[str, float] = {}
    for feat in r.json().get("features", []):
        props = feat.get("properties", {})
        d = props.get("datetime", "")[:10]
        cc = props.get("eo:cloud_cover")
        if not d or cc is None:
            continue
        prev = by_date.get(d)
        if prev is None or float(cc) < prev:
            by_date[d] = float(cc)
    return sorted([d for d, cc in by_date.items() if cc <= scene_cloud_max])


# ── Orchestrator ───────────────────────────────────────────────────────────

@dataclass
class FetchPlan:
    """Result of optimal_fetch_dates — selected dates plus per-stage metrics."""
    mode: str
    dates: list[str]
    n_candidates_after: dict[str, int] = field(default_factory=dict)
    elapsed_s: dict[str, float] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)


def optimal_fetch_dates(
    bbox_wgs84: dict,
    date_start: str,
    date_end: str,
    *,
    mode: str = "era5_then_scl",
    max_aoi_cloud: float = DEFAULT_SCL_CLOUD_THRESHOLD,
    scene_cloud_max: float = DEFAULT_STAC_CLOUD_MAX,
    atmosphere_rules: dict | None = None,
    scl_backend: str = "des",
) -> FetchPlan:
    """Select Sentinel-2 dates worth fetching, given the requested strategy.

    See module docstring for `mode` options. The plan returned reports
    elapsed time per stage so cost-models stay honest.

    Args:
        scl_backend: "des" (default) or "cdse" — which openEO endpoint
            runs the SCL stack. Defaults to DES to preserve historical
            behaviour. Set to "cdse" when DES is throttled and CDSE
            monthly credits are available; this keeps DES capacity free
            for downstream spectral fetches.
    """
    import time as _t

    plan = FetchPlan(mode=mode, dates=[])

    # --- ERA5 (Stage 1) — only run if the mode needs it
    era5_dates: list[str] | None = None
    if mode in ("atmosphere", "era5_then_scl", "era5_then_stac"):
        t0 = _t.time()
        era5_dates = era5_prefilter_dates(
            bbox_wgs84, date_start, date_end, rules=atmosphere_rules,
        )
        plan.elapsed_s["era5"] = round(_t.time() - t0, 2)
        plan.n_candidates_after["era5"] = len(era5_dates)

    # --- STAC granule filter
    stac_dates: set[str] | None = None
    if mode in ("stac_only", "stac_then_scl", "era5_then_stac", "atmosphere"):
        # For "atmosphere" we still need a STAC call — but only as the S2
        # overpass calendar (no cc filter). ERA5 doesn't know which days
        # have a Sentinel-2 pass; intersecting with the calendar prevents
        # us from "spending" fetches on days with no scene.
        cc_threshold = 100.0 if mode == "atmosphere" else scene_cloud_max
        t0 = _t.time()
        stac_dates = set(stac_filter_dates(
            bbox_wgs84, date_start, date_end,
            scene_cloud_max=cc_threshold,
        ))
        plan.elapsed_s["stac"] = round(_t.time() - t0, 2)
        plan.n_candidates_after["stac"] = len(stac_dates)

    # --- SCL-stack screen
    scl_fracs: dict[str, float] | None = None
    if mode in ("scl_only", "era5_then_scl", "stac_then_scl"):
        t0 = _t.time()
        scl_fracs = scl_stack_screen(
            bbox_wgs84, date_start, date_end, backend=scl_backend,
        )
        plan.elapsed_s["scl_stack"] = round(_t.time() - t0, 2)
        plan.n_candidates_after["scl_pre_threshold"] = len(scl_fracs)
        plan.notes["scl_backend"] = scl_backend

    # --- Combine
    if mode == "stac_only":
        keep = sorted(stac_dates or set())
    elif mode == "atmosphere":
        # ERA5 ∩ STAC calendar — atmosphere can't know S2 pass days alone
        keep = sorted(set(era5_dates or []) & (stac_dates or set()))
    elif mode == "scl_only":
        keep = sorted(d for d, f in (scl_fracs or {}).items() if f <= max_aoi_cloud)
    elif mode == "stac_then_scl":
        keep = sorted(
            (stac_dates or set())
            & {d for d, f in (scl_fracs or {}).items() if f <= max_aoi_cloud}
        )
    elif mode == "era5_then_scl":
        keep = sorted(
            set(era5_dates or [])
            & {d for d, f in (scl_fracs or {}).items() if f <= max_aoi_cloud}
        )
    elif mode == "era5_then_stac":
        keep = sorted(set(era5_dates or []) & (stac_dates or set()))
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    plan.dates = keep
    plan.n_candidates_after["final"] = len(keep)
    plan.notes["max_aoi_cloud"] = str(max_aoi_cloud)
    plan.notes["scene_cloud_max"] = str(scene_cloud_max)
    return plan
