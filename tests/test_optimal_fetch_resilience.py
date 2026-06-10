"""Rate-limit resilience + ERA5 disk cache for optimal_fetch.

Guards the cold-start thundering-herd failure mode: 16 concurrent selector
workers tripping the Open-Meteo and CDSE-STAC WAF burst limiters, which
silently dropped throttled tiles from scene selection.
"""
import json

import pytest

from imint.training import optimal_fetch as of


class _FakeResp:
    def __init__(self, status_code: int):
        self.status_code = status_code


class _HTTPErrorLike(Exception):
    """Stand-in for requests.HTTPError — carries a .response."""

    def __init__(self, status_code: int):
        super().__init__(f"{status_code} Client Error")
        self.response = _FakeResp(status_code)


# ── _is_rate_limited ──────────────────────────────────────────────────────

def test_is_rate_limited_httperror():
    assert of._is_rate_limited(_HTTPErrorLike(429))
    assert not of._is_rate_limited(_HTTPErrorLike(500))


def test_is_rate_limited_apierror_waf_body():
    # pystac_client.APIError carries the WAF body as its message
    exc = Exception('{"error":"WAF","message":"Rate limit exceeded","status":429}')
    assert of._is_rate_limited(exc)


def test_is_rate_limited_ignores_other_errors():
    assert not of._is_rate_limited(ValueError("bad bbox"))


# ── retry_on_rate_limit ───────────────────────────────────────────────────

def test_retry_succeeds_after_throttle():
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise _HTTPErrorLike(429)
        return "ok"

    assert of.retry_on_rate_limit(fn, base_delay=0.0) == "ok"
    assert len(calls) == 3


def test_retry_reraises_non_rate_limit_immediately():
    calls = []

    def fn():
        calls.append(1)
        raise ValueError("nope")

    with pytest.raises(ValueError):
        of.retry_on_rate_limit(fn, base_delay=0.0)
    assert len(calls) == 1


def test_retry_exhausts_attempts():
    calls = []

    def fn():
        calls.append(1)
        raise _HTTPErrorLike(429)

    with pytest.raises(_HTTPErrorLike):
        of.retry_on_rate_limit(fn, attempts=3, base_delay=0.0)
    assert len(calls) == 3


# ── ERA5 disk cache ───────────────────────────────────────────────────────

def test_era5_cache_dedups_within_grid_cell(tmp_path, monkeypatch):
    monkeypatch.setattr(of, "_ERA5_CACHE_DIR", tmp_path)
    seen = []
    monkeypatch.setattr(
        of, "_request_era5_daily",
        lambda lat, lon, ds, de: (
            seen.append((lat, lon))
            or [{"date": "2016-06-01", "t2m_mean": 15.0, "precip_mm": 0.0}]
        ),
    )

    # bbox_a and bbox_b centroids both snap to the same 0.25° cell.
    bbox_a = {"west": 14.90, "south": 58.20, "east": 14.95, "north": 58.25}
    bbox_b = {"west": 15.00, "south": 58.28, "east": 15.10, "north": 58.32}
    bbox_far = {"west": 19.9, "south": 59.9, "east": 20.1, "north": 60.1}

    of._era5_daily_open_meteo(bbox_a, "2016-06-01", "2016-08-31")
    of._era5_daily_open_meteo(bbox_a, "2016-06-01", "2016-08-31")  # cache hit
    of._era5_daily_open_meteo(bbox_b, "2016-06-01", "2016-08-31")  # same cell
    assert len(seen) == 1, "tiles in one ERA5 cell must share one network call"

    of._era5_daily_open_meteo(bbox_far, "2016-06-01", "2016-08-31")
    assert len(seen) == 2, "a distant cell must miss the cache"


def test_era5_cache_persists_to_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(of, "_ERA5_CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        of, "_request_era5_daily",
        lambda *a: [{"date": "2016-06-01", "t2m_mean": 15.0, "precip_mm": 0.0}],
    )

    out = of._era5_daily_open_meteo(
        {"west": 14.9, "south": 58.2, "east": 14.95, "north": 58.25},
        "2016-06-01", "2016-08-31",
    )
    files = list(tmp_path.glob("era5*.json"))  # era5v2_… (glob tolerates schema bumps)
    assert len(files) == 1, "one JSON cache file written"
    assert json.loads(files[0].read_text()) == out, "disk payload round-trips"
