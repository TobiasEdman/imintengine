"""SCL-screen throttle retry — the 429/408 backoff added 2026-07-05.

Both openEO endpoints throttle the date-selection screen under load (DES
[408], CDSE [429] with Retry-After). The client doesn't back off, so
throttled chunks were dropped and two legs sharing the CDSE SCL endpoint
429'd every chunk → empty screen → 0 tiles. These tests pin the retry:
throttle statuses retry (honouring Retry-After), everything else does not.
"""
from __future__ import annotations

import pytest

from imint.training import optimal_fetch as of


def test_retry_after_parsing():
    # The exact shape the openEO client surfaces: "'Retry-After': '3'".
    assert of._retry_after_seconds("[429] ... 'Retry-After': '3' ...", 99) == 3.0
    assert of._retry_after_seconds("Retry-After: 1.5", 99) == 1.5
    # No header → fall back to the caller's backoff.
    assert of._retry_after_seconds("[429] Too Many Requests", 7.0) == 7.0


def test_is_throttle_error():
    assert of._is_throttle_error("[429] Too Many Requests")
    assert of._is_throttle_error("[408] RequestTimeout: Request timed out.")
    assert not of._is_throttle_error("NoDataAvailable")
    assert not of._is_throttle_error("some parse error")


def _make_conn():
    class _Cube:
        def load_collection(self, *a, **k):
            return self
    return _Cube()


def test_scl_chunk_retries_429_then_succeeds(monkeypatch):
    sleeps = []
    monkeypatch.setattr(of.time, "sleep", lambda s: sleeps.append(s))
    calls = {"n": 0}

    def _flaky(cube, band):
        calls["n"] += 1
        if calls["n"] < 3:                       # 429 twice, then succeed
            raise RuntimeError("[429] 'Too Many Requests' 'Retry-After': '2'")
        return {"2021-06-01": (0.05, 0.0)}

    monkeypatch.setattr(of, "_read_scl_netcdf", _flaky)
    out = of._scl_chunk(_make_conn(), {"west": 0, "south": 0, "east": 1, "north": 1},
                        "2021-06-01", "2021-06-15", backend="cdse")
    assert out == {"2021-06-01": (0.05, 0.0)}
    assert calls["n"] == 3
    assert sleeps == [2.0, 2.0], "must honour Retry-After: 2 on each retry"


def test_scl_chunk_gives_up_after_max(monkeypatch):
    monkeypatch.setattr(of.time, "sleep", lambda s: None)

    def _always_429(cube, band):
        raise RuntimeError("[429] Too Many Requests")

    monkeypatch.setattr(of, "_read_scl_netcdf", _always_429)
    with pytest.raises(RuntimeError, match="429"):
        of._scl_chunk(_make_conn(), {"west": 0, "south": 0, "east": 1, "north": 1},
                      "2021-06-01", "2021-06-15", backend="cdse")


def test_scl_chunk_does_not_retry_nodata(monkeypatch):
    slept = []
    monkeypatch.setattr(of.time, "sleep", lambda s: slept.append(s))
    calls = {"n": 0}

    def _nodata(cube, band):
        calls["n"] += 1
        raise RuntimeError("NoDataAvailable: no scenes")

    monkeypatch.setattr(of, "_read_scl_netcdf", _nodata)
    with pytest.raises(RuntimeError, match="NoDataAvailable"):
        of._scl_chunk(_make_conn(), {"west": 0, "south": 0, "east": 1, "north": 1},
                      "2021-06-01", "2021-06-15", backend="cdse")
    assert calls["n"] == 1, "NoDataAvailable must not be retried"
    assert slept == []
