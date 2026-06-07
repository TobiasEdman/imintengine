"""Tests for cdse_vpp cache-first + circuit-breaker source routing.

CDSE (metered SH Process) and the WEkEO COG cache serve the same HR-VPP
product, so routing is by cost/coverage/availability:
  - cache-first: WEkEO when it covers the tile (free, local)
  - CDSE only for coverage gaps, guarded by the shared SH-Process PU breaker
  - first PU exhaustion trips the breaker so later tiles skip doomed calls
"""
from __future__ import annotations

import numpy as np
import pytest

from imint.training import cdse_vpp
from imint.training import openeo_tile_graph as guard

H = W = 16


def _covered():
    # SOSD with >5% valid -> "covers the tile"
    return {"sosd": np.full((H, W), 18122, np.float32),
            "eosd": np.full((H, W), 18277, np.float32)}


def _empty():
    return {"sosd": np.zeros((H, W), np.float32),
            "eosd": np.zeros((H, W), np.float32)}


@pytest.fixture(autouse=True)
def _reset_guard(monkeypatch):
    # Isolate the process-global credit guard between tests.
    guard._DEAD_SOURCES.clear()
    monkeypatch.delenv("VPP_SOURCE", raising=False)
    yield
    guard._DEAD_SOURCES.clear()


def test_coverage_predicate():
    assert cdse_vpp._has_sufficient_coverage(_covered()) is True
    assert cdse_vpp._has_sufficient_coverage(_empty()) is False
    assert cdse_vpp._has_sufficient_coverage(None) is False


def test_cache_first_uses_wekeo_without_calling_cdse(monkeypatch):
    calls = {"cdse": 0}
    monkeypatch.setattr(cdse_vpp, "_read_wekeo_vpp", lambda *a, **k: _covered())
    def _no_cdse(*a, **k):
        calls["cdse"] += 1
        raise AssertionError("CDSE must not be called when cache covers")
    monkeypatch.setattr(cdse_vpp, "_fetch_cdse_vpp", _no_cdse)
    out = cdse_vpp._auto_fetch_vpp(0, 0, 1, 1, H, W, 2021)
    assert calls["cdse"] == 0
    assert int(np.median(out["sosd"])) == 18122


def test_gap_falls_through_to_cdse(monkeypatch):
    monkeypatch.setattr(cdse_vpp, "_read_wekeo_vpp", lambda *a, **k: _empty())
    sentinel = {"sosd": np.full((H, W), 21125, np.float32)}
    monkeypatch.setattr(cdse_vpp, "_fetch_cdse_vpp", lambda *a, **k: sentinel)
    out = cdse_vpp._auto_fetch_vpp(0, 0, 1, 1, H, W, 2021)
    assert out is sentinel  # used CDSE for the coverage gap


def test_pu_exhaustion_trips_breaker_and_falls_back(monkeypatch):
    monkeypatch.setattr(cdse_vpp, "_read_wekeo_vpp", lambda *a, **k: _empty())
    def _pu(*a, **k):
        raise RuntimeError(
            "Sentinel Hub Process API error (HTTP 403): "
            '{"error":{"code":"ACCESS_INSUFFICIENT_PROCESSING_UNITS"}}')
    monkeypatch.setattr(cdse_vpp, "_fetch_cdse_vpp", _pu)
    out = cdse_vpp._auto_fetch_vpp(0, 0, 1, 1, H, W, 2021)
    assert guard.is_source_dead("cdse") is True       # breaker tripped
    assert out["sosd"].shape == (H, W)                # WEkEO best-effort returned


def test_breaker_open_skips_cdse(monkeypatch):
    guard.mark_source_dead("cdse", "prior exhaustion")
    monkeypatch.setattr(cdse_vpp, "_read_wekeo_vpp", lambda *a, **k: _empty())
    def _no_cdse(*a, **k):
        raise AssertionError("CDSE must be skipped when breaker is open")
    monkeypatch.setattr(cdse_vpp, "_fetch_cdse_vpp", _no_cdse)
    out = cdse_vpp._auto_fetch_vpp(0, 0, 1, 1, H, W, 2021)
    assert out["sosd"].shape == (H, W)  # returned WEkEO best-effort, no CDSE


def test_no_cache_and_cdse_dead_raises(monkeypatch):
    guard.mark_source_dead("cdse", "exhausted")
    monkeypatch.setattr(cdse_vpp, "_read_wekeo_vpp", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="VPP unavailable"):
        cdse_vpp._auto_fetch_vpp(0, 0, 1, 1, H, W, 2021)


def test_non_pu_cdse_error_does_not_trip_breaker(monkeypatch):
    monkeypatch.setattr(cdse_vpp, "_read_wekeo_vpp", lambda *a, **k: _empty())
    monkeypatch.setattr(
        cdse_vpp, "_fetch_cdse_vpp",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("HTTP 500 transient")))
    out = cdse_vpp._auto_fetch_vpp(0, 0, 1, 1, H, W, 2021)
    assert guard.is_source_dead("cdse") is False  # transient != exhaustion
    assert out["sosd"].shape == (H, W)            # WEkEO best-effort
