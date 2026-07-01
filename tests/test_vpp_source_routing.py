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


# ── VPP_SOURCE=wekeo forced path: coverage check (2026-07-01 silent zero-fill)
#
# A WEkEO read for a bbox outside cached MGRS returns an all-zero array, not
# None. Without _has_sufficient_coverage the wekeo-forced path silently
# accepts that and 130/390 orphan-512 tiles landed with all-zero VPP fields
# (2019: 3/44 MGRS + 2020: 0/44 MGRS covered in cache). Fix: raise loud on
# insufficient coverage so gap-fill or CDSE fallback is triggered explicitly.

def test_wekeo_forced_covered_returns_result(monkeypatch):
    monkeypatch.setenv("VPP_SOURCE", "wekeo")
    monkeypatch.setattr(cdse_vpp, "_read_wekeo_vpp", lambda *a, **k: _covered())
    out = cdse_vpp.fetch_vpp_tiles(0, 0, 1, 1, size_px=(H, W), year=2021)
    assert out["sosd"].shape == (H, W)
    assert int(np.median(out["sosd"])) == 18122


def test_wekeo_forced_none_raises(monkeypatch):
    """Cache truly absent (no COGs for MGRS × year) → raise, don't silent-degrade."""
    monkeypatch.setenv("VPP_SOURCE", "wekeo")
    monkeypatch.setattr(cdse_vpp, "_read_wekeo_vpp", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="no covering WEkEO cache"):
        cdse_vpp.fetch_vpp_tiles(0, 0, 1, 1, size_px=(H, W), year=2020)


def test_wekeo_forced_all_zero_raises(monkeypatch):
    """The 2026-07-01 shape: WEkEO reader returns all-zero (partial coverage,
    bbox in nodata region within cached COG). Previously silent-accepted →
    ~130 tiles landed with all-zero VPP. Must raise now."""
    monkeypatch.setenv("VPP_SOURCE", "wekeo")
    monkeypatch.setattr(cdse_vpp, "_read_wekeo_vpp", lambda *a, **k: _empty())
    with pytest.raises(RuntimeError, match="no covering WEkEO cache"):
        cdse_vpp.fetch_vpp_tiles(0, 0, 1, 1, size_px=(H, W), year=2019)
