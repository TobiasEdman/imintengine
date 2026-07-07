"""Per-tile call-flow optimizations (2026-07-07).

Three fixes to the date-selection flow that dominated per-tile cost
(~12-24 openEO raster downloads of ~25-35 total remote calls):

1. scl_stack_screen memoizes complete screens in-process — the relaxed
   date-rescue re-screens the SAME window strictly screened moments
   earlier (the threshold is applied AFTER the screen), so rescue must
   cost zero new downloads.
2. Disk cache ($SCL_FRACS_CACHE) — cross-run retries (gate-failed tiles
   release claims and are re-attempted) must not re-pay the screens.
   ONLY complete screens are stored: a partial (throttled) screen cached
   would freeze storm-blindness in as truth.
3. Early-exit ordering — growing-season slots (1-3) screen FIRST; a tile
   doomed by a dateless growing window must not pay for the autumn
   (largest, ~5 chunks) and 2016-background screens, nor any spectral.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from imint.training import optimal_fetch as of
from imint.training import tile_fetch as tf


@pytest.fixture(autouse=True)
def _clear_memo():
    of._SCL_SCREEN_MEMO.clear()
    yield
    of._SCL_SCREEN_MEMO.clear()


_BBOX = {"west": 15.0, "south": 59.0, "east": 15.1, "north": 59.1}


@pytest.fixture()
def chunk_counter(monkeypatch):
    """Fake _scl_chunk: every date in the chunk at frac 0.2; counts calls."""
    calls: list[tuple[str, str]] = []

    def _fake(conn, bbox, cs, ce, *, backend="des"):
        calls.append((cs, ce))
        from datetime import date, timedelta
        d = date.fromisoformat(cs)
        out = {}
        while d <= date.fromisoformat(ce):
            out[d.isoformat()] = 0.2
            d += timedelta(days=5)
        return out

    monkeypatch.setattr(of, "_scl_chunk", _fake)
    monkeypatch.setattr(of, "_connect_des_openeo", lambda: object())
    monkeypatch.setattr(of, "_connect_cdse_openeo", lambda: object())
    return calls


def test_memo_second_screen_is_free(chunk_counter):
    a = of.scl_stack_screen(_BBOX, "2021-06-01", "2021-06-14")
    n = len(chunk_counter)
    assert n >= 1
    b = of.scl_stack_screen(_BBOX, "2021-06-01", "2021-06-14")
    assert len(chunk_counter) == n, "identical screen must be a memo hit"
    assert a == b


def test_disk_cache_survives_new_process(chunk_counter, tmp_path):
    of.scl_stack_screen(_BBOX, "2021-06-01", "2021-06-14",
                        cache_dir=str(tmp_path))
    n = len(chunk_counter)
    assert list(tmp_path.glob("*.json")), "complete screen must persist"
    of._SCL_SCREEN_MEMO.clear()          # simulate a fresh pod
    out = of.scl_stack_screen(_BBOX, "2021-06-01", "2021-06-14",
                              cache_dir=str(tmp_path))
    assert len(chunk_counter) == n, "disk hit must not re-download"
    assert out and all(v == 0.2 for v in out.values())


def test_partial_screen_never_cached(monkeypatch, tmp_path):
    calls = []

    def _flaky(conn, bbox, cs, ce, *, backend="des"):
        calls.append(cs)
        if len(calls) == 1:
            raise RuntimeError("boom")   # non-throttle, non-NoData
        return {cs: 0.2}

    monkeypatch.setattr(of, "_scl_chunk", _flaky)
    monkeypatch.setattr(of, "_connect_des_openeo", lambda: object())
    out = of.scl_stack_screen(_BBOX, "2021-06-01", "2021-07-28",
                              cache_dir=str(tmp_path))
    assert out, "partial screen is still returned"
    assert not list(tmp_path.glob("*.json")), "partial must NOT persist"
    assert not of._SCL_SCREEN_MEMO, "partial must NOT memoize"


def test_nodata_chunk_counts_as_complete(monkeypatch, tmp_path):
    def _nodata_then_ok(conn, bbox, cs, ce, *, backend="des"):
        if cs == "2021-06-01":
            raise RuntimeError("NoDataAvailable: no scenes")
        return {cs: 0.2}

    monkeypatch.setattr(of, "_scl_chunk", _nodata_then_ok)
    monkeypatch.setattr(of, "_connect_des_openeo", lambda: object())
    of.scl_stack_screen(_BBOX, "2021-06-01", "2021-07-28",
                        cache_dir=str(tmp_path))
    assert list(tmp_path.glob("*.json")), (
        "a no-overpass chunk is benign — screen is complete and cacheable")


def test_rescue_pays_zero_new_downloads(chunk_counter, monkeypatch):
    """Flagship: fracs are 0.2 everywhere → strict (≤0.10) finds nothing,
    rescue (≤0.40) succeeds — and every window is downloaded EXACTLY once."""
    monkeypatch.setattr(
        of, "era5_prefilter_dates", lambda bbox, ds, de, rules=None: [])
    monkeypatch.setattr(
        of, "stac_filter_dates",
        lambda bbox, ds, de, scene_cloud_max=30.0: ["2016-07-01"])

    vpp = [(140, 153), (170, 183), (200, 213)]      # three 14-day windows
    dates = tf.select_slot_dates(
        _BBOX, tile_year=2021, vpp_windows=vpp)

    assert all(s in dates for s in (1, 2, 3)), "rescue must fill all slots"
    from collections import Counter
    dupes = {w: c for w, c in Counter(chunk_counter).items() if c > 1}
    assert not dupes, f"windows downloaded more than once: {dupes}"


def test_early_exit_skips_autumn_and_background(monkeypatch):
    """Slot 1 dateless in BOTH passes → no slot-0/slot-4 screens, and
    fetch_tile pre-gates before any spectral call."""
    windows_called = []

    class _Plan:
        dates: list = []
        n_candidates_after: dict = {}    # mirrors FetchPlan (funnel log reads it)

    def _empty(bbox, ds, de, **kw):
        windows_called.append(ds)
        return _Plan()

    monkeypatch.setattr(of, "optimal_fetch_dates", _empty)
    vpp = [(140, 153), (170, 183), (200, 213)]
    dates = tf.select_slot_dates(_BBOX, tile_year=2021, vpp_windows=vpp)
    assert 1 not in dates
    assert not any(w.startswith("2020-08") for w in windows_called), (
        "autumn (slot 0) must not be screened for a doomed tile")
    assert not any(w.startswith("2016") for w in windows_called), (
        "2016 background must not be screened for a doomed tile")

    # fetch_tile pre-gate: partial dates → cheap fail, no spectral.
    import fetch_unified_tiles as fut
    monkeypatch.setattr(fut, "select_slot_dates",
                        lambda *a, **k: {2: "2021-06-20", 3: "2021-07-20"})
    monkeypatch.setattr(fut, "fetch_tile_spectral",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("spectral must not be called")))
    monkeypatch.setattr(fut, "_valid_existing_tile", lambda p: False)
    monkeypatch.setattr(fut, "bbox_3006_to_wgs84", lambda b: _BBOX)
    from imint.training.tile_config import TileConfig
    loc = {"name": "t", "year": 2021,
           "bbox_3006": {"west": 600000, "south": 6600000,
                         "east": 605120, "north": 6605120}}
    r = fut.fetch_tile(loc, ["2021"], "/tmp", TileConfig(size_px=512),
                       vpp_cache={"t": [(140, 153)]})
    assert r["status"] == "failed" and r["reason"] == "no_dates_slots_[1]"


# ── ERA5 temperature rule: snow/frost guard, not a growing-season gate ────

def test_t2m_rule_is_snow_guard_contract():
    """User-set 2026-07-07: 'ta med alla bilder över 0 grader, vi vill bara
    inte ha snö och frost'. The 10°C value zeroed autumn/spring windows in
    mid/north Sweden before SCL ever ran. SCL does not count snow (class 11)
    as cloud, so this ERA5 gate is the pipeline's only snow guard — it must
    exist, at 0.0, not at a vegetation-season threshold."""
    assert of.DEFAULT_ATMOSPHERE_RULES["t2m_mean_min_c"] == 0.0


def test_era5_keeps_cold_but_snowfree_dates(monkeypatch):
    daily = [
        {"date": "2021-10-01", "t2m_mean": 4.0, "precip_mm": 0.0,
         "overpass_cloud_pct": 20.0},   # cold autumn, dry, clear → KEEP now
        {"date": "2021-10-02", "t2m_mean": -2.0, "precip_mm": 0.0,
         "overpass_cloud_pct": 10.0},   # below freezing → frost/snow risk → drop
        {"date": "2021-10-03", "t2m_mean": 12.0, "precip_mm": 0.0,
         "overpass_cloud_pct": 20.0},   # warm & dry → keep (unchanged)
    ]
    monkeypatch.setattr(of, "_era5_daily_open_meteo", lambda *a, **k: daily)
    kept = of.era5_prefilter_dates(_BBOX, "2021-10-01", "2021-10-03")
    assert "2021-10-01" in kept, "4°C snow-free autumn date must now pass"
    assert "2021-10-02" not in kept, "sub-zero date must still be rejected"
    assert "2021-10-03" in kept
