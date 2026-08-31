"""Tests for the Planetary Computer RTC γ⁰ S1 backend (``pc_s1_rtc``, v3).

Covers the pure logic that does NOT need a live PC STAC connection:

    * per-pixel NaN-aware median math (``_nan_median``) — RTC nodata is NaN
      (mapped from the -32768 sentinel), NOT the GRD path's 0
    * RTC nodata masking (``_mask_nodata``): -32768 and non-finite → NaN
    * orbit consistency — the composite loop refuses a mixed-orbit scene and
      returns the requested orbit
    * ≤max_scenes cap on contributing scenes
    * v3 composite return contract (shape/dtype/orbit)
    * dominant-orbit probe returns the per-window items for reuse
    * linear-γ⁰ units: the default path stores linear, not dB

The window read + signing path is mocked (``_read_rtc_scene`` /
``_search_rtc``) so tests run offline; the reproject-window read itself is
exercised by the cluster smoke.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from imint.training import pc_s1_rtc as rtc
from imint.training import s1_shared


# ── Fake STAC item ────────────────────────────────────────────────────────

class _FakeAsset:
    def __init__(self, href):
        self.href = href


class _FakeItem:
    def __init__(self, item_id, orbit, dt, bbox, mode="IW", ptype="GRD"):
        self.id = item_id
        self.datetime = dt
        self.bbox = bbox
        self.properties = {
            "sar:instrument_mode": mode,
            "sar:product_type": ptype,
            "sat:orbit_state": orbit,
            "datetime": dt.isoformat(),
        }
        self.assets = {
            "vv": _FakeAsset(f"https://blob/{item_id}_vv.tiff?sig=x"),
            "vh": _FakeAsset(f"https://blob/{item_id}_vh.tiff?sig=x"),
        }


def _mk_items(orbits_dates, bbox=(13.0, 55.0, 13.1, 55.1)):
    return [
        _FakeItem(f"S1_{i}", orbit, dt, bbox)
        for i, (orbit, dt) in enumerate(orbits_dates)
    ]


def _dt(year, month, day):
    return datetime(year, month, day, tzinfo=timezone.utc)


# ── _mask_nodata ───────────────────────────────────────────────────────────

def test_mask_nodata_sentinel_and_nonfinite():
    arr = np.array([[0.05, -32768.0], [np.inf, 0.30]], dtype=np.float32)
    out = rtc._mask_nodata(arr)
    assert np.isnan(out[0, 1])  # -32768 sentinel → NaN
    assert np.isnan(out[1, 0])  # inf → NaN
    assert out[0, 0] == pytest.approx(0.05)  # valid γ⁰ preserved
    assert out[1, 1] == pytest.approx(0.30)


def test_mask_nodata_below_sentinel():
    # Anything <= -32768 is nodata (defensive on rounding).
    arr = np.array([[-40000.0]], dtype=np.float32)
    assert np.isnan(rtc._mask_nodata(arr)[0, 0])


# ── _nan_median ────────────────────────────────────────────────────────────

def test_nan_median_plain():
    stack = np.array([
        [[1.0, 2.0]],
        [[3.0, 4.0]],
        [[5.0, 6.0]],
    ], dtype=np.float32)  # (3, 1, 2)
    med = rtc._nan_median(stack)
    assert med.shape == (1, 2)
    np.testing.assert_allclose(med, [[3.0, 4.0]])


def test_nan_median_ignores_nan_nodata():
    # Pixel [0,0]: values NaN, 4, 8 → median of {4,8} = 6.
    stack = np.array([[[np.nan]], [[4.0]], [[8.0]]], dtype=np.float32)
    assert rtc._nan_median(stack)[0, 0] == pytest.approx(6.0)


def test_nan_median_all_nan_becomes_zero():
    stack = np.full((3, 1, 1), np.nan, dtype=np.float32)
    assert rtc._nan_median(stack)[0, 0] == 0.0  # genuine gap → 0


def test_median_suppresses_speckle():
    # Median of independent multiplicative-speckle looks reduces local
    # variance vs a single scene (the composite's core claim).
    rng = np.random.default_rng(0)
    truth = np.full((64, 64), 0.3, dtype=np.float32)
    scenes = [
        (truth * rng.gamma(shape=2.0, scale=0.5, size=truth.shape)).astype(np.float32)
        for _ in range(3)
    ]
    composite = rtc._nan_median(np.stack(scenes, axis=0))
    assert float(np.var(composite)) < float(np.var(scenes[0]))


# ── Orbit consistency ──────────────────────────────────────────────────────

def test_composite_refuses_mixed_orbit(monkeypatch):
    items = _mk_items([
        ("ASCENDING", _dt(2023, 6, 1)),
        ("DESCENDING", _dt(2023, 6, 7)),
        ("ASCENDING", _dt(2023, 6, 13)),
    ])
    monkeypatch.setattr(rtc, "_search_rtc", lambda *a, **k: items)
    # Force the mixed set past the IW filter so the loop's own guard is tested.
    monkeypatch.setattr(s1_shared, "filter_iw_grdh", lambda its, orbit: its)

    seen = []

    def _fake_read(item, *a, **k):
        seen.append(s1_shared.orbit_from_item(item))
        return np.full((2, 256, 256), 0.1, dtype=np.float32)

    monkeypatch.setattr(rtc, "_read_rtc_scene", _fake_read)
    monkeypatch.setattr(rtc, "_open_client", lambda: object())

    res = rtc.fetch_s1_season_composite(
        380000, 6170000, 382560, 6172560,
        doy_window=(152, 244), year=2023,
        orbit_direction="ASCENDING", size_px=256, max_scenes=3,
    )
    assert res is not None
    _, _, orbit = res
    assert orbit == "ASCENDING"
    assert "DESCENDING" not in seen


# ── ≤max_scenes cap ─────────────────────────────────────────────────────────

def test_max_scenes_cap(monkeypatch):
    base = _dt(2023, 5, 1)
    items = _mk_items([
        ("ASCENDING", base + timedelta(days=12 * i)) for i in range(8)
    ])
    monkeypatch.setattr(rtc, "_search_rtc", lambda *a, **k: items)
    monkeypatch.setattr(rtc, "_open_client", lambda: object())
    n_read = {"count": 0}

    def _fake_read(item, *a, **k):
        n_read["count"] += 1
        return np.full((2, 256, 256), 0.1, dtype=np.float32)

    monkeypatch.setattr(rtc, "_read_rtc_scene", _fake_read)
    res = rtc.fetch_s1_season_composite(
        380000, 6170000, 382560, 6172560,
        doy_window=(121, 273), year=2023,
        orbit_direction="ASCENDING", size_px=256, max_scenes=3,
    )
    assert res is not None
    sar, dates, orbit = res
    assert len(dates) <= 3
    assert n_read["count"] <= 3


# ── v3 return contract ──────────────────────────────────────────────────────

def test_composite_shape_orbit_and_linear(monkeypatch):
    items = _mk_items([("DESCENDING", _dt(2023, 6, 1)),
                       ("DESCENDING", _dt(2023, 6, 13))])
    monkeypatch.setattr(rtc, "_search_rtc", lambda *a, **k: items)
    monkeypatch.setattr(rtc, "_open_client", lambda: object())
    # Linear γ⁰ (~0.05); the composite must carry it through unchanged (no
    # dB conversion in the default path).
    monkeypatch.setattr(
        rtc, "_read_rtc_scene",
        lambda item, *a, **k: np.full((2, 256, 256), 0.05, dtype=np.float32),
    )
    res = rtc.fetch_s1_season_composite(
        380000, 6170000, 382560, 6172560,
        doy_window=(152, 244), year=2023,
        orbit_direction="DESCENDING", size_px=256, max_scenes=3,
    )
    assert res is not None
    sar, dates, orbit = res
    assert sar.shape == (2, 256, 256)
    assert sar.dtype == np.float32
    assert orbit == "DESCENDING"
    assert np.allclose(sar, 0.05)  # linear, not dB
    assert all(len(d) == 10 for d in dates)  # YYYY-MM-DD


def test_composite_none_when_no_scene(monkeypatch):
    monkeypatch.setattr(rtc, "_search_rtc", lambda *a, **k: [])
    monkeypatch.setattr(rtc, "_open_client", lambda: object())
    res = rtc.fetch_s1_season_composite(
        380000, 6170000, 382560, 6172560,
        doy_window=(152, 244), year=2023,
        orbit_direction="ASCENDING", size_px=256, max_scenes=3,
    )
    assert res is None


def test_nodata_scene_rejected(monkeypatch):
    # A scene >10% NaN (swath edge) is dropped; a clean one survives.
    items = _mk_items([("ASCENDING", _dt(2023, 6, 1)),
                       ("ASCENDING", _dt(2023, 6, 13))])
    monkeypatch.setattr(rtc, "_search_rtc", lambda *a, **k: items)
    monkeypatch.setattr(rtc, "_open_client", lambda: object())
    bad = np.full((2, 256, 256), 0.1, dtype=np.float32)
    bad[:, :, :100] = np.nan  # ~39% NaN
    good = np.full((2, 256, 256), 0.3, dtype=np.float32)
    calls = iter([bad, good])
    monkeypatch.setattr(rtc, "_read_rtc_scene", lambda *a, **k: next(calls))
    res = rtc.fetch_s1_season_composite(
        380000, 6170000, 382560, 6172560,
        doy_window=(152, 244), year=2023,
        orbit_direction="ASCENDING", size_px=256, max_scenes=3,
        nodata_threshold=0.10,
    )
    assert res is not None
    sar, dates, _ = res
    assert len(dates) == 1  # only the good scene
    assert np.allclose(sar, 0.3)


# ── probe_orbits_with_items ─────────────────────────────────────────────────

def test_probe_orbit_picks_dominant(monkeypatch):
    def _fake_search(client, bbox, dt_from, dt_to, label):
        return _mk_items([
            ("DESCENDING", _dt(2023, 6, 1)),
            ("DESCENDING", _dt(2023, 6, 13)),
            ("ASCENDING", _dt(2023, 6, 5)),
            ("DESCENDING", _dt(2023, 6, 25)),
        ])
    monkeypatch.setattr(rtc, "_search_rtc", _fake_search)
    orbit, _ = rtc.probe_orbits_with_items(
        380000, 6170000, 382560, 6172560,
        windows=[((152, 244), 2023)], client=object(),
    )
    assert orbit == "DESCENDING"


def test_probe_returns_items_per_window(monkeypatch):
    win0 = _mk_items([("DESCENDING", _dt(2023, 6, 1))])
    win1 = _mk_items([("DESCENDING", _dt(2016, 6, 1))])
    calls = iter([win0, win1])
    monkeypatch.setattr(rtc, "_search_rtc", lambda *a, **k: next(calls))
    orbit, items_by_window = rtc.probe_orbits_with_items(
        380000, 6170000, 382560, 6172560,
        windows=[((152, 244), 2023), ((152, 244), 2016)], client=object(),
    )
    assert orbit == "DESCENDING"
    assert items_by_window[0] == win0
    assert items_by_window[1] == win1


def test_probe_none_when_empty(monkeypatch):
    monkeypatch.setattr(rtc, "_search_rtc", lambda *a, **k: [])
    orbit, _ = rtc.probe_orbits_with_items(
        380000, 6170000, 382560, 6172560,
        windows=[((152, 244), 2023)], client=object(),
    )
    assert orbit is None


def test_composite_reuses_items_no_research(monkeypatch):
    """Passing items= must NOT trigger a STAC search."""
    items = _mk_items([("ASCENDING", _dt(2023, 6, 1)),
                       ("ASCENDING", _dt(2023, 6, 13))])
    search_calls = {"n": 0}

    def _fake_search(*a, **k):
        search_calls["n"] += 1
        return items

    monkeypatch.setattr(rtc, "_search_rtc", _fake_search)
    monkeypatch.setattr(
        rtc, "_read_rtc_scene",
        lambda item, *a, **k: np.full((2, 256, 256), 0.2, dtype=np.float32),
    )
    res = rtc.fetch_s1_season_composite(
        380000, 6170000, 382560, 6172560,
        doy_window=(152, 244), year=2023,
        orbit_direction="ASCENDING", size_px=256, max_scenes=3,
        items=items, client=object(),
    )
    assert res is not None
    assert search_calls["n"] == 0


def test_rtc_measurement_urls():
    it = _FakeItem("S1_x", "ASCENDING", _dt(2023, 6, 1), (13, 55, 13.1, 55.1))
    vv, vh = rtc._rtc_measurement_urls(it)
    assert "vv" in vv and "vh" in vh


def test_rtc_measurement_urls_missing_raises():
    it = _FakeItem("S1_x", "ASCENDING", _dt(2023, 6, 1), (13, 55, 13.1, 55.1))
    it.assets = {"vv": _FakeAsset("x")}  # no vh
    with pytest.raises(RuntimeError, match="missing vv/vh"):
        rtc._rtc_measurement_urls(it)
