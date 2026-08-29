"""Tests for the S1 season-composite enrichment (v2).

Covers the pure logic that does NOT need a live CDSE STAC connection:

    * per-pixel nodata-aware median math (`_nodata_median`)
    * speckle suppression: median of ≤3 scenes has lower local variance
      than any single contributing scene
    * orbit consistency — `filter_iw_grdh` never returns the wrong orbit,
      and the composite loop refuses a mixed-orbit scene
    * ≤max_scenes cap on the number of contributing scenes
    * v2 key layout written by the composite (shape/dtype/orbit contract)
    * dataset fail-loud on a pre-v3 tile fed to a SAR model

The download + calibration path is mocked (`_read_calibrated_scene`) so the
tests run offline; the σ⁰ math itself is exercised by s1_shared and the
cluster smoke.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from imint.training import cdse_s1_stac as stac
from imint.training import s1_shared
from imint.training.s1_enrichment import S1_ENRICH_VERSION


# ── Fake STAC item ────────────────────────────────────────────────────────

class _FakeItem:
    def __init__(self, item_id, orbit, dt, bbox, mode="IW", ptype="IW_GRDH"):
        self.id = item_id
        self.datetime = dt
        self.bbox = bbox
        self.properties = {
            "sar:instrument_mode": mode,
            "sar:product_type": ptype,
            "sat:orbit_state": orbit,
            "datetime": dt.isoformat(),
        }
        self.assets = {}


def _mk_items(orbits_dates, bbox=(13.0, 55.0, 13.1, 55.1)):
    return [
        _FakeItem(f"S1_{i}", orbit, dt, bbox)
        for i, (orbit, dt) in enumerate(orbits_dates)
    ]


def _dt(year, month, day):
    return datetime(year, month, day, tzinfo=timezone.utc)


# ── _nodata_median ─────────────────────────────────────────────────────────

def test_nodata_median_plain():
    stack = np.array([
        [[1.0, 2.0]],
        [[3.0, 4.0]],
        [[5.0, 6.0]],
    ], dtype=np.float32)  # (3, 1, 2)
    med = stac._nodata_median(stack)
    assert med.shape == (1, 2)
    np.testing.assert_allclose(med, [[3.0, 4.0]])  # medians of columns


def test_nodata_median_ignores_zero_nodata():
    # Pixel [0,0]: values 0 (nodata), 4, 8 → median of {4,8} = 6, not 4.
    stack = np.array([
        [[0.0]],
        [[4.0]],
        [[8.0]],
    ], dtype=np.float32)
    med = stac._nodata_median(stack)
    assert med[0, 0] == pytest.approx(6.0)


def test_nodata_median_all_zero_stays_zero():
    stack = np.zeros((3, 1, 1), dtype=np.float32)
    med = stac._nodata_median(stack)
    assert med[0, 0] == 0.0


def test_median_suppresses_speckle():
    # SAR speckle ~ multiplicative; median of independent looks reduces local
    # variance vs a single scene. Build 3 noisy looks of the same flat field.
    rng = np.random.default_rng(0)
    truth = np.full((64, 64), 5.0, dtype=np.float32)
    scenes = [
        (truth * rng.gamma(shape=2.0, scale=0.5, size=truth.shape)).astype(np.float32)
        for _ in range(3)
    ]
    composite = stac._nodata_median(np.stack(scenes, axis=0))
    single_var = float(np.var(scenes[0]))
    composite_var = float(np.var(composite))
    assert composite_var < single_var, (
        f"median var {composite_var:.3f} should be < single-scene var "
        f"{single_var:.3f}"
    )


# ── Orbit consistency ──────────────────────────────────────────────────────

def test_filter_iw_grdh_single_orbit():
    items = _mk_items([
        ("ASCENDING", _dt(2023, 6, 1)),
        ("DESCENDING", _dt(2023, 6, 3)),
        ("ASCENDING", _dt(2023, 6, 13)),
    ])
    asc = s1_shared.filter_iw_grdh(items, "ASCENDING")
    assert len(asc) == 2
    assert all(s1_shared.orbit_from_item(it) == "ASCENDING" for it in asc)
    desc = s1_shared.filter_iw_grdh(items, "DESCENDING")
    assert len(desc) == 1
    assert s1_shared.orbit_from_item(desc[0]) == "DESCENDING"


def test_composite_refuses_mixed_orbit(monkeypatch):
    # Even if a DESCENDING scene slips past the filter, the composite loop's
    # per-item orbit re-check must skip it — never mix into the median.
    items = _mk_items([
        ("ASCENDING", _dt(2023, 6, 1)),
        ("DESCENDING", _dt(2023, 6, 7)),
        ("ASCENDING", _dt(2023, 6, 13)),
    ])
    monkeypatch.setattr(stac, "_stac_search_with_backoff",
                        lambda *a, **k: items)
    # Bypass the real IW/GRDH filter to force the mixed set through, so we test
    # the loop's own orbit guard.
    monkeypatch.setattr(s1_shared, "filter_iw_grdh",
                        lambda its, orbit: its)

    seen_orbits = []

    def _fake_read(item, *a, **k):
        seen_orbits.append(s1_shared.orbit_from_item(item))
        return np.ones((2, 256, 256), dtype=np.float32)

    monkeypatch.setattr(stac, "_read_calibrated_scene", _fake_read)

    res = stac.fetch_s1_season_composite(
        380000, 6170000, 382560, 6172560,
        doy_window=(152, 244), year=2023,
        orbit_direction="ASCENDING", size_px=256, max_scenes=3,
    )
    assert res is not None
    _, _, orbit = res
    assert orbit == "ASCENDING"
    # The DESCENDING scene must never have been read into the median.
    assert "DESCENDING" not in seen_orbits


# ── ≤max_scenes cap + spread ───────────────────────────────────────────────

def test_max_scenes_cap(monkeypatch):
    # 8 valid same-orbit scenes across the season → composite uses ≤3.
    base = _dt(2023, 5, 1)
    items = _mk_items([
        ("ASCENDING", base + timedelta(days=12 * i)) for i in range(8)
    ])
    monkeypatch.setattr(stac, "_stac_search_with_backoff",
                        lambda *a, **k: items)
    n_read = {"count": 0}

    def _fake_read(item, *a, **k):
        n_read["count"] += 1
        return np.ones((2, 256, 256), dtype=np.float32)

    monkeypatch.setattr(stac, "_read_calibrated_scene", _fake_read)

    res = stac.fetch_s1_season_composite(
        380000, 6170000, 382560, 6172560,
        doy_window=(121, 273), year=2023,
        orbit_direction="ASCENDING", size_px=256, max_scenes=3,
    )
    assert res is not None
    sar, dates, orbit = res
    assert len(dates) <= 3
    assert n_read["count"] <= 3


def test_select_spread_scenes_temporal_spread():
    # 10 scenes; take a 3-spread → first, middle-ish, last are represented.
    base = _dt(2023, 5, 1)
    items = _mk_items([
        ("ASCENDING", base + timedelta(days=3 * i)) for i in range(10)
    ])
    spread = stac._select_spread_scenes(items, (13.0, 55.0, 13.1, 55.1), 3)
    dates = sorted(s1_shared.item_datetime(it) for it in spread)
    # Prefix (2*max_scenes=6) is chosen spread across the sorted list, so it
    # spans early → late, not a cluster.
    assert dates[0] == base
    assert dates[-1] >= base + timedelta(days=21)


# ── v2 key layout (composite return contract) ──────────────────────────────

def test_composite_shape_and_orbit(monkeypatch):
    items = _mk_items([("DESCENDING", _dt(2023, 6, 1)),
                       ("DESCENDING", _dt(2023, 6, 13))])
    monkeypatch.setattr(stac, "_stac_search_with_backoff",
                        lambda *a, **k: items)
    monkeypatch.setattr(
        stac, "_read_calibrated_scene",
        lambda item, *a, **k: (np.full((2, 256, 256), -12.0, dtype=np.float32)),
    )
    res = stac.fetch_s1_season_composite(
        380000, 6170000, 382560, 6172560,
        doy_window=(152, 244), year=2023,
        orbit_direction="DESCENDING", size_px=256, max_scenes=3,
    )
    assert res is not None
    sar, dates, orbit = res
    assert sar.shape == (2, 256, 256)
    assert sar.dtype == np.float32
    assert orbit == "DESCENDING"
    assert all(len(d) == 10 for d in dates)  # YYYY-MM-DD


def test_composite_none_when_no_scene(monkeypatch):
    monkeypatch.setattr(stac, "_stac_search_with_backoff", lambda *a, **k: [])
    res = stac.fetch_s1_season_composite(
        380000, 6170000, 382560, 6172560,
        doy_window=(152, 244), year=2023,
        orbit_direction="ASCENDING", size_px=256, max_scenes=3,
    )
    assert res is None


def test_nodata_scene_rejected(monkeypatch):
    # A scene that is >10% zero (swath edge) is dropped; a clean one survives.
    items = _mk_items([("ASCENDING", _dt(2023, 6, 1)),
                       ("ASCENDING", _dt(2023, 6, 13))])
    monkeypatch.setattr(stac, "_stac_search_with_backoff",
                        lambda *a, **k: items)
    bad = np.ones((2, 256, 256), dtype=np.float32)
    bad[:, :, :100] = 0.0  # ~39% nodata
    good = np.full((2, 256, 256), 3.0, dtype=np.float32)
    calls = iter([bad, good])
    monkeypatch.setattr(stac, "_read_calibrated_scene",
                        lambda *a, **k: next(calls))
    res = stac.fetch_s1_season_composite(
        380000, 6170000, 382560, 6172560,
        doy_window=(152, 244), year=2023,
        orbit_direction="ASCENDING", size_px=256, max_scenes=3,
        nodata_threshold=0.10,
    )
    assert res is not None
    sar, dates, _ = res
    assert len(dates) == 1  # only the good scene contributed
    assert np.allclose(sar, 3.0)


# ── probe_orbit_availability ────────────────────────────────────────────────

def test_probe_orbit_picks_dominant(monkeypatch):
    # 3 DESC vs 1 ASC across the windows → DESCENDING wins.
    def _fake_search(Client, bbox, dt_from, dt_to, label):
        return _mk_items([
            ("DESCENDING", _dt(2023, 6, 1)),
            ("DESCENDING", _dt(2023, 6, 13)),
            ("ASCENDING", _dt(2023, 6, 5)),
            ("DESCENDING", _dt(2023, 6, 25)),
        ])
    monkeypatch.setattr(stac, "_stac_search_with_backoff", _fake_search)
    orbit = stac.probe_orbit_availability(
        380000, 6170000, 382560, 6172560,
        windows=[((152, 244), 2023)],
    )
    assert orbit == "DESCENDING"


def test_probe_orbit_none_when_empty(monkeypatch):
    monkeypatch.setattr(stac, "_stac_search_with_backoff", lambda *a, **k: [])
    orbit = stac.probe_orbit_availability(
        380000, 6170000, 382560, 6172560,
        windows=[((152, 244), 2023)],
    )
    assert orbit is None


def test_probe_returns_items_per_window(monkeypatch):
    win0 = _mk_items([("DESCENDING", _dt(2023, 6, 1))])
    win1 = _mk_items([("DESCENDING", _dt(2016, 6, 1))])
    calls = iter([win0, win1])
    monkeypatch.setattr(stac, "_stac_search_with_backoff",
                        lambda *a, **k: next(calls))
    orbit, items_by_window = stac.probe_orbits_with_items(
        380000, 6170000, 382560, 6172560,
        windows=[((152, 244), 2023), ((152, 244), 2016)],
    )
    assert orbit == "DESCENDING"
    assert items_by_window[0] == win0
    assert items_by_window[1] == win1


def test_composite_reuses_items_no_research(monkeypatch):
    """Passing items= must NOT trigger a second STAC search."""
    items = _mk_items([("ASCENDING", _dt(2023, 6, 1)),
                       ("ASCENDING", _dt(2023, 6, 13))])
    search_calls = {"n": 0}

    def _fake_search(*a, **k):
        search_calls["n"] += 1
        return items

    monkeypatch.setattr(stac, "_stac_search_with_backoff", _fake_search)
    monkeypatch.setattr(
        stac, "_read_calibrated_scene",
        lambda item, *a, **k: np.full((2, 256, 256), 2.0, dtype=np.float32),
    )
    res = stac.fetch_s1_season_composite(
        380000, 6170000, 382560, 6172560,
        doy_window=(152, 244), year=2023,
        orbit_direction="ASCENDING", size_px=256, max_scenes=3,
        items=items,  # pre-fetched → no search
    )
    assert res is not None
    assert search_calls["n"] == 0


def test_stac_rate_limit_spaces_searches(monkeypatch):
    import time as _time

    monkeypatch.setattr(stac, "_STAC_MIN_INTERVAL_S", 0.05)
    stac._stac_last_search_t[0] = 0.0
    t0 = _time.monotonic()
    stac._stac_rate_limit()
    stac._stac_rate_limit()
    # Second call must wait ~one interval after the first.
    assert _time.monotonic() - t0 >= 0.05


# ── Growing-season window resolution (enrich script) ────────────────────────

def test_latitude_fallback_window_shifts_north():
    from scripts import enrich_tiles_s1 as enrich

    # Southern tile (low northing) → early May start; northern → June start.
    south = {"easting": 400000, "northing": 6180000}   # ~55.7°N
    north = {"easting": 650000, "northing": 7600000}   # ~68.5°N
    (s_start, s_end) = enrich._latitude_fallback_window(south)
    (n_start, n_end) = enrich._latitude_fallback_window(north)
    assert n_start > s_start   # green-up later in the north
    assert n_end < s_end       # senescence earlier in the north
    assert 121 <= s_start <= 152
    assert 244 <= s_end <= 273


def test_growing_season_prefers_vpp():
    from scripts import enrich_tiles_s1 as enrich
    from imint.training.vpp_windows import compute_growing_season_doy

    # YYDDD-encoded VPP: SOSD ~ DOY 110 (Apr 20), EOSD ~ DOY 260 (Sep 17).
    sosd = np.full((64, 64), 23110.0, dtype=np.float32)
    eosd = np.full((64, 64), 23260.0, dtype=np.float32)
    expected = compute_growing_season_doy(sosd, eosd)
    assert expected is not None  # sanity: this VPP decodes to a valid span
    (win, source) = enrich._growing_season_window(
        {"vpp_sosd": sosd, "vpp_eosd": eosd,
         "easting": 400000, "northing": 6180000}
    )
    assert source == "vpp"
    assert win == expected


def test_tile_year_priority():
    from scripts import enrich_tiles_s1 as enrich
    assert enrich._tile_year({"lpis_year": 2022, "year": 2020}) == 2022
    assert enrich._tile_year({"year": 2019}) == 2019
    assert enrich._tile_year({"dates": np.array(["2021-06-01", ""])}) == 2021
    assert enrich._tile_year({}) is None


# ── Dataset fail-loud on v1 ─────────────────────────────────────────────────

def test_dataset_faildloud_on_v1_sar_read():
    """A v1 tile (no s1_enrich_v) fed to a SAR model must raise, not silently
    feed the old (T*2,H,W) stack.

    Uses terramind_v1_base (SAR + raw 6-band S2, no b08/rededge dependency) so
    the only enrichment under test is the S1 read.
    """
    from imint.training.unified_dataset import UnifiedDataset

    # Minimal fake tile with a v1-style S1 stack (8 = 4 frames * 2), no
    # s1_enrich_v → v1. 4 optical frames so best-frame selection has choices.
    data = {
        "spectral": np.ones((24, 256, 256), dtype=np.float32),
        "temporal_mask": np.ones(4, dtype=np.uint8),
        "doy": np.array([150, 180, 210, 240], dtype=np.int32),
        "s1_vv_vh": np.zeros((8, 256, 256), dtype=np.float32),
    }

    ds = UnifiedDataset.__new__(UnifiedDataset)
    ds.model_keys = ("terramind_v1_base",)

    with pytest.raises(KeyError, match="s1_enrich_v"):
        ds._build_model_specific_tensors(data, source="lulc")


def test_dataset_reads_current_composite():
    """A current-version tile ((2,H,W) linear-γ⁰ composite) reads directly."""
    from imint.training.unified_dataset import UnifiedDataset

    data = {
        "spectral": np.ones((24, 32, 32), dtype=np.float32),
        "temporal_mask": np.ones(4, dtype=np.uint8),
        "doy": np.array([150, 180, 210, 240], dtype=np.int32),
        # Linear γ⁰ (RTC) — the normalizer log-transforms this internally.
        "s1_vv_vh": np.full((2, 32, 32), 0.05, dtype=np.float32),
        "s1_enrich_v": np.int32(S1_ENRICH_VERSION),
    }
    ds = UnifiedDataset.__new__(UnifiedDataset)
    ds.model_keys = ("terramind_v1_base",)
    out = ds._build_model_specific_tensors(data, source="lulc")
    assert out["s1_vv_vh"].shape == (2, 32, 32)
    assert np.allclose(out["s1_vv_vh"], 0.05)
