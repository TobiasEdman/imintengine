"""Unit tests for ``select_slot_dates`` — the 5-slot fresh-fetch date selector.

No network: ``optimal_fetch_dates`` (the only external call) is patched at its
source module to return ``[earliest, midpoint, latest]`` per window, so the
midpoint-nearest pick is observable and the per-window ``mode`` is recorded. The
real ``doy_to_date_range`` runs (deterministic) for the VPP windows.
"""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from imint.training import tile_fetch as tf

_VPP = [(150, 170), (180, 200), (210, 230)]
_COORDS = {"west": 14.0, "south": 60.0, "east": 14.1, "north": 60.1}


def _mid(ds: str, de: str) -> str:
    return date.fromordinal(
        (date.fromisoformat(ds).toordinal() + date.fromisoformat(de).toordinal()) // 2
    ).isoformat()


def _patch_ofd(monkeypatch, *, empty_windows=frozenset()):
    """Patch optimal_fetch_dates → [earliest, midpoint, latest] per window
    (so midpoint-nearest selection is observable), recording (ds, de, mode)."""
    calls: list[tuple[str, str, str]] = []

    def _fake(coords, ds, de, *, mode, scl_backend="des"):
        calls.append((ds, de, mode))
        if (ds, de) in empty_windows:
            return SimpleNamespace(dates=[])
        return SimpleNamespace(dates=[ds, _mid(ds, de), de])

    monkeypatch.setattr("imint.training.optimal_fetch.optimal_fetch_dates", _fake)
    return calls


def test_five_slots_fresh(monkeypatch):
    calls = _patch_ofd(monkeypatch)
    dates = tf.select_slot_dates(_COORDS, tile_year=2022, vpp_windows=_VPP)

    assert set(dates) == {0, 1, 2, 3, 4}
    # Slot 0 — autumn from the PREVIOUS year (2021); midpoint-nearest.
    assert dates[0] == _mid("2021-08-15", "2021-10-31")
    # Slots 1-3 — VPP windows, CURRENT year (2022).
    for slot, (s, e) in zip((1, 2, 3), _VPP):
        ds, de = tf.doy_to_date_range(2022, s, e)
        assert dates[slot] == _mid(ds, de)
    # Slot 4 — 2016 summer background.
    assert dates[4] == _mid("2016-06-01", "2016-08-16")

    # Modes: slots 0-3 use era5_then_scl; the 2016 slot uses era5_then_stac
    # (pre-2018: DES SCL is blind, so STAC existence is the gate).
    modes = [m for _, _, m in calls]
    assert modes[:4] == ["era5_then_scl"] * 4
    assert ("2016-06-01", "2016-08-16", "era5_then_stac") in calls


def test_background_falls_back_to_2015(monkeypatch):
    calls = _patch_ofd(monkeypatch, empty_windows={("2016-06-01", "2016-08-16")})
    dates = tf.select_slot_dates(
        _COORDS, tile_year=2022, vpp_windows=_VPP,
        background_year=2016, background_fallback_year=2015)
    assert dates[4] == _mid("2015-06-01", "2015-08-16")
    assert ("2015-06-01", "2015-08-16", "era5_then_stac") in calls


def test_slot_with_no_clean_date_is_omitted(monkeypatch):
    # Autumn window has no clean date → slot 0 omitted; the rest still present.
    _patch_ofd(monkeypatch, empty_windows={("2021-08-15", "2021-10-31")})
    dates = tf.select_slot_dates(_COORDS, tile_year=2022, vpp_windows=_VPP)
    assert 0 not in dates
    assert set(dates) == {1, 2, 3, 4}


def test_no_vpp_windows_still_does_autumn_and_background(monkeypatch):
    _patch_ofd(monkeypatch)
    dates = tf.select_slot_dates(_COORDS, tile_year=2022, vpp_windows=None)
    assert set(dates) == {0, 4}            # no growing slots, but autumn + 2016 bg


def test_vpp_capped_at_three_slots(monkeypatch):
    # More than 3 VPP windows must not spill past slot 3 (slot 4 is the background).
    _patch_ofd(monkeypatch)
    dates = tf.select_slot_dates(
        _COORDS, tile_year=2022,
        vpp_windows=[(150, 160), (170, 180), (190, 200), (210, 220)])
    assert set(dates) == {0, 1, 2, 3, 4}   # 4th VPP window dropped, not slot 5
