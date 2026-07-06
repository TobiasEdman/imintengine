"""Date-rescue ladder in select_slot_dates (2026-07-06).

A growing-season VPP window with no date at the production AOI-cloud
ceiling (0.10) retries as scl_only at RELAXED_SCL_CLOUD (0.40) instead of
leaving the slot empty — an empty growing-season slot fails the fetch
write-gate, so without the rescue ~30% of tiles could never land. Slots
that succeed strictly must NOT trigger the extra (PU-costing) pass.
"""
from __future__ import annotations

import pytest

from imint.training import optimal_fetch as of
from imint.training import tile_fetch as tf


class _Plan:
    def __init__(self, dates):
        self.dates = dates


@pytest.fixture()
def selector_calls(monkeypatch):
    """Route optimal_fetch_dates to a scripted fake, recording every call."""
    calls = []
    script = {}   # (window_key, mode) -> dates

    def _fake(bbox, ds, de, *, mode, scl_backend, max_aoi_cloud=None, **kw):
        calls.append({"window": (ds, de), "mode": mode,
                      "max_aoi_cloud": max_aoi_cloud})
        return _Plan(script.get((ds, mode), []))

    monkeypatch.setattr(of, "optimal_fetch_dates", _fake)
    return calls, script


def test_rescue_only_for_strict_empty_slots(selector_calls):
    calls, script = selector_calls
    year = 2021
    # Windows: slot1 strict-OK; slot2 strict-empty, relaxed-OK; slot3 strict-OK.
    # vpp_windows are DOY pairs; doy_to_date_range(2021, 130, 160) etc.
    vpp = [(130, 160), (170, 200), (210, 240)]
    from imint.training.tile_fetch import doy_to_date_range
    w1, _ = doy_to_date_range(year, *vpp[0])
    w2, _ = doy_to_date_range(year, *vpp[1])
    w3, _ = doy_to_date_range(year, *vpp[2])
    a1, _ = doy_to_date_range(year - 1, 227, 304)  # autumn — unused key below

    script[(f"{year-1}-08-15", "era5_then_scl")] = ["2020-09-10"]
    script[(w1, "era5_then_scl")] = ["2021-05-20"]
    script[(w2, "era5_then_scl")] = []                 # strict: nothing
    script[(w2, "scl_only")] = ["2021-06-25"]          # relaxed: rescued
    script[(w3, "era5_then_scl")] = ["2021-08-05"]
    script[(f"2016-06-01", "era5_then_stac")] = ["2016-07-01"]

    dates = tf.select_slot_dates(
        {"west": 0, "south": 0, "east": 1, "north": 1},
        tile_year=year, vpp_windows=vpp)

    assert dates[2] == "2021-06-25", "slot 2 must be rescued by relaxed pass"
    assert dates[1] == "2021-05-20" and dates[3] == "2021-08-05"

    relaxed = [c for c in calls if c["mode"] == "scl_only"]
    assert len(relaxed) == 1, "only the strict-empty slot may pay for a rescue"
    assert relaxed[0]["max_aoi_cloud"] == tf.RELAXED_SCL_CLOUD == 0.40
    # Strict passes never override the production ceiling.
    assert all(c["max_aoi_cloud"] is None
               for c in calls if c["mode"] == "era5_then_scl")


def test_rescue_still_none_leaves_slot_empty(selector_calls):
    calls, script = selector_calls
    vpp = [(170, 200)]
    from imint.training.tile_fetch import doy_to_date_range
    w1, _ = doy_to_date_range(2021, *vpp[0])
    # Both passes empty → slot omitted (tile will gate-fail downstream —
    # a genuinely dateless window stays an honest failure, never a guess).
    script[(w1, "era5_then_scl")] = []
    script[(w1, "scl_only")] = []

    dates = tf.select_slot_dates(
        {"west": 0, "south": 0, "east": 1, "north": 1},
        tile_year=2021, vpp_windows=vpp)
    assert 1 not in dates
