"""Stage-0 granule-map builder — pre-2018 detection, dedup, ERA5 fallback.

Hermetic: the only repo dependency exercised for real is the
``DES_L2A_FLOOR`` era-split; STAC granule resolution and the ERA5 date
fallback are injected as deterministic fakes.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_pre2018_granule_map as bgm  # noqa: E402


# ── Pure date logic: which frames are pre-2018? ──────────────────────────────

def test_2018_tile_yields_slot0_2017_plus_2016bg():
    # 2018-labelled tile: slot 0 autumn = 2017 (pre-2018), VPP slots ≥2018.
    frames = bgm.pre2018_frames_for_tile(
        ["2017-10-05", "2018-05-01", "2018-06-15", "2018-08-01"], "2016-07-20"
    )
    assert ("slot0", "2017-10-05") in frames
    assert (bgm.BG_LABEL, "2016-07-20") in frames
    assert len(frames) == 2  # only slot0 + the 2016 background


def test_2022_tile_yields_only_2016bg():
    # 2022-labelled tile: slot 0 autumn = 2021 (≥ floor, des-able) -> not pre-2018.
    frames = bgm.pre2018_frames_for_tile(
        ["2021-10-05", "2022-05-01", "2022-06-15", "2022-08-01"], "2016-07-22"
    )
    assert frames == [(bgm.BG_LABEL, "2016-07-22")]


def test_missing_2016_date_flags_fallback():
    frames = bgm.pre2018_frames_for_tile(
        ["2021-10-05", "2022-05-01", "2022-06-15", "2022-08-01"], ""
    )
    assert (bgm.BG_LABEL, None) in frames  # None -> ERA5 fallback downstream


def test_floor_boundary_is_exclusive():
    # Exactly DES_L2A_FLOOR (2018-01-01) is NOT pre-2018; one day earlier is.
    assert ("slot0", "2017-12-31") in bgm.pre2018_frames_for_tile(["2017-12-31"], "")
    assert all(lbl != "slot0" for lbl, _ in bgm.pre2018_frames_for_tile(["2018-01-01"], ""))


def test_empty_slots_still_request_background():
    assert bgm.pre2018_frames_for_tile([], "2016-07-20") == [(bgm.BG_LABEL, "2016-07-20")]
    assert bgm.pre2018_frames_for_tile([], "") == [(bgm.BG_LABEL, None)]


# ── End-to-end build: dedup across tiles + ERA5 fallback path ────────────────

def _fake_stac(coords, d0, d1):
    """One deterministic granule per date; identical date -> identical scene_id."""
    return [{"scene_id": f"S2_{d0}", "datetime": f"{d0}T10:00:00Z",
             "mgrs_tile": "33VWF", "cloud_pct": 5.0}]


class _FakePlan:
    def __init__(self, dates):
        self.dates = dates


def test_build_map_dedups_shared_granules_and_uses_era5_fallback():
    era5_calls = []

    def fake_era5(coords, start, end, mode="era5_then_scl"):
        era5_calls.append((start, end, mode))
        return _FakePlan(["2016-07-15"])

    bbox = {"west": 17.0, "south": 59.0, "east": 17.05, "north": 59.05}
    tiles = [
        # A and B are 2018 tiles with IDENTICAL stored dates -> share both granules.
        {"name": "tile_A", "coords_wgs84": bbox,
         "slot_dates": ["2017-10-05", "2018-05-01", "2018-06-15", "2018-08-01"],
         "frame_2016_date": "2016-07-20"},
        {"name": "tile_B", "coords_wgs84": bbox,
         "slot_dates": ["2017-10-05", "2018-05-01", "2018-06-15", "2018-08-01"],
         "frame_2016_date": "2016-07-20"},
        # C is a 2022 tile with NO stored 2016 date -> ERA5 fallback selects one.
        {"name": "tile_C", "coords_wgs84": bbox,
         "slot_dates": ["2021-10-05", "2022-05-01", "2022-06-15", "2022-08-01"],
         "frame_2016_date": ""},
    ]

    tile_map, unique = bgm.build_granule_map(tiles, stac_fn=_fake_stac, era5_fn=fake_era5)

    by_sid = {g["scene_id"]: g for g in unique}
    # A+B share the 2017 autumn granule and the 2016 background granule.
    assert by_sid["S2_2017-10-05"]["n_tiles"] == 2
    assert by_sid["S2_2017-10-05"]["tiles"] == ["tile_A", "tile_B"]
    assert by_sid["S2_2016-07-20"]["n_tiles"] == 2
    # C went through the ERA5 fallback exactly once, over the 2016 summer window.
    assert era5_calls == [(*bgm.WINDOW_2016, "era5_then_scl")]
    assert by_sid["S2_2016-07-15"]["n_tiles"] == 1
    assert by_sid["S2_2016-07-15"]["tiles"] == ["tile_C"]

    # 5 frame-refs (A:2, B:2, C:1) collapse to 3 unique granules.
    n_refs = sum(len(v) for v in tile_map.values())
    assert n_refs == 5
    assert len(unique) == 3

    # C's emitted entry carries the fallback-selected date.
    c_bg = [e for e in tile_map["tile_C"] if e["label"] == bgm.BG_LABEL]
    assert c_bg and c_bg[0]["date"] == "2016-07-15"


def test_resolve_granules_keeps_only_same_day_scenes():
    def stac_multi(coords, d0, d1):
        return [
            {"scene_id": "S2_on", "datetime": f"{d0}T10:00:00Z", "mgrs_tile": "33VWF"},
            {"scene_id": "S2_off", "datetime": "2099-01-01T10:00:00Z", "mgrs_tile": "33VWF"},
        ]

    got = bgm.resolve_granules({"west": 0, "south": 0, "east": 1, "north": 1},
                               "2016-07-20", stac_fn=stac_multi)
    assert [s["scene_id"] for s in got] == ["S2_on"]
