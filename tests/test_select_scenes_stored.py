"""select_scenes — stored-date campaign mode (--reuse-stored-dates) + UTM zone.

Unit-level: `_utm_zone` math, stored-date extraction (np.str_ / np.bytes_ /
slot / missing), closest-in-window scene pick, and the dispatch. The
antimeridian zone filter itself lives inside `_stac_l1c_scenes` (real CDSE
STAC) and is verified on-cluster; its logic mirrors the unit-tested
build_pre2018 zone filter.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "sen2cor_pipeline"))
import select_scenes as ss  # noqa: E402


def test_utm_zone_sweden_and_antimeridian():
    assert ss._utm_zone(16.4) == 33       # central Sweden
    assert ss._utm_zone(22.0) == 34       # north-east Sweden
    assert ss._utm_zone(-180.0) == 1      # zone 01 — the antimeridian noise we drop
    assert ss._utm_zone(179.9) == 60      # zone 60 — likewise
    # Sweden (11-24°E) is always within ±1 of zone 33; antimeridian is far away.
    assert all(abs(ss._utm_zone(lon) - 33) <= 1 for lon in (11.5, 16.0, 23.5))
    assert abs(1 - 33) > 1 and abs(60 - 33) > 1


def test_stored_date_frame_2016_str_and_bytes(tmp_path):
    p1 = tmp_path / "a.npz"
    np.savez_compressed(p1, frame_2016_date=np.str_("2016-07-20"))
    assert ss._stored_target_date(p1, "frame_2016") == "2016-07-20"
    p2 = tmp_path / "b.npz"  # legacy fetch_unified_tiles wrote this as np.bytes_
    np.savez_compressed(p2, frame_2016_date=np.bytes_(b"2016-08-01"))
    assert ss._stored_target_date(p2, "frame_2016") == "2016-08-01"


def test_stored_date_slot_and_missing(tmp_path):
    p = tmp_path / "c.npz"
    np.savez_compressed(p, dates=np.array(["2017-10-05", "2018-05-01", "", ""]))
    assert ss._stored_target_date(p, "slot:0") == "2017-10-05"
    assert ss._stored_target_date(p, "slot:2") == ""        # empty slot
    p2 = tmp_path / "d.npz"
    np.savez_compressed(p2, spectral=np.zeros((1, 1, 1)))   # no frame_2016_date
    assert ss._stored_target_date(p2, "frame_2016") == ""
    assert ss._stored_target_date(tmp_path / "nope.npz", "frame_2016") == ""


def test_resolve_stored_picks_closest_date(monkeypatch):
    def fake_stac(bbox, d0, d1):
        return [
            {"scene_id": "far", "datetime": "2016-07-13T10:30:00Z", "mgrs_tile": "33VWF", "cloud_pct": 0.0},
            {"scene_id": "near", "datetime": "2016-07-21T10:30:00Z", "mgrs_tile": "33VWF", "cloud_pct": 80.0},
        ]
    monkeypatch.setattr(ss, "_stac_l1c_scenes", fake_stac)
    bbox = {"west": 16.3, "south": 58.3, "east": 16.4, "north": 58.4}
    _name, hit = ss._resolve_tile_stored("t", bbox, "2016-07-20")
    assert [s["scene_id"] for s in hit] == ["near"]  # 1 day off beats 7, worse cloud notwithstanding


def test_resolve_stored_empty_and_bad_date(monkeypatch):
    monkeypatch.setattr(ss, "_stac_l1c_scenes", lambda b, d0, d1: [])
    bbox = {"west": 16, "east": 16.1, "south": 58, "north": 58.1}
    assert ss._resolve_tile_stored("t", bbox, "2016-07-20")[1] is None
    assert ss._resolve_tile_stored("t", bbox, "")[1] is None  # unparsable date


def test_dispatch_stored_when_available_else_era5(monkeypatch, tmp_path):
    calls = {"stored": 0, "era5": 0}

    def fake_stored(name, bbox, sd, window_days=7):
        calls["stored"] += 1
        return name, [{"scene_id": "s"}]

    def fake_era5(name, bbox, ds, de):
        calls["era5"] += 1
        return name, None

    monkeypatch.setattr(ss, "_resolve_tile_stored", fake_stored)
    monkeypatch.setattr(ss, "_resolve_tile", fake_era5)
    bbox = {"west": 16, "east": 16.1, "south": 58, "north": 58.1}
    np.savez_compressed(tmp_path / "z.npz", frame_2016_date=np.str_("2016-07-20"))

    ss._resolve_tile_or_stored("z", bbox, "2016-06-01", "2016-08-31", str(tmp_path), "frame_2016", True)
    assert calls == {"stored": 1, "era5": 0}                       # has stored date -> stored
    ss._resolve_tile_or_stored("missing", bbox, "2016-06-01", "2016-08-31", str(tmp_path), "frame_2016", True)
    assert calls == {"stored": 1, "era5": 1}                       # no stored date -> ERA5 fallback
    ss._resolve_tile_or_stored("z", bbox, "2016-06-01", "2016-08-31", str(tmp_path), "frame_2016", False)
    assert calls == {"stored": 1, "era5": 2}                       # reuse off -> ERA5


def test_cross_dir_reads_date_from_source_dir(monkeypatch, tmp_path):
    """Cross-dir re-coreg: the stored date is read from ``date_source_dir``, not
    the enumerate dir. Mirrors the campaign — the ``_recoreg`` tile carries NO
    ``frame_2016_date`` (Phase-1 dropped the pre-2018 frame), and the date is
    reused from the original ``unified_v2_512``."""
    captured = {}

    def fake_stored(name, bbox, sd, window_days=7):
        captured["sd"] = sd
        return name, [{"scene_id": "s"}]
    monkeypatch.setattr(ss, "_resolve_tile_stored", fake_stored)
    monkeypatch.setattr(ss, "_resolve_tile",
                        lambda *a: pytest.fail("must resolve via stored date"))

    recoreg = tmp_path / "recoreg"; recoreg.mkdir()
    original = tmp_path / "original"; original.mkdir()
    # _recoreg tile lacks the date; the original carries it.
    np.savez_compressed(recoreg / "z.npz", spectral=np.zeros((1, 1, 1)))
    np.savez_compressed(original / "z.npz", frame_2016_date=np.str_("2016-07-20"))

    bbox = {"west": 16, "east": 16.1, "south": 58, "north": 58.1}
    ss._resolve_tile_or_stored(
        "z", bbox, "2016-06-01", "2016-08-31", str(original), "frame_2016", True)
    assert captured["sd"] == "2016-07-20"     # read from the original, cross-dir
