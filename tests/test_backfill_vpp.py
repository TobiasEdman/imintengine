"""Tests for scripts/backfill_vpp.py — VPP backfill into _recoreg tiles.

Builds a synthetic ``_recoreg`` directory (a tile with empty VPP, a tile
with present VPP, a no-year tile) and mocks the ``cdse_vpp.fetch_vpp_tiles``
fetch — keyed on the ``$VPP_SOURCE`` env switch the real router uses — so
every path is exercised without any network / cluster / CDSE PU spend:

  * enumeration finds ONLY the empty tile (present-VPP + no-year skipped);
  * the 5 ``vpp_*`` bands are written and read back, all other fields
    preserved byte-for-byte;
  * a second run is an idempotent no-op (tile now non-empty);
  * WEkEO-miss → CDSE fallback fills from the metered path;
  * a forced double-miss (WEkEO + CDSE) writes NOTHING and records the tile
    in the ``vpp_known_empty.json`` sidecar — no zero-fill.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import scripts.backfill_vpp as bf

H = W = 8
_RAW = ("sosd", "eosd", "length", "maxv", "minv")


# ── synthetic fetch results (keys are the bare metric names) ──────────────

def _covered(value: float = 18122.0) -> dict[str, np.ndarray]:
    """A "covers the tile" VPP result — SOSD/EOSD well above the >5% floor."""
    return {
        "sosd": np.full((H, W), value, np.float32),
        "eosd": np.full((H, W), value + 155, np.float32),
        "length": np.full((H, W), 155.0, np.float32),
        "maxv": np.full((H, W), 0.8, np.float32),
        "minv": np.full((H, W), 0.05, np.float32),
    }


def _empty_result() -> dict[str, np.ndarray]:
    """An all-zero VPP result — what WEkEO returns for a coverage gap."""
    return {k: np.zeros((H, W), np.float32) for k in _RAW}


# ── synthetic tiles ───────────────────────────────────────────────────────

def _base_tile() -> dict:
    """Minimal valid _recoreg tile: spectral cube + bbox + tessera_year."""
    return {
        "spectral": np.ones((6, H, W), np.float32),
        "bbox_3006": np.array([600000.0, 6500000.0, 600080.0, 6500080.0]),
        "tessera_year": np.int32(2021),
        "label": np.full((H, W), 7, np.uint8),   # a field that must survive
    }


def _write(path: Path, data: dict) -> None:
    np.savez_compressed(path, **data)


@pytest.fixture
def recoreg_dir(tmp_path: Path) -> Path:
    """A _recoreg dir: one empty-VPP tile, one has-VPP tile, one no-year tile."""
    d = tmp_path / "unified_v2_512_recoreg"
    d.mkdir()

    # (1) empty-VPP: VPP channels entirely absent → must be filled.
    _write(d / "tile_600000_6500000.npz", _base_tile())

    # (2) has-VPP: all 5 channels present + non-zero → must be skipped.
    has = _base_tile()
    for k in _RAW:
        has[f"vpp_{k}"] = np.full((H, W), 123.0, np.float32)
    _write(d / "tile_700000_6600000.npz", has)

    # (3) all-zero-VPP: channels present but identically zero → counts as empty.
    zero = _base_tile()
    for k in _RAW:
        zero[f"vpp_{k}"] = np.zeros((H, W), np.float32)
    _write(d / "tile_800000_6700000.npz", zero)

    return d


# ── emptiness / year / bbox unit checks ──────────────────────────────────

def test_vpp_is_empty_detects_absent_and_allzero():
    absent = {"spectral": np.ones((6, H, W), np.float32)}
    assert bf._vpp_is_empty(absent) is True

    allzero = {f"vpp_{k}": np.zeros((H, W), np.float32) for k in _RAW}
    assert bf._vpp_is_empty(allzero) is True

    present = {f"vpp_{k}": np.zeros((H, W), np.float32) for k in _RAW}
    present["vpp_sosd"] = np.full((H, W), 18122.0, np.float32)
    assert bf._vpp_is_empty(present) is False


def test_tile_year_precedence():
    assert bf._tile_year({"tessera_year": np.int32(2021),
                          "lpis_year": np.int32(2019)}) == 2021
    assert bf._tile_year({"lpis_year": np.int32(2019)}) == 2019
    assert bf._tile_year({"year": np.int32(2020)}) == 2020
    assert bf._tile_year({"dates": np.array(["2022-06-01", "2022-08-15"])}) == 2022
    assert bf._tile_year({"spectral": np.ones((6, H, W))}) is None


def test_tile_bbox_3006_roundtrip():
    d = {"bbox_3006": np.array([1.0, 2.0, 3.0, 4.0])}
    assert bf._tile_bbox_3006(d) == (1.0, 2.0, 3.0, 4.0)
    assert bf._tile_bbox_3006({"spectral": np.ones((6, H, W))}) is None


# ── enumeration: only the empty tiles get fetched ────────────────────────

def test_enumeration_targets_only_empty_tiles(recoreg_dir, monkeypatch):
    """The mocked fetch is invoked for tiles (1) + (3) only, not the has-VPP one."""
    fetched_bboxes: list[tuple] = []

    def _fake_fetch(west, south, east, north, *, size_px, year, cache_dir):
        fetched_bboxes.append((west, south, east, north))
        assert size_px == (H, W)        # size derived from the spectral cube
        assert year == 2021             # from tessera_year
        return _covered()

    monkeypatch.setattr(bf, "fetch_vpp_tiles", _fake_fetch)

    stats = bf.run(str(recoreg_dir), workers=1)

    # 2 empty tiles filled (absent + all-zero), 1 skipped (has VPP).
    assert stats["filled"] == 2
    assert stats["skipped"] == 1
    assert stats["empty"] == 0 and stats["failed"] == 0
    assert len(fetched_bboxes) == 2
    # The has-VPP tile's bbox (700000/6600000) was never fetched.
    assert (700000.0, 6600000.0, 700080.0, 6600080.0) not in fetched_bboxes


# ── write + read-back, with field preservation ───────────────────────────

def test_vpp_written_and_read_back_preserving_other_fields(recoreg_dir, monkeypatch):
    monkeypatch.setattr(bf, "fetch_vpp_tiles", lambda *a, **k: _covered(19000.0))

    tile = recoreg_dir / "tile_600000_6500000.npz"
    before = dict(np.load(tile, allow_pickle=True))

    bf.run(str(recoreg_dir), workers=1)

    after = dict(np.load(tile, allow_pickle=True))
    # All 5 VPP channels present, float32, correct shape + value.
    for k in _RAW:
        arr = after[f"vpp_{k}"]
        assert arr.shape == (H, W) and arr.dtype == np.float32
    assert float(after["vpp_sosd"][0, 0]) == 19000.0
    assert bf._vpp_is_empty(after) is False

    # Every pre-existing field is preserved byte-for-byte.
    np.testing.assert_array_equal(after["spectral"], before["spectral"])
    np.testing.assert_array_equal(after["label"], before["label"])
    np.testing.assert_array_equal(after["bbox_3006"], before["bbox_3006"])
    assert int(after["tessera_year"]) == 2021


# ── idempotency: a second run does not re-fetch ──────────────────────────

def test_idempotent_second_run_skips(recoreg_dir, monkeypatch):
    calls = {"n": 0}

    def _counting_fetch(*a, **k):
        calls["n"] += 1
        return _covered()

    monkeypatch.setattr(bf, "fetch_vpp_tiles", _counting_fetch)

    bf.run(str(recoreg_dir), workers=1)
    first = calls["n"]
    assert first == 2  # the 2 empty tiles

    stats2 = bf.run(str(recoreg_dir), workers=1)
    assert calls["n"] == first          # no further fetches
    assert stats2["filled"] == 0
    assert stats2["skipped"] == 3       # all 3 now carry VPP


# ── WEkEO miss → CDSE fallback fills from the metered path ────────────────

def test_wekeo_miss_falls_through_to_cdse(recoreg_dir, monkeypatch):
    """WEkEO returns all-zero (gap); CDSE covers it. Filled, sourced cdse."""
    seen_sources: list[str] = []

    def _by_source(*a, **k):
        import os
        src = os.environ.get("VPP_SOURCE")
        seen_sources.append(src)
        if src == "wekeo":
            return _empty_result()      # coverage gap → miss
        return _covered()               # cdse covers it

    monkeypatch.setattr(bf, "fetch_vpp_tiles", _by_source)

    # Restrict to the single absent-VPP tile for a clean source assertion.
    stats = bf.run(str(recoreg_dir), workers=1, max_tiles=1)

    assert stats["filled"] == 1
    assert "wekeo" in seen_sources and "cdse" in seen_sources


def test_wekeo_runtimeerror_falls_through_to_cdse(recoreg_dir, monkeypatch):
    """No WEkEO cache (RuntimeError) is also a miss → CDSE fallback."""
    def _by_source(*a, **k):
        import os
        if os.environ.get("VPP_SOURCE") == "wekeo":
            raise RuntimeError("VPP_SOURCE=wekeo but no WEkEO cache at /data/vpp_wekeo")
        return _covered()

    monkeypatch.setattr(bf, "fetch_vpp_tiles", _by_source)
    stats = bf.run(str(recoreg_dir), workers=1, max_tiles=1)
    assert stats["filled"] == 1


# ── double-miss: NO zero-fill, recorded in the known-empty sidecar ───────

def test_double_miss_records_known_empty_and_writes_nothing(recoreg_dir, monkeypatch):
    """WEkEO + CDSE both miss → VPP left absent, tile catalogued, never faked."""
    monkeypatch.setattr(bf, "fetch_vpp_tiles", lambda *a, **k: _empty_result())

    tile = recoreg_dir / "tile_600000_6500000.npz"
    before = dict(np.load(tile, allow_pickle=True))

    stats = bf.run(str(recoreg_dir), workers=1)

    assert stats["empty"] == 2          # both empty tiles missed everywhere
    assert stats["filled"] == 0

    # The tile is UNCHANGED — no zero-filled VPP channels were written.
    after = dict(np.load(tile, allow_pickle=True))
    assert bf._vpp_is_empty(after) is True
    for k in _RAW:
        assert f"vpp_{k}" not in after  # nothing fabricated
    np.testing.assert_array_equal(after["spectral"], before["spectral"])

    # The known-empty sidecar lists the missed tiles with a reason.
    sidecar = recoreg_dir / "vpp_known_empty.json"
    assert sidecar.exists()
    mapping = json.loads(sidecar.read_text())
    assert "tile_600000_6500000" in mapping
    assert "tile_800000_6700000" in mapping
    assert mapping["tile_600000_6500000"]  # non-empty reason string


# ── dry-run: fetch + report, but write nothing ───────────────────────────

def test_dry_run_writes_nothing(recoreg_dir, monkeypatch):
    monkeypatch.setattr(bf, "fetch_vpp_tiles", lambda *a, **k: _covered())

    tile = recoreg_dir / "tile_600000_6500000.npz"
    before = dict(np.load(tile, allow_pickle=True))

    stats = bf.run(str(recoreg_dir), workers=1, dry_run=True)

    assert stats["filled"] == 2         # reported as would-fill
    after = dict(np.load(tile, allow_pickle=True))
    assert bf._vpp_is_empty(after) is True            # untouched on disk
    np.testing.assert_array_equal(after["spectral"], before["spectral"])
    assert not (recoreg_dir / "vpp_known_empty.json").exists()
