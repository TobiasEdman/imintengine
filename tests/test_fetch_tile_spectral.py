"""Phase-0 contract test for imint.training.fetch_spectral.fetch_tile_spectral.

Proves the canonical per-tile entry reproduces scripts/regrid_national_512.py's
M1→M2→crop→assemble composition exactly, with no network: the openEO tile-graph
fetch is monkeypatched to return synthetic halo frames. A known inter-frame shift
is injected so M2 (coregister_interframe) actually moves the movers — that makes
the equality test catch a "M2 skipped / wrong args" wiring bug, not pass vacuously.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from imint.coregistration import (
    clearest_frame_idx,
    coregister_interframe,
    subpixel_shift,
)
from imint.training import fetch_spectral as fs
from imint.training.openeo_tile_graph import ALL_BANDS
from imint.training.tile_assemble import assemble_fresh, crop_halo
from imint.training.tile_config import TileConfig

_CANON = 128                        # ≥ estimate_mi_offset's 96 px window, so M2 is real
_HALO_PX = 8
_HALO = _CANON + _HALO_PX           # 136
_CROP = _HALO_PX // 2               # 4
_N = 4
_CENTRE = (500000, 6500000)         # 10 m-aligned EPSG:3006 centre


def _structured_frame(rng, dy: float, dx: float) -> np.ndarray:
    """A (12, halo, halo) frame whose B04 (band 2) carries a band-limited field
    shifted by (dy, dx) — the geometry M2 estimates on."""
    base = gaussian_filter(rng.standard_normal((_HALO, _HALO)).astype(np.float32), 1.6)
    base = (base - base.min()) / (base.max() - base.min())          # → [0, 1]
    frame = np.repeat(base[None], len(ALL_BANDS), axis=0).astype(np.float32)
    frame[2] = subpixel_shift(base.astype(np.float64), dy, dx).astype(np.float32)
    return frame


def _frames() -> dict[int, np.ndarray]:
    rng = np.random.default_rng(0)
    # slot 2 missing on purpose → exercises the zero-fill + temporal_mask path.
    return {
        0: _structured_frame(rng, 0.0, 0.0),     # anchor candidate (no shift)
        1: _structured_frame(rng, 0.45, -0.30),  # mover
        3: _structured_frame(rng, -0.50, 0.25),  # mover
    }


def _patch_fetch(monkeypatch, frames: dict[int, np.ndarray]) -> None:
    def _fake(bbox_3006, slot_dates, source):       # signature of fetch_tile_at_specific_dates
        return {fi: (frames[fi], None) for fi in slot_dates if fi in frames}
    monkeypatch.setattr(fs, "fetch_tile_at_specific_dates", _fake)


def _reference_composition(frames: dict[int, np.ndarray], *, coregister: bool):
    """The regrid_one_tile recipe, run independently for comparison."""
    fresh = {fi: f.copy() for fi, f in frames.items()}
    if coregister and len(fresh) >= 2:
        ref_idx = clearest_frame_idx(fresh)
        fresh, _ = coregister_interframe(fresh, ref_idx, search_px=float(_CROP))
    else:
        ref_idx = next(iter(fresh))
    cropped = {fi: crop_halo(a, crop=_CROP, canon=_CANON) for fi, a in fresh.items()}
    dates_list = [_DATES.get(fi, "") for fi in range(_N)]
    spectral, extras = assemble_fresh(cropped, dates_list, _N, canon=_CANON)
    return spectral, extras, ref_idx


_DATES = {0: "2021-09-10", 1: "2022-06-01", 2: "2022-07-15", 3: "2022-08-20"}


@pytest.fixture(autouse=True)
def _no_l1c_fallthrough(monkeypatch):
    """Default: the entry's l1c_sen2cor halo fallthrough is a no-op — there is no
    sen2cor binary in the test env, and we must not touch the process-global
    _DEAD_SOURCES set. Tests that exercise the fallthrough override this."""
    monkeypatch.setattr(fs, "_l1c_sen2cor_allband_cube", lambda *a, **k: None)


def test_composition_matches_regrid_recipe(monkeypatch):
    frames = _frames()
    _patch_fetch(monkeypatch, frames)
    res = fs.fetch_tile_spectral(
        _CENTRE, tile=TileConfig(size_px=_CANON), dates=_DATES, n_frames=_N,
        backend="des", halo_px=_HALO_PX, coregister=True,
    )
    ref_spec, ref_extras, ref_idx = _reference_composition(frames, coregister=True)

    assert res is not None
    assert res["spectral"].shape == (_N * 6, _CANON, _CANON)
    assert np.array_equal(res["spectral"], ref_spec)
    for k in ("b08", "rededge", "b01", "b09"):
        assert np.array_equal(res[k], ref_extras[k]), k
    assert int(res["coreg_ref_frame"]) == ref_idx
    assert int(res["coreg_m2"]) == 1
    assert list(res["temporal_mask"]) == [1, 1, 0, 1]   # slot 2 missing
    assert list(res["dates"]) == ["2021-09-10", "2022-06-01", "", "2022-08-20"]
    # M2 actually moved the movers — the cube is NOT the raw cropped frames.
    raw_spec, _, _ = _reference_composition(frames, coregister=False)
    assert not np.array_equal(res["spectral"], raw_spec)


def test_l1c_sen2cor_fallthrough_frame_is_m2_coregistered(monkeypatch):
    """A pre-2018 slot filled by the l1c_sen2cor halo fallthrough is NOT merely
    concatenated — it rides through M2 inter-frame coreg with the rest of the
    stack. The fallthrough frame carries a KNOWN shift so a no-op/broken M2 on
    that slot is detectable (a zero-shift frame would pass vacuously)."""
    tg_frames = _frames()                 # tile-graph yields slots 0,1,3; slot 2 missing
    _patch_fetch(monkeypatch, tg_frames)
    fall_frame = _structured_frame(np.random.default_rng(7), 0.40, -0.25)
    calls: dict = {}

    def _fake_cube(bbox_3006, coords_wgs84, date_str, *, size_px):
        calls["args"] = (date_str, size_px)
        return fall_frame.copy()          # (len(ALL_BANDS), _HALO, _HALO)
    monkeypatch.setattr(fs, "_l1c_sen2cor_allband_cube", _fake_cube)

    res = fs.fetch_tile_spectral(
        _CENTRE, tile=TileConfig(size_px=_CANON), dates=_DATES, n_frames=_N,
        backend="des", halo_px=_HALO_PX, coregister=True,
    )
    assert res is not None
    assert list(res["temporal_mask"]) == [1, 1, 1, 1]       # slot 2 filled
    # fetched on the halo grid (size_px == halo) for the missing slot's date.
    assert calls["args"] == ("2022-07-15", _HALO)

    # Headline: the entry's output equals the full M1→M2→crop→assemble recipe
    # over ALL 4 frames (slot 2 from the fallthrough). If slot 2 were concatenated
    # WITHOUT coreg, its channels would be the raw shifted frame and this equality
    # would fail — so it proves slot 2 went THROUGH M2.
    all_frames = {**tg_frames, 2: fall_frame}
    ref_spec, _, _ = _reference_composition(all_frames, coregister=True)
    assert np.array_equal(res["spectral"], ref_spec)
    # And M2 was not a global no-op — the corrected cube differs from the raw one.
    raw_spec, _, _ = _reference_composition(all_frames, coregister=False)
    assert not np.array_equal(res["spectral"], raw_spec)


def test_pre2018_slot_skips_des_tilegraph(monkeypatch):
    """A pre-2018 slot must NOT enter the des tile-graph: des is L2A-indexed
    2018+, and a pre-2018 date in the merged download hangs des server-side
    (regression for the 4b-3 step-4 cluster failure — a 2016 background slot
    408'd the 60-band merged fetch). It's routed to the l1c_sen2cor fallthrough
    instead, so only the >=2018 slots reach the tile-graph."""
    rng = np.random.default_rng(3)
    tg_frames = {0: _structured_frame(rng, 0.0, 0.0),
                 1: _structured_frame(rng, 0.30, -0.20),
                 3: _structured_frame(rng, -0.40, 0.15)}
    dates = {0: "2021-09-10", 1: "2022-06-01", 2: "2016-07-15", 3: "2022-08-20"}

    seen: dict = {}

    def _fake_des(bbox_3006, slot_dates, source):
        seen["slots"] = dict(slot_dates)
        return {fi: (tg_frames[fi], None) for fi in slot_dates if fi in tg_frames}
    monkeypatch.setattr(fs, "fetch_tile_at_specific_dates", _fake_des)

    fall = _structured_frame(np.random.default_rng(9), 0.35, -0.20)
    l1c_dates: list = []

    def _fake_cube(bbox_3006, coords_wgs84, date_str, *, size_px):
        l1c_dates.append(date_str)
        return fall.copy()
    monkeypatch.setattr(fs, "_l1c_sen2cor_allband_cube", _fake_cube)

    res = fs.fetch_tile_spectral(
        _CENTRE, tile=TileConfig(size_px=_CANON), dates=dates, n_frames=_N,
        backend="des", halo_px=_HALO_PX, coregister=True,
    )
    assert res is not None
    # The pre-2018 slot (2) never reached the des tile-graph.
    assert set(seen["slots"]) == {0, 1, 3}
    assert "2016-07-15" not in seen["slots"].values()
    # It was filled by l1c_sen2cor instead → all 4 slots present.
    assert l1c_dates == ["2016-07-15"]
    assert list(res["temporal_mask"]) == [1, 1, 1, 1]


def test_all_pre2018_skips_des_entirely(monkeypatch):
    """All slots pre-2018 → des_dates is empty → the `if des_dates:` guard holds
    and the tile-graph is never called; every slot is filled by l1c_sen2cor."""
    called = {"des": 0}

    def _fake_des(bbox_3006, slot_dates, source):
        called["des"] += 1
        return {}
    monkeypatch.setattr(fs, "fetch_tile_at_specific_dates", _fake_des)

    l1c_dates: list = []

    def _fake_cube(bbox_3006, coords_wgs84, date_str, *, size_px):
        l1c_dates.append(date_str)
        return _structured_frame(np.random.default_rng(1), 0.0, 0.0)
    monkeypatch.setattr(fs, "_l1c_sen2cor_allband_cube", _fake_cube)

    dates = {0: "2016-09-10", 1: "2017-06-01", 2: "2017-07-15", 3: "2016-08-20"}
    res = fs.fetch_tile_spectral(
        _CENTRE, tile=TileConfig(size_px=_CANON), dates=dates, n_frames=_N,
        backend="des", halo_px=_HALO_PX, coregister=True,
    )
    assert res is not None
    assert called["des"] == 0                              # tile-graph never called
    assert sorted(l1c_dates) == sorted(dates.values())    # every slot via l1c
    assert list(res["temporal_mask"]) == [1, 1, 1, 1]


def test_coreg_quality_fields(monkeypatch):
    frames = _frames()
    _patch_fetch(monkeypatch, frames)
    res = fs.fetch_tile_spectral(
        _CENTRE, tile=TileConfig(size_px=_CANON), dates=_DATES, n_frames=_N,
        backend="des", halo_px=_HALO_PX, coregister=True,
    )
    shifts = res["coreg_shifts"]
    ref = int(res["coreg_ref_frame"])
    assert shifts.shape == (_N, 2)
    assert tuple(shifts[ref]) == (0.0, 0.0)             # anchor is the fixed reference
    assert tuple(shifts[2]) == (0.0, 0.0)               # slot 2 missing → zero
    # n_aligned == count of non-anchor slots actually shifted; M2 aligned >= 1 mover
    n_real = sum(1 for fi in range(_N)
                 if fi != ref and abs(shifts[fi][0]) + abs(shifts[fi][1]) > 0)
    assert int(res["coreg_n_aligned"]) == n_real
    assert n_real >= 1
    assert float(res["coreg_max_shift"]) == pytest.approx(
        max(float(np.hypot(*shifts[fi])) for fi in range(_N)))
    assert 0.9 < float(res["coreg_anchor_valid_frac"]) <= 1.0


def test_coregister_false_skips_m2(monkeypatch):
    frames = _frames()
    _patch_fetch(monkeypatch, frames)
    res = fs.fetch_tile_spectral(
        _CENTRE, tile=TileConfig(size_px=_CANON), dates=_DATES, n_frames=_N,
        backend="des", halo_px=_HALO_PX, coregister=False,
    )
    raw_spec, _, _ = _reference_composition(frames, coregister=False)
    assert int(res["coreg_m2"]) == 0
    assert np.array_equal(res["spectral"], raw_spec)
    # no M2 → zeroed coreg-quality signals (anchor frac is still measured).
    assert int(res["coreg_n_aligned"]) == 0
    assert float(res["coreg_max_shift"]) == 0.0
    assert np.array_equal(res["coreg_shifts"], np.zeros((_N, 2), np.float32))


def test_geometry_co_centred(monkeypatch):
    frames = _frames()
    _patch_fetch(monkeypatch, frames)
    res = fs.fetch_tile_spectral(
        _CENTRE, tile=TileConfig(size_px=_CANON), dates=_DATES, n_frames=_N,
    )
    west, south, east, north = res["bbox_3006"]
    assert east - west == _CANON * 10 and north - south == _CANON * 10
    assert int(res["easting"]) == (west + east) // 2
    assert int(res["northing"]) == (south + north) // 2
    assert int(res["tile_size_px"]) == _CANON


def test_returns_none_when_no_slot_fetched(monkeypatch):
    _patch_fetch(monkeypatch, {})   # tile-graph yields nothing
    res = fs.fetch_tile_spectral(
        _CENTRE, tile=TileConfig(size_px=_CANON), dates=_DATES, n_frames=_N,
    )
    assert res is None


def test_rejects_non_m2_backend():
    with pytest.raises(ValueError, match="cannot do M1"):
        fs.fetch_tile_spectral(
            _CENTRE, tile=TileConfig(size_px=_CANON), dates=_DATES, n_frames=_N,
            backend="cdse",
        )
