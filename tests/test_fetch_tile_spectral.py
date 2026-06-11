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
