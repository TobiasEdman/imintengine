"""Step-B SCL contract tests — per-slot SCL persistence + categorical integrity.

Two properties this pins, no network:

1. **SCL rides M2 as an INTEGER shift, never a sinc shift.** A categorical SCL
   raster with a single dot at a known position is fetched alongside a spectral
   frame carrying a KNOWN inter-frame drift. After ``fetch_tile_spectral`` runs
   M2 + halo-crop, the dot must have moved by EXACTLY the rounded M2 shift and
   the SCL must still contain ONLY its original class codes — a sinc/bilinear
   shift would ring across the class boundary and invent codes (jfr
   ``test_coregistration.py::test_coregister_to_reference_removes_shift_dot_com``,
   which is the spectral-side dot test this mirrors for the categorical band).

2. **npz persistence.** A synthetic 5-slot entry with a per-slot SCL block goes
   through ``fetch_unified_tiles._split_entry_result`` and lands as ``scl``
   ``(4, H, W)`` uint8 in ``core``, with a missing slot = all-zero (no_data).
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

from imint.training import fetch_spectral as fs
from imint.training.openeo_tile_graph import ALL_BANDS
from imint.training.tile_config import TileConfig

# _split_entry_result lives in scripts/fetch_unified_tiles.py — load it by path
# (scripts/ is not an importable package).
_FUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "fetch_unified_tiles.py")
_spec = importlib.util.spec_from_file_location("fetch_unified_tiles", _FUT_PATH)
fut = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fut)


_CANON = 128
_HALO_PX = 8
_HALO = _CANON + _HALO_PX
_CROP = _HALO_PX // 2
_N = 4
_CENTRE = (500000, 6500000)
_DATES = {0: "2021-09-10", 1: "2022-06-01", 2: "2022-07-15", 3: "2022-08-20"}


@pytest.fixture(autouse=True)
def _no_l1c_fallthrough(monkeypatch):
    monkeypatch.setattr(fs, "_l1c_sen2cor_allband_cube", lambda *a, **k: None)


def _structured_spectral(rng, dy: float, dx: float) -> np.ndarray:
    """(12, halo, halo) frame whose B04 (idx 2) carries a (dy, dx) sub-pixel drift
    — the geometry M2 estimates on."""
    from scipy.ndimage import gaussian_filter

    from imint.coregistration import subpixel_shift
    base = gaussian_filter(rng.standard_normal((_HALO, _HALO)).astype(np.float32), 1.6)
    base = (base - base.min()) / (base.max() - base.min())
    frame = np.repeat(base[None], len(ALL_BANDS), axis=0).astype(np.float32)
    frame[2] = subpixel_shift(base.astype(np.float64), dy, dx).astype(np.float32)
    return frame


def _dot_scl(row: int, col: int) -> np.ndarray:
    """(halo, halo) uint8 SCL: background class 4 (vegetation), one dot of class 9
    (cloud_high) at (row, col). Two distinct codes — a sinc shift would create
    intermediate values between 4 and 9, which the test forbids."""
    scl = np.full((_HALO, _HALO), 4, np.uint8)
    scl[row, col] = 9
    return scl


def test_scl_rides_m2_as_integer_shift_dot(monkeypatch):
    """The mover slot's SCL dot moves by EXACTLY the rounded M2 shift, and the SCL
    stays categorical (only {4, 9}). A near-1px drift → a 1px integer roll."""
    rng = np.random.default_rng(0)
    frames = {
        0: _structured_spectral(rng, 0.0, 0.0),
        1: _structured_spectral(rng, 0.0, 0.0),
    }
    dot_r, dot_c = _HALO // 2, _HALO // 2
    scls = {0: _dot_scl(dot_r, dot_c), 1: _dot_scl(dot_r, dot_c)}

    def _fake(bbox_3006, slot_dates, source, with_scl=False):
        assert with_scl, "entry must request SCL by default (Step-B contract)"
        return {fi: (frames[fi], scls[fi], None)
                for fi in slot_dates if fi in frames}
    monkeypatch.setattr(fs, "fetch_tile_at_specific_dates", _fake)

    # Pin the coreg step deterministically: the unit under test is the SCL
    # ROLL APPLICATION (sign + magnitude + categorical integrity), not MI
    # estimation quality (covered by tests/test_coregistration.py). Live MI
    # on the synthetic field estimated an off-axis shift whose rounded dx
    # was 0 — `+0 == -0` made the column assertion blind to a dx-only roll
    # sign flip (the 54b30a3 class of bug). A fixed (dy, dx) = (+1.3, -1.3)
    # rounds to (+1, -1): nonzero in BOTH axes, so a sign flip on either
    # axis moves the dot the wrong way and fails the position check.
    dy, dx = 1.3, -1.3
    monkeypatch.setattr(fs, "clearest_frame_idx", lambda frames, band=2: 0)
    monkeypatch.setattr(
        fs, "coregister_interframe",
        lambda frames, ref_idx, *, search_px: (frames, {0: (0.0, 0.0),
                                                        1: (dy, dx)}))

    res = fs.fetch_tile_spectral(
        _CENTRE, tile=TileConfig(size_px=_CANON), dates=_DATES, n_frames=_N,
        backend="des", halo_px=_HALO_PX, coregister=True,
    )
    assert res is not None
    scl_out = res["scl"]
    assert scl_out.shape == (_N, _CANON, _CANON)
    assert scl_out.dtype == np.uint8

    ref_idx = int(res["coreg_ref_frame"])
    mover = 1 if ref_idx == 0 else 0
    r_dy, r_dx = (int(round(v)) for v in res["coreg_shifts"][mover])
    # Sign-sensitivity precondition: both axes must exercise a nonzero
    # integer roll, or the position assertion below cannot catch a
    # per-axis sign flip.
    assert r_dy != 0 and r_dx != 0, (
        f"fixture must yield a nonzero rounded shift in both axes, "
        f"got ({r_dy}, {r_dx}) — increase the injected drift")

    # Categorical integrity: the mover's SCL must contain ONLY the two source
    # codes — no sinc-ringing artefacts between 4 and 9.
    assert set(np.unique(scl_out[mover])).issubset({4, 9}), (
        f"SCL gained non-source codes {set(np.unique(scl_out[mover]))} — "
        f"categorical band was sinc-shifted, not integer-rolled")

    # The dot moved by EXACTLY the rounded shift. Dot position in the CROPPED
    # frame = (halo dot pos + integer roll) - crop.
    ys, xs = np.where(scl_out[mover] == 9)
    assert len(ys) == 1, f"expected exactly one dot, got {len(ys)}"
    assert (ys[0], xs[0]) == (dot_r + r_dy - _CROP, dot_c + r_dx - _CROP)

    # The anchor's dot did NOT move (reference shift is (0,0)).
    ys0, xs0 = np.where(scl_out[ref_idx] == 9)
    assert (ys0[0], xs0[0]) == (dot_r - _CROP, dot_c - _CROP)


def test_scl_missing_slot_is_zero(monkeypatch):
    """A slot the tile-graph did not return has SCL all-zero (0 = no_data),
    coupled to temporal_mask==0."""
    rng = np.random.default_rng(1)
    frames = {  # slot 2 missing
        0: _structured_spectral(rng, 0.0, 0.0),
        1: _structured_spectral(rng, 0.2, -0.1),
        3: _structured_spectral(rng, -0.3, 0.15),
    }
    scls = {fi: np.full((_HALO, _HALO), 4, np.uint8) for fi in frames}

    def _fake(bbox_3006, slot_dates, source, with_scl=False):
        return {fi: (frames[fi], scls[fi], None)
                for fi in slot_dates if fi in frames}
    monkeypatch.setattr(fs, "fetch_tile_at_specific_dates", _fake)

    res = fs.fetch_tile_spectral(
        _CENTRE, tile=TileConfig(size_px=_CANON), dates=_DATES, n_frames=_N,
        backend="des", halo_px=_HALO_PX, coregister=True,
    )
    assert res is not None
    assert list(res["temporal_mask"]) == [1, 1, 0, 1]
    assert not np.any(res["scl"][2]), "missing slot's SCL must be all-zero no_data"
    assert np.any(res["scl"][0]) and np.any(res["scl"][3])


def _entry_with_scl(h: int = 8, w: int = 8) -> dict:
    """A minimal 5-slot fetch_tile_spectral result with a per-slot SCL block
    (slot fi filled with class code fi+1; slot 2 = 0 = missing/no_data)."""
    nb, nf = 6, 5
    spectral = np.zeros((nb * nf, h, w), np.float32)
    for fi in range(nf):
        spectral[fi * nb:(fi + 1) * nb] = fi + 1
    mask = np.ones(nf, np.uint8)
    mask[2] = 0
    scl = np.zeros((nf, h, w), np.uint8)
    for fi in range(nf):
        if fi != 2:
            scl[fi] = fi + 1
    return {
        "spectral": spectral,
        "temporal_mask": mask,
        "doy": np.zeros(nf, np.int32),
        "dates": np.array(["2021-09-10", "2022-06-01", "", "2022-08-20", "2016-07-15"]),
        "multitemporal": np.int32(1),
        "num_frames": np.int32(nf),
        "num_bands": np.int32(nb),
        "bbox_3006": np.array([400000, 6400000, 405120, 6405120], np.int32),
        "easting": np.int32(402560),
        "northing": np.int32(6402560),
        "tile_size_px": np.int32(512),
        "source": "des",
        "coreg_ref_frame": np.int32(0),
        "coreg_m2": np.int32(1),
        "coreg_n_aligned": np.int32(2),
        "coreg_max_shift": np.float32(1.0),
        "coreg_anchor_valid_frac": np.float32(0.95),
        "coreg_shifts": np.zeros((nf, 2), np.float32),
        "b08": np.ones((nf, h, w), np.float32), "b08_dates": np.array([""] * nf),
        "has_b08": np.int32(1),
        "rededge": np.ones((nf * 3, h, w), np.float32),
        "rededge_dates": np.array([""] * nf), "has_rededge": np.int32(1),
        "b01": np.ones((nf, h, w), np.float32), "b01_dates": np.array([""] * nf),
        "has_b01": np.int32(1),
        "b09": np.ones((nf, h, w), np.float32), "b09_dates": np.array([""] * nf),
        "has_b09": np.int32(1),
        "scl": scl,
    }


def test_split_entry_persists_scl_4frame_uint8(tmp_path):
    """_split_entry_result slices scl to the 4 persisted frames as uint8, and a
    round-trip through np.savez preserves it (disk-IO, per the repo rule)."""
    core, _, _ = fut._split_entry_result(_entry_with_scl())
    assert "scl" in core
    assert core["scl"].shape == (4, 8, 8)
    assert core["scl"].dtype == np.uint8
    # Slot fi carries class fi+1; slot 2 (missing) is all-zero no_data.
    for fi in range(4):
        expected = 0 if fi == 2 else fi + 1
        assert np.all(core["scl"][fi] == expected), fi

    out = tmp_path / "tile.npz"
    np.savez_compressed(out, **core)
    with np.load(out) as z:
        assert "scl" in z.files
        assert z["scl"].shape == (4, 8, 8)
        assert z["scl"].dtype == np.uint8
        assert np.array_equal(z["scl"], core["scl"])


def test_split_entry_no_scl_when_absent():
    """An entry fetched without SCL (with_scl=False) has no ``scl`` key → the
    split must not fabricate one."""
    entry = _entry_with_scl()
    del entry["scl"]
    core, _, _ = fut._split_entry_result(entry)
    assert "scl" not in core
