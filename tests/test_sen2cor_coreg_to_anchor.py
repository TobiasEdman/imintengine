"""`_coreg_frame_to_anchor` in run_sen2cor_per_scene — sign convention + guards.

Mirrors `test_coregistration.test_coregister_to_reference_removes_shift_dot_com`
(CLAUDE.md: "Testa koreg ALLTID med dot/center-of-mass"): inject a known
sub-pixel shift into the fresh frame and confirm the wrapper drives its bright-dot
centre-of-mass BACK onto the tile's anchor (misregistration removed, not doubled —
the 54b30a3 sign-inverted trap), reading the anchor from `coreg_ref_frame`. Plus
the three no-valid-anchor guards.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import center_of_mass, gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "sen2cor_pipeline"))
import run_sen2cor_per_scene as rsp  # noqa: E402
from imint.coregistration import subpixel_shift  # noqa: E402


def _dotted(size: int = 160) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    centres = [(58.0, 64.0), (96.0, 102.0), (72.0, 110.0), (104.0, 60.0), (120.0, 120.0)]
    tex = gaussian_filter(
        np.random.default_rng(7).standard_normal((size, size)).astype(np.float32), 2.0
    )
    base = 0.05 * (tex - tex.min())
    for cy, cx in centres:
        base = base + np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.0 ** 2)).astype(np.float32)
    return base.astype(np.float32)


def _dot_com(band2: np.ndarray) -> np.ndarray:
    return np.array(center_of_mass(np.clip(band2 - 0.3, 0.0, None)))


def _write_tile(tmp_path: Path, anchor6: np.ndarray, ref_idx: int = 1, n_slots: int = 4) -> Path:
    h, w = anchor6.shape[1], anchor6.shape[2]
    spectral = np.zeros((n_slots * 6, h, w), dtype=np.float32)
    spectral[ref_idx * 6:(ref_idx + 1) * 6] = anchor6
    p = tmp_path / "tile_400000_6400000.npz"
    np.savez_compressed(p, spectral=spectral, coreg_ref_frame=np.int32(ref_idx))
    return p


def test_coreg_drives_fresh_back_onto_anchor(tmp_path):
    base = _dotted()
    anchor6 = np.repeat(base[None], 6, axis=0)              # band 2 carries the dots
    dy0, dx0 = 0.6, -0.5                                    # sub-pixel; a sign flip doubles it
    shifted = subpixel_shift(base.astype(np.float64), dy0, dx0).astype(np.float32)
    fresh6 = np.repeat(shifted[None], 6, axis=0)

    aligned, meta = rsp._coreg_frame_to_anchor(fresh6, _write_tile(tmp_path, anchor6, ref_idx=1))
    assert aligned is not None and aligned.shape == fresh6.shape

    com_anchor = _dot_com(anchor6[2])
    pre = math.hypot(*(_dot_com(fresh6[2]) - com_anchor))
    post = math.hypot(*(_dot_com(aligned[2]) - com_anchor))
    assert pre > 0.4, f"injected shift too small to test: {pre:.3f}"
    assert post < pre * 0.6, f"coreg did not drive fresh back onto anchor: pre={pre:.3f} post={post:.3f}"


def test_coreg_skips_when_no_anchor_recorded(tmp_path):
    p = tmp_path / "t.npz"
    np.savez_compressed(p, spectral=np.ones((24, 32, 32), np.float32))  # no coreg_ref_frame
    aligned, meta = rsp._coreg_frame_to_anchor(np.ones((6, 32, 32), np.float32), p)
    assert aligned is None and meta["reason"] == "no_anchor"


def test_coreg_skips_when_anchor_empty(tmp_path):
    p = tmp_path / "t.npz"  # slot 1 all-zero
    np.savez_compressed(p, spectral=np.zeros((24, 32, 32), np.float32), coreg_ref_frame=np.int32(1))
    aligned, meta = rsp._coreg_frame_to_anchor(np.ones((6, 32, 32), np.float32), p)
    assert aligned is None and meta["reason"] == "anchor_empty"


def test_coreg_skips_when_ref_out_of_range(tmp_path):
    p = tmp_path / "t.npz"  # 4 slots, ref_idx 9 is out of range
    np.savez_compressed(p, spectral=np.ones((24, 32, 32), np.float32), coreg_ref_frame=np.int32(9))
    aligned, meta = rsp._coreg_frame_to_anchor(np.ones((6, 32, 32), np.float32), p)
    assert aligned is None and meta["reason"] == "anchor_out_of_range"
