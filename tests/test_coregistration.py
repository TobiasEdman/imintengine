"""Tests for imint.coregistration.estimate_mi_offset.

The load-bearing property: mutual information recovers a sub-pixel geometric
shift even when the two frames have *different content* (the multi-temporal
Sentinel-2 case — autumn stubble vs summer canopy), where intensity-based phase
correlation latches onto phenology instead of geometry.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.ndimage import center_of_mass, gaussian_filter

from imint.coregistration import (
    coregister_to_reference,
    estimate_mi_offset,
    subpixel_shift,
)

_N = 220
# Season-stable geometry: 3 field polygons + a road cross. Boundaries/roads are
# the shared signal; field *values* are what changes between "seasons".
_LAB = np.zeros((_N, _N), int)
_LAB[30:110, 20:90] = 1
_LAB[120:185, 40:150] = 2
_LAB[45:165, 120:195] = 3
_ROAD = np.zeros((_N, _N), np.float32)
_ROAD[:, 98:101] = 1.0
_ROAD[85:88, :] = 1.0


def _scene(field_vals: dict[int, float], texture: bool, seed: int) -> np.ndarray:
    r = np.random.default_rng(seed)
    img = np.zeros((_N, _N), np.float32)
    for k, v in field_vals.items():
        img[_LAB == k] = v
    img += 0.5 * _ROAD
    if texture:  # summer crops: within-field texture, absent in autumn
        img += 0.18 * gaussian_filter(r.standard_normal((_N, _N)).astype(np.float32), 1.5) * (_LAB > 0)
    img += 0.02 * r.standard_normal((_N, _N))
    return gaussian_filter(img, 1.0).astype(np.float32)


_AUTUMN = _scene({0: 0.20, 1: 0.25, 2: 0.22, 3: 0.28}, texture=False, seed=1)
_SUMMER = _scene({0: 0.20, 1: 0.62, 2: 0.14, 3: 0.70}, texture=True, seed=2)  # different content


def _recovers(moving_base, sy, sx, ref, tol):
    """estimate_mi_offset returns the shift to APPLY to moving to reach ref, so
    for moving = shift(moving_base, sy, sx) the answer must be ≈ (-sy, -sx)."""
    moving = subpixel_shift(moving_base, sy, sx)
    dy, dx = estimate_mi_offset(moving, ref, search_px=4.0)
    return math.hypot(dy + sy, dx + sx), (dy, dx)


@pytest.mark.parametrize("sy,sx", [(1.3, -0.7), (-0.9, 1.1), (2.0, 0.0)])
def test_recovers_known_shift_same_content(sy, sx):
    err, got = _recovers(_AUTUMN, sy, sx, _AUTUMN, tol=0.25)
    assert err < 0.25, f"same-content shift ({sy},{sx}) not recovered: got {got}, err={err:.2f}"


@pytest.mark.parametrize("sy,sx", [(1.3, -0.7), (-1.1, 0.8)])
def test_recovers_shift_through_content_change(sy, sx):
    # ref = autumn, moving = a SHIFTED summer (different field values + texture).
    # MI must still recover the geometry; phase correlation cannot.
    err, got = _recovers(_SUMMER, sy, sx, _AUTUMN, tol=0.4)
    assert err < 0.4, f"cross-season shift ({sy},{sx}) not recovered: got {got}, err={err:.2f}"


def test_out_of_range_returns_zero():
    moving = subpixel_shift(_AUTUMN, 6.0, 0.0)   # beyond search_px=4
    assert estimate_mi_offset(moving, _AUTUMN, search_px=4.0) == (0.0, 0.0)


def test_no_shift_returns_near_zero():
    dy, dx = estimate_mi_offset(_AUTUMN.copy(), _AUTUMN, search_px=4.0)
    assert max(abs(dy), abs(dx)) < 0.1, (dy, dx)


def test_coregister_to_reference_removes_shift_dot_com():
    """Sign-convention guard for the production MI swap — the 54b30a3 trap.

    ``coregister_to_reference`` shifts its *reference* onto its *target*. We inject
    a known sub-pixel shift into the reference and confirm the reference's
    bright-dot centre-of-mass is driven BACK onto the target (misregistration
    removed), not pushed to ~2x (the sign-inverted failure). COM is a direct
    geometric position measure, so — unlike a residual metric — it cannot be
    fooled by an estimator that rejects an out-of-range (doubled) offset as
    ``(0, 0)``. (CLAUDE.md: "Testa koreg ALLTID med dot/center-of-mass".)
    """
    size = 160
    yy, xx = np.mgrid[0:size, 0:size]
    # Sharp dots (sigma=2) on faint non-negative texture: the dots dominate the COM
    # while the texture gives MI a rich joint histogram. Centres sit inside the
    # central MI window and clear of the edges (no Fourier-shift wrap into the COM).
    centres = [(58.0, 64.0), (96.0, 102.0), (72.0, 110.0), (104.0, 60.0), (120.0, 120.0)]
    sigma = 2.0
    tex = gaussian_filter(
        np.random.default_rng(7).standard_normal((size, size)).astype(np.float32), 2.0
    )
    base = 0.05 * (tex - tex.min())
    for cy, cx in centres:
        base = base + np.exp(
            -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2)
        ).astype(np.float32)

    dy0, dx0 = 0.6, -0.5  # sub-pixel (< 0.95·search_px); a sign flip doubles to 1.2/1.0 px
    shifted = subpixel_shift(base.astype(np.float64), dy0, dx0).astype(np.float32)
    target = np.repeat(base[..., None], 3, axis=-1)  # fixed anchor
    reference = np.repeat(shifted[..., None], 3, axis=-1)  # misregistered by +(dy0, dx0)

    def _dot_com(frame):  # isolate the dot cores from the texture floor
        return np.array(center_of_mass(np.clip(frame[..., 2] - 0.3, 0.0, None)))

    com_target = _dot_com(target)
    pre_err = math.hypot(*(_dot_com(reference) - com_target))
    assert pre_err > 0.4, f"injected misregistration too small to test: {pre_err:.3f}"

    _aligned_tgt, aligned_ref, meta = coregister_to_reference(
        target=target.copy(),
        reference=reference.copy(),
        target_transform=None,
        reference_transform=None,
        subpixel=True,
        reference_band=2,
    )
    post_err = math.hypot(*(_dot_com(aligned_ref) - com_target))

    # Misregistration REMOVED: dot driven onto the target, residual well sub-pixel.
    assert post_err < 0.1, (
        f"reference dot not driven onto target: pre={pre_err:.3f} "
        f"post={post_err:.3f} applied={tuple(round(v, 3) for v in meta['subpixel_offset'])}"
    )
    assert post_err < pre_err * 0.3
    # Sign guard: the APPLIED shift is the negative of the injected one (the
    # reference dot moves from c+delta back to c). A positive/doubled result == 54b30a3.
    s_dy, s_dx = meta["subpixel_offset"]
    assert s_dy * dy0 < 0 and s_dx * dx0 < 0, (
        f"sign inverted/wrong: injected=({dy0},{dx0}) applied=({s_dy:.3f},{s_dx:.3f})"
    )
