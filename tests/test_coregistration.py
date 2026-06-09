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
from scipy.ndimage import gaussian_filter

from imint.coregistration import estimate_mi_offset, subpixel_shift

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
