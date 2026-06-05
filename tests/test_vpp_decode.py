"""Regression tests for the HR-VPP YYDDD date decode.

HR-VPP SOSD/EOSD bands encode dates as ``(year-2000)*1000 + day_of_year``
(YYDDD). A prior bug read the raw integer as a CNES/1960 Julian day, which
silently mis-dated every tile: 2018 tiles fell back to hardcoded windows,
2021/2022 tiles produced ~1-month-early windows that merely looked valid.

These tests pin the correct decode at every consumer:
  * vpp_windows.vpp_yyddd_to_doy / compute_growing_season_doy / _windows
  * unified_dataset.normalize_aux_channel (the shared aux transform)

Raw values used here are taken from the real WEkEO COGs:
  SOSD 18122 -> 2018 DOY 122 (May 2); EOSD 18277 -> 2018 DOY 277 (Oct 4).
"""
from __future__ import annotations

import numpy as np
import pytest

from imint.training.vpp_windows import (
    compute_growing_season_doy,
    compute_growing_season_windows,
    vpp_yyddd_to_doy,
)
from imint.training.unified_dataset import AUX_NORM, normalize_aux_channel


def _const_arr(value: float, n: int = 64) -> np.ndarray:
    return np.full((n, n), value, dtype=np.float32)


# ── decode ──────────────────────────────────────────────────────────────

def test_yyddd_decode_elementwise():
    raw = np.array([18122, 21125, 22162, 17049, 0], dtype=np.float32)
    doy = vpp_yyddd_to_doy(raw)
    np.testing.assert_array_equal(doy, [122, 125, 162, 49, 0])


def test_yyddd_decode_is_year_independent():
    # Same day-of-year across different product years -> same DOY.
    assert int(vpp_yyddd_to_doy(np.array([18122]))[0]) == 122
    assert int(vpp_yyddd_to_doy(np.array([22122]))[0]) == 122


# ── growing-season DOY ──────────────────────────────────────────────────

def test_growing_season_doy_decodes_yyddd():
    # 2018 tile: SOSD May 2 (DOY 122), EOSD Oct 4 (DOY 277).
    gs = compute_growing_season_doy(_const_arr(18122), _const_arr(18277))
    assert gs == (122, 277)


def test_growing_season_doy_not_cnes():
    # Regression: the old CNES misread of 18122 gave DOY 225 (Aug 13).
    gs = compute_growing_season_doy(_const_arr(18122), _const_arr(18277))
    assert gs is not None and gs[0] == 122 and gs[0] != 225


def test_growing_season_doy_year_independent():
    # 2018 and 2022 tiles with identical phenology -> identical window.
    gs_2018 = compute_growing_season_doy(_const_arr(18122), _const_arr(18277))
    gs_2022 = compute_growing_season_doy(_const_arr(22122), _const_arr(22277))
    assert gs_2018 == gs_2022 == (122, 277)


def test_growing_season_doy_insufficient_pixels_falls_back():
    sosd = np.zeros((64, 64), dtype=np.float32)
    sosd[0, :5] = 18122  # < 5% valid
    assert compute_growing_season_doy(sosd, _const_arr(18277)) is None


def test_growing_season_doy_implausible_rejected():
    # EOSD before SOSD-plausible window -> rejected (gate guards garbage).
    assert compute_growing_season_doy(_const_arr(18300), _const_arr(18020)) is None


# ── window division ─────────────────────────────────────────────────────

def test_windows_span_growing_season():
    wins = compute_growing_season_windows(
        _const_arr(18122), _const_arr(18277), num_frames=3,
    )
    assert len(wins) == 3
    # Spring start (May) and capped late-summer end (Sep 1 = DOY 244).
    assert 110 <= wins[0][0] <= 135
    assert wins[-1][1] <= 244
    # Monotonic, non-overlapping.
    flat = [d for w in wins for d in w]
    assert flat == sorted(flat)


# ── aux normalization ───────────────────────────────────────────────────

def test_normalize_aux_vpp_decodes_and_is_year_independent():
    mean, std = AUX_NORM["vpp_sosd"]
    expected = (122 - mean) / std
    v2018 = normalize_aux_channel("vpp_sosd", 18122.0)
    v2022 = normalize_aux_channel("vpp_sosd", 22122.0)
    assert v2018 == pytest.approx(expected, abs=1e-4)
    assert v2018 == pytest.approx(v2022, abs=1e-6)  # the key regression


def test_normalize_aux_vpp_nodata_maps_to_zero():
    # NoData (0) must normalize to ~0, not a huge negative outlier.
    assert normalize_aux_channel("vpp_sosd", 0.0) == pytest.approx(0.0, abs=1e-6)


def test_normalize_aux_vpp_array():
    arr = np.array([[18122, 0], [18277, 22122]], dtype=np.float32)
    out = normalize_aux_channel("vpp_sosd", arr)
    mean, std = AUX_NORM["vpp_sosd"]
    assert out.shape == (2, 2)
    assert out[0, 1] == pytest.approx(0.0, abs=1e-6)          # nodata
    assert out[0, 0] == pytest.approx((122 - mean) / std, abs=1e-4)
    assert out[1, 1] == pytest.approx((122 - mean) / std, abs=1e-4)  # year-indep


def test_normalize_aux_nondate_channel_unchanged_behaviour():
    # dem is a plain z-score channel: (264 - 264.03)/215.37 ~ 0.
    mean, std = AUX_NORM["dem"]
    assert normalize_aux_channel("dem", mean) == pytest.approx(0.0, abs=1e-4)
