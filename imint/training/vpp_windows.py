"""VPP-guided per-tile growing season window computation.

Uses HR-VPP phenology data (SOSD/EOSD) to determine the actual growing
season for each tile location, then divides it into N equal temporal
windows for Sentinel-2 fetching.

This handles the natural variation across Sweden:
  - South Sweden: growing season ~April to ~October (~200 days)
  - North Sweden: shorter growing season, later start
  - Coastal vs mountain: different phenological timing

HR-VPP Season 1 SOSD/EOSD bands are **YYDDD-encoded**:
``raw = (year - 2000) * 1000 + day_of_year`` (verified against the WEkEO
COGs, V101 and V105 alike — e.g. raw 21125 → 2021 DOY 125 = May 5).
  - SOSD = Start Of Season = spring green-up (DOY ~85–165)
  - EOSD = End Of Season   = autumn senescence (DOY ~200–310)

Growing season = SOSD_doy → EOSD_doy. Decode the day-of-year with
``raw % 1000``. (Reading the raw integer as a CNES/1960 Julian day — a
prior bug — silently mis-dated every tile; see git history.)
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

# Fallback growing-season windows (DOY) when VPP data is unavailable.
# Per num_frames: each list spans the full Apr–Aug growing season so the
# last window always reaches the late-summer / harvest signal that
# Clay/CROMA single-snapshot training needs. PR-#15 cap_doy=244 is the
# effective right edge for the last window (Sep 1).
_FALLBACK_BY_NUM_FRAMES: dict[int, list[tuple[int, int]]] = {
    3: [
        (91,  150),   # spring        (Apr–end May, 60d)
        (151, 200),   # early-summer  (Jun–mid Jul, 50d)
        (201, 244),   # late-summer   (mid-Jul–Aug, 44d, capped Sep 1)
    ],
    4: [
        (91,  120),   # April:          leaf-out, conifer vs deciduous
        (121, 166),   # May–mid June:   vegetation green-up
        (167, 212),   # mid June–July:  peak NDVI, forest type separation
        (213, 244),   # August:         senescence, harvest (was 273, capped)
    ],
}
# Backward-compat alias — callers that historically used the 4-frame list
# directly still see the same shape, but the last window is now capped.
_FALLBACK_DOY_WINDOWS: list[tuple[int, int]] = _FALLBACK_BY_NUM_FRAMES[4]

# Sanity bounds for growing season detection
_MIN_GROWING_SEASON_DOY = 60     # No earlier than March 1
_MAX_GROWING_SEASON_DOY = 330    # No later than November 26
_MIN_GROWING_SEASON_LENGTH = 60  # At least 60 days

# Cap for the growing-season END used in window division. VPP-derived
# EOSD for Swedish vegetation routinely extends into Oct-Nov (forest
# leaf-fall, late-grass dormancy), which made the LAST of N divided
# windows fall in late autumn — useless for crop discrimination, where
# the most spectrally distinctive signal is the May-September period
# (peak vegetation through cereal maturation/harvest). Cap at DOY 244
# (Sep 1) so the last window captures late-summer / early-autumn
# maturity rather than full senescence.
# See conversation_log entry 2026-05-21 for the bug analysis: tiles
# 43983958 + 43983968 (lat 58.7-58.8) ended up with no June-July frame,
# falling back to Oct 19 / Sep 1 for the third growing-season frame.
_GROWING_SEASON_END_CAP_DOY = 244  # September 1 (non-leap; leap = Aug 31)


def vpp_yyddd_to_doy(values: np.ndarray) -> np.ndarray:
    """Decode HR-VPP YYDDD date values to day-of-year, elementwise.

    HR-VPP SOSD/EOSD bands store ``(year - 2000) * 1000 + day_of_year``.
    The day-of-year is therefore ``value % 1000``. NoData (0) decodes to
    0 and must be masked by the caller.

    Args:
        values: SOSD or EOSD raw band values (any shape).

    Returns:
        Day-of-year array, same shape as ``values``.
    """
    return np.mod(values, 1000)


def compute_growing_season_doy(
    sosd_arr: np.ndarray,
    eosd_arr: np.ndarray,
) -> tuple[int, int] | None:
    """Compute growing season start/end DOY from HR-VPP phenology arrays.

    SOSD (Start Of Season, spring green-up) and EOSD (End Of Season,
    autumn senescence) are YYDDD-encoded. We decode each valid pixel to
    its day-of-year (robust to mixed-year pixels at tile edges), take the
    median, and return the ``(SOSD_doy, EOSD_doy)`` growing-season span.

    Args:
        sosd_arr: (H, W) SOSD values, YYDDD-encoded. NoData = 0.
        eosd_arr: (H, W) EOSD values, YYDDD-encoded. NoData = 0.

    Returns:
        (gs_start_doy, gs_end_doy) tuple, or None if VPP data is
        insufficient or the decoded span is implausible for Swedish
        vegetation.
    """
    sosd_valid = sosd_arr[sosd_arr > 0]
    eosd_valid = eosd_arr[eosd_arr > 0]

    # Need enough valid pixels for reliable estimate
    min_pixels = max(10, sosd_arr.size * 0.05)
    if len(sosd_valid) < min_pixels or len(eosd_valid) < min_pixels:
        return None

    gs_start = int(np.median(vpp_yyddd_to_doy(sosd_valid)))  # spring green-up
    gs_end = int(np.median(vpp_yyddd_to_doy(eosd_valid)))    # autumn senescence

    # Plausibility gate: green-up in spring, senescence in autumn, and a
    # season at least _MIN_GROWING_SEASON_LENGTH long. Rejects tiles whose
    # decoded phenology is degenerate (sparse/edge pixels, water).
    if (
        _MIN_GROWING_SEASON_DOY <= gs_start <= 180
        and 200 <= gs_end <= _MAX_GROWING_SEASON_DOY
        and gs_end - gs_start >= _MIN_GROWING_SEASON_LENGTH
    ):
        return (gs_start, gs_end)

    return None


def compute_growing_season_windows(
    sosd_arr: np.ndarray,
    eosd_arr: np.ndarray,
    num_frames: int = 4,
) -> list[tuple[int, int]]:
    """Compute per-tile DOY windows from VPP phenology data.

    Divides the growing season into ``num_frames`` equal windows,
    customized to the tile's actual phenological cycle.

    Args:
        sosd_arr: (H, W) float32 SOSD in CNES Julian Days.
        eosd_arr: (H, W) float32 EOSD in CNES Julian Days.
        num_frames: Number of temporal windows (default: 4).

    Returns:
        List of ``(doy_start, doy_end)`` tuples, one per frame.
        Falls back to fixed growing-season windows if VPP data
        is unavailable or unreliable.
    """
    gs = compute_growing_season_doy(sosd_arr, eosd_arr)

    # Pick a fallback that's appropriate for the requested num_frames —
    # 3-frame is NOT just _FALLBACK_DOY_WINDOWS[:3] (which caps at DOY 212
    # = Jul 31, losing the late-summer/harvest signal needed for Clay/
    # CROMA single-snapshot input). Use _FALLBACK_BY_NUM_FRAMES for a
    # full Apr–Aug span tailored per num_frames count.
    fallback = _FALLBACK_BY_NUM_FRAMES.get(num_frames, _FALLBACK_DOY_WINDOWS)
    fallback = fallback[:num_frames]

    if gs is None:
        return fallback

    gs_start, gs_end = gs
    # Cap the effective growing-season end so the last window doesn't
    # extend into senescence/leaf-fall (see _GROWING_SEASON_END_CAP_DOY
    # rationale at module top).
    gs_end = min(gs_end, _GROWING_SEASON_END_CAP_DOY)

    # After capping, ensure the remaining window is still meaningful.
    # If gs_start is already very late (extreme north / weird tile),
    # the capped range may be too short to divide. Fall back in that
    # case rather than producing degenerate single-pixel windows.
    if gs_end - gs_start < _MIN_GROWING_SEASON_LENGTH:
        return fallback

    gs_length = gs_end - gs_start
    window_length = gs_length / num_frames

    windows = []
    for i in range(num_frames):
        w_start = int(gs_start + i * window_length)
        w_end = int(gs_start + (i + 1) * window_length) - 1
        if i == num_frames - 1:
            w_end = gs_end  # ensure last window reaches the (capped) end
        windows.append((w_start, w_end))

    return windows


def doy_to_date_str(year: int, doy: int) -> str:
    """Convert year + day-of-year to ISO date string.

    Args:
        year: Calendar year.
        doy: Day of year (1-based, 1=Jan 1).

    Returns:
        ``"YYYY-MM-DD"`` string.
    """
    dt = datetime(year, 1, 1) + timedelta(days=doy - 1)
    return dt.strftime("%Y-%m-%d")


def doy_windows_to_month_windows(
    doy_windows: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Convert DOY windows to approximate (start_month, end_month) tuples.

    Useful for backward-compatible display and logging.

    Args:
        doy_windows: List of (doy_start, doy_end) tuples.

    Returns:
        List of (start_month, end_month) tuples.
    """
    month_windows = []
    for doy_start, doy_end in doy_windows:
        m_start = (datetime(2019, 1, 1) + timedelta(days=doy_start - 1)).month
        m_end = (datetime(2019, 1, 1) + timedelta(days=doy_end - 1)).month
        month_windows.append((m_start, m_end))
    return month_windows
