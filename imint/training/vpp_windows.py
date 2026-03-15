"""VPP-guided per-tile growing season window computation.

Uses HR-VPP phenology data (SOSD/EOSD) to determine the actual growing
season for each tile location, then divides it into N equal temporal
windows for Sentinel-2 fetching.

This handles the natural variation across Sweden:
  - South Sweden: growing season ~April to ~October (~200 days)
  - North Sweden: shorter growing season, later start
  - Coastal vs mountain: different phenological timing

HR-VPP Season 1 encodes dates as CNES Julian Days (days since 1960-01-01).
The "season" in SOSD/EOSD refers to the dormancy season:
  - SOSD = Start of (dormancy) Season = autumn senescence
  - EOSD = End of (dormancy) Season = spring green-up

Growing season = EOSD_doy → SOSD_doy (from spring green-up to autumn dormancy).
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

# CNES Julian Day epoch
_CNES_EPOCH = datetime(1960, 1, 1)

# Fallback growing-season windows (DOY) if VPP data unavailable
# Covers April through September in 4 roughly equal windows
_FALLBACK_DOY_WINDOWS: list[tuple[int, int]] = [
    (91, 120),    # April:          leaf-out, conifer vs deciduous
    (121, 166),   # May–mid June:   vegetation green-up
    (167, 212),   # mid June–July:  peak NDVI, forest type separation
    (213, 273),   # August–Sept:    senescence, colour change
]

# Sanity bounds for growing season detection
_MIN_GROWING_SEASON_DOY = 60     # No earlier than March 1
_MAX_GROWING_SEASON_DOY = 330    # No later than November 26
_MIN_GROWING_SEASON_LENGTH = 60  # At least 60 days


def cnes_to_doy(cnes_val: float) -> int:
    """Convert CNES Julian Day to day-of-year (1-365/366).

    Args:
        cnes_val: Days since 1960-01-01.

    Returns:
        Day-of-year (1-based).
    """
    dt = _CNES_EPOCH + timedelta(days=int(cnes_val))
    return dt.timetuple().tm_yday


def cnes_to_month(cnes_val: float) -> int:
    """Convert CNES Julian Day to month (1-12)."""
    dt = _CNES_EPOCH + timedelta(days=int(cnes_val))
    return dt.month


def compute_growing_season_doy(
    sosd_arr: np.ndarray,
    eosd_arr: np.ndarray,
) -> tuple[int, int] | None:
    """Compute growing season start/end DOY from VPP arrays.

    Takes the median of non-zero pixels for SOSD and EOSD,
    converts from CNES Julian Days to DOY, and determines the
    growing season boundaries.

    Args:
        sosd_arr: (H, W) float32 SOSD values in CNES Julian Days.
        eosd_arr: (H, W) float32 EOSD values in CNES Julian Days.

    Returns:
        (gs_start_doy, gs_end_doy) tuple, or None if VPP data
        is insufficient for reliable window computation.
    """
    sosd_valid = sosd_arr[sosd_arr > 0]
    eosd_valid = eosd_arr[eosd_arr > 0]

    # Need enough valid pixels for reliable estimate
    min_pixels = max(10, sosd_arr.size * 0.05)
    if len(sosd_valid) < min_pixels or len(eosd_valid) < min_pixels:
        return None

    sosd_median = float(np.median(sosd_valid))
    eosd_median = float(np.median(eosd_valid))

    sosd_doy = cnes_to_doy(sosd_median)
    eosd_doy = cnes_to_doy(eosd_median)

    # HR-VPP dormancy interpretation:
    # SOSD = dormancy start (autumn), EOSD = dormancy end (spring)
    # Growing season = EOSD_doy → SOSD_doy
    #
    # But we also handle the alternative where SOSD < EOSD
    # (standard start/end interpretation), by checking which
    # produces a reasonable growing season.

    # Try both interpretations and pick the one that makes sense
    candidate_a = (eosd_doy, sosd_doy)  # dormancy interpretation
    candidate_b = (sosd_doy, eosd_doy)  # standard interpretation

    for gs_start, gs_end in [candidate_a, candidate_b]:
        length = gs_end - gs_start
        if (
            _MIN_GROWING_SEASON_DOY <= gs_start <= 180
            and 200 <= gs_end <= _MAX_GROWING_SEASON_DOY
            and length >= _MIN_GROWING_SEASON_LENGTH
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

    if gs is None:
        return _FALLBACK_DOY_WINDOWS[:num_frames]

    gs_start, gs_end = gs
    gs_length = gs_end - gs_start
    window_length = gs_length / num_frames

    windows = []
    for i in range(num_frames):
        w_start = int(gs_start + i * window_length)
        w_end = int(gs_start + (i + 1) * window_length) - 1
        if i == num_frames - 1:
            w_end = gs_end  # ensure last window reaches the end
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
