"""Lightweight single source of truth for tile-year resolution."""
from __future__ import annotations

from collections import Counter
from datetime import date

import numpy as np


SMOKE_AUTUMN_DOY_MIN = 227  # Aug 15, the fetch pipeline's autumn window
SMOKE_AUTUMN_DOY_MAX = 304  # Oct 31
SMOKE_GROWING_START = (4, 1)
SMOKE_GROWING_END = (9, 30)


def _get(npz_data, key):
    try:
        return npz_data.get(key)
    except AttributeError:
        return npz_data[key] if key in npz_data else None


def resolve_tile_year(npz_data) -> int | None:
    """Resolve one tile year and reject contradictory temporal metadata.

    The first temporal frame may legitimately be the previous autumn, so the
    date-derived year is the modal year rather than a requirement that every
    frame share one calendar year.  Explicit ``year`` and ``lpis_year`` values
    must agree with each other and with that modal growing-season year.
    """
    explicit = {
        key: int(value)
        for key in ("year", "lpis_year")
        if (value := _get(npz_data, key)) is not None
    }
    if len(set(explicit.values())) > 1:
        raise ValueError(f"Conflicting explicit tile years: {explicit}")
    dates = _get(npz_data, "dates")
    if dates is None:
        return next(iter(explicit.values()), None)
    years = []
    for value in dates:
        text = str(value)
        if len(text) >= 4 and text[:4].isdigit():
            years.append(int(text[:4]))
    date_year = None
    if years:
        counts = Counter(years)
        top = max(counts.values())
        date_year = max(
            year for year, count in counts.items() if count == top
        )
    explicit_year = next(iter(explicit.values()), None)
    if (
        explicit_year is not None
        and date_year is not None
        and explicit_year != date_year
    ):
        raise ValueError(
            "Explicit tile year disagrees with modal spectral date year: "
            f"explicit={explicit_year}, dates={date_year}"
        )
    return explicit_year if explicit_year is not None else date_year


def resolve_growing_cutoff(npz_data, year: int) -> str:
    """Return the last usable current-year growing frame, or fail loudly."""
    raw_dates = _get(npz_data, "dates")
    if raw_dates is None:
        raise ValueError("tile has no spectral dates")
    dates = [str(value) for value in raw_dates]
    if len(dates) < 2:
        raise ValueError("tile has no growing frames")

    raw_mask = _get(npz_data, "temporal_mask")
    if raw_mask is None:
        mask = np.ones(len(dates), dtype=np.uint8)
    else:
        mask = np.asarray(raw_mask).reshape(-1)
        if mask.size != len(dates):
            raise ValueError("temporal mask/date length mismatch")

    usable: list[str] = []
    for index, value in enumerate(dates[1:], start=1):
        if mask[index] <= 0:
            continue
        try:
            parsed = date.fromisoformat(value[:10])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"usable spectral frame has invalid date: index={index}, value={value!r}"
            ) from exc
        if parsed.year != year:
            raise ValueError(
                "spectral/label year mismatch: "
                f"growing_date={parsed.isoformat()}, label_year={year}"
            )
        usable.append(parsed.isoformat())
    if not usable:
        raise ValueError("tile has no usable current-year growing date")
    return max(usable)


def validate_smoke_temporal_metadata(
    npz_data,
    year: int,
    *,
    num_frames: int = 4,
    min_valid_frames: int = 3,
) -> dict:
    """Validate the exact temporal contract consumed by the sealed smoke.

    The normal dataset supports partial and legacy temporal layouts. The
    scientific A/B cohort deliberately requires autumn-(year-1) followed by
    growing-season observations, at least 3/4 real frames, and date/DOY
    agreement so masked-frame duplication is never counted as an acquisition.
    """
    raw_dates = _get(npz_data, "dates")
    raw_mask = _get(npz_data, "temporal_mask")
    raw_doy = _get(npz_data, "doy")
    if raw_dates is None or raw_mask is None or raw_doy is None:
        raise ValueError(
            "sealed smoke tile requires dates, temporal_mask, and doy"
        )
    dates = np.asarray(raw_dates).reshape(-1)
    mask = np.asarray(raw_mask).reshape(-1)
    doy = np.asarray(raw_doy).reshape(-1)
    if not (len(dates) == len(mask) == len(doy) == num_frames):
        raise ValueError(
            "sealed smoke temporal metadata/frame length mismatch: "
            f"dates={len(dates)}, mask={len(mask)}, doy={len(doy)}, "
            f"expected={num_frames}"
        )
    try:
        mask_numeric = mask.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("sealed smoke temporal mask is not numeric") from exc
    if (
        not np.isfinite(mask_numeric).all()
        or not np.isin(mask_numeric, (0.0, 1.0)).all()
    ):
        raise ValueError("sealed smoke temporal mask must contain only 0/1")
    valid = mask_numeric.astype(bool)
    valid_count = int(valid.sum())
    if valid_count < min_valid_frames:
        raise ValueError(
            f"sealed smoke tile has {valid_count}/{num_frames} valid frames; "
            f"requires at least {min_valid_frames}/{num_frames}"
        )
    if not valid[0]:
        raise ValueError("sealed smoke tile requires a valid autumn frame")

    parsed_dates: list[str | None] = []
    for index, (raw_date, is_valid, raw_day) in enumerate(
        zip(dates, valid, doy)
    ):
        if not is_valid:
            parsed_dates.append(None)
            continue
        try:
            parsed = date.fromisoformat(str(raw_date)[:10])
            parsed_doy = int(raw_day)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"sealed smoke frame {index} has invalid date/DOY"
            ) from exc
        expected_year = year - 1 if index == 0 else year
        if parsed.year != expected_year:
            raise ValueError(
                f"sealed smoke frame {index} year={parsed.year}, "
                f"expected={expected_year}"
            )
        if parsed_doy != parsed.timetuple().tm_yday:
            raise ValueError(
                f"sealed smoke frame {index} date/DOY mismatch: "
                f"date={parsed.isoformat()}, doy={parsed_doy}"
            )
        if index == 0:
            if not SMOKE_AUTUMN_DOY_MIN <= parsed_doy <= SMOKE_AUTUMN_DOY_MAX:
                raise ValueError(
                    f"sealed smoke autumn frame is outside DOY "
                    f"{SMOKE_AUTUMN_DOY_MIN}..{SMOKE_AUTUMN_DOY_MAX}: "
                    f"{parsed.isoformat()}"
                )
        elif not (
            date(year, *SMOKE_GROWING_START)
            <= parsed
            <= date(year, *SMOKE_GROWING_END)
        ):
            raise ValueError(
                "sealed smoke growing frame is outside Apr 1..Sep 30: "
                f"{parsed.isoformat()}"
            )
        parsed_dates.append(parsed.isoformat())

    cutoff = resolve_growing_cutoff(npz_data, year)
    return {
        "dates": parsed_dates,
        "mask": [int(value) for value in valid],
        "doy": [int(value) for value in doy],
        "valid_frames": valid_count,
        "cutoff_date": cutoff,
    }
