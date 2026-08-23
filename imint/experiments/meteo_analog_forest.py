"""Core logic for meteorology-matched forest Sentinel-2 comparisons."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from metafilter import AnalogMatch, AnalogModel


FOREST_CLASSES = frozenset({1, 2, 3, 4, 5})
SCL_REJECTED_CLASSES = frozenset({1, 3, 7, 8, 9, 10, 11})
PRITHVI_BANDS = ("B02", "B03", "B04", "B8A", "B11", "B12")
COMPARISON_BANDS = ("B02", "B03", "B04", "B08", "B8A", "B11", "B12")


@dataclass(frozen=True)
class TileCandidate:
    easting: int
    northing: int
    forest_fraction: float
    source: str = ""
    classified_fraction: float = 1.0


@dataclass(frozen=True)
class DateMatch:
    reference_year: int
    reference_date: str
    candidate_year: int
    candidate_date: str
    meteorology_distance: float
    calendar_displacement_days: int
    feature_values: dict[str, float]
    normalized_deltas: dict[str, float]
    selection_method: str = "meteorology"


def forest_mask(nmd_label: np.ndarray) -> np.ndarray:
    """Return stable NMD forest classes, excluding clearcuts class 6."""
    return np.isin(nmd_label, tuple(FOREST_CLASSES))


def forest_fraction(nmd_label: np.ndarray) -> float:
    """Forest fraction across the complete tile, including NMD nodata."""
    if not nmd_label.size:
        return 0.0
    return float(forest_mask(nmd_label).mean())


def classified_fraction(nmd_label: np.ndarray) -> float:
    """Fraction of tile pixels carrying a non-background NMD class."""
    if not nmd_label.size:
        return 0.0
    return float(np.mean(nmd_label != 0))


def select_stratified_tiles(
    candidates: Iterable[TileCandidate],
    *,
    count: int = 10,
    min_forest_fraction: float = 0.80,
) -> list[TileCandidate]:
    """Pick one strongest-forest candidate from each northing stratum."""
    eligible = sorted(
        (item for item in candidates if item.forest_fraction >= min_forest_fraction),
        key=lambda item: (item.northing, item.easting),
    )
    if len(eligible) < count:
        raise ValueError(
            f"only {len(eligible)} tiles meet forest fraction "
            f">= {min_forest_fraction:.2f}; need {count}"
        )
    strata = np.array_split(np.array(eligible, dtype=object), count)
    selected = []
    for stratum in strata:
        best = max(
            stratum.tolist(),
            key=lambda item: (item.forest_fraction, -item.easting),
        )
        selected.append(best)
    return selected


def choose_reference_date(
    valid_dates: Iterable[str],
    *,
    target_month_day: str = "06-01",
) -> str:
    """Choose the cloud-valid date nearest the target day."""
    dates = sorted({pd.Timestamp(item).normalize() for item in valid_dates})
    if not dates:
        raise ValueError("no cloud-valid reference dates")
    target = pd.Timestamp(f"{dates[0].year}-{target_month_day}")
    return min(dates, key=lambda item: (abs((item - target).days), item)).strftime(
        "%Y-%m-%d"
    )


def rank_cloud_valid_analogs(
    frames: Mapping[int, pd.DataFrame],
    valid_dates_by_year: Mapping[int, Iterable[str]],
    *,
    reference_year: int = 2019,
    target_month_day: str = "06-01",
    model: AnalogModel | None = None,
) -> list[DateMatch]:
    """Select one meteorological analog from each candidate year."""
    return [
        match
        for match in rank_cloud_valid_comparisons(
            frames,
            valid_dates_by_year,
            reference_year=reference_year,
            target_month_day=target_month_day,
            model=model,
        )
        if match.selection_method == "meteorology"
    ]


def rank_cloud_valid_comparisons(
    frames: Mapping[int, pd.DataFrame],
    valid_dates_by_year: Mapping[int, Iterable[str]],
    *,
    reference_year: int = 2019,
    target_month_day: str = "06-01",
    model: AnalogModel | None = None,
) -> list[DateMatch]:
    """Select calendar and meteorological comparators after one cloud gate.

    Both strategies use the same SCL-screened candidate dates. ``closest_date``
    chooses the acquisition nearest the reference acquisition's month/day;
    ``meteorology`` chooses the candidate with minimum fitted analog distance.
    Keeping the cloud gate common isolates the effect of the selection strategy
    instead of confounding it with cloud contamination.
    """
    normalized_dates = {
        int(year): list(dates) for year, dates in valid_dates_by_year.items()
    }
    if reference_year not in normalized_dates:
        raise ValueError(f"missing valid dates for reference year {reference_year}")
    for year, frame in frames.items():
        wrong_frame_year = [
            item for item in frame["date"] if pd.Timestamp(item).year != int(year)
        ]
        if wrong_frame_year:
            raise ValueError(
                f"meteorology frame for {year} contains another year: "
                f"{wrong_frame_year[0]}"
            )
    for year, dates in normalized_dates.items():
        wrong_year = [item for item in dates if pd.Timestamp(item).year != int(year)]
        if wrong_year:
            raise ValueError(f"valid dates for {year} contain another year: {wrong_year[0]}")
    reference_date = choose_reference_date(
        normalized_dates[reference_year], target_month_day=target_month_day
    )
    reference_month_day = reference_date[5:]
    fitted = (model or AnalogModel()).fit(frames)
    output = []
    for year in sorted(normalized_dates):
        if year == reference_year:
            continue
        allowed = normalized_dates[year]
        matches = fitted.query(
            reference_year=reference_year,
            reference_date=reference_date,
            candidate_years=[year],
            candidate_dates=allowed,
        )
        if not matches:
            continue
        meteorology_best = min(
            matches,
            key=lambda match: (
                match.distance,
                _month_day_displacement(match.date, reference_month_day),
                match.date,
            ),
        )
        closest_date = min(
            allowed,
            key=lambda item: (
                _month_day_displacement(item, reference_month_day),
                item,
            ),
        )
        closest_matches = fitted.query(
            reference_year=reference_year,
            reference_date=reference_date,
            candidate_years=[year],
            candidate_dates=[closest_date],
        )
        if not closest_matches:
            continue
        output.extend((
            _date_match(
                reference_year,
                reference_date,
                closest_matches[0],
                reference_month_day,
                selection_method="closest_date",
            ),
            _date_match(
                reference_year,
                reference_date,
                meteorology_best,
                reference_month_day,
                selection_method="meteorology",
            ),
        ))
    return output


def frame_bands(fetch_result: Mapping[str, np.ndarray], frame_index: int) -> dict[str, np.ndarray]:
    """Extract all seven comparison bands from a canonical fetch result."""
    n_bands = int(fetch_result.get("num_bands", 6))
    spectral = np.asarray(fetch_result["spectral"], dtype=np.float32)
    start = frame_index * n_bands
    frame = spectral[start:start + n_bands]
    if frame.shape[0] != 6:
        raise ValueError(f"frame {frame_index} has {frame.shape[0]} bands, expected 6")
    bands = {name: frame[index] for index, name in enumerate(PRITHVI_BANDS)}
    b08 = np.asarray(fetch_result["b08"], dtype=np.float32)
    if b08.ndim == 3:
        bands["B08"] = b08[frame_index]
    elif b08.ndim == 2 and frame_index == 0:
        bands["B08"] = b08
    else:
        raise ValueError("fetch result b08 has unsupported shape")
    return {name: bands[name] for name in COMPARISON_BANDS}


def validate_fetch_result(
    fetch_result: Mapping[str, np.ndarray],
    requested_dates: Mapping[int, str],
    *,
    expected_bbox: Mapping[str, float],
    expected_center: tuple[int, int],
    expected_size_px: int,
    min_valid_fraction: float = 0.5,
) -> None:
    """Reject incomplete, misaligned, or geometrically wrong S2 output."""
    slots = sorted(requested_dates)
    expected_count = max(slots, default=-1) + 1
    dates = np.asarray(fetch_result.get("dates", []), dtype=str)
    temporal_mask = np.asarray(fetch_result.get("temporal_mask", []))
    valid_fraction = np.asarray(fetch_result.get("frame_valid_frac", []), dtype=float)
    scl = np.asarray(fetch_result.get("scl", []))
    for name, values in (
        ("dates", dates),
        ("temporal_mask", temporal_mask),
        ("frame_valid_frac", valid_fraction),
        ("scl", scl),
    ):
        if len(values) != expected_count:
            raise ValueError(
                f"fetch result {name} has {len(values)} slots, "
                f"expected exactly {expected_count}"
            )
    if dates.ndim != 1 or temporal_mask.ndim != 1 or valid_fraction.ndim != 1:
        raise ValueError("fetch result slot metadata must be one-dimensional")
    if not np.isfinite(valid_fraction).all() or np.any(
        (valid_fraction < 0) | (valid_fraction > 1)
    ):
        raise ValueError("fetch result has invalid frame-valid fractions")
    for slot, requested in requested_dates.items():
        if dates[slot] != requested:
            raise ValueError(
                f"fetch slot {slot} returned {dates[slot]!r}, expected {requested!r}"
            )
        if int(temporal_mask[slot]) != 1:
            raise ValueError(f"fetch slot {slot} is not temporally valid")
        if not np.isfinite(valid_fraction[slot]) or valid_fraction[slot] < min_valid_fraction:
            raise ValueError(
                f"fetch slot {slot} valid fraction {valid_fraction[slot]:.3f} "
                f"is below {min_valid_fraction:.3f}"
            )
    if expected_count > 1 and int(fetch_result.get("coreg_m2", 0)) != 1:
        raise ValueError("multi-frame fetch did not complete M2 coregistration")
    anchor_fraction = float(fetch_result.get("coreg_anchor_valid_frac", np.nan))
    if not np.isfinite(anchor_fraction) or anchor_fraction <= 0:
        raise ValueError("fetch result has no valid coregistration anchor")
    source = fetch_result.get("source")
    if not isinstance(source, str) or not source:
        raise ValueError("fetch result has no source provenance")
    reference_frame = int(fetch_result.get("coreg_ref_frame", -1))
    if not 0 <= reference_frame < expected_count:
        raise ValueError("fetch result has an invalid coregistration reference")
    shifts = np.asarray(fetch_result.get("coreg_shifts", []), dtype=float)
    if shifts.shape != (expected_count, 2) or not np.isfinite(shifts).all():
        raise ValueError("fetch result has invalid coregistration shifts")
    aligned = int(fetch_result.get("coreg_n_aligned", -1))
    measured_aligned = int(sum(
        index != reference_frame and abs(dy) + abs(dx) > 0.0
        for index, (dy, dx) in enumerate(shifts)
    ))
    if aligned != measured_aligned:
        raise ValueError(
            f"fetch result aligned-frame count {aligned} does not match "
            f"shift vectors ({measured_aligned})"
        )
    max_shift = float(fetch_result.get("coreg_max_shift", np.nan))
    if not np.isfinite(max_shift) or max_shift < 0:
        raise ValueError("fetch result has invalid maximum coregistration shift")
    measured_max_shift = float(np.max(np.linalg.norm(shifts, axis=1)))
    if not np.isclose(max_shift, measured_max_shift, rtol=1e-5, atol=1e-6):
        raise ValueError(
            f"fetch result maximum shift {max_shift} does not match "
            f"shift vectors ({measured_max_shift})"
        )

    expected_bbox_values = np.asarray(
        [expected_bbox[key] for key in ("west", "south", "east", "north")],
        dtype=float,
    )
    returned_bbox = np.asarray(fetch_result.get("bbox_3006", []), dtype=float)
    if returned_bbox.shape != (4,) or not np.array_equal(
        returned_bbox, expected_bbox_values
    ):
        raise ValueError(
            f"fetch result bbox {returned_bbox.tolist()} does not match "
            f"requested bbox {expected_bbox_values.tolist()}"
        )
    returned_center = np.asarray(
        [fetch_result.get("easting", np.nan), fetch_result.get("northing", np.nan)],
        dtype=float,
    )
    expected_center_values = np.asarray(expected_center, dtype=float)
    if (
        not np.isfinite(returned_center).all()
        or not np.equal(returned_center, np.rint(returned_center)).all()
        or not np.array_equal(returned_center, expected_center_values)
    ):
        raise ValueError(
            f"fetch result center {returned_center.tolist()} does not match "
            f"requested center {expected_center_values.tolist()}"
        )
    returned_size = int(fetch_result.get("tile_size_px", -1))
    if returned_size != int(expected_size_px):
        raise ValueError(
            f"fetch result tile size {returned_size} does not match "
            f"requested size {expected_size_px}"
        )
    if int(fetch_result.get("num_frames", -1)) != expected_count:
        raise ValueError("fetch result has the wrong frame count")
    if int(fetch_result.get("num_bands", -1)) != len(PRITHVI_BANDS):
        raise ValueError("fetch result has the wrong spectral band count")
    raster_shape = (int(expected_size_px), int(expected_size_px))
    spectral = np.asarray(fetch_result.get("spectral", []))
    b08 = np.asarray(fetch_result.get("b08", []))
    expected_shapes = {
        "spectral": (expected_count * len(PRITHVI_BANDS), *raster_shape),
        "b08": (expected_count, *raster_shape),
        "scl": (expected_count, *raster_shape),
    }
    actual_shapes = {
        "spectral": spectral.shape,
        "b08": b08.shape,
        "scl": scl.shape,
    }
    wrong_shapes = {
        name: {"actual": actual_shapes[name], "expected": shape}
        for name, shape in expected_shapes.items()
        if actual_shapes[name] != shape
    }
    if wrong_shapes:
        raise ValueError(f"fetch result has wrong raster shapes: {wrong_shapes}")


def common_valid_mask(
    forest: np.ndarray,
    reference_scl: np.ndarray,
    candidate_scl: np.ndarray,
    reference_bands: Mapping[str, np.ndarray],
    candidate_bands: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Mask stable forest pixels valid and cloud-free in both scenes."""
    mask = np.asarray(forest, dtype=bool).copy()
    mask &= reference_scl != 0
    mask &= candidate_scl != 0
    mask &= ~np.isin(reference_scl, tuple(SCL_REJECTED_CLASSES))
    mask &= ~np.isin(candidate_scl, tuple(SCL_REJECTED_CLASSES))
    for name in COMPARISON_BANDS:
        mask &= np.isfinite(reference_bands[name]) & (reference_bands[name] > 0)
        mask &= np.isfinite(candidate_bands[name]) & (candidate_bands[name] > 0)
    return mask


def compare_spectral_pair(
    reference_bands: Mapping[str, np.ndarray],
    candidate_bands: Mapping[str, np.ndarray],
    valid_mask: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Compute forest-masked NDVI, per-band, and spectral-angle differences."""
    valid = np.asarray(valid_mask, dtype=bool)
    if not valid.any():
        raise ValueError("comparison has no common valid forest pixels")

    ref_ndvi = _normalized_difference(reference_bands["B08"], reference_bands["B04"])
    cand_ndvi = _normalized_difference(candidate_bands["B08"], candidate_bands["B04"])
    ndvi_difference = cand_ndvi - ref_ndvi

    arrays = {
        "valid_mask": valid,
        "ndvi_reference": np.where(valid, ref_ndvi, np.nan).astype(np.float32),
        "ndvi_candidate": np.where(valid, cand_ndvi, np.nan).astype(np.float32),
        "ndvi_difference": np.where(valid, ndvi_difference, np.nan).astype(np.float32),
    }
    summary = {
        "valid_pixel_fraction": float(valid.mean()),
        "ndvi_reference_median": float(np.median(ref_ndvi[valid])),
        "ndvi_candidate_median": float(np.median(cand_ndvi[valid])),
        "ndvi_difference_median": float(np.median(ndvi_difference[valid])),
        "ndvi_absolute_difference_median": float(np.median(np.abs(ndvi_difference[valid]))),
        "ndvi_difference_iqr": float(_iqr(ndvi_difference[valid])),
    }

    ref_stack = np.stack([reference_bands[name] for name in COMPARISON_BANDS], axis=-1)
    cand_stack = np.stack([candidate_bands[name] for name in COMPARISON_BANDS], axis=-1)
    for index, name in enumerate(COMPARISON_BANDS):
        signed = cand_stack[..., index] - ref_stack[..., index]
        relative = signed / (np.abs(ref_stack[..., index]) + 1e-6)
        arrays[f"{name.lower()}_difference"] = np.where(valid, signed, np.nan).astype(np.float32)
        arrays[f"{name.lower()}_relative_difference"] = np.where(valid, relative, np.nan).astype(np.float32)
        summary[f"{name.lower()}_difference_median"] = float(np.median(signed[valid]))
        summary[f"{name.lower()}_absolute_difference_median"] = float(
            np.median(np.abs(signed[valid]))
        )

    dot = np.sum(ref_stack * cand_stack, axis=-1)
    norms = np.linalg.norm(ref_stack, axis=-1) * np.linalg.norm(cand_stack, axis=-1)
    cosine = np.clip(dot / np.maximum(norms, 1e-12), -1.0, 1.0)
    angle = np.arccos(cosine)
    arrays["spectral_angle_rad"] = np.where(valid, angle, np.nan).astype(np.float32)
    summary["spectral_angle_median_rad"] = float(np.median(angle[valid]))
    summary["spectral_angle_iqr_rad"] = float(_iqr(angle[valid]))
    return arrays, summary


def summarize_vpp_phase(
    vpp: Mapping[str, np.ndarray],
    acquisition_date: str,
    *,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Summarize forest VPP timing; peak date is an explicit midpoint proxy."""
    acquisition = pd.Timestamp(acquisition_date)
    acquisition_doy = _common_year_doy(acquisition.month, acquisition.day)
    _validate_yyddd_product_year(
        vpp["sosd"],
        acquisition.year,
        "SOSD",
        allowed_years=(acquisition.year - 1, acquisition.year),
    )
    _validate_yyddd_product_year(
        vpp["eosd"],
        acquisition.year,
        "EOSD",
        allowed_years=(acquisition.year, acquisition.year + 1),
    )
    sos_source = _positive_median(vpp["sosd"], mask=mask)
    eos_source = _positive_median(vpp["eosd"], mask=mask)
    sos_doy = _finite_median(_yyddd_relative_doy(vpp["sosd"], acquisition.year), mask=mask)
    eos_doy = _finite_median(_yyddd_relative_doy(vpp["eosd"], acquisition.year), mask=mask)
    sos = _normalize_relative_doy(sos_doy, acquisition.year)
    eos = _normalize_relative_doy(eos_doy, acquisition.year)
    midpoint = (sos + eos) / 2.0
    return {
        "sos_doy_median": sos,
        "eos_doy_median": eos,
        "sos_source_yyddd_median": sos_source,
        "eos_source_yyddd_median": eos_source,
        "season_midpoint_proxy_doy": midpoint,
        "acquisition_doy": float(acquisition_doy),
        "days_from_sos": float(acquisition_doy - sos),
        "days_from_midpoint_proxy": float(acquisition_doy - midpoint),
        "days_to_eos": float(eos - acquisition_doy),
        "maxv_median": _positive_median(vpp["maxv"], mask=mask),
        "minv_median": _positive_median(vpp["minv"], mask=mask),
    }


def _days_in_year(year: int) -> int:
    return int(pd.Timestamp(int(year), 12, 31).dayofyear)


def _validate_yyddd_product_year(
    values: np.ndarray,
    product_year: int,
    label: str,
    *,
    allowed_years: tuple[int, ...] | None = None,
) -> None:
    """Require integral YYDDD values within the allowed season years.

    HR-VPP season events are not confined to the product year: season
    starts can fall in the previous autumn and season ends can spill
    into the next winter, and the YYDDD prefix names the year the event
    actually occurred in.
    """
    if allowed_years is None:
        allowed_years = (int(product_year),)
    array = np.asarray(values, dtype=float)
    valid = array[np.isfinite(array) & (array > 0)]
    if not valid.size:
        raise ValueError(f"VPP {label} has no positive YYDDD values")
    rounded = np.rint(valid)
    if not np.array_equal(valid, rounded):
        raise ValueError(f"VPP {label} contains non-integral YYDDD values")
    encoded = rounded.astype(np.int64)
    prefixes = encoded // 1000
    allowed_prefixes = sorted(int(year) - 2000 for year in allowed_years)
    if not np.all(np.isin(prefixes, allowed_prefixes)):
        observed = sorted(set(prefixes.tolist()))
        raise ValueError(
            f"VPP {label} YYDDD prefix {observed} is outside the allowed "
            f"season years {sorted(int(year) for year in allowed_years)} "
            f"for product year {product_year}"
        )
    doy = encoded % 1000
    for year in sorted(int(year) for year in allowed_years):
        in_year = doy[prefixes == year - 2000]
        if in_year.size and not np.all((in_year >= 1) & (in_year <= _days_in_year(year))):
            raise ValueError(f"VPP {label} contains DOY outside year {year}")


def _yyddd_relative_doy(values: np.ndarray, product_year: int) -> np.ndarray:
    """Decode YYDDD onto the product-year DOY axis.

    Events in the previous year map below 1 and events in the next year
    map above the year length, so day arithmetic against product-year
    dates stays correct across the season's calendar-year boundary.
    """
    array = np.asarray(values, dtype=float)
    result = np.full(array.shape, np.nan)
    valid = np.isfinite(array) & (array > 0)
    encoded = np.rint(array[valid]).astype(np.int64)
    years = 2000 + encoded // 1000
    doy = (encoded % 1000).astype(float)
    doy[years == int(product_year) - 1] -= _days_in_year(int(product_year) - 1)
    doy[years == int(product_year) + 1] += _days_in_year(int(product_year))
    result[valid] = doy
    return result


def _finite_median(values: np.ndarray, *, mask: np.ndarray | None = None) -> float:
    array = np.asarray(values, dtype=float)
    valid_mask = np.isfinite(array)
    if mask is not None:
        if np.asarray(mask).shape != array.shape:
            raise ValueError("VPP mask shape does not match VPP raster")
        valid_mask &= np.asarray(mask, dtype=bool)
    valid = array[valid_mask]
    if not valid.size:
        raise ValueError("VPP band has no valid pixels")
    return float(np.median(valid))


def _normalize_relative_doy(day_of_year: float, year: int) -> float:
    """Normalize a product-year-axis DOY that may extend into adjacent years."""
    value = float(day_of_year)
    if value < 1:
        previous = int(year) - 1
        return normalize_doy_for_year(value + _days_in_year(previous), previous) - 365
    days = _days_in_year(int(year))
    if value > days:
        return normalize_doy_for_year(value - days, int(year) + 1) + 365
    return normalize_doy_for_year(value, year)


def normalize_doy_for_year(day_of_year: float, year: int) -> float:
    """Map a source-year DOY to a common non-leap-year month/day axis."""
    days_in_year = 366 if pd.Timestamp(year=year, month=12, day=31).dayofyear == 366 else 365
    value = float(day_of_year)
    if not np.isfinite(value) or not 1 <= value <= days_in_year:
        raise ValueError(f"VPP DOY {value} is outside year {year}")
    lower = int(np.floor(value))
    upper = int(np.ceil(value))

    def map_day(day):
        timestamp = pd.Timestamp(year=int(year), month=1, day=1) + pd.Timedelta(days=day - 1)
        if timestamp.month == 2 and timestamp.day == 29:
            return 59.5
        return float(_common_year_doy(timestamp.month, timestamp.day))

    if lower == upper:
        return map_day(lower)
    fraction = value - lower
    return map_day(lower) * (1 - fraction) + map_day(upper) * fraction


def _common_year_doy(month: int, day: int) -> int:
    return int(pd.Timestamp(year=2001, month=month, day=day).dayofyear)


def write_manifest(path: str | Path, payload: Mapping) -> None:
    """Atomically persist a run manifest for resumability."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    temporary.replace(destination)


def _date_match(
    reference_year: int,
    reference_date: str,
    match: AnalogMatch,
    target: str,
    *,
    selection_method: str,
) -> DateMatch:
    return DateMatch(
        reference_year=reference_year,
        reference_date=reference_date,
        candidate_year=match.year,
        candidate_date=match.date,
        meteorology_distance=match.distance,
        calendar_displacement_days=_month_day_displacement(match.date, target),
        feature_values=match.feature_values,
        normalized_deltas=match.normalized_deltas,
        selection_method=selection_method,
    )


def _month_day_displacement(date_string: str, month_day: str) -> int:
    timestamp = pd.Timestamp(date_string)
    target = pd.Timestamp(f"{timestamp.year}-{month_day}")
    return abs((timestamp - target).days)


def _normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b) / (a + b + 1e-10)


def _iqr(values: np.ndarray) -> float:
    q25, q75 = np.percentile(values, [25, 75])
    return float(q75 - q25)


def _positive_median(values: np.ndarray, *, mask: np.ndarray | None = None) -> float:
    array = np.asarray(values, dtype=float)
    valid_mask = np.isfinite(array) & (array > 0)
    if mask is not None:
        if np.asarray(mask).shape != array.shape:
            raise ValueError("VPP mask shape does not match VPP raster")
        valid_mask &= np.asarray(mask, dtype=bool)
    valid = array[valid_mask]
    if not valid.size:
        raise ValueError("VPP band has no positive valid pixels")
    return float(np.median(valid))


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")
