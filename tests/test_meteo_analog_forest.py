from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from imint.experiments.meteo_analog_forest import (
    TileCandidate,
    choose_reference_date,
    common_valid_mask,
    compare_spectral_pair,
    forest_fraction,
    forest_mask,
    rank_cloud_valid_analogs,
    rank_cloud_valid_comparisons,
    select_stratified_tiles,
    summarize_vpp_phase,
    validate_fetch_result,
    write_manifest,
)


FEATURES = (
    "gdd_prev30d_c",
    "precip_prev30d_mm",
    "precip_prev7d_mm",
    "swvl1_prev30d_mean",
    "ssrd_prev30d_mj_m2",
)


def _meteo_frame(year, rows):
    data = {"date": [f"{year}-{month_day}" for month_day, _ in rows]}
    for index, feature in enumerate(FEATURES):
        data[feature] = [values[index] for _, values in rows]
    return pd.DataFrame(data)


def test_forest_fraction_excludes_background_and_clearcut():
    labels = np.array([[0, 1, 2], [5, 6, 10]], dtype=np.uint8)
    assert forest_mask(labels).sum() == 3
    assert forest_fraction(labels) == pytest.approx(3 / 6)


def test_selects_one_highest_forest_tile_per_northing_stratum():
    candidates = [
        TileCandidate(100 + index, 1000 + index * 100, 0.81 + (index % 2) * 0.1)
        for index in range(20)
    ]
    selected = select_stratified_tiles(candidates, count=10)
    assert len(selected) == 10
    assert all(tile.forest_fraction >= 0.8 for tile in selected)
    assert len({tile.northing for tile in selected}) == 10


def test_reference_date_uses_closest_cloud_valid_pass():
    assert choose_reference_date(["2019-05-20", "2019-06-03", "2019-06-10"]) == "2019-06-03"
    with pytest.raises(ValueError, match="no cloud-valid"):
        choose_reference_date([])


def test_rank_analogs_only_uses_cloud_valid_candidates():
    exact = [300.0, 40.0, 5.0, 0.3, 500.0]
    frames = {
        2019: _meteo_frame(2019, [("06-01", exact)]),
        2020: _meteo_frame(2020, [("05-20", exact), ("06-05", [500, 100, 30, 0.5, 700])]),
    }
    result = rank_cloud_valid_analogs(
        frames,
        {2019: ["2019-06-01"], 2020: ["2020-06-05"]},
    )
    assert result[0].candidate_date == "2020-06-05"
    assert result[0].meteorology_distance > 0


def test_comparisons_include_cloud_valid_calendar_and_meteorology_choices():
    exact = [300.0, 40.0, 5.0, 0.3, 500.0]
    frames = {
        2019: _meteo_frame(2019, [("06-02", exact)]),
        2020: _meteo_frame(2020, [
            ("06-03", [600.0, 120.0, 30.0, 0.6, 800.0]),
            ("05-20", exact),
        ]),
    }
    result = rank_cloud_valid_comparisons(
        frames,
        {
            2019: ["2019-06-02"],
            2020: ["2020-06-03", "2020-05-20"],
        },
    )
    by_method = {match.selection_method: match for match in result}
    assert by_method["closest_date"].candidate_date == "2020-06-03"
    assert by_method["meteorology"].candidate_date == "2020-05-20"
    assert by_method["meteorology"].meteorology_distance < by_method["closest_date"].meteorology_distance


def test_rank_analogs_rejects_dates_from_wrong_year():
    exact = [300.0, 40.0, 5.0, 0.3, 500.0]
    frames = {
        2019: _meteo_frame(2019, [("06-01", exact)]),
        2020: _meteo_frame(2020, [("06-01", exact)]),
    }
    with pytest.raises(ValueError, match="another year"):
        rank_cloud_valid_analogs(
            frames, {2019: ["2019-06-01"], 2020: ["2019-06-01"]}
        )


def test_common_mask_and_spectral_metrics_exclude_clouds():
    shape = (3, 3)
    forest = np.ones(shape, dtype=bool)
    ref_scl = np.full(shape, 4, dtype=np.uint8)
    cand_scl = ref_scl.copy()
    cand_scl[0, 0] = 3
    cand_scl[0, 1] = 11
    reference = {name: np.full(shape, 0.2, np.float32) for name in ("B02", "B03", "B04", "B08", "B8A", "B11", "B12")}
    candidate = {name: np.full(shape, 0.3, np.float32) for name in reference}
    candidate["B08"][:] = 0.5
    mask = common_valid_mask(forest, ref_scl, cand_scl, reference, candidate)
    assert mask.sum() == 7
    arrays, summary = compare_spectral_pair(reference, candidate, mask)
    assert np.isnan(arrays["ndvi_difference"][0, 0])
    assert np.isnan(arrays["ndvi_difference"][0, 1])
    assert summary["valid_pixel_fraction"] == pytest.approx(7 / 9)
    assert summary["ndvi_candidate_median"] > summary["ndvi_reference_median"]
    assert np.isfinite(summary["spectral_angle_median_rad"])


def test_common_mask_rejects_saturated_and_unclassified_scl():
    shape = (2, 2)
    forest = np.ones(shape, dtype=bool)
    reference_scl = np.full(shape, 4, dtype=np.uint8)
    candidate_scl = reference_scl.copy()
    candidate_scl[0, 0] = 1
    candidate_scl[0, 1] = 7
    bands = {
        name: np.ones(shape, np.float32)
        for name in ("B02", "B03", "B04", "B08", "B8A", "B11", "B12")
    }
    mask = common_valid_mask(
        forest, reference_scl, candidate_scl, bands, bands
    )
    assert mask.tolist() == [[False, False], [True, True]]


def test_empty_common_mask_fails_loudly():
    bands = {name: np.ones((2, 2), np.float32) for name in ("B02", "B03", "B04", "B08", "B8A", "B11", "B12")}
    with pytest.raises(ValueError, match="no common valid"):
        compare_spectral_pair(bands, bands, np.zeros((2, 2), bool))


def test_vpp_phase_labels_midpoint_as_proxy():
    vpp = {
        "sosd": np.full((2, 2), 20120),
        "eosd": np.full((2, 2), 20280),
        "maxv": np.full((2, 2), 1.2),
        "minv": np.full((2, 2), 0.2),
    }
    result = summarize_vpp_phase(vpp, "2020-06-01")
    assert result["season_midpoint_proxy_doy"] == 199
    assert result["days_from_sos"] == 33


def test_vpp_phase_normalizes_leap_year_doy():
    vpp = {
        "sosd": np.full((2, 2), 20153),  # 1 June in leap year 2020
        "eosd": np.full((2, 2), 20280),
        "maxv": np.full((2, 2), 1.2),
        "minv": np.full((2, 2), 0.2),
    }
    result = summarize_vpp_phase(vpp, "2020-06-01")
    assert result["sos_doy_median"] == 152
    assert result["days_from_sos"] == 0
    vpp["sosd"][:] = 20367
    with pytest.raises(ValueError, match="outside year"):
        summarize_vpp_phase(vpp, "2020-06-01")
    with pytest.raises(ValueError, match="non-integral"):
        summarize_vpp_phase(
            {**vpp, "sosd": np.full((2, 2), 20152.5)}, "2020-06-01"
        )


def _valid_fetch_result():
    return {
        "dates": np.array(["2019-06-01", "2020-06-02"]),
        "temporal_mask": np.array([1, 1]),
        "frame_valid_frac": np.array([0.9, 0.8]),
        "scl": np.ones((2, 4, 4), np.uint8),
        "spectral": np.ones((12, 4, 4), np.float32),
        "b08": np.ones((2, 4, 4), np.float32),
        "bbox_3006": np.array([0, 0, 40, 40]),
        "easting": np.int32(20),
        "northing": np.int32(20),
        "tile_size_px": np.int32(4),
        "num_frames": np.int32(2),
        "num_bands": np.int32(6),
        "coreg_m2": np.int32(1),
        "coreg_anchor_valid_frac": np.float32(0.9),
        "source": "des",
        "coreg_ref_frame": np.int32(0),
        "coreg_n_aligned": np.int32(1),
        "coreg_max_shift": np.float32(np.sqrt(0.05)),
        "coreg_shifts": np.array([[0.0, 0.0], [0.2, -0.1]], np.float32),
    }


def test_fetch_result_requires_exact_valid_coregistered_slots():
    dates = {0: "2019-06-01", 1: "2020-06-02"}
    expected = {
        "expected_bbox": {"west": 0, "south": 0, "east": 40, "north": 40},
        "expected_center": (20, 20),
        "expected_size_px": 4,
    }
    validate_fetch_result(_valid_fetch_result(), dates, **expected)
    for key, replacement, message in (
        ("dates", np.array(["2019-06-01", "2020-06-03"]), "returned"),
        ("temporal_mask", np.array([1, 0]), "temporally valid"),
        ("frame_valid_frac", np.array([0.9, 0.4]), "below"),
        ("coreg_m2", np.int32(0), "M2"),
    ):
        result = _valid_fetch_result()
        result[key] = replacement
        with pytest.raises(ValueError, match=message):
            validate_fetch_result(result, dates, **expected)

    for key, replacement, message in (
        ("bbox_3006", np.array([0, 0, 30, 40]), "bbox"),
        ("easting", np.int32(30), "center"),
        ("tile_size_px", np.int32(8), "tile size"),
        ("num_frames", np.int32(3), "frame count"),
        ("spectral", np.ones((6, 4, 4), np.float32), "raster shapes"),
        ("easting", np.float32(20.5), "center"),
        ("coreg_n_aligned", np.int32(0), "aligned-frame count"),
        ("coreg_max_shift", np.float32(0.4), "maximum shift"),
        ("frame_valid_frac", np.array([0.9, 1.1]), "fractions"),
        (
            "dates",
            np.array(["2019-06-01", "2020-06-02", "2021-06-03"]),
            "exactly",
        ),
    ):
        result = _valid_fetch_result()
        result[key] = replacement
        with pytest.raises(ValueError, match=message):
            validate_fetch_result(result, dates, **expected)


def test_manifest_write_is_readable_and_replaces_existing(tmp_path):
    path = tmp_path / "manifest.json"
    write_manifest(path, {"stage": "selected", "count": np.int32(10)})
    write_manifest(path, {"stage": "compared", "count": np.int32(10)})
    assert json.loads(path.read_text()) == {"count": 10, "stage": "compared"}
    assert not path.with_suffix(".json.tmp").exists()
