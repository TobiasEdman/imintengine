from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import socket

import numpy as np
import pandas as pd

from scripts.run_meteo_analog_forest_poc import (
    _safe_run_dir,
    _group_dates,
    _load_selected_tiles,
    _pair_succeeded,
    _validate_years,
    _validate_live_environment,
    fetch_and_compare_tile,
    load_inventory,
    plan_tile_dates,
    select_forest_tiles,
)
import scripts.run_meteo_analog_forest_poc as poc_script
from imint.experiments.meteo_analog_forest import DateMatch, TileCandidate
from imint.training import optimal_fetch as optimal_fetch_module
from imint.training import tile_fetch as tile_fetch_module
from imint.training.tile_config import TileConfig


def test_load_inventory_deduplicates_npz_centers(tmp_path):
    np.savez(tmp_path / "a.npz", easting=500000, northing=6500000)
    np.savez(tmp_path / "b.npz", easting=500000, northing=6500000)
    np.savez(tmp_path / "ignored.npz", value=1)
    rows = load_inventory(tmp_path)
    assert len(rows) == 1
    assert rows[0][:2] == (500000, 6500000)


def test_select_forest_tiles_uses_local_labels(monkeypatch, tmp_path):
    raster = tmp_path / "nmd.tif"
    raster.touch()
    centers = [(1000 + i * 10, 6000000 + i * 1000, str(i)) for i in range(20)]
    monkeypatch.setattr(
        "scripts.run_meteo_analog_forest_poc.fetch_nmd_label_local",
        lambda *args, **kwargs: np.ones((8, 8), np.uint8),
    )
    selected = select_forest_tiles(
        centers,
        tile=TileConfig(size_px=8),
        nmd_raster=raster,
        count=10,
        min_fraction=0.8,
        candidates_per_stratum=2,
    )
    assert len(selected) == 10
    assert all(item.forest_fraction == 1.0 for item in selected)


def test_select_forest_tiles_fails_when_original_stratum_has_no_forest(monkeypatch, tmp_path):
    raster = tmp_path / "nmd.tif"
    raster.touch()
    centers = [(1000 + i * 10, 6000000 + i * 1000, str(i)) for i in range(20)]
    calls = iter([np.zeros((8, 8), np.uint8)] * 2 + [np.ones((8, 8), np.uint8)] * 18)
    monkeypatch.setattr(
        "scripts.run_meteo_analog_forest_poc.fetch_nmd_label_local",
        lambda *args, **kwargs: next(calls),
    )
    with np.testing.assert_raises(ValueError):
        select_forest_tiles(
            centers, tile=TileConfig(size_px=8), nmd_raster=raster,
            count=10, min_fraction=0.8, candidates_per_stratum=2,
        )


def test_strict_local_nmd_never_falls_back_to_remote(monkeypatch, tmp_path):
    monkeypatch.setattr(
        tile_fetch_module,
        "_fetch_nmd_label_remote",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("remote fallback called")
        ),
    )
    tile = TileConfig(size_px=8)
    bbox = tile.bbox_from_center(500000, 6500000)
    with np.testing.assert_raises(FileNotFoundError):
        tile_fetch_module.fetch_nmd_label_local(
            bbox,
            tile,
            str(tmp_path / "missing.tif"),
            allow_remote_fallback=False,
        )


def test_plan_tile_dates_combines_cloud_gate_and_metafilter(monkeypatch, tmp_path):
    features = {
        "gdd_prev30d_c": [300.0, 350.0],
        "precip_prev30d_mm": [40.0, 80.0],
        "precip_prev7d_mm": [5.0, 12.0],
        "swvl1_prev30d_mean": [0.3, 0.4],
        "ssrd_prev30d_mj_m2": [500.0, 600.0],
    }

    def fake_meteo(**kwargs):
        year = int(kwargs["date_start"][:4])
        frame = pd.DataFrame({"date": [f"{year}-06-01", f"{year}-06-05"], **features})
        return SimpleNamespace(frame=frame)

    plan_calls = []

    def fake_plan(bbox, start, end, **kwargs):
        plan_calls.append(kwargs)
        year = int(start[:4])
        return SimpleNamespace(
            mode=kwargs["mode"],
            dates=[f"{year}-06-01"],
            n_candidates_after={"final": 1}, elapsed_s={}, notes={},
            scl_gate={
                f"{year}-06-01": {
                    "cloud_fraction": 0.01,
                    "snow_fraction": 0.0,
                    "coverage_fraction": 1.0,
                    "accepted": True,
                }
            },
            scl_screen_complete=True,
            scl_thresholds={
                "max_aoi_cloud": 0.1,
                "max_aoi_snow": 0.01,
                "min_aoi_coverage": 0.8,
            },
        )

    monkeypatch.setattr(
        "scripts.run_meteo_analog_forest_poc.fetch_daily_meteorology", fake_meteo
    )
    monkeypatch.setattr(
        "scripts.run_meteo_analog_forest_poc.optimal_fetch_dates", fake_plan
    )
    args = SimpleNamespace(
        size_px=512,
        reference_year=2019,
        candidate_years=[2020],
        window=["05-15", "06-15"],
        max_aoi_cloud=0.1,
        fetch_source="des",
    )
    result = plan_tile_dates(TileCandidate(500000, 6500000, 0.9), args, tmp_path)
    assert {match.selection_method for match in result["matches"]} == {
        "closest_date", "meteorology",
    }
    assert all(match.candidate_date == "2020-06-01" for match in result["matches"])
    assert result["fetch_plans"][2020]["n_candidates_after"] == {"final": 1}
    assert result["fetch_plans"][2020]["mode"] == "scl_only"
    assert result["fetch_plans"][2020]["scl_gate"]["2020-06-01"]["accepted"]
    assert all(call["mode"] == "scl_only" for call in plan_calls)
    assert result["plan_fingerprint"]
    assert len(result["meteorology_by_year"]["2020"]) == 2


def test_plan_tile_dates_rejects_accepted_date_without_complete_scl_evidence(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "scripts.run_meteo_analog_forest_poc.fetch_daily_meteorology",
        lambda **kwargs: SimpleNamespace(
            frame=pd.DataFrame({
                "date": [f"{kwargs['date_start'][:4]}-06-01"],
                "gdd_prev30d_c": [300.0],
                "precip_prev30d_mm": [40.0],
                "precip_prev7d_mm": [5.0],
                "swvl1_prev30d_mean": [0.3],
                "ssrd_prev30d_mj_m2": [500.0],
            })
        ),
    )

    def incomplete_plan(_bbox, start, _end, **_kwargs):
        observed = f"{start[:4]}-06-01"
        return SimpleNamespace(
            mode="scl_only",
            dates=[observed],
            n_candidates_after={"final": 1},
            elapsed_s={},
            notes={},
            scl_gate={
                observed: {"cloud_fraction": 0.01, "accepted": True}
            },
            scl_screen_complete=True,
            scl_thresholds={
                "max_aoi_cloud": 0.1,
                "max_aoi_snow": 0.01,
                "min_aoi_coverage": 0.8,
            },
        )

    monkeypatch.setattr(
        "scripts.run_meteo_analog_forest_poc.optimal_fetch_dates",
        incomplete_plan,
    )
    args = SimpleNamespace(
        size_px=512,
        reference_year=2019,
        candidate_years=[2020],
        window=["05-15", "06-15"],
        max_aoi_cloud=0.1,
        fetch_source="des",
    )
    with np.testing.assert_raises_regex(ValueError, "malformed gate"):
        plan_tile_dates(TileCandidate(500000, 6500000, 0.9), args, tmp_path)


def test_scl_only_plan_records_complete_gate_table(monkeypatch):
    monkeypatch.setattr(
        optimal_fetch_module,
        "scl_stack_screen",
        lambda *args, **kwargs: {
            "2020-06-01": (0.02, 0.0, 1.0),
            "2020-06-03": (0.25, 0.0, 1.0),
            "2020-06-05": (0.01, 0.2, 1.0),
        },
    )
    plan = optimal_fetch_module.optimal_fetch_dates(
        {"west": 10.0, "south": 60.0, "east": 10.1, "north": 60.1},
        "2020-05-15",
        "2020-06-15",
        mode="scl_only",
        max_aoi_cloud=0.1,
    )
    assert plan.dates == ["2020-06-01"]
    assert set(plan.scl_gate) == {"2020-06-01", "2020-06-03", "2020-06-05"}
    assert plan.scl_gate["2020-06-01"]["accepted"]
    assert not plan.scl_gate["2020-06-03"]["accepted"]
    assert not plan.scl_gate["2020-06-05"]["accepted"]
    assert plan.scl_screen_complete is True
    assert plan.scl_thresholds["max_aoi_cloud"] == 0.1


def test_strict_scl_screen_rejects_failed_chunks(monkeypatch):
    optimal_fetch_module._SCL_SCREEN_MEMO.clear()
    monkeypatch.delenv("SCL_FRACS_CACHE", raising=False)
    monkeypatch.setattr(
        optimal_fetch_module,
        "_scl_chunk",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic throttle")
        ),
    )
    with np.testing.assert_raises_regex(RuntimeError, "SCL screen incomplete"):
        optimal_fetch_module.scl_stack_screen(
            {"west": 10.0, "south": 60.0, "east": 10.1, "north": 60.1},
            "2020-05-15",
            "2020-05-16",
            conn=object(),
            require_complete=True,
        )


def test_fetch_compare_writes_arrays_and_figure(monkeypatch, tmp_path):
    size = 8
    n_frames = 3
    spectral = np.zeros((n_frames * 6, size, size), np.float32)
    spectral[:6] = 0.2
    spectral[6:12] = 0.3
    spectral[12:] = 0.4
    scl = np.full((n_frames, size, size), 4, np.uint8)
    scl[1, 0, 0] = 3
    scl[2, 0, 1] = 11
    fetch_result = {
        "spectral": spectral,
        "b08": np.stack([
            np.full((size, size), 0.4, np.float32),
            np.full((size, size), 0.5, np.float32),
            np.full((size, size), 0.6, np.float32),
        ]),
        "scl": scl,
        "num_bands": np.int32(6),
        "coreg_max_shift": np.float32(np.sqrt(0.05)),
        "dates": np.array(["2019-06-01", "2020-06-02", "2020-05-20"]),
        "temporal_mask": np.array([1, 1, 1]),
        "frame_valid_frac": np.array([0.9, 0.9, 0.9]),
        "coreg_m2": np.int32(1),
        "coreg_anchor_valid_frac": np.float32(0.9),
        "source": "des",
        "coreg_ref_frame": np.int32(0),
        "coreg_n_aligned": np.int32(2),
        "coreg_shifts": np.array(
            [[0.0, 0.0], [0.2, -0.1], [-0.1, 0.1]], np.float32
        ),
        "bbox_3006": np.asarray([
            TileConfig(size_px=size).bbox_from_center(500000, 6500000)[key]
            for key in ("west", "south", "east", "north")
        ], dtype=np.int32),
        "easting": np.int32(500000),
        "northing": np.int32(6500000),
        "tile_size_px": np.int32(size),
        "num_frames": np.int32(n_frames),
    }
    fetch_calls = []

    def fake_fetch(*args, **kwargs):
        fetch_calls.append(kwargs)
        return fetch_result

    monkeypatch.setattr(
        "scripts.run_meteo_analog_forest_poc.fetch_tile_spectral",
        fake_fetch,
    )
    monkeypatch.setattr(
        "scripts.run_meteo_analog_forest_poc.fetch_nmd_label_local",
        lambda *args, **kwargs: np.ones((size, size), np.uint8),
    )
    vpp = {
        "maxv": np.full((size, size), 1.2, np.float32),
        "minv": np.full((size, size), 0.2, np.float32),
        "length": np.full((size, size), 160, np.float32),
    }
    def fake_vpp(bbox, year, *args, **kwargs):
        values = {
            **vpp,
            "sosd": np.full((size, size), (year - 2000) * 1000 + 120 + (year == 2020)),
            "eosd": np.full((size, size), (year - 2000) * 1000 + 280 + (year == 2020)),
        }
        provenance = {
            "source": "wekeo",
            "product_year": year,
            "season": poc_script.VPP_SEASON,
            "index_path": "/synthetic/index.json",
            "index_sha256": "a" * 64,
            "products": [{
                "filename": f"synthetic-{year}.tif",
                "sha256": "e" * 64,
            }],
            "versions": [105],
        }
        return values, poc_script._bind_vpp_raster_provenance(
            provenance, values
        )

    monkeypatch.setattr(
        "scripts.run_meteo_analog_forest_poc._fetch_vpp", fake_vpp
    )
    closest = DateMatch(
        reference_date="2019-06-01",
        candidate_date="2020-06-02",
        candidate_year=2020,
        meteorology_distance=0.5,
        reference_year=2019,
        calendar_displacement_days=1,
        feature_values={},
        normalized_deltas={},
        selection_method="closest_date",
    )
    meteorology = DateMatch(
        reference_date="2019-06-01",
        candidate_date="2020-05-20",
        candidate_year=2020,
        meteorology_distance=0.1,
        reference_year=2019,
        calendar_displacement_days=12,
        feature_values={},
        normalized_deltas={},
        selection_method="meteorology",
    )
    args = SimpleNamespace(
        size_px=size,
        fetch_source="des",
        nmd_raster=tmp_path / "nmd.tif",
        reference_year=2019,
    )
    context = {
        "reference_date": "2019-06-01",
        "matches": [closest, meteorology],
        "bbox_3006": TileConfig(size_px=size).bbox_from_center(500000, 6500000),
        "plan_fingerprint": "a" * 64,
    }
    summaries, figures = fetch_and_compare_tile(
        "tile_00",
        TileCandidate(500000, 6500000, 1.0),
        context,
        args,
        tmp_path,
        run_fingerprint="b" * 64,
    )
    assert len(fetch_calls) == 1
    assert fetch_calls[0]["dates"] == {
        0: "2019-06-01", 1: "2020-06-02", 2: "2020-05-20"
    }
    assert len(summaries) == 2
    assert all(row["status"] == "ok" for row in summaries)
    assert {row["fetch_group_id"] for row in summaries} == {summaries[0]["fetch_group_id"]}
    assert {row["common_valid_mask_sha256"] for row in summaries} == {
        summaries[0]["common_valid_mask_sha256"]
    }
    artifact = tmp_path / summaries[0]["array_path"]
    assert artifact.exists()
    assert summaries[0]["array_bytes"] == artifact.stat().st_size
    assert summaries[0]["array_sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    with np.load(artifact, allow_pickle=False) as data:
        assert data["reference_bands"].shape == (7, size, size)
        assert data["candidate_bands"].shape == (7, size, size)
        assert data["reference_vpp"].shape == (5, size, size)
        assert data["candidate_vpp"].shape == (5, size, size)
        assert data["nmd_label"].shape == (size, size)
        assert data["valid_mask"].sum() == size * size - 2
        assert str(data["selection_method"]) == "closest_date"
        assert str(data["grid_crs"]) == poc_script.GRID_CRS
        bbox = context["bbox_3006"]
        np.testing.assert_array_equal(
            data["grid_transform"],
            [10.0, 0.0, bbox["west"], 0.0, -10.0, bbox["north"]],
        )
    with np.load(tmp_path / summaries[1]["array_path"], allow_pickle=False) as data:
        assert data["valid_mask"].sum() == size * size - 2
        assert str(data["selection_method"]) == "meteorology"
    assert all(figure.exists() for figure in figures)


def test_group_dates_deduplicates_equal_strategy_dates():
    matches = [
        DateMatch(2019, "2019-06-01", 2020, "2020-06-02", 0.2, 1, {}, {}, method)
        for method in poc_script.SELECTION_METHODS
    ]
    dates, slots = _group_dates("2019-06-01", matches)
    assert dates == {0: "2019-06-01", 1: "2020-06-02"}
    assert slots["2020-06-02"] == 1


def test_safe_run_id_rejects_path_traversal(tmp_path):
    assert _safe_run_dir(tmp_path, "run-01").parent == tmp_path.resolve()
    for run_id in ("../escape", "/absolute", ".."):
        with np.testing.assert_raises(ValueError):
            _safe_run_dir(tmp_path, run_id)


def test_resume_retries_failed_pairs_and_preserves_tile_coverage(tmp_path):
    pairs = tmp_path / "pairs"
    pairs.mkdir()
    arrays = tmp_path / "arrays"
    arrays.mkdir()
    failed = pairs / "failed.json"
    failed.write_text('{"status":"failed"}')
    size = 2
    identity = {
        "tile_id": "tile_00",
        "dataset_schema_version": poc_script.DATASET_SCHEMA_VERSION,
        "plan_fingerprint": "a" * 64,
        "run_fingerprint": "b" * 64,
        "fetch_group_id": "c" * 24,
        "fetch_source": "des",
        "fetch_requested_dates": ["2019-06-01", "2020-06-02"],
        "candidate_slot": 1,
        "tile_size_px": size,
        "easting": 500000,
        "northing": 6500000,
        "forest_fraction": 1.0,
        "classified_fraction": 1.0,
        "bbox_epsg3006": [499990.0, 6499990.0, 500010.0, 6500010.0],
        "band_names": list(poc_script.COMPARISON_BANDS),
        "vpp_band_names": list(poc_script.VPP_BANDS),
        "reference_year": 2019,
        "reference_date": "2019-06-01",
        "candidate_year": 2020,
        "candidate_date": "2020-06-02",
        "selection_method": "meteorology",
        "grid_crs": poc_script.GRID_CRS,
        "grid_transform": [
            10.0, 0.0, 499990.0, 0.0, -10.0, 6500010.0,
        ],
    }
    provenance = {
        "fetch_attempt_sha256": "f" * 64,
        "fetch_source": "des",
        "fetch_returned_dates": identity["fetch_requested_dates"],
        "fetch_temporal_mask": [1, 1],
        "fetch_frame_valid_fraction": [0.9, 0.8],
        "fetch_coreg_shifts": [[0.0, 0.0], [0.2, -0.1]],
        "fetch_coreg_ref_frame": 0,
        "fetch_coreg_m2": 1,
        "fetch_coreg_n_aligned": 1,
        "fetch_coreg_max_shift": float(np.sqrt(0.05)),
        "fetch_coreg_anchor_valid_fraction": 0.9,
        "fetch_bbox_epsg3006": identity["bbox_epsg3006"],
        "fetch_center_epsg3006": [identity["easting"], identity["northing"]],
        "fetch_tile_size_px": size,
        "fetch_num_frames": len(identity["fetch_requested_dates"]),
        "fetch_num_bands": 6,
        "fetch_spectral_shape": [
            len(identity["fetch_requested_dates"]) * 6, size, size,
        ],
        "fetch_scl_shape": [len(identity["fetch_requested_dates"]), size, size],
    }
    mask = np.ones((size, size), bool)
    mask_hash = hashlib.sha256(
        np.packbits(mask, bitorder="little").tobytes()
    ).hexdigest()
    derived = {
        key: np.ones((size, size), bool if key == "valid_mask" else np.float32)
        for key in poc_script._PAIR_DERIVED_ARRAY_KEYS
    }
    bands = {
        name: np.full((size, size), 0.2, np.float32)
        for name in poc_script.COMPARISON_BANDS
    }
    reference_vpp = {
        "sosd": np.full((size, size), 19120, np.float32),
        "eosd": np.full((size, size), 19280, np.float32),
        "length": np.full((size, size), 160, np.float32),
        "maxv": np.full((size, size), 1.2, np.float32),
        "minv": np.full((size, size), 0.2, np.float32),
    }
    candidate_vpp = {
        **reference_vpp,
        "sosd": np.full((size, size), 20121, np.float32),
        "eosd": np.full((size, size), 20281, np.float32),
    }
    def vpp_provenance(year, values):
        value = {
            "source": "wekeo",
            "product_year": year,
            "season": poc_script.VPP_SEASON,
            "versions": [105],
            "products": [{
                "filename": f"synthetic-{year}.tif",
                "sha256": "e" * 64,
            }],
        }
        return poc_script._bind_vpp_raster_provenance(value, values)

    reference_vpp_provenance = vpp_provenance(2019, reference_vpp)
    candidate_vpp_provenance = vpp_provenance(2020, candidate_vpp)
    artifact = arrays / "ok.npz"
    payload = poc_script._pair_artifact(
        arrays=derived,
        reference_bands=bands,
        candidate_bands=bands,
        reference_scl=np.full((size, size), 4, np.uint8),
        candidate_scl=np.full((size, size), 4, np.uint8),
        nmd=np.ones((size, size), np.uint8),
        stable_forest=mask,
        reference_vpp=reference_vpp,
        candidate_vpp=candidate_vpp,
        identity=identity,
        provenance=provenance,
        shared_mask_sha256=mask_hash,
        common_valid_pixel_count=int(mask.sum()),
        common_valid_fraction_of_forest=1.0,
        reference_vpp_provenance=reference_vpp_provenance,
        candidate_vpp_provenance=candidate_vpp_provenance,
    )
    artifact_meta = poc_script._write_npz_atomic(artifact, payload)
    ok = pairs / "ok.json"
    ok.write_text(json.dumps({
        **identity,
        **provenance,
        "status": "ok",
        "common_valid_mask_sha256": mask_hash,
        "common_valid_pixel_count": int(mask.sum()),
        "common_valid_fraction_of_forest": 1.0,
        "reference_vpp_provenance": reference_vpp_provenance,
        "candidate_vpp_provenance": candidate_vpp_provenance,
        "array_path": "arrays/ok.npz",
        "array_bytes": artifact_meta["bytes"],
        "array_sha256": artifact_meta["sha256"],
    }))
    assert not _pair_succeeded(failed, identity)
    assert _pair_succeeded(ok, identity)
    assert not _pair_succeeded(ok, {**identity, "candidate_date": "2020-06-03"})
    artifact.write_bytes(b"corrupt")
    assert not _pair_succeeded(ok, identity)

    tiles = tmp_path / "tiles.csv"
    tiles.write_text(
        "easting,northing,forest_fraction,classified_fraction,source\n"
        "500000,6500000,0.9,0.94,inventory\n"
    )
    assert _load_selected_tiles(tiles)[0].classified_fraction == 0.94


def test_fetch_group_rows_require_one_attempt_and_shared_provenance():
    shared = {
        key: f"shared-{key}"
        for key in poc_script._FETCH_GROUP_SHARED_KEYS
    }
    rows = [
        {
            **shared,
            "status": "ok",
            "selection_method": method,
        }
        for method in poc_script.SELECTION_METHODS
    ]
    assert poc_script._group_rows_consistent(rows)
    rows[1]["fetch_attempt_sha256"] = "different-attempt"
    assert not poc_script._group_rows_consistent(rows)


def test_candidate_years_are_unique_and_distinct():
    _validate_years(2019, [2020, 2021])
    for years in ([2020, 2020], [2019, 2020]):
        with np.testing.assert_raises(ValueError):
            _validate_years(2019, years)


def test_live_des_environment_requires_both_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("VPP_SOURCE", "wekeo")
    monkeypatch.setenv("VPP_WEKEO_DIR", str(tmp_path))
    monkeypatch.setattr(
        poc_script,
        "connect_des",
        lambda: (_ for _ in ()).throw(RuntimeError("no auth")),
    )
    with np.testing.assert_raises(RuntimeError):
        _validate_live_environment(SimpleNamespace(fetch_source="des"))
    monkeypatch.setattr(poc_script, "connect_des", lambda: object())
    _validate_live_environment(SimpleNamespace(fetch_source="des"))


def test_vpp_product_provenance_records_version_and_rejects_mismatch(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("VPP_SOURCE", "wekeo")
    monkeypatch.setenv("VPP_WEKEO_DIR", str(tmp_path))
    monkeypatch.setattr(
        poc_script,
        "_bbox_3006_to_4326",
        lambda *args: [10.0, 55.0, 25.0, 70.0],
    )
    monkeypatch.setattr(poc_script, "_bounds_overlap", lambda *args: True)
    index = {}
    for metric in (name.upper() for name in poc_script.VPP_BANDS):
        filename = f"VPP_2019_S2_33VWG-010m_V105_s1_{metric}.tif"
        (tmp_path / filename).write_bytes(metric.encode())
        index[filename] = {
            "year": 2019,
            "season": 1,
            "metric": metric,
            "tileId": "33VWG",
            "bounds_4326": [10.0, 55.0, 25.0, 70.0],
        }
    (tmp_path / "index.json").write_text(json.dumps(index))
    bbox = {"west": 0, "south": 0, "east": 10, "north": 10}
    provenance = poc_script._vpp_product_provenance(bbox, 2019)
    assert provenance["versions"] == [105]
    assert {item["metric"] for item in provenance["products"]} == {
        name.upper() for name in poc_script.VPP_BANDS
    }
    assert provenance["fingerprint"]

    sosd = "VPP_2019_S2_33VWG-010m_V105_s1_SOSD.tif"
    mismatched = "VPP_2020_S2_33VWG-010m_V105_s1_SOSD.tif"
    index[mismatched] = index.pop(sosd)
    (tmp_path / mismatched).write_bytes(b"SOSD")
    (tmp_path / "index.json").write_text(json.dumps(index))
    with np.testing.assert_raises_regex(ValueError, "metadata disagrees"):
        poc_script._vpp_product_provenance(bbox, 2019)


def test_vpp_product_provenance_rejects_mixed_versions(monkeypatch, tmp_path):
    monkeypatch.setenv("VPP_SOURCE", "wekeo")
    monkeypatch.setenv("VPP_WEKEO_DIR", str(tmp_path))
    monkeypatch.setattr(
        poc_script,
        "_bbox_3006_to_4326",
        lambda *args: [10.0, 55.0, 25.0, 70.0],
    )
    monkeypatch.setattr(poc_script, "_bounds_overlap", lambda *args: True)
    index = {}
    for metric in (name.upper() for name in poc_script.VPP_BANDS):
        version = 106 if metric == "SOSD" else 105
        filename = f"VPP_2019_S2_33VWG-010m_V{version}_s1_{metric}.tif"
        (tmp_path / filename).write_bytes(metric.encode())
        index[filename] = {
            "year": 2019,
            "season": 1,
            "metric": metric,
            "tileId": "33VWG",
            "bounds_4326": [10.0, 55.0, 25.0, 70.0],
        }
    (tmp_path / "index.json").write_text(json.dumps(index))
    with np.testing.assert_raises_regex(ValueError, "mixes product versions"):
        poc_script._vpp_product_provenance(
            {"west": 0, "south": 0, "east": 10, "north": 10}, 2019
        )


def test_vpp_versions_are_single_and_consistent_with_product_year():
    rows = [
        {
            "status": "ok",
            "reference_year": 2019,
            "candidate_year": 2020,
            "reference_vpp_provenance": {
                "product_year": 2019,
                "versions": [101],
            },
            "candidate_vpp_provenance": {
                "product_year": 2020,
                "versions": [101],
            },
        },
        {
            "status": "ok",
            "reference_year": 2019,
            "candidate_year": 2021,
            "reference_vpp_provenance": {
                "product_year": 2019,
                "versions": [101],
            },
            "candidate_vpp_provenance": {
                "product_year": 2021,
                "versions": [105],
            },
        },
    ]
    assert poc_script._vpp_versions_by_year(rows, [2019, 2020, 2021]) == {
        "2019": [101],
        "2020": [101],
        "2021": [105],
    }

    mixed = [*rows, {
        **rows[1],
        "candidate_vpp_provenance": {
            "product_year": 2021,
            "versions": [106],
        },
    }]
    assert poc_script._vpp_versions_by_year(mixed, [2019, 2020, 2021])["2021"] == [
        105,
        106,
    ]

    wrong_year = [{
        **rows[0],
        "candidate_vpp_provenance": {
            "product_year": 2021,
            "versions": [101],
        },
    }]
    with np.testing.assert_raises_regex(ValueError, "invalid VPP version"):
        poc_script._vpp_versions_by_year(wrong_year, [2019, 2020])


def test_source_provenance_changes_with_nmd_and_vpp_products(monkeypatch, tmp_path):
    inventory = tmp_path / "inventory.csv"
    inventory.write_text("easting,northing\n500000,6500000\n")
    nmd = tmp_path / "nmd.tif"
    nmd.write_bytes(b"nmd-v1")
    vpp_root = tmp_path / "vpp"
    vpp_root.mkdir()
    product = vpp_root / "VPP_2019_S2_33VWG-010m_V105_s1_SOSD.tif"
    product.write_bytes(b"vpp-v1")
    (vpp_root / "index.json").write_text(json.dumps({
        product.name: {
            "year": 2019,
            "season": 1,
            "metric": "SOSD",
            "tileId": "33VWG",
            "bounds_4326": [10.0, 55.0, 25.0, 70.0],
        }
    }))
    monkeypatch.setenv("VPP_SOURCE", "wekeo")
    monkeypatch.setenv("VPP_WEKEO_DIR", str(vpp_root))
    args = SimpleNamespace(
        inventory=inventory,
        nmd_raster=nmd,
        execute_fetch=True,
    )
    command = {
        "reference_year": 2019,
        "candidate_years": [2020],
        "window": ["05-15", "06-15"],
    }

    def fingerprint(source_provenance):
        return poc_script._canonical_sha256(poc_script._run_fingerprint_payload(
            command=command,
            imint_git_sha="a" * 40,
            metafilter_git_sha="b" * 40,
            source_provenance=source_provenance,
            selected_tiles_fingerprint="c" * 64,
            selected_tiles_file={"sha256": "d" * 64, "bytes": 1},
        ))

    initial = poc_script._source_provenance(args)
    nmd.write_bytes(b"nmd-v2-longer")
    changed_nmd = poc_script._source_provenance(args)
    product.write_bytes(b"vpp-v2-longer")
    changed_vpp = poc_script._source_provenance(args)
    assert initial["nmd"]["sha256"] != changed_nmd["nmd"]["sha256"]
    assert (
        changed_nmd["vpp"]["product_inventory_sha256"]
        != changed_vpp["vpp"]["product_inventory_sha256"]
    )
    assert len({
        fingerprint(initial),
        fingerprint(changed_nmd),
        fingerprint(changed_vpp),
    }) == 3


def test_main_returns_nonzero_when_fetch_group_fails(monkeypatch, tmp_path):
    output_dir = tmp_path / "output"
    run_dir = output_dir / "failed-run"
    run_dir.mkdir(parents=True)
    (run_dir / "tiles.csv").write_text(
        "easting,northing,forest_fraction,classified_fraction,source\n"
        "500000,6500000,0.9,0.95,synthetic\n"
    )
    args = SimpleNamespace(
        inventory=tmp_path / "unused.csv",
        nmd_raster=tmp_path / "nmd.tif",
        output_dir=output_dir,
        reference_year=2019,
        candidate_years=[2020],
        window=["05-15", "06-15"],
        tiles=1,
        size_px=8,
        min_forest_fraction=0.8,
        candidates_per_stratum=1,
        fetch_source="des",
        max_aoi_cloud=0.1,
        select_only=False,
        plan_network=False,
        execute_fetch=True,
        run_id="failed-run",
    )
    tile = TileCandidate(500000, 6500000, 0.9, "synthetic", 0.95)
    args.nmd_raster.touch()

    def fake_plan(selected, plan_args, cache_dir):
        matches = [
            DateMatch(
                2019, "2019-06-01", 2020, "2020-06-02", 0.2, 1,
                {}, {}, method,
            )
            for method in poc_script.SELECTION_METHODS
        ]
        context = {
            "bbox_3006": TileConfig(size_px=8).bbox_from_center(500000, 6500000),
            "bbox_wgs84": {},
            "reference_date": "2019-06-01",
            "matches": matches,
            "fetch_plans": {},
            "meteorology_by_year": {},
            "planning_inputs": poc_script._planning_inputs(tile, args),
            "planning_schema_version": poc_script.PLANNING_SCHEMA_VERSION,
        }
        context["plan_fingerprint"] = poc_script._plan_fingerprint(context)
        return context

    monkeypatch.setattr(poc_script, "parse_args", lambda: args)
    monkeypatch.setattr(poc_script, "_validate_live_environment", lambda value: None)
    monkeypatch.setattr(
        poc_script, "_validated_imint_git_sha", lambda **kwargs: "c" * 40
    )
    monkeypatch.setattr(poc_script, "_metafilter_sha", lambda: "d" * 40)
    monkeypatch.setattr(
        poc_script,
        "_source_provenance",
        lambda value: {"inventory": {}, "nmd": {}, "vpp": {}},
    )
    monkeypatch.setattr(poc_script, "plan_tile_dates", fake_plan)
    monkeypatch.setattr(
        poc_script,
        "fetch_and_compare_tile",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("synthetic failure")),
    )
    assert poc_script.main() == 1
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["stage"] == "complete_with_failures"
    assert manifest["pair_counts"]["fetch_failed"] == 2


def test_default_select_mode_never_calls_network(monkeypatch, tmp_path):
    inventory = tmp_path / "inventory.csv"
    inventory.write_text("easting,northing\n500000,6500000\n")
    raster = tmp_path / "nmd.tif"
    raster.touch()
    args = SimpleNamespace(
        inventory=inventory,
        nmd_raster=raster,
        output_dir=tmp_path / "output",
        reference_year=2019,
        candidate_years=[2020],
        window=["05-15", "06-15"],
        tiles=1,
        size_px=8,
        min_forest_fraction=0.8,
        candidates_per_stratum=1,
        fetch_source="des",
        max_aoi_cloud=0.1,
        select_only=False,
        plan_network=False,
        execute_fetch=False,
        run_id="offline-test",
    )
    monkeypatch.setattr(poc_script, "parse_args", lambda: args)
    monkeypatch.setattr(
        poc_script,
        "select_forest_tiles",
        lambda *a, **k: [TileCandidate(500000, 6500000, 0.9, "inventory")],
    )
    monkeypatch.setattr(
        poc_script,
        "fetch_daily_meteorology",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("network called")),
    )
    monkeypatch.setattr(
        poc_script,
        "optimal_fetch_dates",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("network called")),
    )
    monkeypatch.setattr(
        socket.socket,
        "connect",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("socket called")),
    )
    assert poc_script.main() == 0
    assert (args.output_dir / args.run_id / "tiles.csv").exists()


def test_checked_in_tile_config_is_ten_non_overlapping_forest_tiles():
    path = Path(__file__).parents[1] / "config" / "meteo_analog_forest_tiles.csv"
    tiles = _load_selected_tiles(path)
    assert len(tiles) == 10
    centers = {(tile.easting, tile.northing) for tile in tiles}
    assert len(centers) == 10
    assert all(tile.forest_fraction >= 0.8 for tile in tiles)
    assert all(
        tile.easting % 10 == 0 and tile.northing % 10 == 0
        for tile in tiles
    )
    tile_width_m = 512 * 10
    for index, left in enumerate(tiles):
        for right in tiles[index + 1:]:
            assert (
                abs(left.easting - right.easting) >= tile_width_m
                or abs(left.northing - right.northing) >= tile_width_m
            )
