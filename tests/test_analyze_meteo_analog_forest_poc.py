from __future__ import annotations

import json
import hashlib
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import scripts.run_meteo_analog_forest_poc as poc_script
from scripts.analyze_meteo_analog_forest_poc import (
    _cluster_bootstrap_median_ci,
    _verify_exact_group_mask,
    add_vpp_phase_metrics,
    analyze_run,
    paired_strategy_frame,
    strategy_summary,
)
from imint.experiments.meteo_analog_forest import (
    DateMatch,
    TileCandidate,
    compare_spectral_pair,
    summarize_vpp_phase,
    write_manifest,
)
from imint.training.tile_config import TileConfig


def _summary_row(tile_id: str, method: str, *, better: bool) -> dict:
    candidate_phase = 32.0 if better else 40.0
    return {
        "tile_id": tile_id,
        "candidate_year": 2020,
        "selection_method": method,
        "fetch_group_id": f"{tile_id}-2020-group",
        "common_valid_mask_sha256": f"{tile_id}-2020-mask",
        "status": "ok",
        "candidate_date": "2020-05-20" if better else "2020-06-02",
        "calendar_displacement_days": 12 if better else 1,
        "meteorology_distance": 0.2 if better else 0.8,
        "ndvi_absolute_difference_median": 0.03 if better else 0.08,
        "spectral_angle_median_rad": 0.04 if better else 0.10,
        "reference_vpp_days_from_sos": 30.0,
        "candidate_vpp_days_from_sos": candidate_phase,
        "reference_vpp_days_from_midpoint_proxy": -50.0,
        "candidate_vpp_days_from_midpoint_proxy": candidate_phase - 80.0,
        "reference_vpp_days_to_eos": 100.0,
        "candidate_vpp_days_to_eos": 130.0 - candidate_phase,
        "vpp_sos_shift_days": 1.0,
        "vpp_eos_shift_days": -2.0,
        "vpp_midpoint_proxy_shift_days": -0.5,
        "reference_vpp_version": 105,
        "candidate_vpp_version": 105,
        "vpp_version_pair": "V105->V105",
    }


def test_vpp_phase_metrics_are_method_dependent():
    frame = pd.DataFrame([
        _summary_row("tile_00", "closest_date", better=False),
        _summary_row("tile_00", "meteorology", better=True),
    ])
    enriched = add_vpp_phase_metrics(frame)
    paired = paired_strategy_frame(enriched)
    assert paired.loc[0, "vpp_phase_alignment_mae_days_meteorology"] == 2.0
    assert paired.loc[0, "vpp_phase_alignment_mae_days_closest_date"] == 10.0
    assert paired.loc[0, "vpp_phase_alignment_mae_days_meteorology_wins"]


def test_paired_frame_rejects_missing_strategy():
    frame = add_vpp_phase_metrics(pd.DataFrame([
        _summary_row("tile_00", "closest_date", better=False),
    ]))
    with pytest.raises(ValueError, match="do not cover the same pairs"):
        paired_strategy_frame(frame)


def test_bootstrap_resamples_complete_tile_clusters():
    frame = pd.DataFrame({
        "tile_id": ["north", "north", "south", "south"],
        "delta": [1.0, 1.0, 9.0, 9.0],
    })
    first = _cluster_bootstrap_median_ci(
        frame,
        value_column="delta",
        cluster_column="tile_id",
        samples=500,
        seed=42,
    )
    second = _cluster_bootstrap_median_ci(
        frame,
        value_column="delta",
        cluster_column="tile_id",
        samples=500,
        seed=42,
    )
    assert first == second
    assert first[0] == 1.0
    assert first[1] == 9.0


def test_complete_run_analysis_verifies_artifacts_and_writes_outputs(tmp_path):
    _write_synthetic_run(tmp_path, tile_count=2)
    output_dir = tmp_path / "analysis"
    result = analyze_run(
        tmp_path,
        output_dir=output_dir,
        expected_tiles=2,
        expected_years=1,
        bootstrap_samples=100,
    )
    assert result["verified_pairs"] == 4
    assert result["paired_comparisons"] == 2
    assert len(pd.read_csv(output_dir / "pair_metrics.csv")) == 4
    assert len(pd.read_csv(output_dir / "paired_strategy_comparison.csv")) == 2
    assert (output_dir / "strategy_effects.csv").is_file()
    assert (output_dir / "strategy_effects_by_vpp_version.csv").is_file()
    assert (output_dir / "strategy_boxplots.png").stat().st_size > 0
    manifest = json.loads((output_dir / "analysis_manifest.json").read_text())
    assert manifest["artifacts_verified"]
    assert manifest["bootstrap_samples"] == 100
    assert manifest["fixed_year_count"] == 1
    assert manifest["vpp_version_pairs"] == ["V105->V105"]
    assert all(
        item["bytes"] > 0 and len(item["sha256"]) == 64
        for item in manifest["output_files"].values()
    )
    assert analyze_run(
        tmp_path,
        output_dir=output_dir,
        expected_tiles=2,
        expected_years=1,
        bootstrap_samples=100,
    ) == result
    effects = pd.read_csv(output_dir / "strategy_effects.csv")
    assert set(effects["metric"]) == {
        "ndvi_absolute_difference_median",
        "spectral_angle_median_rad",
        "vpp_phase_alignment_mae_days",
    }
    assert set(effects["bootstrap_unit"]) == {"tile_id"}
    assert set(effects["year_n"]) == {1}
    assert "meteorology_distance" not in set(effects["metric"])


def test_strategy_summary_reports_independent_tile_count():
    frame = pd.DataFrame([
        _summary_row(tile_id, method, better=(method == "meteorology"))
        for tile_id in ("north", "south")
        for method in poc_script.SELECTION_METHODS
    ])
    summary = strategy_summary(add_vpp_phase_metrics(frame))
    assert set(summary["n"]) == {2}
    assert set(summary["cluster_n"]) == {2}
    assert set(summary["year_n"]) == {1}


def test_analysis_rejects_tampered_tiles_csv(tmp_path):
    _write_synthetic_run(tmp_path, tile_count=1)
    with (tmp_path / "tiles.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="tiles.csv"):
        analyze_run(
            tmp_path,
            output_dir=tmp_path / "analysis",
            expected_tiles=1,
            expected_years=1,
            bootstrap_samples=10,
        )


def test_analysis_rejects_tampered_vpp_version_inventory(tmp_path):
    _write_synthetic_run(tmp_path, tile_count=1)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["vpp_versions_by_year"]["2020"] = [101]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="VPP versions"):
        analyze_run(
            tmp_path,
            output_dir=tmp_path / "analysis",
            expected_tiles=1,
            expected_years=1,
            bootstrap_samples=10,
        )


def test_analysis_rejects_unexpected_pair_manifest(tmp_path):
    _write_synthetic_run(tmp_path, tile_count=1)
    (tmp_path / "pairs" / "unexpected.json").write_text("{}")
    with pytest.raises(ValueError, match="pair manifest set mismatch"):
        analyze_run(
            tmp_path,
            output_dir=tmp_path / "analysis",
            expected_tiles=1,
            expected_years=1,
            bootstrap_samples=10,
        )


def test_analysis_rejects_tampered_planned_match_fields(tmp_path):
    _write_synthetic_run(tmp_path, tile_count=1)
    pair = next((tmp_path / "pairs").glob("*meteorology*.json"))
    row = json.loads(pair.read_text())
    row["meteorology_distance"] += 1.0
    pair.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="invalid pair artifacts"):
        analyze_run(
            tmp_path,
            output_dir=tmp_path / "analysis",
            expected_tiles=1,
            expected_years=1,
            bootstrap_samples=10,
        )


def test_analysis_rejects_self_consistent_incomplete_scl_plan(tmp_path):
    _write_synthetic_run(tmp_path, tile_count=1)
    plan = next((tmp_path / "plans").glob("*.json"))
    payload = json.loads(plan.read_text())
    payload["fetch_plans"]["2020"]["scl_screen_complete"] = False
    payload.pop("plan_fingerprint")
    payload["plan_fingerprint"] = poc_script._canonical_sha256(payload)
    plan.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="invalid or stale plan"):
        analyze_run(
            tmp_path,
            output_dir=tmp_path / "analysis",
            expected_tiles=1,
            expected_years=1,
            bootstrap_samples=10,
        )


def test_analysis_rejects_forest_mask_not_derived_from_nmd(tmp_path):
    _write_synthetic_run(tmp_path, tile_count=1)
    pair = next((tmp_path / "pairs").glob("*.json"))
    row = json.loads(pair.read_text())
    artifact = tmp_path / row["array_path"]
    with np.load(artifact, allow_pickle=False) as data:
        payload = {
            name: np.asarray(data[name]).copy()
            for name in data.files
        }
    payload["stable_forest_mask"][0, 0] = False
    np.savez_compressed(artifact, **payload)
    row["array_bytes"] = artifact.stat().st_size
    row["array_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    pair.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="invalid pair artifacts"):
        analyze_run(
            tmp_path,
            output_dir=tmp_path / "analysis",
            expected_tiles=1,
            expected_years=1,
            bootstrap_samples=10,
        )


def test_analysis_rejects_orphan_plan_and_array_artifacts(tmp_path):
    orphan_plan_run = tmp_path / "orphan-plan"
    orphan_plan_run.mkdir()
    _write_synthetic_run(orphan_plan_run, tile_count=1)
    (orphan_plan_run / "plans" / "orphan.json").write_text("{}")
    with pytest.raises(ValueError, match="plan artifact set mismatch"):
        analyze_run(
            orphan_plan_run,
            output_dir=tmp_path / "plan-analysis",
            expected_tiles=1,
            expected_years=1,
            bootstrap_samples=10,
        )

    orphan_array_run = tmp_path / "orphan-array"
    orphan_array_run.mkdir()
    _write_synthetic_run(orphan_array_run, tile_count=1)
    np.savez_compressed(orphan_array_run / "arrays" / "orphan.npz", value=1)
    with pytest.raises(ValueError, match="array artifact set mismatch"):
        analyze_run(
            orphan_array_run,
            output_dir=tmp_path / "array-analysis",
            expected_tiles=1,
            expected_years=1,
            bootstrap_samples=10,
        )


def test_analysis_rejects_tampered_published_output(tmp_path):
    _write_synthetic_run(tmp_path, tile_count=1)
    output_dir = tmp_path / "analysis"
    analyze_run(
        tmp_path,
        output_dir=output_dir,
        expected_tiles=1,
        expected_years=1,
        bootstrap_samples=10,
    )
    with (output_dir / "strategy_summary.csv").open("a") as stream:
        stream.write("tampered\n")
    with pytest.raises(ValueError, match="is corrupt"):
        analyze_run(
            tmp_path,
            output_dir=output_dir,
            expected_tiles=1,
            expected_years=1,
            bootstrap_samples=10,
        )


def test_exact_group_mask_rejects_valid_but_narrower_subset(tmp_path):
    _write_synthetic_run(tmp_path, tile_count=1)
    pair_paths = sorted((tmp_path / "pairs").glob("*.json"))
    rows = [json.loads(path.read_text()) for path in pair_paths]
    for row in rows:
        artifact = tmp_path / row["array_path"]
        with np.load(artifact, allow_pickle=False) as data:
            payload = {
                name: np.asarray(data[name]).copy()
                for name in data.files
            }
        narrower = np.asarray(payload["valid_mask"], dtype=bool).copy()
        narrower[0, 0] = False
        payload["valid_mask"] = narrower
        np.savez_compressed(artifact, **payload)
    with pytest.raises(ValueError, match="exact two-strategy intersection"):
        _verify_exact_group_mask(tmp_path, rows)


def _write_synthetic_run(run_dir, *, tile_count):
    size = 4
    command = {
        "reference_year": 2019,
        "candidate_years": [2020],
        "window": ["05-15", "06-15"],
        "size_px": size,
        "max_aoi_cloud": 0.1,
        "fetch_source": "des",
    }
    args = SimpleNamespace(**command)
    tiles = [
        TileCandidate(
            easting=500000 + index * 10000,
            northing=6500000 + index * 10000,
            forest_fraction=0.9,
            source="synthetic",
            classified_fraction=0.95,
        )
        for index in range(tile_count)
    ]
    poc_script._write_rows(
        run_dir / "tiles.csv", [asdict(tile) for tile in tiles]
    )
    selected_tiles_fingerprint = poc_script._canonical_sha256(
        [asdict(tile) for tile in tiles]
    )
    selected_tiles_file = poc_script._file_identity(run_dir / "tiles.csv")
    source_provenance = {
        "inventory": {
            "kind": "file",
            "path": "/synthetic/inventory.csv",
            "bytes": 1,
            "sha256": "1" * 64,
        },
        "nmd": {
            "path": "/synthetic/nmd.tif",
            "bytes": 1,
            "sha256": "2" * 64,
        },
        "vpp": {
            "source": "wekeo",
            "available": True,
            "path": "/synthetic/vpp",
            "index": {
                "path": "/synthetic/vpp/index.json",
                "bytes": 1,
                "sha256": "3" * 64,
            },
            "product_count": 1,
            "product_inventory_sha256": "4" * 64,
            "versions": [105],
        },
    }
    imint_git_sha = "c" * 40
    metafilter_git_sha = "d" * 40
    run_fingerprint = poc_script._canonical_sha256(
        poc_script._run_fingerprint_payload(
            command=command,
            imint_git_sha=imint_git_sha,
            metafilter_git_sha=metafilter_git_sha,
            source_provenance=source_provenance,
            selected_tiles_fingerprint=selected_tiles_fingerprint,
            selected_tiles_file=selected_tiles_file,
        )
    )
    (run_dir / "plans").mkdir()
    (run_dir / "pairs").mkdir()
    for tile_index, tile in enumerate(tiles):
        tile_id = f"tile_{tile_index:02d}_{tile.easting}_{tile.northing}"
        matches = [
            DateMatch(
                reference_year=2019,
                reference_date="2019-06-01",
                candidate_year=2020,
                candidate_date=(
                    "2020-06-02" if method == "closest_date" else "2020-05-20"
                ),
                meteorology_distance=(0.8 if method == "closest_date" else 0.2),
                calendar_displacement_days=(1 if method == "closest_date" else 12),
                feature_values={},
                normalized_deltas={},
                selection_method=method,
            )
            for method in poc_script.SELECTION_METHODS
        ]
        dates_by_year = {
            2019: ["2019-06-01"],
            2020: sorted({match.candidate_date for match in matches}),
        }
        fetch_plans = {}
        meteorology_by_year = {}
        for year, dates in dates_by_year.items():
            fetch_plans[year] = {
                "mode": "scl_only",
                "dates": dates,
                "n_candidates_after": {"final": len(dates)},
                "elapsed_s": {"scl_stack": 0.1},
                "notes": {"scl_backend": "des"},
                "scl_gate": {
                    observed: {
                        "cloud_fraction": 0.01,
                        "snow_fraction": 0.0,
                        "coverage_fraction": 1.0,
                        "accepted": True,
                    }
                    for observed in dates
                },
                "scl_screen_complete": True,
                "scl_thresholds": {
                    "max_aoi_cloud": 0.1,
                    "max_aoi_snow": 0.01,
                    "min_aoi_coverage": 0.8,
                },
            }
            meteorology_by_year[str(year)] = [{"date": f"{year}-06-01"}]
        context = {
            "bbox_3006": TileConfig(size_px=size).bbox_from_center(
                tile.easting, tile.northing
            ),
            "bbox_wgs84": {
                "west": 10.0,
                "south": 60.0,
                "east": 10.1,
                "north": 60.1,
            },
            "reference_date": "2019-06-01",
            "matches": matches,
            "fetch_plans": fetch_plans,
            "meteorology_by_year": meteorology_by_year,
            "planning_inputs": poc_script._planning_inputs(tile, args),
            "planning_schema_version": poc_script.PLANNING_SCHEMA_VERSION,
        }
        context["plan_fingerprint"] = poc_script._plan_fingerprint(context)
        write_manifest(
            run_dir / "plans" / f"{tile_id}.json",
            poc_script._serialize_context(context),
        )
        for match in matches:
            _write_synthetic_pair(
                run_dir,
                tile_id,
                tile,
                context,
                matches,
                match,
                args,
                run_fingerprint,
            )

    write_manifest(run_dir / "manifest.json", {
        "run_id": "synthetic",
        "stage": "complete",
        "command": command,
        "tiles": [asdict(tile) for tile in tiles],
        "planning_schema_version": poc_script.PLANNING_SCHEMA_VERSION,
        "dataset_schema_version": poc_script.DATASET_SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "imint_git_sha": imint_git_sha,
        "metafilter_git_sha": metafilter_git_sha,
        "source_provenance": source_provenance,
        "vpp_versions_by_year": {"2019": [105], "2020": [105]},
        "selected_tiles_fingerprint": selected_tiles_fingerprint,
        "selected_tiles_file": selected_tiles_file,
        "pair_counts": {
            "requested": tile_count * 2,
            "successful": tile_count * 2,
            "fetch_failed": 0,
            "planning_missing": 0,
            "pair_manifests_missing": 0,
            "pair_manifests_unexpected": 0,
            "fetch_groups_inconsistent": 0,
            "vpp_version_inconsistent_years": 0,
        },
    })


def _write_synthetic_pair(
    run_dir, tile_id, tile, context, group_matches, match, args, run_fingerprint
):
    size = args.size_px
    identity = poc_script._pair_identity(
        tile_id,
        tile,
        context,
        group_matches,
        match,
        size_px=size,
        fetch_source=args.fetch_source,
        run_fingerprint=run_fingerprint,
    )
    reference_values = [0.10, 0.15, 0.20, 0.40, 0.30, 0.25, 0.20]
    change = 0.01 if match.selection_method == "meteorology" else 0.08
    reference = {
        name: np.full((size, size), value, np.float32)
        for name, value in zip(poc_script.COMPARISON_BANDS, reference_values)
    }
    candidate = {
        name: np.full(
            (size, size), value + change * (index + 1) / 7, np.float32
        )
        for index, (name, value) in enumerate(
            zip(poc_script.COMPARISON_BANDS, reference_values)
        )
    }
    mask = np.ones((size, size), bool)
    arrays, metrics = compare_spectral_pair(reference, candidate, mask)
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
    ref_phase = summarize_vpp_phase(
        reference_vpp, match.reference_date, mask=mask
    )
    cand_phase = summarize_vpp_phase(
        candidate_vpp, match.candidate_date, mask=mask
    )
    requested = identity["fetch_requested_dates"]
    provenance = {
        "fetch_attempt_sha256": "f" * 64,
        "fetch_source": "des",
        "fetch_returned_dates": requested,
        "fetch_temporal_mask": [1] * len(requested),
        "fetch_frame_valid_fraction": [1.0] * len(requested),
        "fetch_coreg_shifts": [[0.0, 0.0]] * len(requested),
        "fetch_coreg_ref_frame": 0,
        "fetch_coreg_m2": 1,
        "fetch_coreg_n_aligned": 0,
        "fetch_coreg_max_shift": 0.0,
        "fetch_coreg_anchor_valid_fraction": 1.0,
        "fetch_bbox_epsg3006": identity["bbox_epsg3006"],
        "fetch_center_epsg3006": [identity["easting"], identity["northing"]],
        "fetch_tile_size_px": size,
        "fetch_num_frames": len(requested),
        "fetch_num_bands": 6,
        "fetch_spectral_shape": [len(requested) * 6, size, size],
        "fetch_scl_shape": [len(requested), size, size],
    }
    mask_hash = hashlib.sha256(
        np.packbits(mask, bitorder="little").tobytes()
    ).hexdigest()
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
    artifact = poc_script._pair_artifact(
        arrays=arrays,
        reference_bands=reference,
        candidate_bands=candidate,
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
    array_path = (
        run_dir / "arrays" / tile_id
        / f"{match.selection_method}_{match.candidate_year}.npz"
    )
    artifact_meta = poc_script._write_npz_atomic(array_path, artifact)
    row = {
        **identity,
        **asdict(match),
        **metrics,
        **provenance,
        "status": "ok",
        "forest_fraction": tile.forest_fraction,
        "classified_fraction": tile.classified_fraction,
        "common_valid_mask_sha256": mask_hash,
        "common_valid_pixel_count": int(mask.sum()),
        "common_valid_fraction_of_forest": 1.0,
        "reference_vpp_provenance": reference_vpp_provenance,
        "candidate_vpp_provenance": candidate_vpp_provenance,
        **{f"reference_vpp_{key}": value for key, value in ref_phase.items()},
        **{f"candidate_vpp_{key}": value for key, value in cand_phase.items()},
        "vpp_sos_shift_days": cand_phase["sos_doy_median"] - ref_phase["sos_doy_median"],
        "vpp_eos_shift_days": cand_phase["eos_doy_median"] - ref_phase["eos_doy_median"],
        "vpp_midpoint_proxy_shift_days": (
            cand_phase["season_midpoint_proxy_doy"]
            - ref_phase["season_midpoint_proxy_doy"]
        ),
        "array_path": str(array_path.relative_to(run_dir)),
        "array_bytes": artifact_meta["bytes"],
        "array_sha256": artifact_meta["sha256"],
    }
    write_manifest(
        poc_script._pair_manifest_path(run_dir / "pairs", tile_id, match), row
    )
