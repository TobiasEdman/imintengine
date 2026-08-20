#!/usr/bin/env python3
"""Validate and analyze a completed meteorological-analog forest POC run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.run_meteo_analog_forest_poc import (
    DATASET_SCHEMA_VERSION,
    SELECTION_METHODS,
    VPP_BANDS,
    _canonical_sha256,
    _common_vpp_valid_mask,
    _expected_pair_manifest_names,
    _group_rows_consistent,
    _load_context,
    _pair_identity,
    _pair_succeeded,
    _plan_is_current,
    _planning_inputs,
    _run_fingerprint_payload,
    _sha256_file,
    _vpp_versions_by_year,
    _year_matches,
)
from imint.experiments.meteo_analog_forest import (
    COMPARISON_BANDS,
    TileCandidate,
    common_valid_mask,
    compare_spectral_pair,
    forest_mask,
    summarize_vpp_phase,
)


EFFICACY_METRICS = {
    "ndvi_absolute_difference_median": "Median |NDVI difference|",
    "spectral_angle_median_rad": "Median spectral angle (rad)",
    "vpp_phase_alignment_mae_days": "VPP phase alignment error (days)",
}
DIAGNOSTIC_METRICS = {
    "meteorology_distance": "Meteorological distance",
}
REPORT_METRICS = {**EFFICACY_METRICS, **DIAGNOSTIC_METRICS}
VPP_PHASE_COMPONENTS = {
    "sos": "days_from_sos",
    "midpoint": "days_from_midpoint_proxy",
    "eos": "days_to_eos",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-tiles", type=int, default=10)
    parser.add_argument("--expected-years", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--expected-imint-git-sha")
    parser.add_argument("--expected-metafilter-git-sha")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.run_dir / "analysis"
    analyze_run(
        args.run_dir,
        output_dir=output_dir,
        expected_tiles=args.expected_tiles,
        expected_years=args.expected_years,
        bootstrap_samples=args.bootstrap_samples,
        expected_imint_git_sha=args.expected_imint_git_sha,
        expected_metafilter_git_sha=args.expected_metafilter_git_sha,
    )
    print(f"Analysis complete: {output_dir}")
    return 0


def analyze_run(
    run_dir: Path,
    *,
    output_dir: Path,
    expected_tiles: int,
    expected_years: int,
    bootstrap_samples: int = 10_000,
    expected_imint_git_sha: str | None = None,
    expected_metafilter_git_sha: str | None = None,
) -> dict:
    """Verify a complete run and emit tables, plots, and analysis metadata."""
    run_dir = Path(run_dir).resolve()
    output_dir = Path(output_dir).resolve()
    manifest, frame, pair_paths = load_verified_pairs(
        run_dir,
        expected_tiles=expected_tiles,
        expected_years=expected_years,
        expected_imint_git_sha=expected_imint_git_sha,
        expected_metafilter_git_sha=expected_metafilter_git_sha,
    )
    frame = add_vpp_phase_metrics(frame)
    paired = paired_strategy_frame(frame)
    summary = strategy_summary(frame)
    effects = paired_effects(paired, bootstrap_samples=bootstrap_samples)
    version_effects = version_stratified_effects(
        paired, bootstrap_samples=bootstrap_samples
    )
    outputs = {
        "pair_metrics": "pair_metrics.csv",
        "paired_strategy_comparison": "paired_strategy_comparison.csv",
        "strategy_summary": "strategy_summary.csv",
        "strategy_effects": "strategy_effects.csv",
        "strategy_effects_by_vpp_version": (
            "strategy_effects_by_vpp_version.csv"
        ),
        "strategy_boxplots": "strategy_boxplots.png",
        "vpp_phase_comparison": "vpp_phase_comparison.png",
        "meteo_vs_ndvi": "meteo_vs_ndvi.png",
    }
    publication_identity = {
        "source_run": str(run_dir),
        "source_run_id": manifest.get("run_id"),
        "source_manifest_sha256": _sha256_file(run_dir / "manifest.json"),
        "run_fingerprint": manifest["run_fingerprint"],
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "expected_tiles": expected_tiles,
        "expected_years": expected_years,
        "bootstrap_samples": bootstrap_samples,
        "expected_imint_git_sha": expected_imint_git_sha,
        "expected_metafilter_git_sha": expected_metafilter_git_sha,
    }
    existing = _reuse_existing_analysis(
        output_dir, publication_identity=publication_identity, outputs=outputs
    )
    if existing is not None:
        return existing

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent
    ))
    try:
        frame.to_csv(staging / outputs["pair_metrics"], index=False)
        paired.to_csv(
            staging / outputs["paired_strategy_comparison"], index=False
        )
        summary.to_csv(staging / outputs["strategy_summary"], index=False)
        effects.to_csv(staging / outputs["strategy_effects"], index=False)
        version_effects.to_csv(
            staging / outputs["strategy_effects_by_vpp_version"], index=False
        )
        render_analysis_figures(frame, paired, staging)
        output_files = {
            name: _published_file_identity(staging / relative, relative)
            for name, relative in outputs.items()
        }
        analysis_manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            **publication_identity,
            "selection_methods": list(SELECTION_METHODS),
            "verified_pairs": len(frame),
            "paired_comparisons": len(paired),
            "pair_manifest_sha256": _combined_sha256(pair_paths),
            "artifacts_verified": True,
            "metrics_recomputed_from_npz": True,
            "exact_group_masks_reconstructed": True,
            "bootstrap_unit": "tile_id",
            "bootstrap_cluster_count": int(frame["tile_id"].nunique()),
            "fixed_year_count": int(frame["candidate_year"].nunique()),
            "vpp_version_pairs": sorted(frame["vpp_version_pair"].unique()),
            "inference_scope": (
                "tile-cluster bootstrap conditional on the fixed candidate "
                "years; years are not resampled"
            ),
            "efficacy_metrics": list(EFFICACY_METRICS),
            "diagnostic_metrics": list(DIAGNOSTIC_METRICS),
            "outputs": outputs,
            "output_files": output_files,
        }
        destination = staging / "analysis_manifest.json"
        with destination.open("w") as handle:
            json.dump(analysis_manifest, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        staging.replace(output_dir)
        return analysis_manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _published_file_identity(path: Path, relative: str) -> dict:
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _reuse_existing_analysis(output_dir, *, publication_identity, outputs):
    if not output_dir.exists():
        return None
    manifest_path = output_dir / "analysis_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"existing analysis is incomplete: {output_dir}")
    payload = json.loads(manifest_path.read_text())
    if any(payload.get(key) != value for key, value in publication_identity.items()):
        raise ValueError("existing analysis belongs to different inputs or parameters")
    recorded = payload.get("output_files", {})
    if set(recorded) != set(outputs):
        raise ValueError("existing analysis has an incomplete output inventory")
    for name, relative in outputs.items():
        path = output_dir / relative
        identity = recorded[name]
        if (
            identity.get("path") != relative
            or not path.is_file()
            or identity.get("bytes") != path.stat().st_size
            or identity.get("sha256") != _sha256_file(path)
        ):
            raise ValueError(f"existing analysis output {name!r} is corrupt")
    return payload


def load_verified_pairs(
    run_dir: Path,
    *,
    expected_tiles: int,
    expected_years: int,
    expected_imint_git_sha: str | None = None,
    expected_metafilter_git_sha: str | None = None,
) -> tuple[dict, pd.DataFrame, list[Path]]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("stage") != "complete":
        raise ValueError(f"run is not complete: stage={manifest.get('stage')!r}")

    expected_pairs = expected_tiles * expected_years * len(SELECTION_METHODS)
    counts = manifest.get("pair_counts", {})
    if counts.get("requested") != expected_pairs:
        raise ValueError(
            f"manifest requested {counts.get('requested')} pairs; expected {expected_pairs}"
        )
    if counts.get("successful") != expected_pairs:
        raise ValueError(
            f"manifest has {counts.get('successful')} successful pairs; "
            f"expected {expected_pairs}"
        )
    failure_counts = (
        "fetch_failed",
        "planning_missing",
        "pair_manifests_missing",
        "pair_manifests_unexpected",
        "fetch_groups_inconsistent",
        "vpp_version_inconsistent_years",
    )
    if any(counts.get(name) != 0 for name in failure_counts):
        raise ValueError(f"manifest records failures: {counts}")

    if manifest.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("run manifest has the wrong dataset schema")
    if not manifest.get("run_fingerprint"):
        raise ValueError("run manifest has no run fingerprint")
    imint_git_sha = manifest.get("imint_git_sha", "")
    metafilter_git_sha = manifest.get("metafilter_git_sha", "")
    for label, value in (
        ("ImintEngine", imint_git_sha),
        ("metafilter", metafilter_git_sha),
    ):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError(f"run manifest has invalid {label} Git provenance")
    if expected_imint_git_sha and imint_git_sha != expected_imint_git_sha:
        raise ValueError("run manifest does not match expected ImintEngine SHA")
    if (
        expected_metafilter_git_sha
        and metafilter_git_sha != expected_metafilter_git_sha
    ):
        raise ValueError("run manifest does not match expected metafilter SHA")
    _validate_source_provenance(manifest.get("source_provenance"))
    selected_tiles = manifest.get("tiles", [])
    selected_fingerprint = _canonical_sha256(selected_tiles)
    if manifest.get("selected_tiles_fingerprint") != selected_fingerprint:
        raise ValueError("run manifest selected-tile fingerprint is inconsistent")
    tiles_path = run_dir / "tiles.csv"
    selected_file = manifest.get("selected_tiles_file", {})
    if (
        not tiles_path.is_file()
        or selected_file.get("sha256") != _sha256_file(tiles_path)
        or selected_file.get("bytes") != tiles_path.stat().st_size
    ):
        raise ValueError("run tiles.csv does not match its recorded identity")
    fingerprint_payload = _run_fingerprint_payload(
        command=manifest["command"],
        imint_git_sha=manifest["imint_git_sha"],
        metafilter_git_sha=manifest["metafilter_git_sha"],
        source_provenance=manifest.get("source_provenance"),
        selected_tiles_fingerprint=selected_fingerprint,
        selected_tiles_file=selected_file,
    )
    if manifest["run_fingerprint"] != _canonical_sha256(fingerprint_payload):
        raise ValueError("run fingerprint does not bind the recorded source inputs")

    command = manifest.get("command", {})
    plan_args = SimpleNamespace(
        reference_year=int(command["reference_year"]),
        candidate_years=[int(year) for year in command["candidate_years"]],
        window=list(command["window"]),
        size_px=int(command["size_px"]),
        max_aoi_cloud=float(command["max_aoi_cloud"]),
        fetch_source=command["fetch_source"],
    )
    tile_by_id = {}
    for index, payload in enumerate(manifest.get("tiles", [])):
        tile = TileCandidate(**payload)
        tile_id = f"tile_{index:02d}_{tile.easting}_{tile.northing}"
        tile_by_id[tile_id] = tile
    if len(tile_by_id) != expected_tiles:
        raise ValueError("run manifest does not contain the expected tiles")

    expected_plan_names = {f"{tile_id}.json" for tile_id in tile_by_id}
    plans_dir = run_dir / "plans"
    plan_entries = list(plans_dir.iterdir()) if plans_dir.is_dir() else []
    actual_plan_names = {path.name for path in plan_entries if path.is_file()}
    non_file_plans = {path.name for path in plan_entries if not path.is_file()}
    if actual_plan_names != expected_plan_names or non_file_plans:
        raise ValueError(
            "plan artifact set mismatch: "
            f"missing={sorted(expected_plan_names - actual_plan_names)}, "
            f"unexpected={sorted((actual_plan_names - expected_plan_names) | non_file_plans)}"
        )
    plans = {}
    for tile_id, tile in tile_by_id.items():
        path = plans_dir / f"{tile_id}.json"
        context = _load_context(path)
        if not _plan_is_current(context, _planning_inputs(tile, plan_args)):
            raise ValueError(f"invalid or stale plan for {tile_id}")
        plans[tile_id] = context

    candidate_years = [int(year) for year in command["candidate_years"]]
    if len(candidate_years) != expected_years:
        raise ValueError("run command does not contain the expected candidate years")
    expected_pair_names = _expected_pair_manifest_names(
        sorted(tile_by_id), candidate_years
    )
    pairs_dir = run_dir / "pairs"
    actual_pair_entries = list(pairs_dir.iterdir()) if pairs_dir.is_dir() else []
    actual_pair_paths = {
        path.name: path for path in sorted(actual_pair_entries) if path.is_file()
    }
    actual_pair_names = set(actual_pair_paths)
    non_files = {path.name for path in actual_pair_entries if not path.is_file()}
    missing = sorted(expected_pair_names - actual_pair_names)
    unexpected = sorted((actual_pair_names - expected_pair_names) | non_files)
    if missing or unexpected:
        raise ValueError(
            f"pair manifest set mismatch: missing={missing[:10]}, "
            f"unexpected={unexpected[:10]}"
        )
    pair_paths = [actual_pair_paths[name] for name in sorted(expected_pair_names)]
    rows = []
    invalid = []
    for path in pair_paths:
        try:
            row = json.loads(path.read_text())
            tile_id = row["tile_id"]
            context = plans[tile_id]
            group_matches = _year_matches(context, int(row["candidate_year"]))
            match = next(
                item
                for item in group_matches
                if item.selection_method == row["selection_method"]
            )
            expected = _pair_identity(
                tile_id,
                tile_by_id[tile_id],
                context,
                group_matches,
                match,
                size_px=plan_args.size_px,
                fetch_source=plan_args.fetch_source,
                run_fingerprint=manifest["run_fingerprint"],
            )
            if not _pair_succeeded(path, expected):
                raise ValueError("identity or artifact validation failed")
            rows.append(_recompute_pair_row(run_dir, row))
        except (KeyError, OSError, StopIteration, TypeError, ValueError) as error:
            invalid.append(f"{path.name}: {error}")
    if invalid:
        raise ValueError(f"invalid pair artifacts: {invalid[:10]}")

    expected_array_paths = {
        Path(row["array_path"]).as_posix() for row in rows
    }
    arrays_dir = run_dir / "arrays"
    actual_array_paths = {
        path.relative_to(run_dir).as_posix()
        for path in arrays_dir.rglob("*")
        if path.is_file()
    } if arrays_dir.is_dir() else set()
    if actual_array_paths != expected_array_paths:
        raise ValueError(
            "array artifact set mismatch: "
            f"missing={sorted(expected_array_paths - actual_array_paths)}, "
            f"unexpected={sorted(actual_array_paths - expected_array_paths)}"
        )

    frame = pd.DataFrame(rows)
    key = ["tile_id", "candidate_year", "selection_method"]
    if frame.duplicated(key).any():
        raise ValueError("duplicate tile/year/selection pair manifests")
    methods = set(frame["selection_method"])
    if methods != set(SELECTION_METHODS):
        raise ValueError(f"selection methods are {sorted(methods)}, expected {SELECTION_METHODS}")
    if frame["tile_id"].nunique() != expected_tiles:
        raise ValueError("pair manifests do not cover the expected tile count")
    if frame["candidate_year"].nunique() != expected_years:
        raise ValueError("pair manifests do not cover the expected year count")
    grouped = frame.groupby(["tile_id", "candidate_year"], sort=False)
    if (grouped["fetch_group_id"].nunique() != 1).any():
        raise ValueError("strategies do not share one fetch/coreg group")
    if (grouped["common_valid_mask_sha256"].nunique() != 1).any():
        raise ValueError("strategies do not share one valid-pixel mask")
    for _, group in grouped:
        group_rows = group.to_dict("records")
        if not _group_rows_consistent(group_rows):
            raise ValueError("strategies do not share exact fetch-group provenance")
        _verify_exact_group_mask(run_dir, group_rows)
    run_years = [plan_args.reference_year, *candidate_years]
    observed_versions = _vpp_versions_by_year(rows, run_years)
    if manifest.get("vpp_versions_by_year") != observed_versions:
        raise ValueError("run manifest VPP versions do not match pair artifacts")
    if any(len(versions) != 1 for versions in observed_versions.values()):
        raise ValueError("run uses inconsistent VPP versions within a year")
    return manifest, frame, pair_paths


def _validate_source_provenance(source):
    """Require the exact source-identity schema emitted by a live POC run."""
    if not isinstance(source, dict) or set(source) != {"inventory", "nmd", "vpp"}:
        raise ValueError("run manifest has malformed source provenance")

    def require_file_identity(value, label, *, kind_optional=False):
        if not isinstance(value, dict):
            raise ValueError(f"{label} identity is not an object")
        if not isinstance(value.get("path"), str) or not value["path"]:
            raise ValueError(f"{label} identity has no path")
        if not isinstance(value.get("bytes"), int) or value["bytes"] < 0:
            raise ValueError(f"{label} identity has invalid size")
        if not isinstance(value.get("sha256"), str) or not re.fullmatch(
            r"[0-9a-f]{64}", value["sha256"]
        ):
            raise ValueError(f"{label} identity has invalid SHA-256")
        if not kind_optional and value.get("kind") not in (None, "file"):
            raise ValueError(f"{label} identity has unexpected kind")

    inventory = source["inventory"]
    if not isinstance(inventory, dict):
        raise ValueError("inventory identity is not an object")
    if inventory.get("kind") == "file":
        require_file_identity(inventory, "inventory", kind_optional=True)
    elif inventory.get("kind") == "directory":
        if (
            not isinstance(inventory.get("path"), str)
            or not isinstance(inventory.get("file_count"), int)
            or not re.fullmatch(
                r"[0-9a-f]{64}", inventory.get("inventory_sha256", "")
            )
        ):
            raise ValueError("inventory directory identity is malformed")
    else:
        raise ValueError("inventory identity has invalid kind")
    require_file_identity(source["nmd"], "NMD")
    vpp = source["vpp"]
    if (
        not isinstance(vpp, dict)
        or vpp.get("source") != "wekeo"
        or vpp.get("available") is not True
        or not isinstance(vpp.get("product_count"), int)
        or vpp["product_count"] <= 0
        or not re.fullmatch(
            r"[0-9a-f]{64}", vpp.get("product_inventory_sha256", "")
        )
        or not vpp.get("versions")
    ):
        raise ValueError("VPP source identity is malformed")
    require_file_identity(vpp.get("index"), "VPP index")


def _recompute_pair_row(run_dir: Path, row: dict) -> dict:
    """Recompute spectral and VPP summaries from the hashed NPZ payload."""
    artifact = (run_dir / row["array_path"]).resolve()
    with np.load(artifact, allow_pickle=False) as data:
        band_names = np.asarray(data["band_names"], dtype=str).tolist()
        vpp_names = np.asarray(data["vpp_band_names"], dtype=str).tolist()
        reference = {
            name: np.asarray(data["reference_bands"])[index]
            for index, name in enumerate(band_names)
        }
        candidate = {
            name: np.asarray(data["candidate_bands"])[index]
            for index, name in enumerate(band_names)
        }
        stored_mask = np.asarray(data["valid_mask"], dtype=bool)
        nmd = np.asarray(data["nmd_label"])
        stable_forest = np.asarray(data["stable_forest_mask"], dtype=bool)
        if not np.array_equal(stable_forest, forest_mask(nmd)):
            raise ValueError("stored forest mask does not match NMD labels")
        pair_mask = common_valid_mask(
            stable_forest,
            np.asarray(data["reference_scl"]),
            np.asarray(data["candidate_scl"]),
            reference,
            candidate,
        )
        if np.any(stored_mask & ~pair_mask):
            raise ValueError("stored common mask includes invalid pair pixels")
        arrays, spectral_metrics = compare_spectral_pair(
            reference, candidate, stored_mask
        )
        for key, recomputed in arrays.items():
            if not np.allclose(
                np.asarray(data[key]), recomputed, rtol=1e-6, atol=1e-7,
                equal_nan=True,
            ):
                raise ValueError(f"stored derived array {key!r} is inconsistent")

        reference_vpp = {
            name: np.asarray(data["reference_vpp"])[index]
            for index, name in enumerate(vpp_names)
        }
        candidate_vpp = {
            name: np.asarray(data["candidate_vpp"])[index]
            for index, name in enumerate(vpp_names)
        }
        vpp_valid = _common_vpp_valid_mask(reference_vpp, candidate_vpp)
        if np.any(stored_mask & ~vpp_valid):
            raise ValueError("stored common mask includes VPP-invalid pixels")
        reference_phase = summarize_vpp_phase(
            reference_vpp, row["reference_date"], mask=stored_mask
        )
        candidate_phase = summarize_vpp_phase(
            candidate_vpp, row["candidate_date"], mask=stored_mask
        )

    recomputed_metrics = {
        **spectral_metrics,
        **{
            f"reference_vpp_{key}": value
            for key, value in reference_phase.items()
        },
        **{
            f"candidate_vpp_{key}": value
            for key, value in candidate_phase.items()
        },
        "vpp_sos_shift_days": (
            candidate_phase["sos_doy_median"] - reference_phase["sos_doy_median"]
        ),
        "vpp_eos_shift_days": (
            candidate_phase["eos_doy_median"] - reference_phase["eos_doy_median"]
        ),
        "vpp_midpoint_proxy_shift_days": (
            candidate_phase["season_midpoint_proxy_doy"]
            - reference_phase["season_midpoint_proxy_doy"]
        ),
    }
    for key, value in recomputed_metrics.items():
        if key not in row or not np.isclose(
            float(row[key]), float(value), rtol=1e-6, atol=1e-7
        ):
            raise ValueError(f"pair manifest metric {key!r} is inconsistent")
    reference_version = row["reference_vpp_provenance"]["versions"][0]
    candidate_version = row["candidate_vpp_provenance"]["versions"][0]
    return {
        **row,
        **recomputed_metrics,
        "reference_vpp_version": int(reference_version),
        "candidate_vpp_version": int(candidate_version),
        "vpp_version_pair": f"V{reference_version}->V{candidate_version}",
    }


def _verify_exact_group_mask(run_dir: Path, rows: list[dict]) -> None:
    """Rebuild the exact two-strategy S2/VPP intersection from both NPZs."""
    if len(rows) != len(SELECTION_METHODS):
        raise ValueError("fetch group does not contain exactly two strategies")
    artifacts = []
    for row in rows:
        path = (run_dir / row["array_path"]).resolve()
        with np.load(path, allow_pickle=False) as data:
            band_names = np.asarray(data["band_names"], dtype=str).tolist()
            vpp_names = np.asarray(data["vpp_band_names"], dtype=str).tolist()
            artifacts.append({
                "method": row["selection_method"],
                "nmd": np.asarray(data["nmd_label"]).copy(),
                "forest": np.asarray(data["stable_forest_mask"], dtype=bool).copy(),
                "reference_scl": np.asarray(data["reference_scl"]).copy(),
                "candidate_scl": np.asarray(data["candidate_scl"]).copy(),
                "reference_bands": {
                    name: np.asarray(data["reference_bands"])[index].copy()
                    for index, name in enumerate(band_names)
                },
                "candidate_bands": {
                    name: np.asarray(data["candidate_bands"])[index].copy()
                    for index, name in enumerate(band_names)
                },
                "reference_vpp": {
                    name: np.asarray(data["reference_vpp"])[index].copy()
                    for index, name in enumerate(vpp_names)
                },
                "candidate_vpp": {
                    name: np.asarray(data["candidate_vpp"])[index].copy()
                    for index, name in enumerate(vpp_names)
                },
                "stored_mask": np.asarray(data["valid_mask"], dtype=bool).copy(),
            })
    if {item["method"] for item in artifacts} != set(SELECTION_METHODS):
        raise ValueError("fetch group has the wrong strategy methods")
    anchor = artifacts[0]
    if not np.array_equal(anchor["forest"], forest_mask(anchor["nmd"])):
        raise ValueError("fetch group forest mask does not match NMD labels")
    for item in artifacts[1:]:
        if not np.array_equal(item["nmd"], anchor["nmd"]):
            raise ValueError("strategies carry different NMD rasters")
        if not np.array_equal(item["forest"], anchor["forest"]):
            raise ValueError("strategies carry different forest masks")
        if not np.array_equal(item["reference_scl"], anchor["reference_scl"]):
            raise ValueError("strategies carry different reference SCL rasters")
        for collection in ("reference_bands", "reference_vpp", "candidate_vpp"):
            if any(
                not np.array_equal(item[collection][name], anchor[collection][name])
                for name in item[collection]
            ):
                raise ValueError(f"strategies carry different {collection}")

    expected = anchor["forest"].copy()
    for item in artifacts:
        expected &= common_valid_mask(
            anchor["forest"],
            item["reference_scl"],
            item["candidate_scl"],
            item["reference_bands"],
            item["candidate_bands"],
        )
        expected &= _common_vpp_valid_mask(
            item["reference_vpp"], item["candidate_vpp"]
        )
    for item in artifacts:
        if not np.array_equal(item["stored_mask"], expected):
            raise ValueError(
                f"{item['method']} mask is not the exact two-strategy intersection"
            )


def add_vpp_phase_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive method-dependent acquisition alignment on the VPP timeline."""
    output = frame.copy()
    component_columns = []
    for label, source in VPP_PHASE_COMPONENTS.items():
        reference = f"reference_vpp_{source}"
        candidate = f"candidate_vpp_{source}"
        if reference not in output or candidate not in output:
            raise ValueError(f"missing VPP phase columns {reference!r}/{candidate!r}")
        destination = f"vpp_phase_{label}_alignment_error_days"
        output[destination] = (output[candidate] - output[reference]).abs()
        component_columns.append(destination)
    output["vpp_phase_alignment_mae_days"] = output[component_columns].mean(axis=1)
    for name in ("sos", "eos", "midpoint_proxy"):
        source = f"vpp_{name}_shift_days"
        if source in output:
            output[f"{source.removesuffix('_days')}_absolute_days"] = output[source].abs()
    return output


def paired_strategy_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Align closest-date and meteorology rows for paired inference."""
    index = ["tile_id", "candidate_year"]
    closest = frame[frame["selection_method"] == "closest_date"].set_index(index)
    meteorology = frame[frame["selection_method"] == "meteorology"].set_index(index)
    if not closest.index.is_unique or not meteorology.index.is_unique:
        raise ValueError("strategy rows are not unique by tile/year")
    if set(closest.index) != set(meteorology.index):
        raise ValueError("closest-date and meteorology strategies do not cover the same pairs")
    meteorology = meteorology.loc[closest.index]
    output = closest.index.to_frame(index=False)
    copied = {
        "candidate_date",
        "calendar_displacement_days",
        "fetch_group_id",
        "common_valid_mask_sha256",
        *REPORT_METRICS,
    }
    for column in sorted(copied):
        if column not in closest or column not in meteorology:
            raise ValueError(f"missing paired metric {column!r}")
        output[f"{column}_closest_date"] = closest[column].to_numpy()
        output[f"{column}_meteorology"] = meteorology[column].to_numpy()
    if not (
        output["fetch_group_id_closest_date"]
        == output["fetch_group_id_meteorology"]
    ).all():
        raise ValueError("paired strategies use different fetch/coreg groups")
    if not (
        output["common_valid_mask_sha256_closest_date"]
        == output["common_valid_mask_sha256_meteorology"]
    ).all():
        raise ValueError("paired strategies use different pixel masks")
    for column in (
        "reference_vpp_version",
        "candidate_vpp_version",
        "vpp_version_pair",
    ):
        if column not in closest or column not in meteorology:
            raise ValueError(f"missing paired VPP provenance {column!r}")
        if not (closest[column] == meteorology[column]).all():
            raise ValueError(f"paired strategies use different {column}")
        output[column] = closest[column].to_numpy()
    for metric in EFFICACY_METRICS:
        delta = (
            output[f"{metric}_meteorology"]
            - output[f"{metric}_closest_date"]
        )
        output[f"{metric}_delta"] = delta
        output[f"{metric}_meteorology_wins"] = delta < 0
    return output


def strategy_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in SELECTION_METHODS:
        subset = frame[frame["selection_method"] == method]
        for metric, label in REPORT_METRICS.items():
            values = pd.to_numeric(subset[metric], errors="raise").to_numpy(float)
            rows.append({
                "selection_method": method,
                "metric": metric,
                "label": label,
                "role": (
                    "efficacy" if metric in EFFICACY_METRICS else "diagnostic"
                ),
                "n": len(values),
                "cluster_n": int(subset["tile_id"].nunique()),
                "year_n": int(subset["candidate_year"].nunique()),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
            })
    return pd.DataFrame(rows)


def paired_effects(
    paired: pd.DataFrame,
    *,
    bootstrap_samples: int,
) -> pd.DataFrame:
    rows = []
    for offset, (metric, label) in enumerate(EFFICACY_METRICS.items()):
        delta_column = f"{metric}_delta"
        delta = paired[delta_column].to_numpy(float)
        low, high = _cluster_bootstrap_median_ci(
            paired,
            value_column=delta_column,
            cluster_column="tile_id",
            samples=bootstrap_samples,
            seed=20260820 + offset,
        )
        rows.append({
            "metric": metric,
            "label": label,
            "paired_n": len(delta),
            "cluster_n": int(paired["tile_id"].nunique()),
            "year_n": int(paired["candidate_year"].nunique()),
            "bootstrap_unit": "tile_id",
            "median_delta_meteorology_minus_closest": float(np.median(delta)),
            "mean_delta_meteorology_minus_closest": float(np.mean(delta)),
            "median_delta_ci95_low": low,
            "median_delta_ci95_high": high,
            "meteorology_win_rate": float(np.mean(delta < 0)),
            "tie_rate": float(np.mean(delta == 0)),
        })
    return pd.DataFrame(rows)


def version_stratified_effects(
    paired: pd.DataFrame,
    *,
    bootstrap_samples: int,
) -> pd.DataFrame:
    """Report paired effects separately for each VPP processing-version pair."""
    rows = []
    for version_pair, subset in paired.groupby("vpp_version_pair", sort=True):
        effects = paired_effects(
            subset.reset_index(drop=True),
            bootstrap_samples=bootstrap_samples,
        )
        effects.insert(0, "vpp_version_pair", version_pair)
        rows.append(effects)
    if not rows:
        raise ValueError("no VPP version strata are available")
    return pd.concat(rows, ignore_index=True)


def _cluster_bootstrap_median_ci(
    frame: pd.DataFrame,
    *,
    value_column: str,
    cluster_column: str,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if value_column not in frame or cluster_column not in frame:
        raise ValueError("cluster bootstrap columns are missing")
    clusters = [
        group[value_column].to_numpy(float)
        for _, group in frame.groupby(cluster_column, sort=True)
    ]
    values = np.concatenate(clusters) if clusters else np.asarray([])
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be non-empty and finite")
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    rng = np.random.default_rng(seed)
    medians = np.empty(samples, dtype=float)
    for index in range(samples):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        medians[index] = np.median(
            np.concatenate([clusters[item] for item in sampled])
        )
    low, high = np.quantile(medians, [0.025, 0.975])
    return float(low), float(high)


def render_analysis_figures(
    frame: pd.DataFrame,
    paired: pd.DataFrame,
    output_dir: Path,
) -> None:
    _render_strategy_boxplots(frame, output_dir / "strategy_boxplots.png")
    _render_vpp_phase_comparison(paired, output_dir / "vpp_phase_comparison.png")
    _render_meteo_vs_ndvi(frame, output_dir / "meteo_vs_ndvi.png")


def _render_strategy_boxplots(frame: pd.DataFrame, path: Path) -> None:
    metrics = [
        "ndvi_absolute_difference_median",
        "spectral_angle_median_rad",
        "vpp_phase_alignment_mae_days",
    ]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    colors = ["#6b7280", "#1A4338"]
    for axis, metric in zip(axes, metrics):
        values = [
            frame.loc[frame["selection_method"] == method, metric].to_numpy(float)
            for method in SELECTION_METHODS
        ]
        plot = axis.boxplot(values, patch_artist=True, tick_labels=["Date", "Meteo"])
        for patch, color in zip(plot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
        axis.set_title(EFFICACY_METRICS[metric])
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def _render_vpp_phase_comparison(paired: pd.DataFrame, path: Path) -> None:
    metric = "vpp_phase_alignment_mae_days"
    x = paired[f"{metric}_closest_date"].to_numpy(float)
    y = paired[f"{metric}_meteorology"].to_numpy(float)
    upper = max(float(np.max(x)), float(np.max(y)), 1.0)
    figure, axis = plt.subplots(figsize=(6, 5.5))
    axis.scatter(x, y, color="#1A4338", alpha=0.75)
    axis.plot([0, upper], [0, upper], color="#9ca3af", linestyle="--")
    axis.set(xlabel="Closest date (days)", ylabel="Meteorology (days)",
             title="VPP phase alignment error")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def _render_meteo_vs_ndvi(frame: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 5.2))
    styles = {
        "closest_date": ("#6b7280", "Closest date"),
        "meteorology": ("#1A4338", "Meteorology"),
    }
    for method, (color, label) in styles.items():
        subset = frame[frame["selection_method"] == method]
        axis.scatter(
            subset["meteorology_distance"],
            subset["ndvi_absolute_difference_median"],
            color=color,
            alpha=0.7,
            label=label,
        )
    axis.set(xlabel="Meteorological distance", ylabel="Median |NDVI difference|",
             title="Meteorology and spectral similarity")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def _combined_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
