#!/usr/bin/env python3
"""Run the meteorology-matched forest Sentinel-2 proof of concept."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.metadata
import json
import os
import re
import subprocess
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metafilter import fetch_daily_meteorology
from imint.fetch import _connect as connect_des
from imint.experiments.meteo_analog_forest import (
    COMPARISON_BANDS,
    DateMatch,
    TileCandidate,
    classified_fraction,
    common_valid_mask,
    compare_spectral_pair,
    forest_fraction,
    forest_mask,
    frame_bands,
    rank_cloud_valid_comparisons,
    summarize_vpp_phase,
    validate_fetch_result,
    write_manifest,
)
from imint.training.cdse_vpp import fetch_vpp_tiles
from imint.training.fetch_spectral import fetch_tile_spectral
from imint.training.optimal_fetch import optimal_fetch_dates
from imint.training.tile_config import TileConfig
from imint.training.tile_fetch import bbox_3006_to_wgs84, fetch_nmd_label_local
from imint.training.wekeo_vpp import _bbox_3006_to_4326, _bounds_overlap


SELECTION_METHODS = ("closest_date", "meteorology")
VPP_BANDS = ("sosd", "eosd", "length", "maxv", "minv")
VPP_SEASON = 1
GRID_CRS = "EPSG:3006"
GRID_RESOLUTION_M = 10.0
MIN_COMMON_FOREST_PIXELS_512 = 10_000
MIN_COMMON_FOREST_FRACTION = 0.25
PLANNING_SCHEMA_VERSION = 4
DATASET_SCHEMA_VERSION = 4
_VPP_FILENAME_RE = re.compile(
    r"^VPP_(?P<year>\d{4})_S2_(?P<tile>[0-9A-Z]+)-0?\d+m_"
    r"V(?P<version>\d+)_s(?P<season>\d)_(?P<metric>[A-Z]+)\.tif$"
)
_PAIR_DERIVED_ARRAY_KEYS = frozenset({
    "valid_mask",
    "ndvi_reference",
    "ndvi_candidate",
    "ndvi_difference",
    "spectral_angle_rad",
    *(
        key
        for band in COMPARISON_BANDS
        for key in (
            f"{band.lower()}_difference",
            f"{band.lower()}_relative_difference",
        )
    ),
})
_PAIR_REQUIRED_ARRAY_KEYS = frozenset({
    "dataset_schema_version",
    "band_names",
    "reference_bands",
    "candidate_bands",
    "reference_scl",
    "candidate_scl",
    "nmd_label",
    "stable_forest_mask",
    "vpp_band_names",
    "reference_vpp",
    "candidate_vpp",
    "tile_id",
    "tile_size_px",
    "plan_fingerprint",
    "run_fingerprint",
    "fetch_group_id",
    "fetch_source",
    "fetch_requested_dates",
    "fetch_returned_dates",
    "fetch_temporal_mask",
    "fetch_frame_valid_fraction",
    "fetch_coreg_shifts",
    "fetch_coreg_ref_frame",
    "fetch_coreg_m2",
    "fetch_coreg_n_aligned",
    "fetch_coreg_max_shift",
    "fetch_coreg_anchor_valid_fraction",
    "fetch_bbox_epsg3006",
    "fetch_center_epsg3006",
    "fetch_tile_size_px",
    "fetch_num_frames",
    "fetch_num_bands",
    "fetch_spectral_shape",
    "fetch_scl_shape",
    "fetch_attempt_sha256",
    "candidate_slot",
    "common_valid_mask_sha256",
    "common_valid_pixel_count",
    "common_valid_fraction_of_forest",
    "reference_vpp_provenance_json",
    "candidate_vpp_provenance_json",
    "bbox_epsg3006",
    "center_epsg3006",
    "reference_date",
    "candidate_date",
    "reference_year",
    "candidate_year",
    "selection_method",
    "grid_crs",
    "grid_transform",
    *_PAIR_DERIVED_ARRAY_KEYS,
})
_FETCH_GROUP_SHARED_KEYS = (
    "fetch_group_id",
    "fetch_attempt_sha256",
    "fetch_source",
    "fetch_returned_dates",
    "fetch_temporal_mask",
    "fetch_frame_valid_fraction",
    "fetch_coreg_shifts",
    "fetch_coreg_ref_frame",
    "fetch_coreg_m2",
    "fetch_coreg_n_aligned",
    "fetch_coreg_max_shift",
    "fetch_coreg_anchor_valid_fraction",
    "fetch_bbox_epsg3006",
    "fetch_center_epsg3006",
    "fetch_tile_size_px",
    "fetch_num_frames",
    "fetch_num_bands",
    "fetch_spectral_shape",
    "fetch_scl_shape",
    "common_valid_mask_sha256",
    "common_valid_pixel_count",
    "common_valid_fraction_of_forest",
    "reference_vpp_provenance",
    "candidate_vpp_provenance",
    "grid_crs",
    "grid_transform",
)
_FILE_DIGEST_CACHE: dict[tuple[str, int, int, int], str] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True,
                        help="CSV or directory of NPZ tiles carrying easting/northing")
    parser.add_argument("--nmd-raster", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/meteo_analog_forest_poc"))
    parser.add_argument("--reference-year", type=int, default=2019)
    parser.add_argument("--candidate-years", type=int, nargs="+",
                        default=[2020, 2021, 2022, 2023, 2024])
    parser.add_argument("--window", nargs=2, default=["05-15", "06-15"],
                        metavar=("START_MM_DD", "END_MM_DD"))
    parser.add_argument("--tiles", type=int, default=10)
    parser.add_argument("--size-px", type=int, default=512)
    parser.add_argument("--min-forest-fraction", type=float, default=0.80)
    parser.add_argument("--candidates-per-stratum", type=int, default=100)
    parser.add_argument("--fetch-source", choices=("des", "cdse-openeo"), default="des")
    parser.add_argument("--max-aoi-cloud", type=float, default=0.10)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--select-only", action="store_true",
                      help="select forest tiles locally (default; no network)")
    mode.add_argument("--plan-network", action="store_true",
                      help="select tiles and perform Open-Meteo/SCL date planning")
    mode.add_argument("--execute-fetch", action="store_true",
                      help="perform live planning plus S2/VPP fetch")
    parser.add_argument("--run-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_years(args.reference_year, args.candidate_years)
    if args.execute_fetch:
        _validate_live_environment(args)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = _safe_run_dir(args.output_dir, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"

    tile_config = TileConfig(size_px=args.size_px)
    tiles_path = run_dir / "tiles.csv"
    if tiles_path.exists():
        selected = _load_selected_tiles(tiles_path)
    else:
        centers = load_inventory(args.inventory)
        selected = select_forest_tiles(
            centers,
            tile=tile_config,
            nmd_raster=args.nmd_raster,
            count=args.tiles,
            min_fraction=args.min_forest_fraction,
            candidates_per_stratum=args.candidates_per_stratum,
        )
        _write_rows(tiles_path, [asdict(tile) for tile in selected])
    _validate_selected_tiles(selected, args)
    manifest = _load_or_create_manifest(
        manifest_path,
        args,
        run_id,
        selected_tiles=selected,
        selected_tiles_path=tiles_path,
    )
    manifest["stage"] = "tiles_selected"
    manifest["tiles"] = [asdict(tile) for tile in selected]
    write_manifest(manifest_path, manifest)

    if not (args.plan_network or args.execute_fetch):
        print(f"Offline tile selection complete: {run_dir}")
        return 0

    all_matches = []
    tile_context = {}
    plans_dir = run_dir / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    for tile_index, selected_tile in enumerate(selected):
        tile_id = f"tile_{tile_index:02d}_{selected_tile.easting}_{selected_tile.northing}"
        plan_path = plans_dir / f"{tile_id}.json"
        expected_inputs = _planning_inputs(selected_tile, args)
        context = None
        if plan_path.exists():
            try:
                context = _load_context(plan_path)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                context = None
        if not _plan_is_current(context, expected_inputs):
            try:
                context = plan_tile_dates(
                    selected_tile, args, run_dir / "cache" / tile_id
                )
            except Exception as error:
                context = {
                    "matches": [],
                    "planning_inputs": expected_inputs,
                    "planning_schema_version": PLANNING_SCHEMA_VERSION,
                    "planning_error": f"{type(error).__name__}: {error}",
                }
            write_manifest(plan_path, _serialize_context(context))
        if context.get("planning_error"):
            all_matches.append({
                "tile_id": tile_id,
                "status": "planning_failed",
                "reason": context["planning_error"],
            })
        tile_context[tile_id] = context
        for match in context["matches"]:
            all_matches.append({"tile_id": tile_id, "status": "matched", **asdict(match)})
        matched_pairs = {
            (match.candidate_year, match.selection_method)
            for match in context["matches"]
        }
        for year in args.candidate_years:
            for selection_method in SELECTION_METHODS:
                if (year, selection_method) not in matched_pairs:
                    all_matches.append({
                        "tile_id": tile_id,
                        "status": "missing",
                        "candidate_year": year,
                        "selection_method": selection_method,
                        "reason": context.get(
                            "planning_error", "no cloud-valid comparison date"
                        ),
                    })
    _write_rows(run_dir / "matches.csv", all_matches)
    manifest["stage"] = "dates_planned"
    manifest["planned_tiles"] = sorted(tile_context)
    write_manifest(manifest_path, manifest)

    if not args.execute_fetch:
        print(f"Network plan complete: {run_dir}")
        return 0

    pairs_dir = run_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    for tile_id, selected_tile in zip(tile_context, selected):
        context = tile_context[tile_id]
        for candidate_year in args.candidate_years:
            group_matches = _year_matches(context, candidate_year)
            if {match.selection_method for match in group_matches} != set(
                SELECTION_METHODS
            ):
                continue
            expected = {
                match.selection_method: _pair_identity(
                    tile_id,
                    selected_tile,
                    context,
                    group_matches,
                    match,
                    size_px=args.size_px,
                    fetch_source=args.fetch_source,
                    run_fingerprint=manifest["run_fingerprint"],
                )
                for match in group_matches
            }
            pair_paths = {
                match.selection_method: _pair_manifest_path(
                    pairs_dir, tile_id, match
                )
                for match in group_matches
            }
            if _fetch_group_succeeded(pair_paths, expected):
                continue
            group_context = dict(context)
            group_context["matches"] = group_matches
            try:
                tile_summaries, _ = fetch_and_compare_tile(
                    tile_id,
                    selected_tile,
                    group_context,
                    args,
                    run_dir,
                    run_fingerprint=manifest["run_fingerprint"],
                )
            except Exception as error:
                tile_summaries = [
                    {
                        **expected[match.selection_method],
                        "status": "failed",
                        "reason": f"{type(error).__name__}: {error}",
                    }
                    for match in group_matches
                ]
            for match, row in zip(group_matches, tile_summaries):
                write_manifest(
                    _pair_manifest_path(pairs_dir, tile_id, match), row
                )
        write_manifest(manifest_path, manifest)
    tile_ids = [
        f"tile_{index:02d}_{tile.easting}_{tile.northing}"
        for index, tile in enumerate(selected)
    ]
    expected_pair_names = _expected_pair_manifest_names(
        tile_ids, args.candidate_years
    )
    actual_pair_paths = {
        path.name: path for path in sorted(pairs_dir.glob("*.json"))
    }
    missing_pair_names = sorted(expected_pair_names - set(actual_pair_paths))
    unexpected_pair_names = sorted(set(actual_pair_paths) - expected_pair_names)
    summaries = [
        json.loads(actual_pair_paths[name].read_text())
        for name in sorted(expected_pair_names & set(actual_pair_paths))
    ]
    figure_paths = sorted((run_dir / "figures").glob("*.png"))
    _write_rows(run_dir / "summary.csv", summaries)
    write_report(run_dir / "report.html", summaries, figure_paths)
    fetch_failed = sum(row.get("status") != "ok" for row in summaries)
    planning_missing = sum(row.get("status") == "missing" for row in all_matches)
    requested = len(selected) * len(args.candidate_years) * len(SELECTION_METHODS)
    successful = sum(row.get("status") == "ok" for row in summaries)
    rows_by_group = {}
    for row in summaries:
        rows_by_group.setdefault(
            (row.get("tile_id"), row.get("candidate_year")), []
        ).append(row)
    inconsistent_groups = sum(
        not _group_rows_consistent(
            rows_by_group.get((tile_id, int(year)), [])
        )
        for tile_id in tile_ids
        for year in args.candidate_years
    )
    run_years = [args.reference_year, *args.candidate_years]
    version_validation_error = None
    try:
        versions_by_year = _vpp_versions_by_year(summaries, run_years)
        inconsistent_version_years = sum(
            len(versions_by_year[str(year)]) != 1 for year in run_years
        )
    except (KeyError, TypeError, ValueError) as error:
        versions_by_year = {str(int(year)): [] for year in run_years}
        inconsistent_version_years = len(run_years)
        version_validation_error = f"{type(error).__name__}: {error}"
    failed = bool(
        fetch_failed
        or planning_missing
        or missing_pair_names
        or unexpected_pair_names
        or inconsistent_groups
        or inconsistent_version_years
        or successful != requested
    )
    manifest["stage"] = "complete_with_failures" if failed else "complete"
    manifest["pair_counts"] = {
        "requested": requested,
        "successful": successful,
        "fetch_failed": fetch_failed,
        "planning_missing": planning_missing,
        "pair_manifests_missing": len(missing_pair_names),
        "pair_manifests_unexpected": len(unexpected_pair_names),
        "fetch_groups_inconsistent": inconsistent_groups,
        "vpp_version_inconsistent_years": inconsistent_version_years,
    }
    manifest["vpp_versions_by_year"] = versions_by_year
    if version_validation_error:
        manifest["vpp_version_validation_error"] = version_validation_error
    else:
        manifest.pop("vpp_version_validation_error", None)
    manifest["pair_manifest_inventory"] = {
        "expected_sha256": _canonical_sha256(sorted(expected_pair_names)),
        "missing": missing_pair_names,
        "unexpected": unexpected_pair_names,
    }
    write_manifest(manifest_path, manifest)
    print(f"POC complete: {run_dir / 'report.html'}")
    return 1 if failed else 0


def load_inventory(path: Path) -> list[tuple[int, int, str]]:
    """Load unique EPSG:3006 centers from a CSV or NPZ directory."""
    records = {}
    if path.is_file():
        frame = pd.read_csv(path)
        required = {"easting", "northing"}
        if not required <= set(frame.columns):
            raise ValueError(f"inventory CSV must contain {sorted(required)}")
        for row in frame.itertuples():
            center = (int(row.easting), int(row.northing))
            records[center] = str(path)
    elif path.is_dir():
        for npz_path in sorted(path.glob("*.npz")):
            try:
                with np.load(npz_path, allow_pickle=False) as data:
                    if "easting" not in data or "northing" not in data:
                        continue
                    center = (int(data["easting"]), int(data["northing"]))
                    records.setdefault(center, str(npz_path))
            except (OSError, ValueError):
                continue
    else:
        raise FileNotFoundError(path)
    if not records:
        raise ValueError(f"no tile centers found in {path}")
    return [(east, north, source) for (east, north), source in sorted(records.items())]


def select_forest_tiles(
    centers: list[tuple[int, int, str]],
    *,
    tile: TileConfig,
    nmd_raster: Path,
    count: int,
    min_fraction: float,
    candidates_per_stratum: int,
) -> list[TileCandidate]:
    """Evaluate deterministic east-west samples within northing strata."""
    if not nmd_raster.exists():
        raise FileNotFoundError(
            f"local NMD raster required for offline selection: {nmd_raster}"
        )
    ordered = sorted(centers, key=lambda item: (item[1], item[0]))
    strata = np.array_split(np.array(ordered, dtype=object), count)
    selected = []
    missing = []
    for stratum_index, stratum in enumerate(strata):
        evaluated = []
        rows = sorted(stratum.tolist(), key=lambda item: item[0])
        if len(rows) > candidates_per_stratum:
            indexes = np.linspace(0, len(rows) - 1, candidates_per_stratum, dtype=int)
            rows = [rows[index] for index in sorted(set(indexes.tolist()))]
        for east, north, source in rows:
            bbox = tile.bbox_from_center(int(east), int(north))
            label = fetch_nmd_label_local(
                bbox,
                tile,
                str(nmd_raster),
                allow_remote_fallback=False,
            )
            if label is None:
                continue
            evaluated.append(
                TileCandidate(
                    easting=int(east),
                    northing=int(north),
                    forest_fraction=forest_fraction(label),
                    classified_fraction=classified_fraction(label),
                    source=str(source),
                )
            )
        eligible = [item for item in evaluated if item.forest_fraction >= min_fraction]
        if not eligible:
            missing.append(stratum_index)
            continue
        selected.append(max(eligible, key=lambda item: (item.forest_fraction, -item.easting)))
    if missing:
        raise ValueError(f"no >= {min_fraction:.2f} forest tile in strata {missing}")
    return selected


def plan_tile_dates(tile: TileCandidate, args, cache_dir: Path) -> dict:
    tile_config = TileConfig(size_px=args.size_px)
    bbox_3006 = tile_config.bbox_from_center(tile.easting, tile.northing)
    bbox_wgs84 = bbox_3006_to_wgs84(bbox_3006)
    years = [args.reference_year, *args.candidate_years]
    frames = {}
    meteorology_by_year = {}
    valid_dates = {}
    fetch_plans = {}
    for year in years:
        meteorology = fetch_daily_meteorology(
            bbox_wgs84=bbox_wgs84,
            date_start=f"{year}-04-15",
            date_end=f"{year}-{args.window[1]}",
            cache_dir=cache_dir / "meteorology" / str(year),
        )
        frames[year] = meteorology.frame
        meteorology_by_year[str(year)] = _frame_records(meteorology.frame)
        plan = optimal_fetch_dates(
            bbox_wgs84,
            f"{year}-{args.window[0]}",
            f"{year}-{args.window[1]}",
            mode="scl_only",
            max_aoi_cloud=args.max_aoi_cloud,
            scl_backend=(
                "cdse" if args.fetch_source == "cdse-openeo" else "des"
            ),
            require_complete_scl=True,
        )
        _validate_scl_plan(plan, year)
        valid_dates[year] = plan.dates
        fetch_plans[year] = {
            "mode": getattr(plan, "mode", "scl_only"),
            "dates": plan.dates,
            "n_candidates_after": plan.n_candidates_after,
            "elapsed_s": plan.elapsed_s,
            "notes": plan.notes,
            "scl_gate": getattr(plan, "scl_gate", {}),
            "scl_screen_complete": getattr(
                plan, "scl_screen_complete", None
            ),
            "scl_thresholds": getattr(plan, "scl_thresholds", {}),
        }
    matches = rank_cloud_valid_comparisons(
        frames,
        valid_dates,
        reference_year=args.reference_year,
    )
    context = {
        "bbox_3006": bbox_3006,
        "bbox_wgs84": bbox_wgs84,
        "reference_date": (
            matches[0].reference_date if matches else None
        ),
        "matches": matches,
        "fetch_plans": fetch_plans,
        "meteorology_by_year": meteorology_by_year,
        "planning_inputs": _planning_inputs(tile, args),
        "planning_schema_version": PLANNING_SCHEMA_VERSION,
    }
    context["plan_fingerprint"] = _plan_fingerprint(context)
    return context


def _validate_scl_plan(plan, year):
    """Require a complete SCL screen and internally consistent gate table."""

    def field(name, default=None):
        return (
            plan.get(name, default)
            if isinstance(plan, dict)
            else getattr(plan, name, default)
        )

    if field("scl_screen_complete") is not True:
        raise ValueError(f"SCL plan {year} is not a complete screen")
    thresholds = field("scl_thresholds", {})
    expected_thresholds = {
        "max_aoi_cloud", "max_aoi_snow", "min_aoi_coverage",
    }
    if not isinstance(thresholds, dict) or set(thresholds) != expected_thresholds:
        raise ValueError(f"SCL plan {year} has malformed thresholds")
    threshold_values = {
        name: float(value) for name, value in thresholds.items()
    }
    if not all(
        np.isfinite(value) and 0 <= value <= 1
        for value in threshold_values.values()
    ):
        raise ValueError(f"SCL plan {year} has invalid thresholds")

    gates = field("scl_gate", {})
    dates = field("dates", [])
    if (
        not isinstance(gates, dict)
        or not isinstance(dates, list)
        or dates != sorted(set(dates))
    ):
        raise ValueError(f"SCL plan {year} has malformed dates or gates")
    accepted_dates = []
    gate_fields = {
        "cloud_fraction", "snow_fraction", "coverage_fraction", "accepted",
    }
    for observed_date, gate in sorted(gates.items()):
        try:
            timestamp = pd.Timestamp(observed_date)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"SCL plan {year} has invalid date {observed_date!r}"
            ) from error
        if timestamp.strftime("%Y-%m-%d") != observed_date or timestamp.year != int(year):
            raise ValueError(
                f"SCL plan {year} has out-of-year date {observed_date!r}"
            )
        if not isinstance(gate, dict) or set(gate) != gate_fields:
            raise ValueError(
                f"SCL plan {year} has malformed gate for {observed_date}"
            )
        fractions = {
            name: float(gate[name])
            for name in (
                "cloud_fraction", "snow_fraction", "coverage_fraction",
            )
        }
        if not all(
            np.isfinite(value) and 0 <= value <= 1
            for value in fractions.values()
        ):
            raise ValueError(
                f"SCL plan {year} has invalid fractions for {observed_date}"
            )
        accepted = (
            fractions["cloud_fraction"]
            <= threshold_values["max_aoi_cloud"]
            and fractions["snow_fraction"]
            <= threshold_values["max_aoi_snow"]
            and fractions["coverage_fraction"]
            >= threshold_values["min_aoi_coverage"]
        )
        if not isinstance(gate["accepted"], bool) or gate["accepted"] != accepted:
            raise ValueError(
                f"SCL plan {year} has inconsistent gate for {observed_date}"
            )
        if accepted:
            accepted_dates.append(observed_date)
    if dates != accepted_dates:
        raise ValueError(
            f"SCL plan {year} dates do not equal its accepted gate dates"
        )


def _year_matches(context, candidate_year):
    matches = [
        match
        for match in context.get("matches", [])
        if match.candidate_year == candidate_year
    ]
    by_method = {}
    for match in matches:
        if match.selection_method in by_method:
            raise ValueError(
                f"duplicate {match.selection_method!r} match for {candidate_year}"
            )
        by_method[match.selection_method] = match
    return [by_method[method] for method in SELECTION_METHODS if method in by_method]


def _group_dates(reference_date, matches):
    ordered = [reference_date, *(match.candidate_date for match in matches)]
    unique = list(dict.fromkeys(ordered))
    dates = {slot: value for slot, value in enumerate(unique)}
    return dates, {value: slot for slot, value in dates.items()}


def _pair_identity(
    tile_id,
    selected_tile,
    context,
    group_matches,
    match,
    *,
    size_px,
    fetch_source,
    run_fingerprint,
):
    dates, slot_by_date = _group_dates(match.reference_date, group_matches)
    bbox = context["bbox_3006"]
    pixel_width = (float(bbox["east"]) - float(bbox["west"])) / int(size_px)
    pixel_height = (float(bbox["north"]) - float(bbox["south"])) / int(size_px)
    if not (
        np.isclose(pixel_width, GRID_RESOLUTION_M)
        and np.isclose(pixel_height, GRID_RESOLUTION_M)
    ):
        raise ValueError(
            f"comparison grid is {pixel_width}x{pixel_height} m, expected "
            f"{GRID_RESOLUTION_M} m"
        )
    grid_transform = [
        pixel_width,
        0.0,
        float(bbox["west"]),
        0.0,
        -pixel_height,
        float(bbox["north"]),
    ]
    fetch_group_id = _canonical_sha256({
        "tile_id": tile_id,
        "candidate_year": match.candidate_year,
        "dates": dates,
        "plan_fingerprint": context["plan_fingerprint"],
        "run_fingerprint": run_fingerprint,
    })[:24]
    return {
        **asdict(match),
        "tile_id": tile_id,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "plan_fingerprint": context["plan_fingerprint"],
        "run_fingerprint": run_fingerprint,
        "fetch_group_id": fetch_group_id,
        "fetch_source": fetch_source,
        "fetch_requested_dates": [dates[index] for index in sorted(dates)],
        "candidate_slot": slot_by_date[match.candidate_date],
        "tile_size_px": int(size_px),
        "easting": int(selected_tile.easting),
        "northing": int(selected_tile.northing),
        "forest_fraction": float(selected_tile.forest_fraction),
        "classified_fraction": float(selected_tile.classified_fraction),
        "bbox_epsg3006": [
            float(bbox[edge]) for edge in ("west", "south", "east", "north")
        ],
        "band_names": list(COMPARISON_BANDS),
        "vpp_band_names": list(VPP_BANDS),
        "reference_year": int(match.reference_year),
        "reference_date": match.reference_date,
        "candidate_year": int(match.candidate_year),
        "candidate_date": match.candidate_date,
        "selection_method": match.selection_method,
        "grid_crs": GRID_CRS,
        "grid_transform": grid_transform,
    }


def fetch_and_compare_tile(
    tile_id,
    selected_tile,
    context,
    args,
    run_dir,
    *,
    run_fingerprint,
):
    matches = context["matches"]
    if not matches:
        return [], []
    if len({match.candidate_year for match in matches}) != 1:
        raise ValueError("one fetch/coreg group must contain exactly one candidate year")
    matches = _year_matches(
        {"matches": matches}, matches[0].candidate_year
    )
    reference_date = context.get("reference_date")
    if not reference_date:
        raise ValueError("tile context has no independent reference date")
    if any(match.reference_date != reference_date for match in matches):
        raise ValueError("tile matches disagree with the planned reference date")
    dates, slot_by_date = _group_dates(reference_date, matches)
    identities = {
        match.selection_method: _pair_identity(
            tile_id,
            selected_tile,
            context,
            matches,
            match,
            size_px=args.size_px,
            fetch_source=args.fetch_source,
            run_fingerprint=run_fingerprint,
        )
        for match in matches
    }
    bbox = context["bbox_3006"]
    result = fetch_tile_spectral(
        (selected_tile.easting, selected_tile.northing),
        tile=TileConfig(size_px=args.size_px),
        dates=dates,
        n_frames=len(dates),
        backend=args.fetch_source,
        coregister=True,
        with_scl=True,
    )
    if result is None:
        return [
            {
                **identities[match.selection_method],
                "status": "spectral_fetch_failed",
            }
            for match in matches
        ], []
    validate_fetch_result(
        result,
        dates,
        expected_bbox=bbox,
        expected_center=(selected_tile.easting, selected_tile.northing),
        expected_size_px=args.size_px,
    )
    provenance = _fetch_provenance(result)
    if provenance["fetch_source"] != args.fetch_source:
        raise ValueError(
            f"fetch returned source {provenance['fetch_source']!r}, "
            f"expected {args.fetch_source!r}"
        )
    nmd = fetch_nmd_label_local(
        bbox,
        TileConfig(size_px=args.size_px),
        str(args.nmd_raster),
        allow_remote_fallback=False,
    )
    if nmd is None:
        return [{
            **identities[match.selection_method],
            "status": "nmd_missing",
        } for match in matches], []
    stable_forest = forest_mask(nmd)
    reference_bands = frame_bands(result, 0)
    reference_scl = result["scl"][0]
    candidate_frames = {
        match.selection_method: (
            slot_by_date[match.candidate_date],
            frame_bands(result, slot_by_date[match.candidate_date]),
            result["scl"][slot_by_date[match.candidate_date]],
        )
        for match in matches
    }
    shared_mask = stable_forest.copy()
    for _, candidate_bands, candidate_scl in candidate_frames.values():
        shared_mask &= common_valid_mask(
            stable_forest,
            reference_scl,
            candidate_scl,
            reference_bands,
            candidate_bands,
        )
    reference_vpp, reference_vpp_provenance = _fetch_vpp(
        bbox, args.reference_year, args, run_dir
    )
    candidate_vpp, candidate_vpp_provenance = _fetch_vpp(
        bbox, matches[0].candidate_year, args, run_dir
    )
    shared_mask &= _common_vpp_valid_mask(reference_vpp, candidate_vpp)
    stable_forest_pixels = int(stable_forest.sum())
    common_valid_pixels = int(shared_mask.sum())
    common_valid_fraction = (
        common_valid_pixels / stable_forest_pixels if stable_forest_pixels else 0.0
    )
    minimum_pixels = _minimum_common_forest_pixels(args.size_px)
    if (
        common_valid_pixels < minimum_pixels
        or common_valid_fraction < MIN_COMMON_FOREST_FRACTION
    ):
        raise ValueError(
            "fetch group has insufficient common S2/VPP-valid forest: "
            f"pixels={common_valid_pixels} (minimum={minimum_pixels}), "
            f"fraction={common_valid_fraction:.3f} "
            f"(minimum={MIN_COMMON_FOREST_FRACTION:.3f})"
        )
    shared_mask_sha256 = hashlib.sha256(
        np.packbits(shared_mask, bitorder="little").tobytes()
    ).hexdigest()
    ref_vpp_summary = summarize_vpp_phase(
        reference_vpp, dates[0], mask=shared_mask
    )

    summaries = []
    figures = []
    for match in matches:
        identity = identities[match.selection_method]
        try:
            _, candidate_bands, candidate_scl = candidate_frames[
                match.selection_method
            ]
            arrays, metrics = compare_spectral_pair(
                reference_bands, candidate_bands, shared_mask
            )
            candidate_vpp_summary = summarize_vpp_phase(
                candidate_vpp, match.candidate_date, mask=shared_mask
            )
            pair_dir = run_dir / "arrays" / tile_id
            pair_dir.mkdir(parents=True, exist_ok=True)
            artifact_stem = f"{match.selection_method}_{match.candidate_year}"
            array_path = pair_dir / f"{artifact_stem}.npz"
            artifact = _pair_artifact(
                arrays=arrays,
                reference_bands=reference_bands,
                candidate_bands=candidate_bands,
                reference_scl=reference_scl,
                candidate_scl=candidate_scl,
                nmd=nmd,
                stable_forest=stable_forest,
                reference_vpp=reference_vpp,
                candidate_vpp=candidate_vpp,
                identity=identity,
                provenance=provenance,
                shared_mask_sha256=shared_mask_sha256,
                common_valid_pixel_count=common_valid_pixels,
                common_valid_fraction_of_forest=common_valid_fraction,
                reference_vpp_provenance=reference_vpp_provenance,
                candidate_vpp_provenance=candidate_vpp_provenance,
            )
            artifact_meta = _write_npz_atomic(array_path, artifact)
            row = {
                **identity,
                "status": "ok",
                **asdict(match),
                **metrics,
                **provenance,
                "common_valid_mask_sha256": shared_mask_sha256,
                "common_valid_pixel_count": common_valid_pixels,
                "common_valid_fraction_of_forest": common_valid_fraction,
                "reference_vpp_provenance": reference_vpp_provenance,
                "candidate_vpp_provenance": candidate_vpp_provenance,
                **{f"reference_vpp_{key}": value for key, value in ref_vpp_summary.items()},
                **{f"candidate_vpp_{key}": value for key, value in candidate_vpp_summary.items()},
                "vpp_sos_shift_days": candidate_vpp_summary["sos_doy_median"] - ref_vpp_summary["sos_doy_median"],
                "vpp_eos_shift_days": candidate_vpp_summary["eos_doy_median"] - ref_vpp_summary["eos_doy_median"],
                "vpp_midpoint_proxy_shift_days": (
                    candidate_vpp_summary["season_midpoint_proxy_doy"]
                    - ref_vpp_summary["season_midpoint_proxy_doy"]
                ),
                "array_path": str(array_path.relative_to(run_dir)),
                "array_bytes": artifact_meta["bytes"],
                "array_sha256": artifact_meta["sha256"],
            }
            figure = run_dir / "figures" / f"{tile_id}_{artifact_stem}.png"
            render_pair_figure(
                figure,
                reference_bands,
                candidate_bands,
                arrays,
                match.reference_date,
                match.candidate_date,
                match.meteorology_distance,
                match.selection_method,
            )
            row["figure_path"] = str(figure.relative_to(run_dir))
            summaries.append(row)
            write_manifest(
                _pair_manifest_path(run_dir / "pairs", tile_id, match), row
            )
            figures.append(figure)
        except Exception as error:
            failed = {
                **identity,
                "status": "failed",
                "reason": f"{type(error).__name__}: {error}",
            }
            summaries.append(failed)
            write_manifest(
                _pair_manifest_path(run_dir / "pairs", tile_id, match),
                failed,
            )
    return summaries, figures


def _fetch_vpp(bbox, year, args, run_dir):
    values = fetch_vpp_tiles(
        bbox["west"], bbox["south"], bbox["east"], bbox["north"],
        size_px=args.size_px,
        cache_dir=run_dir / "cache" / "vpp" / str(year),
        year=year,
    )
    _validate_vpp_payload(values, year=year, size_px=args.size_px)
    provenance = _bind_vpp_raster_provenance(
        _vpp_product_provenance(bbox, year), values
    )
    return values, provenance


def _pair_artifact(
    *,
    arrays,
    reference_bands,
    candidate_bands,
    reference_scl,
    candidate_scl,
    nmd,
    stable_forest,
    reference_vpp,
    candidate_vpp,
    identity,
    provenance,
    shared_mask_sha256,
    common_valid_pixel_count,
    common_valid_fraction_of_forest,
    reference_vpp_provenance,
    candidate_vpp_provenance,
):
    """Build the complete, reproducible raster payload for one comparison."""
    reference_stack = _stack_named(reference_bands, COMPARISON_BANDS, "reference bands")
    candidate_stack = _stack_named(candidate_bands, COMPARISON_BANDS, "candidate bands")
    reference_vpp_stack = _stack_named(reference_vpp, VPP_BANDS, "reference VPP")
    candidate_vpp_stack = _stack_named(candidate_vpp, VPP_BANDS, "candidate VPP")
    shape = reference_stack.shape[1:]
    raster_shapes = {
        "candidate_bands": candidate_stack.shape[1:],
        "reference_scl": np.asarray(reference_scl).shape,
        "candidate_scl": np.asarray(candidate_scl).shape,
        "nmd_label": np.asarray(nmd).shape,
        "stable_forest_mask": np.asarray(stable_forest).shape,
        "reference_vpp": reference_vpp_stack.shape[1:],
        "candidate_vpp": candidate_vpp_stack.shape[1:],
    }
    wrong = {name: value for name, value in raster_shapes.items() if value != shape}
    if wrong:
        raise ValueError(f"pair artifact raster shapes differ from {shape}: {wrong}")

    return {
        **arrays,
        "dataset_schema_version": np.int16(DATASET_SCHEMA_VERSION),
        "band_names": np.asarray(COMPARISON_BANDS),
        "reference_bands": reference_stack,
        "candidate_bands": candidate_stack,
        "reference_scl": np.asarray(reference_scl, dtype=np.uint8),
        "candidate_scl": np.asarray(candidate_scl, dtype=np.uint8),
        "nmd_label": np.asarray(nmd),
        "stable_forest_mask": np.asarray(stable_forest, dtype=bool),
        "vpp_band_names": np.asarray(VPP_BANDS),
        "reference_vpp": reference_vpp_stack,
        "candidate_vpp": candidate_vpp_stack,
        "tile_id": np.asarray(identity["tile_id"]),
        "tile_size_px": np.int16(identity["tile_size_px"]),
        "plan_fingerprint": np.asarray(identity["plan_fingerprint"]),
        "run_fingerprint": np.asarray(identity["run_fingerprint"]),
        "fetch_group_id": np.asarray(identity["fetch_group_id"]),
        "fetch_source": np.asarray(identity["fetch_source"]),
        "fetch_requested_dates": np.asarray(identity["fetch_requested_dates"]),
        "fetch_returned_dates": np.asarray(provenance["fetch_returned_dates"]),
        "fetch_temporal_mask": np.asarray(
            provenance["fetch_temporal_mask"], dtype=np.uint8
        ),
        "fetch_frame_valid_fraction": np.asarray(
            provenance["fetch_frame_valid_fraction"], dtype=np.float32
        ),
        "fetch_coreg_shifts": np.asarray(
            provenance["fetch_coreg_shifts"], dtype=np.float32
        ),
        "fetch_coreg_ref_frame": np.int16(provenance["fetch_coreg_ref_frame"]),
        "fetch_coreg_m2": np.int8(provenance["fetch_coreg_m2"]),
        "fetch_coreg_n_aligned": np.int16(provenance["fetch_coreg_n_aligned"]),
        "fetch_coreg_max_shift": np.float32(provenance["fetch_coreg_max_shift"]),
        "fetch_coreg_anchor_valid_fraction": np.float32(
            provenance["fetch_coreg_anchor_valid_fraction"]
        ),
        "fetch_bbox_epsg3006": np.asarray(
            provenance["fetch_bbox_epsg3006"], dtype=np.float64
        ),
        "fetch_center_epsg3006": np.asarray(
            provenance["fetch_center_epsg3006"], dtype=np.float64
        ),
        "fetch_tile_size_px": np.int16(provenance["fetch_tile_size_px"]),
        "fetch_num_frames": np.int16(provenance["fetch_num_frames"]),
        "fetch_num_bands": np.int16(provenance["fetch_num_bands"]),
        "fetch_spectral_shape": np.asarray(
            provenance["fetch_spectral_shape"], dtype=np.int32
        ),
        "fetch_scl_shape": np.asarray(
            provenance["fetch_scl_shape"], dtype=np.int32
        ),
        "fetch_attempt_sha256": np.asarray(
            provenance["fetch_attempt_sha256"]
        ),
        "candidate_slot": np.int16(identity["candidate_slot"]),
        "common_valid_mask_sha256": np.asarray(shared_mask_sha256),
        "common_valid_pixel_count": np.int32(common_valid_pixel_count),
        "common_valid_fraction_of_forest": np.float32(
            common_valid_fraction_of_forest
        ),
        "reference_vpp_provenance_json": np.asarray(
            _canonical_json(reference_vpp_provenance)
        ),
        "candidate_vpp_provenance_json": np.asarray(
            _canonical_json(candidate_vpp_provenance)
        ),
        "bbox_epsg3006": np.asarray(identity["bbox_epsg3006"], dtype=np.float64),
        "center_epsg3006": np.asarray([
            identity["easting"], identity["northing"],
        ], dtype=np.float64),
        "reference_date": np.asarray(identity["reference_date"]),
        "candidate_date": np.asarray(identity["candidate_date"]),
        "reference_year": np.int16(identity["reference_year"]),
        "candidate_year": np.int16(identity["candidate_year"]),
        "selection_method": np.asarray(identity["selection_method"]),
        "grid_crs": np.asarray(identity["grid_crs"]),
        "grid_transform": np.asarray(
            identity["grid_transform"], dtype=np.float64
        ),
    }


def _fetch_provenance(result):
    """Return JSON-native fetch/coregistration evidence for one date group."""
    return {
        "fetch_attempt_sha256": _fetch_attempt_sha256(result),
        "fetch_source": str(result["source"]),
        "fetch_returned_dates": np.asarray(result["dates"], dtype=str).tolist(),
        "fetch_temporal_mask": np.asarray(
            result["temporal_mask"], dtype=np.uint8
        ).tolist(),
        "fetch_frame_valid_fraction": np.asarray(
            result["frame_valid_frac"], dtype=float
        ).tolist(),
        "fetch_coreg_shifts": np.asarray(
            result["coreg_shifts"], dtype=float
        ).tolist(),
        "fetch_coreg_ref_frame": int(result["coreg_ref_frame"]),
        "fetch_coreg_m2": int(result["coreg_m2"]),
        "fetch_coreg_n_aligned": int(result["coreg_n_aligned"]),
        "fetch_coreg_max_shift": float(result["coreg_max_shift"]),
        "fetch_coreg_anchor_valid_fraction": float(
            result["coreg_anchor_valid_frac"]
        ),
        "fetch_bbox_epsg3006": np.asarray(
            result["bbox_3006"], dtype=float
        ).tolist(),
        "fetch_center_epsg3006": [
            int(result["easting"]), int(result["northing"]),
        ],
        "fetch_tile_size_px": int(result["tile_size_px"]),
        "fetch_num_frames": int(result["num_frames"]),
        "fetch_num_bands": int(result["num_bands"]),
        "fetch_spectral_shape": list(np.asarray(result["spectral"]).shape),
        "fetch_scl_shape": list(np.asarray(result["scl"]).shape),
    }


def _fetch_attempt_sha256(result):
    """Fingerprint every returned value from one shared S2 fetch attempt."""
    digest = hashlib.sha256()
    for name in sorted(result):
        value = result[name]
        digest.update(name.encode("utf-8"))
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(_canonical_json({
                "dtype": array.dtype.str,
                "shape": list(array.shape),
            }).encode("utf-8"))
            digest.update(array.view(np.uint8).tobytes())
        else:
            digest.update(_canonical_json(value).encode("utf-8"))
    return digest.hexdigest()


def _minimum_common_forest_pixels(size_px):
    """Scale the 512-pixel absolute floor for small synthetic test tiles."""
    numerator = MIN_COMMON_FOREST_PIXELS_512 * int(size_px) * int(size_px)
    denominator = 512 * 512
    return max(1, (numerator + denominator - 1) // denominator)


def _validate_vpp_payload(values, *, year, size_px):
    missing = [name for name in VPP_BANDS if name not in values]
    if missing:
        raise ValueError(f"VPP {year} is missing bands {missing}")
    shape = (int(size_px), int(size_px))
    for name in VPP_BANDS:
        array = np.asarray(values[name], dtype=float)
        if array.shape != shape:
            raise ValueError(
                f"VPP {year} {name} shape {array.shape} does not match {shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"VPP {year} {name} contains non-finite values")
        if np.any(array < 0):
            raise ValueError(f"VPP {year} {name} contains negative values")
    # summarize_vpp_phase applies the strict integral/prefix/DOY check. Calling
    # it here makes a wrong product fail before any pair artifact is written.
    summarize_vpp_phase(values, f"{int(year):04d}-06-01")


def _common_vpp_valid_mask(reference_vpp, candidate_vpp):
    """Return one mask valid in every required VPP band for both years."""
    first = np.asarray(reference_vpp[VPP_BANDS[0]])
    mask = np.ones(first.shape, dtype=bool)
    for payload in (reference_vpp, candidate_vpp):
        for name in VPP_BANDS:
            values = np.asarray(payload[name], dtype=float)
            if values.shape != mask.shape:
                raise ValueError("reference/candidate VPP rasters use different grids")
            mask &= np.isfinite(values)
            if name == "minv":
                mask &= values >= 0
            else:
                mask &= values > 0
    return mask


def _named_arrays_sha256(mapping, names):
    """Hash named raster values including dtype, shape, and byte content."""
    digest = hashlib.sha256()
    for name in names:
        array = np.ascontiguousarray(mapping[name])
        digest.update(name.encode("utf-8"))
        digest.update(_canonical_json({
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }).encode("utf-8"))
        digest.update(array.view(np.uint8).tobytes())
    return digest.hexdigest()


def _bind_vpp_raster_provenance(provenance, values):
    """Bind source-product provenance to the exact consumed VPP pixels."""
    payload = {
        key: value for key, value in provenance.items() if key != "fingerprint"
    }
    payload["raster_payload_sha256"] = _named_arrays_sha256(values, VPP_BANDS)
    payload["fingerprint"] = _canonical_sha256(payload)
    return payload


def _vpp_product_provenance(bbox, year):
    """Resolve exact WEkEO COG filenames and versions covering one tile-year."""
    source = os.environ.get("VPP_SOURCE", "").strip().lower()
    if source != "wekeo":
        raise RuntimeError(
            f"reproducible POC requires VPP_SOURCE=wekeo, got {source!r}"
        )
    root = Path(os.environ["VPP_WEKEO_DIR"]).resolve()
    index_path = root / "index.json"
    index = json.loads(index_path.read_text())
    bounds = _bbox_3006_to_4326(
        bbox["west"], bbox["south"], bbox["east"], bbox["north"]
    )
    products = []
    for filename, metadata in sorted(index.items()):
        if int(metadata.get("year", -1)) != int(year):
            continue
        required_metadata = {"year", "season", "metric", "tileId", "bounds_4326"}
        missing_metadata = sorted(required_metadata - set(metadata))
        if missing_metadata:
            raise ValueError(
                f"VPP index entry {filename!r} is missing {missing_metadata}"
            )
        indexed_bounds = np.asarray(metadata["bounds_4326"], dtype=float)
        if indexed_bounds.shape != (4,) or not np.isfinite(indexed_bounds).all():
            raise ValueError(f"VPP index entry {filename!r} has invalid bounds")
        if (
            int(metadata["season"]) != VPP_SEASON
            or metadata["metric"] not in {name.upper() for name in VPP_BANDS}
            or not _bounds_overlap(indexed_bounds.tolist(), bounds)
        ):
            continue
        parsed = _parse_vpp_filename_strict(filename)
        expected = {
            "year": int(metadata["year"]),
            "season": int(metadata["season"]),
            "metric": str(metadata["metric"]),
            "tile_id": str(metadata["tileId"]),
        }
        if any(parsed[key] != value for key, value in expected.items()):
            raise ValueError(
                f"VPP index metadata disagrees with filename {filename!r}"
            )
        product_path = root / filename
        stat = product_path.stat()
        products.append({
            **parsed,
            "filename": filename,
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": _sha256_file_cached(product_path),
        })
    present = {item["metric"] for item in products}
    required = {name.upper() for name in VPP_BANDS}
    if present != required:
        raise ValueError(
            f"VPP provenance for {year} covers {sorted(present)}, "
            f"expected {sorted(required)}"
        )
    versions = sorted({item["version"] for item in products})
    if len(versions) != 1:
        raise ValueError(
            f"VPP provenance for {year} mixes product versions {versions}"
        )
    product_keys = [(item["tile_id"], item["metric"]) for item in products]
    if len(product_keys) != len(set(product_keys)):
        raise ValueError(f"VPP provenance for {year} has duplicate tile/metric products")
    payload = {
        "source": "wekeo",
        "product_year": int(year),
        "season": VPP_SEASON,
        "index_path": str(index_path),
        "index_sha256": _sha256_file(index_path),
        "products": products,
        "versions": versions,
    }
    payload["fingerprint"] = _canonical_sha256(payload)
    return payload


def _parse_vpp_filename_strict(filename):
    match = _VPP_FILENAME_RE.fullmatch(Path(filename).name)
    if match is None:
        raise ValueError(f"unexpected HR-VPP filename {filename!r}")
    return {
        "metric": match.group("metric"),
        "tile_id": match.group("tile"),
        "year": int(match.group("year")),
        "version": int(match.group("version")),
        "season": int(match.group("season")),
    }


def _stack_named(mapping, names, label):
    missing = [name for name in names if name not in mapping]
    if missing:
        raise ValueError(f"{label} missing {missing}")
    values = [np.asarray(mapping[name], dtype=np.float32) for name in names]
    shapes = {value.shape for value in values}
    if len(shapes) != 1:
        raise ValueError(f"{label} have inconsistent shapes: {sorted(shapes)}")
    return np.stack(values, axis=0)


def _write_npz_atomic(path, arrays):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "bytes": destination.stat().st_size,
        "sha256": _sha256_file(destination),
    }


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_file_cached(path):
    """Cache immutable-file digests while invalidating on any stat change."""
    resolved = Path(path).resolve()
    stat = resolved.stat()
    key = (
        str(resolved),
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )
    if key not in _FILE_DIGEST_CACHE:
        _FILE_DIGEST_CACHE[key] = _sha256_file(resolved)
    return _FILE_DIGEST_CACHE[key]


def _pair_manifest_path(pairs_dir, tile_id, match):
    if match.selection_method not in SELECTION_METHODS:
        raise ValueError(f"unknown selection method {match.selection_method!r}")
    return Path(pairs_dir) / (
        f"{tile_id}_{match.selection_method}_{match.candidate_year}.json"
    )


def _expected_pair_manifest_names(tile_ids, candidate_years):
    return {
        f"{tile_id}_{method}_{int(year)}.json"
        for tile_id in tile_ids
        for year in candidate_years
        for method in SELECTION_METHODS
    }


def _fetch_group_succeeded(paths_by_method, expected_by_method):
    """Accept a resume group only when both pairs came from one fetch attempt."""
    if set(paths_by_method) != set(SELECTION_METHODS):
        return False
    if set(expected_by_method) != set(SELECTION_METHODS):
        return False
    rows = []
    for method in SELECTION_METHODS:
        path = Path(paths_by_method[method])
        if not _pair_succeeded(path, expected_by_method[method]):
            return False
        try:
            rows.append(json.loads(path.read_text()))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False
    return _group_rows_consistent(rows)


def _group_rows_consistent(rows):
    """Verify shared fetch/mask/VPP evidence across both strategy rows."""
    if len(rows) != len(SELECTION_METHODS):
        return False
    if {row.get("selection_method") for row in rows} != set(SELECTION_METHODS):
        return False
    if any(row.get("status") != "ok" for row in rows):
        return False
    if any(
        key not in row
        for row in rows
        for key in _FETCH_GROUP_SHARED_KEYS
    ):
        return False
    anchor = rows[0]
    return all(
        _canonical_json(row.get(key)) == _canonical_json(anchor.get(key))
        for row in rows[1:]
        for key in _FETCH_GROUP_SHARED_KEYS
    )


def _vpp_versions_by_year(rows, years):
    """Collect exact VPP product versions used for each requested year."""
    output = {str(int(year)): set() for year in years}
    for row in rows:
        if row.get("status") != "ok":
            continue
        for prefix, year_key in (
            ("reference", "reference_year"),
            ("candidate", "candidate_year"),
        ):
            year = str(int(row[year_key]))
            if year not in output:
                raise ValueError(f"pair row carries unexpected VPP year {year}")
            provenance = row.get(f"{prefix}_vpp_provenance", {})
            versions = provenance.get("versions", [])
            if (
                provenance.get("product_year") != int(year)
                or len(versions) != 1
                or not isinstance(versions[0], int)
            ):
                raise ValueError(f"pair row has invalid VPP version for {year}")
            output[year].add(versions[0])
    return {
        year: sorted(versions) for year, versions in sorted(output.items())
    }


def render_pair_figure(
    path, reference, candidate, arrays, ref_date, candidate_date, distance, method,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes[0, 0].imshow(_rgb(reference)); axes[0, 0].set_title(ref_date)
    axes[0, 1].imshow(_rgb(candidate)); axes[0, 1].set_title(candidate_date)
    im = axes[0, 2].imshow(arrays["ndvi_difference"], cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    axes[0, 2].set_title("NDVI difference"); figure.colorbar(im, ax=axes[0, 2])
    axes[1, 0].imshow(arrays["ndvi_reference"], cmap="RdYlGn", vmin=-1, vmax=1)
    axes[1, 0].set_title("Reference NDVI")
    axes[1, 1].imshow(arrays["ndvi_candidate"], cmap="RdYlGn", vmin=-1, vmax=1)
    axes[1, 1].set_title("Candidate NDVI")
    im = axes[1, 2].imshow(arrays["spectral_angle_rad"], cmap="magma")
    axes[1, 2].set_title(f"Spectral angle ({method}, meteo d={distance:.2f})")
    figure.colorbar(im, ax=axes[1, 2])
    for axis in axes.flat:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def write_report(path, summaries, figures):
    rows = []
    figure_by_stem = {item.stem: item for item in figures}
    for row in summaries:
        if row.get("status") != "ok":
            rows.append(f"<tr><td>{html.escape(str(row))}</td></tr>")
            continue
        stem = (
            f"{row['tile_id']}_{row['selection_method']}_"
            f"{row['candidate_year']}"
        )
        figure = figure_by_stem.get(stem)
        image = f'<img src="{html.escape(str(figure.relative_to(path.parent)))}" width="900">' if figure else ""
        rows.append(
            "<tr><td>" + html.escape(row["tile_id"]) + "</td><td>" +
            html.escape(str(row["candidate_year"])) + "</td><td>" +
            html.escape(row["selection_method"]) + "</td><td>" +
            html.escape(f"{row['meteorology_distance']:.3f}") + "</td><td>" + image + "</td></tr>"
        )
    path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Meteorology analog forest POC</title>"
        "<h1>Meteorology-matched forest Sentinel-2 POC</h1>"
        "<p>VPP midpoint is a proxy derived from (SOSD + EOSD) / 2; it is not an observed peak date.</p>"
        "<table><tr><th>Tile</th><th>Year</th><th>Selection</th>"
        "<th>Meteo distance</th><th>Comparison</th></tr>" +
        "".join(rows) + "</table>"
    )


def _rgb(bands):
    rgb = np.stack([bands["B04"], bands["B03"], bands["B02"]], axis=-1)
    valid = np.isfinite(rgb) & (rgb > 0)
    if not valid.any():
        return np.zeros_like(rgb)
    low, high = np.percentile(rgb[valid], [2, 98])
    return np.clip((rgb - low) / max(high - low, 1e-6), 0, 1)


def _validate_live_environment(args):
    if os.environ.get("VPP_SOURCE", "").lower() != "wekeo":
        raise RuntimeError("live POC requires VPP_SOURCE=wekeo")
    if not os.environ.get("VPP_WEKEO_DIR"):
        raise RuntimeError("live POC requires VPP_WEKEO_DIR")
    if args.fetch_source == "des":
        try:
            connect_des()
        except Exception as error:
            raise RuntimeError(f"DES authentication preflight failed: {error}") from error


def _validate_years(reference_year, candidate_years):
    if len(set(candidate_years)) != len(candidate_years):
        raise ValueError("candidate years must be unique")
    if reference_year in candidate_years:
        raise ValueError("reference year must not also be a candidate year")


def _pair_succeeded(path, expected):
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
        if (
            payload.get("status") != "ok"
            or payload.get("dataset_schema_version") != DATASET_SCHEMA_VERSION
        ):
            return False
        if any(
            _canonical_json(payload.get(key)) != _canonical_json(value)
            for key, value in expected.items()
        ):
            return False
        relative = Path(payload["array_path"])
        if relative.is_absolute():
            return False
        run_dir = path.parent.parent.resolve()
        artifact = (run_dir / relative).resolve()
        artifact.relative_to(run_dir)
        if (
            not artifact.is_file()
            or artifact.stat().st_size != int(payload["array_bytes"])
            or _sha256_file(artifact) != payload["array_sha256"]
        ):
            return False
        with np.load(artifact, allow_pickle=False) as data:
            if not _PAIR_REQUIRED_ARRAY_KEYS <= set(data.files):
                return False
            size = int(expected["tile_size_px"])
            raster_shape = (size, size)
            if int(data["dataset_schema_version"]) != DATASET_SCHEMA_VERSION:
                return False
            scalar_identity = {
                "tile_id": "tile_id",
                "tile_size_px": "tile_size_px",
                "plan_fingerprint": "plan_fingerprint",
                "run_fingerprint": "run_fingerprint",
                "fetch_group_id": "fetch_group_id",
                "fetch_source": "fetch_source",
                "candidate_slot": "candidate_slot",
                "reference_date": "reference_date",
                "candidate_date": "candidate_date",
                "reference_year": "reference_year",
                "candidate_year": "candidate_year",
                "selection_method": "selection_method",
                "grid_crs": "grid_crs",
            }
            for array_key, expected_key in scalar_identity.items():
                if _canonical_json(np.asarray(data[array_key]).item()) != _canonical_json(
                    expected[expected_key]
                ):
                    return False
            if np.asarray(data["band_names"], dtype=str).tolist() != expected["band_names"]:
                return False
            if np.asarray(data["vpp_band_names"], dtype=str).tolist() != expected[
                "vpp_band_names"
            ]:
                return False
            requested = np.asarray(data["fetch_requested_dates"], dtype=str).tolist()
            returned = np.asarray(data["fetch_returned_dates"], dtype=str).tolist()
            if requested != expected["fetch_requested_dates"] or returned != requested:
                return False
            if payload.get("fetch_returned_dates") != returned:
                return False
            fetch_attempt = str(
                np.asarray(data["fetch_attempt_sha256"]).item()
            )
            if (
                not re.fullmatch(r"[0-9a-f]{64}", fetch_attempt)
                or payload.get("fetch_attempt_sha256") != fetch_attempt
            ):
                return False
            candidate_slot = int(data["candidate_slot"])
            if returned[candidate_slot] != expected["candidate_date"]:
                return False
            frame_count = len(requested)
            if np.asarray(data["fetch_temporal_mask"]).shape != (frame_count,):
                return False
            if not np.asarray(data["fetch_temporal_mask"], dtype=bool).all():
                return False
            if np.asarray(data["fetch_frame_valid_fraction"]).shape != (frame_count,):
                return False
            if np.asarray(data["fetch_coreg_shifts"]).shape != (frame_count, 2):
                return False
            valid_fraction = np.asarray(
                data["fetch_frame_valid_fraction"], dtype=float
            )
            shifts = np.asarray(data["fetch_coreg_shifts"], dtype=float)
            if (
                not np.isfinite(valid_fraction).all()
                or np.any(valid_fraction < 0.5)
                or not np.isfinite(shifts).all()
            ):
                return False
            coreg_reference = int(data["fetch_coreg_ref_frame"])
            coreg_m2 = int(data["fetch_coreg_m2"])
            coreg_aligned = int(data["fetch_coreg_n_aligned"])
            coreg_max_shift = float(data["fetch_coreg_max_shift"])
            coreg_anchor = float(data["fetch_coreg_anchor_valid_fraction"])
            measured_aligned = int(sum(
                index != coreg_reference and abs(dy) + abs(dx) > 0.0
                for index, (dy, dx) in enumerate(shifts)
            ))
            if (
                not 0 <= coreg_reference < frame_count
                or coreg_m2 != 1
                or coreg_aligned != measured_aligned
                or not np.isfinite(coreg_max_shift)
                or coreg_max_shift < 0
                or not np.isfinite(coreg_anchor)
                or coreg_anchor <= 0
            ):
                return False
            if not np.isclose(
                coreg_max_shift,
                np.max(np.linalg.norm(shifts, axis=1)),
                rtol=1e-5,
                atol=1e-6,
            ):
                return False
            expected_shapes = {
                "reference_bands": (len(COMPARISON_BANDS), *raster_shape),
                "candidate_bands": (len(COMPARISON_BANDS), *raster_shape),
                "reference_scl": raster_shape,
                "candidate_scl": raster_shape,
                "nmd_label": raster_shape,
                "stable_forest_mask": raster_shape,
                "reference_vpp": (len(VPP_BANDS), *raster_shape),
                "candidate_vpp": (len(VPP_BANDS), *raster_shape),
                **{key: raster_shape for key in _PAIR_DERIVED_ARRAY_KEYS},
            }
            if any(np.asarray(data[key]).shape != shape for key, shape in expected_shapes.items()):
                return False
            fetch_bbox = np.asarray(data["fetch_bbox_epsg3006"], dtype=float)
            fetch_center = np.asarray(data["fetch_center_epsg3006"], dtype=float)
            if (
                fetch_bbox.shape != (4,)
                or fetch_center.shape != (2,)
                or not np.array_equal(fetch_bbox, expected["bbox_epsg3006"])
                or not np.array_equal(
                    fetch_center, [expected["easting"], expected["northing"]]
                )
                or int(data["fetch_tile_size_px"]) != size
                or int(data["fetch_num_frames"]) != frame_count
                or int(data["fetch_num_bands"]) != 6
                or np.asarray(data["fetch_spectral_shape"], dtype=int).tolist()
                != [frame_count * 6, size, size]
                or np.asarray(data["fetch_scl_shape"], dtype=int).tolist()
                != [frame_count, size, size]
            ):
                return False
            if not np.array_equal(
                np.asarray(data["bbox_epsg3006"], dtype=float),
                np.asarray(expected["bbox_epsg3006"], dtype=float),
            ):
                return False
            if not np.array_equal(
                np.asarray(data["center_epsg3006"], dtype=float),
                np.asarray([expected["easting"], expected["northing"]], dtype=float),
            ):
                return False
            if not np.array_equal(
                np.asarray(data["grid_transform"], dtype=float),
                np.asarray(expected["grid_transform"], dtype=float),
            ):
                return False
            mask_hash = hashlib.sha256(
                np.packbits(
                    np.asarray(data["valid_mask"], dtype=bool), bitorder="little"
                ).tobytes()
            ).hexdigest()
            if (
                str(np.asarray(data["common_valid_mask_sha256"]).item()) != mask_hash
                or payload.get("common_valid_mask_sha256") != mask_hash
            ):
                return False
            stored_mask = np.asarray(data["valid_mask"], dtype=bool)
            forest = np.asarray(data["stable_forest_mask"], dtype=bool)
            nmd = np.asarray(data["nmd_label"])
            if not np.array_equal(forest, forest_mask(nmd)):
                return False
            common_count = int(data["common_valid_pixel_count"])
            common_fraction = float(data["common_valid_fraction_of_forest"])
            expected_fraction = float(stored_mask.sum() / forest.sum()) if forest.any() else 0.0
            if (
                common_count != int(stored_mask.sum())
                or payload.get("common_valid_pixel_count") != common_count
                or common_count < _minimum_common_forest_pixels(size)
                or common_fraction < MIN_COMMON_FOREST_FRACTION
                or not np.isclose(common_fraction, expected_fraction, rtol=1e-5, atol=1e-7)
                or not np.isclose(
                    float(payload.get("common_valid_fraction_of_forest", -1)),
                    common_fraction,
                    rtol=1e-5,
                    atol=1e-7,
                )
            ):
                return False
            reference_vpp = {
                name: np.asarray(data["reference_vpp"])[index]
                for index, name in enumerate(VPP_BANDS)
            }
            candidate_vpp = {
                name: np.asarray(data["candidate_vpp"])[index]
                for index, name in enumerate(VPP_BANDS)
            }
            _validate_vpp_payload(
                reference_vpp, year=expected["reference_year"], size_px=size
            )
            _validate_vpp_payload(
                candidate_vpp, year=expected["candidate_year"], size_px=size
            )
            if np.any(stored_mask & ~_common_vpp_valid_mask(reference_vpp, candidate_vpp)):
                return False
            for prefix in ("reference", "candidate"):
                artifact_provenance = json.loads(
                    str(np.asarray(data[f"{prefix}_vpp_provenance_json"]).item())
                )
                if _canonical_json(artifact_provenance) != _canonical_json(
                    payload[f"{prefix}_vpp_provenance"]
                ):
                    return False
                product_year = (
                    expected["reference_year"]
                    if prefix == "reference"
                    else expected["candidate_year"]
                )
                vpp_values = (
                    reference_vpp if prefix == "reference" else candidate_vpp
                )
                versions = artifact_provenance.get("versions", [])
                products = artifact_provenance.get("products", [])
                if (
                    artifact_provenance.get("source") != "wekeo"
                    or artifact_provenance.get("product_year") != product_year
                    or artifact_provenance.get("season") != VPP_SEASON
                    or len(versions) != 1
                    or not products
                    or any(
                        not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", ""))
                        for item in products
                    )
                    or artifact_provenance.get("raster_payload_sha256")
                    != _named_arrays_sha256(vpp_values, VPP_BANDS)
                    or artifact_provenance.get("fingerprint")
                    != _canonical_sha256({
                        key: value
                        for key, value in artifact_provenance.items()
                        if key != "fingerprint"
                    })
                ):
                    return False
            for key in (
                "fetch_temporal_mask",
                "fetch_frame_valid_fraction",
                "fetch_coreg_shifts",
                "fetch_bbox_epsg3006",
                "fetch_center_epsg3006",
                "fetch_spectral_shape",
                "fetch_scl_shape",
            ):
                if not np.allclose(
                    np.asarray(data[key], dtype=float),
                    np.asarray(payload[key], dtype=float),
                    equal_nan=False,
                ):
                    return False
            scalar_provenance = {
                "fetch_coreg_ref_frame": coreg_reference,
                "fetch_coreg_m2": coreg_m2,
                "fetch_coreg_n_aligned": coreg_aligned,
                "fetch_coreg_max_shift": coreg_max_shift,
                "fetch_coreg_anchor_valid_fraction": coreg_anchor,
                "fetch_tile_size_px": int(data["fetch_tile_size_px"]),
                "fetch_num_frames": int(data["fetch_num_frames"]),
                "fetch_num_bands": int(data["fetch_num_bands"]),
            }
            for key, value in scalar_provenance.items():
                if key not in payload or not np.isclose(
                    float(payload[key]), float(value), rtol=1e-5, atol=1e-6
                ):
                    return False
            return True
    except (
        EOFError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ):
        return False


def _safe_run_dir(output_dir: Path, run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id) or run_id in {".", ".."}:
        raise ValueError("run-id may contain only letters, digits, dot, underscore, and dash")
    root = output_dir.resolve()
    destination = (root / run_id).resolve()
    if destination.parent != root:
        raise ValueError("run-id escapes output directory")
    return destination


def _load_or_create_manifest(
    path,
    args,
    run_id,
    *,
    selected_tiles,
    selected_tiles_path,
):
    proposed = _base_manifest(
        args,
        run_id,
        selected_tiles=selected_tiles,
        selected_tiles_path=selected_tiles_path,
    )
    if not path.exists():
        write_manifest(path, proposed)
        return proposed
    existing = json.loads(path.read_text())
    if _immutable_command(existing.get("command", {})) != _immutable_command(vars(args)):
        raise ValueError("existing run-id was created with different immutable options")
    if existing.get("run_fingerprint") != proposed["run_fingerprint"]:
        raise ValueError("existing run-id belongs to different code or schema revisions")
    return existing


def _immutable_command(command):
    ignored = {"select_only", "plan_network", "execute_fetch"}
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in command.items()
        if key not in ignored
    }


def _load_selected_tiles(path):
    frame = pd.read_csv(path)
    return [
        TileCandidate(
            easting=int(row.easting),
            northing=int(row.northing),
            forest_fraction=float(row.forest_fraction),
            source=str(row.source),
            classified_fraction=float(getattr(row, "classified_fraction", 1.0)),
        )
        for row in frame.itertuples()
    ]


def _validate_selected_tiles(selected, args):
    if len(selected) != int(args.tiles):
        raise ValueError(
            f"tiles.csv has {len(selected)} tiles, expected exactly {args.tiles}"
        )
    centers = [(item.easting, item.northing) for item in selected]
    if len(set(centers)) != len(centers):
        raise ValueError("tiles.csv contains duplicate centers")
    for item in selected:
        if item.forest_fraction < float(args.min_forest_fraction):
            raise ValueError(
                f"tile {(item.easting, item.northing)} forest fraction "
                f"{item.forest_fraction:.3f} is below {args.min_forest_fraction:.3f}"
            )
        if not 0 <= item.classified_fraction <= 1:
            raise ValueError("tiles.csv classified fractions must be within [0, 1]")
        if not item.source:
            raise ValueError("tiles.csv source values must be non-empty")


def _serialize_context(context):
    payload = dict(context)
    payload["matches"] = [asdict(match) for match in context.get("matches", [])]
    return payload


def _planning_inputs(tile, args):
    return {
        "tile": asdict(tile),
        "reference_year": int(args.reference_year),
        "candidate_years": [int(year) for year in args.candidate_years],
        "window": list(args.window),
        "size_px": int(args.size_px),
        "max_aoi_cloud": float(args.max_aoi_cloud),
        "fetch_source": args.fetch_source,
    }


def _plan_fingerprint(context):
    payload = _serialize_context(context)
    payload.pop("plan_fingerprint", None)
    return _canonical_sha256(payload)


def _plan_is_current(context, expected_inputs):
    if not context or context.get("planning_error"):
        return False
    if context.get("planning_schema_version") != PLANNING_SCHEMA_VERSION:
        return False
    if _canonical_json(context.get("planning_inputs")) != _canonical_json(
        expected_inputs
    ):
        return False
    fingerprint = context.get("plan_fingerprint")
    if not isinstance(fingerprint, str) or fingerprint != _plan_fingerprint(context):
        return False
    try:
        _validate_plan_context(context, expected_inputs)
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _validate_plan_context(context, expected_inputs):
    """Validate the exact durable planning schema used by fetch and analysis."""
    required = {
        "bbox_3006",
        "bbox_wgs84",
        "reference_date",
        "matches",
        "fetch_plans",
        "meteorology_by_year",
        "planning_inputs",
        "planning_schema_version",
        "plan_fingerprint",
    }
    if set(context) != required:
        raise ValueError("plan context has unexpected or missing fields")
    bbox_3006 = context["bbox_3006"]
    bbox_wgs84 = context["bbox_wgs84"]
    bbox_fields = {"west", "south", "east", "north"}
    for label, bbox in (("EPSG:3006", bbox_3006), ("WGS84", bbox_wgs84)):
        if (
            not isinstance(bbox, dict)
            or set(bbox) != bbox_fields
            or not all(np.isfinite(float(value)) for value in bbox.values())
            or float(bbox["west"]) >= float(bbox["east"])
            or float(bbox["south"]) >= float(bbox["north"])
        ):
            raise ValueError(f"plan context has invalid {label} bbox")

    years = [
        int(expected_inputs["reference_year"]),
        *[int(year) for year in expected_inputs["candidate_years"]],
    ]
    fetch_plans = {
        int(year): payload for year, payload in context["fetch_plans"].items()
    }
    meteorology = {
        int(year): payload
        for year, payload in context["meteorology_by_year"].items()
    }
    if set(fetch_plans) != set(years) or set(meteorology) != set(years):
        raise ValueError("plan context does not cover the exact requested years")
    plan_fields = {
        "mode",
        "dates",
        "n_candidates_after",
        "elapsed_s",
        "notes",
        "scl_gate",
        "scl_screen_complete",
        "scl_thresholds",
    }
    for year in years:
        plan = fetch_plans[year]
        if not isinstance(plan, dict) or set(plan) != plan_fields:
            raise ValueError(f"plan context has malformed fetch plan for {year}")
        if plan["mode"] != "scl_only":
            raise ValueError(f"plan context has unexpected mode for {year}")
        if not all(
            isinstance(plan[name], dict)
            for name in ("n_candidates_after", "elapsed_s", "notes")
        ):
            raise ValueError(f"plan context has malformed diagnostics for {year}")
        _validate_scl_plan(plan, year)
        rows = meteorology[year]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"plan context has no meteorology for {year}")
        for row in rows:
            if not isinstance(row, dict) or "date" not in row:
                raise ValueError(f"plan context has malformed meteorology for {year}")
            timestamp = pd.Timestamp(row["date"])
            if timestamp.year != year:
                raise ValueError(f"plan context has out-of-year meteorology for {year}")

    matches = context["matches"]
    if not isinstance(matches, list):
        raise ValueError("plan context matches are not a list")
    seen = set()
    reference_dates = set()
    for match in matches:
        if not isinstance(match, DateMatch):
            raise ValueError("plan context has malformed match entries")
        key = (int(match.candidate_year), match.selection_method)
        if (
            key in seen
            or int(match.reference_year) != years[0]
            or int(match.candidate_year) not in years[1:]
            or match.selection_method not in SELECTION_METHODS
            or match.reference_date not in fetch_plans[years[0]]["dates"]
            or match.candidate_date
            not in fetch_plans[int(match.candidate_year)]["dates"]
            or not np.isfinite(float(match.meteorology_distance))
            or int(match.calendar_displacement_days) < 0
        ):
            raise ValueError("plan context has inconsistent match entries")
        seen.add(key)
        reference_dates.add(match.reference_date)
    expected_reference = (
        next(iter(reference_dates)) if len(reference_dates) == 1 else None
    )
    if context["reference_date"] != expected_reference:
        raise ValueError("plan context has inconsistent reference date")


def _frame_records(frame):
    """Return JSON-native daily meteorology rows for durable plan evidence."""
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _load_context(path):
    payload = json.loads(path.read_text())
    payload["matches"] = [DateMatch(**match) for match in payload.get("matches", [])]
    return payload


def _load_pair_rows(path):
    return [json.loads(item.read_text()) for item in sorted(path.glob("*.json"))]


def _base_manifest(args, run_id, *, selected_tiles, selected_tiles_path):
    imint_git_sha = _validated_imint_git_sha(required=bool(args.execute_fetch))
    metafilter_git_sha = _metafilter_sha()
    if bool(args.execute_fetch) and not re.fullmatch(
        r"[0-9a-f]{40}", metafilter_git_sha
    ):
        raise RuntimeError(
            "live POC requires metafilter installed from an exact Git commit"
        )
    source_provenance = _source_provenance(args)
    selected_payload = [asdict(tile) for tile in selected_tiles]
    selected_tiles_fingerprint = _canonical_sha256(selected_payload)
    selected_tiles_file = _file_identity(Path(selected_tiles_path))
    provenance = _run_fingerprint_payload(
        command=vars(args),
        imint_git_sha=imint_git_sha,
        metafilter_git_sha=metafilter_git_sha,
        source_provenance=source_provenance,
        selected_tiles_fingerprint=selected_tiles_fingerprint,
        selected_tiles_file=selected_tiles_file,
    )
    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "started",
        "command": vars(args),
        "planning_schema_version": PLANNING_SCHEMA_VERSION,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "imint_git_sha": imint_git_sha,
        "metafilter_git_sha": metafilter_git_sha,
        "source_provenance": source_provenance,
        "selected_tiles_fingerprint": selected_tiles_fingerprint,
        "selected_tiles_file": selected_tiles_file,
        "tiles": selected_payload,
        "run_fingerprint": _canonical_sha256(provenance),
    }


def _run_fingerprint_payload(
    *,
    command,
    imint_git_sha,
    metafilter_git_sha,
    source_provenance,
    selected_tiles_fingerprint,
    selected_tiles_file,
):
    return {
        "command": _immutable_command(command),
        "planning_schema_version": PLANNING_SCHEMA_VERSION,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "imint_git_sha": imint_git_sha,
        "metafilter_git_sha": metafilter_git_sha,
        "source_provenance": source_provenance,
        "selected_tiles_fingerprint": selected_tiles_fingerprint,
        "selected_tiles_file": selected_tiles_file,
    }


def _source_provenance(args):
    return {
        "inventory": _path_identity(Path(args.inventory)),
        "nmd": _file_identity(Path(args.nmd_raster)),
        "vpp": _vpp_source_identity(required=bool(args.execute_fetch)),
    }


def _file_identity(path):
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "bytes": stat.st_size,
        "sha256": _sha256_file(resolved),
    }


def _path_identity(path):
    resolved = Path(path).resolve()
    if resolved.is_file():
        return {"kind": "file", **_file_identity(resolved)}
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    entries = []
    for item in sorted(candidate for candidate in resolved.rglob("*") if candidate.is_file()):
        stat = item.stat()
        entries.append({
            "path": str(item.relative_to(resolved)),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": _sha256_file_cached(item),
        })
    return {
        "kind": "directory",
        "path": str(resolved),
        "file_count": len(entries),
        "inventory_sha256": _canonical_sha256(entries),
    }


def _vpp_source_identity(*, required):
    source = os.environ.get("VPP_SOURCE", "").strip().lower()
    root_value = os.environ.get("VPP_WEKEO_DIR")
    if source != "wekeo" or not root_value:
        if required:
            raise RuntimeError("live POC requires VPP_SOURCE=wekeo and VPP_WEKEO_DIR")
        return {"source": source or "unset", "available": False}
    root = Path(root_value).resolve()
    index_path = root / "index.json"
    if not index_path.is_file():
        if required:
            raise FileNotFoundError(index_path)
        return {"source": "wekeo", "path": str(root), "available": False}
    index = json.loads(index_path.read_text())
    products = []
    for filename, metadata in sorted(index.items()):
        parsed = _parse_vpp_filename_strict(filename)
        product_path = root / filename
        if not product_path.is_file():
            if required:
                raise FileNotFoundError(product_path)
            continue
        stat = product_path.stat()
        products.append({
            "filename": filename,
            "year": parsed["year"],
            "version": parsed["version"],
            "season": parsed["season"],
            "metric": parsed["metric"],
            "tile_id": parsed["tile_id"],
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
            "index_metadata_sha256": _canonical_sha256(metadata),
        })
    return {
        "source": "wekeo",
        "available": True,
        "path": str(root),
        "index": _file_identity(index_path),
        "product_count": len(products),
        "product_inventory_sha256": _canonical_sha256(products),
        "versions": sorted({item["version"] for item in products}),
    }


def _canonical_json(payload):
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _canonical_sha256(payload):
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _metafilter_sha():
    try:
        direct_url = importlib.metadata.distribution(
            "space-datalab-metafilter"
        ).read_text("direct_url.json")
        if direct_url:
            payload = json.loads(direct_url)
            commit = payload.get("vcs_info", {}).get("commit_id")
            if commit:
                return commit
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError):
        pass
    import metafilter
    return _git_sha(Path(metafilter.__file__).resolve().parent)


def _git_sha(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "installed-package"


def _validated_imint_git_sha(*, required):
    """Return HEAD and reject any uncommitted code for live execution."""
    root = Path(__file__).resolve().parents[1]
    sha = _git_sha(root)
    if not required:
        return sha
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise RuntimeError("live POC requires an exact ImintEngine Git commit")
    expected = os.environ.get("IMINT_GIT_SHA")
    if expected and expected != sha:
        raise RuntimeError(
            f"IMINT_GIT_SHA {expected!r} does not match checked-out HEAD {sha!r}"
        )
    try:
        dirty = subprocess.check_output(
            [
                "git", "-C", str(root), "status", "--porcelain",
                "--untracked-files=all",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("cannot verify ImintEngine working-tree identity") from error
    if dirty:
        raise RuntimeError("live POC refuses a dirty ImintEngine worktree")
    return sha


def _write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", newline="") as handle:
            if rows:
                normalized = []
                for row in rows:
                    normalized.append({
                        key: (
                            json.dumps(value, sort_keys=True)
                            if isinstance(value, (dict, list))
                            else value
                        )
                        for key, value in row.items()
                    })
                fields = sorted({key for row in normalized for key in row})
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(normalized)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
