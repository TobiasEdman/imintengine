#!/usr/bin/env python3
"""Persist ERA5-Land growing-season AUX sidecars for unified tiles."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imint.training.era5_aux import (
    era5_api_cell_coords_match,
    era5_atmosphere_cell_id,
    era5_grid_context,
    fetch_era5_land_growing_season,
    format_era5_cell_id,
)
from imint.training.tile_time import resolve_growing_cutoff, resolve_tile_year
from imint.training.tile_bbox import resolve_tile_bbox
from imint.training.tile_config import TileConfig

COVERAGE_SCHEMA = "era5-smoke-sidecar-bundle-v1"
COHORT_SCHEMA = "era5-smoke-cohort-v4"
WEATHER_FIELDS = (
    "era5_t2m_mean", "era5_tp_sum", "era5_swvl1_mean",
    "era5_ssrd_sum", "era5_gdd",
)
GRID_FIELDS = (
    "era5_request_lat", "era5_request_lon",
    "era5_land_cell_lat", "era5_land_cell_lon",
    "era5_atmosphere_cell_lat", "era5_atmosphere_cell_lon",
)


def _grid_matches(actual: dict, expected: dict) -> bool:
    # Request coordinates are produced and persisted by us, so retain the
    # strict identity check.  Only API-selected response cells may contain
    # Open-Meteo's float-serialization noise.
    request_pairs = (
        (actual["request_lat"], expected["request_lat"]),
        (actual["request_lon"], expected["request_lon"]),
    )
    if not all(
        np.isfinite(float(a))
        and np.isfinite(float(b))
        and np.isclose(a, b, rtol=0.0, atol=1e-7)
        for a, b in request_pairs
    ):
        return False
    return (
        era5_api_cell_coords_match(
            actual["land_cell"]["lat"],
            actual["land_cell"]["lon"],
            expected["land_cell"]["lat"],
            expected["land_cell"]["lon"],
        )
        and era5_api_cell_coords_match(
            actual["atmosphere_cell"]["lat"],
            actual["atmosphere_cell"]["lon"],
            expected["atmosphere_cell"]["lat"],
            expected["atmosphere_cell"]["lon"],
        )
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bundle_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def tile_context(path: Path, *, manifest_path: str | None = None) -> tuple[float, float, int, str]:
    """Return WGS84 centroid and spectral label year from one tile."""
    with np.load(path, allow_pickle=False) as tile:
        size = int(tile["tile_size_px"].item()) if "tile_size_px" in tile.files else 512
        bbox = resolve_tile_bbox(
            name=path.stem, tile=TileConfig(size_px=size), npz_data=tile,
            manifest_path=manifest_path,
        )
        year = resolve_tile_year(tile)
        if bbox is None or year is None:
            raise ValueError(f"Could not resolve bbox/year for {path.name}")
        try:
            cutoff = resolve_growing_cutoff(tile, year)
        except ValueError as exc:
            raise ValueError(f"Invalid temporal metadata in {path.name}: {exc}") from exc
    west, south, east, north = (bbox[k] for k in ("west", "south", "east", "north"))
    lon, lat = Transformer.from_crs(3006, 4326, always_xy=True).transform(
        (west + east) / 2.0, (south + north) / 2.0,
    )
    return lat, lon, year, cutoff


def _valid_sidecar(
    path: Path,
    tile_name: str,
    year: int,
    cutoff: str,
    lat: float,
    lon: float,
    expected_item: dict[str, Any] | None = None,
) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {
                *WEATHER_FIELDS, *GRID_FIELDS,
                "tile_name", "year", "cutoff_date", "lat", "lon",
            }
            if not required <= set(data.files):
                return False
            if not (
                str(data["tile_name"].item()) == tile_name
                and int(data["year"].item()) == year
                and str(data["cutoff_date"].item()) == cutoff
                and np.isclose(float(data["lat"].item()), lat, rtol=0.0, atol=1e-7)
                and np.isclose(float(data["lon"].item()), lon, rtol=0.0, atol=1e-7)
                and all(
                    np.isfinite(float(data[name].item()))
                    for name in (*WEATHER_FIELDS, *GRID_FIELDS)
                )
            ):
                return False
            grid = era5_grid_context(lat, lon)
            actual_grid = {
                "request_lat": float(data["era5_request_lat"].item()),
                "request_lon": float(data["era5_request_lon"].item()),
                "land_cell": {
                    "lat": float(data["era5_land_cell_lat"].item()),
                    "lon": float(data["era5_land_cell_lon"].item()),
                },
                "atmosphere_cell": {
                    "lat": float(data["era5_atmosphere_cell_lat"].item()),
                    "lon": float(data["era5_atmosphere_cell_lon"].item()),
                },
            }
            if not _grid_matches(actual_grid, grid):
                return False
            if expected_item is not None:
                expected_item_grid = {
                    "request_lat": expected_item.get("era5_request", {}).get("lat"),
                    "request_lon": expected_item.get("era5_request", {}).get("lon"),
                    "land_cell": expected_item.get("era5_land_cell", {}),
                    "atmosphere_cell": expected_item.get(
                        "era5_atmosphere_cell", {},
                    ),
                }
                if not (
                    expected_item.get("year") == year
                    and expected_item.get("cutoff_date") == cutoff
                    and _grid_matches(actual_grid, expected_item_grid)
                    and expected_item.get("era5_cell")
                    == format_era5_cell_id(
                        actual_grid["atmosphere_cell"]["lat"],
                        actual_grid["atmosphere_cell"]["lon"],
                    )
                ):
                    return False
            return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _atomic_savez(path: Path, values: dict) -> None:
    fd, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".npz.tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **values)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
        raise


def _write_once_or_verify_json(path: Path, values: dict) -> None:
    """Publish immutable coverage metadata, accepting an exact retry."""
    if path.exists():
        try:
            actual = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid existing coverage: {path}") from exc
        if actual != values:
            raise ValueError(f"Refusing to overwrite mismatched coverage: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".create",
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(values, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            actual = json.loads(path.read_text())
            if actual != values:
                raise ValueError(
                    f"Refusing to overwrite mismatched coverage: {path}"
                )
    except BaseException:
        raise
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_cohort(cohort_dir: Path) -> tuple[dict, dict[str, dict], dict[str, str]]:
    manifest_path = cohort_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid cohort manifest: {manifest_path}") from exc
    if manifest.get("schema") != COHORT_SCHEMA:
        raise ValueError(
            f"Expected {COHORT_SCHEMA}, got {manifest.get('schema')!r}"
        )
    items: dict[str, dict] = {}
    splits: dict[str, str] = {}
    for split in ("train", "val"):
        split_path = cohort_dir / f"split_{split}.txt"
        names = [
            line.strip() for line in split_path.read_text().splitlines()
            if line.strip()
        ]
        declared = manifest.get("splits", {}).get(split)
        if not isinstance(declared, list) or [item.get("name") for item in declared] != names:
            raise ValueError(f"Cohort {split} split and manifest disagree")
        for item in declared:
            if item["name"] in items:
                raise ValueError(f"Duplicate cohort tile: {item['name']}")
            items[item["name"]] = item
            splits[item["name"]] = split
    return manifest, items, splits


def _coverage_entry(
    sidecar: Path,
    *,
    context: tuple[float, float, int, str],
    item: dict,
    split: str,
) -> dict[str, Any]:
    lat, lon, year, cutoff = context
    if not _valid_sidecar(
        sidecar, sidecar.name, year, cutoff, lat, lon, item,
    ):
        raise ValueError(f"Invalid ERA5 sidecar: {sidecar}")
    with np.load(sidecar, allow_pickle=False) as data:
        land_cell = format_era5_cell_id(
            float(data["era5_land_cell_lat"].item()),
            float(data["era5_land_cell_lon"].item()),
        )
        atmosphere_cell = format_era5_cell_id(
            float(data["era5_atmosphere_cell_lat"].item()),
            float(data["era5_atmosphere_cell_lon"].item()),
        )
    return {
        "name": sidecar.name,
        "split": split,
        "year": year,
        "cutoff_date": cutoff,
        "source_lat": lat,
        "source_lon": lon,
        "land_cell": land_cell,
        "atmosphere_cell": atmosphere_cell,
        "sidecar_sha256": _sha256_file(sidecar),
    }


def _build_coverage(
    *,
    output_dir: Path,
    cohort_dir: Path,
    cohort_manifest: dict,
    cohort_items: dict[str, dict],
    splits: dict[str, str],
    contexts: dict[str, tuple[float, float, int, str]],
) -> dict[str, Any]:
    entries = [
        _coverage_entry(
            output_dir / name,
            context=contexts[name],
            item=cohort_items[name],
            split=splits[name],
        )
        for name in sorted(cohort_items)
    ]
    cells = {
        split: sorted({
            entry["atmosphere_cell"]
            for entry in entries if entry["split"] == split
        })
        for split in ("train", "val")
    }
    overlap = sorted(set(cells["train"]) & set(cells["val"]))
    if overlap:
        raise ValueError(
            f"Actual ERA5 atmosphere cells overlap: {overlap[:3]}"
        )
    sidecars = [output_dir / entry["name"] for entry in entries]
    return {
        "schema": COVERAGE_SCHEMA,
        "cohort_manifest_sha256": _sha256_file(cohort_dir / "manifest.json"),
        "cohort_split_sha256": cohort_manifest["split_sha256"],
        "requested": len(entries),
        "valid": len(entries),
        "atmosphere_cells": {
            "train": cells["train"],
            "val": cells["val"],
            "overlap": [],
        },
        "sidecar_bundle_sha256": _bundle_sha256(sidecars),
        "tiles": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--cohort-dir", required=True, type=Path)
    parser.add_argument("--manifest-path", type=str)
    parser.add_argument(
        "--validate-existing", action="store_true",
        help="Validate the sealed sidecar bundle without fetching or writing.",
    )
    args = parser.parse_args()

    cohort_manifest, cohort_items, splits = _load_cohort(args.cohort_dir)
    paths = [args.tile_dir / name for name in sorted(cohort_items)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cohort contains missing source tiles: {missing[:3]}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contexts = {
        path.name: tile_context(path, manifest_path=args.manifest_path)
        for path in paths
    }
    coverage_path = args.output_dir / "coverage.json"
    sealed = coverage_path.exists()
    if args.validate_existing and not sealed:
        raise FileNotFoundError(f"ERA5 coverage is not sealed: {coverage_path}")

    for i, path in enumerate(paths, 1):
        output = args.output_dir / path.name
        lat, lon, year, cutoff = contexts[path.name]
        expected_item = cohort_items[path.name]
        if output.exists() and _valid_sidecar(
            output, path.name, year, cutoff, lat, lon, expected_item,
        ):
            continue
        if args.validate_existing or sealed:
            raise ValueError(f"Sealed ERA5 sidecar is missing or invalid: {output}")
        weather = fetch_era5_land_growing_season(
            lat, lon, year, cache_dir=args.cache_dir, cutoff_date=cutoff,
        )
        _atomic_savez(output, {
            **weather, "tile_name": path.name, "lat": lat, "lon": lon,
            "year": year, "cutoff_date": cutoff,
        })
        if i % 25 == 0:
            print(f"ERA5 {i}/{len(paths)}", flush=True)
    coverage = _build_coverage(
        output_dir=args.output_dir,
        cohort_dir=args.cohort_dir,
        cohort_manifest=cohort_manifest,
        cohort_items=cohort_items,
        splits=splits,
        contexts=contexts,
    )
    _write_once_or_verify_json(coverage_path, coverage)
    print(json.dumps({
        "status": "valid" if args.validate_existing else "sealed",
        "requested": coverage["requested"],
        "valid": coverage["valid"],
        "sidecar_bundle_sha256": coverage["sidecar_bundle_sha256"],
        "atmosphere_cell_overlap": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
