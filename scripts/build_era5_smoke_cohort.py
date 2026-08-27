#!/usr/bin/env python3
"""Build a persisted, auditable ERA5-cell-disjoint smoke cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imint.training.tile_bbox import resolve_tile_bbox
from imint.training.tile_config import TileConfig
from imint.training.tile_time import (
    resolve_tile_year,
    validate_smoke_temporal_metadata,
)
from imint.training.era5_aux import (
    ERA5_ATMOSPHERE_GRID_DEGREES,
    era5_atmosphere_cell_id,
    era5_grid_context,
    fetch_era5_land_growing_season,
)

COHORT_SCHEMA = "era5-smoke-cohort-v4"
DEFAULT_MAX_LABEL = 22
DEFAULT_MIN_CROP_TRAIN_TILES = 2
DEFAULT_MIN_CROP_VAL_TILES = 2
DEFAULT_MIN_CROP_TRAIN_PIXELS = 1024
DEFAULT_MIN_CROP_VAL_PIXELS = 1024
MODEL_PATCH_PX = 504
TRANSFORMER = Transformer.from_crs(3006, 4326, always_xy=True)
_CROP_CLASS_NAMES = {
    11: "vete",
    12: "korn",
    13: "havre",
    14: "oljeväxter",
    15: "slåttervall",
    16: "bete",
    17: "potatis",
    18: "sockerbetor",
    19: "trindsäd",
    20: "råg",
    21: "majs",
}
_BBOX_KEYS = ("west", "south", "east", "north")
_COMPONENT_FIELDS = ("spatial_component", "split_component")


def _stable(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _label_metadata(label: np.ndarray) -> dict:
    label = np.asarray(label)
    if label.ndim != 2 or label.size == 0:
        raise ValueError(f"label must be a non-empty 2-D array, got {label.shape}")
    if not np.issubdtype(label.dtype, np.integer):
        raise ValueError(f"label must have integer dtype, got {label.dtype}")
    labels, counts = np.unique(label.astype(np.int64, copy=False), return_counts=True)
    contiguous = np.ascontiguousarray(label)
    return {
        "shape": [int(value) for value in label.shape],
        "dtype": str(label.dtype),
        "sha256": hashlib.sha256(contiguous.view(np.uint8)).hexdigest(),
        "min": int(labels[0]),
        "max": int(labels[-1]),
        "pixel_counts": {
            str(int(value)): int(count) for value, count in zip(labels, counts)
        },
    }


def _model_label(label: np.ndarray) -> np.ndarray:
    """Return the exact deterministic validation crop consumed by the model."""
    label = np.asarray(label)
    h, w = label.shape
    if h < MODEL_PATCH_PX or w < MODEL_PATCH_PX:
        raise ValueError(
            f"label {label.shape} is smaller than model patch {MODEL_PATCH_PX}"
        )
    top = (h - MODEL_PATCH_PX) // 2
    left = (w - MODEL_PATCH_PX) // 2
    return label[top:top + MODEL_PATCH_PX, left:left + MODEL_PATCH_PX]


def _tile_metadata(
    path: Path,
    *,
    label_dir: Path | None,
    manifest_path: str | None,
    max_label: int,
) -> dict:
    with np.load(path, allow_pickle=False) as tile:
        size_px = (
            int(tile["tile_size_px"].item())
            if "tile_size_px" in tile.files else 512
        )
        bbox = resolve_tile_bbox(
            name=path.stem,
            tile=TileConfig(size_px=size_px),
            npz_data=tile,
            manifest_path=manifest_path,
        )
        year = resolve_tile_year(tile)
        temporal = (
            validate_smoke_temporal_metadata(tile, year)
            if year is not None else None
        )
    if bbox is None or year is None:
        raise ValueError(f"could not resolve bbox/year for {path.name}")

    label_path = label_dir / path.name if label_dir is not None else path
    if not label_path.is_file():
        raise ValueError(f"missing label source for {path.name}: {label_path}")
    with np.load(label_path, allow_pickle=False) as label_source:
        if "label" not in label_source.files:
            raise ValueError(f"missing label array for {path.name}: {label_path}")
        label = label_source["label"]
        label_meta = _label_metadata(label)
        model_label_meta = _label_metadata(_model_label(label))
    if label_meta["min"] < 0 or label_meta["max"] > max_label:
        raise ValueError(
            f"label range for {path.name} is {label_meta['min']}..{label_meta['max']}; "
            f"expected 0..{max_label}"
        )
    if not any(int(key) > 0 for key in label_meta["pixel_counts"]):
        raise ValueError(f"label for {path.name} has no foreground support")

    easting = (bbox["west"] + bbox["east"]) // 2
    northing = (bbox["south"] + bbox["north"]) // 2
    lon, lat = TRANSFORMER.transform(easting, northing)
    grid = era5_grid_context(lat, lon)
    return {
        "name": path.name,
        "bbox_3006": {key: int(bbox[key]) for key in _BBOX_KEYS},
        "location_3006": {
            "easting": int(easting),
            "northing": int(northing),
        },
        "location_wgs84": {"lat": float(lat), "lon": float(lon)},
        "year": int(year),
        "cutoff_date": temporal["cutoff_date"],
        "temporal": temporal,
        # Split on the coarsest of the two consumed products.  This is the
        # expected ERA5 0.25-degree nearest cell and is later checked against
        # the actual coordinates returned by Open-Meteo for every sidecar.
        "era5_cell": era5_atmosphere_cell_id(lat, lon),
        "era5_request": {
            "lat": grid["request_lat"],
            "lon": grid["request_lon"],
        },
        "era5_land_cell": grid["land_cell"],
        "era5_atmosphere_cell": grid["atmosphere_cell"],
        "label": label_meta,
        "model_label": model_label_meta,
    }


_CEREAL_CLASSES = (11, 12, 13)   # vete / korn / havre — what ERA5 is meant to separate


def _era5_land_covered(items: list[dict], cache_dir: Path) -> list[dict]:
    """Keep only items whose cell ERA5-Land actually covers, order preserved.

    A cell over sea or a large lake gets served by the plain ERA5 reanalysis
    instead, on a 0.25 deg grid rather than 0.1 deg. Detect that by comparing
    the cell the fetch actually returned against the ERA5-Land cell
    ``era5_grid_context`` predicts: if they match, ERA5-Land served it.

    The fetch is cached, so this pre-warms exactly the entries the real fetch
    will need — the probe costs nothing beyond the first call per cell.
    """
    kept: list[dict] = []
    for item in items:
        lat = item["era5_request"]["lat"]
        lon = item["era5_request"]["lon"]
        expected = era5_grid_context(lat, lon)["land_cell"]
        try:
            w = fetch_era5_land_growing_season(
                lat, lon, int(item["year"]), cache_dir=cache_dir,
                cutoff_date=item["cutoff_date"],
            )
        except Exception as exc:  # noqa: BLE001 — an unreachable cell is excluded, not fatal
            print(f"  probe: {item['name']} excluded ({type(exc).__name__}: "
                  f"{str(exc)[:70]})", flush=True)
            continue
        served = (float(w["era5_land_cell_lat"]), float(w["era5_land_cell_lon"]))
        if math.isclose(served[0], expected["lat"], abs_tol=1e-4) and \
           math.isclose(served[1], expected["lon"], abs_tol=1e-4):
            kept.append(item)
        else:
            print(f"  probe: {item['name']} excluded — ERA5-Land gap, served "
                  f"from {served} not ({expected['lat']}, {expected['lon']})",
                  flush=True)
    return kept


def _support(items: list[dict]) -> dict:
    pixel_counts: Counter[int] = Counter()
    tile_counts: Counter[int] = Counter()
    for item in items:
        for raw_label, count in item["model_label"]["pixel_counts"].items():
            label = int(raw_label)
            pixel_counts[label] += int(count)
            tile_counts[label] += 1
    observed = sorted(pixel_counts)
    return {
        "observed_labels": observed,
        "foreground_labels": [label for label in observed if label > 0],
        "pixel_counts": {str(label): pixel_counts[label] for label in observed},
        "tile_counts": {str(label): tile_counts[label] for label in observed},
        "min_label": min(observed),
        "max_label": max(observed),
    }


def _supported_crop_classes(
    support: dict[str, dict],
    thresholds: dict[str, int],
) -> list[str]:
    """Return preregistered crop classes with real train and val support."""
    result: list[str] = []
    for label, name in _CROP_CLASS_NAMES.items():
        key = str(label)
        if (
            int(support["train"]["tile_counts"].get(key, 0))
            >= thresholds["min_train_tiles"]
            and int(support["val"]["tile_counts"].get(key, 0))
            >= thresholds["min_val_tiles"]
            and int(support["train"]["pixel_counts"].get(key, 0))
            >= thresholds["min_train_pixels"]
            and int(support["val"]["pixel_counts"].get(key, 0))
            >= thresholds["min_val_pixels"]
        ):
            result.append(name)
    return result


def _split_text(items: list[dict]) -> str:
    return "".join(f"{item['name']}\n" for item in items)


def _read_split(path: Path) -> tuple[str, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing cohort split: {path}")
    text = path.read_text()
    names = [line.strip() for line in text.splitlines() if line.strip()]
    if not names:
        raise ValueError(f"Cohort split is empty: {path}")
    if len(names) != len(set(names)):
        raise ValueError(f"Cohort split contains duplicate names: {path}")
    return text, names


class _DisjointSet:
    """Minimal deterministic union-find used for cohort component closure."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        # A canonical root makes the result independent of edge discovery order.
        low, high = sorted((left_root, right_root))
        self.parent[high] = low


def _bbox_values(bbox: dict) -> tuple[int, int, int, int]:
    if not isinstance(bbox, dict) or set(bbox) != set(_BBOX_KEYS):
        raise ValueError(f"bbox_3006 must contain exactly {_BBOX_KEYS}: {bbox!r}")
    try:
        west, south, east, north = (int(bbox[key]) for key in _BBOX_KEYS)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"bbox_3006 coordinates must be integers: {bbox!r}") from exc
    if east <= west or north <= south:
        raise ValueError(f"bbox_3006 has non-positive area: {bbox!r}")
    return west, south, east, north


def _bbox_intersection_area(left: dict, right: dict) -> int:
    """Return exact axis-aligned intersection area in square EPSG:3006 metres."""
    left_w, left_s, left_e, left_n = _bbox_values(left)
    right_w, right_s, right_e, right_n = _bbox_values(right)
    width = max(0, min(left_e, right_e) - max(left_w, right_w))
    height = max(0, min(left_n, right_n) - max(left_s, right_s))
    return width * height


def _groups(disjoint_set: _DisjointSet, size: int) -> list[list[int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(size):
        grouped[disjoint_set.find(index)].append(index)
    return list(grouped.values())


def _spatial_component_indices(items: list[dict]) -> list[list[int]]:
    """Find transitive positive-area bbox-overlap components via a sweep line."""
    boxes = [_bbox_values(item["bbox_3006"]) for item in items]
    disjoint_set = _DisjointSet(len(items))
    sweep_order = sorted(
        range(len(items)),
        key=lambda index: (*boxes[index], items[index]["name"]),
    )
    active: list[int] = []
    for index in sweep_order:
        west, south, _, north = boxes[index]
        # east == west means edge contact only, with zero intersection area.
        active = [other for other in active if boxes[other][2] > west]
        for other in active:
            _, other_south, _, other_north = boxes[other]
            if min(north, other_north) > max(south, other_south):
                disjoint_set.union(index, other)
        active.append(index)
    return _groups(disjoint_set, len(items))


def _component_id(kind: str, members: list[dict]) -> str:
    payload = [
        {
            "name": item["name"],
            "bbox_3006": item["bbox_3006"],
            "era5_cell": item["era5_cell"],
            **(
                {"spatial_component": item["spatial_component"]}
                if kind == "split" else {}
            ),
        }
        for item in sorted(members, key=lambda item: item["name"])
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"{kind}:{_sha256_text(encoded)}"


def _bind_components(items: list[dict]) -> dict[str, list[dict]]:
    """Bind bbox components and their ERA5-cell closure to tile metadata.

    ``spatial_component`` is the transitive closure of positive-area bbox
    intersections. ``split_component`` additionally joins every spatial
    component that shares an ERA5 atmosphere cell. Assigning whole split
    components therefore enforces both spatial and ERA5-cell disjointness,
    including a bbox component that bridges two otherwise distinct cells.
    """
    if len({item["name"] for item in items}) != len(items):
        raise ValueError("Cohort candidates contain duplicate tile names")
    if not items:
        return {}

    spatial_groups = _spatial_component_indices(items)
    for indices in spatial_groups:
        members = [items[index] for index in indices]
        component_id = _component_id("spatial", members)
        for item in members:
            item["spatial_component"] = component_id

    split_set = _DisjointSet(len(items))
    for indices in spatial_groups:
        first = indices[0]
        for index in indices[1:]:
            split_set.union(first, index)
    first_by_cell: dict[str, int] = {}
    for index, item in enumerate(items):
        cell = item["era5_cell"]
        if cell in first_by_cell:
            split_set.union(first_by_cell[cell], index)
        else:
            first_by_cell[cell] = index

    result: dict[str, list[dict]] = {}
    for indices in _groups(split_set, len(items)):
        members = [items[index] for index in indices]
        component_id = _component_id("split", members)
        for item in members:
            item["split_component"] = component_id
        result[component_id] = members
    return result


def _validation_component_ids(
    split_components: dict[str, list[dict]],
    *,
    train_tiles: int,
    val_tiles: int,
    seed: int,
) -> set[str]:
    """Choose a deterministic feasible validation-component subset.

    Whole components are assigned to a side, while an assigned side may use a
    deterministic subset of its tiles. A bitset subset-sum finds the smallest
    validation capacity that can satisfy the exact requested tile count while
    leaving enough capacity for training. This avoids a greedy false failure.
    """
    component_ids = sorted(
        split_components,
        key=lambda component_id: _stable(component_id, seed),
    )
    capacities = [len(split_components[component_id]) for component_id in component_ids]
    total_tiles = sum(capacities)
    max_val_capacity = total_tiles - train_tiles
    if max_val_capacity < val_tiles:
        raise ValueError(
            "insufficient compatible, spatial/ERA5-disjoint tiles: "
            f"requested train={train_tiles}, val={val_tiles}; "
            f"available total={total_tiles}"
        )

    reachable = 1  # bit N means that a component subset with N tiles exists.
    mask = (1 << (max_val_capacity + 1)) - 1
    previous_sum = [-1] * (max_val_capacity + 1)
    previous_component = [-1] * (max_val_capacity + 1)
    for component_index, capacity in enumerate(capacities):
        shifted = (reachable << capacity) & mask
        newly_reachable = shifted & ~reachable
        remaining = newly_reachable
        while remaining:
            bit = remaining & -remaining
            tile_sum = bit.bit_length() - 1
            previous_sum[tile_sum] = tile_sum - capacity
            previous_component[tile_sum] = component_index
            remaining ^= bit
        reachable |= shifted

    validation_capacity = next(
        (
            capacity
            for capacity in range(val_tiles, max_val_capacity + 1)
            if reachable & (1 << capacity)
        ),
        None,
    )
    if validation_capacity is None:
        preview = sorted(capacities, reverse=True)[:10]
        raise ValueError(
            "insufficient compatible, spatial/ERA5-disjoint component partition: "
            f"requested train={train_tiles}, val={val_tiles}; "
            f"component capacities (largest first)={preview}"
        )

    chosen: set[str] = set()
    cursor = validation_capacity
    while cursor:
        component_index = previous_component[cursor]
        if component_index < 0:
            raise AssertionError("component subset reconstruction failed")
        chosen.add(component_ids[component_index])
        cursor = previous_sum[cursor]
    return chosen


def _component_envelope(members: list[dict]) -> dict[str, int]:
    boxes = [_bbox_values(item["bbox_3006"]) for item in members]
    return {
        "west": min(box[0] for box in boxes),
        "south": min(box[1] for box in boxes),
        "east": max(box[2] for box in boxes),
        "north": max(box[3] for box in boxes),
    }


def _component_catalog(items: list[dict], field: str) -> list[dict]:
    by_component: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_component[item[field]].append(item)
    catalog: list[dict] = []
    for component_id, members in sorted(by_component.items()):
        entry = {
            "id": component_id,
            "tiles": sorted(item["name"] for item in members),
            "bbox_3006": _component_envelope(members),
            "era5_cells": sorted({item["era5_cell"] for item in members}),
        }
        if field == "spatial_component":
            split_components = {item["split_component"] for item in members}
            if len(split_components) != 1:
                raise AssertionError("spatial component spans split components")
            entry["split_component"] = split_components.pop()
        else:
            entry["spatial_components"] = sorted({
                item["spatial_component"] for item in members
            })
        catalog.append(entry)
    return catalog


def _cross_split_bbox_intersections(
    train: list[dict], val: list[dict],
) -> list[dict]:
    intersections: list[dict] = []
    for train_item in train:
        for val_item in val:
            area = _bbox_intersection_area(
                train_item["bbox_3006"], val_item["bbox_3006"],
            )
            if area > 0:
                intersections.append({
                    "train": train_item["name"],
                    "val": val_item["name"],
                    "area_m2": area,
                })
    return intersections


def _spatial_partition_manifest(train: list[dict], val: list[dict]) -> dict:
    all_items = [*train, *val]
    train_spatial = {item["spatial_component"] for item in train}
    val_spatial = {item["spatial_component"] for item in val}
    train_split = {item["split_component"] for item in train}
    val_split = {item["split_component"] for item in val}
    return {
        "crs": "EPSG:3006",
        "bbox_intersection": "positive_area",
        "component_definition": (
            "transitive_closure_of_positive_area_bbox_intersections"
        ),
        "split_component_definition": (
            "transitive_closure_of_spatial_components_and_shared_era5_cells"
        ),
        "spatial_components": {
            "train": sorted(train_spatial),
            "val": sorted(val_spatial),
            "overlap": sorted(train_spatial & val_spatial),
        },
        "split_components": {
            "train": sorted(train_split),
            "val": sorted(val_split),
            "overlap": sorted(train_split & val_split),
        },
        "spatial_component_catalog": _component_catalog(
            all_items, "spatial_component",
        ),
        "split_component_catalog": _component_catalog(
            all_items, "split_component",
        ),
        "bbox_area_intersections": _cross_split_bbox_intersections(train, val),
    }


def build_cohort(
    *,
    tile_dir: Path,
    train_tiles: int,
    val_tiles: int,
    seed: int,
    max_label: int = DEFAULT_MAX_LABEL,
    min_val_crop_classes: int = 5,
    min_crop_train_tiles: int = DEFAULT_MIN_CROP_TRAIN_TILES,
    min_crop_val_tiles: int = DEFAULT_MIN_CROP_VAL_TILES,
    min_crop_train_pixels: int = DEFAULT_MIN_CROP_TRAIN_PIXELS,
    min_crop_val_pixels: int = DEFAULT_MIN_CROP_VAL_PIXELS,
    label_dir: Path | None = None,
    manifest_path: str | None = None,
    prefer_cereal_tiles: bool = False,
    era5_land_probe_cache: Path | None = None,
    oversample: float = 1.25,
) -> tuple[dict, str, str]:
    """Select a deterministic cohort and return its manifest and split text."""
    if train_tiles <= 0 or val_tiles <= 0:
        raise ValueError("train_tiles and val_tiles must be positive")
    if max_label < 1:
        raise ValueError("max_label must include at least one foreground class")
    if max_label != DEFAULT_MAX_LABEL:
        raise ValueError("ERA5 smoke cohort is fixed to 23 logits and max_label=22")
    if not 5 <= min_val_crop_classes <= len(_CROP_CLASS_NAMES):
        raise ValueError(
            f"min_val_crop_classes must be 5..{len(_CROP_CLASS_NAMES)}"
        )
    thresholds = {
        "min_train_tiles": min_crop_train_tiles,
        "min_val_tiles": min_crop_val_tiles,
        "min_train_pixels": min_crop_train_pixels,
        "min_val_pixels": min_crop_val_pixels,
    }
    if any(not isinstance(value, int) or value <= 0 for value in thresholds.values()):
        raise ValueError(f"Crop support thresholds must be positive integers: {thresholds}")

    bbox_manifest_sha256 = (
        _sha256_file(Path(manifest_path)) if manifest_path is not None else None
    )

    candidates: list[dict] = []
    skipped: list[dict[str, str]] = []
    for path in sorted(tile_dir.glob("*.npz")):
        try:
            item = _tile_metadata(
                path,
                label_dir=label_dir,
                manifest_path=manifest_path,
                max_label=max_label,
            )
            candidates.append(item)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            skipped.append({"name": path.name, "reason": str(exc)})

    split_components = _bind_components(candidates)
    val_component_ids = _validation_component_ids(
        split_components,
        train_tiles=train_tiles,
        val_tiles=val_tiles,
        seed=seed,
    )
    train_pool = [
        item
        for component_id, items in split_components.items()
        if component_id not in val_component_ids
        for item in items
    ]
    val_pool = [
        item
        for component_id, items in split_components.items()
        if component_id in val_component_ids
        for item in items
    ]
    # Uniform hash-order put vete in 13 of 128 val tiles and cereals at 0.62% of
    # val pixels (audit 2026-08-26), which cannot resolve a per-class effect —
    # one tile's outcome dominates the IoU. Rank cereal-bearing tiles first and
    # keep the hash tie-break, so selection stays deterministic and seed-stable.
    def _rank(item: dict):
        if not prefer_cereal_tiles:
            return (0, _stable(item["name"], seed))
        counts = item["model_label"]["pixel_counts"]
        cereal_px = sum(int(counts.get(str(c), 0)) for c in _CEREAL_CLASSES)
        return (0 if cereal_px else 1, _stable(item["name"], seed))

    # ERA5-Land has no ocean coverage, so ~9% of cells fall back to the 0.25 deg
    # ERA5 grid — a different cell centre that six separate validators reject
    # (2026-08-26). Rather than teach every one of them to tolerate it, exclude
    # those cells here: over-select, probe, and keep the first N that ERA5-Land
    # actually covers. Probing writes through the shared fetch cache, so the
    # real fetch later reuses it and nothing is thrown away.
    take_train, take_val = train_tiles, val_tiles
    if era5_land_probe_cache is not None:
        take_train = int(train_tiles * oversample) + 1
        take_val = int(val_tiles * oversample) + 1

    train = sorted(train_pool, key=_rank)[:take_train]
    val = sorted(val_pool, key=_rank)[:take_val]

    if era5_land_probe_cache is not None:
        n_train_cand, n_val_cand = len(train), len(val)
        train = _era5_land_covered(train, era5_land_probe_cache)[:train_tiles]
        val = _era5_land_covered(val, era5_land_probe_cache)[:val_tiles]
        # Over-selection only helps where the pool has slack. train_pool has
        # thousands spare; val_pool is constrained by spatial disjointness to
        # roughly the requested size, so a single exclusion can make the count
        # unreachable. Say that plainly — the generic "insufficient compatible
        # tiles" error below blames pool construction and sent the reader
        # looking in the wrong place.
        for label, got, want, cand in (("train", len(train), train_tiles, n_train_cand),
                                       ("val", len(val), val_tiles, n_val_cand)):
            if got < want:
                raise ValueError(
                    f"ERA5-Land probe left too few {label} tiles: {got} of "
                    f"{want} wanted, from {cand} candidates "
                    f"(oversample={oversample}). The {label} pool cannot "
                    f"absorb the ~9% of cells ERA5-Land does not cover — "
                    f"lower --{label}-tiles or raise --oversample if the pool "
                    f"has room."
                )
    if len(train) != train_tiles or len(val) != val_tiles:
        raise ValueError(
            "insufficient compatible, spatial/ERA5-disjoint tiles: "
            f"requested train={train_tiles}, val={val_tiles}; "
            f"available train={len(train_pool)}, val={len(val_pool)}; skipped={len(skipped)}"
        )

    # Rebind the manifest components to the exact selected cohort. Candidate
    # components have already governed allocation; selected-only identities
    # make the persisted component contract independently reproducible.
    _bind_components([*train, *val])

    # Bind every consumed source byte only after selection. Hashing the full
    # corpus would re-read hundreds of GB; hashing the 384 selected NPZ files
    # proves spectral, dates, bbox, labels, and all existing AUX inputs are
    # identical across the sequential A/B arms.
    for item in (*train, *val):
        item["source_npz_sha256"] = _sha256_file(tile_dir / item["name"])

    train_cell_set = {item["era5_cell"] for item in train}
    val_cell_set = {item["era5_cell"] for item in val}
    if not train_cell_set.isdisjoint(val_cell_set):
        raise AssertionError("ERA5 cells overlap after cohort selection")
    spatial_partition = _spatial_partition_manifest(train, val)
    if spatial_partition["spatial_components"]["overlap"]:
        raise AssertionError("spatial components overlap after cohort selection")
    if spatial_partition["split_components"]["overlap"]:
        raise AssertionError("split components overlap after cohort selection")
    if spatial_partition["bbox_area_intersections"]:
        first = spatial_partition["bbox_area_intersections"][0]
        raise AssertionError(
            "train/val bbox intersection area must be zero; "
            f"first overlap={first}"
        )

    train_text = _split_text(train)
    val_text = _split_text(val)
    support = {"train": _support(train), "val": _support(val)}
    if not support["train"]["foreground_labels"] or not support["val"]["foreground_labels"]:
        raise ValueError("both cohort splits must contain foreground label support")

    unseen_val_labels = (
        set(support["val"]["foreground_labels"])
        - set(support["train"]["foreground_labels"])
    )
    if unseen_val_labels:
        raise ValueError(
            "validation contains classes absent from training: "
            f"{sorted(unseen_val_labels)}"
        )
    val_supported_crop_classes = _supported_crop_classes(support, thresholds)
    if len(val_supported_crop_classes) < min_val_crop_classes:
        raise ValueError(
            "validation split has insufficient crop-class support: "
            f"required={min_val_crop_classes}, eligible={val_supported_crop_classes}, "
            f"thresholds={thresholds}"
        )

    manifest = {
        "schema": COHORT_SCHEMA,
        "seed": seed,
        "source": {
            "tile_dir": str(tile_dir),
            "label_dir": str(label_dir) if label_dir is not None else None,
            "bbox_manifest": manifest_path,
            "bbox_manifest_sha256": bbox_manifest_sha256,
        },
        "label_schema": {
            "num_logits": max_label + 1,
            "min_label": 0,
            "max_label": max_label,
        },
        "model_patch_px": MODEL_PATCH_PX,
        "counts": {"train": len(train), "val": len(val)},
        "split_sha256": {
            "train": _sha256_text(train_text),
            "val": _sha256_text(val_text),
        },
        "era5_cells": {
            "train": sorted(train_cell_set),
            "val": sorted(val_cell_set),
            "overlap": [],
        },
        "era5_cell_definition": {
            "model": "era5",
            "resolution_degrees": float(ERA5_ATMOSPHERE_GRID_DEGREES),
            "cell_selection": "nearest",
            "identity": "verified_response_latitude_longitude",
        },
        "spatial_partition": spatial_partition,
        "year_counts": {
            "train": dict(sorted(Counter(item["year"] for item in train).items())),
            "val": dict(sorted(Counter(item["year"] for item in val).items())),
        },
        "label_support": support,
        "min_val_crop_classes": min_val_crop_classes,
        "crop_support_thresholds": thresholds,
        "val_supported_crop_classes": val_supported_crop_classes,
        "splits": {"train": train, "val": val},
        "skipped": skipped,
    }
    return manifest, train_text, val_text


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def write_cohort(output_dir: Path, manifest: dict, train_text: str, val_text: str) -> None:
    """Persist split files first and the hash-bearing manifest last."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(output_dir / "split_train.txt", train_text)
    _atomic_write(output_dir / "split_val.txt", val_text)
    _atomic_write(output_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")


def validate_existing_cohort(
    *,
    tile_dir: Path,
    output_dir: Path,
    label_dir: Path | None = None,
    manifest_path: str | None = None,
) -> dict:
    """Validate a persisted cohort against its splits and current tile labels."""
    cohort_manifest_path = output_dir / "manifest.json"
    if not cohort_manifest_path.is_file():
        raise FileNotFoundError(f"Missing cohort manifest: {cohort_manifest_path}")
    try:
        manifest = json.loads(cohort_manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid cohort manifest: {cohort_manifest_path}") from exc
    if manifest.get("schema") != COHORT_SCHEMA:
        raise ValueError(
            f"Unsupported cohort schema {manifest.get('schema')!r}; expected {COHORT_SCHEMA!r}"
        )

    label_schema = manifest.get("label_schema", {})
    max_label = label_schema.get("max_label")
    if label_schema != {"num_logits": 23, "min_label": 0, "max_label": 22}:
        raise ValueError(f"Expected 23-logit label schema 0..22, got {label_schema}")
    if manifest.get("model_patch_px") != MODEL_PATCH_PX:
        raise ValueError(f"Expected model_patch_px={MODEL_PATCH_PX}")

    source = manifest.get("source", {})
    recorded_bbox_manifest = source.get("bbox_manifest")
    effective_manifest_path = manifest_path or recorded_bbox_manifest
    if manifest_path is not None and manifest_path != recorded_bbox_manifest:
        raise ValueError(
            "Validation bbox manifest does not match the persisted cohort source"
        )
    actual_bbox_manifest_sha256 = (
        _sha256_file(Path(effective_manifest_path))
        if effective_manifest_path is not None else None
    )
    if source.get("bbox_manifest_sha256") != actual_bbox_manifest_sha256:
        raise ValueError("Cohort bbox manifest is stale or has changed")

    manifest_splits = manifest.get("splits", {})
    if not all(isinstance(manifest_splits.get(split), list) for split in ("train", "val")):
        raise ValueError("Cohort manifest must contain train and val tile metadata")
    raw_cells = {
        split: {item.get("era5_cell") for item in manifest_splits[split]}
        for split in ("train", "val")
    }
    if any(None in values for values in raw_cells.values()):
        raise ValueError("Cohort tile metadata is missing an ERA5 cell")
    cells = {split: sorted(values) for split, values in raw_cells.items()}
    cell_overlap = set(cells["train"]) & set(cells["val"])
    if cell_overlap:
        raise ValueError(f"Cohort ERA5 cells overlap: {sorted(cell_overlap)[:3]}")
    if manifest.get("era5_cells") != {
        "train": cells["train"], "val": cells["val"], "overlap": [],
    }:
        raise ValueError("Cohort ERA5-cell manifest is stale or inconsistent")

    split_names: dict[str, list[str]] = {}
    actual_splits: dict[str, list[dict]] = {"train": [], "val": []}
    expected_by_name: dict[str, dict] = {}
    for split in ("train", "val"):
        text, names = _read_split(output_dir / f"split_{split}.txt")
        split_names[split] = names
        if manifest.get("split_sha256", {}).get(split) != _sha256_text(text):
            raise ValueError(f"Stale or partial split_{split}.txt: manifest hash mismatch")
        if manifest.get("counts", {}).get(split) != len(names):
            raise ValueError(f"Cohort {split} count does not match split file")
        expected_items = manifest.get("splits", {}).get(split)
        if (
            not isinstance(expected_items, list)
            or [item.get("name") for item in expected_items] != names
        ):
            raise ValueError(f"Cohort {split} tile list does not match split file")
        if manifest.get("label_support", {}).get(split) != _support(expected_items):
            raise ValueError(f"Cohort {split} label support is stale or inconsistent")
        for expected in expected_items:
            if any(
                not isinstance(expected.get(field), str) or not expected[field]
                for field in _COMPONENT_FIELDS
            ):
                raise ValueError(
                    f"Cohort tile {expected.get('name')} is missing component identity"
                )
            if expected["name"] in expected_by_name:
                raise ValueError(
                    f"Cohort tile appears more than once: {expected['name']}"
                )
            expected_by_name[expected["name"]] = expected
            path = tile_dir / expected["name"]
            if not path.is_file():
                raise FileNotFoundError(f"Cohort tile is missing: {path}")
            actual = _tile_metadata(
                path,
                label_dir=label_dir,
                manifest_path=effective_manifest_path,
                max_label=max_label,
            )
            actual["source_npz_sha256"] = _sha256_file(path)
            actual_splits[split].append(actual)

    _bind_components([*actual_splits["train"], *actual_splits["val"]])
    for split in ("train", "val"):
        for actual in actual_splits[split]:
            expected = expected_by_name[actual["name"]]
            if actual != expected:
                raise ValueError(
                    f"Cohort metadata is stale for {actual['name']}: "
                    f"expected label={expected.get('label', {}).get('sha256')}, "
                    f"actual label={actual['label']['sha256']}"
                )

    spatial_partition = _spatial_partition_manifest(
        actual_splits["train"], actual_splits["val"],
    )
    if spatial_partition["spatial_components"]["overlap"]:
        raise ValueError("Cohort spatial components overlap across train/val")
    if spatial_partition["split_components"]["overlap"]:
        raise ValueError("Cohort split components overlap across train/val")
    if spatial_partition["bbox_area_intersections"]:
        first = spatial_partition["bbox_area_intersections"][0]
        raise ValueError(
            "Cohort train/val bbox intersection area is non-zero: "
            f"{first}"
        )
    if manifest.get("spatial_partition") != spatial_partition:
        raise ValueError("Cohort spatial-partition manifest is stale or inconsistent")

    train_foreground = set(manifest["label_support"]["train"]["foreground_labels"])
    val_foreground = set(manifest["label_support"]["val"]["foreground_labels"])
    unseen_val_labels = val_foreground - train_foreground
    if unseen_val_labels:
        raise ValueError(
            "validation contains classes absent from training: "
            f"{sorted(unseen_val_labels)}"
        )
    thresholds = manifest.get("crop_support_thresholds")
    required_threshold_keys = {
        "min_train_tiles", "min_val_tiles",
        "min_train_pixels", "min_val_pixels",
    }
    if (
        not isinstance(thresholds, dict)
        or set(thresholds) != required_threshold_keys
        or any(not isinstance(value, int) or value <= 0 for value in thresholds.values())
    ):
        raise ValueError(f"Invalid crop_support_thresholds: {thresholds}")
    expected_crop_classes = _supported_crop_classes(
        manifest["label_support"], thresholds,
    )
    if manifest.get("val_supported_crop_classes") != expected_crop_classes:
        raise ValueError(
            "val_supported_crop_classes does not match validation label support"
        )
    min_val_crop_classes = manifest.get("min_val_crop_classes")
    if (
        not isinstance(min_val_crop_classes, int)
        or min_val_crop_classes < 5
        or len(expected_crop_classes) < min_val_crop_classes
    ):
        raise ValueError(
            "validation split does not satisfy min_val_crop_classes >= 5"
        )

    expected_cell_definition = {
        "model": "era5",
        "resolution_degrees": float(ERA5_ATMOSPHERE_GRID_DEGREES),
        "cell_selection": "nearest",
        "identity": "verified_response_latitude_longitude",
    }
    if manifest.get("era5_cell_definition") != expected_cell_definition:
        raise ValueError("Cohort ERA5 cell definition is stale or unsupported")

    name_overlap = set(split_names["train"]) & set(split_names["val"])
    if name_overlap:
        raise ValueError(f"Cohort split names overlap: {sorted(name_overlap)[:3]}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-tiles", type=int, default=256)
    parser.add_argument("--val-tiles", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--max-label", type=int, default=DEFAULT_MAX_LABEL)
    parser.add_argument("--min-val-crop-classes", type=int, default=5)
    parser.add_argument(
        "--prefer-cereal-tiles", action="store_true",
        help="Rank tiles containing vete/korn/havre ahead of the rest. Uniform "
             "sampling put vete in 13 of 128 val tiles (cereals 0.62%% of val "
             "pixels), too thin for a per-class conclusion.",
    )
    parser.add_argument(
        "--era5-land-probe-cache", type=Path, default=None,
        help="Probe ERA5-Land coverage for over-selected candidates and drop "
             "cells it does not cover, so the cohort never needs the 0.25 deg "
             "ERA5 fallback that six grid validators reject. Writes through "
             "this fetch cache, which the real fetch then reuses.",
    )
    parser.add_argument(
        "--oversample", type=float, default=1.25,
        help="Candidate multiplier when probing (~9%% of cells fall back).",
    )
    parser.add_argument(
        "--min-crop-train-tiles", type=int,
        default=DEFAULT_MIN_CROP_TRAIN_TILES,
    )
    parser.add_argument(
        "--min-crop-val-tiles", type=int,
        default=DEFAULT_MIN_CROP_VAL_TILES,
    )
    parser.add_argument(
        "--min-crop-train-pixels", type=int,
        default=DEFAULT_MIN_CROP_TRAIN_PIXELS,
    )
    parser.add_argument(
        "--min-crop-val-pixels", type=int,
        default=DEFAULT_MIN_CROP_VAL_PIXELS,
    )
    parser.add_argument("--label-dir", type=Path)
    parser.add_argument("--manifest-path")
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="Validate the existing manifest and splits against current tiles; do not rebuild.",
    )
    args = parser.parse_args()

    if args.validate_existing:
        manifest = validate_existing_cohort(
            tile_dir=args.tile_dir,
            output_dir=args.output_dir,
            label_dir=args.label_dir,
            manifest_path=args.manifest_path,
        )
        print(json.dumps({
            "status": "valid",
            "schema": manifest["schema"],
            "counts": manifest["counts"],
            "label_schema": manifest["label_schema"],
            "era5_cell_overlap": len(manifest["era5_cells"]["overlap"]),
            "min_val_crop_classes": manifest["min_val_crop_classes"],
            "val_supported_crop_classes": manifest["val_supported_crop_classes"],
        }, indent=2))
        return

    manifest, train_text, val_text = build_cohort(
        tile_dir=args.tile_dir,
        train_tiles=args.train_tiles,
        val_tiles=args.val_tiles,
        seed=args.seed,
        max_label=args.max_label,
        min_val_crop_classes=args.min_val_crop_classes,
        min_crop_train_tiles=args.min_crop_train_tiles,
        min_crop_val_tiles=args.min_crop_val_tiles,
        min_crop_train_pixels=args.min_crop_train_pixels,
        min_crop_val_pixels=args.min_crop_val_pixels,
        label_dir=args.label_dir,
        manifest_path=args.manifest_path,
        prefer_cereal_tiles=args.prefer_cereal_tiles,
        era5_land_probe_cache=args.era5_land_probe_cache,
        oversample=args.oversample,
    )
    write_cohort(args.output_dir, manifest, train_text, val_text)
    print(json.dumps({
        "train": manifest["counts"]["train"],
        "val": manifest["counts"]["val"],
        "train_cells": len(manifest["era5_cells"]["train"]),
        "val_cells": len(manifest["era5_cells"]["val"]),
        "overlap": len(manifest["era5_cells"]["overlap"]),
        "label_support": manifest["label_support"],
        "min_val_crop_classes": manifest["min_val_crop_classes"],
        "val_supported_crop_classes": manifest["val_supported_crop_classes"],
        "year_counts": manifest["year_counts"],
        "skipped": len(manifest["skipped"]),
    }, indent=2))


if __name__ == "__main__":
    main()
