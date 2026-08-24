"""
imint/training/unified_dataset.py -- Unified PyTorch Dataset for LULC + Crop segmentation

Loads tiles from unified_v2/ producing 23-class per-pixel segmentation labels.
Labels are pre-built offline by build_labels.py (NMD 19-class + gated LPIS + gated SKS)
and stored directly in data["label"] — no runtime merging.

Tile sources (unified_v2/*.npz):
    spectral  (24, 256, 256) = 4 frames × 6 bands, reflectance [0, 1]
    label     (256, 256)     unified 23-class label (pre-built by build_labels.py)
    label_mask (256, 256)    raw SJV grödkoder (uint16, for reference)
    harvest_mask (256, 256)  binary SKS harvest indicator
    bbox_3006 (4,)           [west, south, east, north] in SWEREF 99 TM
    height/volume/basal_area/diameter/dem (256, 256) forest/terrain aux
    vpp_sosd/vpp_eosd (256, 256) phenology aux
    dates (4,), temporal_mask (4,), doy (4,)

Output dict (matches LULCDataset format for trainer.py compatibility):
    image:               (6, 224, 224) float32, single-date Prithvi-normalized
    label:               (224, 224) int64, unified 19-class per-pixel
    height:              (1, 224, 224) float32, z-score normalized
    volume:              (1, 224, 224) float32
    basal_area:          (1, 224, 224) float32
    diameter:            (1, 224, 224) float32
    dem:                 (1, 224, 224) float32
    vpp_sosd:            (1, 224, 224) float32
    vpp_eosd:            (1, 224, 224) float32
    metadata:            {"tile": str, "source": "lulc" | "crop"}

Unified 19-class schema (from unified_schema.py + harvest):
    0  bakgrund (ignore_index)
    1  tallskog            (NMD pine)
    2  granskog            (NMD spruce)
    3  lovskog             (NMD deciduous)
    4  blandskog           (NMD mixed)
    5  sumpskog            (NMD wetland forest)
    6  vatmark             (NMD open wetland)
    7  oppen mark          (NMD open land)
    8  bebyggelse          (NMD developed)
    9  vatten              (NMD water)
    10 vete                (LPIS wheat)
    11 korn                (LPIS barley)
    12 havre               (LPIS oats)
    13 oljevaxter          (LPIS rapeseed)
    14 vall                (LPIS ley/grass)
    15 potatis             (LPIS potato)
    16 trindsad            (LPIS pulses)
    17 ovrig aker          (LPIS other / unmapped cropland)
    18 hygge               (harvested forest)
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from pathlib import Path
from typing import Sequence

import numpy as np
from pyproj import Transformer

from .era5_aux import (
    ERA5_ATMOSPHERE_GRID_DEGREES,
    ERA5_AUX_NORM,
    era5_api_cell_coords_match,
    era5_grid_context,
    format_era5_cell_id,
)
from .tile_bbox import resolve_tile_bbox
from .tile_config import TileConfig
from .tile_time import (
    resolve_growing_cutoff,
    resolve_tile_year,
    validate_smoke_temporal_metadata,
)

try:
    import torch
    from torch.utils.data import Dataset, WeightedRandomSampler
except ImportError:
    raise ImportError(
        "PyTorch is required for training. Install with: pip install torch"
    )

from .unified_schema import NUM_UNIFIED_CLASSES, HARVEST_CLASS
from .losses import parcel_area_to_pixel_weights
from .sampler import _sweref99_to_wgs84

logger = logging.getLogger(__name__)

_SWEREF99_TO_WGS84_EXACT = Transformer.from_crs(
    "EPSG:3006", "EPSG:4326", always_xy=True,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Total classes: 0-22 (23 classes including hygge/harvest at index 22)
NUM_CLASSES = NUM_UNIFIED_CLASSES  # 23

# Prithvi-EO-2.0 normalization constants (DN scale, per band)
# Bands: B02, B03, B04, B8A, B11, B12
PRITHVI_MEAN = np.array(
    [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0], dtype=np.float32,
)
PRITHVI_STD = np.array(
    [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0], dtype=np.float32,
)

# DOY target for peak-summer frame selection (mid-July)
PEAK_SUMMER_DOY = 195

# Number of spectral bands per frame (Prithvi 6-band)
N_BANDS = 6

# Auxiliary channel z-score normalization: (mean, std)
# volume and basal_area use log(1+x) pre-transform (lognormal distributions)
AUX_NORM = {
    "height": (7.36, 6.55),
    "volume": (3.56, 1.14),          # log1p-transformed
    "basal_area": (2.42, 0.71),      # log1p-transformed
    "diameter": (16.33, 7.84),
    "dem": (264.03, 215.37),
    # vpp_sosd/eosd are decoded YYDDD->DOY before normalization (see
    # AUX_YYDDD_DATE_CHANNELS); stats below are day-of-year mean/std.
    "vpp_sosd": (129.40, 31.11),   # DOY, spring green-up (~May 9)
    "vpp_eosd": (263.40, 38.32),   # DOY, autumn senescence (~Sep 20)
    "vpp_length": (141.61, 41.39),
    "vpp_maxv": (0.88, 0.57),
    "vpp_minv": (0.04, 0.05),
    # markfukt: SLU soil-moisture probability, already float32 [0.01, 1.01]
    # in the npz with NaN nodata. See config.aux_norm for provenance.
    "markfukt": (0.50, 0.25),
    # ΔVV/ΔVH: SAR backscatter change in dB between the growing-season γ⁰
    # composite (`s1_vv_vh`) and the 2016 γ⁰ composite (`s1_vv_vh_2016`),
    # computed at read time (not stored). Distribution is centred on 0 dB
    # (no change) and roughly symmetric; a fresh clearcut collapses volume
    # scattering, producing a sharp NEGATIVE step of ~−3 to −10 dB (VV) /
    # ~−2 to −8 dB (VH). std=4.0 dB places a −8 dB harvest at z≈−2 while
    # keeping ordinary phenological/moisture jitter (±1–2 dB) inside ±0.5 z,
    # so the neutral zero-change signal normalizes to exactly 0 (mean=0).
    "delta_vv": (0.0, 4.0),
    "delta_vh": (0.0, 4.0),
    # Fixed Sweden priors for a leakage-free smoke test. A production run
    # should replace these with train-split-only statistics.
    **ERA5_AUX_NORM,
}

# Channels that need log(1+x) pre-transform before z-score
AUX_LOG_TRANSFORM = {"volume", "basal_area"}

# Channels whose npz nodata is encoded as NaN (float) rather than 0.
# For these, NaN is filled with the channel's z-score mean BEFORE the
# transform so nodata normalizes to ~0 and never leaks NaN into the model.
# (Every other aux channel uses 0 as its nodata sentinel.)
AUX_NAN_NODATA_CHANNELS = {"markfukt"}

# HR-VPP date channels: raw band is YYDDD = (year-2000)*1000 + DOY.
# Decode to day-of-year (value % 1000) before z-score; NoData (0) maps to
# the channel mean so it normalizes to 0 instead of a huge outlier.
AUX_YYDDD_DATE_CHANNELS = {"vpp_sosd", "vpp_eosd"}

# ΔSAR aux channels are COMPUTED at read time from the tile's S1 keys
# (`s1_vv_vh` season composite − `s1_vv_vh_2016`), not read from a stored
# array. They are opt-in via config.enabled_aux_names like every other aux
# channel, but `_load_aux_channels` routes them through `compute_delta_sar`
# instead of a direct `data[name]` lookup. Tiles missing the 2016 composite
# (~10%) emit a zero Δ ("no change signal" — the correct neutral for fusion).
AUX_COMPUTED_CHANNELS = {"delta_vv", "delta_vh"}

# Floor (linear γ⁰) applied before the 10·log10 so a zero/negative nodata
# pixel never produces −inf. −40 dB (1e-4 linear) is well below the sensor
# noise floor, so a genuine low-backscatter pixel is unaffected.
_S1_LINEAR_FLOOR = 1e-4


def compute_delta_sar(data) -> "np.ndarray | None":
    """Compute the (2, H, W) ΔVV/ΔVH dB-difference stack from S1 keys.

    ΔX = 10·log10(season_X) − 10·log10(2016_X), on the LINEAR γ⁰ composites
    `s1_vv_vh` (2, H, W) and `s1_vv_vh_2016` (2, H, W). A sharp negative Δ is
    the clearcut signal: volume scattering collapses when a stand is felled.

    Returns:
        (2, H, W) float32 [ΔVV, ΔVH] in dB, NaN/inf-scrubbed to 0, OR None
        when either composite is absent (missing-2016 tile → caller emits
        zeros = "no change"). Never raises: inputs are floored before log so
        zero-nodata pixels map to a finite (near-zero after subtraction) Δ.
    """
    season = data.get("s1_vv_vh", None)
    baseline = data.get("s1_vv_vh_2016", None)
    if season is None or baseline is None:
        return None
    season = np.asarray(season, dtype=np.float32)
    baseline = np.asarray(baseline, dtype=np.float32)
    if season.shape != baseline.shape or season.ndim != 3 or season.shape[0] != 2:
        return None
    # Floor before log: guards linear-γ⁰ zeros (nodata) and any residual
    # negative/NaN so 10·log10 is always finite. NaN → floor via nan_to_num.
    s = np.maximum(np.nan_to_num(season, nan=_S1_LINEAR_FLOOR), _S1_LINEAR_FLOOR)
    b = np.maximum(np.nan_to_num(baseline, nan=_S1_LINEAR_FLOOR), _S1_LINEAR_FLOOR)
    delta = 10.0 * np.log10(s) - 10.0 * np.log10(b)  # (2, H, W) dB
    return np.nan_to_num(
        delta, nan=0.0, posinf=0.0, neginf=0.0,
    ).astype(np.float32)


def normalize_aux_channel(name: str, x):
    """Decode, optionally log-transform, and z-score-normalize one aux channel.

    Single source of truth for auxiliary-channel normalization — every
    consumer (UnifiedDataset, PixelDataset, preview/visualization tooling)
    must call this so the transform can never drift between code paths.
    NoData is encoded as 0 in all aux channels.

    Args:
        name: Channel name (one of AUX_CHANNEL_NAMES).
        x: Scalar or ndarray of raw channel values.

    Returns:
        Normalized value(s) — a float for scalar input, ndarray otherwise.
    """
    scalar = np.isscalar(x) or np.ndim(x) == 0
    arr = np.asarray(x, dtype=np.float32)
    # NaN-nodata channels (markfukt): replace NaN with the channel mean so
    # nodata normalizes to ~0 rather than propagating NaN through the model.
    # Fall back to a constant if the whole tile is nodata (no finite pixels).
    if name in AUX_NAN_NODATA_CHANNELS:
        fill = AUX_NORM.get(name, (0.0, 1.0))[0]
        arr = np.where(np.isnan(arr), np.float32(fill), arr)
    if name in AUX_YYDDD_DATE_CHANNELS:
        # YYDDD -> DOY; NoData (0) -> mean so it normalizes to 0 rather
        # than a large negative outlier.
        mean, std = AUX_NORM[name]
        arr = np.where(arr > 0, np.mod(arr, 1000.0), mean).astype(np.float32)
        arr = (arr - mean) / max(std, 1e-6)
    else:
        if name in AUX_LOG_TRANSFORM:
            arr = np.log1p(np.maximum(arr, 0.0))
        if name in AUX_NORM:
            mean, std = AUX_NORM[name]
            arr = (arr - mean) / max(std, 1e-6)
    return float(arr) if scalar else arr

# Canonical ordered list of auxiliary channels
# Must match order from config.enabled_aux_names
AUX_CHANNEL_NAMES = [
    "height", "volume", "basal_area", "diameter", "dem",
    "vpp_sosd", "vpp_eosd", "vpp_length", "vpp_maxv", "vpp_minv",
]

ERA5_AUX_CHANNELS = frozenset({
    "era5_t2m_mean", "era5_tp_sum", "era5_swvl1_mean",
    "era5_ssrd_sum", "era5_gdd",
})

_ERA5_SMOKE_COHORT_SCHEMA = "era5-smoke-cohort-v4"
_ERA5_SMOKE_MAX_LABEL = 22
_ERA5_SMOKE_MODEL_PATCH_PX = 504
_ERA5_SMOKE_CROP_CLASS_NAMES = {
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


def _split_file_names(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not names:
        raise ValueError(f"Explicit split file is empty: {path}")
    if len(names) != len(set(names)):
        raise ValueError(f"Explicit split file contains duplicate tile names: {path}")
    return names


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _support_from_manifest_items(items: list[dict]) -> dict:
    pixel_counts: dict[int, int] = {}
    tile_counts: dict[int, int] = {}
    for item in items:
        model_label = item.get("model_label")
        if not isinstance(model_label, dict):
            raise ValueError("Cohort tile is missing exact model-label support")
        for raw_label, count in model_label["pixel_counts"].items():
            label = int(raw_label)
            pixel_counts[label] = pixel_counts.get(label, 0) + int(count)
            tile_counts[label] = tile_counts.get(label, 0) + 1
    observed = sorted(pixel_counts)
    if not observed:
        raise ValueError("Cohort manifest contains no label support")
    return {
        "observed_labels": observed,
        "foreground_labels": [label for label in observed if label > 0],
        "pixel_counts": {str(label): pixel_counts[label] for label in observed},
        "tile_counts": {str(label): tile_counts[label] for label in observed},
        "min_label": min(observed),
        "max_label": max(observed),
    }


def _supported_smoke_crop_classes(
    support: dict[str, dict],
    thresholds: dict[str, int],
) -> list[str]:
    result: list[str] = []
    for label, name in _ERA5_SMOKE_CROP_CLASS_NAMES.items():
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


_SPATIAL_COMPONENT_RE = re.compile(r"^spatial:[0-9a-f]{64}$")
_SPLIT_COMPONENT_RE = re.compile(r"^split:[0-9a-f]{64}$")
_BBOX_KEYS = ("west", "south", "east", "north")


def _manifest_bbox_values(item: dict) -> tuple[int, int, int, int]:
    bbox = item.get("bbox_3006")
    if not isinstance(bbox, dict) or set(bbox) != set(_BBOX_KEYS):
        raise ValueError(
            f"Cohort tile {item.get('name')} has invalid exact bbox_3006"
        )
    values = tuple(bbox[key] for key in _BBOX_KEYS)
    if any(type(value) is not int for value in values):
        raise ValueError(
            f"Cohort tile {item.get('name')} bbox_3006 must contain integers"
        )
    west, south, east, north = values
    if east <= west or north <= south:
        raise ValueError(
            f"Cohort tile {item.get('name')} bbox_3006 has non-positive area"
        )
    return west, south, east, north


def _bbox_intersection_area(left: dict, right: dict) -> int:
    left_w, left_s, left_e, left_n = _manifest_bbox_values(left)
    right_w, right_s, right_e, right_n = _manifest_bbox_values(right)
    width = max(0, min(left_e, right_e) - max(left_w, right_w))
    height = max(0, min(left_n, right_n) - max(left_s, right_s))
    return width * height


def _manifest_component_catalog(items: list[dict], field: str) -> list[dict]:
    by_component: dict[str, list[dict]] = {}
    for item in items:
        by_component.setdefault(item[field], []).append(item)
    result: list[dict] = []
    for component_id, members in sorted(by_component.items()):
        boxes = [_manifest_bbox_values(item) for item in members]
        entry = {
            "id": component_id,
            "tiles": sorted(item["name"] for item in members),
            "bbox_3006": {
                "west": min(box[0] for box in boxes),
                "south": min(box[1] for box in boxes),
                "east": max(box[2] for box in boxes),
                "north": max(box[3] for box in boxes),
            },
            "era5_cells": sorted({item["era5_cell"] for item in members}),
        }
        if field == "spatial_component":
            split_components = {item["split_component"] for item in members}
            if len(split_components) != 1:
                raise ValueError(
                    f"Cohort spatial component {component_id} spans split components"
                )
            entry["split_component"] = split_components.pop()
        else:
            entry["spatial_components"] = sorted({
                item["spatial_component"] for item in members
            })
        result.append(entry)
    return result


def _validate_spatial_partition(manifest: dict) -> None:
    """Validate the sealed bbox/component split contract without source I/O."""
    splits = manifest.get("splits", {})
    split_items = {
        split: splits.get(split) if isinstance(splits.get(split), list) else []
        for split in ("train", "val")
    }
    all_items = [*split_items["train"], *split_items["val"]]
    for item in all_items:
        spatial_id = item.get("spatial_component")
        split_id = item.get("split_component")
        if not isinstance(spatial_id, str) or not _SPATIAL_COMPONENT_RE.fullmatch(
            spatial_id
        ):
            raise ValueError(
                f"Cohort tile {item.get('name')} has invalid spatial_component"
            )
        if not isinstance(split_id, str) or not _SPLIT_COMPONENT_RE.fullmatch(
            split_id
        ):
            raise ValueError(
                f"Cohort tile {item.get('name')} has invalid split_component"
            )
        _manifest_bbox_values(item)

    for left_index, left in enumerate(all_items):
        for right in all_items[left_index + 1:]:
            if (
                left["era5_cell"] == right["era5_cell"]
                and left["split_component"] != right["split_component"]
            ):
                raise ValueError(
                    "Cohort items in one ERA5 cell have different split components"
                )
            if _bbox_intersection_area(left, right) > 0 and not (
                left["spatial_component"] == right["spatial_component"]
                and left["split_component"] == right["split_component"]
            ):
                raise ValueError(
                    "Positive-area cohort bbox overlap crosses component identity"
                )

    component_sets = {
        field: {
            split: {item[field] for item in split_items[split]}
            for split in ("train", "val")
        }
        for field in ("spatial_component", "split_component")
    }
    for field, by_split in component_sets.items():
        overlap = by_split["train"] & by_split["val"]
        if overlap:
            raise ValueError(
                f"Cohort {field} identities overlap across train/val: "
                f"{sorted(overlap)[:3]}"
            )
    for train_item in split_items["train"]:
        for val_item in split_items["val"]:
            if _bbox_intersection_area(train_item, val_item) > 0:
                raise ValueError(
                    "Cohort train/val bbox intersection area must be zero"
                )

    partition = manifest.get("spatial_partition")
    if not isinstance(partition, dict):
        raise ValueError("Cohort lacks a spatial_partition manifest")
    expected_scalars = {
        "crs": "EPSG:3006",
        "bbox_intersection": "positive_area",
        "component_definition": (
            "transitive_closure_of_positive_area_bbox_intersections"
        ),
        "split_component_definition": (
            "transitive_closure_of_spatial_components_and_shared_era5_cells"
        ),
    }
    if any(partition.get(key) != value for key, value in expected_scalars.items()):
        raise ValueError("Cohort spatial_partition definitions are unsupported")
    for field, key in (
        ("spatial_component", "spatial_components"),
        ("split_component", "split_components"),
    ):
        expected = {
            "train": sorted(component_sets[field]["train"]),
            "val": sorted(component_sets[field]["val"]),
            "overlap": [],
        }
        if partition.get(key) != expected:
            raise ValueError(f"Cohort {key} manifest is stale or inconsistent")
    if partition.get("bbox_area_intersections") != []:
        raise ValueError("Cohort bbox_area_intersections must be empty")
    expected_catalogs = {
        "spatial_component_catalog": _manifest_component_catalog(
            all_items, "spatial_component",
        ),
        "split_component_catalog": _manifest_component_catalog(
            all_items, "split_component",
        ),
    }
    for key, expected in expected_catalogs.items():
        if partition.get(key) != expected:
            raise ValueError(f"Cohort {key} is stale or inconsistent")


def _load_explicit_split_manifest(
    split_dir: Path,
) -> tuple[dict, dict[str, list[str]]]:
    """Load and structurally validate an ERA5 smoke split bundle."""
    paths = {
        split: split_dir / f"split_{split}.txt" for split in ("train", "val")
    }
    missing = [
        str(path)
        for path in (*paths.values(), split_dir / "manifest.json")
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Explicit split_dir requires split_train.txt, split_val.txt, and "
            f"manifest.json; missing={missing}"
        )

    names = {split: _split_file_names(path) for split, path in paths.items()}
    overlap = set(names["train"]) & set(names["val"])
    if overlap:
        raise ValueError(f"Train/val split files overlap: {sorted(overlap)[:3]}")

    manifest_path = split_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Invalid cohort manifest {manifest_path}: {exc}") from exc
    if manifest.get("schema") != _ERA5_SMOKE_COHORT_SCHEMA:
        raise ValueError(
            f"Unsupported cohort schema {manifest.get('schema')!r}; "
            f"expected {_ERA5_SMOKE_COHORT_SCHEMA!r}"
        )

    label_schema = manifest.get("label_schema", {})
    expected_schema = {
        "num_logits": _ERA5_SMOKE_MAX_LABEL + 1,
        "min_label": 0,
        "max_label": _ERA5_SMOKE_MAX_LABEL,
    }
    if label_schema != expected_schema:
        raise ValueError(
            f"ERA5 smoke requires the 23-logit 0..22 label schema; got {label_schema}"
        )
    if manifest.get("model_patch_px") != _ERA5_SMOKE_MODEL_PATCH_PX:
        raise ValueError(
            f"ERA5 smoke requires model_patch_px={_ERA5_SMOKE_MODEL_PATCH_PX}"
        )

    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ValueError("Cohort manifest lacks source metadata")
    bbox_manifest = source.get("bbox_manifest")
    bbox_manifest_sha256 = source.get("bbox_manifest_sha256")
    if bbox_manifest is None:
        if bbox_manifest_sha256 is not None:
            raise ValueError(
                "Cohort bbox_manifest_sha256 must be null without a bbox manifest"
            )
    elif not (
        isinstance(bbox_manifest, str)
        and isinstance(bbox_manifest_sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", bbox_manifest_sha256)
    ):
        raise ValueError("Cohort bbox manifest path/hash contract is invalid")
    elif Path(bbox_manifest).is_file() and (
        _sha256_file(Path(bbox_manifest)) != bbox_manifest_sha256
    ):
        raise ValueError("Cohort bbox manifest is stale or has changed")

    for split in ("train", "val"):
        split_text = paths[split].read_text()
        expected_hash = hashlib.sha256(split_text.encode()).hexdigest()
        if manifest.get("split_sha256", {}).get(split) != expected_hash:
            raise ValueError(f"Stale or partial split_{split}.txt: manifest hash mismatch")
        if manifest.get("counts", {}).get(split) != len(names[split]):
            raise ValueError(f"Cohort {split} count does not match split file")
        items = manifest.get("splits", {}).get(split)
        if (
            not isinstance(items, list)
            or [item.get("name") for item in items] != names[split]
        ):
            raise ValueError(f"Cohort {split} tile list does not match split file")
        for item in items:
            label = item.get("label", {})
            if (
                label.get("min", -1) < 0
                or label.get("max", _ERA5_SMOKE_MAX_LABEL + 1)
                > _ERA5_SMOKE_MAX_LABEL
            ):
                raise ValueError(
                    f"Cohort tile {item.get('name')} exceeds label range 0..22"
                )
        actual_support = _support_from_manifest_items(items)
        if manifest.get("label_support", {}).get(split) != actual_support:
            raise ValueError(f"Cohort {split} label support is stale or inconsistent")
        if not actual_support["foreground_labels"]:
            raise ValueError(f"Cohort {split} has no foreground label support")

    train_foreground = set(manifest["label_support"]["train"]["foreground_labels"])
    val_foreground = set(manifest["label_support"]["val"]["foreground_labels"])
    unseen_val_labels = val_foreground - train_foreground
    if unseen_val_labels:
        raise ValueError(
            "Validation contains classes absent from training: "
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
    expected_crop_classes = _supported_smoke_crop_classes(
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

    declared_cells = manifest.get("era5_cells", {})
    raw_item_cells = {
        split: {item.get("era5_cell") for item in manifest["splits"][split]}
        for split in ("train", "val")
    }
    if any(None in cells for cells in raw_item_cells.values()):
        raise ValueError("Cohort manifest has a tile without an ERA5 cell")
    item_cells = {split: sorted(cells) for split, cells in raw_item_cells.items()}
    if (
        declared_cells.get("train") != item_cells["train"]
        or declared_cells.get("val") != item_cells["val"]
    ):
        raise ValueError("Cohort ERA5-cell lists are stale or inconsistent")
    cell_overlap = set(item_cells["train"]) & set(item_cells["val"])
    if cell_overlap or declared_cells.get("overlap") != []:
        raise ValueError(f"ERA5 train/val cells overlap: {sorted(cell_overlap)[:3]}")
    _validate_spatial_partition(manifest)
    return manifest, names


def _resolve_explicit_cohort_location(
    *,
    data: dict,
    tile_name: str,
    manifest: dict,
    manifest_item: dict,
) -> dict[str, float | str]:
    """Resolve and attest the location used by a sealed smoke sample.

    The ERA5 cohort builder and sidecar fetcher both use
    :func:`resolve_tile_bbox`.  The sealed dataset must use that same source
    for Prithvi's location token; falling back to the historical hard-coded
    Sweden coordinate would silently give the model weather and location
    metadata for different places.
    """
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ValueError("Cohort manifest lacks location source metadata")
    bbox_manifest = source.get("bbox_manifest")
    if bbox_manifest is not None and not isinstance(bbox_manifest, str):
        raise ValueError("Cohort source bbox_manifest must be a path or null")

    try:
        size_px = int(np.asarray(data.get("tile_size_px", 512)).item())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Sealed cohort tile {tile_name} has invalid tile_size_px"
        ) from exc
    if size_px <= 0:
        raise ValueError(
            f"Sealed cohort tile {tile_name} has invalid tile_size_px={size_px}"
        )

    sealed_bbox_raw = manifest_item.get("bbox_3006")
    sealed_center_raw = manifest_item.get("location_3006")
    sealed_wgs84_raw = manifest_item.get("location_wgs84")
    try:
        bbox_values = [
            sealed_bbox_raw[key]
            for key in ("west", "south", "east", "north")
        ]
        center_values = [
            sealed_center_raw[key] for key in ("easting", "northing")
        ]
        if any(type(value) is not int for value in (*bbox_values, *center_values)):
            raise TypeError("EPSG:3006 coordinates must be exact integers")
        sealed_bbox = {
            key: int(sealed_bbox_raw[key])
            for key in ("west", "south", "east", "north")
        }
        easting = float(sealed_center_raw["easting"])
        northing = float(sealed_center_raw["northing"])
        sealed_lat = float(sealed_wgs84_raw["lat"])
        sealed_lon = float(sealed_wgs84_raw["lon"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Cohort exact location metadata is incomplete for {tile_name}"
        ) from exc
    if not np.isfinite((easting, northing, sealed_lat, sealed_lon)).all():
        raise ValueError(
            f"Cohort exact location metadata is non-finite for {tile_name}"
        )
    expected_bbox = TileConfig(size_px=size_px).bbox_from_center(
        easting, northing,
    )
    if sealed_bbox != expected_bbox:
        raise ValueError(
            f"Cohort exact bbox/centroid mismatch for {tile_name}: "
            f"bbox={sealed_bbox}, center={sealed_center_raw}"
        )

    # Re-resolve when the original source is available and reject any drift,
    # including a legacy external-manifest edit that stays within one ERA5
    # cell.  The sealed bbox remains authoritative when that optional source
    # is not mounted at training time.
    resolved_bbox = resolve_tile_bbox(
        name=Path(tile_name).stem,
        tile=TileConfig(size_px=size_px),
        npz_data=data,
        manifest_path=bbox_manifest,
    )
    if resolved_bbox is not None and resolved_bbox != sealed_bbox:
        raise ValueError(
            f"Cohort location source drift for {tile_name}: "
            f"sealed_bbox={sealed_bbox}, resolved_bbox={resolved_bbox}"
        )
    lon, lat = _SWEREF99_TO_WGS84_EXACT.transform(easting, northing)
    if not np.isfinite((lat, lon)).all():
        raise ValueError(
            f"Sealed cohort tile {tile_name} resolves to invalid WGS84 coordinates"
        )
    if not np.allclose(
        (lat, lon), (sealed_lat, sealed_lon), rtol=0.0, atol=1e-9,
    ):
        raise ValueError(
            f"Cohort exact WGS84 location mismatch for {tile_name}: "
            f"transformed={(lat, lon)}, manifest={(sealed_lat, sealed_lon)}"
        )

    grid = era5_grid_context(lat, lon)
    expected_request = manifest_item.get("era5_request")
    expected_land = manifest_item.get("era5_land_cell")
    expected_atmosphere = manifest_item.get("era5_atmosphere_cell")
    try:
        request_matches = np.allclose(
            (grid["request_lat"], grid["request_lon"]),
            (expected_request["lat"], expected_request["lon"]),
            rtol=0.0,
            atol=1e-7,
        )
        land_matches = era5_api_cell_coords_match(
            grid["land_cell"]["lat"],
            grid["land_cell"]["lon"],
            expected_land["lat"],
            expected_land["lon"],
        )
        atmosphere_matches = era5_api_cell_coords_match(
            grid["atmosphere_cell"]["lat"],
            grid["atmosphere_cell"]["lon"],
            expected_atmosphere["lat"],
            expected_atmosphere["lon"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Cohort location metadata is incomplete for {tile_name}"
        ) from exc
    expected_cell_id = format_era5_cell_id(
        grid["atmosphere_cell"]["lat"],
        grid["atmosphere_cell"]["lon"],
    )
    if not (
        request_matches
        and land_matches
        and atmosphere_matches
        and manifest_item.get("era5_cell") == expected_cell_id
    ):
        raise ValueError(
            f"Cohort location metadata mismatch for {tile_name}: "
            f"resolved_request=({grid['request_lat']}, {grid['request_lon']}), "
            f"manifest_request={expected_request}"
        )

    return {
        "source": "sealed_cohort_manifest_v4",
        "bbox_west_3006": float(sealed_bbox["west"]),
        "bbox_south_3006": float(sealed_bbox["south"]),
        "bbox_east_3006": float(sealed_bbox["east"]),
        "bbox_north_3006": float(sealed_bbox["north"]),
        "easting_3006": easting,
        "northing_3006": northing,
        "lat": sealed_lat,
        "lon": sealed_lon,
        "era5_request_lat": float(grid["request_lat"]),
        "era5_request_lon": float(grid["request_lon"]),
    }

# Forest classes in the unified schema (eligible for harvest override)
_FOREST_CLASSES = frozenset({1, 2, 3, 4, 5})

# Label-derived keys the dataset consumes for the segmentation target and
# per-pixel loss weighting. When a `label_dir` sidecar is supplied, exactly
# these keys are sourced from the sidecar npz (if present there) instead of
# the source tile; every other key (spectral/aux/temporal/coords) still comes
# from the source tile. `label` is the training target (`_build_label`);
# `parcel_area_ha`/`nmd_area_ha` drive inverse-area pixel weighting in
# `__getitem__`. Only these three are read in `__getitem__`; the sidecar's
# other bookkeeping keys (nmd_label_raw, label_mask, harvest_mask, n_parcels,
# …) are not consumed here and are left untouched.
_LABEL_SIDECAR_KEYS = ("label", "parcel_area_ha", "nmd_area_ha")


class _LabelOverlay:
    """Read-only overlay of a label sidecar npz over a source tile npz.

    Presents the union of both mappings so downstream ``data[...]`` /
    ``data.get(...)`` / ``key in data`` calls work unchanged. For the
    keys in ``_LABEL_SIDECAR_KEYS`` the sidecar value wins when present;
    every other lookup falls through to the source tile. Only the label
    target and area-weighting rasters are ever overridden — spectral, aux,
    temporal and coordinate arrays always resolve to the source tile.
    """

    __slots__ = ("_source", "_sidecar")

    def __init__(self, source, sidecar):
        self._source = source
        self._sidecar = sidecar

    def _from_sidecar(self, key: str) -> bool:
        return key in _LABEL_SIDECAR_KEYS and key in self._sidecar

    def __contains__(self, key: str) -> bool:
        return self._from_sidecar(key) or key in self._source

    def __getitem__(self, key: str):
        if self._from_sidecar(key):
            return self._sidecar[key]
        return self._source[key]

    def get(self, key: str, default=None):
        if self._from_sidecar(key):
            return self._sidecar[key]
        # np.lib.npyio.NpzFile has no .get(); emulate it.
        return self._source[key] if key in self._source else default


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class UnifiedDataset(Dataset):
    """PyTorch Dataset that loads both LULC and crop tiles for
    unified 19-class segmentation training.

    Merges NMD land-cover classes with LPIS crop detail and a
    harvest-detection class into a single per-pixel label space.
    Outputs are single-date, Prithvi-normalized, and compatible
    with the LULCDataset dict format expected by trainer.py.

    Args:
        lulc_dir: Directory containing LULC seasonal tiles
            (expected layout: ``<lulc_dir>/tiles/*.npz``).
        crop_dir: Directory containing crop tiles
            (expected layout: ``<crop_dir>/*.npz``).
        split: Dataset split -- ``"train"`` or ``"val"``.
            If ``split_<split>.txt`` files exist in the respective
            directories, they are used. Otherwise all tiles are loaded.
        patch_size: Output spatial dimension (default 224 for Prithvi).
        enable_aux: Whether to include auxiliary channels in output.
        augment_override: Explicit augmentation flag. If None (default),
            augmentation is enabled when ``split == "train"``.

    Raises:
        FileNotFoundError: If both tile directories are empty or missing.
    """

    # Model registry keys for which this dataset can build per-model
    # input tensors. Each key selects which extra tensors __getitem__ emits:
    #   clay_v1_5        → s2_clay (10-band optical stack), optical-only
    #   croma_base       → s2_croma (12-band optical) + s1_vv_vh (2-band SAR)
    #   terramind_v1_base→ s1_vv_vh (2-band SAR); S2 comes from `spectral`
    # Prithvi/Tessera route through `spectral`/`tessera` and set no model key.
    _SUPPORTED_MODEL_KEYS: frozenset[str] = frozenset({
        "clay_v1_5", "croma_base", "terramind_v1_base",
    })

    # Per-model-key set of tile .npz keys that MUST be present for the
    # emitted sample to be usable. Tiles missing any required key for the
    # active backbone are dropped at index-construction time (fail-loud +
    # logged) so __getitem__ never KeyErrors mid-epoch. `s1_vv_vh` is the
    # constraining one: present on ~6011/7882 tiles.
    _MODEL_REQUIRED_TILE_KEYS: dict[str, tuple[str, ...]] = {
        "clay_v1_5":         ("b08", "rededge"),
        "croma_base":        ("b08", "rededge", "s1_vv_vh"),
        "terramind_v1_base": ("s1_vv_vh",),
    }

    def __init__(
        self,
        lulc_dir: str | Path | None = None,
        crop_dir: str | Path | None = None,
        split: str = "train",
        patch_size: int = 224,
        enable_aux: bool = True,
        augment_override: bool | None = None,
        multitemporal: bool = False,
        num_temporal_frames: int = 4,
        model_keys: tuple[str, ...] = (),
        aux_channel_names: Sequence[str] | None = None,
        label_dir: str | Path | None = None,
        frac_dir: str | Path | None = None,
        era5_dir: str | Path | None = None,
        era5_mode: str = "off",
        split_dir: str | Path | None = None,
        limit_tiles: int | None = None,
        backbone_family: str = "prithvi",
    ):
        super().__init__()
        self.patch_size = patch_size
        # Which backbone family the emitted samples feed. "prithvi" (the
        # default) reads the multi-frame `spectral` reflectance and applies
        # Prithvi z-score normalization + temporal/location coords. "tessera"
        # reads the pre-baked `tessera` embedding (128, H, W) instead — no
        # normalization (embeddings ship normalized), no temporal reshape
        # (annual, single-frame). Routed explicitly on family, never on
        # tensor shape.
        self.backbone_family = backbone_family
        if backbone_family == "tessera" and multitemporal:
            raise ValueError(
                "backbone_family='tessera' is single-frame (annual "
                "embeddings) — do not combine with multitemporal=True."
            )
        # Optional non-destructive label sidecar directory. When set, the
        # label-derived keys (_LABEL_SIDECAR_KEYS) are read from
        # <label_dir>/<tile_name>.npz per tile; spectral/aux/temporal/coords
        # still come from the source tile. None → byte-identical legacy path.
        self.label_dir = Path(label_dir) if label_dir is not None else None
        # Optional Trädslag fraction sidecar directory. When set, each sample
        # carries `frac` (K,H,W) target crown-cover in [0,1] and `frac_mask`
        # (H,W) — 1 where supervision applies. Tiles without a frac sidecar
        # return an all-masked frac (no supervision) but are NOT dropped:
        # the hard label still supervises them. None → no frac keys emitted
        # (byte-identical legacy path).
        self.frac_dir = Path(frac_dir) if frac_dir is not None else None
        # Weather lives in non-destructive per-tile sidecars. Missing files
        # deliberately yield neutral (z=0) values, which is also the A/B
        # smoke-test baseline while preserving an identical architecture.
        if era5_mode not in {"off", "control", "treatment"}:
            raise ValueError(f"Unknown era5_mode: {era5_mode!r}")
        if era5_mode == "treatment" and era5_dir is None:
            raise ValueError("era5_mode='treatment' requires era5_dir")
        if era5_mode != "treatment" and era5_dir is not None:
            raise ValueError("era5_dir is only valid with era5_mode='treatment'")
        self.era5_mode = era5_mode
        self.era5_dir = Path(era5_dir) if era5_dir is not None else None
        self.split_dir = Path(split_dir) if split_dir is not None else None
        self._split_manifest: dict | None = None
        self._explicit_split_names: dict[str, list[str]] | None = None
        if self.split_dir is not None:
            if split not in {"train", "val"}:
                raise FileNotFoundError(
                    f"Explicit ERA5 smoke cohort has no split_{split}.txt"
                )
            if limit_tiles is not None:
                raise ValueError(
                    "limit_tiles cannot be combined with an explicit, "
                    "manifest-validated split_dir"
                )
            self._split_manifest, self._explicit_split_names = (
                _load_explicit_split_manifest(self.split_dir)
            )
        self.enable_aux = enable_aux
        # Ordered aux-channel list. Defaults to the canonical 10-channel
        # AUX_CHANNEL_NAMES so the historical behaviour is byte-identical;
        # pass config.enabled_aux_names to opt in to markfukt (11 channels).
        self.aux_channel_names: tuple[str, ...] = (
            tuple(AUX_CHANNEL_NAMES) if aux_channel_names is None
            else tuple(aux_channel_names)
        )
        self.augment = (split == "train") if augment_override is None else augment_override
        self.multitemporal = multitemporal
        self.num_temporal_frames = num_temporal_frames

        # Opt-in per-model tensor emission. Empty tuple preserves the
        # historical Prithvi-only behaviour byte-for-byte (no extra keys,
        # no extra I/O, no extra cost per __getitem__).
        bad = set(model_keys) - self._SUPPORTED_MODEL_KEYS
        if bad:
            raise ValueError(
                f"Unsupported model_keys: {sorted(bad)}. "
                f"Supported: {sorted(self._SUPPORTED_MODEL_KEYS)}"
            )
        self.model_keys = tuple(model_keys)

        # Prithvi normalization reshaped for broadcasting over (6, H, W)
        self._mean = PRITHVI_MEAN.reshape(N_BANDS, 1, 1)
        self._std = PRITHVI_STD.reshape(N_BANDS, 1, 1)

        # Discover tiles from both sources
        self._entries: list[dict] = []
        self._load_tile_list(lulc_dir, "lulc", split)
        self._load_tile_list(crop_dir, "crop", split)

        # Drop tiles whose label sidecar is missing so training never crashes
        # mid-epoch on an absent sidecar. Done at construction; logged.
        if self.label_dir is not None:
            kept: list[dict] = []
            dropped = 0
            for e in self._entries:
                if (self.label_dir / e["name"]).exists():
                    kept.append(e)
                else:
                    dropped += 1
            self._entries = kept
            logger.info(
                "UnifiedDataset[%s]: label_dir=%s — kept %d tiles, "
                "dropped %d with missing sidecar",
                split, self.label_dir, len(kept), dropped,
            )

        # Drop tiles that lack a required key for the active backbone so
        # __getitem__ never KeyErrors mid-epoch. The constraining key is
        # `s1_vv_vh` (SAR), present on ~6011/7882 tiles — CROMA/TerraMind
        # need it, Clay/Prithvi/Tessera do not. Reading `.files` inspects
        # only the npz zip directory (no array decompression), so this is
        # cheap even over the full 7882-tile set.
        required = set()
        for k in self.model_keys:
            required.update(self._MODEL_REQUIRED_TILE_KEYS.get(k, ()))
        # SAR models (CROMA/TerraMind) require the v3 RTC γ⁰ season composite;
        # an older v1 ±3-day (T*2,H,W) stack or a v2 CDSE-dB composite under
        # the same `s1_vv_vh` key is a silent trap (wrong shape / wrong units).
        # Gate on `s1_enrich_v==3` at index time so pre-v3 leftovers are
        # dropped here (logged) instead of raising mid-epoch in __getitem__.
        needs_sar_v3 = bool(
            {"croma_base", "terramind_v1_base"} & set(self.model_keys)
        )
        if required:
            kept: list[dict] = []
            dropped = 0
            dropped_old_s1 = 0
            for e in self._entries:
                try:
                    with np.load(e["path"], allow_pickle=True) as z:
                        present = set(z.files)
                        if not required.issubset(present):
                            dropped += 1
                            continue
                        if needs_sar_v3 and int(z["s1_enrich_v"].item()
                                                if "s1_enrich_v" in present
                                                else 0) != 3:
                            dropped_old_s1 += 1
                            continue
                except Exception:
                    dropped += 1
                    continue
                kept.append(e)
            self._entries = kept
            logger.info(
                "UnifiedDataset[%s]: model_keys=%s require tile keys %s — "
                "kept %d tiles, dropped %d missing a required key, "
                "dropped %d with s1_enrich_v!=3 (pre-v3 S1 — re-run "
                "enrich_tiles_s1.py --s1-backend pc-rtc)",
                split, self.model_keys, sorted(required),
                len(kept), dropped, dropped_old_s1,
            )

        # Apply any smoke limit only after label/model eligibility filtering,
        # so the requested cohort size is real rather than shrinking later.
        if limit_tiles is not None:
            if limit_tiles <= 0:
                raise ValueError("limit_tiles must be positive")
            self._entries = sorted(
                self._entries,
                key=lambda entry: hashlib.md5(entry["name"].encode()).hexdigest(),
            )[:limit_tiles]

        if self._split_manifest is not None:
            self._validate_explicit_split_entries(split)

        if self.era5_mode == "treatment":
            missing_sidecars = [
                entry["name"] for entry in self._entries
                if not (self.era5_dir / entry["name"]).is_file()
            ]
            if missing_sidecars:
                raise FileNotFoundError(
                    f"ERA5 treatment coverage incomplete for split={split}: "
                    f"{len(missing_sidecars)} missing; first={missing_sidecars[:3]}"
                )

        if not self._entries:
            raise FileNotFoundError(
                f"No tiles found for split='{split}'. "
                f"Searched lulc_dir={lulc_dir}, crop_dir={crop_dir}"
                + (f", label_dir={self.label_dir} (all sidecars missing?)"
                   if self.label_dir is not None else "")
                + (f", model_keys={self.model_keys} required tile keys "
                   f"{sorted(required)} (none present?)"
                   if required else "")
            )

        logger.info(
            "UnifiedDataset[%s]: %d tiles (%d LULC, %d crop)",
            split,
            len(self._entries),
            sum(1 for e in self._entries if e["source"] == "lulc"),
            sum(1 for e in self._entries if e["source"] == "crop"),
        )

    # ------------------------------------------------------------------
    # Tile discovery
    # ------------------------------------------------------------------

    def _validate_explicit_split_entries(self, split: str) -> None:
        """Fail if selected files or labels drift from the cohort manifest."""
        if split not in {"train", "val"}:
            raise FileNotFoundError(
                f"Explicit ERA5 smoke cohort has no split_{split}.txt"
            )
        expected_names = self._explicit_split_names[split]
        actual_names = [entry["name"] for entry in self._entries]
        if len(actual_names) != len(set(actual_names)):
            raise ValueError(
                f"Explicit cohort split={split} resolves duplicate tile names across data dirs"
            )
        if set(actual_names) != set(expected_names):
            missing = sorted(set(expected_names) - set(actual_names))
            unexpected = sorted(set(actual_names) - set(expected_names))
            raise FileNotFoundError(
                f"Explicit cohort split={split} is partial or stale: "
                f"missing={missing[:3]}, unexpected={unexpected[:3]}"
            )

        expected_items = {
            item["name"]: item for item in self._split_manifest["splits"][split]
        }
        for entry in self._entries:
            label_path = (
                self.label_dir / entry["name"]
                if self.label_dir is not None else entry["path"]
            )
            try:
                with np.load(label_path, allow_pickle=False) as label_source:
                    if "label" not in label_source.files:
                        raise ValueError("missing label array")
                    actual_label = _label_metadata(label_source["label"])
            except (OSError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Could not validate cohort label {label_path}: {exc}"
                ) from exc
            expected_label = expected_items[entry["name"]].get("label")
            if actual_label != expected_label:
                raise ValueError(
                    f"Cohort label is stale for {entry['name']}: "
                    f"expected sha256={expected_label.get('sha256') if expected_label else None}, "
                    f"actual sha256={actual_label['sha256']}"
                )
            expected_source_hash = expected_items[entry["name"]].get(
                "source_npz_sha256"
            )
            actual_source_hash = _sha256_file(Path(entry["path"]))
            if actual_source_hash != expected_source_hash:
                raise ValueError(
                    f"Cohort source NPZ is stale for {entry['name']}: "
                    f"expected sha256={expected_source_hash}, "
                    f"actual sha256={actual_source_hash}"
                )
            try:
                with np.load(entry["path"], allow_pickle=False) as source_data:
                    entry["_sealed_location"] = _resolve_explicit_cohort_location(
                        data=source_data,
                        tile_name=entry["name"],
                        manifest=self._split_manifest,
                        manifest_item=expected_items[entry["name"]],
                    )
            except (OSError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Could not validate cohort location for {entry['name']}: {exc}"
                ) from exc
            if (
                actual_label["min"] < 0
                or actual_label["max"] > _ERA5_SMOKE_MAX_LABEL
            ):
                raise ValueError(
                    f"Cohort label range for {entry['name']} is "
                    f"{actual_label['min']}..{actual_label['max']}; expected 0..22"
                )

    def _load_tile_list(
        self,
        data_dir: str | Path | None,
        source: str,
        split: str,
    ) -> None:
        """Discover tiles from a directory and add them to ``_entries``.

        If a ``split_<split>.txt`` file is present, only listed tiles
        are included. Otherwise, all ``.npz`` files are loaded.

        Args:
            data_dir: Root directory for this tile source.
            source: ``"lulc"`` or ``"crop"`` tag.
            split: ``"train"`` or ``"val"``.
        """
        if data_dir is None:
            return

        data_dir = Path(data_dir)
        if not data_dir.exists():
            logger.warning("Tile directory not found: %s", data_dir)
            return

        # LULC tiles are under a 'tiles' subdirectory; crop tiles are
        # directly in the directory
        if source == "lulc":
            tiles_dir = data_dir / "tiles"
            if not tiles_dir.exists():
                tiles_dir = data_dir
        else:
            tiles_dir = data_dir

        # Check for a split file
        split_file = (self.split_dir or data_dir) / f"split_{split}.txt"
        if split_file.exists():
            with open(split_file) as f:
                requested_names = [
                    line.strip() for line in f if line.strip()
                ]
            paths_by_name = {
                path.name: path for path in tiles_dir.glob("*.npz")
            }
            # Explicit scientific cohorts bind sample order as well as
            # membership. Preserve the split ledger exactly so seeded A/B
            # samplers and the disk-backed preflight see the same sequence.
            tile_paths = [
                paths_by_name[name]
                for name in requested_names
                if name in paths_by_name
            ]
        else:
            # Deterministic 90/10 train/val split by MD5 hash of tile name.
            # This ensures consistent splits without needing split files.
            all_tiles = sorted(tiles_dir.glob("*.npz"))
            VAL_FRACTION = 10  # 10% val (every tile where hash % 100 < 10)
            if split == "val":
                tile_paths = [
                    p for p in all_tiles
                    if int(hashlib.md5(p.name.encode()).hexdigest(), 16) % 100 < VAL_FRACTION
                ]
            else:  # train
                tile_paths = [
                    p for p in all_tiles
                    if int(hashlib.md5(p.name.encode()).hexdigest(), 16) % 100 >= VAL_FRACTION
                ]

        for p in tile_paths:
            self._entries.append({
                "path": p,
                "source": source,
                "name": p.name,
            })

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tile_names(self) -> list[str]:
        """Tile filenames (for sampler compatibility with LULCDataset)."""
        return [e["name"] for e in self._entries]

    # ------------------------------------------------------------------
    # Core __getitem__
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int, _retries: int = 0) -> dict:
        """Load a single tile and return the trainer-compatible dict.

        Returns:
            Dict with keys ``image``, ``label``, auxiliary channels,
            and ``metadata``.
        """
        entry = self._entries[idx]
        # The tessera path needs the `tessera` embedding present; the Prithvi
        # path needs `spectral`/`image`. Require the right key up front so a
        # tile missing it is skipped by the retry loop, not silently zeroed.
        required_key = "tessera" if self.backbone_family == "tessera" else None
        try:
            data = np.load(entry["path"], allow_pickle=True)
            if "spectral" not in data and "image" not in data:
                raise KeyError("missing spectral")
            if required_key is not None and required_key not in data:
                raise KeyError(f"missing {required_key}")
        except Exception as exc:
            if self._split_manifest is not None:
                raise RuntimeError(
                    f"Explicit cohort tile is unreadable or lacks model input: "
                    f"{entry['path']}"
                ) from exc
            if _retries >= 50:
                raise RuntimeError(f"No valid tiles found after {_retries} retries from idx {idx}")
            alt = (idx + 1) % len(self._entries)
            return self.__getitem__(alt, _retries=_retries + 1)

        # Overlay the label sidecar (if configured) so `label` and the
        # area-weighting rasters resolve to <label_dir>/<tile>.npz while
        # spectral/aux/temporal/coords still come from the source tile.
        if self.label_dir is not None:
            try:
                sidecar = np.load(
                    self.label_dir / entry["name"], allow_pickle=True
                )
                data = _LabelOverlay(data, sidecar)
            except Exception as exc:
                if self._split_manifest is not None:
                    raise RuntimeError(
                        f"Explicit cohort label sidecar is unreadable: "
                        f"{self.label_dir / entry['name']}"
                    ) from exc
                # Sidecar unreadable at read-time (should have been dropped
                # at construction) — skip to the next tile rather than crash.
                if _retries >= 50:
                    raise RuntimeError(
                        f"No valid tiles found after {_retries} retries "
                        f"from idx {idx}"
                    )
                alt = (idx + 1) % len(self._entries)
                return self.__getitem__(alt, _retries=_retries + 1)

        source = entry["source"]
        sealed_location = entry.get("_sealed_location")
        if self._split_manifest is not None and sealed_location is None:
            raise RuntimeError(
                f"Explicit cohort tile lacks an attested location: {entry['name']}"
            )

        # --- Image tensor -----------------------------------------------
        # Tessera consumes the pre-baked (128, H, W) annual embedding; every
        # other family reads Sentinel-2 reflectance (single- or multi-frame).
        if self.backbone_family == "tessera":
            image = np.asarray(data["tessera"], dtype=np.float32)  # (128, H, W)
            temporal_mask = None
            doy = None
        elif self.multitemporal:
            image, temporal_mask, doy = self._extract_all_frames(
                data, source, self.num_temporal_frames,
                strict_smoke=self._split_manifest is not None,
            )
        else:
            if source == "lulc":
                image = self._extract_lulc_frame(data)
            else:
                image = self._extract_crop_frame(data)
            temporal_mask = None
            doy = None

        # --- Label construction ----------------------------------------
        label = self._build_label(data, source)

        # --- Auxiliary channels ----------------------------------------
        h, w = label.shape

        # Unified area map for inverse-area pixel weighting.
        # LPIS parcel area takes precedence for crop pixels; NMD connected-
        # component area fills in for all other labeled pixels (forest, water,
        # wetland, urban etc.).  Both are float32 ha-per-pixel rasters built
        # by build_labels.py.  Tiles that predate the NMD area stamp fall back
        # to zeros, which parcel_area_to_pixel_weights treats as weight=1.0.
        raw_lpis = data.get("parcel_area_ha", None)
        raw_nmd  = data.get("nmd_area_ha",    None)

        lpis_arr = (np.asarray(raw_lpis, dtype=np.float32)
                    if raw_lpis is not None else np.zeros((h, w), dtype=np.float32))
        nmd_arr  = (np.asarray(raw_nmd,  dtype=np.float32)
                    if raw_nmd  is not None else np.zeros((h, w), dtype=np.float32))

        if lpis_arr.shape != (h, w): lpis_arr = lpis_arr[:h, :w]
        if nmd_arr.shape  != (h, w): nmd_arr  = nmd_arr[:h, :w]

        # LPIS area where crop parcel exists, NMD component area everywhere else
        area_map = np.where(lpis_arr > 0, lpis_arr, nmd_arr)

        aux_stack = self._load_aux_channels(
            data,
            h,
            w,
            entry["name"],
            expected_location=(
                (sealed_location["lat"], sealed_location["lon"])
                if sealed_location is not None else None
            ),
        ) if self.enable_aux else None

        # Fold area_map into aug_stack as channel 0 for spatial consistency
        area_as_channel = area_map[np.newaxis]  # (1, H, W)
        if aux_stack is not None:
            aug_stack = np.concatenate([area_as_channel, aux_stack], axis=0)
        else:
            aug_stack = area_as_channel

        # --- Per-model extras (Clay / CROMA): build at native resolution
        # in raw reflectance [0,1] and fold into aug_stack so the same
        # crop+flip transform is applied. Channel-counts tracked so we
        # can slice them back out post-crop.
        n_area = 1
        n_aux = len(self.aux_channel_names) if self.enable_aux else 0
        extras_specs: list[tuple[str, int]] = []  # [(name, n_channels)]
        if self.model_keys:
            extras_tensors = self._build_model_specific_tensors(data, source)
            for name, arr in extras_tensors.items():
                aug_stack = np.concatenate([aug_stack, arr], axis=0)
                extras_specs.append((name, arr.shape[0]))

        # --- Trädslag fraction bundle (folded into aug_stack for crop
        # consistency): K target channels in [0,1] + 1 supervision-mask
        # channel (1 = supervise this pixel). Missing sidecar / disabled →
        # all-masked (mask all-zero), so the loss contributes nothing but the
        # tile still trains via the hard label. Stored as the LAST block so
        # the area/aux/extras offsets above are untouched.
        n_frac_bundle = 0
        if self.frac_dir is not None:
            frac_bundle = self._load_frac_bundle(entry["name"], h, w)  # (K+1,H,W)
            aug_stack = np.concatenate([aug_stack, frac_bundle], axis=0)
            n_frac_bundle = frac_bundle.shape[0]

        # --- Prithvi normalization: reflectance [0,1] -> DN -> z-score -
        # Normalize all T frames identically (mean/std tile across frames).
        # Skipped for tessera: the embeddings ship already normalized.
        if self.backbone_family != "tessera":
            n_frames = image.shape[0] // N_BANDS
            mean_t = np.tile(self._mean, (n_frames, 1, 1))  # (T*6, 1, 1)
            std_t = np.tile(self._std, (n_frames, 1, 1))
            image = (image * 10000.0 - mean_t) / std_t

        # --- Spatial augmentation / crop --------------------------------
        if self.augment:
            image, label, aug_stack = self._augment(image, label, aug_stack)
        else:
            image, label, aug_stack = self._center_crop(image, label, aug_stack)

        # Extract area_map (chan 0), aux_stack (next n_aux), and any
        # per-model extras (whatever follows) using explicit offsets so
        # adding new tail channels never silently breaks the aux unpack.
        area_map_cropped = aug_stack[0]                                  # (H', W')
        aux_stack = (aug_stack[n_area:n_area + n_aux]
                     if self.enable_aux else None)                       # (N, H', W')
        extras_cropped: dict[str, np.ndarray] = {}
        offset = n_area + n_aux
        for name, n_ch in extras_specs:
            extras_cropped[name] = aug_stack[offset:offset + n_ch]
            offset += n_ch
        # Frac bundle is the trailing block: K target channels + 1 mask.
        frac_target_cropped = None
        frac_mask_cropped = None
        if n_frac_bundle:
            bundle = aug_stack[offset:offset + n_frac_bundle]  # (K+1, H', W')
            frac_target_cropped = bundle[:-1]                  # (K, H', W')
            frac_mask_cropped = bundle[-1]                     # (H', W')
            offset += n_frac_bundle

        # Compute per-pixel loss weights from cropped area map
        area_t = torch.from_numpy(np.ascontiguousarray(area_map_cropped))
        pixel_weight = parcel_area_to_pixel_weights(
            area_t,
            mmu_ha=getattr(self, "_mmu_ha", 0.25),
            max_weight=getattr(self, "_area_weight_max", 4.0),
        )

        # --- Prithvi TL coordinate tensors ------------------------------
        # Tessera has no per-frame/location coords (annual embedding); the
        # head ignores them. Emit None so the batch key stays present.
        if self.backbone_family == "tessera":
            temporal_coords, location_coords = None, None
        else:
            n_frames = image.shape[0] // N_BANDS
            temporal_coords, location_coords = self._build_coords(
                data,
                doy,
                n_frames,
                resolved_location=(
                    (sealed_location["lat"], sealed_location["lon"])
                    if sealed_location is not None else None
                ),
            )

        # --- Build output dict ------------------------------------------
        # `spectral` carries the image tensor for ALL families (Prithvi
        # reflectance or tessera embedding); the trainer routes on family,
        # not on this key's channel count. Coord keys are only emitted when
        # present so every sample in a (single-family) batch has the same
        # key set: Prithvi always includes them, tessera always omits them.
        result: dict = {
            "spectral":     torch.from_numpy(np.ascontiguousarray(image)),
            "label":        torch.from_numpy(np.ascontiguousarray(label)),
            "pixel_weight": pixel_weight,
            "metadata": {
                "tile":   entry["name"],
                "source": source,
            },
        }
        if sealed_location is not None:
            result["metadata"].update({
                "location_source": sealed_location["source"],
                "location_bbox_west_3006": sealed_location["bbox_west_3006"],
                "location_bbox_south_3006": sealed_location["bbox_south_3006"],
                "location_bbox_east_3006": sealed_location["bbox_east_3006"],
                "location_bbox_north_3006": sealed_location["bbox_north_3006"],
                "location_easting_3006": sealed_location["easting_3006"],
                "location_northing_3006": sealed_location["northing_3006"],
                "location_lat": sealed_location["lat"],
                "location_lon": sealed_location["lon"],
                "era5_request_lat": sealed_location["era5_request_lat"],
                "era5_request_lon": sealed_location["era5_request_lon"],
            })
        if temporal_coords is not None:
            result["temporal_coords"] = temporal_coords
        if location_coords is not None:
            result["location_coords"] = location_coords

        # Multitemporal metadata
        if temporal_mask is not None:
            result["temporal_mask"] = torch.from_numpy(
                np.ascontiguousarray(temporal_mask)
            )
        if doy is not None:
            result["doy"] = torch.from_numpy(np.ascontiguousarray(doy))

        # Attach each auxiliary channel as (1, H, W) tensor
        if aux_stack is not None:
            for i, ch_name in enumerate(self.aux_channel_names):
                result[ch_name] = torch.from_numpy(
                    np.ascontiguousarray(aux_stack[i:i + 1])
                )  # (1, H', W')

        # Attach per-model extras (Clay / CROMA stacks) as full (C, H', W')
        # tensors. Raw reflectance [0,1] — the model's normalizer (e.g.
        # ClayNormalizer, CromaNormalizer) applies at forward time.
        for name, arr in extras_cropped.items():
            result[name] = torch.from_numpy(np.ascontiguousarray(arr))

        # Trädslag fraction target + supervision mask (crop-consistent).
        if frac_target_cropped is not None:
            result["frac"] = torch.from_numpy(
                np.ascontiguousarray(frac_target_cropped)
            )  # (K, H', W') float32 in [0,1]
            result["frac_mask"] = torch.from_numpy(
                np.ascontiguousarray(frac_mask_cropped)
            )  # (H', W') float32 {0,1}

        return result

    # ------------------------------------------------------------------
    # Trädslag fraction sidecar
    # ------------------------------------------------------------------

    def _load_frac_bundle(self, name: str, h: int, w: int) -> np.ndarray:
        """Load the Trädslag fraction sidecar as a (K+1, H, W) float32 bundle.

        Channels 0..K-1 are per-species crown-cover targets normalized to
        [0, 1] (raw 0-100 / 100). Channel K is the per-pixel supervision mask
        (1 = supervise, 0 = mask out), which is 0 where the pixel is flagged
        unreliable (``frac_unreliable``) OR carries no signal (all species 0).

        A missing / unreadable sidecar returns an all-masked bundle (mask
        all-zero, targets zero) so the tile still trains via the hard label
        but contributes nothing to the fraction loss.

        The bundle is folded into ``aug_stack`` by the caller so the same
        crop/flip/rotate applies — keeping the target pixel-aligned with the
        label and spectral crops.
        """
        NUM_SPECIES = 4
        bundle = np.zeros((NUM_SPECIES + 1, h, w), dtype=np.float32)  # mask=0
        if self.frac_dir is None:
            return bundle
        path = self.frac_dir / name
        if not path.exists():
            return bundle
        try:
            sc = np.load(path, allow_pickle=True)
            frac = np.asarray(sc["frac"], dtype=np.float32)          # (K,H,W) 0-100
            unreliable = np.asarray(sc["frac_unreliable"]).astype(bool)  # (H,W)
        except Exception:
            return bundle

        # Spatial alignment: crop/pad to (h, w) defensively (sidecars are
        # built at the tile's own size, so this is normally a no-op).
        if frac.shape[1:] != (h, w):
            k = frac.shape[0]
            fitted = np.zeros((k, h, w), dtype=np.float32)
            hh = min(h, frac.shape[1]); ww = min(w, frac.shape[2])
            fitted[:, :hh, :ww] = frac[:, :hh, :ww]
            frac = fitted
        if unreliable.shape != (h, w):
            fitted_u = np.zeros((h, w), dtype=bool)
            hh = min(h, unreliable.shape[0]); ww = min(w, unreliable.shape[1])
            fitted_u[:hh, :ww] = unreliable[:hh, :ww]
            unreliable = fitted_u

        k = min(frac.shape[0], NUM_SPECIES)
        bundle[:k] = frac[:k] / 100.0
        # Supervision mask: reliable AND at least one species present.
        has_signal = (bundle[:NUM_SPECIES] > 0).any(axis=0)
        bundle[NUM_SPECIES] = (has_signal & (~unreliable)).astype(np.float32)
        return bundle

    # ------------------------------------------------------------------
    # Prithvi TL coordinates
    # ------------------------------------------------------------------

    @staticmethod
    def _build_coords(
        data: dict,
        doy: np.ndarray | None,
        n_frames: int,
        *,
        resolved_location: tuple[float, float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build Prithvi TL coordinate tensors from tile metadata.

        Returns:
            temporal_coords: (T, 2) float32 [year, doy] per frame.
            location_coords: (2,) float32 [lat, lon] in WGS84 degrees.
        """
        year = resolve_tile_year(data)
        if year is None:
            raise ValueError("Prithvi coordinates require a resolved tile year")

        temporal_coords = np.zeros((n_frames, 2), dtype=np.float32)
        if doy is not None:
            temporal_coords[:len(doy), 1] = doy[:n_frames].astype(np.float32)

        # Per-frame year: frame_2016 gets its own year, autumn gets year-1
        bg_year = int(data.get("frame_2016_year", 2016))
        has_bg = int(data.get("has_frame_2016", 0)) == 1
        if n_frames >= 5 and has_bg:
            temporal_coords[0, 0] = float(bg_year)       # frame 0: background
            temporal_coords[1, 0] = float(year - 1)      # frame 1: autumn yr-1
            temporal_coords[2:, 0] = float(year)          # frames 2-4: growing season
        else:
            temporal_coords[0, 0] = float(year - 1) if n_frames >= 2 else float(year)
            temporal_coords[1:, 0] = float(year)

        if resolved_location is None:
            # Preserve the historical generic-dataset behaviour.  Sealed
            # smoke samples never enter this branch: construction resolves
            # and attests their real tile centroid first.
            easting = float(data.get("easting", 500_000))
            northing = float(data.get("northing", 6_500_000))
            lat, lon = _sweref99_to_wgs84(easting, northing)
        else:
            lat, lon = (float(value) for value in resolved_location)
            if not np.isfinite((lat, lon)).all():
                raise ValueError("Prithvi resolved location must be finite")
        location_coords = np.array([lat, lon], dtype=np.float32)

        return (
            torch.from_numpy(temporal_coords),
            torch.from_numpy(location_coords),
        )

    # ------------------------------------------------------------------
    # Frame selection
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_lulc_frame(data: np.lib.npyio.NpzFile) -> np.ndarray:
        """Select peak-summer frame from a 4-frame LULC tile.

        The ``image`` array has shape (24, H, W) = 4 frames x 6 bands.
        We pick the frame whose ``doy`` value is closest to
        ``PEAK_SUMMER_DOY`` (195, mid-July). Falls back to frame index 1
        if ``doy`` is missing or all zeros.

        Args:
            data: Loaded .npz file.

        Returns:
            (6, H, W) float32 single-date reflectance.
        """
        raw = data.get("spectral", data.get("image"))
        image = raw.astype(np.float32)  # (24, H, W)
        n_frames = image.shape[0] // N_BANDS

        # Determine best frame via day-of-year
        doy = data.get("doy", None)
        if doy is not None:
            doy = np.asarray(doy).ravel()
            if doy.shape[0] >= n_frames and np.any(doy > 0):
                # Pick frame closest to peak summer DOY
                valid_mask = doy[:n_frames] > 0
                if np.any(valid_mask):
                    distances = np.abs(doy[:n_frames].astype(np.float64) - PEAK_SUMMER_DOY)
                    distances[~valid_mask] = 9999
                    frame_idx = int(np.argmin(distances))
                else:
                    frame_idx = min(1, n_frames - 1)
            else:
                frame_idx = min(1, n_frames - 1)
        else:
            frame_idx = min(1, n_frames - 1)

        start = frame_idx * N_BANDS
        return image[start:start + N_BANDS]  # (6, H, W)

    @staticmethod
    def _extract_crop_frame(data: np.lib.npyio.NpzFile) -> np.ndarray:
        """Select peak-summer frame from a 3-frame crop tile.

        The ``spectral`` array has shape (18, H, W) = 3 frames x 6 bands.
        Frame index 1 corresponds to Jun-Jul (peak summer).

        Args:
            data: Loaded .npz file.

        Returns:
            (6, H, W) float32 single-date reflectance.
        """
        spectral = data["spectral"].astype(np.float32)  # (18, H, W)
        n_frames = spectral.shape[0] // N_BANDS

        # Prefer frame 1 (Jun-Jul); verify via seasons_valid if available
        frame_idx = min(1, n_frames - 1)
        seasons_valid = data.get("seasons_valid", None)
        if seasons_valid is not None:
            sv = np.asarray(seasons_valid).ravel()
            if sv.shape[0] >= n_frames and not sv[frame_idx]:
                # Frame 1 invalid -- pick first valid frame
                valid = np.where(sv[:n_frames])[0]
                if len(valid) > 0:
                    frame_idx = int(valid[0])

        start = frame_idx * N_BANDS
        return spectral[start:start + N_BANDS]  # (6, H, W)

    # ------------------------------------------------------------------
    # Per-model input tensors (Clay / CROMA)
    # ------------------------------------------------------------------

    @staticmethod
    def _select_best_frame_idx(
        data: np.lib.npyio.NpzFile, source: str, n_frames: int,
    ) -> int:
        """Pick the single best frame index for single-frame models.

        Mirrors the logic in ``_extract_lulc_frame`` (DOY closest to
        peak-summer) and ``_extract_crop_frame`` (index 1, falling back
        to first valid via ``seasons_valid``). Returned index is in
        ``[0, n_frames)``.
        """
        if source == "lulc":
            doy = data.get("doy", None)
            if doy is not None:
                doy = np.asarray(doy).ravel()
                if doy.shape[0] >= n_frames and np.any(doy > 0):
                    valid_mask = doy[:n_frames] > 0
                    if np.any(valid_mask):
                        distances = np.abs(
                            doy[:n_frames].astype(np.float64) - PEAK_SUMMER_DOY
                        )
                        distances[~valid_mask] = 9999
                        return int(np.argmin(distances))
            return min(1, n_frames - 1)
        # crop branch
        frame_idx = min(1, n_frames - 1)
        seasons_valid = data.get("seasons_valid", None)
        if seasons_valid is not None:
            sv = np.asarray(seasons_valid).ravel()
            if sv.shape[0] >= n_frames and not sv[frame_idx]:
                valid = np.where(sv[:n_frames])[0]
                if len(valid) > 0:
                    return int(valid[0])
        return frame_idx

    def _select_aux_band_frame(
        self,
        data: np.lib.npyio.NpzFile,
        key: str,
        has_flag: str,
        idx: int,
    ) -> "np.ndarray | None":
        """Return the (H, W) frame ``idx`` of a per-frame aux band, or None.

        Used for CROMA's B01/B09 (and extensible to other optional bands).
        The band is enriched as a (T, H, W) array with a ``has_<band>`` flag.
        Returns None (→ caller zero-pads) only when the band is genuinely
        absent: the has-flag is 0 OR the key is missing OR the requested
        frame is entirely nodata. NaN pixels within an otherwise-valid frame
        are scrubbed to 0. A None return is logged once so a real coverage
        gap is visible rather than silently zero-padded.

        Args:
            data: Loaded tile (or _LabelOverlay wrapper).
            key: Band array key, e.g. ``"b01"``.
            has_flag: Presence flag key, e.g. ``"has_b01"``.
            idx: Best-frame index (same as the optical/SAR selection).

        Returns:
            (H, W) float32 frame, or None if the band is unavailable.
        """
        # Respect the has-flag first: an explicit 0 means "not fetched".
        flag = data.get(has_flag, None)
        if flag is not None and int(np.asarray(flag).ravel()[0]) == 0:
            self._log_band_missing(key, reason="has-flag=0")
            return None

        arr = data.get(key, None)
        if arr is None:
            self._log_band_missing(key, reason="key absent")
            return None

        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2:
            frame = arr  # already a single (H, W) frame
        else:
            n = arr.shape[0]
            f = idx if idx < n else n - 1
            frame = arr[f]  # (H, W)

        # A frame that is entirely NaN/zero carries no signal → treat as
        # absent so the loader zero-pads consistently (and logs it).
        if not np.isfinite(frame).any() or np.all(frame == 0):
            self._log_band_missing(key, reason="frame all-nodata")
            return None

        return np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)

    def _log_band_missing(self, key: str, *, reason: str) -> None:
        """Log a per-band zero-pad fallback at most once per (key, reason).

        Keeps the training log readable: a systematic coverage gap surfaces
        once, not once per tile per epoch.
        """
        seen = getattr(self, "_band_miss_logged", None)
        if seen is None:
            seen = set()
            self._band_miss_logged = seen
        tag = (key, reason)
        if tag not in seen:
            seen.add(tag)
            logger.warning(
                "CROMA s2_croma: band %r unavailable (%s) — zero-padding "
                "this band. (Logged once per reason.)",
                key, reason,
            )

    def _build_model_specific_tensors(
        self, data: np.lib.npyio.NpzFile, source: str,
    ) -> dict[str, np.ndarray]:
        """Build per-model input stacks at native tile resolution.

        Returns a dict of ``{model_key: (C, H, W) float32 array}`` in
        raw reflectance [0, 1]. Caller is responsible for stacking
        these into the augment pipeline so the same crop/flip is
        applied; model-side normalizers handle scaling at forward time.

        Raises:
            KeyError: if a required enrichment key (``b08``, ``rededge``)
                is missing for a tile that the caller asked Clay/CROMA
                tensors for. Lets the dataset's retry-on-error loop
                pick a different tile instead of silently emitting zeros.
        """
        # Defer the build_*_tensor imports — they pull in torch as a
        # transitive dep through their type guards, but unified_dataset
        # is imported during dataset discovery before torch is needed.
        from imint.fm.loaders.clay import build_s2_clay_tensor
        from imint.fm.loaders.croma import build_s2_croma_tensor

        raw_spectral = data.get("spectral", data.get("image"))
        if raw_spectral is None:
            raise KeyError("tile missing 'spectral'/'image'")
        spectral = np.asarray(raw_spectral, dtype=np.float32)  # (T*6, H, W)
        n_frames = spectral.shape[0] // N_BANDS

        # Single, shared best-frame index so the optical stack AND the SAR
        # frame refer to the SAME temporal frame (year-consistent inputs).
        idx = self._select_best_frame_idx(data, source, n_frames)
        spectral_6band = spectral[idx * N_BANDS:(idx + 1) * N_BANDS]  # (6, H, W)

        needs_optical = bool(
            {"clay_v1_5", "croma_base"} & set(self.model_keys)
        )
        needs_sar = bool(
            {"croma_base", "terramind_v1_base"} & set(self.model_keys)
        )
        # TerraMind consumes a RAW 6-band S2 frame (its own normalizer runs
        # in the forward router). Our `spectral` order (B02,B03,B04,B8A,B11,
        # B12) matches TerraMind's [BLUE,GREEN,RED,NIR_NARROW,SWIR_1,SWIR_2],
        # so we emit the un-z-scored frame directly under `s2_terramind`.
        needs_s2_raw6 = "terramind_v1_base" in self.model_keys

        # b08 / rededge only needed for the optical stacks (Clay/CROMA).
        b08_frame = None
        rededge_frame = None
        if needs_optical:
            # b08 layout: (T, H, W) per scripts/enrich_tiles_b08.py
            b08_all = data.get("b08", None)
            if b08_all is None:
                raise KeyError("tile missing 'b08' (run enrich_tiles_b08.py)")
            b08_all = np.asarray(b08_all, dtype=np.float32)
            b08_frame = b08_all[idx]  # (H, W)

            # rededge layout: (T*3, H, W) per scripts/enrich_tiles_rededge.py
            rededge_all = data.get("rededge", None)
            if rededge_all is None:
                raise KeyError(
                    "tile missing 'rededge' (run enrich_tiles_rededge.py)"
                )
            rededge_all = np.asarray(rededge_all, dtype=np.float32)
            rededge_frame = rededge_all[idx * 3:(idx + 1) * 3]  # (3, H, W)

        out: dict[str, np.ndarray] = {}
        if needs_s2_raw6:
            # Raw 6-band reflectance frame for TerraMind's S2L2A modality.
            out["s2_terramind"] = np.ascontiguousarray(
                spectral_6band, dtype=np.float32,
            )  # (6, H, W)
        if "clay_v1_5" in self.model_keys:
            out["s2_clay"] = build_s2_clay_tensor(
                spectral_6band, b08_frame, rededge=rededge_frame,
            ).astype(np.float32)  # (10, H, W)
        if "croma_base" in self.model_keys:
            # CROMA's native S2 input is the FULL 12-band stack. B01 (coastal
            # aerosol) and B09 (water vapour) are enriched onto the tile as
            # (T, H, W) per-frame arrays with has_b01/has_b09 flags. Feed the
            # REAL bands (same best-frame idx, NaN-scrubbed) so CROMA is not
            # handicapped by two zero-padded channels. Only fall back to
            # zero-pad if a band is genuinely absent — and log it, so a
            # coverage gap is visible rather than silent.
            b01_frame = self._select_aux_band_frame(data, "b01", "has_b01", idx)
            b09_frame = self._select_aux_band_frame(data, "b09", "has_b09", idx)
            out["s2_croma"] = build_s2_croma_tensor(
                spectral_6band, b08_frame, rededge=rededge_frame,
                b01=b01_frame, b09=b09_frame,
            ).astype(np.float32)  # (12, H, W)

        # SAR (VV/VH) for CROMA joint + TerraMind S1GRD. v3 layout is a single
        # per-orbit growing-season median composite (2, H, W) in **linear** γ⁰
        # (RTC, PC) per scripts/enrich_tiles_s1.py — a direct read, no frame
        # selection. The normalizer log-transforms internally, so the stored
        # composite must be linear. Older v1 (±3-day (T*2,H,W)) and v2
        # (CDSE-dB) layouts are a hard break: require s1_enrich_v==3 and fail
        # loud rather than feed a mis-shaped or double-logged stack to the
        # SAR encoders.
        if needs_sar:
            s1_ver = int(data.get("s1_enrich_v", 0))
            if s1_ver != 3:
                raise KeyError(
                    f"tile requires s1_enrich_v==3 RTC γ⁰ season composite for "
                    f"model_keys={sorted(set(self.model_keys) & {'croma_base', 'terramind_v1_base'})}"
                    f" but found s1_enrich_v={s1_ver}. Re-run the S1 season "
                    f"enrichment job (scripts/enrich_tiles_s1.py "
                    f"--s1-backend pc-rtc / k8s/enrich-s1-season-job.yaml) "
                    f"over this tile."
                )
            s1_comp = data.get("s1_vv_vh", None)
            if s1_comp is None:
                raise KeyError(
                    "tile missing 's1_vv_vh' composite (run "
                    "scripts/enrich_tiles_s1.py / k8s/enrich-s1-season-job.yaml)"
                )
            s1_comp = np.asarray(s1_comp, dtype=np.float32)  # (2, H, W)
            if s1_comp.ndim != 3 or s1_comp.shape[0] != 2:
                raise ValueError(
                    f"s1_vv_vh must be (2, H, W) in v3, got {s1_comp.shape}. "
                    f"Re-run the S1 season enrichment over this tile."
                )
            # NaN-scrub guard: a fully-nodata pixel is 0 in the composite; any
            # residual non-finite → 0 so the dB normalizer never sees NaN.
            out["s1_vv_vh"] = np.nan_to_num(
                s1_comp, nan=0.0, posinf=0.0, neginf=0.0,
            ).astype(np.float32)
        return out

    # ------------------------------------------------------------------
    # Label construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_label(data: np.lib.npyio.NpzFile, source: str) -> np.ndarray:
        """Return the pre-built unified 23-class label from the tile.

        Tiles built by build_labels.py store the fully-merged label in
        data["label"] (NMD + gated LPIS + gated SKS). Return it directly —
        no runtime merging needed.

        Args:
            data: Loaded .npz file.
            source: ``"lulc"`` or ``"crop"``.

        Returns:
            (H, W) int64 unified label array.
        """
        # Always use "label" — it contains the unified 23-class output from
        # build_labels.py.  "nmd_label" holds raw NMD indices (0-255) and
        # must NOT be used as training labels (causes CUDA assert on
        # values >= num_classes).
        if "label" not in data:
            img = data.get("spectral", data.get("image"))
            h, w = img.shape[1], img.shape[2]
            return np.zeros((h, w), dtype=np.int64)
        label = np.asarray(data["label"]).astype(np.int64)
        # Safety clamp: any out-of-range values → background (0)
        label[label >= NUM_CLASSES] = 0
        label[label < 0] = 0
        return label

    @staticmethod
    def _extract_all_frames(
        data: np.lib.npyio.NpzFile,
        source: str,
        num_frames: int,
        strict_smoke: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract all temporal frames for multi-temporal training.

        Frame layout when ``num_frames=5`` and ``frame_2016`` is present:
            0: 2016 summer background (clearcut change detection anchor)
            1: autumn year-1 (Sep-Oct)
            2-4: VPP-guided growing season

        When ``frame_2016`` is absent, frame 0 is zero-padded (masked).
        When ``num_frames=4``, the background frame is omitted entirely.

        Returns:
            image: (T*6, H, W) float32 stacked frames.
            temporal_mask: (T,) uint8, 1 = valid frame, 0 = padded.
            doy: (T,) int32 day-of-year per frame.
        """
        raw = data.get("spectral", data.get("image")).astype(np.float32)
        c, h, w = raw.shape
        tile_frames = c // N_BANDS

        if strict_smoke:
            if raw.ndim != 3 or c != num_frames * N_BANDS:
                raise ValueError(
                    f"Sealed smoke tile must contain exactly {num_frames}x{N_BANDS} "
                    f"spectral channels, got {raw.shape}"
                )
            year = resolve_tile_year(data)
            if year is None:
                raise ValueError("Sealed smoke tile lacks a resolved year")
            validate_smoke_temporal_metadata(
                data, year, num_frames=num_frames, min_valid_frames=3,
            )

        # Temporal metadata from the 4 base frames in the tile
        tile_mask = data.get("temporal_mask", None)
        tile_doy = data.get("doy", None)
        if tile_mask is not None:
            tile_mask = np.asarray(tile_mask).ravel()[:tile_frames]
        else:
            tile_mask = np.ones(tile_frames, dtype=np.uint8)
        if tile_doy is not None:
            tile_doy = np.asarray(tile_doy).ravel()[:tile_frames].astype(np.int32)
        else:
            tile_doy = np.zeros(tile_frames, dtype=np.int32)

        # --- Prepend frame_2016 as background anchor (frame 0) ---
        # Only when num_frames > tile_frames (i.e. 5 requested, tile has 4)
        use_bg = num_frames > tile_frames
        if use_bg:
            has_bg = int(data.get("has_frame_2016", 0)) == 1
            if has_bg:
                bg_frame = np.asarray(data["frame_2016"], dtype=np.float32)  # (6, H, W)
                bg_doy = int(data.get("frame_2016_doy", 166))
            else:
                bg_frame = np.zeros((N_BANDS, h, w), dtype=np.float32)
                bg_doy = 0

            # Build output: [bg_frame, base_frames..., zero-pad if needed]
            n_base = min(tile_frames, num_frames - 1)
            image = np.zeros((num_frames * N_BANDS, h, w), dtype=np.float32)
            image[:N_BANDS] = bg_frame
            image[N_BANDS:N_BANDS + n_base * N_BANDS] = raw[:n_base * N_BANDS]

            temporal_mask = np.zeros(num_frames, dtype=np.uint8)
            temporal_mask[0] = 1 if has_bg else 0
            temporal_mask[1:1 + n_base] = tile_mask[:n_base]

            doy = np.zeros(num_frames, dtype=np.int32)
            doy[0] = bg_doy
            doy[1:1 + n_base] = tile_doy[:n_base]
        else:
            # Standard 4-frame path (no background frame)
            if tile_frames >= num_frames:
                image = raw[:num_frames * N_BANDS]
                temporal_mask = tile_mask[:num_frames]
                doy = tile_doy[:num_frames]
            else:
                image = np.zeros((num_frames * N_BANDS, h, w), dtype=np.float32)
                image[:tile_frames * N_BANDS] = raw
                temporal_mask = np.zeros(num_frames, dtype=np.uint8)
                temporal_mask[:tile_frames] = tile_mask
                doy = np.zeros(num_frames, dtype=np.int32)
                doy[:tile_frames] = tile_doy

        # Replace zero-padded frames with nearest valid frame
        for t in range(num_frames):
            if temporal_mask[t] == 0:
                valid_indices = np.where(temporal_mask > 0)[0]
                if len(valid_indices) > 0:
                    nearest = valid_indices[
                        np.argmin(np.abs(valid_indices - t))
                    ]
                    start_dst = t * N_BANDS
                    start_src = nearest * N_BANDS
                    image[start_dst:start_dst + N_BANDS] = (
                        image[start_src:start_src + N_BANDS]
                    )

        return image, temporal_mask, doy

    # ------------------------------------------------------------------
    # Auxiliary channels
    # ------------------------------------------------------------------

    def _load_aux_channels(
        self,
        data: np.lib.npyio.NpzFile,
        h: int,
        w: int,
        tile_name: str,
        *,
        expected_location: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """Load, fill, and z-score-normalize the enabled auxiliary channels.

        Channels and their order come from ``self.aux_channel_names``
        (defaults to the canonical 10-channel ``AUX_CHANNEL_NAMES``).
        A missing channel is filled with the channel's nodata sentinel:
        ``NaN`` for NaN-nodata channels (markfukt) so it normalizes to the
        channel mean, ``0`` otherwise (existing behaviour).

        Args:
            data: Loaded .npz file.
            h: Spatial height of label array.
            w: Spatial width of label array.

        Returns:
            (N, H, W) float32 stacked aux channels, normalized.
        """
        # ΔSAR channels are computed once (both come from one dB-difference
        # call) and cached so `delta_vv`/`delta_vh` don't recompute. Missing
        # 2016 composite → None → each channel emits zeros ("no change").
        delta_stack: np.ndarray | None = None
        if AUX_COMPUTED_CHANNELS & set(self.aux_channel_names):
            delta_stack = compute_delta_sar(data)  # (2, H, W) [ΔVV, ΔVH] or None

        era5_data = None
        if self.era5_mode == "treatment" and ERA5_AUX_CHANNELS & set(self.aux_channel_names):
            sidecar = self.era5_dir / tile_name
            if not sidecar.exists():
                raise FileNotFoundError(f"Missing ERA5 treatment sidecar: {sidecar}")
            with np.load(sidecar, allow_pickle=False) as loaded:
                era5_data = {name: loaded[name] for name in loaded.files}
            required_sidecar_fields = ERA5_AUX_CHANNELS | {
                "tile_name", "year", "cutoff_date", "lat", "lon",
                "era5_request_lat", "era5_request_lon",
                "era5_land_cell_lat", "era5_land_cell_lon",
                "era5_atmosphere_cell_lat", "era5_atmosphere_cell_lon",
            }
            missing = required_sidecar_fields - set(era5_data)
            if missing:
                raise ValueError(f"ERA5 sidecar {sidecar} lacks {sorted(missing)}")
            if str(np.asarray(era5_data.get("tile_name", "")).item()) != tile_name:
                raise ValueError(f"ERA5 sidecar identity mismatch: {sidecar}")
            expected_year = resolve_tile_year(data)
            stored_year = int(np.asarray(era5_data.get("year", -1)).item())
            if expected_year is None or stored_year != expected_year:
                raise ValueError(
                    f"ERA5 year mismatch for {tile_name}: tile={expected_year}, sidecar={stored_year}"
                )
            expected_cutoff = resolve_growing_cutoff(data, expected_year)
            stored_cutoff = str(np.asarray(era5_data.get("cutoff_date", "")).item())
            if stored_cutoff != expected_cutoff:
                raise ValueError(
                    f"ERA5 cutoff mismatch for {tile_name}: tile={expected_cutoff}, "
                    f"sidecar={stored_cutoff}"
                )
            if not all(np.isfinite(float(np.asarray(era5_data[name]).item()))
                       for name in ERA5_AUX_CHANNELS):
                raise ValueError(f"ERA5 sidecar {sidecar} contains non-finite values")
            if not all(
                np.isfinite(float(np.asarray(era5_data[name]).item()))
                for name in (
                    "lat", "lon", "era5_request_lat", "era5_request_lon",
                    "era5_land_cell_lat", "era5_land_cell_lon",
                    "era5_atmosphere_cell_lat", "era5_atmosphere_cell_lon",
                )
            ):
                raise ValueError(f"ERA5 sidecar {sidecar} has invalid coordinates")
            sidecar_lat = float(np.asarray(era5_data["lat"]).item())
            sidecar_lon = float(np.asarray(era5_data["lon"]).item())
            if expected_location is not None and not np.allclose(
                (sidecar_lat, sidecar_lon),
                expected_location,
                rtol=0.0,
                atol=1e-7,
            ):
                raise ValueError(
                    f"ERA5 sidecar location mismatch for {tile_name}: "
                    f"resolved_tile={expected_location}, "
                    f"sidecar={(sidecar_lat, sidecar_lon)}"
                )
            grid = era5_grid_context(
                sidecar_lat,
                sidecar_lon,
            )
            stored_grid = {
                "request_lat": float(np.asarray(era5_data["era5_request_lat"]).item()),
                "request_lon": float(np.asarray(era5_data["era5_request_lon"]).item()),
                "land_cell": {
                    "lat": float(np.asarray(era5_data["era5_land_cell_lat"]).item()),
                    "lon": float(np.asarray(era5_data["era5_land_cell_lon"]).item()),
                },
                "atmosphere_cell": {
                    "lat": float(np.asarray(era5_data["era5_atmosphere_cell_lat"]).item()),
                    "lon": float(np.asarray(era5_data["era5_atmosphere_cell_lon"]).item()),
                },
            }
            request_matches = np.allclose(
                (stored_grid["request_lat"], stored_grid["request_lon"]),
                (grid["request_lat"], grid["request_lon"]),
                rtol=0.0,
                atol=1e-7,
            )
            response_cells_match = (
                era5_api_cell_coords_match(
                    stored_grid["land_cell"]["lat"],
                    stored_grid["land_cell"]["lon"],
                    grid["land_cell"]["lat"],
                    grid["land_cell"]["lon"],
                )
                and era5_api_cell_coords_match(
                    stored_grid["atmosphere_cell"]["lat"],
                    stored_grid["atmosphere_cell"]["lon"],
                    grid["atmosphere_cell"]["lat"],
                    grid["atmosphere_cell"]["lon"],
                )
            )
            if not request_matches or not response_cells_match:
                raise ValueError(f"ERA5 sidecar {sidecar} has inconsistent grid cells")

        aux_arrays: list[np.ndarray] = []
        for ch_name in self.aux_channel_names:
            if ch_name in AUX_COMPUTED_CHANNELS:
                # Computed at read time from S1 keys (not a stored array).
                # delta_vv → row 0, delta_vh → row 1. Missing 2016 → zeros
                # (a zero Δ is the neutral "no change" the fusion expects).
                if delta_stack is None:
                    arr = np.zeros((h, w), dtype=np.float32)
                else:
                    row = 0 if ch_name == "delta_vv" else 1
                    arr = delta_stack[row]
                    if arr.shape != (h, w):
                        arr = arr[:h, :w]
                        if arr.shape[0] < h or arr.shape[1] < w:
                            padded = np.zeros((h, w), dtype=np.float32)
                            padded[:arr.shape[0], :arr.shape[1]] = arr
                            arr = padded
            elif ch_name in ERA5_AUX_CHANNELS:
                if self.era5_mode == "control":
                    # The neutral arm is exclusive: source NPZ ERA5 keys are
                    # deliberately ignored so they cannot contaminate control.
                    value = AUX_NORM[ch_name][0]
                elif self.era5_mode == "treatment" and era5_data is not None:
                    value = float(np.asarray(era5_data[ch_name]).item())
                else:
                    raise ValueError(
                        f"ERA5 channel {ch_name} requires control or treatment mode"
                    )
                arr = np.full((h, w), value, dtype=np.float32)
            elif ch_name in data:
                arr = data[ch_name].astype(np.float32)
                # Ensure spatial dimensions match
                if arr.shape != (h, w):
                    arr = arr[:h, :w]
                    # Pad if smaller (edge case)
                    if arr.shape[0] < h or arr.shape[1] < w:
                        fill = (np.nan if ch_name in AUX_NAN_NODATA_CHANNELS
                                else 0.0)
                        padded = np.full((h, w), fill, dtype=np.float32)
                        padded[:arr.shape[0], :arr.shape[1]] = arr
                        arr = padded
            else:
                # Absent channel → all-nodata. NaN sentinel for markfukt
                # (normalizes to channel mean), 0 for every other channel.
                # ERA5 absence means neutral after normalization, not a raw
                # physical zero (which would be an extreme weather signal).
                fill = (np.nan if ch_name in AUX_NAN_NODATA_CHANNELS
                        else 0.0)
                arr = np.full((h, w), fill, dtype=np.float32)
            aux_arrays.append(arr)

        aux_stack = np.stack(aux_arrays, axis=0)  # (N, H, W)

        # Decode / log-transform / z-score, per channel (shared transform).
        for i, ch_name in enumerate(self.aux_channel_names):
            aux_stack[i] = normalize_aux_channel(ch_name, aux_stack[i])

        return aux_stack

    # ------------------------------------------------------------------
    # Augmentation and cropping
    # ------------------------------------------------------------------

    def _augment(
        self,
        image: np.ndarray,
        label: np.ndarray,
        aux: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Apply data augmentation (training only).

        Augmentations (all applied consistently to image, label, aux):
            - Random crop from fetch size (256) to patch size (224)
            - Random horizontal flip (p=0.5)
            - Random vertical flip (p=0.5)
            - Random 90-degree rotation (k in {0, 1, 2, 3})

        Args:
            image: (C, H, W) float32.
            label: (H, W) int64.
            aux: Optional (N, H, W) float32 stacked aux channels.

        Returns:
            Tuple of (image, label, aux) after augmentation.
        """
        # Random crop
        image, label, aux = self._random_crop(image, label, aux)

        # Random horizontal flip
        if random.random() > 0.5:
            image = image[:, :, ::-1]
            label = label[:, ::-1]
            if aux is not None:
                aux = aux[:, :, ::-1]

        # Random vertical flip
        if random.random() > 0.5:
            image = image[:, ::-1, :]
            label = label[::-1, :]
            if aux is not None:
                aux = aux[:, ::-1, :]

        # Random 90-degree rotation
        k = random.randint(0, 3)
        if k > 0:
            image = np.rot90(image, k, axes=(1, 2))
            label = np.rot90(label, k, axes=(0, 1))
            if aux is not None:
                aux = np.rot90(aux, k, axes=(1, 2))

        return image, label, aux

    def _random_crop(
        self,
        image: np.ndarray,
        label: np.ndarray,
        aux: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Random spatial crop to ``patch_size``.

        Args:
            image: (C, H, W).
            label: (H, W).
            aux: Optional (N, H, W).

        Returns:
            Cropped (image, label, aux).
        """
        p = self.patch_size
        _, h, w = image.shape
        if h <= p and w <= p:
            return image, label, aux

        y = random.randint(0, max(h - p, 0))
        x = random.randint(0, max(w - p, 0))

        a_crop = aux[:, y:y + p, x:x + p] if aux is not None else None
        return image[:, y:y + p, x:x + p], label[y:y + p, x:x + p], a_crop

    def _center_crop(
        self,
        image: np.ndarray,
        label: np.ndarray,
        aux: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Center spatial crop to ``patch_size``.

        Args:
            image: (C, H, W).
            label: (H, W).
            aux: Optional (N, H, W).

        Returns:
            Cropped (image, label, aux).
        """
        p = self.patch_size
        _, h, w = image.shape
        if h <= p and w <= p:
            return image, label, aux

        y = max((h - p) // 2, 0)
        x = max((w - p) // 2, 0)

        a_crop = aux[:, y:y + p, x:x + p] if aux is not None else None
        return image[:, y:y + p, x:x + p], label[y:y + p, x:x + p], a_crop


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def compute_unified_class_weights(
    dataset: UnifiedDataset,
    max_weight: float = 10.0,
    max_tiles: int | None = None,
) -> np.ndarray:
    """Compute inverse-frequency class weights by scanning all tiles.

    Loads each tile's label, counts per-class pixels, and returns
    weights inversely proportional to class frequency.  Weights are
    capped at ``max_weight`` to prevent training instability on very
    rare classes.  Background (class 0) always receives weight 0.

    This can be slow for large datasets.  Use ``max_tiles`` to limit
    the scan to a random subset for faster approximation.

    Args:
        dataset: A UnifiedDataset instance.
        max_weight: Maximum per-class weight (default 10.0).
        max_tiles: If set, scan at most this many tiles (randomly
            selected) instead of the full dataset.

    Returns:
        (NUM_CLASSES,) float32 array of class weights.  Index 0 is
        background (weight = 0).
    """
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)

    # Determine which tiles to scan
    indices = list(range(len(dataset)))
    if max_tiles is not None and max_tiles < len(indices):
        rng = random.Random(42)
        indices = rng.sample(indices, max_tiles)

    for idx in indices:
        entry = dataset._entries[idx]
        try:
            data = np.load(entry["path"], allow_pickle=True)
        except Exception:
            continue

        label = UnifiedDataset._build_label(data, entry["source"])
        classes, pixel_counts = np.unique(label, return_counts=True)
        for cls, cnt in zip(classes, pixel_counts):
            if 0 <= cls < NUM_CLASSES:
                counts[cls] += cnt

    # Inverse-frequency weighting
    total = counts.sum()
    weights = np.ones(NUM_CLASSES, dtype=np.float32)
    for i in range(1, NUM_CLASSES):
        if counts[i] > 0:
            w = total / (NUM_CLASSES * counts[i])
            weights[i] = min(w, max_weight)
        else:
            weights[i] = max_weight  # Never-seen class gets max weight

    weights[0] = 0.0  # Ignore background
    return weights


# ---------------------------------------------------------------------------
# Weighted sampler
# ---------------------------------------------------------------------------

def build_unified_sampler(
    dataset: UnifiedDataset,
    max_tile_weight: float = 5.0,
) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that oversamples tiles containing
    rare classes.

    For each tile, we check which classes are present and assign the
    tile a weight equal to the maximum inverse-frequency weight of its
    constituent classes.  This ensures tiles with rare classes (e.g.
    harvest, potato, pulses) are drawn more often.

    Args:
        dataset: A UnifiedDataset instance.
        max_tile_weight: Cap on per-tile sampling weight to prevent
            extreme oversampling of any single tile.

    Returns:
        A ``WeightedRandomSampler`` suitable for ``DataLoader(sampler=...)``.
    """
    n_tiles = len(dataset)
    class_tile_counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    tile_classes: list[set[int]] = []

    # First pass: discover which classes each tile contains
    for entry in dataset._entries:
        try:
            data = np.load(entry["path"], allow_pickle=True)
            label = UnifiedDataset._build_label(data, entry["source"])
            present = set(np.unique(label).tolist())
        except Exception:
            present = set()

        tile_classes.append(present)
        for cls in present:
            if 0 < cls < NUM_CLASSES:
                class_tile_counts[cls] += 1

    # Compute inverse tile-frequency per class
    class_tile_counts = np.maximum(class_tile_counts, 1.0)
    class_rarity = n_tiles / (NUM_CLASSES * class_tile_counts)

    # Second pass: assign per-tile weight = max rarity of present classes
    sample_weights: list[float] = []
    for present in tile_classes:
        if not present:
            sample_weights.append(1.0)
            continue
        w = max(
            class_rarity[cls] for cls in present
            if 0 < cls < NUM_CLASSES
        ) if any(0 < cls < NUM_CLASSES for cls in present) else 1.0
        sample_weights.append(min(float(w), max_tile_weight))

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=n_tiles,
        replacement=True,
    )
