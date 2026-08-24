#!/usr/bin/env python3
"""Read every sealed smoke sample through the real dataset before training."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imint.training.config import TrainingConfig
from imint.training.era5_aux import (
    era5_api_cell_coords_match,
    era5_grid_context,
    format_era5_cell_id,
)
from imint.training.unified_dataset import UnifiedDataset


def _atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _expected_support(manifest: dict, split: str) -> dict[int, int]:
    counts = manifest.get("label_support", {}).get(split, {}).get(
        "pixel_counts", {}
    )
    if not isinstance(counts, dict):
        raise ValueError(f"Cohort lacks {split} pixel support")
    return {int(label): int(count) for label, count in counts.items()}


def _attest_location(
    *,
    sample: dict,
    manifest_item: dict,
    tile_name: str,
) -> dict[str, float | str]:
    """Bind Prithvi's location token to the cohort's persisted ERA5 request."""
    metadata = sample.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Missing sample metadata for {tile_name}")
    required = {
        "location_source",
        "location_bbox_west_3006",
        "location_bbox_south_3006",
        "location_bbox_east_3006",
        "location_bbox_north_3006",
        "location_easting_3006",
        "location_northing_3006",
        "location_lat",
        "location_lon",
        "era5_request_lat",
        "era5_request_lon",
    }
    missing = required - set(metadata)
    if missing:
        raise ValueError(
            f"Missing sealed location attestation for {tile_name}: {sorted(missing)}"
        )
    if metadata["location_source"] != "sealed_cohort_manifest_v4":
        raise ValueError(
            f"Unexpected location source for {tile_name}: "
            f"{metadata['location_source']!r}"
        )

    lat = float(metadata["location_lat"])
    lon = float(metadata["location_lon"])
    if not math.isfinite(lat) or not math.isfinite(lon):
        raise ValueError(f"Non-finite resolved location for {tile_name}")
    location = sample.get("location_coords")
    expected_location = torch.tensor([lat, lon], dtype=torch.float32)
    if (
        location is None
        or tuple(location.shape) != (2,)
        or not torch.allclose(
            location.to(torch.float32), expected_location, rtol=0.0, atol=1e-5,
        )
    ):
        raise ValueError(
            f"Prithvi location token differs from resolved tile location for {tile_name}"
        )

    try:
        manifest_bbox = manifest_item["bbox_3006"]
        manifest_center = manifest_item["location_3006"]
        manifest_wgs84 = manifest_item["location_wgs84"]
        exact_pairs = (
            (metadata["location_bbox_west_3006"], manifest_bbox["west"]),
            (metadata["location_bbox_south_3006"], manifest_bbox["south"]),
            (metadata["location_bbox_east_3006"], manifest_bbox["east"]),
            (metadata["location_bbox_north_3006"], manifest_bbox["north"]),
            (metadata["location_easting_3006"], manifest_center["easting"]),
            (metadata["location_northing_3006"], manifest_center["northing"]),
        )
        exact_location_matches = all(
            float(actual) == float(expected) for actual, expected in exact_pairs
        ) and all(
            math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-9)
            for actual, expected in (
                (lat, manifest_wgs84["lat"]),
                (lon, manifest_wgs84["lon"]),
            )
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Incomplete exact cohort location metadata for {tile_name}"
        ) from exc
    if not exact_location_matches:
        raise ValueError(
            f"Sample location metadata differs from sealed cohort for {tile_name}"
        )

    grid = era5_grid_context(lat, lon)
    expected_request = manifest_item.get("era5_request")
    expected_land = manifest_item.get("era5_land_cell")
    expected_atmosphere = manifest_item.get("era5_atmosphere_cell")
    try:
        request_matches = all(
            math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-7)
            for actual, expected in (
                (grid["request_lat"], expected_request["lat"]),
                (grid["request_lon"], expected_request["lon"]),
                (grid["request_lat"], metadata["era5_request_lat"]),
                (grid["request_lon"], metadata["era5_request_lon"]),
            )
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
            f"Incomplete cohort location metadata for {tile_name}"
        ) from exc
    cell_id = format_era5_cell_id(
        grid["atmosphere_cell"]["lat"], grid["atmosphere_cell"]["lon"],
    )
    if not (
        request_matches
        and land_matches
        and atmosphere_matches
        and manifest_item.get("era5_cell") == cell_id
    ):
        raise ValueError(
            f"Resolved Prithvi/ERA5 location differs from cohort for {tile_name}"
        )

    return {
        "tile": tile_name,
        "bbox_3006": {
            key: float(manifest_bbox[key])
            for key in ("west", "south", "east", "north")
        },
        "easting_3006": float(manifest_center["easting"]),
        "northing_3006": float(manifest_center["northing"]),
        "lat": lat,
        "lon": lon,
        "era5_request_lat": float(grid["request_lat"]),
        "era5_request_lon": float(grid["request_lon"]),
        "era5_cell": cell_id,
    }


def validate_split(
    *,
    tile_dir: Path,
    cohort_dir: Path,
    manifest: dict,
    split: str,
    era5_mode: str,
    era5_dir: Path | None,
) -> dict:
    config = TrainingConfig(
        enable_markfukt_channel=True,
        enable_era5_channels=True,
        era5_mode=era5_mode,
    )
    dataset = UnifiedDataset(
        lulc_dir=tile_dir,
        split=split,
        patch_size=int(manifest["model_patch_px"]),
        enable_aux=True,
        augment_override=False,
        multitemporal=True,
        num_temporal_frames=4,
        aux_channel_names=config.enabled_aux_names,
        era5_dir=era5_dir,
        era5_mode=era5_mode,
        split_dir=cohort_dir,
        backbone_family="prithvi",
    )
    expected_names = [
        line.strip()
        for line in (cohort_dir / f"split_{split}.txt").read_text().splitlines()
        if line.strip()
    ]
    if dataset.tile_names != expected_names:
        raise ValueError(f"Dataset order differs from sealed {split} split")

    support: Counter[int] = Counter()
    valid_frame_counts: Counter[int] = Counter()
    manifest_items = {
        item.get("name"): item
        for item in manifest.get("splits", {}).get(split, [])
        if isinstance(item, dict)
    }
    if set(manifest_items) != set(expected_names):
        raise ValueError(f"Cohort lacks exact {split} location metadata")
    location_records: list[dict[str, float | str]] = []
    for index, expected_name in enumerate(expected_names):
        sample = dataset[index]
        if sample.get("metadata", {}).get("tile") != expected_name:
            raise ValueError(
                f"Dataset substituted {sample.get('metadata')} for {expected_name}"
            )
        spectral = sample["spectral"]
        label = sample["label"]
        expected_shape = (
            int(manifest["model_patch_px"]),
            int(manifest["model_patch_px"]),
        )
        if tuple(spectral.shape) != (24, *expected_shape):
            raise ValueError(
                f"Unexpected spectral shape for {expected_name}: {tuple(spectral.shape)}"
            )
        if tuple(label.shape) != expected_shape:
            raise ValueError(
                f"Unexpected label shape for {expected_name}: {tuple(label.shape)}"
            )
        if not torch.isfinite(spectral).all():
            raise ValueError(f"Non-finite spectral values in {expected_name}")
        if int(label.min()) < 0 or int(label.max()) > 22:
            raise ValueError(f"Label range outside 0..22 in {expected_name}")
        for channel in config.enabled_aux_names:
            values = sample.get(channel)
            if values is None or tuple(values.shape) != (1, *expected_shape):
                raise ValueError(
                    f"Missing/wrong AUX channel {channel} in {expected_name}"
                )
            if not torch.isfinite(values).all():
                raise ValueError(
                    f"Non-finite AUX channel {channel} in {expected_name}"
                )
        temporal = sample.get("temporal_coords")
        location = sample.get("location_coords")
        mask = sample.get("temporal_mask")
        if (
            temporal is None or tuple(temporal.shape) != (4, 2)
            or location is None or tuple(location.shape) != (2,)
            or mask is None or tuple(mask.shape) != (4,)
            or not torch.isfinite(temporal).all()
            or not torch.isfinite(location).all()
        ):
            raise ValueError(f"Invalid Prithvi coordinates/mask in {expected_name}")
        location_records.append(_attest_location(
            sample=sample,
            manifest_item=manifest_items[expected_name],
            tile_name=expected_name,
        ))
        valid_frames = int(mask.to(torch.int64).sum().item())
        if valid_frames < 3 or int(mask[0].item()) != 1:
            raise ValueError(
                f"Invalid temporal coverage in {expected_name}: "
                f"mask={mask.tolist()}"
            )
        valid_frame_counts[valid_frames] += 1
        labels, counts = torch.unique(label, return_counts=True)
        support.update({
            int(raw_label): int(count)
            for raw_label, count in zip(labels.tolist(), counts.tolist())
        })

    expected_support = _expected_support(manifest, split)
    actual_support = dict(sorted(support.items()))
    if actual_support != expected_support:
        raise ValueError(
            f"Actual {split} model-label support differs from cohort: "
            f"actual={actual_support}, expected={expected_support}"
        )
    location_sha256 = hashlib.sha256(json.dumps(
        location_records,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()).hexdigest()
    return {
        "tiles": len(dataset),
        "pixels": sum(actual_support.values()),
        "pixel_counts": {str(key): value for key, value in actual_support.items()},
        "valid_frame_counts": {
            str(key): value for key, value in sorted(valid_frame_counts.items())
        },
        "minimum_valid_frames": min(valid_frame_counts),
        "autumn_frame_required": True,
        "location_attestation": {
            "source": "sealed_cohort_manifest_v4",
            "verified_tiles": len(location_records),
            "sha256": location_sha256,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-dir", required=True, type=Path)
    parser.add_argument("--cohort-dir", required=True, type=Path)
    parser.add_argument(
        "--era5-mode", required=True, choices=("control", "treatment"),
    )
    parser.add_argument("--era5-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.era5_mode == "treatment" and args.era5_dir is None:
        parser.error("treatment requires --era5-dir")
    if args.era5_mode == "control" and args.era5_dir is not None:
        parser.error("control must not receive --era5-dir")

    manifest = json.loads((args.cohort_dir / "manifest.json").read_text())
    if manifest.get("schema") != "era5-smoke-cohort-v4":
        raise ValueError("Unexpected cohort schema")
    result = {
        "schema": "era5-smoke-data-preflight-v1",
        "era5_mode": args.era5_mode,
        "splits": {
            split: validate_split(
                tile_dir=args.tile_dir,
                cohort_dir=args.cohort_dir,
                manifest=manifest,
                split=split,
                era5_mode=args.era5_mode,
                era5_dir=args.era5_dir,
            )
            for split in ("train", "val")
        },
    }
    _atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
