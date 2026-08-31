"""Focused regressions for the sealed ERA5 smoke data preflight."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from pyproj import Transformer

from imint.training.config import TrainingConfig
from imint.training.era5_aux import (
    ERA5_ATMOSPHERE_GRID_DEGREES,
    era5_grid_context,
    format_era5_cell_id,
)
from imint.training.tile_bbox import _load_manifest
from imint.training.unified_dataset import (
    UnifiedDataset,
    _label_metadata,
    _support_from_manifest_items,
)
from scripts import preflight_era5_smoke_data as preflight


def _sample(tile_name: str) -> dict:
    shape = (2, 2)
    grid = era5_grid_context(59.3, 18.1)
    sample = {
        "metadata": {
            "tile": tile_name,
            "location_source": "sealed_cohort_manifest_v4",
            "location_bbox_west_3006": 0.0,
            "location_bbox_south_3006": 0.0,
            "location_bbox_east_3006": 100.0,
            "location_bbox_north_3006": 100.0,
            "location_easting_3006": 50.0,
            "location_northing_3006": 50.0,
            "location_lat": 59.3,
            "location_lon": 18.1,
            "era5_request_lat": grid["request_lat"],
            "era5_request_lon": grid["request_lon"],
        },
        "spectral": torch.ones(24, *shape),
        "label": torch.tensor([[0, 11], [11, 11]], dtype=torch.long),
        "temporal_coords": torch.zeros(4, 2),
        "location_coords": torch.tensor([59.3, 18.1]),
        "temporal_mask": torch.ones(4),
    }
    config = TrainingConfig(
        enable_markfukt_channel=True,
        enable_era5_channels=True,
        era5_mode="control",
    )
    sample.update({
        name: torch.zeros(1, *shape)
        for name in config.enabled_aux_names
    })
    return sample


def _install_dataset(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tile_names: list[str],
    samples: list[dict],
) -> None:
    class Dataset:
        def __init__(self, **kwargs):
            assert kwargs["augment_override"] is False
            assert kwargs["multitemporal"] is True
            assert kwargs["num_temporal_frames"] == 4
            self.tile_names = tile_names

        def __len__(self) -> int:
            return len(samples)

        def __getitem__(self, index: int) -> dict:
            return samples[index]

    monkeypatch.setattr(preflight, "UnifiedDataset", Dataset)


def _manifest(
    *,
    tile_names: list[str],
    pixel_counts: dict[str, int] | None = None,
) -> dict:
    grid = era5_grid_context(59.3, 18.1)
    cell_id = format_era5_cell_id(
        grid["atmosphere_cell"]["lat"], grid["atmosphere_cell"]["lon"],
    )
    return {
        "model_patch_px": 2,
        "label_support": {
            "train": {
                "pixel_counts": pixel_counts or {"0": 1, "11": 3},
            },
        },
        "splits": {
            "train": [
                {
                    "name": name,
                    "era5_cell": cell_id,
                    "bbox_3006": {
                        "west": 0,
                        "south": 0,
                        "east": 100,
                        "north": 100,
                    },
                    "location_3006": {"easting": 50, "northing": 50},
                    "location_wgs84": {"lat": 59.3, "lon": 18.1},
                    "era5_request": {
                        "lat": grid["request_lat"],
                        "lon": grid["request_lon"],
                    },
                    "era5_land_cell": grid["land_cell"],
                    "era5_atmosphere_cell": grid["atmosphere_cell"],
                }
                for name in tile_names
            ],
        },
    }


def test_preflight_reads_every_sealed_tile_and_recounts_exact_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort_dir = tmp_path / "cohort"
    cohort_dir.mkdir()
    (cohort_dir / "split_train.txt").write_text("first.npz\nsecond.npz\n")
    samples = [_sample("first.npz"), _sample("second.npz")]
    _install_dataset(
        monkeypatch,
        tile_names=["first.npz", "second.npz"],
        samples=samples,
    )

    result = preflight.validate_split(
        tile_dir=tmp_path / "tiles",
        cohort_dir=cohort_dir,
        manifest=_manifest(
            tile_names=["first.npz", "second.npz"],
            pixel_counts={"0": 2, "11": 6},
        ),
        split="train",
        era5_mode="control",
        era5_dir=None,
    )

    location_attestation = result.pop("location_attestation")
    assert location_attestation["source"] == "sealed_cohort_manifest_v4"
    assert location_attestation["verified_tiles"] == 2
    assert len(location_attestation["sha256"]) == 64
    assert result == {
        "tiles": 2,
        "pixels": 8,
        "pixel_counts": {"0": 2, "11": 6},
        "valid_frame_counts": {"4": 2},
        "minimum_valid_frames": 4,
        "autumn_frame_required": True,
    }


def test_preflight_rejects_dataset_substitution_even_when_shapes_are_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort_dir = tmp_path / "cohort"
    cohort_dir.mkdir()
    (cohort_dir / "split_train.txt").write_text("sealed.npz\n")
    _install_dataset(
        monkeypatch,
        tile_names=["sealed.npz"],
        samples=[_sample("substitute.npz")],
    )

    with pytest.raises(ValueError, match="Dataset substituted"):
        preflight.validate_split(
            tile_dir=tmp_path / "tiles",
            cohort_dir=cohort_dir,
            manifest=_manifest(tile_names=["sealed.npz"]),
            split="train",
            era5_mode="control",
            era5_dir=None,
        )


def test_preflight_rejects_support_drift_after_reading_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort_dir = tmp_path / "cohort"
    cohort_dir.mkdir()
    (cohort_dir / "split_train.txt").write_text("tile.npz\n")
    _install_dataset(
        monkeypatch,
        tile_names=["tile.npz"],
        samples=[_sample("tile.npz")],
    )

    with pytest.raises(ValueError, match="model-label support differs"):
        preflight.validate_split(
            tile_dir=tmp_path / "tiles",
            cohort_dir=cohort_dir,
            manifest=_manifest(
                tile_names=["tile.npz"],
                pixel_counts={"0": 2, "11": 2},
            ),
            split="train",
            era5_mode="control",
            era5_dir=None,
        )


@pytest.mark.parametrize(
    "mask",
    [
        torch.tensor([1, 1, 0, 0], dtype=torch.float32),
        torch.tensor([0, 1, 1, 1], dtype=torch.float32),
    ],
)
def test_preflight_rejects_insufficient_or_missing_autumn_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mask: torch.Tensor,
):
    cohort_dir = tmp_path / "cohort"
    cohort_dir.mkdir()
    (cohort_dir / "split_train.txt").write_text("tile.npz\n")
    sample = _sample("tile.npz")
    sample["temporal_mask"] = mask
    _install_dataset(
        monkeypatch,
        tile_names=["tile.npz"],
        samples=[sample],
    )

    with pytest.raises(ValueError, match="Invalid temporal coverage"):
        preflight.validate_split(
            tile_dir=tmp_path / "tiles",
            cohort_dir=cohort_dir,
            manifest=_manifest(tile_names=["tile.npz"]),
            split="train",
            era5_mode="control",
            era5_dir=None,
        )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bbox_only_tile(path: Path, *, easting: int, northing: int) -> np.ndarray:
    size = 504
    label = np.zeros((size, size), dtype=np.uint8)
    for offset, crop_class in enumerate(range(11, 16)):
        left = offset * 100
        label[:, left:left + 100] = crop_class
    np.savez_compressed(
        path,
        spectral=np.zeros((24, size, size), dtype=np.float16),
        label=label,
        year=np.int32(2022),
        bbox_3006=np.asarray([
            easting - size * 5,
            northing - size * 5,
            easting + size * 5,
            northing + size * 5,
        ], dtype=np.int32),
        tile_size_px=np.int32(size),
        dates=np.asarray([
            "2021-09-20", "2022-05-10", "2022-07-01", "2022-08-12",
        ]),
        temporal_mask=np.ones(4, dtype=np.uint8),
        doy=np.asarray([263, 130, 182, 224], dtype=np.int32),
    )
    return label


def _sealed_item(
    path: Path,
    label: np.ndarray,
    *,
    easting: int,
    northing: int,
) -> dict:
    lon, lat = Transformer.from_crs(
        "EPSG:3006", "EPSG:4326", always_xy=True,
    ).transform(easting, northing)
    grid = era5_grid_context(lat, lon)
    spatial_component = (
        "spatial:" + hashlib.sha256(f"spatial:{path.name}".encode()).hexdigest()
    )
    split_component = (
        "split:" + hashlib.sha256(f"split:{path.name}".encode()).hexdigest()
    )
    return {
        "name": path.name,
        "year": 2022,
        "cutoff_date": "2022-08-12",
        "bbox_3006": {
            "west": easting - 2_520,
            "south": northing - 2_520,
            "east": easting + 2_520,
            "north": northing + 2_520,
        },
        "location_3006": {
            "easting": easting,
            "northing": northing,
        },
        "location_wgs84": {"lat": lat, "lon": lon},
        "spatial_component": spatial_component,
        "split_component": split_component,
        "era5_cell": format_era5_cell_id(
            grid["atmosphere_cell"]["lat"],
            grid["atmosphere_cell"]["lon"],
        ),
        "era5_request": {
            "lat": grid["request_lat"],
            "lon": grid["request_lon"],
        },
        "era5_land_cell": grid["land_cell"],
        "era5_atmosphere_cell": grid["atmosphere_cell"],
        "label": _label_metadata(label),
        "model_label": _label_metadata(label),
        "source_npz_sha256": _sha256_file(path),
    }


def _write_sealed_bundle(tmp_path: Path) -> tuple[Path, Path, dict]:
    tile_dir = tmp_path / "tiles"
    cohort_dir = tmp_path / "cohort"
    tile_dir.mkdir()
    cohort_dir.mkdir()
    specs = {
        "train": ("sealed_train.npz", 400_000, 6_400_000),
        "val": ("sealed_val.npz", 700_000, 7_300_000),
    }
    items: dict[str, list[dict]] = {}
    split_text: dict[str, str] = {}
    for split, (name, easting, northing) in specs.items():
        path = tile_dir / name
        label = _write_bbox_only_tile(
            path, easting=easting, northing=northing,
        )
        items[split] = [
            _sealed_item(
                path, label, easting=easting, northing=northing,
            ),
        ]
        split_text[split] = f"{name}\n"
        (cohort_dir / f"split_{split}.txt").write_text(split_text[split])

    support = {
        split: _support_from_manifest_items(split_items)
        for split, split_items in items.items()
    }
    cell_sets = {
        split: sorted({item["era5_cell"] for item in split_items})
        for split, split_items in items.items()
    }
    assert set(cell_sets["train"]).isdisjoint(cell_sets["val"])
    all_items = [*items["train"], *items["val"]]
    spatial_catalog = sorted(({
        "id": item["spatial_component"],
        "tiles": [item["name"]],
        "bbox_3006": item["bbox_3006"],
        "era5_cells": [item["era5_cell"]],
        "split_component": item["split_component"],
    } for item in all_items), key=lambda entry: entry["id"])
    split_catalog = sorted(({
        "id": item["split_component"],
        "tiles": [item["name"]],
        "bbox_3006": item["bbox_3006"],
        "era5_cells": [item["era5_cell"]],
        "spatial_components": [item["spatial_component"]],
    } for item in all_items), key=lambda entry: entry["id"])
    manifest = {
        "schema": "era5-smoke-cohort-v4",
        "source": {
            "tile_dir": str(tile_dir),
            "label_dir": None,
            "bbox_manifest": None,
            "bbox_manifest_sha256": None,
        },
        "label_schema": {"num_logits": 23, "min_label": 0, "max_label": 22},
        "model_patch_px": 504,
        "counts": {"train": 1, "val": 1},
        "split_sha256": {
            split: hashlib.sha256(text.encode()).hexdigest()
            for split, text in split_text.items()
        },
        "era5_cells": {
            "train": cell_sets["train"],
            "val": cell_sets["val"],
            "overlap": [],
        },
        "era5_cell_definition": {
            "model": "era5",
            "resolution_degrees": float(ERA5_ATMOSPHERE_GRID_DEGREES),
            "cell_selection": "nearest",
            "identity": "verified_response_latitude_longitude",
        },
        "spatial_partition": {
            "crs": "EPSG:3006",
            "bbox_intersection": "positive_area",
            "component_definition": (
                "transitive_closure_of_positive_area_bbox_intersections"
            ),
            "split_component_definition": (
                "transitive_closure_of_spatial_components_and_shared_era5_cells"
            ),
            "spatial_components": {
                "train": [items["train"][0]["spatial_component"]],
                "val": [items["val"][0]["spatial_component"]],
                "overlap": [],
            },
            "split_components": {
                "train": [items["train"][0]["split_component"]],
                "val": [items["val"][0]["split_component"]],
                "overlap": [],
            },
            "spatial_component_catalog": spatial_catalog,
            "split_component_catalog": split_catalog,
            "bbox_area_intersections": [],
        },
        "label_support": support,
        "crop_support_thresholds": {
            "min_train_tiles": 1,
            "min_val_tiles": 1,
            "min_train_pixels": 1,
            "min_val_pixels": 1,
        },
        "min_val_crop_classes": 5,
        "val_supported_crop_classes": [
            "vete", "korn", "havre", "oljeväxter", "slåttervall",
        ],
        "splits": items,
    }
    (cohort_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return tile_dir, cohort_dir, manifest


def _sealed_dataset(tile_dir: Path, cohort_dir: Path) -> UnifiedDataset:
    return UnifiedDataset(
        lulc_dir=tile_dir,
        split="train",
        patch_size=504,
        enable_aux=False,
        augment_override=False,
        multitemporal=True,
        num_temporal_frames=4,
        era5_mode="control",
        split_dir=cohort_dir,
        backbone_family="prithvi",
    )


def test_real_sealed_dataset_uses_bbox_only_location_for_prithvi(tmp_path: Path):
    tile_dir, cohort_dir, manifest = _write_sealed_bundle(tmp_path)

    sample = _sealed_dataset(tile_dir, cohort_dir)[0]

    lon, lat = Transformer.from_crs(
        "EPSG:3006", "EPSG:4326", always_xy=True,
    ).transform(400_000, 6_400_000)
    np.testing.assert_allclose(
        sample["location_coords"].numpy(), [lat, lon], rtol=0.0, atol=1e-5,
    )
    assert sample["metadata"]["location_source"] == "sealed_cohort_manifest_v4"
    assert sample["metadata"]["location_easting_3006"] == 400_000
    assert sample["metadata"]["location_northing_3006"] == 6_400_000
    assert sample["metadata"]["era5_request_lat"] == (
        manifest["splits"]["train"][0]["era5_request"]["lat"]
    )
    assert sample["metadata"]["era5_request_lon"] == (
        manifest["splits"]["train"][0]["era5_request"]["lon"]
    )


def test_real_sealed_dataset_rejects_manifest_location_mismatch(tmp_path: Path):
    tile_dir, cohort_dir, manifest = _write_sealed_bundle(tmp_path)
    manifest["splits"]["train"][0]["era5_request"]["lat"] += 0.1
    (cohort_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    with pytest.raises(ValueError, match="location metadata mismatch"):
        _sealed_dataset(tile_dir, cohort_dir)


def test_real_sealed_dataset_rejects_missing_exact_location_not_default(
    tmp_path: Path,
):
    tile_dir, cohort_dir, manifest = _write_sealed_bundle(tmp_path)
    path = tile_dir / "sealed_train.npz"
    with np.load(path, allow_pickle=False) as archive:
        values = {
            name: archive[name]
            for name in archive.files
            if name != "bbox_3006"
        }
    np.savez_compressed(path, **values)
    manifest["splits"]["train"][0]["source_npz_sha256"] = _sha256_file(path)
    for field in ("bbox_3006", "location_3006", "location_wgs84"):
        del manifest["splits"]["train"][0][field]
    (cohort_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    with pytest.raises(ValueError, match="invalid exact bbox_3006"):
        _sealed_dataset(tile_dir, cohort_dir)


def test_real_sealed_dataset_rejects_external_location_drift_within_era5_cell(
    tmp_path: Path,
):
    tile_dir, cohort_dir, manifest = _write_sealed_bundle(tmp_path)
    path = tile_dir / "sealed_train.npz"
    with np.load(path, allow_pickle=False) as archive:
        values = {
            name: archive[name]
            for name in archive.files
            if name != "bbox_3006"
        }
    np.savez_compressed(path, **values)
    manifest["splits"]["train"][0]["source_npz_sha256"] = _sha256_file(path)
    external_path = tmp_path / "tile_locations.json"
    original_bbox = manifest["splits"]["train"][0]["bbox_3006"]
    external_path.write_text(json.dumps([{
        "name": "sealed_train",
        "bbox_3006": original_bbox,
    }]))
    manifest["source"]["bbox_manifest"] = str(external_path)
    manifest["source"]["bbox_manifest_sha256"] = _sha256_file(external_path)
    (cohort_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _load_manifest.cache_clear()
    _sealed_dataset(tile_dir, cohort_dir)

    shifted_bbox = {
        "west": original_bbox["west"] + 10,
        "south": original_bbox["south"],
        "east": original_bbox["east"] + 10,
        "north": original_bbox["north"],
    }
    external_path.write_text(json.dumps([{
        "name": "sealed_train",
        "bbox_3006": shifted_bbox,
    }]))
    _load_manifest.cache_clear()
    with pytest.raises(
        ValueError, match="bbox manifest is stale|location source drift",
    ):
        _sealed_dataset(tile_dir, cohort_dir)
