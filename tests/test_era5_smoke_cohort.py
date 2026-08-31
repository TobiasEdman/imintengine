"""Regression tests for persisted ERA5 smoke cohort guards."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from imint.training.unified_dataset import UnifiedDataset

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_era5_smoke_cohort.py"
_SPEC = importlib.util.spec_from_file_location("build_era5_smoke_cohort", _SCRIPT)
assert _SPEC and _SPEC.loader
cohort = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cohort)


def _write_tile(
    path: Path,
    index: int,
    *,
    max_label: int | None = None,
    center: tuple[int, int] | None = None,
) -> None:
    label = np.zeros((512, 512), dtype=np.uint8)
    if max_label is not None:
        label[:, 256:] = max_label
    else:
        for offset, crop_class in enumerate(range(11, 16)):
            top = 20 + offset * 90
            label[top:top + 80, 20:492] = crop_class
    center_east, center_north = center or (
        300_000 + index * 20_000,
        6_200_000 + index * 20_000,
    )
    np.savez_compressed(
        path,
        label=label,
        year=np.int32(2022),
        bbox_3006=np.array([
            center_east - 2_560,
            center_north - 2_560,
            center_east + 2_560,
            center_north + 2_560,
        ]),
        tile_size_px=np.int32(512),
        dates=np.asarray([
            "2021-09-20", "2022-05-10", "2022-07-01", "2022-08-12",
        ]),
        temporal_mask=np.ones(4, dtype=np.uint8),
        doy=np.asarray([263, 130, 182, 224], dtype=np.int32),
    )


def _build(tmp_path: Path) -> tuple[Path, Path, dict]:
    tile_dir = tmp_path / "tiles"
    split_dir = tmp_path / "cohort"
    tile_dir.mkdir()
    for index in range(15):
        _write_tile(tile_dir / f"tile_{index:02d}.npz", index)
    manifest, train_text, val_text = cohort.build_cohort(
        tile_dir=tile_dir,
        train_tiles=4,
        val_tiles=2,
        seed=17,
        min_crop_train_tiles=1,
        min_crop_val_tiles=1,
        min_crop_train_pixels=1,
        min_crop_val_pixels=1,
    )
    cohort.write_cohort(split_dir, manifest, train_text, val_text)
    return tile_dir, split_dir, manifest


def test_build_persists_disjoint_counts_schema_and_support(tmp_path):
    tile_dir, split_dir, manifest = _build(tmp_path)

    assert manifest["schema"] == "era5-smoke-cohort-v4"
    assert manifest["counts"] == {"train": 4, "val": 2}
    assert manifest["label_schema"] == {
        "num_logits": 23, "min_label": 0, "max_label": 22,
    }
    assert manifest["min_val_crop_classes"] == 5
    assert manifest["crop_support_thresholds"] == {
        "min_train_tiles": 1, "min_val_tiles": 1,
        "min_train_pixels": 1, "min_val_pixels": 1,
    }
    assert manifest["val_supported_crop_classes"] == [
        "vete", "korn", "havre", "oljeväxter", "slåttervall",
    ]
    assert not (
        set(manifest["era5_cells"]["train"])
        & set(manifest["era5_cells"]["val"])
    )
    spatial = manifest["spatial_partition"]
    assert spatial["crs"] == "EPSG:3006"
    assert spatial["bbox_intersection"] == "positive_area"
    assert spatial["spatial_components"]["overlap"] == []
    assert spatial["split_components"]["overlap"] == []
    assert spatial["bbox_area_intersections"] == []
    for split in ("train", "val"):
        support = manifest["label_support"][split]
        assert support["foreground_labels"]
        assert support["max_label"] <= 22
        assert (
            sum(support["pixel_counts"].values())
            == manifest["counts"][split] * 504 * 504
        )
        assert all(
            len(item["source_npz_sha256"]) == 64
            for item in manifest["splits"][split]
        )
        for item in manifest["splits"][split]:
            assert set(item["bbox_3006"]) == {"west", "south", "east", "north"}
            assert set(item["location_3006"]) == {"easting", "northing"}
            assert set(item["location_wgs84"]) == {"lat", "lon"}
            assert item["spatial_component"].startswith("spatial:")
            assert item["split_component"].startswith("split:")

    assert all(
        cohort._bbox_intersection_area(
            train_item["bbox_3006"], val_item["bbox_3006"],
        ) == 0
        for train_item in manifest["splits"]["train"]
        for val_item in manifest["splits"]["val"]
    )

    validated = cohort.validate_existing_cohort(
        tile_dir=tile_dir, output_dir=split_dir,
    )
    assert validated == json.loads(json.dumps(manifest))
    train_dataset = UnifiedDataset(
        lulc_dir=tile_dir, split="train", split_dir=split_dir,
    )
    val_dataset = UnifiedDataset(
        lulc_dir=tile_dir, split="val", split_dir=split_dir,
    )
    train_text_names = [
        line for line in (split_dir / "split_train.txt").read_text().splitlines()
        if line
    ]
    assert train_dataset.tile_names == train_text_names
    assert val_dataset.tile_names == [
        line for line in (split_dir / "split_val.txt").read_text().splitlines()
        if line
    ]
    assert len(train_text_names) == 4
    assert len(val_dataset) == 2


def test_explicit_split_dir_requires_both_splits_and_manifest(tmp_path):
    tile_dir = tmp_path / "tiles"
    split_dir = tmp_path / "cohort"
    tile_dir.mkdir()
    split_dir.mkdir()
    _write_tile(tile_dir / "tile_00.npz", 0)
    (split_dir / "split_train.txt").write_text("tile_00.npz\n")

    with pytest.raises(FileNotFoundError, match="requires split_train.txt, split_val.txt"):
        UnifiedDataset(lulc_dir=tile_dir, split="train", split_dir=split_dir)


def test_validate_existing_cli_is_read_only_and_reports_contract(tmp_path):
    tile_dir, split_dir, _ = _build(tmp_path)
    before = {path.name: path.read_bytes() for path in split_dir.iterdir()}

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--tile-dir", str(tile_dir),
            "--output-dir", str(split_dir),
            "--validate-existing",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "status": "valid",
        "schema": "era5-smoke-cohort-v4",
        "counts": {"train": 4, "val": 2},
        "label_schema": {"num_logits": 23, "min_label": 0, "max_label": 22},
        "era5_cell_overlap": 0,
        "min_val_crop_classes": 5,
        "val_supported_crop_classes": [
            "vete", "korn", "havre", "oljeväxter", "slåttervall",
        ],
    }
    assert {path.name: path.read_bytes() for path in split_dir.iterdir()} == before


def test_validator_rejects_partial_or_modified_split(tmp_path):
    tile_dir, split_dir, _ = _build(tmp_path)
    with (split_dir / "split_train.txt").open("a") as handle:
        handle.write("missing.npz\n")

    with pytest.raises(ValueError, match="hash mismatch"):
        cohort.validate_existing_cohort(tile_dir=tile_dir, output_dir=split_dir)
    with pytest.raises(ValueError, match="hash mismatch"):
        UnifiedDataset(lulc_dir=tile_dir, split="train", split_dir=split_dir)


def test_validator_rejects_stale_label_content(tmp_path):
    tile_dir, split_dir, manifest = _build(tmp_path)
    name = manifest["splits"]["train"][0]["name"]
    _write_tile(tile_dir / name, 0, max_label=9)

    with pytest.raises(ValueError, match="metadata is stale"):
        cohort.validate_existing_cohort(tile_dir=tile_dir, output_dir=split_dir)
    with pytest.raises(ValueError, match="label is stale"):
        UnifiedDataset(lulc_dir=tile_dir, split="train", split_dir=split_dir)


def test_validator_rejects_changed_non_label_source_input(tmp_path):
    tile_dir, split_dir, manifest = _build(tmp_path)
    name = manifest["splits"]["train"][0]["name"]
    path = tile_dir / name
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    values["dates"] = np.asarray([
        "2021-09-20", "2022-05-10", "2022-07-01", "2022-08-13",
    ])
    values["doy"] = np.asarray([263, 130, 182, 225], dtype=np.int32)
    np.savez_compressed(path, **values)

    with pytest.raises(ValueError, match="metadata is stale"):
        cohort.validate_existing_cohort(tile_dir=tile_dir, output_dir=split_dir)
    with pytest.raises(ValueError, match="source NPZ is stale"):
        UnifiedDataset(lulc_dir=tile_dir, split="train", split_dir=split_dir)


def test_builder_excludes_labels_outside_23_logit_schema(tmp_path):
    tile = tmp_path / "bad.npz"
    _write_tile(tile, 0, max_label=23)

    with pytest.raises(ValueError, match=r"expected 0\.\.22"):
        cohort._tile_metadata(
            tile, label_dir=None, manifest_path=None, max_label=22,
        )


def test_builder_excludes_insufficient_or_invalid_temporal_coverage(tmp_path):
    tile = tmp_path / "tile_00.npz"
    _write_tile(tile, 0)
    with np.load(tile, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    values["temporal_mask"] = np.asarray([1, 1, 0, 0], dtype=np.uint8)
    np.savez_compressed(tile, **values)
    with pytest.raises(ValueError, match="requires at least 3/4"):
        cohort._tile_metadata(
            tile, label_dir=None, manifest_path=None, max_label=22,
        )

    values["temporal_mask"] = np.ones(4, dtype=np.uint8)
    values["dates"] = np.asarray([
        "2021-07-01", "2022-05-10", "2022-07-01", "2022-08-12",
    ])
    values["doy"] = np.asarray([182, 130, 182, 224], dtype=np.int32)
    np.savez_compressed(tile, **values)
    with pytest.raises(ValueError, match="autumn frame is outside"):
        cohort._tile_metadata(
            tile, label_dir=None, manifest_path=None, max_label=22,
        )


def test_tile_metadata_never_reads_spectral_payload(tmp_path, monkeypatch):
    tile = tmp_path / "tile.npz"
    _write_tile(tile, 0)
    real_load = np.load

    class GuardedArchive:
        def __init__(self, archive):
            self.archive = archive
            self.files = archive.files

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.archive.close()

        def get(self, key, default=None):
            if key in {"spectral", "image"}:
                raise AssertionError("cohort scan must not decompress image payloads")
            return self.archive.get(key, default)

        def __getitem__(self, key):
            if key in {"spectral", "image"}:
                raise AssertionError("cohort scan must not decompress image payloads")
            return self.archive[key]

    monkeypatch.setattr(
        cohort.np,
        "load",
        lambda *args, **kwargs: GuardedArchive(real_load(*args, **kwargs)),
    )

    metadata = cohort._tile_metadata(
        tile, label_dir=None, manifest_path=None, max_label=22,
    )
    assert metadata["label"]["max"] == 15


def test_builder_requires_five_validation_crop_classes(tmp_path):
    tile_dir = tmp_path / "tiles"
    tile_dir.mkdir()
    for index in range(15):
        _write_tile(tile_dir / f"tile_{index:02d}.npz", index, max_label=11)

    with pytest.raises(ValueError, match="insufficient crop-class support"):
        cohort.build_cohort(
            tile_dir=tile_dir,
            train_tiles=4,
            val_tiles=2,
            seed=17,
            min_crop_train_tiles=1,
            min_crop_val_tiles=1,
            min_crop_train_pixels=1,
            min_crop_val_pixels=1,
        )


def test_validator_rejects_stale_supported_crop_class_list(tmp_path):
    tile_dir, split_dir, manifest = _build(tmp_path)
    manifest["val_supported_crop_classes"] = ["vete"]
    (split_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    with pytest.raises(ValueError, match="does not match validation label support"):
        cohort.validate_existing_cohort(tile_dir=tile_dir, output_dir=split_dir)
    with pytest.raises(ValueError, match="does not match validation label support"):
        UnifiedDataset(lulc_dir=tile_dir, split="train", split_dir=split_dir)


def test_validator_rejects_manifest_cell_overlap(tmp_path):
    tile_dir, split_dir, manifest = _build(tmp_path)
    manifest["splits"]["val"][0]["era5_cell"] = manifest["splits"]["train"][0]["era5_cell"]
    manifest["era5_cells"]["val"] = sorted({
        item["era5_cell"] for item in manifest["splits"]["val"]
    })
    (split_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    with pytest.raises(ValueError, match="ERA5 cells overlap"):
        cohort.validate_existing_cohort(tile_dir=tile_dir, output_dir=split_dir)
    with pytest.raises(ValueError, match="ERA5 train/val cells overlap"):
        UnifiedDataset(lulc_dir=tile_dir, split="train", split_dir=split_dir)


def test_cross_cell_overlap_and_cell_peers_form_one_split_component(tmp_path):
    """A 5.12 km footprint crossing an ERA5 boundary must not leak."""
    tile_dir = tmp_path / "tiles"
    split_dir = tmp_path / "cohort"
    tile_dir.mkdir()

    # First two footprints overlap by about 2.8 km but their centroids select
    # the adjacent +14.00 and +14.25 ERA5 atmosphere cells. The other two do
    # not overlap them; shared cells must nevertheless pull all four into one
    # split component through the component/cell transitive closure.
    bridge_centers = {
        "bridge_w.npz": (450_020, 6_540_380),
        "bridge_e.npz": (452_320, 6_540_350),
        "west_cell_peer.npz": (450_020, 6_548_380),
        "east_cell_peer.npz": (452_320, 6_532_350),
    }
    for index, (name, center) in enumerate(bridge_centers.items()):
        _write_tile(tile_dir / name, index, center=center)
    ordinary_centers = [
        (300_000, 6_200_000),
        (340_000, 6_240_000),
        (380_000, 6_280_000),
        (420_000, 6_320_000),
        (520_000, 6_400_000),
        (600_000, 6_500_000),
    ]
    for index, center in enumerate(ordinary_centers, start=4):
        _write_tile(tile_dir / f"ordinary_{index}.npz", index, center=center)

    raw_bridge = {
        name: cohort._tile_metadata(
            tile_dir / name,
            label_dir=None,
            manifest_path=None,
            max_label=22,
        )
        for name in bridge_centers
    }
    assert raw_bridge["bridge_w.npz"]["era5_cell"] != raw_bridge["bridge_e.npz"][
        "era5_cell"
    ]
    assert raw_bridge["bridge_w.npz"]["era5_cell"] == raw_bridge[
        "west_cell_peer.npz"
    ]["era5_cell"]
    assert raw_bridge["bridge_e.npz"]["era5_cell"] == raw_bridge[
        "east_cell_peer.npz"
    ]["era5_cell"]
    assert cohort._bbox_intersection_area(
        raw_bridge["bridge_w.npz"]["bbox_3006"],
        raw_bridge["bridge_e.npz"]["bbox_3006"],
    ) > 0

    build_kwargs = {
        "tile_dir": tile_dir,
        "train_tiles": 6,
        "val_tiles": 4,
        "seed": 29,
        "min_crop_train_tiles": 1,
        "min_crop_val_tiles": 1,
        "min_crop_train_pixels": 1,
        "min_crop_val_pixels": 1,
    }
    first = cohort.build_cohort(**build_kwargs)
    second = cohort.build_cohort(**build_kwargs)
    assert first == second
    manifest, train_text, val_text = first
    cohort.write_cohort(split_dir, manifest, train_text, val_text)

    split_by_name = {
        item["name"]: split
        for split in ("train", "val")
        for item in manifest["splits"][split]
    }
    assert set(split_by_name) == {path.name for path in tile_dir.glob("*.npz")}
    assert len({split_by_name[name] for name in bridge_centers}) == 1

    selected = {
        item["name"]: item
        for split in ("train", "val")
        for item in manifest["splits"][split]
    }
    assert selected["bridge_w.npz"]["spatial_component"] == selected[
        "bridge_e.npz"
    ]["spatial_component"]
    assert len({selected[name]["split_component"] for name in bridge_centers}) == 1
    assert manifest["counts"] == {"train": 6, "val": 4}
    assert manifest["spatial_partition"]["bbox_area_intersections"] == []
    cohort.validate_existing_cohort(tile_dir=tile_dir, output_dir=split_dir)


def test_bbox_components_are_transitive_but_edge_contact_has_zero_area():
    items = [
        {
            "name": "a.npz",
            "bbox_3006": {"west": 0, "south": 0, "east": 10, "north": 10},
            "era5_cell": "cell-a",
        },
        {
            "name": "b.npz",
            "bbox_3006": {"west": 9, "south": 0, "east": 19, "north": 10},
            "era5_cell": "cell-b",
        },
        {
            "name": "c.npz",
            "bbox_3006": {"west": 18, "south": 0, "east": 28, "north": 10},
            "era5_cell": "cell-c",
        },
        {
            "name": "touch.npz",
            "bbox_3006": {"west": 28, "south": 0, "east": 38, "north": 10},
            "era5_cell": "cell-touch",
        },
    ]

    cohort._bind_components(items)

    assert len({item["spatial_component"] for item in items[:3]}) == 1
    assert items[2]["spatial_component"] != items[3]["spatial_component"]
    assert cohort._bbox_intersection_area(
        items[2]["bbox_3006"], items[3]["bbox_3006"],
    ) == 0


def test_exact_counts_fail_loud_when_one_component_cannot_be_split():
    only_component = {
        "split:only": [{"name": f"tile_{index}.npz"} for index in range(6)]
    }

    with pytest.raises(ValueError, match="component partition"):
        cohort._validation_component_ids(
            only_component,
            train_tiles=4,
            val_tiles=2,
            seed=17,
        )


def test_component_partition_preserves_seeded_256_128_contract():
    split_components = {
        f"split:{index:064x}": [{"name": f"tile_{index:03d}.npz"}]
        for index in range(384)
    }

    first = cohort._validation_component_ids(
        split_components,
        train_tiles=256,
        val_tiles=128,
        seed=20260821,
    )
    second = cohort._validation_component_ids(
        split_components,
        train_tiles=256,
        val_tiles=128,
        seed=20260821,
    )

    assert first == second
    assert len(first) == 128
    assert sum(
        len(items)
        for component_id, items in split_components.items()
        if component_id not in first
    ) == 256


def test_validator_rejects_tampered_component_identity(tmp_path):
    tile_dir, split_dir, manifest = _build(tmp_path)
    manifest["splits"]["train"][0]["spatial_component"] = "spatial:" + "0" * 64
    (split_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    with pytest.raises(ValueError, match="metadata is stale"):
        cohort.validate_existing_cohort(tile_dir=tile_dir, output_dir=split_dir)


def test_external_bbox_manifest_is_byte_sealed(tmp_path):
    tile_dir = tmp_path / "tiles"
    split_dir = tmp_path / "cohort"
    tile_dir.mkdir()
    for index in range(6):
        name = "legacy.npz" if index == 0 else f"ordinary_{index}.npz"
        _write_tile(tile_dir / name, index)

    legacy_path = tile_dir / "legacy.npz"
    with np.load(legacy_path, allow_pickle=False) as archive:
        legacy_values = {
            key: archive[key] for key in archive.files if key != "bbox_3006"
        }
    np.savez_compressed(legacy_path, **legacy_values)
    original_bbox = {
        "west": 297_440,
        "south": 6_197_440,
        "east": 302_560,
        "north": 6_202_560,
    }
    bbox_manifest_path = tmp_path / "locations.json"
    bbox_manifest_path.write_text(json.dumps([
        {"name": "legacy", "bbox_3006": original_bbox},
    ]))

    manifest, train_text, val_text = cohort.build_cohort(
        tile_dir=tile_dir,
        train_tiles=4,
        val_tiles=2,
        seed=17,
        min_crop_train_tiles=1,
        min_crop_val_tiles=1,
        min_crop_train_pixels=1,
        min_crop_val_pixels=1,
        manifest_path=str(bbox_manifest_path),
    )
    cohort.write_cohort(split_dir, manifest, train_text, val_text)
    assert len(manifest["source"]["bbox_manifest_sha256"]) == 64

    moved_bbox = dict(original_bbox)
    moved_bbox["west"] += 100
    moved_bbox["east"] += 100
    bbox_manifest_path.write_text(json.dumps([
        {"name": "legacy", "bbox_3006": moved_bbox},
    ]))

    with pytest.raises(ValueError, match="bbox manifest is stale"):
        cohort.validate_existing_cohort(
            tile_dir=tile_dir,
            output_dir=split_dir,
            manifest_path=str(bbox_manifest_path),
        )
