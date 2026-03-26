"""Tests for imint/training/crop_dataset.py and crop_schema.py."""
from __future__ import annotations

import os
import numpy as np
import pytest

from imint.training.crop_schema import (
    lucas_code_to_class,
    is_agricultural,
    CLASS_NAMES,
    NUM_CLASSES,
    LUCAS_TO_CROP,
    LUCAS_SUPPORTED_YEARS,
    load_lucas_sweden,
    summarize_lucas_sweden,
)


# ── Crop schema tests ────────────────────────────────────────────────────

class TestCropSchema:
    """Verify LUCAS → Swedish crop class mapping."""

    def test_num_classes(self):
        assert NUM_CLASSES == 8

    def test_class_names_count(self):
        assert len(CLASS_NAMES) == NUM_CLASSES

    def test_wheat_mapping(self):
        assert lucas_code_to_class("B11") == 1  # Common wheat
        assert lucas_code_to_class("B13") == 1  # Durum wheat

    def test_barley_mapping(self):
        assert lucas_code_to_class("B14") == 2

    def test_oats_mapping(self):
        assert lucas_code_to_class("B15") == 3

    def test_rapeseed_mapping(self):
        assert lucas_code_to_class("B32") == 4

    def test_ley_grass_mapping(self):
        for code in ["B51", "B52", "B53", "B54", "B55"]:
            assert lucas_code_to_class(code) == 5, f"{code} should be ley_grass"

    def test_potato_mapping(self):
        assert lucas_code_to_class("B21") == 6

    def test_other_crop_mapping(self):
        for code in ["B12", "B16", "B22", "B31"]:
            assert lucas_code_to_class(code) == 7, f"{code} should be other_crop"

    def test_non_crop_returns_zero(self):
        assert lucas_code_to_class("A11") == 0  # Forest
        assert lucas_code_to_class("C10") == 0  # Urban
        assert lucas_code_to_class("") == 0

    def test_is_agricultural(self):
        assert is_agricultural("B11")
        assert is_agricultural("B14")
        assert not is_agricultural("A11")
        assert not is_agricultural("")

    def test_supported_years(self):
        assert 2018 in LUCAS_SUPPORTED_YEARS
        assert 2022 in LUCAS_SUPPORTED_YEARS


# ── CSV loading tests ────────────────────────────────────────────────────

class TestLoadLucasSweden:
    """Test LUCAS CSV loading and filtering."""

    def _write_csv(self, path, rows):
        """Write a minimal LUCAS CSV."""
        import csv
        fieldnames = ["POINT_ID", "GPS_LAT", "GPS_LONG", "LC1", "NUTS0", "SURVEY_YEAR"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_filters_sweden(self, tmp_path):
        csv_path = str(tmp_path / "lucas.csv")
        self._write_csv(csv_path, [
            {"POINT_ID": "1", "GPS_LAT": "59.3", "GPS_LONG": "18.1",
             "LC1": "B11", "NUTS0": "SE", "SURVEY_YEAR": "2018"},
            {"POINT_ID": "2", "GPS_LAT": "52.5", "GPS_LONG": "13.4",
             "LC1": "B11", "NUTS0": "DE", "SURVEY_YEAR": "2018"},
        ])
        points = load_lucas_sweden(csv_path)
        assert len(points) == 1
        assert points[0]["nuts0"] == "SE"

    def test_crop_only_filter(self, tmp_path):
        csv_path = str(tmp_path / "lucas.csv")
        self._write_csv(csv_path, [
            {"POINT_ID": "1", "GPS_LAT": "59.3", "GPS_LONG": "18.1",
             "LC1": "B14", "NUTS0": "SE", "SURVEY_YEAR": "2018"},
            {"POINT_ID": "2", "GPS_LAT": "58.0", "GPS_LONG": "16.0",
             "LC1": "A11", "NUTS0": "SE", "SURVEY_YEAR": "2018"},
        ])
        points = load_lucas_sweden(csv_path, crop_only=True)
        assert len(points) == 1
        assert points[0]["crop_class"] == 2  # barley

    def test_multiple_csvs_dedup(self, tmp_path):
        """2022 should take precedence over 2018 for same point_id."""
        csv_2018 = str(tmp_path / "lucas_2018.csv")
        csv_2022 = str(tmp_path / "lucas_2022.csv")
        self._write_csv(csv_2018, [
            {"POINT_ID": "P1", "GPS_LAT": "59.3", "GPS_LONG": "18.1",
             "LC1": "B11", "NUTS0": "SE", "SURVEY_YEAR": "2018"},
        ])
        self._write_csv(csv_2022, [
            {"POINT_ID": "P1", "GPS_LAT": "59.3", "GPS_LONG": "18.1",
             "LC1": "B14", "NUTS0": "SE", "SURVEY_YEAR": "2022"},
        ])
        points = load_lucas_sweden([csv_2018, csv_2022])
        assert len(points) == 1
        assert points[0]["year"] == 2022
        assert points[0]["crop_class"] == 2  # barley (B14), not wheat (B11)

    def test_summarize(self, tmp_path):
        csv_path = str(tmp_path / "lucas.csv")
        self._write_csv(csv_path, [
            {"POINT_ID": "1", "GPS_LAT": "59.3", "GPS_LONG": "18.1",
             "LC1": "B11", "NUTS0": "SE", "SURVEY_YEAR": "2018"},
            {"POINT_ID": "2", "GPS_LAT": "58.0", "GPS_LONG": "16.0",
             "LC1": "B11", "NUTS0": "SE", "SURVEY_YEAR": "2018"},
            {"POINT_ID": "3", "GPS_LAT": "57.0", "GPS_LONG": "15.0",
             "LC1": "B14", "NUTS0": "SE", "SURVEY_YEAR": "2018"},
        ])
        points = load_lucas_sweden(csv_path)
        summary = summarize_lucas_sweden(points)
        assert summary["total"] == 3
        assert summary["per_class"]["wheat"] == 2
        assert summary["per_class"]["barley"] == 1


# ── CropDataset tests ────────────────────────────────────────────────────

class TestCropDataset:
    """Test CropDataset with synthetic tiles."""

    @pytest.fixture
    def tile_dir(self, tmp_path):
        """Create a directory with synthetic crop tiles."""
        for i in range(10):
            np.savez_compressed(
                tmp_path / f"tile_{i:03d}.npz",
                spectral=np.random.rand(18, 256, 256).astype(np.float32),
                label=np.uint8(i % NUM_CLASSES),
                seasons_valid=np.array([True, True, True]),
                lat=59.0 + i * 0.01,
                lon=18.0 + i * 0.01,
                point_id=f"P{i:04d}",
                bbox_3006=np.array([670000, 6580000, 672560, 6582560]),
            )
        return str(tmp_path)

    def test_loads_tiles(self, tile_dir):
        from imint.training.crop_dataset import CropDataset
        ds = CropDataset(tile_dir, split="train", patch_size=224)
        assert len(ds) > 0

    def test_output_shape(self, tile_dir):
        from imint.training.crop_dataset import CropDataset
        ds = CropDataset(tile_dir, split="train", patch_size=224)
        sample = ds[0]
        assert sample["image"].shape == (18, 224, 224)
        assert sample["label"].shape == ()
        assert sample["label"].dtype == torch.int64

    def test_label_range(self, tile_dir):
        from imint.training.crop_dataset import CropDataset
        ds = CropDataset(tile_dir, split="train", patch_size=224)
        for i in range(min(5, len(ds))):
            label = ds[i]["label"].item()
            assert 0 <= label < NUM_CLASSES

    def test_val_split(self, tile_dir):
        from imint.training.crop_dataset import CropDataset
        ds_train = CropDataset(tile_dir, split="train", val_fraction=0.3)
        ds_val = CropDataset(tile_dir, split="val", val_fraction=0.3)
        assert len(ds_train) + len(ds_val) == 10
        assert len(ds_val) >= 1

    def test_seasons_valid(self, tile_dir):
        from imint.training.crop_dataset import CropDataset
        ds = CropDataset(tile_dir, split="train", patch_size=224)
        sample = ds[0]
        assert "seasons_valid" in sample
        assert sample["seasons_valid"].shape == (3,)

    def test_weighted_sampler(self, tile_dir):
        from imint.training.crop_dataset import CropDataset, build_crop_sampler
        ds = CropDataset(tile_dir, split="train")
        sampler = build_crop_sampler(ds)
        assert len(sampler) == len(ds)


try:
    import torch
except ImportError:
    torch = None

pytestmark = pytest.mark.skipif(
    torch is None, reason="torch not installed"
)
