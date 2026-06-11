"""Tests for imint/training/crop_schema.py — the kept crop-class mappings.

Salvaged from the retired tests/test_crop_dataset.py (the CropDataset half went
away with the LUCAS crop-only pipeline). Covers the LUCAS→crop and SJV→crop
mappings + the LUCAS CSV reader that crop_schema still provides as validation
reference, plus the SJV_TO_CROP table relocated here from build_crop_dataset.
"""
from __future__ import annotations

from imint.training.crop_schema import (
    CLASS_NAMES,
    LUCAS_SUPPORTED_YEARS,
    LUCAS_TO_CROP,
    NUM_CLASSES,
    SJV_TO_CROP,
    is_agricultural,
    load_lucas_sweden,
    lucas_code_to_class,
    sjv_grodkod_to_class,
    summarize_lucas_sweden,
)


# ── LUCAS → crop class ───────────────────────────────────────────────────

class TestCropSchema:
    """Verify LUCAS → Swedish crop class mapping."""

    def test_num_classes(self):
        assert NUM_CLASSES == 9

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
        # ovrig_akergroda (class 8): rye (B12), triticale (B16), sugar beet (B22)
        for code in ["B12", "B16", "B22"]:
            assert lucas_code_to_class(code) == 8, f"{code} should be ovrig_akergroda"
        # peas / field beans (B31) are now their own class: trindsad (7)
        assert lucas_code_to_class("B31") == 7

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


# ── SJV grödkod → crop class (relocated from build_crop_dataset) ──────────

class TestSjvToCrop:
    """Verify the SJV grödkod → crop class table moved into crop_schema."""

    def test_representative_codes(self):
        assert sjv_grodkod_to_class(1) == 1    # Höstvete
        assert sjv_grodkod_to_class(3) == 2    # Höstkorn
        assert sjv_grodkod_to_class(5) == 3    # Havre
        assert sjv_grodkod_to_class(85) == 4   # Höstraps
        assert sjv_grodkod_to_class(50) == 5   # Slåtter-/betesvall
        assert sjv_grodkod_to_class(70) == 6   # Matpotatis
        assert sjv_grodkod_to_class(30) == 7   # Ärtor
        assert sjv_grodkod_to_class(22) == 8   # Sockerbetor

    def test_unknown_code_is_zero(self):
        assert sjv_grodkod_to_class(9999) == 0
        assert SJV_TO_CROP.get(9999, 0) == 0

    def test_classes_in_range(self):
        assert all(0 < v < NUM_CLASSES for v in SJV_TO_CROP.values())


# ── LUCAS CSV reader (validation reference) ──────────────────────────────

class TestLoadLucasSweden:
    """Test LUCAS CSV loading and filtering (synthetic CSVs — no real data)."""

    def _write_csv(self, path, rows):
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
        """2022 should take precedence over 2018 for the same point_id."""
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
        assert summary["per_class"]["vete"] == 2
        assert summary["per_class"]["korn"] == 1
