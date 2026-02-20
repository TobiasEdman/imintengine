"""Tests for the change detection analyzer."""
from __future__ import annotations

import os
import numpy as np
from imint.analyzers.change_detection import ChangeDetectionAnalyzer


class TestBaselineBehavior:
    """First run should save baseline, second run should detect changes."""

    def test_first_run_saves_baseline(self, rgb_uniform, tmp_output_dir, coords):
        """First run returns 0 changes and saves a baseline file."""
        analyzer = ChangeDetectionAnalyzer(config={"threshold": 0.15})
        result = analyzer.run(
            rgb_uniform, date="2022-06-15", coords=coords, output_dir=tmp_output_dir,
        )

        assert result.success
        assert result.outputs["change_fraction"] == 0.0
        assert result.outputs["n_regions"] == 0
        assert "baseline_saved" in result.metadata

        # Verify baseline file was actually written
        baseline_path = result.metadata["baseline_saved"]
        assert os.path.exists(baseline_path)

    def test_second_run_detects_changes(self, tmp_output_dir, coords):
        """Second run with different image should detect changed pixels."""
        analyzer = ChangeDetectionAnalyzer(config={"threshold": 0.1, "min_region_pixels": 5})

        # First run: uniform image
        img1 = np.full((64, 64, 3), 0.3, dtype=np.float32)
        result1 = analyzer.run(img1, date="2022-06-15", coords=coords, output_dir=tmp_output_dir)
        assert result1.outputs["change_fraction"] == 0.0

        # Second run: very different image
        img2 = np.full((64, 64, 3), 0.9, dtype=np.float32)
        result2 = analyzer.run(img2, date="2022-06-15", coords=coords, output_dir=tmp_output_dir)

        assert result2.success
        assert result2.outputs["change_fraction"] > 0.5, "Expected significant change"
        assert result2.outputs["change_mask"].any(), "Expected some changed pixels"

    def test_identical_images_no_change(self, rgb_uniform, tmp_output_dir, coords):
        """Two identical images should produce zero change."""
        analyzer = ChangeDetectionAnalyzer(config={"threshold": 0.15})

        # Run twice with identical image
        analyzer.run(rgb_uniform, date="2022-06-15", coords=coords, output_dir=tmp_output_dir)
        result = analyzer.run(rgb_uniform, date="2022-06-15", coords=coords, output_dir=tmp_output_dir)

        assert result.success
        assert result.outputs["change_fraction"] == 0.0
        assert result.outputs["n_regions"] == 0


class TestChangeRegions:
    """Verify connected region detection."""

    def test_localized_change_produces_region(self, tmp_output_dir, coords):
        """A localized bright patch should produce exactly one region."""
        analyzer = ChangeDetectionAnalyzer(config={"threshold": 0.1, "min_region_pixels": 5})

        # Baseline: dark image
        img1 = np.zeros((64, 64, 3), dtype=np.float32)
        analyzer.run(img1, date="2022-06-15", coords=coords, output_dir=tmp_output_dir)

        # Changed: bright 20x20 patch in the center
        img2 = np.zeros((64, 64, 3), dtype=np.float32)
        img2[22:42, 22:42, :] = 1.0
        result = analyzer.run(img2, date="2022-06-15", coords=coords, output_dir=tmp_output_dir)

        assert result.success
        assert result.outputs["n_regions"] >= 1
        region = result.outputs["regions"][0]
        assert "bbox" in region
        assert region["pixel_count"] > 0


class TestChangeThreshold:
    """Verify threshold parameter controls sensitivity."""

    def test_high_threshold_less_detection(self, tmp_output_dir, coords):
        """Higher threshold should detect fewer changes."""
        img1 = np.full((64, 64, 3), 0.5, dtype=np.float32)
        img2 = np.full((64, 64, 3), 0.7, dtype=np.float32)  # Modest change

        # Low threshold
        a_low = ChangeDetectionAnalyzer(config={"threshold": 0.1})
        a_low.run(img1, date="2022-06-15", coords=coords, output_dir=tmp_output_dir)
        r_low = a_low.run(img2, date="2022-06-15", coords=coords, output_dir=tmp_output_dir)

        # High threshold (need separate baseline dir)
        out2 = tmp_output_dir + "_high"
        os.makedirs(out2, exist_ok=True)
        a_high = ChangeDetectionAnalyzer(config={"threshold": 0.5})
        a_high.run(img1, date="2022-06-15", coords=coords, output_dir=out2)
        r_high = a_high.run(img2, date="2022-06-15", coords=coords, output_dir=out2)

        assert r_low.outputs["change_fraction"] >= r_high.outputs["change_fraction"]
