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


class TestPreCreatedBaseline:
    """Verify change detection works with pre-created cloud-free baselines."""

    def test_detects_changes_against_prefetched_baseline(self, tmp_output_dir, coords):
        """When a baseline already exists (from ensure_baseline), compare against it."""
        analyzer = ChangeDetectionAnalyzer(config={"threshold": 0.1, "min_region_pixels": 5})

        # Simulate a pre-fetched cloud-free baseline
        baseline_dir = os.path.join(tmp_output_dir, "..", "baselines")
        os.makedirs(baseline_dir, exist_ok=True)
        season = "summer"  # 2022-06-15 is summer
        area = f"{coords['west']}_{coords['south']}_{coords['east']}_{coords['north']}"
        baseline_path = os.path.join(baseline_dir, f"{season}_{area}.npy")

        baseline = np.zeros((64, 64, 3), dtype=np.float32)
        np.save(baseline_path, baseline)

        # Run analyzer with bright image — should detect changes
        bright = np.full((64, 64, 3), 0.9, dtype=np.float32)
        result = analyzer.run(bright, date="2022-06-15", coords=coords, output_dir=tmp_output_dir)

        assert result.success
        assert result.outputs["change_fraction"] > 0.5
        assert "baseline_saved" not in result.metadata

    def test_no_change_against_identical_prefetched_baseline(self, tmp_output_dir, coords):
        """Pre-created baseline identical to current image -> zero change."""
        analyzer = ChangeDetectionAnalyzer(config={"threshold": 0.15})

        baseline_dir = os.path.join(tmp_output_dir, "..", "baselines")
        os.makedirs(baseline_dir, exist_ok=True)
        season = "summer"
        area = f"{coords['west']}_{coords['south']}_{coords['east']}_{coords['north']}"
        baseline_path = os.path.join(baseline_dir, f"{season}_{area}.npy")

        img = np.full((64, 64, 3), 0.5, dtype=np.float32)
        np.save(baseline_path, img)

        result = analyzer.run(img, date="2022-06-15", coords=coords, output_dir=tmp_output_dir)

        assert result.success
        assert result.outputs["change_fraction"] == 0.0


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


class TestCloudMasking:
    """Verify SCL-based cloud masking in change detection."""

    def test_cloudy_pixels_excluded_from_change(self, tmp_output_dir, coords):
        """Pixels cloudy in current SCL should not count as changed."""
        analyzer = ChangeDetectionAnalyzer(config={"threshold": 0.1, "min_region_pixels": 1})

        # Baseline: dark image
        img1 = np.zeros((64, 64, 3), dtype=np.float32)
        analyzer.run(img1, date="2022-06-15", coords=coords, output_dir=tmp_output_dir)

        # Current: bright everywhere, but top half is cloud (SCL=9)
        img2 = np.ones((64, 64, 3), dtype=np.float32)
        scl = np.zeros((64, 64), dtype=np.uint8)
        scl[:32, :] = 9  # top half = cloud_high_probability

        result = analyzer.run(img2, date="2022-06-15", coords=coords,
                              output_dir=tmp_output_dir, scl=scl)

        assert result.success
        # Only bottom half (non-cloud) should be detected as changed
        mask = result.outputs["change_mask"]
        assert mask[:32, :].sum() == 0, "Cloudy pixels should not be flagged"
        assert mask[32:, :].sum() > 0, "Non-cloudy changed pixels should be flagged"
        assert result.metadata["cloud_masked_pixels"] == 32 * 64
        assert result.metadata["valid_pixels"] == 32 * 64

    def test_baseline_scl_also_masks(self, tmp_output_dir, coords):
        """Pixels cloudy in baseline SCL should also be excluded."""
        analyzer = ChangeDetectionAnalyzer(config={"threshold": 0.1, "min_region_pixels": 1})

        # Create baseline + baseline SCL with clouds in bottom half
        baseline_dir = os.path.join(tmp_output_dir, "..", "baselines")
        os.makedirs(baseline_dir, exist_ok=True)
        area = f"{coords['west']}_{coords['south']}_{coords['east']}_{coords['north']}"
        baseline_path = os.path.join(baseline_dir, f"summer_{area}.npy")
        scl_path = baseline_path.replace(".npy", "_scl.npy")

        baseline = np.zeros((64, 64, 3), dtype=np.float32)
        np.save(baseline_path, baseline)
        baseline_scl = np.zeros((64, 64), dtype=np.uint8)
        baseline_scl[32:, :] = 8  # bottom half = cloud_medium_probability
        np.save(scl_path, baseline_scl)

        # Current: bright, cloud in top half
        img2 = np.ones((64, 64, 3), dtype=np.float32)
        current_scl = np.zeros((64, 64), dtype=np.uint8)
        current_scl[:32, :] = 10  # top half = thin_cirrus

        result = analyzer.run(img2, date="2022-06-15", coords=coords,
                              output_dir=tmp_output_dir, scl=current_scl)

        assert result.success
        # Both halves are cloudy (one in current, one in baseline) -> no valid changes
        mask = result.outputs["change_mask"]
        assert mask.sum() == 0, "All pixels should be masked by combined cloud mask"
        assert result.metadata["cloud_masked_pixels"] == 64 * 64

    def test_no_scl_means_no_masking(self, tmp_output_dir, coords):
        """Without SCL, all pixels are compared (backward compatible)."""
        analyzer = ChangeDetectionAnalyzer(config={"threshold": 0.1, "min_region_pixels": 1})

        img1 = np.zeros((64, 64, 3), dtype=np.float32)
        analyzer.run(img1, date="2022-06-15", coords=coords, output_dir=tmp_output_dir)

        img2 = np.ones((64, 64, 3), dtype=np.float32)
        result = analyzer.run(img2, date="2022-06-15", coords=coords,
                              output_dir=tmp_output_dir, scl=None)

        assert result.success
        assert result.metadata["cloud_masked_pixels"] == 0
        assert result.metadata["valid_pixels"] == 64 * 64
        assert result.outputs["change_fraction"] > 0.9
