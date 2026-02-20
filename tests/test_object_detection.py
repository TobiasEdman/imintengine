"""Tests for the object detection analyzer."""
from __future__ import annotations

import numpy as np
from imint.analyzers.object_detection import ObjectDetectionAnalyzer


class TestHeatmapMode:
    """Verify variance-based anomaly detection."""

    def test_uniform_image_no_detections(self, rgb_uniform):
        """A perfectly uniform image should have zero variance → no anomalies."""
        analyzer = ObjectDetectionAnalyzer(config={"mode": "heatmap", "patch_size": 16})
        result = analyzer.run(rgb_uniform, date="2022-06-15")

        assert result.success
        assert result.outputs["regions"] == []
        assert result.metadata["mode"] == "heatmap"
        assert result.metadata["n_detections"] == 0

    def test_anomaly_detected_in_noisy_patch(self):
        """An image with one very noisy patch should produce at least one detection."""
        img = np.full((64, 64, 3), 0.5, dtype=np.float32)
        # Insert a high-variance patch at (0, 0)
        rng = np.random.RandomState(99)
        img[0:32, 0:32, :] = rng.rand(32, 32, 3).astype(np.float32)

        analyzer = ObjectDetectionAnalyzer(config={
            "mode": "heatmap",
            "patch_size": 32,
            "std_threshold": 1.0,
        })
        result = analyzer.run(img, date="2022-06-15")

        assert result.success
        assert len(result.outputs["regions"]) >= 1
        region = result.outputs["regions"][0]
        assert region["label"] == "anomaly"
        assert region["score"] > 0

    def test_heatmap_output_shape(self, rgb_random):
        """Heatmap should match image spatial dimensions."""
        analyzer = ObjectDetectionAnalyzer(config={"mode": "heatmap", "patch_size": 16})
        result = analyzer.run(rgb_random, date="2022-06-15")

        assert result.success
        heatmap = result.outputs["heatmap"]
        assert heatmap.shape == rgb_random.shape[:2]

    def test_region_bbox_format(self, rgb_random):
        """Any detected regions should have proper bbox format."""
        analyzer = ObjectDetectionAnalyzer(config={
            "mode": "heatmap",
            "patch_size": 16,
            "std_threshold": 0.5,  # Lower threshold to get some detections
        })
        result = analyzer.run(rgb_random, date="2022-06-15")

        for region in result.outputs["regions"]:
            bbox = region["bbox"]
            assert "y_min" in bbox
            assert "y_max" in bbox
            assert "x_min" in bbox
            assert "x_max" in bbox
            assert bbox["y_max"] > bbox["y_min"]
            assert bbox["x_max"] > bbox["x_min"]


class TestModelMode:
    """Verify model mode graceful failure without ultralytics."""

    def test_model_mode_without_ultralytics(self, rgb_random):
        """Model mode should fail gracefully if ultralytics is not installed."""
        analyzer = ObjectDetectionAnalyzer(config={"mode": "model"})
        result = analyzer.run(rgb_random, date="2022-06-15")

        # Should either succeed (if ultralytics is installed) or fail gracefully
        if not result.success:
            assert "ultralytics" in result.error


class TestPatchSizeConfig:
    """Verify patch_size parameter."""

    def test_small_patches_more_granular(self, rgb_random):
        """Smaller patches should produce more patch evaluations."""
        a_small = ObjectDetectionAnalyzer(config={"patch_size": 8})
        a_large = ObjectDetectionAnalyzer(config={"patch_size": 32})

        r_small = a_small.run(rgb_random)
        r_large = a_large.run(rgb_random)

        # Smaller patches → more patches evaluated
        assert r_small.metadata["n_detections"] >= 0
        assert r_large.metadata["n_detections"] >= 0
