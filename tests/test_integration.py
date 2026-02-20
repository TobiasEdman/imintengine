"""
Integration test — runs the full engine pipeline end-to-end.

Verifies that run_job() produces valid output files in the correct formats.
"""
from __future__ import annotations

import os
import json
import numpy as np
from PIL import Image

from imint.job import IMINTJob
from imint.engine import run_job


def _make_job(tmp_path, date="2022-06-15"):
    """Create a fully populated IMINTJob with synthetic data."""
    rng = np.random.RandomState(42)
    h, w = 64, 64
    output_dir = str(tmp_path / "outputs" / date)
    os.makedirs(output_dir, exist_ok=True)

    return IMINTJob(
        date=date,
        coords={"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0},
        rgb=rng.rand(h, w, 3).astype(np.float32),
        bands={
            "B02": rng.rand(h, w).astype(np.float32),
            "B03": rng.rand(h, w).astype(np.float32),
            "B04": rng.rand(h, w).astype(np.float32),
            "B08": (rng.rand(h, w) + 0.3).astype(np.float32),
            "B11": rng.rand(h, w).astype(np.float32),
        },
        output_dir=output_dir,
        config_path="config/analyzers.yaml",
        job_id="test_integration",
    )


class TestRunJobEndToEnd:
    """Full pipeline integration tests."""

    def test_run_job_succeeds(self, tmp_path):
        """run_job() should complete without errors."""
        job = _make_job(tmp_path)
        result = run_job(job)

        assert result.success, f"run_job failed: {result.error}"
        assert result.job_id == "test_integration"
        assert result.date == "2022-06-15"
        assert len(result.analyzer_results) == 3

    def test_rgb_png_created(self, tmp_path):
        """RGB composite PNG should be a valid image."""
        job = _make_job(tmp_path)
        run_job(job)

        path = os.path.join(job.output_dir, "2022-06-15_rgb.png")
        assert os.path.exists(path)
        img = Image.open(path)
        assert img.size == (64, 64)
        assert img.mode == "RGB"

    def test_ndvi_png_created(self, tmp_path):
        """NDVI colormap PNG should be a valid image."""
        job = _make_job(tmp_path)
        run_job(job)

        path = os.path.join(job.output_dir, "2022-06-15_ndvi.png")
        assert os.path.exists(path)
        img = Image.open(path)
        assert img.size[0] > 0 and img.size[1] > 0

    def test_land_cover_tif_created(self, tmp_path):
        """Land cover GeoTIFF should exist and be non-empty."""
        job = _make_job(tmp_path)
        run_job(job)

        path = os.path.join(job.output_dir, "2022-06-15_land_cover.tif")
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_summary_json_valid(self, tmp_path):
        """Summary JSON should be valid and contain all analyzers."""
        job = _make_job(tmp_path)
        result = run_job(job)

        assert result.summary_path is not None
        assert os.path.exists(result.summary_path)

        with open(result.summary_path) as f:
            summary = json.load(f)

        assert summary["date"] == "2022-06-15"
        assert len(summary["analyzers"]) == 3
        names = [a["name"] for a in summary["analyzers"]]
        assert "change_detection" in names
        assert "spectral" in names
        assert "object_detection" in names

    def test_detections_geojson_valid(self, tmp_path):
        """Detections GeoJSON should be valid GeoJSON format."""
        job = _make_job(tmp_path)
        run_job(job)

        path = os.path.join(job.output_dir, "2022-06-15_detections.geojson")
        if os.path.exists(path):
            with open(path) as f:
                geojson = json.load(f)
            assert geojson["type"] == "FeatureCollection"
            for feature in geojson["features"]:
                assert feature["type"] == "Feature"
                assert "geometry" in feature
                assert feature["geometry"]["type"] == "Polygon"


class TestRunJobEdgeCases:
    """Edge cases for run_job()."""

    def test_rgb_none_returns_error(self):
        """run_job with rgb=None should return an error, not crash."""
        job = IMINTJob(
            date="2022-06-15",
            coords={"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0},
            rgb=None,
        )
        result = run_job(job)
        assert not result.success
        assert "rgb is None" in result.error

    def test_second_run_change_detection(self, tmp_path):
        """Running twice should trigger change detection on second run."""
        job1 = _make_job(tmp_path, date="2022-06-15")
        run_job(job1)

        # Second run with different data
        rng = np.random.RandomState(99)
        job2 = IMINTJob(
            date="2022-06-15",
            coords=job1.coords,
            rgb=rng.rand(64, 64, 3).astype(np.float32),
            bands=job1.bands,
            output_dir=job1.output_dir,
            config_path=job1.config_path,
            job_id="test_run2",
        )
        result2 = run_job(job2)
        assert result2.success

        # Change detection should now find differences
        cd_result = [r for r in result2.analyzer_results if r.analyzer == "change_detection"][0]
        assert cd_result.outputs["change_fraction"] > 0
