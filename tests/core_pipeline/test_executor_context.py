"""C01: preserve fetched spatial and cloud context through executors."""

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from executors.colonyos import ColonyOSExecutor
from executors.local import LocalExecutor
from imint import engine
from imint.analyzers.change_detection import ChangeDetectionAnalyzer
from imint.exporters.export import save_geotiff
from imint.fetch import FetchResult
from imint.job import GeoContext


@pytest.fixture
def scene():
    coords = dict(west=14.0, south=58.0, east=14.01, north=58.01)
    geo = GeoContext(
        "EPSG:3006",
        from_origin(440000, 6440000, 10, 10),
        dict(west=440000, south=6439680, east=440320, north=6440000),
        coords,
        (32, 32),
    )
    scl = np.full((32, 32), 4, np.uint8)
    scl[12:20, 12:20] = 9
    return FetchResult(
        {}, scl, 64 / 1024, np.zeros((32, 32, 3), np.float32), geo
    )


def _make_executor(monkeypatch, tmp_path, scene, kind):
    monkeypatch.setattr("imint.fetch.fetch_des_data", lambda **kwargs: scene)
    monkeypatch.setattr(
        "executors.colonyos.fetch_des_data", lambda **kwargs: scene
    )
    if kind == "local":
        return LocalExecutor(output_dir=str(tmp_path))

    for key, value in {
        "DATE": "2026-01-01",
        "WEST": "14",
        "SOUTH": "58",
        "EAST": "14.01",
        "NORTH": "58.01",
        "OUTPUTS_DIR": str(tmp_path),
    }.items():
        monkeypatch.setenv(key, value)
    return ColonyOSExecutor()


def test_C01_local_geotiff_preserves_fetched_grid(monkeypatch, tmp_path, scene):
    runner = _make_executor(monkeypatch, tmp_path, scene, "local")
    job = runner.build_job(date="2026-01-01", coords=scene.geo.bounds_wgs84)
    path = tmp_path / "export.tif"
    save_geotiff(
        np.ones((32, 32), np.float32),
        str(path),
        geo=job.geo,
        coords=job.coords,
    )
    with rasterio.open(path) as src:
        print("C01 observed CRS", src.crs, "transform", src.transform)
        assert src.crs.to_epsg() == 3006
        assert src.transform == scene.geo.transform
        np.testing.assert_array_equal(src.read(1), 1)


@pytest.mark.parametrize("kind", ["local", "colony"])
def test_C01_executor_scl_reaches_real_engine(monkeypatch, tmp_path, scene, kind):
    runner = _make_executor(monkeypatch, tmp_path, scene, kind)
    monkeypatch.setattr(
        engine, "ANALYZER_REGISTRY", {"change_detection": ChangeDetectionAnalyzer}
    )
    monkeypatch.setattr(
        engine,
        "_load_config",
        lambda path: {"change_detection": {"min_region_pixels": 1}},
    )
    # Known report issues are outside this regression's scope.
    monkeypatch.setattr(engine, "_generate_html_report", lambda *args: None)

    job = runner.build_job(date="2026-01-01", coords=scene.geo.bounds_wgs84)
    first = engine.run_job(job)
    assert first.success

    scene.rgb[12:20, 12:20] = 1
    job = runner.build_job(date="2026-01-01", coords=scene.geo.bounds_wgs84)
    result = engine.run_job(job)
    assert result.success

    analyzer_result = result.analyzer_results[0]
    saved = json.loads(Path(result.summary_path).read_text())["analyzers"][0]
    print(
        "C01",
        kind,
        "masked",
        analyzer_result.metadata["cloud_masked_pixels"],
        "change",
        analyzer_result.outputs["change_fraction"],
    )
    assert analyzer_result.metadata["cloud_masked_pixels"] == 64
    assert analyzer_result.outputs["change_fraction"] == 0
    assert saved["metadata"]["cloud_masked_pixels"] == 64
    assert job.scl is scene.scl


@pytest.mark.parametrize("kind", ["local", "colony"])
def test_executor_cloud_rejection_control(monkeypatch, tmp_path, scene, kind):
    scene.cloud_fraction = 0.9
    runner = _make_executor(monkeypatch, tmp_path, scene, kind)
    job = runner.build_job(date="2026-01-01", coords=scene.geo.bounds_wgs84)
    assert job.rgb is None
