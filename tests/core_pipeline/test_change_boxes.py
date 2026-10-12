"""C03: exported change boxes cover the complete pixel footprint."""

import json

import numpy as np
import pytest
from rasterio.transform import from_origin

from imint.analyzers.change_detection import ChangeDetectionAnalyzer
from imint.exporters.export import save_regions_geojson
from imint.job import GeoContext


def test_C03_change_boxes_cover_pixel_footprint(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    analyzer = ChangeDetectionAnalyzer(
        {"threshold": 0.15, "min_region_pixels": 1}
    )
    rgb = np.zeros((32, 32, 3), np.float32)
    assert analyzer.run(rgb, output_dir=str(out)).success

    rgb[10:20, 10:20] = 1
    result = analyzer.run(rgb, output_dir=str(out))
    assert result.success and result.outputs["n_regions"] == 1
    region = result.outputs["regions"][0]
    ys, xs = np.where(result.outputs["change_mask"])

    path = out / "regions.json"
    geo = GeoContext(
        "EPSG:4326", from_origin(10, 60, 0.01, 0.01), {}, {}, (32, 32)
    )
    save_regions_geojson([region], str(path), geo=geo)
    ring = json.loads(path.read_text())["features"][0]["geometry"]["coordinates"][0]
    expected_max = geo.transform * (int(xs.max()) + 1, int(ys.max()) + 1)
    print(
        "C03 bbox",
        region["bbox"],
        "max changed pixel",
        int(xs.max()),
        int(ys.max()),
        "export east/south",
        max(point[0] for point in ring),
        min(point[1] for point in ring),
        "expected",
        expected_max,
    )
    assert max(point[0] for point in ring) == pytest.approx(expected_max[0])
    assert min(point[1] for point in ring) == pytest.approx(expected_max[1])
