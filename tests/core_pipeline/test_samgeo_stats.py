"""C02: count foreground segments whether or not background is present."""

import sys
import types

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from imint.analyzers.samgeo import SAMGeoAnalyzer


@pytest.mark.parametrize(
    "values,expected",
    [
        (np.ones((8, 8), np.uint16), (1, 64, 64)),
        (np.zeros((8, 8), np.uint16), (0, 0, 0)),
        (
            np.repeat(
                [[1], [1], [1], [2], [2], [2], [2], [2]], 8, axis=1
            ).astype(np.uint16),
            (2, 40, 32),
        ),
        (np.pad(np.ones((4, 4), np.uint16), 2), (1, 16, 16)),
    ],
)
def test_C02_samgeo_foreground_stats(monkeypatch, tmp_path, values, expected):
    class FakeSam:
        def __init__(self, **kwargs):
            pass

        def generate(self, input_path, output):
            with rasterio.open(
                output,
                "w",
                driver="GTiff",
                height=8,
                width=8,
                count=1,
                dtype="uint16",
                transform=from_origin(100, 200, 10, 10),
            ) as dst:
                dst.write(values, 1)

    fake = types.ModuleType("samgeo")
    fake.SamGeo = FakeSam
    fake.SamGeo2 = FakeSam
    monkeypatch.setitem(sys.modules, "samgeo", fake)
    monkeypatch.setitem(sys.modules, "samgeo.samgeo", fake)

    analyzer = SAMGeoAnalyzer({"min_mask_area": 0, "device": "cpu"})
    result = analyzer.run(
        np.zeros((8, 8, 3), np.float32), output_dir=str(tmp_path)
    )
    print(
        "C02 labels",
        np.unique(values),
        "success",
        result.success,
        "error",
        result.error,
        "stats",
        result.outputs.get("segment_stats"),
    )
    assert result.success, result.error
    stats = result.outputs["segment_stats"]
    assert (
        stats["n_segments"],
        stats["largest_segment_pixels"],
        stats["median_segment_pixels"],
    ) == expected
    np.testing.assert_array_equal(result.outputs["seg_mask"], values)
