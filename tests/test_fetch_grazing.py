"""Tests for grazing timeseries fetching (imint.fetch grazing functions).

Tests polygon-level cloud fraction, polygon rasterization, bbox projection,
band reordering, and the GrazingTimeseriesResult dataclass.
"""
from __future__ import annotations

import numpy as np
import pytest

from imint.fetch import (
    _polygon_cloud_fraction,
    _rasterize_polygon,
    _polygon_to_projected_bbox,
    _lpis_bbox_string,
    fetch_lpis_polygons,
    fetch_grazing_lpis,
    _GRAZING_REORDER,
    GRAZING_BAND_ORDER,
    GrazingTimeseriesResult,
    FetchError,
    SCL_CLOUD_CLASSES,
    NMD_GRID_SIZE,
    LPIS_WFS_URL,
    LPIS_LAYER,
    LPIS_PASTURE_AGOSLAG,
)
from imint.job import GeoContext


# ── _polygon_cloud_fraction ──────────────────────────────────────────────────

class TestPolygonCloudFraction:

    def test_all_clear_in_polygon(self):
        """Polygon area has no clouds → fraction = 0.0."""
        scl = np.full((10, 10), 4, dtype=np.uint8)  # 4 = vegetation
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:8, 2:8] = True  # 36 pixels inside polygon
        assert _polygon_cloud_fraction(scl, mask) == 0.0

    def test_all_cloudy_in_polygon(self):
        """Polygon area is fully cloudy → fraction = 1.0."""
        scl = np.full((10, 10), 9, dtype=np.uint8)  # 9 = cloud high
        mask = np.ones((10, 10), dtype=bool)
        assert _polygon_cloud_fraction(scl, mask) == 1.0

    def test_cloud_outside_polygon_ignored(self):
        """Clouds outside polygon mask should NOT count."""
        scl = np.full((10, 10), 9, dtype=np.uint8)  # all cloud
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True  # single clear pixel inside polygon
        scl[5, 5] = 4  # vegetation (clear) at the polygon pixel
        assert _polygon_cloud_fraction(scl, mask) == 0.0

    def test_partial_cloud_in_polygon(self):
        """Half of polygon pixels are cloudy → 0.5."""
        scl = np.full((10, 10), 4, dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=bool)
        mask[0:2, 0:5] = True  # 10 pixels
        # Make 5 of them cloudy
        scl[0, 0:5] = 9  # 5 cloud pixels in polygon
        assert _polygon_cloud_fraction(scl, mask) == pytest.approx(0.5)

    def test_empty_mask_returns_one(self):
        """Zero-area mask returns 1.0 (fail-safe: no valid pixels)."""
        scl = np.full((5, 5), 4, dtype=np.uint8)
        mask = np.zeros((5, 5), dtype=bool)
        assert _polygon_cloud_fraction(scl, mask) == 1.0

    def test_all_scl_cloud_classes_detected(self):
        """All SCL cloud classes (3, 8, 9, 10) should be detected."""
        for cls in SCL_CLOUD_CLASSES:
            scl = np.full((3, 3), cls, dtype=np.uint8)
            mask = np.ones((3, 3), dtype=bool)
            assert _polygon_cloud_fraction(scl, mask) == 1.0, (
                f"SCL class {cls} not detected as cloud"
            )

    def test_non_cloud_classes_ignored(self):
        """Non-cloud SCL classes (0,1,2,4,5,6,7,11) should NOT count."""
        non_cloud = [0, 1, 2, 4, 5, 6, 7, 11]
        for cls in non_cloud:
            scl = np.full((3, 3), cls, dtype=np.uint8)
            mask = np.ones((3, 3), dtype=bool)
            assert _polygon_cloud_fraction(scl, mask) == 0.0, (
                f"SCL class {cls} incorrectly counted as cloud"
            )


# ── _rasterize_polygon ───────────────────────────────────────────────────────

class TestRasterizePolygon:

    def test_simple_rectangle(self):
        """Rasterize a rectangle that covers most of the bbox."""
        from shapely.geometry import box

        bounds = {"west": 500000, "south": 6500000,
                  "east": 500100, "north": 6500100}
        # Polygon covers inner 80x80m of 100x100m bbox
        poly = box(500010, 6500010, 500090, 6500090)
        mask = _rasterize_polygon(poly, bounds, pixel_size=10)

        assert mask.shape == (10, 10)
        assert mask.dtype == bool
        # Inner 8x8 should be True
        assert mask.sum() > 0
        assert mask.sum() <= 64  # 8x8

    def test_polygon_smaller_than_bbox(self):
        """Small polygon in large bbox → partial mask."""
        from shapely.geometry import box

        bounds = {"west": 500000, "south": 6500000,
                  "east": 500200, "north": 6500200}
        # Small 20x20m polygon in corner
        poly = box(500000, 6500180, 500020, 6500200)
        mask = _rasterize_polygon(poly, bounds, pixel_size=10)

        assert mask.shape == (20, 20)
        # Only a small part should be True
        assert 0 < mask.sum() < mask.size

    def test_output_shape_matches_bounds(self):
        """Output shape should be (north-south)/px × (east-west)/px."""
        from shapely.geometry import box

        bounds = {"west": 100000, "south": 6000000,
                  "east": 100050, "north": 6000070}
        poly = box(100010, 6000010, 100040, 6000060)
        mask = _rasterize_polygon(poly, bounds, pixel_size=10)

        expected_h = (6000070 - 6000000) // 10  # 7
        expected_w = (100050 - 100000) // 10     # 5
        assert mask.shape == (expected_h, expected_w)


# ── _polygon_to_projected_bbox ───────────────────────────────────────────────

class TestPolygonToProjectedBbox:

    def test_buffer_expands_bbox(self):
        """Buffer should make the bbox larger."""
        from shapely.geometry import box

        # A small polygon in EPSG:3006 coordinates
        poly = box(500050, 6500050, 500100, 6500100)

        bbox_no_buf = _polygon_to_projected_bbox(
            poly, src_crs="EPSG:3006", buffer_m=0.0,
        )
        bbox_with_buf = _polygon_to_projected_bbox(
            poly, src_crs="EPSG:3006", buffer_m=50.0,
        )

        assert bbox_with_buf["west"] <= bbox_no_buf["west"]
        assert bbox_with_buf["south"] <= bbox_no_buf["south"]
        assert bbox_with_buf["east"] >= bbox_no_buf["east"]
        assert bbox_with_buf["north"] >= bbox_no_buf["north"]

    def test_grid_snapping(self):
        """Output should be snapped to NMD 10m grid."""
        from shapely.geometry import box

        # Polygon with non-aligned coordinates
        poly = box(500013, 6500017, 500087, 6500093)
        bbox = _polygon_to_projected_bbox(
            poly, src_crs="EPSG:3006", buffer_m=0.0,
        )

        assert bbox["west"] % NMD_GRID_SIZE == 0
        assert bbox["south"] % NMD_GRID_SIZE == 0
        assert bbox["east"] % NMD_GRID_SIZE == 0
        assert bbox["north"] % NMD_GRID_SIZE == 0

    def test_crs_key_present(self):
        """Result should include 'crs' key."""
        from shapely.geometry import box

        poly = box(500000, 6500000, 500100, 6500100)
        bbox = _polygon_to_projected_bbox(
            poly, src_crs="EPSG:3006", buffer_m=0,
        )
        assert bbox["crs"] == "EPSG:3006"


# ── Band reorder indices ─────────────────────────────────────────────────────

class TestBandReorder:

    def test_reorder_length(self):
        """Reorder array has 12 elements (one per spectral band)."""
        assert len(_GRAZING_REORDER) == 12

    def test_reorder_is_permutation(self):
        """Reorder indices should be a permutation of 0..11."""
        assert sorted(_GRAZING_REORDER) == list(range(12))

    def test_reorder_produces_correct_order(self):
        """Applying reorder to download order should give GRAZING_BAND_ORDER."""
        download_order = [
            # 10m
            "b02", "b03", "b04", "b08",
            # 20m
            "b05", "b06", "b07", "b8a", "b11", "b12",
            # 60m all
            "b01", "b09",
        ]
        reordered = [download_order[i] for i in _GRAZING_REORDER]
        assert reordered == GRAZING_BAND_ORDER

    def test_reorder_with_numpy(self):
        """Numpy fancy indexing with reorder should work correctly."""
        # Create fake (13, 4, 4) array where band i has all values = i
        fake = np.zeros((13, 4, 4), dtype=np.float32)
        for i in range(13):
            fake[i] = float(i)

        spectral = fake[_GRAZING_REORDER]  # (12, 4, 4)
        assert spectral.shape == (12, 4, 4)
        # B01 = download index 10, should be first in reordered
        assert spectral[0, 0, 0] == 10.0
        # B02 = download index 0, should be second
        assert spectral[1, 0, 0] == 0.0
        # B04 = download index 2, should be fourth (index 3)
        assert spectral[3, 0, 0] == 2.0


# ── GrazingTimeseriesResult ──────────────────────────────────────────────────

class TestGrazingTimeseriesResult:

    def test_create_with_data(self):
        """Create result with T=3 timesteps."""
        data = np.random.rand(3, 12, 46, 46).astype(np.float32)
        geo = GeoContext(
            crs="EPSG:3006", transform=None,
            bounds_projected={"west": 0, "south": 0, "east": 460, "north": 460},
            bounds_wgs84={"west": 13.0, "south": 55.0, "east": 13.1, "north": 55.1},
            shape=(46, 46),
        )
        result = GrazingTimeseriesResult(
            data=data,
            dates=["2023-05-01", "2023-06-15", "2023-08-01"],
            cloud_fractions=[0.0, 0.005, 0.002],
            polygon_id="field_42",
            geo=geo,
            shape_hw=(46, 46),
        )
        assert result.data.shape == (3, 12, 46, 46)
        assert len(result.dates) == 3
        assert result.polygon_id == "field_42"

    def test_create_empty(self):
        """Create result with T=0 (no cloud-free dates)."""
        data = np.empty((0, 12, 46, 46), dtype=np.float32)
        geo = GeoContext(
            crs="EPSG:3006", transform=None,
            bounds_projected={}, bounds_wgs84={}, shape=(46, 46),
        )
        result = GrazingTimeseriesResult(
            data=data, dates=[], cloud_fractions=[],
            polygon_id=0, geo=geo, shape_hw=(46, 46),
        )
        assert result.data.shape[0] == 0
        assert result.data.shape[1] == 12
        assert len(result.dates) == 0


# ── LPIS bbox helpers ────────────────────────────────────────────────────

class TestLpisBboxString:

    def test_dict_wgs84_autodetect(self):
        """Dict with small coords → WGS84."""
        bbox_str, crs = _lpis_bbox_string(
            {"west": 13.4, "south": 55.9, "east": 13.5, "north": 56.0}
        )
        assert crs == "EPSG:4326"
        assert "13.4,55.9,13.5,56.0" in bbox_str
        assert bbox_str.endswith(",EPSG:4326")

    def test_dict_epsg3006_autodetect(self):
        """Dict with large coords → EPSG:3006."""
        bbox_str, crs = _lpis_bbox_string(
            {"west": 400000, "south": 6200000, "east": 410000, "north": 6210000}
        )
        assert crs == "EPSG:3006"
        assert bbox_str.endswith(",EPSG:3006")

    def test_tuple_input(self):
        """Tuple (w, s, e, n) works."""
        bbox_str, crs = _lpis_bbox_string((13.4, 55.9, 13.5, 56.0))
        assert crs == "EPSG:4326"
        assert "13.4,55.9" in bbox_str

    def test_explicit_crs_overrides(self):
        """Explicit bbox_crs takes precedence."""
        bbox_str, crs = _lpis_bbox_string(
            {"west": 13.4, "south": 55.9, "east": 13.5, "north": 56.0},
            bbox_crs="EPSG:3006",
        )
        assert crs == "EPSG:3006"

    def test_invalid_bbox_raises(self):
        """Non-dict, non-tuple raises ValueError."""
        with pytest.raises(ValueError):
            _lpis_bbox_string("invalid")


# ── LPIS constants ───────────────────────────────────────────────────────

class TestLpisConstants:

    def test_wfs_url(self):
        assert "epub.sjv.se" in LPIS_WFS_URL
        assert "wfs" in LPIS_WFS_URL

    def test_layer_name(self):
        assert LPIS_LAYER == "inspire:senaste_arslager_block"

    def test_pasture_agoslag(self):
        assert LPIS_PASTURE_AGOSLAG == "Bete"


# ── fetch_lpis_polygons (mocked) ─────────────────────────────────────────

# Minimal GeoJSON fixture simulating Jordbruksverket WFS response
_LPIS_GEOJSON_FIXTURE = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[400010, 6200010], [400110, 6200010],
                                 [400110, 6200110], [400010, 6200110],
                                 [400010, 6200010]]],
            },
            "properties": {
                "blockid": "12345678901",
                "agoslag": "Bete",
                "kategori": "Gård/Miljö",
                "areal": 1.0,
                "region_kod": "1234",
                "arslager": 2024,
            },
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[400200, 6200200], [400300, 6200200],
                                 [400300, 6200300], [400200, 6200300],
                                 [400200, 6200200]]],
            },
            "properties": {
                "blockid": "12345678902",
                "agoslag": "Åker",
                "kategori": "Gård/Miljö",
                "areal": 2.5,
                "region_kod": "1234",
                "arslager": 2024,
            },
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[400400, 6200400], [400500, 6200400],
                                 [400500, 6200500], [400400, 6200500],
                                 [400400, 6200400]]],
            },
            "properties": {
                "blockid": "12345678903",
                "agoslag": "Bete",
                "kategori": "Ej stödberättigande",
                "areal": 0.8,
                "region_kod": "1234",
                "arslager": 2024,
            },
        },
    ],
}


class TestFetchLpisPolygons:

    def _mock_urlopen(self, monkeypatch):
        """Patch urllib to return the GeoJSON fixture."""
        import io
        import json

        class FakeResponse:
            def __init__(self):
                self._data = json.dumps(_LPIS_GEOJSON_FIXTURE).encode()
            def read(self):
                return self._data
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass

        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda req, timeout=None: FakeResponse(),
        )

    def test_returns_geodataframe(self, monkeypatch):
        """Should return a GeoDataFrame with expected columns."""
        self._mock_urlopen(monkeypatch)
        gdf = fetch_lpis_polygons(
            {"west": 400000, "south": 6200000, "east": 401000, "north": 6201000},
            agoslag=None,
        )
        assert hasattr(gdf, "geometry")
        assert hasattr(gdf, "crs")
        assert len(gdf) == 3
        assert "blockid" in gdf.columns
        assert "agoslag" in gdf.columns

    def test_filter_bete(self, monkeypatch):
        """Default filter should keep only 'Bete' rows."""
        self._mock_urlopen(monkeypatch)
        gdf = fetch_lpis_polygons(
            (400000, 6200000, 401000, 6201000),
            agoslag="Bete",
        )
        assert len(gdf) == 2
        assert set(gdf["agoslag"]) == {"Bete"}

    def test_filter_none_returns_all(self, monkeypatch):
        """agoslag=None should return all land-use types."""
        self._mock_urlopen(monkeypatch)
        gdf = fetch_lpis_polygons(
            (400000, 6200000, 401000, 6201000),
            agoslag=None,
        )
        assert len(gdf) == 3

    def test_filter_multiple(self, monkeypatch):
        """List of ägoslag should filter correctly."""
        self._mock_urlopen(monkeypatch)
        gdf = fetch_lpis_polygons(
            (400000, 6200000, 401000, 6201000),
            agoslag=["Bete", "Åker"],
        )
        assert len(gdf) == 3

    def test_crs_is_epsg3006(self, monkeypatch):
        """Result CRS should be EPSG:3006."""
        self._mock_urlopen(monkeypatch)
        gdf = fetch_lpis_polygons(
            (400000, 6200000, 401000, 6201000),
            agoslag=None,
        )
        assert "3006" in str(gdf.crs)

    def test_empty_response(self, monkeypatch):
        """Empty WFS response → empty GeoDataFrame."""
        import json

        class FakeEmpty:
            def read(self):
                return json.dumps({"type": "FeatureCollection", "features": []}).encode()
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass

        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda req, timeout=None: FakeEmpty(),
        )
        gdf = fetch_lpis_polygons(
            (13.4, 55.9, 13.5, 56.0),
            agoslag="Bete",
        )
        assert len(gdf) == 0
        assert "blockid" in gdf.columns

    def test_cache_roundtrip(self, monkeypatch, tmp_path):
        """Cached file should be used on second call."""
        self._mock_urlopen(monkeypatch)

        # First call — fetches from WFS
        gdf1 = fetch_lpis_polygons(
            (400000, 6200000, 401000, 6201000),
            agoslag="Bete",
            cache_dir=str(tmp_path),
        )
        assert len(gdf1) == 2

        # Replace urlopen with one that raises (should not be called)
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda req, timeout=None: (_ for _ in ()).throw(
                RuntimeError("should not be called")
            ),
        )

        # Second call — should use cache
        gdf2 = fetch_lpis_polygons(
            (400000, 6200000, 401000, 6201000),
            agoslag="Bete",
            cache_dir=str(tmp_path),
        )
        assert len(gdf2) == 2


# ── fetch_grazing_lpis ───────────────────────────────────────────────────

class TestFetchGrazingLpis:

    def test_no_polygons_raises(self, monkeypatch):
        """FetchError if LPIS returns no polygons."""
        import json

        class FakeEmpty:
            def read(self):
                return json.dumps({"type": "FeatureCollection", "features": []}).encode()
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass

        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda req, timeout=None: FakeEmpty(),
        )
        with pytest.raises(FetchError):
            fetch_grazing_lpis(
                bbox=(13.4, 55.9, 13.5, 56.0),
                year=2024,
            )
