"""Unit tests for imint.training.spatial_parquet.SpatialParquet.

Covers the contract:
    - Preprocessed parquet (with _bbox_* cols) uses row-group filtering
    - Fallback to full parquet load when spatial version missing (warns)
    - Query returns only polygons whose bbox intersects tile bbox
    - Empty result returns empty GeoDataFrame (not None, no crash)
    - Dict bbox and tuple bbox both accepted
"""
from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from imint.training.spatial_parquet import SpatialParquet


def _make_test_parquet(path: str, polygons_per_cell: int = 5) -> gpd.GeoDataFrame:
    """Build a synthetic parquet covering a 3×3 grid of 100km cells,
    matching what preprocess_sks_lpis_spatial.py writes.

    Cells span EPSG:3006 [0, 300000] × [6000000, 6300000] in 100 km steps.
    Each cell gets ``polygons_per_cell`` little 1 km squares inside it.
    """
    rows = []
    grid_m = 100_000
    for gx in range(3):
        for gy in range(3):
            cell_x0 = gx * grid_m
            cell_y0 = 6_000_000 + gy * grid_m
            for i in range(polygons_per_cell):
                # Stagger polygons inside the cell
                px = cell_x0 + 10_000 + i * 15_000
                py = cell_y0 + 10_000 + i * 15_000
                geom = box(px, py, px + 1_000, py + 1_000)
                rows.append({
                    "id": f"poly_{gx}_{gy}_{i}",
                    "Avvdatum": pd.Timestamp(f"202{i % 5}-06-15"),
                    "geometry": geom,
                    "_bbox_minx": float(px),
                    "_bbox_miny": float(py),
                    "_bbox_maxx": float(px + 1_000),
                    "_bbox_maxy": float(py + 1_000),
                    "_grid_x": int(px // grid_m),
                    "_grid_y": int(py // grid_m),
                })
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:3006")
    # Sort by grid cell (matches preprocess output)
    gdf = gdf.sort_values(by=["_grid_x", "_grid_y"]).reset_index(drop=True)
    # Write with small row groups so we exercise row-group filtering
    gdf.to_parquet(path, index=False, compression="snappy", row_group_size=5)
    return gdf


@pytest.fixture
def spatial_parquet_path(tmp_path):
    path = str(tmp_path / "test_spatial.parquet")
    _make_test_parquet(path, polygons_per_cell=5)
    return path


class TestSpatialParquetFastPath:
    def test_loads_lazily(self, spatial_parquet_path):
        sp = SpatialParquet(spatial_parquet_path)
        # File not opened until query
        assert sp._parq is None
        assert sp._rg_bboxes is None

        sp.query({"west": 0, "south": 6_000_000, "east": 5_000, "north": 6_005_000})
        assert sp._parq is not None
        assert sp._rg_bboxes is not None

    def test_query_returns_intersecting_polygons(self, spatial_parquet_path):
        sp = SpatialParquet(spatial_parquet_path)
        # Query the first 1km polygon in the (0, 0) cell
        bbox = {"west": 10_000, "south": 6_010_000, "east": 11_000, "north": 6_011_000}
        result = sp.query(bbox)
        assert len(result) == 1
        assert result.iloc[0]["id"] == "poly_0_0_0"

    def test_tuple_bbox_accepted(self, spatial_parquet_path):
        sp = SpatialParquet(spatial_parquet_path)
        bbox = (10_000, 6_010_000, 11_000, 6_011_000)
        result = sp.query(bbox)
        assert len(result) == 1

    def test_query_spanning_cells(self, spatial_parquet_path):
        """Bbox that straddles two 100km cells should return polygons
        from both cells."""
        sp = SpatialParquet(spatial_parquet_path)
        bbox = {
            "west": 90_000, "east": 110_000,
            "south": 6_000_000, "north": 6_100_000,
        }
        result = sp.query(bbox)
        # Should have polygons from grid cells touching both sides
        assert len(result) > 0

    def test_empty_query_returns_empty_gdf(self, spatial_parquet_path):
        sp = SpatialParquet(spatial_parquet_path)
        # Far from any cell
        bbox = {"west": 999_000, "east": 999_100,
                "south": 7_000_000, "north": 7_000_100}
        result = sp.query(bbox)
        assert len(result) == 0
        assert isinstance(result, gpd.GeoDataFrame)

    def test_exact_filter_applied(self, spatial_parquet_path):
        """Polygons in the same row group but outside tile bbox should
        be dropped by the exact per-row filter."""
        sp = SpatialParquet(spatial_parquet_path)
        # Tiny query that matches only 1 of 5 polygons in (0, 0) cell
        bbox = {"west": 10_000, "south": 6_010_000,
                "east": 11_000, "north": 6_011_000}
        result = sp.query(bbox)
        ids = list(result["id"])
        assert ids == ["poly_0_0_0"]

    def test_extracts_row_group_bboxes(self, spatial_parquet_path):
        sp = SpatialParquet(spatial_parquet_path)
        sp._ensure_open()
        assert sp._rg_bboxes is not None
        # 9 cells × 5 polygons = 45 rows; row_group_size=5 → 9 row groups
        assert len(sp._rg_bboxes) == 9
        # Each row group bbox should be within the 3×3 grid extent
        for minx, miny, maxx, maxy in sp._rg_bboxes:
            assert 0 <= minx < 300_000
            assert 6_000_000 <= miny < 6_300_000


class TestSpatialParquetFallback:
    def test_missing_spatial_uses_fallback(self, tmp_path):
        # Create only the legacy (no _bbox_* cols) parquet
        legacy_path = str(tmp_path / "legacy.parquet")
        gdf = gpd.GeoDataFrame(
            [{"id": "a", "geometry": box(0, 0, 10, 10)}],
            crs="EPSG:3006",
        )
        gdf.to_parquet(legacy_path)

        spatial_path = str(tmp_path / "missing_spatial.parquet")
        sp = SpatialParquet(spatial_path, fallback_path=legacy_path)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = sp.query({"west": 0, "south": 0, "east": 20, "north": 20})
            # Warning should have been emitted about the missing spatial
            assert any("missing" in str(warning.message).lower() for warning in w)

        assert len(result) == 1
        assert result.iloc[0]["id"] == "a"

    def test_missing_both_raises(self, tmp_path):
        sp = SpatialParquet(
            str(tmp_path / "nope.parquet"),
            fallback_path=str(tmp_path / "also_nope.parquet"),
        )
        with pytest.raises(FileNotFoundError):
            sp.query({"west": 0, "south": 0, "east": 1, "north": 1})

    def test_missing_bbox_columns_raises(self, tmp_path):
        """Parquet without _bbox_* columns should raise on ensure_open."""
        path = str(tmp_path / "no_bbox_cols.parquet")
        gdf = gpd.GeoDataFrame(
            [{"id": "a", "geometry": box(0, 0, 10, 10)}],
            crs="EPSG:3006",
        )
        gdf.to_parquet(path)

        sp = SpatialParquet(path)
        with pytest.raises(ValueError, match="_bbox_"):
            sp.query({"west": 0, "south": 0, "east": 1, "north": 1})


class TestSpatialParquetCaching:
    def test_get_spatial_parquet_caches(self, spatial_parquet_path):
        from imint.training.spatial_parquet import get_spatial_parquet
        a = get_spatial_parquet(spatial_parquet_path)
        b = get_spatial_parquet(spatial_parquet_path)
        assert a is b
