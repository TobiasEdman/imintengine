"""Reverse-fit tests for NFI plot → tile pixel co-location.

The repo's bulletproof discipline (CLAUDE.md): verify a transform by
injecting a known input and checking the exact output — here, stamp a point
at a known pixel centre and confirm it maps back to that pixel, and confirm
the y-axis flip (north = row 0) so we never repeat a silent half-tile flip.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from imint.training.nfi_colocate import (
    build_plot_index,
    colocate_plots,
    point_to_pixel,
    tile_year,
)
from imint.training.tile_config import TileConfig

SIZE = 512
TILE = TileConfig(size_px=SIZE)
CENTER_E, CENTER_N = 500_000, 6_500_000
BBOX = TILE.bbox_from_center(CENTER_E, CENTER_N)  # 497440/6497440/502560/6502560


def _pixel_center(row: int, col: int) -> tuple[float, float]:
    """World (E, N) of a pixel centre — north-up, row 0 at the north edge."""
    e = BBOX["west"] + (col + 0.5) * 10
    n = BBOX["north"] - (row + 0.5) * 10
    return e, n


@pytest.mark.parametrize("row,col", [(0, 0), (511, 511), (256, 256), (100, 400), (400, 100)])
def test_point_to_pixel_reverse_fit(row, col):
    e, n = _pixel_center(row, col)
    assert point_to_pixel(e, n, BBOX, SIZE) == (row, col)


def test_y_axis_flip():
    """A point near the NORTH edge is row 0; near the SOUTH edge, the last row."""
    near_north = point_to_pixel(CENTER_E, BBOX["north"] - 1, BBOX, SIZE)
    near_south = point_to_pixel(CENTER_E, BBOX["south"] + 1, BBOX, SIZE)
    assert near_north[0] == 0
    assert near_south[0] == SIZE - 1


def test_top_left_corner_inclusive_far_edges_exclusive():
    assert point_to_pixel(BBOX["west"], BBOX["north"], BBOX, SIZE) == (0, 0)
    # east / south edges belong to the neighbouring tile (floor → size_px)
    assert point_to_pixel(BBOX["east"], CENTER_N, BBOX, SIZE) is None
    assert point_to_pixel(CENTER_E, BBOX["south"], BBOX, SIZE) is None


def test_out_of_bounds_returns_none():
    assert point_to_pixel(BBOX["west"] - 10, CENTER_N, BBOX, SIZE) is None
    assert point_to_pixel(BBOX["east"] + 10, CENTER_N, BBOX, SIZE) is None
    assert point_to_pixel(CENTER_E, BBOX["north"] + 10, BBOX, SIZE) is None


def test_tile_year_resolution_order():
    assert tile_year({"year": 2023, "lpis_year": 2019}) == 2023  # year wins
    assert tile_year({"lpis_year": 2021}) == 2021                # falls to lpis_year
    assert tile_year({"dates": ["2022-09-01", "2022-06-01", "2021-10-01"]}) == 2022  # modal
    assert tile_year({"dates": ["2020-01-01", "2021-01-01"]}) == 2021  # tie → most recent
    assert tile_year({}) is None  # no fallback default (unlike build_labels)


def _plots():
    return pd.DataFrame(
        {
            "PlotID": [1, 2, 3, 4],
            "Easting": [CENTER_E, CENTER_E, CENTER_E, 600_000],  # 4th is far outside
            "Northing": [CENTER_N, CENTER_N, CENTER_N, CENTER_N],
            "Year": [2023, 2022, 2023, 2023],
        }
    )


def test_colocate_plots_year_and_bbox_filter():
    npz = {"easting": CENTER_E, "northing": CENTER_N, "year": 2023}
    got = colocate_plots(_plots(), name="tile_500000_6500000", npz_data=npz, tile=TILE)
    # Year 2023 + inside bbox → PlotID 1 and 3 (not 2: wrong year; not 4: outside)
    assert sorted(got["PlotID"]) == [1, 3]
    # Tile centre maps to pixel (256, 256)
    assert set(zip(got["row"], got["col"])) == {(256, 256)}


def test_colocate_plots_year_match_disabled():
    npz = {"easting": CENTER_E, "northing": CENTER_N, "year": 2023}
    got = colocate_plots(
        _plots(), name="t", npz_data=npz, tile=TILE, require_year_match=False,
    )
    assert sorted(got["PlotID"]) == [1, 2, 3]  # year ignored; 4 still outside


def test_colocate_skips_tile_with_unknown_year():
    npz = {"easting": CENTER_E, "northing": CENTER_N}  # no year/lpis_year/dates
    got = colocate_plots(_plots(), name="t", npz_data=npz, tile=TILE)
    assert got.empty


def test_build_plot_index_over_npz(tmp_path):
    path = tmp_path / "tile_500000_6500000.npz"
    np.savez(path, easting=CENTER_E, northing=CENTER_N, year=2023)
    idx = build_plot_index([path], _plots(), TILE)
    assert list(idx["tile_name"]) == ["tile_500000_6500000", "tile_500000_6500000"]
    assert sorted(idx["PlotID"]) == [1, 3]
    assert set(zip(idx["row"], idx["col"])) == {(256, 256)}


def test_build_plot_index_empty_when_no_overlap(tmp_path):
    path = tmp_path / "tile_500000_6500000.npz"
    np.savez(path, easting=CENTER_E, northing=CENTER_N, year=2019)  # no 2019 plots
    idx = build_plot_index([path], _plots(), TILE)
    assert idx.empty
    assert {"tile_name", "row", "col"}.issubset(idx.columns)
