"""Regression tests for the SJV/LPIS (N,E) axis-swap fix.

The swap has recurred twice in this repo's history (ae5fe17 2026-04, then
2026-07-20). These lock the detection + in-place correction so it can't
silently regress again.
"""
import importlib.util
from pathlib import Path

import pytest

gpd = pytest.importorskip("geopandas")
shapely_geom = pytest.importorskip("shapely.geometry")

_SPEC = importlib.util.spec_from_file_location(
    "convert_lpis", Path(__file__).resolve().parents[1]
    / "scripts" / "convert_lpis_wfs_zip.py")
conv = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(conv)


def _box(minx, miny, maxx, maxy):
    from shapely.geometry import box
    return box(minx, miny, maxx, maxy)


def _gdf(coords_en: bool):
    """A one-parcel GeoDataFrame in Sweden. If coords_en=False the geometry is
    stored (N, E) — the WFS-official swap we must correct."""
    # A real Swedish parcel footprint in (E, N).
    e0, n0, e1, n1 = 400000, 6400000, 400100, 6400100
    geom = _box(e0, n0, e1, n1) if coords_en else _box(n0, e0, n1, e1)
    return gpd.GeoDataFrame({"grdkod_mar": [4]}, geometry=[geom], crs=3006)


def test_axes_ok_detects_orientation():
    assert conv._axes_ok(_gdf(True).total_bounds)
    assert not conv._axes_ok(_gdf(False).total_bounds)


def test_ensure_en_swaps_only_when_needed():
    ok = _gdf(True)
    assert conv._ensure_en_axes(ok).total_bounds.tolist() == ok.total_bounds.tolist()
    fixed = conv._ensure_en_axes(_gdf(False))
    assert conv._axes_ok(fixed.total_bounds)


def test_fix_parquet_is_idempotent_and_backs_up(tmp_path):
    p = tmp_path / "lpis.parquet"
    _gdf(False).to_parquet(p, index=False)          # write a swapped parquet

    assert conv.fix_parquet_axes(str(p)) == "fixed"
    assert (tmp_path / "lpis.parquet.bak").exists()  # original backed up
    assert conv._axes_ok(gpd.read_parquet(p).total_bounds)

    # backup preserves the ORIGINAL (still swapped)
    assert not conv._axes_ok(gpd.read_parquet(str(p) + ".bak").total_bounds)

    # second run is a no-op, no double-swap
    assert conv.fix_parquet_axes(str(p)) == "ok"
    assert conv._axes_ok(gpd.read_parquet(p).total_bounds)


def test_fix_parquet_leaves_correct_file_untouched(tmp_path):
    p = tmp_path / "good.parquet"
    _gdf(True).to_parquet(p, index=False)
    assert conv.fix_parquet_axes(str(p)) == "ok"
    assert not (tmp_path / "good.parquet.bak").exists()
