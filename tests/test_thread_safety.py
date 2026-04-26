"""Regression test: SpatialParquet + tile_fetch are thread-safe.

The label-misalignment bug (2026-04) was caused by sharing a single
``rasterio.DatasetReader`` and a single ``pyarrow.parquet.ParquetFile``
across many threads inside ``build_labels.py``'s ``ThreadPoolExecutor``.
Both libraries hold mutable cursor / read state and are not thread-safe
for concurrent reads on the same handle.

The fix moved the handles into ``threading.local()`` so each thread
gets its own. This test pins that invariant: the result of N threads
each calling the read path with their own bbox MUST equal the result
of running the same N calls sequentially in a single thread.

Run:
    pytest tests/test_thread_safety.py
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_raster(tmpdir: Path) -> Path:
    """Create a 1024×1024 raster where pixel value = (row // 10) so any
    row/column drift between concurrent reads shows up as a wrong value.
    """
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    path = tmpdir / "synth_nmd.tif"
    arr = np.zeros((1024, 1024), dtype=np.uint8)
    # vary by (row, col) so any window misalignment is detectable
    for r in range(1024):
        arr[r, :] = r % 19  # values 0..18, matching NMD sequential range
    transform = from_origin(0, 10240, 10, 10)  # 10 m pixels, north-up
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=1024, width=1024, count=1,
        dtype="uint8",
        crs="EPSG:3006",
        transform=transform,
    ) as dst:
        dst.write(arr, 1)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fetch_nmd_label_local_thread_safe(tmp_path: Path) -> None:
    """Concurrent calls with distinct bboxes must equal serial calls.

    This is the regression for the 2026-04 label-seam bug.
    """
    pytest.importorskip("rasterio")
    from imint.training.tile_config import TileConfig
    from imint.training.tile_fetch import fetch_nmd_label_local

    raster = _make_synthetic_raster(tmp_path)
    tile = TileConfig(size_px=64)  # 640 m extent

    # Build a list of distinct bboxes that don't overlap. Synthetic
    # raster spans (0,0) → (10240, 10240); we tile it into 64-px windows.
    bboxes = []
    for ix in range(8):
        for iy in range(8):
            west = ix * 640
            south = iy * 640
            bboxes.append({
                "west": west, "south": south,
                "east": west + 640, "north": south + 640,
            })

    # Serial reference
    serial = [fetch_nmd_label_local(b, tile, str(raster)) for b in bboxes]
    assert all(r is not None for r in serial), "serial fetch returned None"

    # Concurrent reads — 16 threads, all hitting the shared TLS handle
    with ThreadPoolExecutor(max_workers=16) as pool:
        parallel = list(pool.map(
            lambda b: fetch_nmd_label_local(b, tile, str(raster)),
            bboxes,
        ))

    assert len(parallel) == len(serial)
    for i, (s, p) in enumerate(zip(serial, parallel)):
        assert p is not None, f"parallel fetch[{i}] returned None"
        np.testing.assert_array_equal(
            s, p,
            err_msg=(
                f"thread-unsafe NMD read at bbox {bboxes[i]}: "
                "serial vs concurrent results differ"
            ),
        )


def test_spatial_parquet_thread_safe(tmp_path: Path) -> None:
    """Concurrent SpatialParquet.query calls must equal serial calls."""
    gpd = pytest.importorskip("geopandas")
    pytest.importorskip("pyarrow")
    pytest.importorskip("shapely")
    from shapely.geometry import box

    from imint.training.spatial_parquet import SpatialParquet

    # Build a tiny synthetic parquet with the bbox columns SpatialParquet
    # expects. 200 polygons in a 1000×1000 m grid, each 50 m square.
    polys = []
    rows = []
    for i in range(20):
        for j in range(10):
            west = i * 50
            south = j * 50
            geom = box(west, south, west + 50, south + 50)
            polys.append(geom)
            rows.append({
                "id": i * 10 + j,
                "_bbox_minx": west,
                "_bbox_miny": south,
                "_bbox_maxx": west + 50,
                "_bbox_maxy": south + 50,
            })
    import pandas as pd
    gdf = gpd.GeoDataFrame(rows, geometry=polys, crs="EPSG:3006")

    spatial_path = tmp_path / "synth_spatial.parquet"
    fallback_path = tmp_path / "synth.parquet"
    gdf.to_parquet(spatial_path, index=False, row_group_size=20)
    gdf.to_parquet(fallback_path, index=False)

    sp = SpatialParquet(str(spatial_path), fallback_path=str(fallback_path))

    # Generate disjoint query bboxes, half hit, half miss.
    queries = []
    for i in range(20):
        for j in range(10):
            queries.append({
                "west": i * 50 + 10, "south": j * 50 + 10,
                "east": i * 50 + 40, "north": j * 50 + 40,
            })

    serial = [len(sp.query(q)) for q in queries]

    with ThreadPoolExecutor(max_workers=16) as pool:
        parallel = list(pool.map(lambda q: len(sp.query(q)), queries))

    assert serial == parallel, (
        "SpatialParquet.query disagreed under concurrent access:\n"
        f"  serial   = {serial}\n"
        f"  parallel = {parallel}"
    )
