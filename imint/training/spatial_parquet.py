"""
imint/training/spatial_parquet.py — Bbox-filtered GeoParquet reader.

Wraps a GeoParquet file that has been preprocessed by
``scripts/preprocess_sks_lpis_spatial.py`` so it carries per-polygon
bbox columns (``_bbox_minx`` / ``_bbox_miny`` / ``_bbox_maxx`` /
``_bbox_maxy``) in small row groups (10 000 rows) sorted by a coarse
spatial grid.

At query time we:
  1. Read pyarrow's per-row-group min/max statistics for the bbox
     columns (essentially free — read from file metadata).
  2. Intersect those row-group bboxes with the query tile bbox to
     decide which row groups to read.
  3. Pull only those row groups (typically 1-2 out of hundreds),
     then apply an exact per-row bbox filter to drop polygons that
     happen to share a row group but don't overlap the tile.

Memory footprint at query time: O(polygons in 1-2 row groups × bbox
size) ≈ a few MB, independent of total file size. No full-file load.

Opens file lazily on first query; holds a ``pyarrow.parquet.ParquetFile``
handle which is cheap (just an open FD).
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import geopandas as gpd


class SpatialParquet:
    """Row-group-filtered reader for a spatially-partitioned GeoParquet.

    Args:
        path: Path to a parquet preprocessed by
            ``preprocess_sks_lpis_spatial.py``. Must contain the bbox
            columns ``_bbox_minx, _bbox_miny, _bbox_maxx, _bbox_maxy``.
        fallback_path: Optional path to the un-indexed original parquet.
            If the spatial version is missing, we fall back to loading
            the full file on first query and filtering in-memory.
            Emits a warning so the caller sees the perf hit.
    """

    def __init__(self, path: str, fallback_path: str | None = None):
        self.path = path
        self.fallback_path = fallback_path
        self._parq = None            # pyarrow.parquet.ParquetFile
        self._rg_bboxes = None       # list[tuple[minx, miny, maxx, maxy]]
        self._fallback_gdf = None    # loaded only if spatial file missing

    def _ensure_open(self) -> None:
        if self._parq is not None or self._fallback_gdf is not None:
            return

        if os.path.exists(self.path):
            import pyarrow.parquet as pq
            self._parq = pq.ParquetFile(self.path)
            self._rg_bboxes = self._extract_row_group_bboxes()
            return

        # Spatial parquet missing — fall back to slow path
        if self.fallback_path and os.path.exists(self.fallback_path):
            import warnings
            warnings.warn(
                f"Spatial parquet {self.path} missing; falling back to "
                f"{self.fallback_path} (full-file load, high memory).",
                RuntimeWarning,
            )
            import geopandas as gpd
            self._fallback_gdf = gpd.read_parquet(self.fallback_path)
            return

        raise FileNotFoundError(
            f"Neither {self.path} nor fallback {self.fallback_path} exists."
        )

    def _extract_row_group_bboxes(self) -> list[tuple[float, float, float, float]]:
        """Read min/max statistics for the bbox columns from each row group.

        Returns a list aligned with ``self._parq.num_row_groups``.
        Each entry is the axis-aligned bbox that encloses all polygon
        bboxes in that row group — i.e. the union of per-polygon
        bboxes restricted to the rows in that group.
        """
        assert self._parq is not None
        md = self._parq.metadata
        # Find column indices by name
        schema = self._parq.schema_arrow
        col_names = schema.names
        try:
            idx_minx = col_names.index("_bbox_minx")
            idx_miny = col_names.index("_bbox_miny")
            idx_maxx = col_names.index("_bbox_maxx")
            idx_maxy = col_names.index("_bbox_maxy")
        except ValueError as e:
            raise ValueError(
                f"{self.path} missing expected bbox columns "
                f"(_bbox_minx / _bbox_miny / _bbox_maxx / _bbox_maxy). "
                f"Reprocess via preprocess_sks_lpis_spatial.py."
            ) from e

        bboxes: list[tuple[float, float, float, float]] = []
        for rg in range(md.num_row_groups):
            rg_md = md.row_group(rg)
            minx = rg_md.column(idx_minx).statistics.min
            maxx = rg_md.column(idx_maxx).statistics.max
            miny = rg_md.column(idx_miny).statistics.min
            maxy = rg_md.column(idx_maxy).statistics.max
            bboxes.append((float(minx), float(miny), float(maxx), float(maxy)))
        return bboxes

    def _relevant_row_groups(
        self, tile_bbox: tuple[float, float, float, float],
    ) -> list[int]:
        """Return row group indices whose bounding box intersects tile_bbox."""
        assert self._rg_bboxes is not None
        tw, ts, te, tn = tile_bbox
        hits = []
        for i, (minx, miny, maxx, maxy) in enumerate(self._rg_bboxes):
            if maxx < tw or minx > te or maxy < ts or miny > tn:
                continue  # no overlap
            hits.append(i)
        return hits

    def query(self, tile_bbox: dict | tuple) -> "gpd.GeoDataFrame":
        """Return a GeoDataFrame of polygons whose bbox intersects tile_bbox.

        Args:
            tile_bbox: Either a dict with keys ``west/south/east/north``
                or a 4-tuple ``(west, south, east, north)`` in the same
                CRS the parquet was written in (EPSG:3006 for our SKS/LPIS).

        Returns:
            GeoDataFrame. Empty (with correct CRS) when no polygons overlap.
        """
        import geopandas as gpd

        if isinstance(tile_bbox, dict):
            tw = tile_bbox["west"]
            ts = tile_bbox["south"]
            te = tile_bbox["east"]
            tn = tile_bbox["north"]
        else:
            tw, ts, te, tn = tile_bbox
        tile_bbox_t = (tw, ts, te, tn)

        self._ensure_open()

        # Slow path — spatial parquet missing
        if self._fallback_gdf is not None:
            b = self._fallback_gdf.geometry.bounds
            mask = ~(
                (b["maxx"] < tw) | (b["minx"] > te)
                | (b["maxy"] < ts) | (b["miny"] > tn)
            )
            return self._fallback_gdf[mask].copy()

        # Fast path — row-group-filtered read
        assert self._parq is not None
        relevant = self._relevant_row_groups(tile_bbox_t)
        if not relevant:
            # Return empty GeoDataFrame with correct schema
            empty = self._parq.schema_arrow.empty_table()
            return gpd.GeoDataFrame.from_arrow(empty)

        table = self._parq.read_row_groups(relevant)
        gdf = gpd.GeoDataFrame.from_arrow(table)

        # Exact per-row bbox filter (row group may contain polygons
        # outside the tile)
        b = gdf.geometry.bounds
        mask = ~(
            (b["maxx"] < tw) | (b["minx"] > te)
            | (b["maxy"] < ts) | (b["miny"] > tn)
        )
        return gdf[mask].copy()

    def __repr__(self) -> str:
        return f"SpatialParquet({self.path!r})"


@lru_cache(maxsize=16)
def get_spatial_parquet(path: str, fallback_path: str | None = None) -> SpatialParquet:
    """Cache SpatialParquet instances by path (keeps file handle open)."""
    return SpatialParquet(path, fallback_path)
