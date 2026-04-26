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

Thread safety
-------------
``pyarrow.parquet.ParquetFile`` holds an open file descriptor and
mutable cursor state. Two threads calling ``read_row_groups()`` on the
same handle can race and return data from the wrong row groups (this
exact bug produced misaligned tile labels in the multi-threaded
build-labels pipeline, 2026-04). To make ``SpatialParquet`` safe to
share across threads we keep one ``ParquetFile`` (and one fallback
``GeoDataFrame``) **per thread** via ``threading.local()``. The
configuration (``path``, ``fallback_path``) and the immutable
row-group bbox metadata are shared.
"""
from __future__ import annotations

import os
import threading
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

    The instance is safe to share across threads. Pyarrow / geopandas
    handles are stored in ``threading.local()`` so each thread gets
    its own ``ParquetFile`` cursor.
    """

    def __init__(self, path: str, fallback_path: str | None = None):
        self.path = path
        self.fallback_path = fallback_path
        # Immutable, computed once on first open from any thread.
        self._rg_bboxes: list[tuple[float, float, float, float]] | None = None
        self._rg_lock = threading.Lock()
        # Per-thread mutable handles.
        self._tls = threading.local()

    # ── per-thread handle accessors ──────────────────────────────────────

    def _ensure_open(self) -> None:
        """Open the parquet file in the calling thread (if not yet open).

        Computes ``self._rg_bboxes`` once across all threads (under a
        lock) so we don't redo the metadata scan per worker.
        """
        # Already open in this thread → fast path.
        if getattr(self._tls, "parq", None) is not None:
            return
        if getattr(self._tls, "fallback_gdf", None) is not None:
            return

        if os.path.exists(self.path):
            import pyarrow.parquet as pq
            self._tls.parq = pq.ParquetFile(self.path)
            self._tls.fallback_gdf = None
            with self._rg_lock:
                if self._rg_bboxes is None:
                    self._rg_bboxes = self._extract_row_group_bboxes(
                        self._tls.parq,
                    )
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
            # Per-thread fallback — geopandas filtering uses pandas, which
            # holds internal numpy buffers that are read-safe but better
            # not to share across threads either.
            self._tls.fallback_gdf = gpd.read_parquet(self.fallback_path)
            self._tls.parq = None
            return

        raise FileNotFoundError(
            f"Neither {self.path} nor fallback {self.fallback_path} exists."
        )

    @staticmethod
    def _extract_row_group_bboxes(parq) -> list[tuple[float, float, float, float]]:
        """Read min/max statistics for the bbox columns from each row group.

        Pure metadata read — no row data is materialized, so calling
        this from a single thread once is cheap (~milliseconds for our
        ~115 row-group LPIS files).
        """
        md = parq.metadata
        schema = parq.schema_arrow
        col_names = schema.names
        try:
            idx_minx = col_names.index("_bbox_minx")
            idx_miny = col_names.index("_bbox_miny")
            idx_maxx = col_names.index("_bbox_maxx")
            idx_maxy = col_names.index("_bbox_maxy")
        except ValueError as e:
            raise ValueError(
                f"missing expected bbox columns "
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

        Thread-safe: each calling thread gets its own pyarrow handle
        via ``threading.local()``.
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
        fallback_gdf = getattr(self._tls, "fallback_gdf", None)
        if fallback_gdf is not None:
            b = fallback_gdf.geometry.bounds
            mask = ~(
                (b["maxx"] < tw) | (b["minx"] > te)
                | (b["maxy"] < ts) | (b["miny"] > tn)
            )
            return fallback_gdf[mask].copy()

        # Fast path — row-group-filtered read
        parq = getattr(self._tls, "parq", None)
        assert parq is not None, "_ensure_open did not initialise per-thread parq"
        relevant = self._relevant_row_groups(tile_bbox_t)
        if not relevant:
            empty = parq.schema_arrow.empty_table()
            return gpd.GeoDataFrame.from_arrow(empty)

        table = parq.read_row_groups(relevant)
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
