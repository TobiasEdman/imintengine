"""Co-locate SLU NFI point plots onto the model's tile pixel grid.

Bridges :mod:`imint.training.slu_nfi` (the plot-table reader) and the tile
geometry (:class:`imint.training.tile_config.TileConfig` +
:func:`imint.training.tile_bbox.resolve_tile_bbox`): given a tile's ``.npz``
metadata, return the NFI plots that fall inside it — year-matched and
stamped with the pixel ``(row, col)`` they land on. Used by the validation
harness (sample predictions at plot pixels) and, later, by sparse training
targets.

Conventions — all EPSG:3006 (the repo's CRS):
  - bbox is ``(west, south, east, north)``; the pixel grid is north-up, so
    **row 0 is the NORTH edge** (the y-axis flips). We map points with the
    *same* affine the label/aux rasterizers use
    (``rasterio.transform.from_bounds`` →
    ``enrich_tiles_{lpis_mask,sks}.py``), so a plot lands on exactly the
    pixel a polygon would rasterize to — no half-pixel offset.
  - Temporal matching is **strict**: a plot's inventory ``Year`` must equal
    the tile year. Unlike :mod:`build_labels` we do NOT fall back to a
    default year — if the tile year is unknown we skip the tile rather than
    risk a year mismatch (CLAUDE.md: never mix spectral year and label year).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from imint.training.slu_nfi import plots_in_bbox
from imint.training.tile_bbox import resolve_tile_bbox
from imint.training.tile_config import TileConfig

if TYPE_CHECKING:
    from affine import Affine

_PIXEL_COLS = ("row", "col")


def tile_transform(bbox: dict, size_px: int) -> "Affine":
    """Affine for a tile bbox — identical to the label/aux rasterizers.

    ``rasterio.transform.from_bounds(west, south, east, north, w, h)`` is the
    exact call in ``enrich_tiles_{lpis_mask,sks}.py``, so points map onto the
    same grid as the rasterized polygons.
    """
    from rasterio.transform import from_bounds

    return from_bounds(
        bbox["west"], bbox["south"], bbox["east"], bbox["north"], size_px, size_px,
    )


def point_to_pixel(
    easting: float, northing: float, bbox: dict, size_px: int,
) -> tuple[int, int] | None:
    """EPSG:3006 ``(E, N)`` → ``(row, col)`` on the tile grid, or ``None``.

    ``row 0`` is the north edge (y flips). Cells are ``[inclusive, exclusive)``
    via floor, so a point exactly on the east or south edge maps past the last
    index and returns ``None`` (it belongs to the neighbouring tile). Uses
    ``rasterio.transform.rowcol`` (floor) for parity with the rasterizer.
    """
    from rasterio.transform import rowcol

    row, col = rowcol(tile_transform(bbox, size_px), easting, northing)
    row, col = int(row), int(col)
    if 0 <= row < size_px and 0 <= col < size_px:
        return row, col
    return None


def tile_year(npz_data) -> int | None:
    """Resolve a tile's inventory year, mirroring ``build_labels.py:224-245``.

    Order: ``year`` → ``lpis_year`` → modal year of ``dates`` (ties → most
    recent). Returns ``None`` when undeterminable — co-location must be
    year-exact, so (unlike build_labels) there is no default-year fallback.
    """
    def _get(key):
        try:
            return npz_data.get(key)
        except AttributeError:
            return npz_data[key] if key in npz_data else None

    for key in ("year", "lpis_year"):
        val = _get(key)
        if val is not None:
            return int(val)

    dates = _get("dates")
    if dates is not None:
        years: list[int] = []
        for d in dates:
            s = str(d)
            if len(s) >= 4:
                try:
                    years.append(int(s[:4]))
                except ValueError:
                    pass
        if years:
            counts = Counter(years)
            top = counts.most_common(1)[0][1]
            return max(y for y, c in counts.items() if c == top)
    return None


def colocate_plots(
    plots: pd.DataFrame,
    *,
    name: str,
    npz_data,
    tile: TileConfig,
    manifest_path: str | None = None,
    require_year_match: bool = True,
) -> pd.DataFrame:
    """Return the NFI plots inside one tile, stamped with ``(row, col)``.

    Args:
        plots: NFI frame from :func:`slu_nfi.load_nfi_plots` (needs
            ``Easting``/``Northing``/``Year``).
        name: tile name (used by the bbox filename fallback).
        npz_data: dict-like of the tile ``.npz`` (``easting``/``northing``/
            ``bbox_3006``/``year``/``lpis_year``/``dates``).
        tile: :class:`TileConfig` — defines ``size_px``.
        manifest_path: optional, forwarded to :func:`resolve_tile_bbox`.
        require_year_match: keep only plots whose ``Year`` equals the tile
            year; skip the whole tile if the tile year is unknown (default —
            the safe path).

    Returns:
        A copy of the matching rows with added int columns ``row``, ``col``.
        Empty (original columns + ``row``/``col``) when nothing co-locates.
    """
    empty = plots.iloc[0:0].assign(
        row=pd.Series(dtype="int64"), col=pd.Series(dtype="int64"),
    )

    bbox = resolve_tile_bbox(
        name=name, tile=tile, npz_data=npz_data, manifest_path=manifest_path,
    )
    if bbox is None:
        return empty

    sub = plots
    if require_year_match:
        year = tile_year(npz_data)
        if year is None:
            return empty
        sub = sub[sub["Year"] == year]

    sub = plots_in_bbox(sub, (bbox["west"], bbox["south"], bbox["east"], bbox["north"]))
    if sub.empty:
        return empty

    from rasterio.transform import rowcol

    rows, cols = rowcol(
        tile_transform(bbox, tile.size_px),
        sub["Easting"].to_numpy(),
        sub["Northing"].to_numpy(),
    )
    rows = np.asarray(rows, dtype="int64")
    cols = np.asarray(cols, dtype="int64")
    inbounds = (rows >= 0) & (rows < tile.size_px) & (cols >= 0) & (cols < tile.size_px)

    out = sub.loc[inbounds].copy()
    out["row"] = rows[inbounds]
    out["col"] = cols[inbounds]
    return out


def build_plot_index(
    tile_npz_paths,
    plots: pd.DataFrame,
    tile: TileConfig,
    *,
    manifest_path: str | None = None,
    require_year_match: bool = True,
) -> pd.DataFrame:
    """Co-locate ``plots`` against many tile ``.npz`` files → one frame.

    Concatenates :func:`colocate_plots` across tiles with an added
    ``tile_name`` column. Tiles whose bbox/year can't be resolved contribute
    nothing. The caller should persist the result to parquet — this is the
    cached plot→pixel index; never recompute it per epoch.
    """
    frames: list[pd.DataFrame] = []
    for path in tile_npz_paths:
        path = Path(path)
        with np.load(path, allow_pickle=True) as npz:
            data = {k: npz[k] for k in npz.files}
        got = colocate_plots(
            plots, name=path.stem, npz_data=data, tile=tile,
            manifest_path=manifest_path, require_year_match=require_year_match,
        )
        if not got.empty:
            got.insert(0, "tile_name", path.stem)
            frames.append(got)

    if frames:
        return pd.concat(frames, ignore_index=True)
    return plots.iloc[0:0].assign(
        tile_name=pd.Series(dtype="object"),
        row=pd.Series(dtype="int64"),
        col=pd.Series(dtype="int64"),
    )
