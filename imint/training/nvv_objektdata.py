"""NVV NMD objekthöjd/objekttäckning — Pirinen 2023 lager #6, #7, #8, #9.

Reads the four Naturvårdsverket NMD2018 tilläggsskikt rasters
(bush/tree height + cover) from a locally extracted ZIP archive and
clips them to an AOI bbox aligned to the NMD 10 m grid in EPSG:3006.

Source ZIPs (NMD2018, ~1 GB each):
    https://geodata.naturvardsverket.se/nedladdning/marktacke/NMD2018/
        Objekt_hojd_intervall_0_5_till_5_v1_3.zip          # bush height
        Objekt_hojd_intervall_5_till_45_v1_3.zip           # tree height
        Objekt_tackning_hojdintervall_0_5_till_5_v1_3.zip  # bush cover
        Objekt_tackning_hojdintervall_5_till_45_v1_3.zip   # tree cover

Encoding:
    Height layers (uint8, decimeters):
        0          no data
        5–50       bush height (0.5 – 5.0 m)
        50–450     tree height (5.0 – 45.0 m)
    Cover layers (uint8, percent):
        0          no data
        1–100      cover percentage in the height interval

The fetcher does not download zips — that is done by
``k8s/prefetch-nvv-aux-job.yaml``. This module only reads pre-extracted
rasters from disk.

Local path resolution: ``data_root`` arg → ``IMINT_NVV_AUX_DIR`` env →
``~/imint_data/nvv_aux``. Expected on-disk layout::

    {data_root}/objektdata/
        Objekt_hojd_intervall_0_5_till_5_v1_3.tif
        Objekt_hojd_intervall_5_till_45_v1_3.tif
        Objekt_tackning_hojdintervall_0_5_till_5_v1_3.tif
        Objekt_tackning_hojdintervall_5_till_45_v1_3.tif

License: CC0 (Naturvårdsverket open data)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np

from .skg_height import _to_nmd_grid_bounds


_DEFAULT_DATA_ROOT = "~/imint_data/nvv_aux"
_SUBDIR = "objektdata"

Kind = Literal["hojd", "tackning"]
HeightRange = Literal["0_5_5", "5_45"]


def _filename(kind: Kind, height_range: HeightRange) -> str:
    """Resolve canonical NVV filename for a given (kind, range) pair."""
    if kind == "hojd":
        # Objekt_hojd_intervall_0_5_till_5_v1_3.tif
        # Objekt_hojd_intervall_5_till_45_v1_3.tif
        suffix = {"0_5_5": "0_5_till_5", "5_45": "5_till_45"}[height_range]
        return f"Objekt_hojd_intervall_{suffix}_v1_3.tif"
    elif kind == "tackning":
        # Objekt_tackning_hojdintervall_0_5_till_5_v1_3.tif
        # Objekt_tackning_hojdintervall_5_till_45_v1_3.tif
        suffix = {"0_5_5": "0_5_till_5", "5_45": "5_till_45"}[height_range]
        return f"Objekt_tackning_hojdintervall_{suffix}_v1_3.tif"
    else:
        raise ValueError(f"unknown kind: {kind!r}")


def _cache_prefix(kind: Kind, height_range: HeightRange) -> str:
    return f"{kind}_{height_range}"


# ── Public API ───────────────────────────────────────────────────────────


def fetch_objektdata_tile(
    west: float,
    south: float,
    east: float,
    north: float,
    *,
    kind: Kind,
    height_range: HeightRange,
    size_px: int | tuple[int, int] = 1000,
    cache_dir: Path | None = None,
    data_root: Path | None = None,
) -> np.ndarray:
    """Fetch an Objekt-höjd or -täckning tile clipped to the EPSG:3006 bbox.

    Args:
        west, south, east, north: Bounding box in EPSG:3006 (meters).
        kind: ``"hojd"`` (height in dm) or ``"tackning"`` (cover %).
        height_range: ``"0_5_5"`` (bushes 0.5–5 m) or ``"5_45"`` (trees 5–45 m).
        size_px: Output raster size, single int or ``(h, w)`` tuple.
        cache_dir: Optional directory for ``.npy`` cache.
        data_root: Override for the local data root.

    Returns:
        NumPy array ``(H, W)`` uint8. Heights are in decimeters; covers are %.
    """
    h_px, w_px = (size_px, size_px) if isinstance(size_px, int) else size_px

    prefix = _cache_prefix(kind, height_range)
    if cache_dir is not None:
        cache_key = f"{prefix}_{int(west)}_{int(south)}_{int(east)}_{int(north)}.npy"
        cache_path = cache_dir / cache_key
        if cache_path.exists():
            cached = np.load(cache_path)
            if cached.shape == (h_px, w_px):
                return cached

    src_path = _resolve_source(kind, height_range, data_root)
    if src_path is None:
        expected = _resolve_root(data_root) / _SUBDIR / _filename(kind, height_range)
        raise FileNotFoundError(
            f"NVV objektdata GeoTIFF not found: {expected}\n"
            f"Extract the corresponding ZIP from "
            f"https://geodata.naturvardsverket.se/nedladdning/marktacke/NMD2018/ "
            f"into {expected.parent}"
        )

    arr = _read_window(src_path, west, south, east, north, h_px, w_px)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache_dir / cache_key.replace(".npy", "_tmp.npy")
        np.save(tmp, arr)
        Path(str(tmp)).rename(cache_path)

    return arr


# ── Convenience wrappers ─────────────────────────────────────────────────


def fetch_bush_height_tile(west, south, east, north, **kw):
    """Pirinen lager #6 — busk-höjd (0.5–5 m), uint8 dm."""
    return fetch_objektdata_tile(west, south, east, north,
                                  kind="hojd", height_range="0_5_5", **kw)


def fetch_bush_cover_tile(west, south, east, north, **kw):
    """Pirinen lager #7 — busk-täckning (0.5–5 m), uint8 %."""
    return fetch_objektdata_tile(west, south, east, north,
                                  kind="tackning", height_range="0_5_5", **kw)


def fetch_tree_height_tile_nvv(west, south, east, north, **kw):
    """Pirinen lager #8 — träd-höjd via NVV (5–45 m), uint8 dm.

    Alternative to ``imint.training.skg_height.fetch_height_tile``,
    sourced from NVV NMD2018 instead of Skogsstyrelsen ImageServer.
    """
    return fetch_objektdata_tile(west, south, east, north,
                                  kind="hojd", height_range="5_45", **kw)


def fetch_tree_cover_tile(west, south, east, north, **kw):
    """Pirinen lager #9 — träd-täckning (5–45 m), uint8 %."""
    return fetch_objektdata_tile(west, south, east, north,
                                  kind="tackning", height_range="5_45", **kw)


def fetch_objektdata_for_coords(
    coords: dict,
    *,
    kind: Kind,
    height_range: HeightRange,
    size_px: int = 1000,
    cache_dir: Path | None = None,
    data_root: Path | None = None,
) -> np.ndarray:
    """Fetch for a WGS84 bbox (snapped to NMD 10 m grid)."""
    projected = _to_nmd_grid_bounds(coords)
    return fetch_objektdata_tile(
        projected["west"], projected["south"],
        projected["east"], projected["north"],
        kind=kind, height_range=height_range,
        size_px=size_px, cache_dir=cache_dir, data_root=data_root,
    )


# ── Internal helpers ─────────────────────────────────────────────────────


def _resolve_root(data_root: Path | None) -> Path:
    if data_root is not None:
        return Path(data_root).expanduser()
    env = os.environ.get("IMINT_NVV_AUX_DIR")
    if env:
        return Path(env).expanduser()
    return Path(_DEFAULT_DATA_ROOT).expanduser()


def _resolve_source(
    kind: Kind, height_range: HeightRange, data_root: Path | None,
) -> Path | None:
    root = _resolve_root(data_root) / _SUBDIR
    candidate = root / _filename(kind, height_range)
    return candidate if candidate.is_file() else None


def _read_window(
    src_path: Path,
    west: float, south: float, east: float, north: float,
    h_px: int, w_px: int,
) -> np.ndarray:
    """Read AOI window; reproject if source CRS differs from EPSG:3006."""
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds as transform_from_bounds

    dst_transform = transform_from_bounds(west, south, east, north, w_px, h_px)
    dst = np.zeros((h_px, w_px), dtype=np.uint8)

    with rasterio.open(src_path) as src:
        sb = src.bounds
        if east <= sb.left or west >= sb.right or north <= sb.bottom or south >= sb.top:
            return dst
        window = from_bounds(west, south, east, north, transform=src.transform)
        data = src.read(
            1,
            window=window,
            out_shape=(h_px, w_px),
            resampling=Resampling.nearest,
            boundless=True,
            fill_value=0,
        )
        if src.crs is not None and src.crs.to_epsg() != 3006:
            reprojected = np.zeros_like(dst)
            reproject(
                source=data,
                destination=reprojected,
                src_transform=src.window_transform(window),
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs="EPSG:3006",
                resampling=Resampling.nearest,
            )
            data = reprojected

    return data.astype(np.uint8)
