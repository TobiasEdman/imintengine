"""NVV Markfuktighetsindex (Soil Moisture Index) — Pirinen 2023 lager #3.

Reads the Naturvårdsverket Markfuktighetsindex_NMD GeoTIFF from a locally
extracted ZIP archive and clips it to an AOI bbox aligned to the NMD 10 m
grid in EPSG:3006.

Source ZIP (Sweden-wide, 12 GB) or per-region (del1–9, ~3 GB each):
    https://geodata.naturvardsverket.se/nedladdning/marktacke/NMD2018/
    Markfuktighetsindex_NMD_Sverige.zip
    Markfuktighetsindex_NMD_del{1..9}.zip

Inside each ZIP: GeoTIFF in EPSG:3006, 10 m, uint8 with values:
    0      no data / outside Sweden
    1-100  soil moisture (low → high)
    255    nodata fill

The fetcher does not download zips — that is the responsibility of
``k8s/prefetch-nvv-aux-job.yaml`` (or manual one-time download for local
development). This module only reads pre-extracted rasters from disk.

Local path resolution (priority order):
    1. ``data_root`` argument (if given)
    2. ``IMINT_NVV_AUX_DIR`` environment variable
    3. ``~/imint_data/nvv_aux``

Expected on-disk layout::

    {data_root}/markfuktighet/
        Markfuktighetsindex_NMD_Sverige.tif        # full Sweden, OR
        Markfuktighetsindex_NMD_del{1..9}.tif      # regional parts

If multiple files are present they are read as a virtual mosaic.

License: CC0 (Naturvårdsverket open data)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .skg_height import _to_nmd_grid_bounds


_DEFAULT_DATA_ROOT = "~/imint_data/nvv_aux"
_SUBDIR = "markfuktighet"
_GLOB_PATTERN = "Markfuktighetsindex_NMD*.tif"


def fetch_smi_tile(
    west: float,
    south: float,
    east: float,
    north: float,
    *,
    size_px: int | tuple[int, int] = 1000,
    cache_dir: Path | None = None,
    data_root: Path | None = None,
) -> np.ndarray:
    """Fetch a Markfuktighetsindex tile clipped to the given EPSG:3006 bbox.

    Args:
        west, south, east, north: Bounding box in EPSG:3006 (meters).
        size_px: Output raster size. Single int or ``(h, w)`` tuple.
        cache_dir: Optional directory for ``.npy`` cache.
        data_root: Override for the local data root. Falls back to env var
            ``IMINT_NVV_AUX_DIR`` then ``~/imint_data/nvv_aux``.

    Returns:
        NumPy array of shape ``(H, W)``, dtype uint8, values 0–100 (255 → 0).
    """
    h_px, w_px = (size_px, size_px) if isinstance(size_px, int) else size_px

    if cache_dir is not None:
        cache_key = f"smi_{int(west)}_{int(south)}_{int(east)}_{int(north)}.npy"
        cache_path = cache_dir / cache_key
        if cache_path.exists():
            cached = np.load(cache_path)
            if cached.shape == (h_px, w_px):
                return cached

    sources = _resolve_sources(data_root)
    if not sources:
        raise FileNotFoundError(
            f"No Markfuktighetsindex GeoTIFF found under "
            f"{_resolve_root(data_root) / _SUBDIR}. "
            f"Extract Markfuktighetsindex_NMD_Sverige.zip (or del1-9) from "
            f"https://geodata.naturvardsverket.se/nedladdning/marktacke/NMD2018/ "
            f"into that directory first."
        )

    arr = _read_window(sources, west, south, east, north, h_px, w_px)

    # Map nodata fill (255) to 0 so consumers can ignore it cleanly
    arr = np.where(arr == 255, 0, arr).astype(np.uint8)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache_dir / cache_key.replace(".npy", "_tmp.npy")
        np.save(tmp, arr)
        Path(str(tmp)).rename(cache_path)

    return arr


def fetch_smi_for_coords(
    coords: dict,
    *,
    size_px: int = 1000,
    cache_dir: Path | None = None,
    data_root: Path | None = None,
) -> np.ndarray:
    """Fetch SMI for a WGS84 bbox (snapped to 10 m EPSG:3006 grid)."""
    projected = _to_nmd_grid_bounds(coords)
    return fetch_smi_tile(
        projected["west"], projected["south"],
        projected["east"], projected["north"],
        size_px=size_px,
        cache_dir=cache_dir,
        data_root=data_root,
    )


# ── Internal helpers ─────────────────────────────────────────────────────


def _resolve_root(data_root: Path | None) -> Path:
    if data_root is not None:
        return Path(data_root).expanduser()
    env = os.environ.get("IMINT_NVV_AUX_DIR")
    if env:
        return Path(env).expanduser()
    return Path(_DEFAULT_DATA_ROOT).expanduser()


def _resolve_sources(data_root: Path | None) -> list[Path]:
    root = _resolve_root(data_root) / _SUBDIR
    if not root.is_dir():
        return []
    return sorted(root.glob(_GLOB_PATTERN))


def _read_window(
    sources: list[Path],
    west: float, south: float, east: float, north: float,
    h_px: int, w_px: int,
) -> np.ndarray:
    """Read AOI window from one or more source GeoTIFFs (virtual mosaic).

    Iterates through sources; returns the first one that overlaps the bbox.
    For multi-file regional splits this is fine since del1-9 are disjoint.
    If no source overlaps, returns zeros.
    """
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds as transform_from_bounds

    dst = np.zeros((h_px, w_px), dtype=np.uint8)
    dst_transform = transform_from_bounds(west, south, east, north, w_px, h_px)
    written = False

    for src_path in sources:
        with rasterio.open(src_path) as src:
            sb = src.bounds
            if east <= sb.left or west >= sb.right or north <= sb.bottom or south >= sb.top:
                continue
            try:
                window = from_bounds(west, south, east, north, transform=src.transform)
                data = src.read(
                    1,
                    window=window,
                    out_shape=(h_px, w_px),
                    resampling=Resampling.nearest,
                    boundless=True,
                    fill_value=0,
                )
                # If source CRS differs from EPSG:3006, reproject
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
                # Merge: keep first non-zero pixels (regional splits don't overlap)
                mask = (dst == 0) & (data != 0)
                dst[mask] = data[mask]
                written = True
            except Exception:
                continue

    if not written:
        return dst
    return dst
