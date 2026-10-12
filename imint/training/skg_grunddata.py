"""Skogsstyrelsen Skogliga grunddata + Trädhöjd, read from the open-data mirror.

Forestry variables on the NMD 10 m grid, as NumPy arrays:

    Variable         Unit       Typical range   Mirror product
    ───────────────  ─────────  ──────────────  ─────────────────────────────
    volume           m³sk/ha    0 – 700         sks_Volym_omdrev{1,2,3}
    basal_area       m²/ha      0 – 60          sks_Grundyta_omdrev{1,2,3}
    diameter         cm         0 – 60          sks_Medeldiameter_omdrev{1,2,3}
    tree_height      dm (m out) 0 – 350         Trädhöjd, 10 m derivative

Until 2026 these came from Skogsstyrelsen's ArcGIS ImageServer through its
public kartportal proxy. The proxy was retired; the same products are
published as per-county GeoTIFF zips and mirrored to the PVC
(``imint.training.skg_open``, backlog IM-022). Reading the mirror with the
newest scanning round (omdrev) per pixel reproduces the proxy-fetched
arrays exactly, so tiles enriched before and after the switch agree.

License: CC0 (Skogsstyrelsen open data).

Typical usage::

    from imint.training.skg_grunddata import fetch_volume_tile
    volume = fetch_volume_tile(west, south, east, north, size_px=512)
    # volume.shape == (512, 512), dtype float32, unit: m³sk/ha
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import skg_open

# Public channel name -> mirror variable.
_VARIABLE = {
    "volume": "volym",
    "basal_area": "grundyta",
    "diameter": "medeldiameter",
}


# ── Public API ───────────────────────────────────────────────────────────

def fetch_volume_tile(
    west: float,
    south: float,
    east: float,
    north: float,
    *,
    size_px: int | tuple[int, int] = 256,
    cache_dir: Path | None = None,
) -> np.ndarray:
    """Timber volume (m³sk/ha), ``(H, W)`` float32, nodata → 0."""
    return fetch_grunddata_tile(
        west, south, east, north,
        variable="volume", size_px=size_px, cache_dir=cache_dir,
    )


def fetch_basal_area_tile(
    west: float,
    south: float,
    east: float,
    north: float,
    *,
    size_px: int | tuple[int, int] = 256,
    cache_dir: Path | None = None,
) -> np.ndarray:
    """Basal area / grundyta (m²/ha), ``(H, W)`` float32, nodata → 0."""
    return fetch_grunddata_tile(
        west, south, east, north,
        variable="basal_area", size_px=size_px, cache_dir=cache_dir,
    )


def fetch_diameter_tile(
    west: float,
    south: float,
    east: float,
    north: float,
    *,
    size_px: int | tuple[int, int] = 256,
    cache_dir: Path | None = None,
) -> np.ndarray:
    """Basal-area-weighted mean diameter / medeldiameter (cm), ``(H, W)``
    float32, nodata → 0."""
    return fetch_grunddata_tile(
        west, south, east, north,
        variable="diameter", size_px=size_px, cache_dir=cache_dir,
    )


def fetch_tree_height_tile(
    west: float,
    south: float,
    east: float,
    north: float,
    *,
    size_px: int | tuple[int, int] = 256,
    cache_dir: Path | None = None,
) -> np.ndarray:
    """Tree height (trädhöjd) in **metres**, ``(H, W)`` float32, nodata → 0.

    The mirror and the ``.npy`` cache hold decimetres (the product's unit);
    the conversion happens here, after the cache, as it always did.
    """
    dm = _cached(
        "tree_height", west, south, east, north, size_px, cache_dir,
        lambda: skg_open.read_tradhojd(west, south, east, north, size_px=size_px),
    )
    return dm.astype(np.float32) / 10.0


def fetch_grunddata_tile(
    west: float,
    south: float,
    east: float,
    north: float,
    *,
    variable: str,
    size_px: int | tuple[int, int] = 256,
    cache_dir: Path | None = None,
) -> np.ndarray:
    """One grunddata variable (``volume``, ``basal_area`` or ``diameter``)
    on the NMD 10 m grid.

    Args:
        west, south, east, north: Bounding box in EPSG:3006 (metres), on
            the 10 m lattice, with ``extent == size_px × 10 m``.
        variable: Channel name.
        size_px: Output size — int for square or ``(H, W)`` tuple.
        cache_dir: Optional ``.npy`` cache directory. The cache key is
            ``<variable>_<west>_<south>_<east>_<north>.npy``.

    Returns:
        ``(H, W)`` float32 array. Nodata pixels are 0.
    """
    if variable not in _VARIABLE:
        raise ValueError(f"unknown grunddata variable {variable!r}; "
                         f"expected one of {tuple(_VARIABLE)}")
    return _cached(
        variable, west, south, east, north, size_px, cache_dir,
        lambda: skg_open.read_grunddata(
            _VARIABLE[variable], west, south, east, north, size_px=size_px),
    )


# ── Cache contract (shared with slu_markfukt) ───────────────────────────

def _cached(prefix, west, south, east, north, size_px, cache_dir, read):
    """``<prefix>_<w>_<s>_<e>_<n>.npy`` under ``cache_dir``; a cached array of
    another shape is ignored and overwritten; writes are tmp + rename.

    The request is validated against the 10 m lattice *before* the cache
    is consulted, so a stale array can never satisfy a malformed request."""
    h_px, w_px = (size_px, size_px) if isinstance(size_px, int) else size_px
    skg_open.check_bbox(west, south, east, north, h_px, w_px)
    if cache_dir is None:
        return read()
    cache_path = Path(cache_dir) / (
        f"{prefix}_{int(west)}_{int(south)}_{int(east)}_{int(north)}.npy")
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape == (h_px, w_px):
            return cached
    data = read()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_name(cache_path.stem + "_tmp.npy")
    np.save(tmp, data)
    tmp.rename(cache_path)
    return data
