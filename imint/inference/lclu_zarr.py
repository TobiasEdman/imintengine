"""TESSERA embeddings for an EPSG:3006 window, read from the published Zarr store.

geotessera publishes dataset 1.0 "vultr" — the dataset of the training cohort
and of the served R4 (``EMBEDDING``) — as NPY tiles and as one Zarr v3 store,
``zarr/v1`` on Source Cooperative. The store has one group per UTM zone
(``utm33``, ``utm34``, …) holding ``embeddings`` int8 (year, 128, y, x),
``scales`` float32 (year, y, x; NaN = water, +inf = no data), ``time`` (the
years) and ``x``/``y`` pixel centres on the zone's 10 m lattice. A 0.1° tile
belongs to the zone of its centre longitude.

The int8 vectors are the tiles' own, copied without resampling, but the store
builder rounds every tile's UTM origin to the zone lattice (up to half a
pixel) and lets the last-written tile win where neighbouring tiles overlap.
A window read here is therefore not the NPY or COG path's window bit for bit,
and it carries its own identity, ``ZARR_EMBEDDING``, never ``EMBEDDING``.
``scripts/compare_tessera_zarr.py`` measures the difference.

Only the native window a grid needs is read, plus a bilinear halo, and every
read is kept on disk under ``cache_dir``. Resampling and merging reuse the
code of the existing paths: ``merge="serving"`` is the rule of
``lclu_cog.assemble_cog_embedding`` and ``prepare_lclu.assemble_embedding``,
``merge="training"`` that of ``enrich_tiles_tessera._fetch_tessera_for_tile``.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path

import numpy as np

from imint.inference.lclu import InputUnavailable

STORE_URL = "https://data.source.coop/tessera/tessera/zarr/v1"
ZARR_EMBEDDING = {"producer": "tessera-zarr", "store": STORE_URL,
                  "dataset": "v1", "variant": "vultr"}
BANDS = 128
PIXEL = 10.0
HALO = 2
MERGE_RULES = ("serving", "training")


@dataclass(frozen=True)
class Zone:
    number: int
    crs: str
    origin_x: float          # west edge of column 0
    origin_y: float          # north edge of row 0
    years: tuple[int, ...]
    shape: tuple[int, int]   # (rows, columns)
    embeddings: object       # zarr array (year, 128, y, x) int8
    scales: object           # zarr array (year, y, x) float32


def open_store(url: str):
    """Open a store read-only: an http(s)/s3 URL through fsspec, or a local path."""
    import zarr
    from zarr.storage import FsspecStore, LocalStore

    store = (FsspecStore.from_url(url, read_only=True) if "://" in url
             else LocalStore(url, read_only=True))
    return zarr.open_group(store, mode="r", zarr_format=3)


def tile_zones(west: float, south: float, east: float, north: float) -> list[int]:
    """UTM zones of the 0.1° tiles covering a lon/lat box (each tile: its centre's zone)."""
    if south < 0:
        raise ValueError("Only northern-hemisphere zones are supported")
    centres = [(k + 0.5) / 10 for k in range(math.floor(west * 10), math.floor(east * 10) + 1)]
    return sorted({int(math.floor((c + 180) / 6)) + 1 for c in centres})


def open_zone(root, number: int) -> Zone:
    """The zone group's arrays and lattice, after checking the store layout."""
    name = f"utm{number:02d}"
    if name not in root:
        raise InputUnavailable(f"TESSERA Zarr store has no zone {name}")
    group = root[name]
    embeddings, scales = group["embeddings"], group["scales"]
    years = tuple(int(v) for v in group["time"][:])
    t, bands, rows, cols = embeddings.shape
    if (embeddings.dtype != np.int8 or bands != BANDS or scales.dtype != np.float32
            or scales.shape != (t, rows, cols) or len(years) != t):
        raise ValueError(f"Unexpected TESSERA Zarr layout in {name}")
    x, y = group["x"][:2], group["y"][:2]
    if not (np.isclose(x[1] - x[0], PIXEL) and np.isclose(y[0] - y[1], PIXEL)):
        raise ValueError(f"{name} is not a 10 m north-up lattice")
    crs = f"EPSG:{32600 + number}"
    if group.attrs.get("proj:code", crs) != crs:
        raise ValueError(f"{name} declares {group.attrs['proj:code']}, expected {crs}")
    return Zone(number, crs, float(x[0]) - PIXEL / 2, float(y[0]) + PIXEL / 2,
                years, (rows, cols), embeddings, scales)


def read_native(zone: Zone, bounds_3006: tuple, year: int, cache: Path):
    """int8 (128, h, w), scales (h, w) and affine of the zone window under a 3006 box.

    None when the box lies outside the zone's lattice. The read is kept in
    ``cache`` and served from there on the next call.
    """
    from rasterio.transform import Affine
    from rasterio.warp import transform_bounds

    if year not in zone.years:
        raise InputUnavailable(f"TESSERA Zarr store has no {year} in utm{zone.number:02d}")
    west, south, east, north = transform_bounds("EPSG:3006", zone.crs, *bounds_3006,
                                                densify_pts=41)
    rows, cols = zone.shape
    c0 = max(0, math.floor((west - zone.origin_x) / PIXEL) - HALO)
    c1 = min(cols, math.ceil((east - zone.origin_x) / PIXEL) + HALO)
    r0 = max(0, math.floor((zone.origin_y - north) / PIXEL) - HALO)
    r1 = min(rows, math.ceil((zone.origin_y - south) / PIXEL) + HALO)
    if c1 <= c0 or r1 <= r0:
        return None
    path = cache / f"utm{zone.number:02d}_{year}_r{r0}-{r1}_c{c0}-{c1}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as saved:
            q, s = saved["embeddings"], saved["scales"]
    else:
        t = zone.years.index(year)
        q = np.asarray(zone.embeddings[t, :, r0:r1, c0:c1])
        s = np.asarray(zone.scales[t, r0:r1, c0:c1])
        cache.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp")
        np.savez_compressed(tmp, embeddings=q, scales=s)  # numpy appends .npz
        os.replace(tmp.with_suffix(".tmp.npz"), path)
    if q.shape != (BANDS, r1 - r0, c1 - c0) or q.dtype != np.int8 or s.shape != q.shape[1:]:
        raise ValueError(f"Cached or stored window {path.name} has the wrong shape")
    transform = Affine(PIXEL, 0.0, zone.origin_x + c0 * PIXEL,
                       0.0, -PIXEL, zone.origin_y - r0 * PIXEL)
    return q, s, transform, {"zone": zone.number, "rows": [r0, r1], "cols": [c0, c1]}


def read_embedding(store_url: str, bounds_3006: tuple, size_px: int, year: int, *,
                   cache_dir: Path, merge: str = "serving") -> tuple[np.ndarray, np.ndarray, dict]:
    """(128, size, size) float16 embedding, (size, size) validity and provenance.

    ``bounds_3006`` is (west, south, east, north). Raises InputUnavailable when
    the year is absent or any pixel of the window has no data in every zone
    that could hold it (a gap, not water).
    """
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject, transform_bounds
    from scripts.enrich_tiles_tessera import _reproject_tessera_to_tile

    if merge not in MERGE_RULES:
        raise ValueError(f"merge must be one of {MERGE_RULES}")
    west, south, east, north = bounds_3006
    box = {"west": west, "south": south, "east": east, "north": north}
    dst = from_bounds(west, south, east, north, size_px, size_px)
    cache = Path(cache_dir) / hashlib.sha256(store_url.encode()).hexdigest()[:16]

    def to_grid(values: np.ndarray, transform, crs) -> np.ndarray:
        out = np.zeros((size_px, size_px), np.float32)
        reproject(values.astype(np.float32), out, src_transform=transform, src_crs=crs,
                  dst_transform=dst, dst_crs="EPSG:3006", resampling=Resampling.bilinear)
        return out

    root = open_store(store_url)
    merged = np.zeros((BANDS, size_px, size_px), np.float16)
    valid = np.zeros((size_px, size_px), bool)
    gap = np.ones((size_px, size_px), bool)   # no zone can rule the pixel out as a gap
    zones = []
    for number in tile_zones(*transform_bounds("EPSG:3006", "EPSG:4326", *bounds_3006,
                                               densify_pts=41)):
        zone = open_zone(root, number)
        native = read_native(zone, bounds_3006, year, cache)
        if native is None:
            continue
        q, scales, transform, window = native
        finite = np.isfinite(scales)
        arr = q.astype(np.float32) * np.where(finite, scales, 0)[None]
        chunk = _reproject_tessera_to_tile(arr.transpose(1, 2, 0), transform, zone.crs, box,
                                           size_px)
        take = np.any(chunk != 0, axis=0)
        if merge == "serving":
            take &= to_grid(np.any(arr != 0, axis=0), transform, zone.crs) >= 1 - 1e-6
        merged[:, take] = chunk[:, take]
        valid |= take
        gap &= to_grid(np.isposinf(scales), transform, zone.crs) > 0
        zones.append({**window, "water_px": int(np.isnan(scales).sum()),
                      "missing_px": int(np.isposinf(scales).sum())})
    gap &= ~valid
    if gap.any():
        raise InputUnavailable(f"TESSERA Zarr has no data for {int(gap.sum())} pixels "
                               f"of the window in {year}")
    return merged, valid, {"embedding": ZARR_EMBEDDING, "store": store_url, "year": year,
                           "merge": merge, "zones": zones}
