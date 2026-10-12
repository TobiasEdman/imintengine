"""Local mirror of Skogsstyrelsen's open GeoTIFF data (backlog IM-022).

Skogsstyrelsen retired its public kartportal ImageServer proxies in 2026.
The same products are published as per-county GeoTIFF zips at
``https://geodpags.skogsstyrelsen.se/geodataport/data/``. Those zips are
deflate-compressed strip TIFFs (one row per strip), so they cannot be read
by window over HTTP; ``scripts/stage_skg_open.py`` (run as
``k8s/skg-open-stage-job.yaml``) downloads them once to the PVC, re-encodes
each raster as a tiled Cloud-Optimized GeoTIFF and builds one VRT mosaic
per product. This module is the only reader of that mirror.

Mirror layout (``SKG_OPEN_DIR``, default ``/data/skg_open``)::

    MANIFEST.json                       source URLs, sha256, bytes, dates, git sha
    raw/<zip as published>              the downloaded zips, unchanged
    grunddata/omdrev{1,2,3}/<var>_<län>.tif   COG per county (int16, nodata -1)
    grunddata/<var>_omdrev{1,2,3}.vrt   county mosaic per variable and omdrev
    markfukt/markfukt_<län>.tif         COG per county (2 m uint16, nodata 65535)
    markfukt/markfukt.vrt
    tradhojd/tradhojd_10m_<län>.tif     COG per county (10 m int16 dm, nodata -1)
    tradhojd/tradhojd_10m.vrt

Products and the policy that reproduces the arrays the tiles already hold
(parity measured 2026-10-12 against proxy-fetched cache arrays, Blekinge):

* **Skogliga grunddata** ``Volym`` (m³sk/ha), ``Grundyta`` (m²/ha),
  ``Medeldiameter`` (cm). The ImageServer mosaic served the newest
  *omdrev* (laser-scanning round) per pixel: omdrev 3 (2026-04, nine
  counties, 10 m) over omdrev 2 (2026-02, all counties, 10 m) over omdrev 1
  (12.5 m). Reading omdrev 3 → 2 → 1 with that precedence matched the
  stored volume, basal area and diameter arrays pixel for pixel.
* **SLU Markfuktighetskarta** index 0–100, 101 = open water, 2 m. The
  tiles hold a nearest-neighbour 10 m sample. The published county files
  (2022-04, four counties re-published 2026-04) are not the version the
  proxy served in 2026-06: same nodata and water masks, index values differ
  by ~5 on average. Documented per-channel diff, not reproducible from any
  open download.
* **Trädhöjd** (tree height, dm). See :func:`read_tradhojd`.

All reads are windowed on the NMD 10 m lattice (``nmd_grid``) and return
the array on the caller's ``(height_px, width_px)`` grid. Nodata is never
silently zero-filled by the caller's request: a missing mirror raises
:class:`SkgMirrorMissingError`; nodata *inside* a product is mapped the
way the retired fetchers mapped it (grunddata and height ``-1 → 0``,
markfukt ``65535 → 0``) so stored tiles and new tiles agree.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .nmd_grid import GRID_M, is_on_grid

DEFAULT_MIRROR_DIR = "/data/skg_open"
ENV_MIRROR_DIR = "SKG_OPEN_DIR"
OPEN_DATA_URL = "https://geodpags.skogsstyrelsen.se/geodataport/data/"

GRUNDDATA_VARIABLES = ("volym", "grundyta", "medeldiameter")
#: Newest scanning round first; a pixel comes from the first round that has it.
OMDREV_PRIORITY = (3, 2, 1)
GRUNDDATA_NODATA = -1
MARKFUKT_NODATA = 65535
TRADHOJD_NODATA = -1


class SkgMirrorMissingError(RuntimeError):
    """The open-data mirror (or one of its products) is not on disk."""


def mirror_dir() -> Path:
    """The mirror root: ``$SKG_OPEN_DIR`` or :data:`DEFAULT_MIRROR_DIR`."""
    return Path(os.environ.get(ENV_MIRROR_DIR) or DEFAULT_MIRROR_DIR)


def grunddata_path(variable: str, omdrev: int) -> Path:
    if variable not in GRUNDDATA_VARIABLES:
        raise ValueError(f"unknown grunddata variable {variable!r}; "
                         f"expected one of {GRUNDDATA_VARIABLES}")
    return mirror_dir() / "grunddata" / f"{variable}_omdrev{omdrev}.vrt"


def markfukt_path() -> Path:
    return mirror_dir() / "markfukt" / "markfukt.vrt"


def tradhojd_path() -> Path:
    return mirror_dir() / "tradhojd" / "tradhojd_10m.vrt"


def _require(path: Path, product: str) -> Path:
    if not path.exists():
        raise SkgMirrorMissingError(
            f"{product}: {path} is missing. The Skogsstyrelsen open-data mirror "
            f"is staged by scripts/stage_skg_open.py (k8s/skg-open-stage-job.yaml) "
            f"from {OPEN_DATA_URL}; set {ENV_MIRROR_DIR} if it lives elsewhere "
            f"than {DEFAULT_MIRROR_DIR}."
        )
    return path


def _check_bbox(west: float, south: float, east: float, north: float,
                h_px: int, w_px: int) -> None:
    """The request must be a whole number of 10 m cells on the NMD lattice.

    Mirrors the cdse_s2 / skg_grunddata extent guard: a bbox whose extent
    does not equal ``size_px × 10 m`` means the caller decoupled ground
    extent from pixels (the 256/512 aux-misalignment bug).
    """
    if not is_on_grid(west, south, east, north):
        raise ValueError(
            f"bbox ({west}, {south}, {east}, {north}) is not on the {GRID_M} m "
            "NMD lattice — snap it with nmd_grid.snap_bounds first.")
    ew, ns = east - west, north - south
    if ew != w_px * GRID_M or ns != h_px * GRID_M:
        raise ValueError(
            f"bbox/size_px mismatch: extent {ew:.0f}x{ns:.0f} m vs {w_px}x{h_px} px "
            f"implies {ew / max(w_px, 1):.2f} m/px, expected {GRID_M} m/px — "
            "caller decoupled extent from pixels.")


def _read_window(path: Path, west: float, south: float, east: float, north: float,
                 out_shape: tuple[int, int], *, resampling, fill):
    """Read ``path`` over the bbox onto ``out_shape``; outside-raster = ``fill``."""
    import rasterio
    from rasterio.windows import from_bounds

    with rasterio.open(path) as ds:
        if ds.crs is not None and ds.crs.to_epsg() != 3006:
            raise ValueError(f"{path}: expected EPSG:3006, got {ds.crs}")
        win = from_bounds(west, south, east, north, ds.transform)
        return ds.read(1, window=win, out_shape=out_shape, boundless=True,
                       fill_value=fill, resampling=resampling)


def read_grunddata(variable: str, west: float, south: float, east: float,
                   north: float, *, size_px: int | tuple[int, int]) -> np.ndarray:
    """Newest-omdrev-per-pixel mosaic of one grunddata variable, float32,
    nodata (no scanning round has the pixel) → 0, negatives clipped to 0."""
    from rasterio.enums import Resampling

    h_px, w_px = (size_px, size_px) if isinstance(size_px, int) else size_px
    _check_bbox(west, south, east, north, h_px, w_px)
    out = np.full((h_px, w_px), GRUNDDATA_NODATA, dtype=np.int16)
    missing = np.ones((h_px, w_px), dtype=bool)
    found_any = False
    for omdrev in OMDREV_PRIORITY:
        path = grunddata_path(variable, omdrev)
        if not path.exists():
            continue          # omdrev not staged (3 exists for nine counties only)
        found_any = True
        if not missing.any():
            break
        # 10 m rounds are read on the exact lattice (nearest == identity);
        # the 12.5 m round is resampled bilinearly like the ImageServer did.
        res = _raster_res(path)
        resampling = Resampling.nearest if res == GRID_M else Resampling.bilinear
        arr = _read_window(path, west, south, east, north, (h_px, w_px),
                           resampling=resampling, fill=GRUNDDATA_NODATA)
        take = missing & (arr != GRUNDDATA_NODATA)
        out[take] = arr[take]
        missing &= ~take
    if not found_any:
        _require(grunddata_path(variable, OMDREV_PRIORITY[1]), f"grunddata {variable}")
    data = out.astype(np.float32)
    data[missing] = 0.0
    return np.clip(data, 0.0, None)


def read_markfukt(west: float, south: float, east: float, north: float,
                  *, size_px: int | tuple[int, int]) -> np.ndarray:
    """Nearest 2 m sample at each 10 m cell centre, uint8 in the service
    encoding: 0 nodata, 1–100 index, 101 water, 255 fill (kept; callers map
    it to NaN as they always did)."""
    from rasterio.enums import Resampling

    h_px, w_px = (size_px, size_px) if isinstance(size_px, int) else size_px
    _check_bbox(west, south, east, north, h_px, w_px)
    path = _require(markfukt_path(), "SLU markfukt")
    arr = _read_window(path, west, south, east, north, (h_px, w_px),
                       resampling=Resampling.nearest, fill=MARKFUKT_NODATA)
    arr = np.where(arr == MARKFUKT_NODATA, 0, arr)
    if arr.max(initial=0) > 255:
        raise ValueError(f"markfukt: unexpected code {int(arr.max())} > 255 in {path}")
    return arr.astype(np.uint8)


def read_tradhojd(west: float, south: float, east: float, north: float,
                  *, size_px: int | tuple[int, int]) -> np.ndarray:
    """Tree height in decimetres at 10 m (the staged ``tradhojd_10m`` product),
    float32, nodata → 0, negatives clipped to 0."""
    from rasterio.enums import Resampling

    h_px, w_px = (size_px, size_px) if isinstance(size_px, int) else size_px
    _check_bbox(west, south, east, north, h_px, w_px)
    path = _require(tradhojd_path(), "Skogsstyrelsen trädhöjd")
    arr = _read_window(path, west, south, east, north, (h_px, w_px),
                       resampling=Resampling.nearest, fill=TRADHOJD_NODATA)
    data = arr.astype(np.float32)
    data[arr == TRADHOJD_NODATA] = 0.0
    return np.clip(data, 0.0, None)


def _raster_res(path: Path) -> float:
    import rasterio

    with rasterio.open(path) as ds:
        return float(ds.res[0])
