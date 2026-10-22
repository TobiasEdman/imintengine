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

import hashlib
import json
import os
import re
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


def check_bbox(west: float, south: float, east: float, north: float,
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


#: GDAL 3.10 reads a large multi-source VRT request in threads and drops a
#: source's read error: an unreadable county raster would come back as
#: nodata, which the readers turn into 0 (a 1280 x 2560 px window, measured
#: 2026-10-21). Read serially, the error raises.
SERIAL_VRT = {"VRT_NUM_THREADS": "1"}


def _read_window(path: Path, west: float, south: float, east: float, north: float,
                 out_shape: tuple[int, int], *, resampling, fill):
    """Read ``path`` over the bbox onto ``out_shape``; outside-raster = ``fill``."""
    import rasterio
    from rasterio.windows import from_bounds

    with rasterio.Env(**SERIAL_VRT), rasterio.open(path) as ds:
        if ds.crs is not None and ds.crs.to_epsg() != 3006:
            raise ValueError(f"{path}: expected EPSG:3006, got {ds.crs}")
        win = from_bounds(west, south, east, north, ds.transform)
        return ds.read(1, window=win, out_shape=out_shape, boundless=True,
                       fill_value=fill, resampling=resampling)


def read_grunddata(variable: str, west: float, south: float, east: float,
                   north: float, *, size_px: int | tuple[int, int]) -> np.ndarray:
    """Newest-omdrev-per-pixel mosaic of one grunddata variable, float32,
    nodata (no scanning round has the pixel) → 0, negatives clipped to 0."""
    out, rounds = _grunddata_mosaic(variable, west, south, east, north, size_px=size_px)
    data = out.astype(np.float32)
    data[rounds == 0] = 0.0
    return np.clip(data, 0.0, None)


def _grunddata_mosaic(variable: str, west: float, south: float, east: float,
                      north: float, *, size_px: int | tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """``(values int16, round int8)``: the newest-omdrev value per pixel and
    the omdrev it came from (0 where no round has the pixel)."""
    from rasterio.enums import Resampling

    h_px, w_px = (size_px, size_px) if isinstance(size_px, int) else size_px
    check_bbox(west, south, east, north, h_px, w_px)
    out = np.full((h_px, w_px), GRUNDDATA_NODATA, dtype=np.int16)
    rounds = np.zeros((h_px, w_px), dtype=np.int8)
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
        rounds[take] = omdrev
        missing &= ~take
    if not found_any:
        _require(grunddata_path(variable, OMDREV_PRIORITY[1]), f"grunddata {variable}")
    return out, rounds


def read_markfukt(west: float, south: float, east: float, north: float,
                  *, size_px: int | tuple[int, int]) -> np.ndarray:
    """Nearest 2 m sample at each 10 m cell centre, uint8 in the service
    encoding: 0 nodata, 1–100 index, 101 water, 255 fill (kept; callers map
    it to NaN as they always did)."""
    from rasterio.enums import Resampling

    h_px, w_px = (size_px, size_px) if isinstance(size_px, int) else size_px
    check_bbox(west, south, east, north, h_px, w_px)
    path = _require(markfukt_path(), "SLU markfukt")
    arr = _read_window(path, west, south, east, north, (h_px, w_px),
                       resampling=Resampling.nearest, fill=MARKFUKT_NODATA)
    arr = np.where(arr == MARKFUKT_NODATA, 0, arr)
    if arr.max(initial=0) > 255:
        raise ValueError(f"markfukt: unexpected code {int(arr.max())} > 255 in {path}")
    return arr.astype(np.uint8)


def _manifest() -> dict:
    return json.loads(_require(mirror_dir() / "MANIFEST.json", "mirror manifest").read_text())


def _county_rasters(manifest: dict, prefix: str) -> dict[str, Path]:
    """``{county: path}`` of the derived rasters recorded under ``prefix`` and on disk."""
    pat = re.compile(re.escape(prefix) + r"(\d{2})\.tif")
    out = {}
    for rel in manifest.get("derived", {}):
        m = pat.fullmatch(rel)
        if m and (mirror_dir() / rel).exists():
            out[m.group(1)] = mirror_dir() / rel
    return out


#: Per-county rasters whose valid pixels mark the county polygon, with the
#: values that are not inside it. Neither alone is a complete mask: measured
#: 2026-10-19 on the 230 NFI top-up tiles, markfukt misses four tiles that
#: grunddata places in counties 01 and 03, and grunddata misses one that
#: markfukt places in 25. Their union gave every tile a county.
COUNTY_MASKS = (("grunddata/omdrev2/volym_", (GRUNDDATA_NODATA,)),
                ("markfukt/markfukt_", (255, MARKFUKT_NODATA)))
TRADHOJD_VRT = "tradhojd/tradhojd_10m.vrt"
TRADHOJD_PREFIX = "tradhojd/tradhojd_10m_"


def staged_tradhojd_counties(manifest: dict) -> set[str]:
    """Counties in the recorded Trädhöjd mosaic. The manifest records the
    VRT's sources only after the VRT is written, so a county counts as staged
    only once the mosaic the reader opens actually contains it."""
    pat = re.compile(re.escape(TRADHOJD_PREFIX) + r"(\d{2})\.tif")
    sources = manifest.get("vrt", {}).get(TRADHOJD_VRT, {}).get("sources", [])
    return {m.group(1) for s in sources if (m := pat.fullmatch(s)) and (mirror_dir() / s).exists()}


def tradhojd_gaps(manifest: dict) -> list[tuple[str, list[float]]]:
    """Recorded upstream gaps of the staged Trädhöjd counties, as
    ``(county, [west, south, east, north])`` in EPSG:3006.

    The staging records a gap where the published data itself is unreadable
    (county 24's tile 720_76 is truncated in the zip). The 10 m raster has
    nodata there, which a reader must not return as height 0."""
    pat = re.compile(re.escape(TRADHOJD_PREFIX) + r"(\d{2})\.tif")
    return [(m.group(1), g) for name, rec in manifest.get("derived", {}).items()
            if (m := pat.fullmatch(name)) for g in rec.get("gaps", [])]


def counties_in_window(west: float, south: float, east: float, north: float,
                       manifest: dict | None = None) -> set[str]:
    """Counties with a valid pixel inside the bbox in any county mask."""
    import rasterio
    from rasterio.windows import from_bounds

    manifest = manifest or _manifest()
    found = set()
    for prefix, invalid in COUNTY_MASKS:
        for county, path in _county_rasters(manifest, prefix).items():
            if county in found:
                continue
            with rasterio.open(path) as ds:
                b = ds.bounds
                if east <= b.left or west >= b.right or north <= b.bottom or south >= b.top:
                    continue
                win = from_bounds(max(west, b.left), max(south, b.bottom), min(east, b.right),
                                  min(north, b.top), ds.transform)
                a = ds.read(1, window=win)
                if a.size and bool((~np.isin(a, invalid)).any()):
                    found.add(county)
    return found


def read_tradhojd(west: float, south: float, east: float, north: float,
                  *, size_px: int | tuple[int, int]) -> np.ndarray:
    """Tree height in decimetres at 10 m (the staged ``tradhojd_10m`` product),
    float32, negatives clipped to 0, nodata NaN (Tobias 2026-10-22). The
    product's own values pass through unchanged, including its 0 over most
    lakes. NaN marks where it has none: outside every county (sea, abroad,
    the large lakes outside the county polygons) and inside a county without
    coverage (county 25 lacks a third of its area, rising northwards). No
    water or land class is inferred here. :func:`county_cells` tells the two
    apart for a caller that needs it.

    Trädhöjd is staged county by county, so its mosaic can lack counties. A
    window is read only when every county it lies in (by the grunddata county
    mask) is staged and it touches no recorded upstream gap
    (:func:`tradhojd_gaps`); otherwise, or when no county covers it, this
    raises :class:`SkgMirrorMissingError` instead of returning the mosaic's
    nodata as zeros."""
    from rasterio.enums import Resampling

    h_px, w_px = (size_px, size_px) if isinstance(size_px, int) else size_px
    check_bbox(west, south, east, north, h_px, w_px)
    path = _require(tradhojd_path(), "Skogsstyrelsen trädhöjd")
    manifest = _manifest()
    need = counties_in_window(west, south, east, north, manifest)
    if not need:
        raise SkgMirrorMissingError(
            f"trädhöjd: no county's grunddata covers bbox ({west}, {south}, {east}, {north}); "
            "its coverage cannot be confirmed")
    missing = need - staged_tradhojd_counties(manifest)
    if missing:
        raise SkgMirrorMissingError(
            f"trädhöjd: county {sorted(missing)} is not staged for bbox ({west}, {south}, {east}, {north}); "
            "stage it with scripts/stage_skg_open.py --products tradhojd")
    hit = sorted({county for county, (gw, gs, ge, gn) in tradhojd_gaps(manifest)
                  if west < ge and east > gw and south < gn and north > gs})
    if hit:
        raise SkgMirrorMissingError(
            f"trädhöjd: bbox ({west}, {south}, {east}, {north}) touches a recorded upstream gap of "
            f"county {hit}: the published data is unreadable there, so its heights are unknown")
    arr = _read_window(path, west, south, east, north, (h_px, w_px),
                       resampling=Resampling.nearest, fill=TRADHOJD_NODATA)
    data = np.clip(arr.astype(np.float32), 0.0, None)
    data[arr == TRADHOJD_NODATA] = np.nan
    return data


def county_cells(west: float, south: float, east: float, north: float, out_shape: tuple[int, int],
                 counties: set[str], manifest: dict | None = None) -> np.ndarray:
    """Boolean ``out_shape`` grid of the bbox: True where any of ``counties``'
    ``COUNTY_MASKS`` rasters is valid (nearest). Only validity counts, never a
    class: markfukt's water code 101 is as inside as any index value."""
    from rasterio.enums import Resampling

    manifest = manifest or _manifest()
    inside = np.zeros(out_shape, bool)
    for prefix, invalid in COUNTY_MASKS:
        rasters = _county_rasters(manifest, prefix)
        for county in sorted(counties & rasters.keys()):
            a = _read_window(rasters[county], west, south, east, north, out_shape,
                             resampling=Resampling.nearest, fill=invalid[0])
            inside |= ~np.isin(a, invalid)
    return inside


def window_provenance(west: float, south: float, east: float, north: float,
                      *, size_px: int | tuple[int, int]) -> dict:
    """Which mirror rasters a tile window is read from, for the tile's record.

    The open products are versions of their own (markfukt and height differ
    from the retired proxies; docs/data/skg_open_mirror.md), so every tile
    names the county rasters behind its channels: each raster's sha256 and
    its source zip's name, sha256 and publication date as recorded in
    MANIFEST.json, the omdrev each grunddata pixel came from, the statistic
    the 10 m height was derived with, and the manifest's own sha256."""
    h_px, w_px = (size_px, size_px) if isinstance(size_px, int) else size_px
    manifest_path = _require(mirror_dir() / "MANIFEST.json", "mirror manifest")
    manifest = json.loads(manifest_path.read_text())
    counties = sorted(counties_in_window(west, south, east, north, manifest))
    omdrev_pixels = {}
    for variable in GRUNDDATA_VARIABLES:
        _, rounds = _grunddata_mosaic(variable, west, south, east, north, size_px=(h_px, w_px))
        omdrev_pixels[variable] = {str(r): int((rounds == r).sum()) for r in (*OMDREV_PRIORITY, 0)}
    wanted = [f"grunddata/omdrev{r}/{v}_{c}.tif" for c in counties for v in GRUNDDATA_VARIABLES
              for r in OMDREV_PRIORITY if omdrev_pixels[v][str(r)]]
    wanted += [f"markfukt/markfukt_{c}.tif" for c in counties]
    wanted += [f"{TRADHOJD_PREFIX}{c}.tif" for c in counties]
    rasters = {}
    for rel in wanted:
        rec = manifest.get("derived", {}).get(rel)
        if rec is None:
            continue                        # e.g. omdrev 3 is published for nine counties only
        src = manifest.get("files", {}).get(rec.get("from", ""), {})
        rasters[rel] = {"sha256": rec.get("sha256"), "source": rec.get("from"),
                        "source_sha256": src.get("sha256"), "published": src.get("index_modified"),
                        **({"statistic": rec["statistic"]} if "statistic" in rec else {})}
    return {"mirror_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "counties": counties, "grunddata_omdrev_pixels": omdrev_pixels,
            "rasters": rasters}


def _raster_res(path: Path) -> float:
    import rasterio

    with rasterio.open(path) as ds:
        return float(ds.res[0])
