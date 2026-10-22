#!/usr/bin/env python3
"""Stage Skogsstyrelsen's open GeoTIFF products on the PVC (backlog IM-022).

Downloads the per-county zips from ``geodpags.skogsstyrelsen.se``, re-encodes
each raster as a tiled Cloud-Optimized GeoTIFF the fetchers can read by
window, builds one VRT mosaic per product and keeps a MANIFEST.json beside
the data. Reader: ``imint.training.skg_open``.

Products (``--products``):

  grunddata   sks_{Volym,Grundyta,Medeldiameter}_omdrev{1,2,3}_<län>.zip
              10 m (omdrev 2, 3) / 12.5 m (omdrev 1) int16, nodata -1,
              re-encoded losslessly.
  markfukt    sksSLUMarkfuktighet<län>.zip — 2 m uint16, nodata 65535,
              re-encoded losslessly (no overviews: the reader samples the
              2 m cell at each 10 m centre itself).
  tradhojd    sksTradhojd<län>.zip — 1 m int16 dm, nodata -1. Reduced to
              the 10 m lattice with ``--tradhojd-method`` (p90 by default:
              the statistic closest to the retired proxy's height channel)
              into tradhojd/tradhojd_10m_<län>.tif; the 1 m raw zip is kept
              unless ``--drop-raw``.

Resumable and idempotent: a raw zip whose sha256 is already in the manifest
and whose derived raster exists is skipped; a partial download continues
with an HTTP Range request. Every manifest update is written atomically.

Usage (inside k8s/skg-open-stage-job.yaml)::

    python scripts/stage_skg_open.py --dest /data/skg_open \\
        --products grunddata markfukt --omdrev 1 2 3

Dry run (plan + sizes only, no writes)::

    python scripts/stage_skg_open.py --dest /tmp/x --products grunddata --dry-run
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import html
import json
import math
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
import zlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SOURCE_URL = "https://geodpags.skogsstyrelsen.se/geodataport/data/"
USER_AGENT = "ImintEngine/1.0 (stage_skg_open)"
MANIFEST_SCHEMA = 1
COUNTIES = ("01", "03", "04", "05", "06", "07", "08", "09", "10", "12", "13",
            "14", "17", "18", "19", "20", "21", "22", "23", "24", "25")
GRUNDDATA = {"volym": "Volym", "grundyta": "Grundyta", "medeldiameter": "Medeldiameter"}
ZIP_DEFLATE, ZIP_DEFLATE64 = 8, 9
COG_OPTS = ["-of", "COG", "-co", "COMPRESS=DEFLATE", "-co", "PREDICTOR=2",
            "-co", "BLOCKSIZE=512", "-co", "OVERVIEWS=NONE", "-co", "BIGTIFF=IF_SAFER",
            "-co", "NUM_THREADS=ALL_CPUS"]


# ── plan ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Item:
    name: str              # zip name as published
    product: str           # grunddata | markfukt | tradhojd
    derived: str           # path under dest of the raster the reader uses
    omdrev: int | None = None
    variable: str | None = None
    county: str | None = None


def parse_index(page: str) -> dict[str, dict]:
    """``{name: {"bytes": int, "modified": str}}`` from the IIS directory page."""
    rows = re.findall(
        r'(\d+/\d+/\d+\s+\d+:\d+ [AP]M)\s+(\d+)\s+<A HREF="[^"]+">([^<]+)</A>', page)
    return {html.unescape(n): {"bytes": int(b), "modified": m} for m, b, n in rows}


def plan(products: list[str], omdrev: list[int], counties: list[str],
         index: dict[str, dict]) -> list[Item]:
    items: list[Item] = []
    if "grunddata" in products:
        for om in omdrev:
            for var, label in GRUNDDATA.items():
                for c in counties:
                    name = f"sks_{label}_omdrev{om}_{c}.zip"
                    if name in index:          # omdrev 3 exists for nine counties
                        items.append(Item(name, "grunddata",
                                          f"grunddata/omdrev{om}/{var}_{c}.tif", om, var, c))
    if "markfukt" in products:
        for c in counties:
            name = f"sksSLUMarkfuktighet{c}.zip"
            if name in index:
                items.append(Item(name, "markfukt", f"markfukt/markfukt_{c}.tif", county=c))
    if "tradhojd" in products:
        for c in counties:
            name = f"sksTradhojd{c}.zip"
            if name in index:
                items.append(Item(name, "tradhojd", f"tradhojd/tradhojd_10m_{c}.tif", county=c))
    missing = [i.name for i in items if i.name not in index]
    assert not missing, missing
    return items


def vrt_targets(items: list[Item]) -> dict[str, list[str]]:
    """``{vrt path under dest: [derived rasters]}``."""
    out: dict[str, list[str]] = {}
    for it in items:
        if it.product == "grunddata":
            key = f"grunddata/{it.variable}_omdrev{it.omdrev}.vrt"
        elif it.product == "markfukt":
            key = "markfukt/markfukt.vrt"
        else:
            key = "tradhojd/tradhojd_10m.vrt"
        out.setdefault(key, []).append(it.derived)
    return out


# ── manifest ────────────────────────────────────────────────────────────────

def load_manifest(dest: Path) -> dict:
    path = dest / "MANIFEST.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"schema": MANIFEST_SCHEMA, "source_url": SOURCE_URL, "files": {},
            "derived": {}, "vrt": {}, "history": []}


def save_manifest(dest: Path, manifest: dict) -> None:
    """Atomic write; the previous manifest is kept as MANIFEST.previous.json
    so a record is never lost by a later run."""
    manifest["updated_at"] = _now()
    manifest["git_sha"] = _git_sha()
    path = dest / "MANIFEST.json"
    if path.exists():
        shutil.copyfile(path, path.with_name("MANIFEST.previous.json"))
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


class UpstreamChanged(RuntimeError):
    """A re-downloaded zip does not match the sha256 recorded for it."""


def check_recovered(rec: dict, new_sha256: str, name: str, *, accept: bool,
                    raw: Path | None = None, dest: Path | None = None) -> dict:
    """A raw zip re-downloaded after deletion must hash to the recorded
    sha256. If upstream changed, the old digest is kept under
    ``superseded_sha256`` and the run stops unless ``accept`` is set: a
    silently replaced digest would turn the manifest into a record of the
    *current* upstream, not of what the COGs were derived from.

    On a refused mismatch the downloaded file is moved aside to
    ``<name>.upstream-changed`` so that no later run can find a raw file
    under the recorded name and skip the check. On an accepted mismatch
    the manifest as it was before the change is archived as an immutable
    ``MANIFEST.superseded-<timestamp>.json`` (the rolling
    ``MANIFEST.previous.json`` is overwritten by the same run's later saves).
    """
    old = rec.get("sha256")
    if not old or old == new_sha256:
        return rec
    if not accept:
        if raw is not None and raw.exists():
            os.replace(raw, raw.with_name(raw.name + ".upstream-changed"))
        raise UpstreamChanged(
            f"{name}: re-downloaded file hashes to {new_sha256[:12]}…, manifest records "
            f"{old[:12]}…; upstream changed since the COG was derived. The download was moved "
            f"to {raw.name + '.upstream-changed' if raw is not None else '<name>.upstream-changed'}. "
            "Re-run with --accept-upstream-change to record the new digest (the old one is "
            "kept under superseded_sha256) and re-derive the raster.")
    if dest is not None and (dest / "MANIFEST.json").exists():
        archive_manifest(dest)
    return {**rec, "superseded_sha256": [*rec.get("superseded_sha256", []), old],
            "upstream_changed_at": _now()}


def archive_manifest(dest: Path) -> Path:
    """Copy MANIFEST.json to an immutable ``MANIFEST.superseded-<ts>[-n].json``.

    The file is created exclusively (O_CREAT | O_EXCL): a name that already
    exists — two acceptances in the same second, or a leftover from an
    earlier run — gets the next ``-n`` suffix instead of being overwritten,
    so every archived record survives."""
    stamp = _now().replace(":", "").replace("-", "")
    src = (dest / "MANIFEST.json").read_bytes()
    for n in range(0, 10_000):
        path = dest / (f"MANIFEST.superseded-{stamp}.json" if n == 0
                       else f"MANIFEST.superseded-{stamp}-{n}.json")
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            continue
        with os.fdopen(fd, "wb") as f:
            f.write(src)
        return path
    raise RuntimeError(f"{dest}: could not allocate a unique manifest archive name")


def is_done(item: Item, manifest: dict, dest: Path) -> bool:
    """Raw hashed and derived raster present — or the raw zip intentionally
    deleted after a verified derivation (``raw_deleted_at``), in which case
    the derived raster alone completes the item."""
    rec = manifest["files"].get(item.name)
    der = manifest["derived"].get(item.derived)
    if not (rec and rec.get("sha256") and der and der.get("sha256") and (dest / item.derived).exists()):
        return False
    return bool(rec.get("raw_deleted_at")) or (dest / "raw" / item.name).exists()


def staged_vrt_sources(manifest: dict, dest: Path, vrt: str) -> list[str]:
    """Every derived raster on disk that belongs to ``vrt``'s product and
    round, from the manifest — not only the counties of the current run, so a
    subset re-run never shrinks a published mosaic."""
    if vrt.startswith("grunddata/"):
        var, om = re.fullmatch(r"grunddata/(\w+)_omdrev(\d)\.vrt", vrt).groups()
        prefix = f"grunddata/omdrev{om}/{var}_"
    elif vrt.startswith("markfukt/"):
        prefix = "markfukt/markfukt_"
    else:
        prefix = "tradhojd/tradhojd_10m_"
    return sorted(p for p in manifest["derived"]
                  if p.startswith(prefix) and p.endswith(".tif") and (dest / p).exists())


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_sha() -> str:
    env = os.environ.get("IMINT_GIT_SHA")
    if env:
        return env
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


# ── download ────────────────────────────────────────────────────────────────

def _request(url: str, headers: dict | None = None, method: str = "GET"):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})},
                                 method=method)
    return urllib.request.urlopen(req, timeout=120)


def download(url: str, target: Path, expected_bytes: int, *, attempts: int = 6) -> dict:
    """Resumable download to ``target`` (via ``target.part``); verifies size."""
    part = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == expected_bytes:
        return {"resumed": False, "attempts": 0}
    meta = {}
    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.exists() else 0
        if have > expected_bytes:
            part.unlink()
            have = 0
        try:
            headers = {"Range": f"bytes={have}-"} if have else {}
            with _request(url, headers) as resp, open(part, "ab" if have else "wb") as out:
                if have and resp.status != 206:
                    out.close()
                    part.unlink()
                    continue                    # server ignored the range: restart
                meta = {"last_modified": resp.headers.get("Last-Modified"),
                        "etag": resp.headers.get("ETag")}
                shutil.copyfileobj(resp, out, 8 << 20)
            if part.stat().st_size == expected_bytes:
                os.replace(part, target)
                return {"resumed": have > 0, "attempts": attempt, **meta}
            raise OSError(f"size mismatch after download: {part.stat().st_size} != {expected_bytes}")
        except (OSError, urllib.error.URLError) as e:
            if attempt == attempts:
                raise
            print(f"    retry {attempt}/{attempts} after {type(e).__name__}: {e}", flush=True)
            time.sleep(min(60, 5 * attempt))
    raise AssertionError("unreachable")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def crc32_of(path: Path) -> int:
    crc = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            crc = zlib.crc32(chunk, crc)
    return crc


# ── extract / convert ───────────────────────────────────────────────────────

def zip_tif_entry(zpath: Path) -> zipfile.ZipInfo:
    with zipfile.ZipFile(zpath) as z:
        tifs = [i for i in z.infolist() if i.filename.lower().endswith(".tif")]
    if len(tifs) != 1:
        raise ValueError(f"{zpath.name}: expected one .tif entry, found {[i.filename for i in tifs]}")
    return tifs[0]


def zip_tiles(zpath: Path) -> list[zipfile.ZipInfo] | None:
    """A county shipped as tiles: its mosaic ``.vrt`` entry first, then the
    ``.tif`` tiles, all in the VRT's directory. ``None`` for a zip with one
    raster and no VRT.

    Read 2026-10-21 from the zip directories: Trädhöjd county 24
    (Västerbotten) is 689 Deflate tiles of 10 km plus
    ``vasterbottens/vasterbottens.vrt``; every other county is one raster."""
    with zipfile.ZipFile(zpath) as z:
        infos = z.infolist()
    vrts = [i for i in infos if i.filename.lower().endswith(".vrt")]
    tifs = [i for i in infos if i.filename.lower().endswith(".tif")]
    if not vrts:
        return None
    if len(vrts) != 1 or not tifs:
        raise ValueError(f"{zpath.name}: expected one .vrt with its tiles, found {len(vrts)} .vrt and {len(tifs)} .tif")
    home = Path(vrts[0].filename).parent
    if strays := [i.filename for i in tifs if Path(i.filename).parent != home]:
        raise ValueError(f"{zpath.name}: tiles outside the VRT's directory {home}: {strays[:5]}")
    return [vrts[0], *tifs]


def _extract_entry(zpath: Path, info: zipfile.ZipInfo, tmp: Path) -> None:
    """Write one entry to ``tmp`` and check its size and CRC-32 read back from disk.

    Deflate64 is streamed by Info-ZIP ``unzip -p`` (GDAL and ``zipfile``
    cannot read it), Deflate and Stored by ``zipfile``. The caller removes
    ``tmp`` on failure."""
    if info.compress_type == ZIP_DEFLATE64:
        with open(tmp, "wb") as out:
            done = subprocess.run(["unzip", "-p", str(zpath), info.filename], stdout=out,
                                  stderr=subprocess.PIPE, text=True)
        if done.returncode != 0:
            raise OSError(f"{zpath.name}: unzip exit {done.returncode}: {done.stderr.strip()[:300]}")
    elif info.compress_type in (ZIP_DEFLATE, zipfile.ZIP_STORED):
        with zipfile.ZipFile(zpath) as z, z.open(info) as src, open(tmp, "wb") as out:
            shutil.copyfileobj(src, out, 16 << 20)
    else:
        raise ValueError(f"{zpath.name}: unsupported zip method {info.compress_type} for {info.filename}")
    if tmp.stat().st_size != info.file_size:
        raise OSError(f"{zpath.name}: extracted {tmp.stat().st_size} bytes, expected {info.file_size}")
    if (crc := crc32_of(tmp)) != info.CRC:
        raise OSError(f"{zpath.name}: extracted file has CRC-32 {crc:08x}, expected {info.CRC:08x}")


#: Nodata of the tiled mosaic where no tile lies, and of the derived raster.
TILED_NODATA = -1

#: Tiles truncated in the published zip itself: their CRC-32 matches the zip
#: directory, but the blocks past the cut cannot be read. Such a tile is staged
#: with those blocks as nodata and its gaps recorded on the derived raster,
#: where ``read_tradhojd`` refuses any window that touches them; any other
#: unreadable block stops the derivation. An entry applies only to the tile
#: with this CRC-32, and the backfill only to the zip with this sha256, so a
#: republished zip is checked afresh. Found 2026-10-21 in run r2: 2456 of the
#: tile's 6241 blocks, every one from (58, 47) on, apart from 14 never-written
#: (all-0) blocks; the gaps are the conservative rectangles around them.
KNOWN_TRUNCATED_TILES = {
    "sksTradhojd24.zip": {
        "sha256": "05c5c6a1f8ad3f3aa0ff093222482b645073fef6cd81d9561f1dbfe548703ac4",
        "tiles": {
            "720_76.tif": {"crc32": 0x170E97A9, "unreadable_blocks": 2456,
                           "gaps": [[767424.0, 7203856.0, 770000.0, 7203984.0],
                                    [760000.0, 7200000.0, 770000.0, 7203856.0]]},
        },
    },
}

#: GDAL 3.10 reads the sources of a large VRT request in threads and then
#: drops a source's read error: a truncated tile read as nodata, silently
#: (run r2, rasterio 1.4.4 / GDAL 3.10.3). Read serially, the error raises.
#: The derivation is already parallel across worker processes.
SERIAL_VRT = {"VRT_NUM_THREADS": "1"}


def known_truncated(zip_name: str, tiles: list[zipfile.ZipInfo]) -> dict[str, dict]:
    """The ``KNOWN_TRUNCATED_TILES`` entries of ``zip_name`` whose tile is among
    ``tiles`` with the listed CRC-32, by tile name."""
    known = KNOWN_TRUNCATED_TILES.get(zip_name, {}).get("tiles", {})
    crcs = {Path(t.filename).name: t.CRC for t in tiles}
    return {name: k for name, k in known.items() if crcs.get(name) == k["crc32"]}


def _patch_truncated_tile(tif: Path, known: dict) -> None:
    """Rewrite a known truncated tile with its unreadable blocks as nodata.

    Its unreadable blocks must be exactly ``known["unreadable_blocks"]`` in
    number and each lie inside one of ``known["gaps"]``; otherwise this
    raises, since the tile is not the one that was examined."""
    import numpy as np
    import rasterio
    from rasterio.errors import RasterioIOError

    with rasterio.open(tif) as src:
        profile = src.profile
        blocks = []
        for _, win in src.block_windows(1):
            try:
                blocks.append((win, src.read(1, window=win)))
            except RasterioIOError:
                blocks.append((win, None))
        bad = [src.window_bounds(win) for win, a in blocks if a is None]
    inside = all(any(gw <= left and gs <= bottom and right <= ge and top <= gn
                     for gw, gs, ge, gn in known["gaps"]) for left, bottom, right, top in bad)
    if len(bad) != known["unreadable_blocks"] or not inside:
        raise ValueError(f"{tif.name}: {len(bad)} unreadable blocks, expected {known['unreadable_blocks']} "
                         f"inside {known['gaps']}")
    patched = tif.with_name(tif.name + ".patched.tif")
    with rasterio.open(patched, "w", **profile) as dst:
        for win, a in blocks:
            if a is None:
                a = np.full((int(win.height), int(win.width)), TILED_NODATA, dtype=profile["dtype"])
            dst.write(a, 1, window=win)
    os.replace(patched, tif)


def _mark_tiled_vrt_nodata(vrt: Path, tiles: list[zipfile.ZipInfo]) -> None:
    """Check the shipped mosaic VRT against the zip's tiles and give it a nodata value.

    The tiles carry no nodata and no CRS tag; the VRT supplies EPSG:3006 but
    no ``NoDataValue``, so GDAL would read the gaps between tiles as 0, a
    valid height. The VRT must list exactly the zip's tiles, relative to
    itself, in one band; ``NoDataValue`` -1 is then set. A VRT that already
    has one is refused: its meaning would have to be checked first.

    The SRS must declare ``AUTHORITY["EPSG","3006"]``, the CRS the derived
    raster is written in. The shipped WKT1 is SWEREF99 TM with EPSG:3006's
    parameters, but GDAL identifies it only at confidence 25, tied with
    3847 and 3012, so ``to_epsg()`` cannot make this check."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(vrt)
    srs = tree.getroot().find("SRS")
    if srs is None or 'AUTHORITY["EPSG","3006"]' not in "".join((srs.text or "").split()):
        raise ValueError(f"{vrt.name}: the SRS does not declare EPSG:3006")
    bands = tree.getroot().findall("VRTRasterBand")
    if len(bands) != 1:
        raise ValueError(f"{vrt.name}: expected one band, found {len(bands)}")
    band = bands[0]
    if band.find("NoDataValue") is not None:
        raise ValueError(f"{vrt.name}: already has a NoDataValue; check what it means before staging")
    sources = band.findall("./*/SourceFilename")
    if any(s.get("relativeToVRT") != "1" for s in sources):
        raise ValueError(f"{vrt.name}: a source is not relative to the VRT")
    listed, shipped = sorted(s.text for s in sources), sorted(Path(t.filename).name for t in tiles)
    if listed != shipped:
        raise ValueError(f"{vrt.name}: lists {len(listed)} sources, the zip has {len(shipped)} tiles; "
                         f"first differences {sorted(set(listed) ^ set(shipped))[:5]}")
    nodata = ET.Element("NoDataValue")
    nodata.text = str(TILED_NODATA)
    band.insert(0, nodata)
    tree.write(vrt)


def _extract_tiled(zpath: Path, work: Path, entries: list[zipfile.ZipInfo]) -> tuple[str, Path]:
    """Extract a tiled county (its VRT and every tile, each size- and
    CRC-checked) under ``work/<zip stem>/`` and return the mosaic VRT and that
    directory. On any failure, including SIGTERM, the directory is removed."""
    vrt_info, tiles = entries[0], entries[1:]
    root = work / zpath.stem
    shutil.rmtree(root, ignore_errors=True)        # tiles left by a SIGKILLed run are not trusted
    try:
        for info in entries:
            out = root / info.filename
            out.parent.mkdir(parents=True, exist_ok=True)
            _extract_entry(zpath, info, out)
        vrt = root / vrt_info.filename
        for name, known in known_truncated(zpath.name, tiles).items():
            _patch_truncated_tile(vrt.parent / name, known)
        _mark_tiled_vrt_nodata(vrt, tiles)
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return str(vrt), root


def gdal_source(zpath: Path, work: Path, *, extract: bool = False) -> tuple[str, Path | None]:
    """A GDAL-openable path for the zip's raster.

    Deflate (method 8) is read in place through ``/vsizip/``. Deflate64
    (method 9, used by the whole-country and most Trädhöjd zips) is not
    supported by GDAL or ``zipfile``; Info-ZIP ``unzip -p`` streams it to a
    temporary file that the caller removes, with a few MB of memory and the
    entry's CRC checked (exit status 2 on a mismatch). ``inflate64`` 1.0.4,
    used before, kept ~0.7-1 GB per GiB inflated and never returned it
    (measured 2026-10-20): the 40-55 GiB counties would OOM-kill the Job.
    ``extract`` writes a Deflate
    entry to such a file too: a fresh ``/vsizip/`` handle that seeks to a
    late row inflates the stream from its start, so parallel readers each
    opening the zip would pay that again and again.

    A work file is returned only once its size and CRC-32, read back from
    disk, equal the zip directory's; the derivation reads exactly those
    bytes. On any failure, including SIGTERM's SystemExit mid-extraction,
    the partial work file (up to ~55 GiB for county 25) is removed.

    A tiled county (``zip_tiles``) is always extracted, to a directory that
    is returned in place of the work file; the source is its mosaic VRT.
    """
    if (entries := zip_tiles(zpath)) is not None:
        return _extract_tiled(zpath, work, entries)
    info = zip_tif_entry(zpath)
    if info.compress_type not in (ZIP_DEFLATE, zipfile.ZIP_STORED, ZIP_DEFLATE64):
        raise ValueError(f"{zpath.name}: unsupported zip method {info.compress_type}")
    if info.compress_type != ZIP_DEFLATE64 and not extract:
        return f"/vsizip/{zpath}/{info.filename}", None
    work.mkdir(parents=True, exist_ok=True)
    tmp = work / info.filename
    try:
        _extract_entry(zpath, info, tmp)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return str(tmp), tmp


def convert(item: Item, zpath: Path, out: Path, work: Path, tradhojd_method: str,
            tradhojd_workers: int = 1) -> dict:
    """Write the derived COG for ``item``; returns the record for the manifest.

    A tiled Trädhöjd county is derived from its mosaic VRT. Its tiles fill
    the sea and the land outside the county with 0; ``mask_to_county`` sets
    those cells to -1 afterwards, as for every county, so they never cover a
    neighbour in the mosaic and the county's own zeros (treeless land, the
    product's lakes) stay 0."""
    entries = zip_tiles(zpath)
    tiled = entries is not None
    gaps = [g for k in known_truncated(zpath.name, entries[1:]).values() for g in k["gaps"]] if tiled else []
    if tiled and item.product != "tradhojd":
        raise ValueError(f"{zpath.name}: tiled zips are handled for tradhojd only, not {item.product}")
    src, tmp = gdal_source(zpath, work, extract=item.product == "tradhojd" and tradhojd_workers > 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".part.tif")
    try:
        if item.product == "tradhojd":
            check = derive_tradhojd_10m(src, partial, tradhojd_method, workers=tradhojd_workers)
            method = f"block {tradhojd_method} of the 1 m cells per 10 m cell, nodata-aware, int16 dm"
            if gaps:
                method += "; truncated upstream tiles: their unreadable blocks are nodata, the gaps recorded"
        else:
            subprocess.run(["gdal_translate", "-q", *COG_OPTS, src, str(partial)], check=True)
            method = "gdal_translate lossless"
            check = verify_cog(src, partial, lossless=True)
        os.replace(partial, out)
    finally:
        if tmp is not None and tmp.is_dir():
            shutil.rmtree(tmp)
        elif tmp is not None and tmp.exists():
            tmp.unlink()
        if partial.exists():
            partial.unlink()
    return {"from": item.name, "method": method, "bytes": out.stat().st_size,
            "sha256": sha256_of(out), "created_at": _now(), **check, **({"gaps": gaps} if gaps else {})}


TRADHOJD_STATS = ("p90", "p95", "average", "max")


def block_stat(block, stat: str, nodata: int):
    """``stat`` of each 10×10 group of 1 m cells in ``block`` (rows and
    columns multiples of 10), ignoring nodata; a group with no valid cell
    is nodata. Returns int16 (dm, rounded)."""
    import numpy as np

    h, w = block.shape
    g = block.reshape(h // 10, 10, w // 10, 10).transpose(0, 2, 1, 3).reshape(h // 10, w // 10, 100)
    valid = g != nodata
    n_valid = valid.sum(axis=2)
    gf = np.where(valid, g, np.nan).astype(np.float32)
    with np.errstate(all="ignore"):
        if stat == "average":
            v = np.nanmean(gf, axis=2)
        elif stat == "max":
            v = np.nanmax(np.where(valid, gf, -np.inf), axis=2)
        elif stat.startswith("p"):
            v = np.nanpercentile(gf, int(stat[1:]), axis=2)
        else:
            raise ValueError(f"unknown tradhojd statistic {stat!r}")
    out = np.where(n_valid > 0, np.rint(v), nodata)
    return np.clip(out, -1, 32767).astype(np.int16)


def _reduce_strip(src: str, row: int, n_rows: int, stat: str, nodata: int):
    """``block_stat`` of ``n_rows`` source rows from ``row``; runs in a worker process."""
    import rasterio
    from rasterio.windows import Window

    with rasterio.Env(**SERIAL_VRT), rasterio.open(src) as ds:
        return block_stat(ds.read(1, window=Window(0, row, ds.width, n_rows)), stat, nodata)


def derive_tradhojd_10m(src: str, out: Path, stat: str, workers: int = 1) -> dict:
    """Reduce the 1 m Trädhöjd raster to the NMD 10 m lattice, one strip of
    640 source rows at a time (memory-bounded; works on a strip TIFF read
    through /vsizip/).

    With ``workers`` > 1 the strips are reduced by that many spawned
    processes, each opening ``src`` and reading its own rows, at most two
    strips per worker in flight; this process writes them in row order.
    Every strip goes through the same ``block_stat``, so the raster equals
    the serial one. Measured 2026-10-20 on county 05 (170000 x 160000 px):
    x6.3 wall time at 8 workers, reading and the statistic both scaling.
    The workers are spawned (no fork of a process holding GDAL handles), so
    the caller's main module must be importable: run this script as a
    file, as the Job does, not from stdin."""
    import multiprocessing
    from collections import deque
    from concurrent.futures import ProcessPoolExecutor

    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.windows import Window

    with rasterio.open(src) as ds:
        if ds.res != (1.0, 1.0) or ds.dtypes[0] != "int16":
            raise ValueError(f"{src}: expected 1 m int16 Trädhöjd, got {ds.res} {ds.dtypes[0]}")
        nod = int(ds.nodata if ds.nodata is not None else -1)
        west, north = ds.transform.c, ds.transform.f
        if west % 10 or north % 10 or ds.width % 10 or ds.height % 10:
            raise ValueError(f"{src}: origin/extent not aligned to the 10 m lattice")
        oh, ow = ds.height // 10, ds.width // 10
        prof = {"driver": "GTiff", "width": ow, "height": oh, "count": 1, "dtype": "int16",
                "crs": "EPSG:3006", "transform": from_origin(west, north, 10, 10), "nodata": -1,
                "tiled": True, "blockxsize": 512, "blockysize": 512, "compress": "DEFLATE",
                "predictor": 2, "BIGTIFF": "IF_SAFER"}
        src_bounds = [float(v) for v in ds.bounds]
        height = ds.height
    rows_per_chunk = 10 * 64
    strips = [(r, min(rows_per_chunk, height - r)) for r in range(0, height, rows_per_chunk)]
    vmin, vmax, n_valid = None, None, 0

    def write(dst, r, n, reduced):
        nonlocal vmin, vmax, n_valid
        dst.write(reduced, 1, window=Window(0, r // 10, ow, n // 10))
        valid = reduced[reduced != -1]
        if valid.size:
            n_valid += int(valid.size)
            lo, hi = int(valid.min()), int(valid.max())
            vmin = lo if vmin is None else min(vmin, lo)
            vmax = hi if vmax is None else max(vmax, hi)

    with rasterio.open(out, "w", **prof) as dst:
        if workers <= 1:
            with rasterio.Env(**SERIAL_VRT), rasterio.open(src) as ds:    # one handle, read in order
                for r, n in strips:
                    write(dst, r, n, block_stat(ds.read(1, window=Window(0, r, ds.width, n)), stat, nod))
        else:
            ex = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))
            try:
                todo, pending = iter(strips), deque()
                for r, n in todo:
                    pending.append((r, n, ex.submit(_reduce_strip, src, r, n, stat, nod)))
                    if len(pending) == 2 * workers:
                        break
                while pending:
                    r, n, fut = pending.popleft()
                    nxt = next(todo, None)
                    if nxt is not None:
                        pending.append((*nxt, ex.submit(_reduce_strip, src, *nxt, stat, nod)))
                    write(dst, r, n, fut.result())
            finally:
                ex.shutdown(wait=False, cancel_futures=True)   # a SIGTERM or error must not wait for the queue
    rec = {"width": ow, "height": oh, "res_m": 10.0, "nodata": -1, "dtype": "int16",
           "bounds": src_bounds, "statistic": stat,
           "value_min_dm": vmin, "value_max_dm": vmax, "valid_cells": n_valid}
    return {**rec, **verify_tradhojd_10m(out, rec)}


#: Plausible range of a 10 m tree-height statistic, dm. Swedish trees stay
#: below ~50 m; the bound only catches decoding garbage, not tall trees.
TRADHOJD_RANGE_DM = (0, 1000)


def verify_tradhojd_10m(out: Path, expected: dict) -> dict:
    """Re-open the derived raster and check it against what the derivation
    expected before the raw zip may be deleted: readable, EPSG:3006, int16,
    nodata -1, 10 m, the source's exact extent, at least one valid cell and
    a plausible value range. Raises on any mismatch."""
    import numpy as np
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(out) as b:
        if b.crs is None or b.crs.to_epsg() != 3006:
            raise ValueError(f"{out}: CRS {b.crs} is not EPSG:3006")
        if b.dtypes[0] != "int16" or b.nodata != -1 or b.res != (10.0, 10.0):
            raise ValueError(f"{out}: expected int16, nodata -1, 10 m; got {b.dtypes[0]}, {b.nodata}, {b.res}")
        if (b.width, b.height) != (expected["width"], expected["height"]):
            raise ValueError(f"{out}: size {b.width}x{b.height}, expected {expected['width']}x{expected['height']}")
        bounds = [float(v) for v in b.bounds]
        if bounds != expected["bounds"]:
            raise ValueError(f"{out}: extent {bounds} differs from the source {expected['bounds']}")
        n = 512
        for r, c in [(0, 0), (max(0, b.height - n), max(0, b.width - n)),
                     (max(0, b.height // 2 - n // 2), max(0, b.width // 2 - n // 2))]:
            b.read(1, window=Window(c, r, min(n, b.width - c), min(n, b.height - r)))   # readable
    if not expected["valid_cells"]:
        raise ValueError(f"{out}: no valid cell")
    lo, hi = TRADHOJD_RANGE_DM
    if not (lo <= expected["value_min_dm"] <= expected["value_max_dm"] <= hi):
        raise ValueError(f"{out}: values {expected['value_min_dm']}..{expected['value_max_dm']} dm "
                         f"outside the plausible {lo}..{hi} dm")
    return {"verified": True, "verified_windows": 3}


def record_known_gaps(item: Item, manifest: dict, dest: Path) -> bool:
    """Record the known gaps on a county derived before they were known.

    Applies only when the recorded zip is the one they were found in (its
    sha256) and no gaps are recorded yet. Every 10 m cell lying wholly inside
    a gap must already be nodata, or this raises. Returns whether the record
    changed."""
    import rasterio
    from rasterio.windows import Window

    known = KNOWN_TRUNCATED_TILES.get(item.name)
    der = manifest["derived"].get(item.derived)
    if (not known or der is None or "gaps" in der
            or manifest["files"].get(item.name, {}).get("sha256") != known["sha256"]):
        return False
    gaps = [g for k in known["tiles"].values() for g in k["gaps"]]
    with rasterio.open(dest / item.derived) as ds:
        west, north, res = ds.transform.c, ds.transform.f, ds.res[0]
        for gw, gs, ge, gn in gaps:
            c0, c1 = math.ceil((gw - west) / res), math.floor((ge - west) / res)
            r0, r1 = math.ceil((north - gn) / res), math.floor((north - gs) / res)
            a = ds.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
            if (a != der["nodata"]).any():
                raise ValueError(f"{item.derived}: {int((a != der['nodata']).sum())} valid cells inside "
                                 f"the known gap {[gw, gs, ge, gn]}")
    der["gaps"], der["gaps_recorded_at"] = gaps, _now()
    return True


def verify_cog(src: str, cog: Path, *, lossless: bool) -> dict:
    """Shape/CRS/nodata equality and, for lossless copies, pixel equality on
    sampled windows (the four corners and the centre)."""
    import numpy as np
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(src) as a, rasterio.open(cog) as b:
        if b.crs is None or b.crs.to_epsg() != 3006:
            raise ValueError(f"{cog}: CRS {b.crs} is not EPSG:3006")
        if b.nodata != a.nodata or b.dtypes[0] != a.dtypes[0]:
            raise ValueError(f"{cog}: dtype/nodata changed ({a.dtypes[0]},{a.nodata}) -> ({b.dtypes[0]},{b.nodata})")
        rec = {"width": b.width, "height": b.height, "res_m": float(b.res[0]),
               "nodata": b.nodata, "dtype": b.dtypes[0],
               "bounds": [float(v) for v in b.bounds], "blocks": list(b.block_shapes[0])}
        if not lossless:
            return rec
        if (a.width, a.height) != (b.width, b.height) or a.transform != b.transform:
            raise ValueError(f"{cog}: grid changed")
        n = 1024
        spots = [(0, 0), (0, max(0, a.width - n)), (max(0, a.height - n), 0),
                 (max(0, a.height - n), max(0, a.width - n)),
                 (max(0, a.height // 2 - n // 2), max(0, a.width // 2 - n // 2))]
        for r, c in spots:
            win = Window(c, r, min(n, a.width - c), min(n, a.height - r))
            if not np.array_equal(a.read(1, window=win), b.read(1, window=win)):
                raise ValueError(f"{cog}: pixel mismatch at window {win}")
        rec["verified_windows"] = len(spots)
        return rec


def vrt_nodata(vrt: str) -> tuple[int, int]:
    """``(srcnodata, vrtnodata)`` for a product mosaic.

    The county rasters are rectangles that overlap their neighbours. Grunddata
    pads the area outside the county polygon with its nodata value (-1), and
    so does every Trädhöjd county raster once ``mask_to_county`` has run
    (as published, county 23 pads it with 0 and county 24 carries heights
    there), so overlapping sources compose correctly. The markfukt
    rasters pad it with the *fill code 255* (their nodata 65535 does not
    occur), so 255 must be the transparent value when the counties are
    mosaicked; it is nodata to every reader anyway (0 and 255 both map to
    "no data" in the service encoding). Measured 2026-10-13: with 65535 as
    the source nodata, half of Sweden read a neighbour's 255 padding.
    """
    if vrt.startswith("markfukt/"):
        return 255, MARKFUKT_NODATA
    return -1, -1


MARKFUKT_NODATA = 65535

#: Rasters whose valid cells mark a county's polygon, with the values outside
#: it: ``skg_open.COUNTY_MASKS``, which this script does not import. Neither
#: alone covers every NFI top-up tile (grunddata misses tiles in 01 and 03,
#: markfukt one in 25); their union does, so a rewrite needs both.
COUNTY_MASKS = (("grunddata/omdrev2/volym_", (-1,)), ("markfukt/markfukt_", (255, MARKFUKT_NODATA)))


def mask_to_county(item: Item, manifest: dict, dest: Path) -> bool:
    """Set a derived Trädhöjd county raster to nodata outside its county.

    The county rasters overlap their neighbours, and in the mosaic a later
    source wins wherever it is valid. As published, county 23 pads the area
    outside the county with 0 and county 24's tiles carry heights there
    (measured 2026-10-22: 92 % 0 and 8 % heights outside 23, 6 % heights
    outside 24), so 23's zeros covered counties 20-22. A cell is inside when
    any of the county's ``COUNTY_MASKS`` rasters is valid there, read onto
    the 10 m grid (nearest); every other cell becomes -1.

    Runs once per raster (record ``county_mask``, with each mask source's
    sha256) through ``_rewrite_county_raster``. Raises, before touching the
    raster, unless both mask sources are staged: masking against one would
    clear the heights only the other marks inside, for good. Returns whether
    the record changed."""
    der = manifest["derived"].get(item.derived)
    if item.product != "tradhojd" or der is None or "county_mask" in der:
        return False
    sources = _county_mask_sources(item, manifest, dest)

    def clear(a, inside, win, transform):
        drop = ~inside & (a != -1)
        counts = {"cleared_cells": int(drop.sum()), "cleared_heights": int((drop & (a > 0)).sum())}
        a[~inside] = -1
        return counts

    _rewrite_county_raster(item, manifest, dest, sources, clear, kind="county_mask", suffix=".masked.tif")
    return True


#: The tiles a tiled county was published with, for ``restore_tiled_zeros``:
#: name = the tile's south-west corner in ``tile_m`` units (northing_easting),
#: bound to the zip's sha256. Read 2026-10-22 from county 24's shipped
#: vasterbottens.vrt (689 SimpleSources; every name matched its DstRect).
#: Inside a published tile every 1 m cell is valid (no nodata tag), so a -1
#: there came from the retired all-zero rule or a recorded gap; outside every
#: tile a -1 is genuine nodata.
TILED_FOOTPRINTS = {
    "sksTradhojd24.zip": {
        "sha256": "05c5c6a1f8ad3f3aa0ff093222482b645073fef6cd81d9561f1dbfe548703ac4",
        "tile_m": 10000,
        "tiles": (
            "703_71 703_72 703_73 704_71 704_72 704_73 704_74 704_75 705_70 705_71 705_72 705_73 705_74 "
            "705_75 706_69 706_70 706_71 706_72 706_73 706_74 706_75 706_76 706_78 706_79 707_68 707_69 "
            "707_70 707_71 707_72 707_73 707_74 707_75 707_76 707_77 707_78 707_79 708_60 708_61 708_62 "
            "708_63 708_64 708_66 708_67 708_68 708_69 708_70 708_71 708_72 708_73 708_74 708_75 708_76 "
            "708_77 708_78 708_79 709_58 709_59 709_60 709_61 709_62 709_63 709_64 709_65 709_66 709_67 "
            "709_68 709_69 709_70 709_71 709_72 709_73 709_74 709_75 709_76 709_77 709_78 709_79 710_57 "
            "710_58 710_59 710_60 710_61 710_62 710_63 710_64 710_65 710_66 710_67 710_68 710_69 710_70 "
            "710_71 710_72 710_73 710_74 710_75 710_76 710_77 710_78 710_79 711_56 711_57 711_58 711_59 "
            "711_60 711_61 711_62 711_63 711_64 711_65 711_66 711_67 711_68 711_69 711_70 711_71 711_72 "
            "711_73 711_74 711_75 711_76 711_77 711_78 711_79 712_54 712_55 712_56 712_57 712_58 712_59 "
            "712_60 712_61 712_62 712_63 712_64 712_65 712_66 712_67 712_68 712_69 712_70 712_71 712_72 "
            "712_73 712_74 712_75 712_76 712_77 712_78 712_79 713_54 713_55 713_56 713_57 713_58 713_59 "
            "713_60 713_61 713_62 713_63 713_64 713_65 713_66 713_67 713_68 713_69 713_70 713_71 713_72 "
            "713_73 713_74 713_75 713_76 713_77 713_78 713_79 713_80 714_52 714_53 714_54 714_55 714_56 "
            "714_57 714_58 714_59 714_60 714_61 714_62 714_63 714_64 714_65 714_66 714_67 714_68 714_69 "
            "714_70 714_71 714_72 714_73 714_74 714_75 714_76 714_77 714_78 714_79 714_80 714_81 715_52 "
            "715_53 715_54 715_55 715_56 715_57 715_58 715_59 715_60 715_61 715_62 715_63 715_64 715_65 "
            "715_66 715_67 715_68 715_69 715_70 715_71 715_72 715_73 715_74 715_75 715_76 715_77 715_78 "
            "715_79 715_80 715_81 716_51 716_52 716_53 716_54 716_55 716_56 716_57 716_58 716_59 716_60 "
            "716_61 716_62 716_63 716_64 716_65 716_66 716_67 716_68 716_69 716_70 716_71 716_72 716_73 "
            "716_74 716_75 716_76 716_77 716_78 716_79 716_80 716_81 717_50 717_51 717_52 717_53 717_54 "
            "717_55 717_56 717_57 717_58 717_59 717_60 717_61 717_62 717_63 717_64 717_65 717_66 717_67 "
            "717_68 717_69 717_70 717_71 717_72 717_73 717_74 717_75 717_76 717_77 717_78 717_79 717_80 "
            "717_81 718_49 718_50 718_51 718_52 718_53 718_54 718_55 718_56 718_57 718_58 718_59 718_60 "
            "718_61 718_62 718_63 718_64 718_65 718_66 718_67 718_68 718_69 718_70 718_71 718_72 718_73 "
            "718_74 718_75 718_76 718_77 718_78 718_79 718_80 718_81 719_48 719_49 719_50 719_51 719_52 "
            "719_53 719_54 719_55 719_56 719_57 719_58 719_59 719_60 719_61 719_62 719_63 719_64 719_65 "
            "719_66 719_67 719_68 719_69 719_70 719_71 719_72 719_73 719_74 719_75 719_76 719_77 719_78 "
            "719_79 719_80 720_48 720_49 720_50 720_51 720_52 720_53 720_54 720_55 720_56 720_57 720_58 "
            "720_59 720_60 720_61 720_62 720_63 720_64 720_65 720_66 720_67 720_68 720_69 720_70 720_71 "
            "720_72 720_73 720_74 720_75 720_76 720_77 720_78 720_79 721_47 721_48 721_49 721_50 721_51 "
            "721_52 721_53 721_54 721_55 721_56 721_57 721_58 721_59 721_60 721_61 721_62 721_63 721_64 "
            "721_65 721_66 721_67 721_68 721_69 721_70 721_71 721_72 721_73 721_74 721_75 721_76 721_77 "
            "721_78 721_79 721_80 722_46 722_47 722_48 722_49 722_50 722_51 722_52 722_53 722_54 722_55 "
            "722_56 722_57 722_58 722_59 722_60 722_61 722_62 722_63 722_64 722_65 722_66 722_67 722_68 "
            "722_69 722_70 722_71 722_72 722_73 722_74 722_75 722_76 722_77 722_78 722_79 722_80 723_46 "
            "723_47 723_48 723_49 723_50 723_51 723_52 723_53 723_54 723_55 723_56 723_57 723_58 723_59 "
            "723_60 723_61 723_62 723_63 723_64 723_65 723_66 723_67 723_68 723_69 723_70 723_71 723_72 "
            "723_73 723_74 723_75 723_76 723_77 723_78 723_79 723_80 724_47 724_48 724_49 724_50 724_51 "
            "724_52 724_53 724_54 724_55 724_56 724_57 724_58 724_59 724_60 724_61 724_62 724_63 724_64 "
            "724_65 724_66 724_67 724_68 724_69 724_71 724_72 724_73 724_74 724_75 724_76 724_77 724_78 "
            "724_79 725_47 725_48 725_49 725_50 725_51 725_52 725_53 725_54 725_55 725_56 725_57 725_58 "
            "725_59 725_60 725_61 725_62 725_63 725_64 725_65 725_66 725_67 725_68 725_72 725_73 725_74 "
            "725_75 726_47 726_48 726_49 726_50 726_51 726_52 726_53 726_54 726_55 726_56 726_57 726_58 "
            "726_59 726_60 726_61 726_62 726_63 726_64 726_65 726_66 726_67 727_47 727_48 727_49 727_50 "
            "727_51 727_52 727_53 727_54 727_55 727_56 727_57 727_58 727_59 727_60 727_61 727_62 727_63 "
            "727_64 727_65 728_47 728_48 728_49 728_50 728_51 728_52 728_53 728_54 728_55 728_56 728_57 "
            "728_58 728_59 728_60 728_61 728_62 728_63 729_47 729_48 729_49 729_50 729_51 729_52 729_53 "
            "729_54 729_55 729_56 729_57 729_58 729_59 729_60 729_61 730_48 730_49 730_50 730_51 730_52 "
            "730_53 730_54 730_55 730_56 730_57 730_58 730_59 730_60 731_47 731_48 731_49 731_50 731_51 "
            "731_52 731_53 731_54 731_55 731_56 731_57 731_58 731_59 732_47 732_48 732_49 732_50 732_51 "
            "732_52 732_53 732_54 732_55 732_56 732_57 732_58 733_47 733_48 733_49 733_50 733_51 733_52 "
            "733_53 733_54 733_55 733_56 734_50 734_51 734_52 734_53 734_54 734_55 735_51 735_52 735_53"
        ).split(),
    },
}


def restore_tiled_zeros(item: Item, manifest: dict, dest: Path) -> bool:
    """Give a tiled county derived under the retired all-zero rule its 0 back
    inside the county.

    County 24 (run r2, 2026-10-21) was derived with ``all_zero_cells_nodata``:
    a 10 m cell whose 1 m cells were all 0 became -1, which kept the tiles' 0
    fill off the neighbours in the mosaic. ``mask_to_county`` does that job
    now. Since 2026-10-22 readers return -1 as NaN (no data), so those
    treeless cells inside the county, and the product's lakes, must read 0
    again. The derived raster alone cannot tell such a cell from one no tile
    covered (both are -1), so the restore is bound to the source: only cells
    inside the county mask and inside a published tile (``TILED_FOOTPRINTS``,
    for the zip with the recorded sha256) become 0, never a cell touching a
    recorded gap. Raises when no footprint matches the recorded zip. Runs once,
    after ``mask_to_county`` (record ``tiled_zeros_restored``). Returns whether
    the record changed."""
    import numpy as np

    der = manifest["derived"].get(item.derived)
    if (item.product != "tradhojd" or der is None or not der.get("all_zero_cells_nodata")
            or "county_mask" not in der or "tiled_zeros_restored" in der):
        return False
    foot = TILED_FOOTPRINTS.get(item.name)
    if not foot or manifest["files"].get(item.name, {}).get("sha256") != foot["sha256"]:
        raise ValueError(f"{item.derived}: no published tile footprint for the recorded {item.name}; "
                         "cannot tell its treeless cells from cells no tile covered")
    sources = _county_mask_sources(item, manifest, dest)
    gaps = der.get("gaps", [])
    tile_m = foot["tile_m"]
    blocks = {tuple(int(v) for v in t.split("_")) for t in foot["tiles"]}

    def restore(a, inside, win, transform):
        res, west = transform.a, transform.c
        top = transform.f - win.row_off * res
        rows = (top - (np.arange(a.shape[0]) + 0.5) * res) // tile_m
        cols = (west + (np.arange(a.shape[1]) + 0.5) * res) // tile_m
        in_tile = np.zeros(a.shape, bool)
        for r in np.unique(rows):
            covered = np.array([(int(r), int(c)) in blocks for c in np.unique(cols)])
            in_tile[rows == r] = covered[np.searchsorted(np.unique(cols), cols)]
        in_gap = np.zeros(a.shape, bool)
        for gw, gs, ge, gn in gaps:                    # every cell touching a gap stays -1
            r0, r1 = math.floor((top - gn) / res), math.ceil((top - gs) / res)
            c0, c1 = math.floor((gw - west) / res), math.ceil((ge - west) / res)
            in_gap[max(r0, 0):max(r1, 0), max(c0, 0):max(c1, 0)] = True
        back = inside & in_tile & (a == -1) & ~in_gap
        a[back] = 0
        return {"restored_cells": int(back.sum())}

    _rewrite_county_raster(item, manifest, dest, sources, restore, kind="tiled_zeros_restored",
                           suffix=".restored.tif")
    return True


def _county_mask_sources(item: Item, manifest: dict, dest: Path) -> list[tuple[str, tuple]]:
    """``(path, values outside)`` of the county's ``COUNTY_MASKS`` rasters;
    raises unless every one is recorded with a sha256 and on disk."""
    sources, missing = [], []
    for prefix, invalid in COUNTY_MASKS:
        rel = f"{prefix}{item.county}.tif"
        if manifest["derived"].get(rel, {}).get("sha256") and (dest / rel).exists():
            sources.append((rel, invalid))
        else:
            missing.append(rel)
    if missing:
        raise ValueError(f"{item.derived}: county mask incomplete for county {item.county}, "
                         f"not staged: {missing}; the union of {[p for p, _ in COUNTY_MASKS]} is required")
    return sources


def _rewrite_county_raster(item: Item, manifest: dict, dest: Path, sources: list[tuple[str, tuple]],
                           edit, *, kind: str, suffix: str) -> dict[str, int]:
    """Rewrite ``item``'s derived raster in strips of 512 rows through
    ``edit(a, inside, window, transform)``, which changes the strip ``a`` in
    place and returns counts to add up. ``inside`` is the county on the strip:
    any of ``sources`` valid, read onto the 10 m grid (nearest).

    The rewrite goes to a sibling file (``suffix``) and is re-verified. The
    record it leads to (sha256, bytes, value statistics and ``kind``, which
    holds the counts and each source's sha256) is saved first as
    ``pending_rewrite``; then the file replaces the raster, and the record is
    applied and saved again. ``finish_pending_rewrite`` completes an
    interruption between the two saves. On a failure before the first save,
    the raster and its record stay as they were. Returns the summed counts."""
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window, bounds as window_bounds, from_bounds

    der = manifest["derived"][item.derived]
    out = dest / item.derived
    partial = out.with_name(out.name + suffix)
    totals: dict[str, int] = {}
    n_valid = 0
    vmin = vmax = None
    try:
        with rasterio.Env(**SERIAL_VRT), rasterio.open(out) as src, contextlib.ExitStack() as stack:
            masks = [(stack.enter_context(rasterio.open(dest / rel)), invalid) for rel, invalid in sources]
            with rasterio.open(partial, "w", **{**src.profile, "BIGTIFF": "IF_SAFER"}) as dst:
                for row in range(0, src.height, 512):
                    win = Window(0, row, src.width, min(512, src.height - row))
                    a = src.read(1, window=win)
                    inside = np.zeros(a.shape, bool)
                    for m, invalid in masks:
                        v = m.read(1, window=from_bounds(*window_bounds(win, src.transform), m.transform),
                                   out_shape=a.shape, boundless=True, fill_value=invalid[0],
                                   resampling=Resampling.nearest)
                        inside |= ~np.isin(v, invalid)
                    for key, n in edit(a, inside, win, src.transform).items():
                        totals[key] = totals.get(key, 0) + n
                    dst.write(a, 1, window=win)
                    valid = a[a != -1]
                    if valid.size:
                        n_valid += int(valid.size)
                        vmin = int(valid.min()) if vmin is None else min(vmin, int(valid.min()))
                        vmax = int(valid.max()) if vmax is None else max(vmax, int(valid.max()))
        check = verify_tradhojd_10m(partial, {"width": der["width"], "height": der["height"],
                                              "bounds": der["bounds"], "valid_cells": n_valid,
                                              "value_min_dm": vmin, "value_max_dm": vmax})
        new_sha = sha256_of(partial)
        der["pending_rewrite"] = {
            "kind": kind, "file": partial.name, "from_sha256": der["sha256"], "to_sha256": new_sha,
            "apply": {"valid_cells": n_valid, "value_min_dm": vmin, "value_max_dm": vmax,
                      "bytes": partial.stat().st_size, "sha256": new_sha, **check,
                      kind: {"sources": {rel: manifest["derived"][rel]["sha256"] for rel, _ in sources},
                             **totals, "at": _now()}}}
        save_manifest(dest, manifest)
    except BaseException:
        der.pop("pending_rewrite", None)
        partial.unlink(missing_ok=True)
        raise
    os.replace(partial, out)
    der.update(der.pop("pending_rewrite")["apply"])
    save_manifest(dest, manifest)
    return totals


def finish_pending_rewrite(item: Item, manifest: dict, dest: Path) -> bool:
    """Finish or drop a county-raster rewrite interrupted between its two
    manifest saves (``_rewrite_county_raster``), going by the raster's
    sha256:

    * it is the new raster: apply the pending record;
    * it is the old one and the verified sibling is intact: replace, then apply;
    * it is the old one otherwise: drop the pending record, and the rewrite
      runs again;
    * anything else raises rather than guess.

    Runs before the raw-deletion check, which compares the raster with its
    record. Returns whether the record changed."""
    der = manifest["derived"].get(item.derived)
    if not der or "pending_rewrite" not in der:
        return False
    pending = der["pending_rewrite"]
    out = dest / item.derived
    partial = out.with_name(pending["file"])
    now = sha256_of(out)
    if now == pending["from_sha256"] and partial.exists() and sha256_of(partial) == pending["to_sha256"]:
        os.replace(partial, out)
        now = pending["to_sha256"]
    if now == pending["to_sha256"]:
        der.update(der.pop("pending_rewrite")["apply"])
    elif now == pending["from_sha256"]:
        del der["pending_rewrite"]
    else:
        raise RuntimeError(f"{item.derived}: sha256 {now[:12]} is neither side of the interrupted "
                           f"{pending['kind']}; refusing to guess")
    partial.unlink(missing_ok=True)
    save_manifest(dest, manifest)
    return True


def build_vrt(dest: Path, vrt: str, rasters: list[str], srcnodata: int, vrtnodata: int) -> dict:
    out = dest / vrt
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".vrt.tmp")
    subprocess.run(["gdalbuildvrt", "-q", "-srcnodata", str(srcnodata), "-vrtnodata", str(vrtnodata),
                    "-resolution", "highest", str(tmp), *[str(dest / r) for r in rasters]],
                   check=True)
    text = tmp.read_text()
    # relative source paths keep the VRT valid when the mirror is mounted elsewhere
    text = text.replace(str(dest) + "/", "")
    text = text.replace('relativeToVRT="0"', 'relativeToVRT="1"')
    text = text.replace('<SourceFilename relativeToVRT="1">' + vrt.rsplit("/", 1)[0] + "/",
                        '<SourceFilename relativeToVRT="1">')
    out.write_text(text)
    tmp.unlink()
    import rasterio
    with rasterio.open(out) as ds:
        if ds.crs.to_epsg() != 3006:
            raise ValueError(f"{out}: CRS {ds.crs}")
        return {"sources": rasters, "width": ds.width, "height": ds.height,
                "res_m": float(ds.res[0]), "nodata": ds.nodata, "srcnodata": srcnodata,
                "built_at": _now()}


# ── main ────────────────────────────────────────────────────────────────────

LOCK_NAME = "MANIFEST.lock"


class ManifestLocked(RuntimeError):
    """Another writer holds the mirror's MANIFEST.json."""


@contextlib.contextmanager
def manifest_lock(dest: Path, owner: str):
    """Exclusive lock for every writer of ``<dest>/MANIFEST.json``.

    The manifest is loaded once and rewritten whole, so two writers would
    erase each other's records (review N6 on #85, B2 on #86; Tobias chose
    this lock on 2026-10-19). It is taken before the first read and held
    past the last save; the file is created with O_CREAT | O_EXCL.

    SIGTERM (Kubernetes' deadline, eviction or delete) is turned into
    ``SystemExit(143)`` while the lock is held, so the lock is released on
    every stop except SIGKILL or the OOM killer. The file carries a random
    token and is removed only if it still holds this writer's token, so a
    writer never removes a lock another writer took after this one's was
    cleared by hand. A stale lock is cleared by hand, only after confirming
    that no Job is ACTIVE."""
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / LOCK_NAME
    record = json.dumps({"owner": owner, "since": _now(), "host": os.uname().nodename,
                         "pid": os.getpid(), "token": secrets.token_hex(16)})
    previous = signal.signal(signal.SIGTERM, _exit_on_sigterm)
    try:
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            raise ManifestLocked(f"REFUSING: {path} exists ({path.read_text().strip()}); another writer holds "
                                 "the mirror manifest. Remove it by hand only if no stage or drop Job is "
                                 "running.") from None
        try:
            with os.fdopen(fd, "w") as f:
                f.write(record)
            yield
        finally:
            try:
                if path.read_text() == record:
                    path.unlink()
            except FileNotFoundError:
                pass
    finally:
        signal.signal(signal.SIGTERM, previous)


def _exit_on_sigterm(signum, frame):
    raise SystemExit(128 + signum)


def main(argv: list[str] | None = None) -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--dest", type=Path)
    pre.add_argument("--dry-run", action="store_true")
    known, _ = pre.parse_known_args(argv)
    if known.dest is None or known.dry_run:
        return _main(argv)
    with manifest_lock(known.dest, "stage_skg_open"):
        return _main(argv)


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--products", nargs="+", default=["grunddata", "markfukt"],
                   choices=["grunddata", "markfukt", "tradhojd"])
    p.add_argument("--omdrev", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--counties", nargs="+", default=list(COUNTIES))
    p.add_argument("--source-url", default=SOURCE_URL)
    p.add_argument("--tradhojd-method", default="p90", choices=list(TRADHOJD_STATS),
                   help="10 m statistic of the 1 m Trädhöjd cells; p90 is closest to the "
                        "retired proxy's height channel (see docs/data/skg_open_mirror.md)")
    p.add_argument("--tradhojd-workers", type=int, default=1,
                   help="processes reducing Trädhöjd strips in parallel; the raster is the same for any value")
    p.add_argument("--drop-raw", action="store_true",
                   help="delete the raw zip after its derived raster is verified (tradhojd only)")
    p.add_argument("--accept-upstream-change", action="store_true",
                   help="when a re-downloaded zip no longer matches its recorded sha256, record the "
                        "new digest (old kept under superseded_sha256) and re-derive instead of stopping")
    p.add_argument("--work-dir", type=Path, default=None, help="scratch for the extracted 1 m raster (default <dest>/_work)")
    p.add_argument("--min-free-gib", type=float, default=0.0,
                   help="stop cleanly, before a download, if free space after it would fall below this")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    with _request(args.source_url) as resp:
        index = parse_index(resp.read().decode("utf-8", "replace"))
    items = plan(args.products, args.omdrev, args.counties, index)
    total = sum(index[i.name]["bytes"] for i in items)
    print(f"plan: {len(items)} files, {total / 2**30:.1f} GiB from {args.source_url}", flush=True)
    for it in items:
        print(f"  {index[it.name]['bytes']:>12,d}  {it.name}  ->  {it.derived}")
    if args.dry_run:
        return 0

    dest: Path = args.dest
    dest.mkdir(parents=True, exist_ok=True)
    work = args.work_dir or dest / "_work"
    manifest = load_manifest(dest)
    manifest["source_url"] = args.source_url
    manifest["history"].append({"at": _now(), "argv": sys.argv[1:], "git_sha": _git_sha()})
    save_manifest(dest, manifest)

    done = skipped = 0
    stopped = None
    for it in items:
        if finish_pending_rewrite(it, manifest, dest):
            print(f"    finished an interrupted rewrite of {it.derived}", flush=True)
        raw = dest / "raw" / it.name
        der = manifest["derived"].get(it.derived)
        if (args.drop_raw and it.product == "tradhojd" and der and der.get("sha256")
                and raw.exists() and (dest / it.derived).exists()
                and not manifest["files"].get(it.name, {}).get("raw_deleted_at")):
            # crashed after the verified derivation was recorded, before the delete:
            # re-verify the raster on disk against its record, then finish the delete
            out = dest / it.derived
            if sha256_of(out) != der["sha256"]:
                raise RuntimeError(f"{it.derived}: sha256 differs from its record; refusing to delete {it.name}")
            verify_tradhojd_10m(out, der)
            raw.unlink()
            manifest["files"][it.name]["raw_deleted_at"] = _now()
            save_manifest(dest, manifest)
            vrt = "tradhojd/tradhojd_10m.vrt"
            manifest["vrt"][vrt] = build_vrt(dest, vrt, staged_vrt_sources(manifest, dest, vrt), *vrt_nodata(vrt))
            save_manifest(dest, manifest)
            print(f"    resumed {it.name}: derived raster re-verified, raw zip deleted", flush=True)
        if is_done(it, manifest, dest):
            if record_known_gaps(it, manifest, dest):
                save_manifest(dest, manifest)
                print(f"    recorded the known upstream gaps of {it.derived}", flush=True)
            if mask_to_county(it, manifest, dest):
                print(f"    masked {it.derived} to its county: "
                      f"{manifest['derived'][it.derived]['county_mask']['cleared_cells']} cells cleared", flush=True)
            if restore_tiled_zeros(it, manifest, dest):
                print(f"    restored the treeless 0 of {it.derived}: "
                      f"{manifest['derived'][it.derived]['tiled_zeros_restored']['restored_cells']} cells", flush=True)
            skipped += 1
            continue
        t0 = time.time()
        raw = dest / "raw" / it.name
        rec = manifest["files"].get(it.name) or {}
        if args.drop_raw and it.product == "tradhojd":
            mine = {it.name, it.name + ".part", it.name + ".upstream-changed"}
            others = sorted(q.name for q in (dest / "raw").glob("sksTradhojd*") if q.name not in mine)
            if others:
                raise RuntimeError(f"refusing {it.name}: raw Trädhöjd zips still on disk {others}; "
                                   "one county at a time")
        if not rec.get("sha256") or not raw.exists():
            free = shutil.disk_usage(dest).free
            if (free - index[it.name]["bytes"]) / 2**30 < args.min_free_gib:
                stopped = (f"STOP before {it.name}: {free / 2**30:.1f} GiB free, the download needs "
                           f"{index[it.name]['bytes'] / 2**30:.1f} GiB, floor {args.min_free_gib:.0f} GiB")
                print(stopped, flush=True)
                break
            print(f"[{done + skipped + 1}/{len(items)}] download {it.name} ({index[it.name]['bytes'] / 2**20:.0f} MiB)", flush=True)
            dl = download(args.source_url + urllib.parse.quote(it.name), raw, index[it.name]["bytes"])
            new_sha = sha256_of(raw)
            kept = check_recovered(rec, new_sha, it.name, accept=args.accept_upstream_change,
                                   raw=raw, dest=dest)
            rec = {**kept, "url": args.source_url + urllib.parse.quote(it.name), "bytes": raw.stat().st_size,
                   "sha256": new_sha, "index_modified": index[it.name]["modified"],
                   "downloaded_at": _now(), "product": it.product, "omdrev": it.omdrev,
                   "variable": it.variable, "county": it.county,
                   **{k: v for k, v in dl.items() if k != "attempts"}}
            rec.pop("raw_deleted_at", None)
            if "superseded_sha256" in rec:
                manifest["derived"].pop(it.derived, None)      # must be re-derived from the new bytes
            manifest["files"][it.name] = rec
            save_manifest(dest, manifest)
        # what gdal_source extracts first: every entry of a tiled county, a Deflate64
        # entry, or a Deflate entry read by parallel workers
        if (entries := zip_tiles(raw)) is not None:
            inflate = sum(i.file_size for i in entries)
        elif ((info := zip_tif_entry(raw)).compress_type == ZIP_DEFLATE64
              or (it.product == "tradhojd" and args.tradhojd_workers > 1)):
            inflate = info.file_size
        else:
            inflate = 0
        if inflate:
            free = shutil.disk_usage(dest).free
            if (free - inflate) / 2**30 < args.min_free_gib:
                stopped = (f"STOP before inflating {it.name}: {free / 2**30:.1f} GiB free, inflation needs "
                           f"{inflate / 2**30:.1f} GiB, floor {args.min_free_gib:.0f} GiB "
                           "(the raw zip stays recorded; a re-run resumes here)")
                print(stopped, flush=True)
                break
        print(f"    convert -> {it.derived}", flush=True)
        manifest["derived"][it.derived] = convert(it, raw, dest / it.derived, work, args.tradhojd_method,
                                                  tradhojd_workers=args.tradhojd_workers)
        save_manifest(dest, manifest)              # the verified derivation is on record first
        mask_to_county(it, manifest, dest)         # before the raw zip goes: a failed mask can be redone
        if args.drop_raw and it.product == "tradhojd":
            raw.unlink()
            manifest["files"][it.name]["raw_deleted_at"] = _now()
            save_manifest(dest, manifest)
            # the mosaic always matches what is staged, even if a later county stops the run
            vrt = "tradhojd/tradhojd_10m.vrt"
            manifest["vrt"][vrt] = build_vrt(dest, vrt, staged_vrt_sources(manifest, dest, vrt), *vrt_nodata(vrt))
            save_manifest(dest, manifest)
        done += 1
        print(f"    done in {time.time() - t0:.0f}s; free {shutil.disk_usage(dest).free / 2**30:.1f} GiB", flush=True)

    for vrt in vrt_targets(items):
        rasters = staged_vrt_sources(manifest, dest, vrt)
        manifest["vrt"][vrt] = build_vrt(dest, vrt, rasters, *vrt_nodata(vrt))
        print(f"vrt {vrt}: {len(rasters)} sources", flush=True)
    save_manifest(dest, manifest)
    staged = sum(r["bytes"] for r in manifest["files"].values() if not r.get("raw_deleted_at"))
    derived = sum(r["bytes"] for r in manifest["derived"].values())
    print(f"converted {done}, skipped {skipped}; raw {staged / 2**30:.1f} GiB, derived {derived / 2**30:.1f} GiB", flush=True)
    if stopped:
        print(stopped, flush=True)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
