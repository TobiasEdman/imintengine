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
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
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
    manifest["updated_at"] = _now()
    manifest["git_sha"] = _git_sha()
    path = dest / "MANIFEST.json"
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


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


# ── extract / convert ───────────────────────────────────────────────────────

def zip_tif_entry(zpath: Path) -> zipfile.ZipInfo:
    with zipfile.ZipFile(zpath) as z:
        tifs = [i for i in z.infolist() if i.filename.lower().endswith(".tif")]
    if len(tifs) != 1:
        raise ValueError(f"{zpath.name}: expected one .tif entry, found {[i.filename for i in tifs]}")
    return tifs[0]


def gdal_source(zpath: Path, work: Path) -> tuple[str, Path | None]:
    """A GDAL-openable path for the zip's raster.

    Deflate (method 8) is read in place through ``/vsizip/``. Deflate64
    (method 9, used by the whole-country and most Trädhöjd zips) is not
    supported by GDAL or ``zipfile``; it is inflated with ``inflate64`` to a
    temporary file that the caller removes.
    """
    info = zip_tif_entry(zpath)
    if info.compress_type == ZIP_DEFLATE or info.compress_type == zipfile.ZIP_STORED:
        return f"/vsizip/{zpath}/{info.filename}", None
    if info.compress_type != ZIP_DEFLATE64:
        raise ValueError(f"{zpath.name}: unsupported zip method {info.compress_type}")
    import inflate64

    work.mkdir(parents=True, exist_ok=True)
    tmp = work / info.filename
    inflater = inflate64.Inflater()
    with open(zpath, "rb") as raw, open(tmp, "wb") as out:
        # local file header: 30 fixed bytes + name + extra, then the deflate64 stream
        raw.seek(info.header_offset + 26)
        fnlen, exlen = int.from_bytes(raw.read(2), "little"), int.from_bytes(raw.read(2), "little")
        raw.seek(info.header_offset + 30 + fnlen + exlen)
        remaining = info.compress_size
        while remaining > 0:
            chunk = raw.read(min(16 << 20, remaining))
            if not chunk:
                raise OSError(f"{zpath.name}: truncated zip entry")
            remaining -= len(chunk)
            out.write(inflater.inflate(chunk))
    if tmp.stat().st_size != info.file_size:
        raise OSError(f"{zpath.name}: inflated {tmp.stat().st_size} bytes, expected {info.file_size}")
    return str(tmp), tmp


def convert(item: Item, zpath: Path, out: Path, work: Path, tradhojd_method: str) -> dict:
    """Write the derived COG for ``item``; returns the record for the manifest."""
    src, tmp = gdal_source(zpath, work)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".part.tif")
    try:
        if item.product == "tradhojd":
            check = derive_tradhojd_10m(src, partial, tradhojd_method)
            method = f"block {tradhojd_method} of the 1 m cells per 10 m cell, nodata-aware, int16 dm"
        else:
            subprocess.run(["gdal_translate", "-q", *COG_OPTS, src, str(partial)], check=True)
            method = "gdal_translate lossless"
            check = verify_cog(src, partial, lossless=True)
        os.replace(partial, out)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()
        if partial.exists():
            partial.unlink()
    return {"from": item.name, "method": method, "bytes": out.stat().st_size,
            "sha256": sha256_of(out), "created_at": _now(), **check}


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


def derive_tradhojd_10m(src: str, out: Path, stat: str) -> dict:
    """Reduce the 1 m Trädhöjd raster to the NMD 10 m lattice, streaming
    ten source rows at a time (memory-bounded, sequential: works on a
    strip TIFF read through /vsizip/)."""
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
        rows_per_chunk = 10 * 64
        with rasterio.open(out, "w", **prof) as dst:
            for r in range(0, ds.height, rows_per_chunk):
                n = min(rows_per_chunk, ds.height - r)
                block = ds.read(1, window=Window(0, r, ds.width, n))
                dst.write(block_stat(block, stat, nod), 1, window=Window(0, r // 10, ow, n // 10))
    with rasterio.open(out) as b:
        return {"width": b.width, "height": b.height, "res_m": 10.0, "nodata": b.nodata,
                "dtype": b.dtypes[0], "bounds": [float(v) for v in b.bounds],
                "blocks": list(b.block_shapes[0]), "statistic": stat}


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
    and Trädhöjd pad the area outside the county polygon with their nodata
    value (-1), so overlapping sources compose correctly. The markfukt
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

def main(argv: list[str] | None = None) -> int:
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
    p.add_argument("--drop-raw", action="store_true",
                   help="delete the raw zip after its derived raster is verified (tradhojd only)")
    p.add_argument("--work-dir", type=Path, default=None, help="scratch for Deflate64 inflation (default <dest>/_work)")
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
    for it in items:
        if is_done(it, manifest, dest):
            skipped += 1
            continue
        t0 = time.time()
        raw = dest / "raw" / it.name
        rec = manifest["files"].get(it.name) or {}
        if not rec.get("sha256") or not raw.exists():
            print(f"[{done + skipped + 1}/{len(items)}] download {it.name} ({index[it.name]['bytes'] / 2**20:.0f} MiB)", flush=True)
            dl = download(args.source_url + urllib.parse.quote(it.name), raw, index[it.name]["bytes"])
            rec = {"url": args.source_url + urllib.parse.quote(it.name), "bytes": raw.stat().st_size,
                   "sha256": sha256_of(raw), "index_modified": index[it.name]["modified"],
                   "downloaded_at": _now(), "product": it.product, "omdrev": it.omdrev,
                   "variable": it.variable, "county": it.county,
                   **{k: v for k, v in dl.items() if k != "attempts"}}
            manifest["files"][it.name] = rec
            save_manifest(dest, manifest)
        print(f"    convert -> {it.derived}", flush=True)
        manifest["derived"][it.derived] = convert(it, raw, dest / it.derived, work, args.tradhojd_method)
        if args.drop_raw and it.product == "tradhojd":
            raw.unlink()
            manifest["files"][it.name]["raw_deleted_at"] = _now()
        save_manifest(dest, manifest)
        done += 1
        print(f"    done in {time.time() - t0:.0f}s", flush=True)

    for vrt in vrt_targets(items):
        rasters = staged_vrt_sources(manifest, dest, vrt)
        manifest["vrt"][vrt] = build_vrt(dest, vrt, rasters, *vrt_nodata(vrt))
        print(f"vrt {vrt}: {len(rasters)} sources", flush=True)
    save_manifest(dest, manifest)
    staged = sum(r["bytes"] for r in manifest["files"].values() if not r.get("raw_deleted_at"))
    derived = sum(r["bytes"] for r in manifest["derived"].values())
    print(f"converted {done}, skipped {skipped}; raw {staged / 2**30:.1f} GiB, derived {derived / 2**30:.1f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
