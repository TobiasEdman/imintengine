#!/usr/bin/env python3
"""Can HR-VPP be read per analysis window instead of per MGRS file? (IM-026)

Read-only probes for package 4 of docs/plans/lclu_national_service.md. One
product already cached in /data/vpp_wekeo is the reference bytes:

1. layout — is the cached file tiled (a window then touches only some
   blocks), with overviews and the first IFD at the front, and how many
   compressed bytes does a 512 × 512 window at its centre touch against the
   whole file?
2. wekeo — order the same product on WEkEO HDA and request two byte ranges of
   its download URL. 206 with the cached file's bytes means Range works. Then
   read the centre window through GDAL /vsicurl and compare it with the cached
   file pixel for pixel.
3. cdse — the HR-VPP data were to move to CDSE in 2025. Ask the anonymous CDSE
   STAC catalogue for HR-VPP collections and an item for the same tile and
   year. With CDSE S3 keys (optional), look for the same object under the
   ``eodata`` bucket and compare the centre window through /vsis3.

No CDSE processing units: STAC is a catalogue and S3 is object storage, not
Sentinel Hub or openEO. A WEkEO order is free and fetches nothing by itself.
Tokens, keys and signed URL query strings are never printed or stored.

    python scripts/probe_hrvpp_access.py --vpp-dir /data/vpp_wekeo --report report.json
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
import json
import os
from pathlib import Path
import struct
import sys
import time
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import rasterio
from rasterio.session import Session
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WINDOW_PX = 512                     # the service's SOURCE_SIZE (imint/inference/lclu.py)
RANGE_BYTES = 1024
CDSE_STAC = "https://stac.dataspace.copernicus.eu/v1"
CDSE_S3_ENDPOINT = "eodata.dataspace.copernicus.eu"
CDSE_S3_BUCKET = "eodata"
STAC_KEYWORDS = ("vpp", "phenolog")
S3_KEYWORDS = ("vpp", "phenolog", "pan-european", "biophysical", "bio-geophysical")
MAX_S3_LISTINGS = 40


def public_url(url: str) -> str:
    """The URL without query or fragment: signed URLs carry credentials there."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def pick_reference(vpp_dir: Path, product: str | None = None) -> tuple[str, dict]:
    """A cached season-1 product: the named one, else the newest SOSD by name."""
    index = json.loads((vpp_dir / "index.json").read_text())
    if product is None:
        sosd = [n for n, m in index.items() if m["metric"] == "SOSD" and m["season"] == 1
                and (vpp_dir / n).exists()]
        if not sosd:
            raise SystemExit(f"No cached season-1 SOSD product in {vpp_dir}")
        product = max(sosd, key=lambda n: (index[n]["year"], n))
    if product not in index or not (vpp_dir / product).exists():
        raise SystemExit(f"{product} is not cached in {vpp_dir}")
    return product, index[product]


class S3Keys(Session):
    """Fixed keys for a non-AWS S3 endpoint as GDAL options. rasterio refuses AWS
    keys as plain Env options, and its AWSSession needs boto3 to carry them."""

    def __init__(self, access: str, secret: str, endpoint: str, *, https: bool = True,
                 region: str = "default"):
        self._options = {"AWS_ACCESS_KEY_ID": access, "AWS_SECRET_ACCESS_KEY": secret,
                         "AWS_S3_ENDPOINT": endpoint, "AWS_REGION": region,
                         "AWS_HTTPS": "YES" if https else "NO", "AWS_VIRTUAL_HOSTING": "FALSE"}

    @classmethod
    def hascreds(cls, config: dict) -> bool:
        return "AWS_ACCESS_KEY_ID" in config and "AWS_SECRET_ACCESS_KEY" in config

    def get_credential_options(self) -> dict:
        return dict(self._options)


def centre_window(width: int, height: int, size: int = WINDOW_PX) -> Window:
    return Window((width - size) // 2, (height - size) // 2, size, size)


def first_ifd_offset(path: Path) -> int:
    """Byte offset of the first IFD, from the classic or BigTIFF header."""
    head = path.open("rb").read(16)
    order = {b"II": "<", b"MM": ">"}[head[:2]]
    magic = struct.unpack(order + "H", head[2:4])[0]
    if magic == 42:
        return struct.unpack(order + "I", head[4:8])[0]
    if magic == 43:
        return struct.unpack(order + "Q", head[8:16])[0]
    raise ValueError(f"{path} is not a TIFF")


def layout(path: Path, size: int = WINDOW_PX) -> dict:
    """Block structure of the cached file and the bytes a centre window touches."""
    with rasterio.open(path) as src:
        block_h, block_w = src.block_shapes[0]
        window = centre_window(src.width, src.height, size)
        rows = range(window.row_off // block_h, (window.row_off + size - 1) // block_h + 1)
        cols = range(window.col_off // block_w, (window.col_off + size - 1) // block_w + 1)
        window_bytes = sum(src.block_size(b, r, c) for b in src.indexes for r in rows for c in cols)
        report = {"width": src.width, "height": src.height, "bands": src.count,
                  "dtype": src.dtypes[0], "crs": str(src.crs), "block": [block_h, block_w],
                  "tiled": block_w < src.width,
                  "compression": src.compression.name if src.compression else None,
                  "overviews": src.overviews(1),
                  "layout_tag": src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT")}
    file_bytes = path.stat().st_size
    return {**report, "first_ifd_offset": first_ifd_offset(path), "window_px": size,
            "window_blocks": len(rows) * len(cols), "window_bytes": window_bytes,
            "file_bytes": file_bytes, "window_fraction": round(window_bytes / file_bytes, 6)}


def range_probe(session, url: str, reference: Path) -> dict:
    """Two ranged GETs — the header and the middle of the file — against the cached bytes.

    Streams and stops after the requested length, so a server that ignores
    Range and answers 200 with the whole file costs only that much.
    """
    size = reference.stat().st_size
    expected = reference.read_bytes()
    checks, final_url = [], None
    for start in (0, size // 2):
        end = min(start + RANGE_BYTES, size) - 1
        with session.get(url, headers={"Range": f"bytes={start}-{end}"}, stream=True,
                         timeout=120, allow_redirects=True) as r:
            body = r.raw.read(end - start + 1, decode_content=True)
            final_url = r.url
            checks.append({"range": [start, end], "status": r.status_code,
                           "content_range": r.headers.get("Content-Range"),
                           "accept_ranges": r.headers.get("Accept-Ranges"),
                           "bytes_read": len(body),
                           "matches_cached": body == expected[start:end + 1],
                           "redirects": [public_url(h.headers.get("Location", ""))
                                         for h in r.history]})
    supported = all(c["status"] == 206 and c["matches_cached"] for c in checks)
    return {"url": public_url(url), "final_url": public_url(final_url),
            "range_supported": supported, "checks": checks}, final_url


def window_parity(remote: str, reference: Path, gdal_env: dict | None = None,
                  session: Session | None = None, size: int = WINDOW_PX) -> dict:
    """Read the centre window remotely and locally; equal means the same product bytes."""
    with rasterio.open(reference) as src:
        dims = (src.width, src.height, src.count)
        local = src.read(window=centre_window(src.width, src.height, size))
    started = time.monotonic()
    with rasterio.Env(session=session, GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                      **(gdal_env or {})), \
            rasterio.open(remote) as src:
        remote_dims = (src.width, src.height, src.count)
        remote_data = src.read(window=centre_window(src.width, src.height, size))
    return {"seconds": round(time.monotonic() - started, 2), "same_dims": remote_dims == dims,
            "equal": remote_dims == dims and bool(np.array_equal(remote_data, local))}


def wekeo(client, product: str, meta: dict, reference: Path) -> dict:
    """Order the cached product on WEkEO HDA, then probe Range and the centre window."""
    from imint.data import wekeo_vpp

    client.accept_tac(wekeo_vpp._HDA_DATASET_ID)
    matches = client.search({"dataset_id": wekeo_vpp._HDA_DATASET_ID,
                             "productType": meta["metric"],
                             "tileId": meta["tileId"].removeprefix("T"),
                             "itemsPerPage": 200, "startIndex": 0})
    result = next((r for r in matches if wekeo_vpp._result_filename(r) == product), None)
    if result is None:
        return {"found": False}
    location = result.results[0].get("properties", {}).get("location")
    [url] = result.get_download_urls()
    report, final_url = range_probe(client.session, url, reference)
    report = {"found": True, "location": location, **report}
    if report["range_supported"]:
        # A redirect to object storage carries its own signature; the bearer
        # token is only for the broker itself.
        redirected = public_url(final_url) != public_url(url)
        env = {} if redirected else {"GDAL_HTTP_HEADERS": f"Authorization: Bearer {client.token}"}
        report["window"] = window_parity(f"/vsicurl/{final_url}", reference, env)
    return report


def stac_items(get_json: Callable[[str], dict], meta: dict) -> dict:
    """HR-VPP collections in the CDSE catalogue and items for the reference tile and year."""
    collections, url = [], f"{CDSE_STAC}/collections"
    while url:
        page = get_json(url)
        collections += page.get("collections", [])
        url = next((link["href"] for link in page.get("links", []) if link.get("rel") == "next"), None)
    matched = [c for c in collections
               if any(k in " ".join([c.get("id", ""), c.get("title", "")]).lower()
                      for k in STAC_KEYWORDS)]
    west, south, east, north = meta["bounds_4326"]
    lon, lat = (west + east) / 2, (south + north) / 2
    year, tile = meta["year"], meta["tileId"].removeprefix("T")
    items = {}
    for collection in matched:
        page = get_json(f"{CDSE_STAC}/search?collections={collection['id']}"
                        f"&bbox={lon - 0.01},{lat - 0.01},{lon + 0.01},{lat + 0.01}"
                        f"&datetime={year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z&limit=100")
        features = page.get("features", [])
        items[collection["id"]] = {
            "in_box": len(features),
            "ids_sample": [f.get("id") for f in features[:10]],
            "tile_items": [
                {"id": f["id"],
                 "assets": {k: sorted({public_url(a["href"]),
                                       *(public_url(alt["href"])
                                         for alt in a.get("alternate", {}).values())})
                            for k, a in f.get("assets", {}).items()}}
                for f in features if tile in f.get("id", "")]}
    return {"collections_total": len(collections),
            "matched": [{"id": c["id"], "title": c.get("title")} for c in matched],
            "items_for_reference": items}


def s3_prefixes(list_prefixes: Callable[[str], list[str]], root: str = "CLMS/") -> list[str]:
    """Breadth-first walk of the bucket's prefixes whose names look like HR-VPP."""
    found, queue, listings = [], [root], 0
    while queue and listings < MAX_S3_LISTINGS:
        prefix = queue.pop(0)
        listings += 1
        for child in list_prefixes(prefix):
            name = child.removeprefix(prefix).lower()
            if any(k in name for k in S3_KEYWORDS):
                found.append(child)
                queue.append(child)
    return found


def cdse_key(location: str | None, product: str, meta: dict) -> str:
    """The ``eodata`` key to try: WEkEO's s3://<bucket>/CLMS/... location without
    its bucket, else the shape WEkEO returned when adb5e686 was written."""
    if location and location.startswith("s3://"):
        _, _, key = location.removeprefix("s3://").partition("/")
        if key.startswith("CLMS/"):
            return key
    return (f"CLMS/Pan-European/Biophysical/VPP/v01/{meta['year']}/s{meta['season']}/"
            f"{product}")


def cdse(get_json, product: str, meta: dict, reference: Path, location: str | None,
         s3=None, s3_keys: S3Keys | None = None) -> dict:
    """Catalogue first; with S3 keys, the bucket and a window parity read."""
    report = {"stac": guarded(lambda: stac_items(get_json, meta))}
    if s3 is None:
        report["s3"] = {"skipped": "no CDSE S3 keys (secret cdse-s3)"}
        return report

    def list_prefixes(prefix):
        page = s3.list_objects_v2(Bucket=CDSE_S3_BUCKET, Prefix=prefix, Delimiter="/")
        return [p["Prefix"] for p in page.get("CommonPrefixes", [])]

    key = cdse_key(location, product, meta)
    page = s3.list_objects_v2(Bucket=CDSE_S3_BUCKET, Prefix=key, MaxKeys=1)
    hit = next((o for o in page.get("Contents", []) if o["Key"] == key), None)
    s3_report = {"prefixes": s3_prefixes(list_prefixes),
                 "same_key": {"key": key, "exists": hit is not None,
                              "size": hit["Size"] if hit else None,
                              "cached_size": reference.stat().st_size}}
    if hit is not None and s3_keys is not None:
        s3_report["window"] = window_parity(f"/vsis3/{CDSE_S3_BUCKET}/{key}", reference,
                                            session=s3_keys)
    report["s3"] = s3_report
    return report


def guarded(probe: Callable[[], dict]) -> dict:
    """One probe's failure is a finding, not the end of the run."""
    try:
        return probe()
    except Exception as exc:  # noqa: BLE001 — recorded, the other probes still run
        return {"error": f"{type(exc).__name__}: {exc}"[:500]}


def summary(report: dict) -> dict:
    wekeo_report, cdse_report = report.get("wekeo", {}), report.get("cdse", {})
    return {"product": report["product"],
            "cached_tiled": report["layout"].get("tiled"),
            "window_bytes": report["layout"].get("window_bytes"),
            "file_bytes": report["layout"].get("file_bytes"),
            "wekeo_range": wekeo_report.get("range_supported"),
            "wekeo_window_equal": wekeo_report.get("window", {}).get("equal"),
            "cdse_collections": [c["id"] for c in cdse_report.get("stac", {}).get("matched", [])],
            "cdse_same_key": cdse_report.get("s3", {}).get("same_key", {}).get("exists"),
            "cdse_window_equal": cdse_report.get("s3", {}).get("window", {}).get("equal"),
            "errors": {**{k: v["error"] for k, v in report.items()
                          if isinstance(v, dict) and "error" in v},
                       **({"cdse.stac": cdse_report["stac"]["error"]}
                          if "error" in cdse_report.get("stac", {}) else {})}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--vpp-dir", type=Path, required=True)
    ap.add_argument("--product", help="cached file name; default: the newest SOSD")
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--skip-wekeo", action="store_true")
    ap.add_argument("--skip-cdse", action="store_true")
    args = ap.parse_args(argv)

    product, meta = pick_reference(args.vpp_dir, args.product)
    reference = args.vpp_dir / product
    report = {"version": 1, "product": product, "meta": meta,
              "probed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "layout": guarded(lambda: layout(reference))}
    if not args.skip_wekeo:
        from imint.data.wekeo_vpp import _hda_client

        report["wekeo"] = guarded(lambda: wekeo(_hda_client(), product, meta, reference))
    if not args.skip_cdse:
        import requests

        def get_json(url):
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            return r.json()

        access, secret = (os.environ.get("CDSE_S3_ACCESS_KEY"),
                          os.environ.get("CDSE_S3_SECRET_KEY"))
        s3 = s3_keys = None
        if access and secret:
            import boto3

            s3 = boto3.client("s3", endpoint_url=f"https://{CDSE_S3_ENDPOINT}",
                              aws_access_key_id=access, aws_secret_access_key=secret,
                              region_name="default")
            s3_keys = S3Keys(access, secret, CDSE_S3_ENDPOINT)
        location = report.get("wekeo", {}).get("location")
        report["cdse"] = guarded(lambda: cdse(get_json, product, meta, reference, location,
                                              s3, s3_keys))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary(report)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
