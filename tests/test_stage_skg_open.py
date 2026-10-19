"""Tests for scripts/stage_skg_open.py: index parsing, the staging plan,
the manifest contract, idempotent skipping and the resumable download.
GDAL conversion is exercised on a tiny zip-wrapped GeoTIFF."""
from __future__ import annotations

import hashlib
import http.server
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest

from scripts import stage_skg_open as st

INDEX = """<html><body><pre>
 2/25/2026 10:38 AM     83807378 <A HREF="/geodataport/data/sks_Volym_omdrev2_01.zip">sks_Volym_omdrev2_01.zip</A><br>
 2/25/2026 10:38 AM     56356781 <A HREF="/geodataport/data/sks_Grundyta_omdrev2_01.zip">sks_Grundyta_omdrev2_01.zip</A><br>
 2/25/2026 10:38 AM     52281246 <A HREF="/geodataport/data/sks_Medeldiameter_omdrev2_01.zip">sks_Medeldiameter_omdrev2_01.zip</A><br>
 4/15/2026  1:00 PM      5942179 <A HREF="/geodataport/data/sks_Volym_omdrev3_07.zip">sks_Volym_omdrev3_07.zip</A><br>
 4/9/2022  3:16 PM   1083208306 <A HREF="/geodataport/data/sksSLUMarkfuktighet01.zip">sksSLUMarkfuktighet01.zip</A><br>
 2/20/2025  1:00 PM   6541420432 <A HREF="/geodataport/data/sksTradhojd01.zip">sksTradhojd01.zip</A><br>
 12/1/2025  5:33 PM   2670278104 <A HREF="/geodataport/data/sks_Medelh%C3%B6jd_omdrev1.zip">sks_Medelh&#246;jd_omdrev1.zip</A><br>
</pre></body></html>"""


def test_parse_index_reads_size_date_and_unescapes_names():
    idx = st.parse_index(INDEX)
    assert idx["sks_Volym_omdrev2_01.zip"] == {"bytes": 83807378, "modified": "2/25/2026 10:38 AM"}
    assert idx["sksSLUMarkfuktighet01.zip"]["bytes"] == 1083208306
    assert "sks_Medelhöjd_omdrev1.zip" in idx


def test_plan_lists_only_published_files_per_product():
    idx = st.parse_index(INDEX)
    items = st.plan(["grunddata", "markfukt", "tradhojd"], [2, 3], ["01", "07"], idx)
    names = [i.name for i in items]
    assert names == ["sks_Volym_omdrev2_01.zip", "sks_Grundyta_omdrev2_01.zip",
                     "sks_Medeldiameter_omdrev2_01.zip", "sks_Volym_omdrev3_07.zip",
                     "sksSLUMarkfuktighet01.zip", "sksTradhojd01.zip"]
    by = {i.name: i for i in items}
    assert by["sks_Volym_omdrev2_01.zip"].derived == "grunddata/omdrev2/volym_01.tif"
    assert by["sks_Volym_omdrev3_07.zip"].omdrev == 3 and by["sks_Volym_omdrev3_07.zip"].county == "07"
    assert by["sksSLUMarkfuktighet01.zip"].derived == "markfukt/markfukt_01.tif"
    assert by["sksTradhojd01.zip"].derived == "tradhojd/tradhojd_10m_01.tif"
    assert st.vrt_targets(items) == {
        "grunddata/volym_omdrev2.vrt": ["grunddata/omdrev2/volym_01.tif"],
        "grunddata/grundyta_omdrev2.vrt": ["grunddata/omdrev2/grundyta_01.tif"],
        "grunddata/medeldiameter_omdrev2.vrt": ["grunddata/omdrev2/medeldiameter_01.tif"],
        "grunddata/volym_omdrev3.vrt": ["grunddata/omdrev3/volym_07.tif"],
        "markfukt/markfukt.vrt": ["markfukt/markfukt_01.tif"],
        "tradhojd/tradhojd_10m.vrt": ["tradhojd/tradhojd_10m_01.tif"],
    }


def test_manifest_roundtrip_is_atomic_and_carries_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("IMINT_GIT_SHA", "abc123")
    m = st.load_manifest(tmp_path)
    assert m["schema"] == st.MANIFEST_SCHEMA and m["files"] == {}
    m["files"]["x.zip"] = {"sha256": "0" * 64, "bytes": 1}
    st.save_manifest(tmp_path, m)
    assert not (tmp_path / "MANIFEST.json.tmp").exists()
    again = json.loads((tmp_path / "MANIFEST.json").read_text())
    assert again["git_sha"] == "abc123" and again["updated_at"].endswith("Z")
    assert again["files"]["x.zip"]["sha256"] == "0" * 64
    assert again["source_url"] == st.SOURCE_URL


def test_is_done_requires_raw_hash_and_derived_raster(tmp_path):
    item = st.Item("a.zip", "markfukt", "markfukt/markfukt_01.tif", county="01")
    m = st.load_manifest(tmp_path)
    assert not st.is_done(item, m, tmp_path)
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "a.zip").write_bytes(b"zip")
    m["files"]["a.zip"] = {"sha256": "f" * 64}
    assert not st.is_done(item, m, tmp_path)                       # derived missing
    m["derived"]["markfukt/markfukt_01.tif"] = {"bytes": 1, "sha256": "e" * 64}
    assert not st.is_done(item, m, tmp_path)
    (tmp_path / "markfukt").mkdir()
    (tmp_path / "markfukt" / "markfukt_01.tif").write_bytes(b"tif")
    assert st.is_done(item, m, tmp_path)


@pytest.fixture
def range_server(tmp_path):
    payload = bytes(range(256)) * 400          # 102400 bytes
    hits = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            rng = self.headers.get("Range")
            hits.append(rng)
            start = int(rng.split("=")[1].rstrip("-")) if rng else 0
            body = payload[start:]
            self.send_response(206 if rng else 200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("ETag", '"abc"')
            self.end_headers()
            # first answer is cut short to force a resume
            self.wfile.write(body if hits.count(None) > 1 or rng else body[:40000])

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/f.zip", payload, hits
    srv.shutdown()


def test_download_resumes_with_a_range_request(range_server, tmp_path):
    url, payload, hits = range_server
    target = tmp_path / "raw" / "f.zip"
    rec = st.download(url, target, len(payload))
    assert target.read_bytes() == payload and not target.with_name("f.zip.part").exists()
    assert rec["resumed"] is True and hits == [None, "bytes=40000-"] and rec["etag"] == '"abc"'
    assert st.download(url, target, len(payload)) == {"resumed": False, "attempts": 0}  # already there


def _tiny_zip(tmp_path: Path, name: str) -> Path:
    import rasterio
    from rasterio.transform import from_origin

    tif = tmp_path / "work" / "x.tif"
    tif.parent.mkdir()
    data = np.arange(64 * 64, dtype=np.int16).reshape(64, 64)
    data[0, :] = -1
    with rasterio.open(tif, "w", driver="GTiff", width=64, height=64, count=1, dtype="int16",
                       crs="EPSG:3006", transform=from_origin(500000, 6250640, 10, 10),
                       nodata=-1) as dst:
        dst.write(data, 1)
    zpath = tmp_path / name
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(tif, "x.tif")
    shutil.rmtree(tif.parent)
    return zpath


def test_gdal_source_reads_deflate_zips_in_place(tmp_path):
    zpath = _tiny_zip(tmp_path, "sks_Volym_omdrev2_01.zip")
    src, tmp = st.gdal_source(zpath, tmp_path / "_work")
    assert src == f"/vsizip/{zpath}/x.tif" and tmp is None
    import rasterio
    with rasterio.open(src) as ds:
        assert ds.width == 64 and ds.nodata == -1


@pytest.mark.skipif(shutil.which("gdal_translate") is None, reason="gdal CLI not installed")
def test_convert_writes_a_verified_cog(tmp_path):
    zpath = _tiny_zip(tmp_path, "sks_Volym_omdrev2_01.zip")
    item = st.Item(zpath.name, "grunddata", "grunddata/omdrev2/volym_01.tif", 2, "volym", "01")
    rec = st.convert(item, zpath, tmp_path / item.derived, tmp_path / "_work", "average")
    assert rec["verified_windows"] == 5 and rec["res_m"] == 10.0 and rec["nodata"] == -1
    assert rec["sha256"] == hashlib.sha256((tmp_path / item.derived).read_bytes()).hexdigest()
    assert rec["blocks"] == [512, 512]


def test_block_stat_p90_ignores_nodata_and_rounds():
    block = np.full((20, 20), -1, dtype=np.int16)
    block[:10, :10] = np.arange(100, dtype=np.int16).reshape(10, 10)        # 0..99 -> p90 = 89.1
    block[:10, 10:] = 7                                                      # constant
    block[10:, :10][0, 0] = 250                                              # one valid cell
    out = st.block_stat(block, "p90", -1)
    assert out.dtype == np.int16 and out.shape == (2, 2)
    assert out.tolist() == [[89, 7], [250, -1]]
    assert st.block_stat(block, "max", -1).tolist() == [[99, 7], [250, -1]]
    assert st.block_stat(block, "average", -1).tolist() == [[50, 7], [250, -1]]   # 49.5 rounds to even


def test_derive_tradhojd_10m_streams_to_the_lattice(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    src = tmp_path / "t.tif"
    data = np.zeros((1300, 640), dtype=np.int16)          # 2 chunks of 640 rows + 20
    data[:, :] = np.arange(640, dtype=np.int16)[None, :] % 100
    data[10:20, :] = -1
    with rasterio.open(src, "w", driver="GTiff", width=640, height=1300, count=1, dtype="int16",
                       crs="EPSG:3006", transform=from_origin(460000, 6270000, 1, 1), nodata=-1) as d:
        d.write(data, 1)
    rec = st.derive_tradhojd_10m(str(src), tmp_path / "out.tif", "p90")
    assert (rec["width"], rec["height"], rec["res_m"], rec["statistic"]) == (64, 130, 10.0, "p90")
    with rasterio.open(tmp_path / "out.tif") as o:
        arr = o.read(1)
        assert o.transform.c == 460000 and o.transform.f == 6270000 and o.block_shapes[0] == (512, 512)
    assert arr[1, 0] == -1 and arr[0, 0] == 8 and arr[129, 63] == 38   # p90 of 0..9 and of 30..39


def test_is_done_accepts_an_intentionally_dropped_raw_zip(tmp_path):
    item = st.Item("sksTradhojd01.zip", "tradhojd", "tradhojd/tradhojd_10m_01.tif", county="01")
    m = st.load_manifest(tmp_path)
    m["files"]["sksTradhojd01.zip"] = {"sha256": "a" * 64, "raw_deleted_at": "2026-10-12T20:00:00Z"}
    m["derived"]["tradhojd/tradhojd_10m_01.tif"] = {"sha256": "b" * 64}
    (tmp_path / "tradhojd").mkdir()
    (tmp_path / "tradhojd" / "tradhojd_10m_01.tif").write_bytes(b"tif")
    assert st.is_done(item, m, tmp_path)                           # no raw on disk, by design
    del m["files"]["sksTradhojd01.zip"]["raw_deleted_at"]
    assert not st.is_done(item, m, tmp_path)                       # genuinely missing input
    m["derived"]["tradhojd/tradhojd_10m_01.tif"] = {}
    m["files"]["sksTradhojd01.zip"]["raw_deleted_at"] = "x"
    assert not st.is_done(item, m, tmp_path)                       # unverified derivation


def test_vrt_sources_come_from_the_whole_staged_inventory(tmp_path):
    m = st.load_manifest(tmp_path)
    for p in ("markfukt/markfukt_01.tif", "markfukt/markfukt_03.tif", "grunddata/omdrev2/volym_01.tif",
              "grunddata/omdrev2/volym_03.tif", "grunddata/omdrev3/volym_07.tif", "grunddata/omdrev2/grundyta_01.tif"):
        m["derived"][p] = {"sha256": "c" * 64}
        (tmp_path / p).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / p).write_bytes(b"tif")
    m["derived"]["markfukt/markfukt_99.tif"] = {"sha256": "d" * 64}     # recorded but not on disk
    assert st.staged_vrt_sources(m, tmp_path, "markfukt/markfukt.vrt") == [
        "markfukt/markfukt_01.tif", "markfukt/markfukt_03.tif"]
    assert st.staged_vrt_sources(m, tmp_path, "grunddata/volym_omdrev2.vrt") == [
        "grunddata/omdrev2/volym_01.tif", "grunddata/omdrev2/volym_03.tif"]
    assert st.staged_vrt_sources(m, tmp_path, "grunddata/volym_omdrev3.vrt") == ["grunddata/omdrev3/volym_07.tif"]
    # a subset run (county 03 only) still publishes both counties
    items = st.plan(["markfukt"], [2], ["03"], {"sksSLUMarkfuktighet03.zip": {"bytes": 1, "modified": ""}})
    for vrt in st.vrt_targets(items):
        assert len(st.staged_vrt_sources(m, tmp_path, vrt)) == 2


def test_markfukt_mosaic_makes_the_255_padding_transparent():
    # county rasters overlap; markfukt pads outside the polygon with 255, grunddata with -1
    assert st.vrt_nodata("markfukt/markfukt.vrt") == (255, 65535)
    assert st.vrt_nodata("grunddata/volym_omdrev2.vrt") == (-1, -1)
    assert st.vrt_nodata("tradhojd/tradhojd_10m.vrt") == (-1, -1)


@pytest.mark.skipif(shutil.which("gdalbuildvrt") is None, reason="gdal CLI not installed")
def test_overlapping_markfukt_counties_compose_without_padding_bleed(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    m = st.load_manifest(tmp_path)
    (tmp_path / "markfukt").mkdir()
    # county A covers x 0-200, real data 40 on the left half, 255 padding on the right
    a = np.full((10, 20), 255, np.uint16); a[:, :10] = 40
    # county B covers x 100-300, real data 60 on its right half, 255 padding on the left
    b = np.full((10, 20), 255, np.uint16); b[:, 10:] = 60
    for name, arr, west in (("markfukt_01.tif", a, 0), ("markfukt_03.tif", b, 100)):
        with rasterio.open(tmp_path / "markfukt" / name, "w", driver="GTiff", width=20, height=10, count=1,
                           dtype="uint16", crs="EPSG:3006", transform=from_origin(west, 100, 10, 10),
                           nodata=65535) as d:
            d.write(arr, 1)
        m["derived"]["markfukt/" + name] = {"sha256": "x" * 64}
    rec = st.build_vrt(tmp_path, "markfukt/markfukt.vrt", st.staged_vrt_sources(m, tmp_path, "markfukt/markfukt.vrt"),
                       *st.vrt_nodata("markfukt/markfukt.vrt"))
    with rasterio.open(tmp_path / "markfukt" / "markfukt.vrt") as v:
        row = v.read(1)[5]
    assert rec["srcnodata"] == 255 and v.nodata == 65535
    assert list(row[:10]) == [40] * 10 and list(row[20:]) == [60] * 10      # neither county's padding won
    assert list(row[10:20]) == [65535] * 10                               # padding in both -> nodata


def test_recovered_download_must_match_the_recorded_digest(tmp_path, monkeypatch):
    monkeypatch.setenv("IMINT_GIT_SHA", "abc")
    rec = {"sha256": "a" * 64, "bytes": 1}
    assert st.check_recovered(rec, "a" * 64, "x.zip", accept=False) == rec        # same bytes: fine
    assert st.check_recovered({}, "b" * 64, "x.zip", accept=False) == {}         # first download
    raw = tmp_path / "raw" / "x.zip"
    raw.parent.mkdir()
    raw.write_bytes(b"new upstream bytes")
    with pytest.raises(st.UpstreamChanged, match="upstream-changed"):
        st.check_recovered(rec, "b" * 64, "x.zip", accept=False, raw=raw, dest=tmp_path)
    assert not raw.exists() and (tmp_path / "raw" / "x.zip.upstream-changed").exists()
    # a later run therefore sees the raw as missing and cannot skip the check
    item = st.Item("x.zip", "markfukt", "markfukt/markfukt_01.tif", county="01")
    m = st.load_manifest(tmp_path)
    m["files"]["x.zip"] = rec
    m["derived"][item.derived] = {"sha256": "d" * 64}
    (tmp_path / "markfukt").mkdir()
    (tmp_path / item.derived).write_bytes(b"tif")
    assert not st.is_done(item, m, tmp_path)
    # acceptance archives the pre-change manifest immutably and keeps the old digest
    st.save_manifest(tmp_path, m)
    raw.write_bytes(b"new upstream bytes")
    out = st.check_recovered(rec, "b" * 64, "x.zip", accept=True, raw=raw, dest=tmp_path)
    assert out["superseded_sha256"] == ["a" * 64] and out["upstream_changed_at"].endswith("Z")
    assert rec == {"sha256": "a" * 64, "bytes": 1}                                 # input untouched
    archived = list(tmp_path.glob("MANIFEST.superseded-*.json"))
    assert len(archived) == 1 and json.loads(archived[0].read_text())["files"]["x.zip"]["sha256"] == "a" * 64
    m["files"]["x.zip"] = {**out, "sha256": "b" * 64}
    st.save_manifest(tmp_path, m)                                                  # later save in the same run
    st.save_manifest(tmp_path, m)
    assert json.loads(archived[0].read_text())["files"]["x.zip"]["sha256"] == "a" * 64   # archive untouched


def test_save_manifest_keeps_the_previous_copy(tmp_path, monkeypatch):
    monkeypatch.setenv("IMINT_GIT_SHA", "abc")
    m = st.load_manifest(tmp_path)
    m["files"]["a.zip"] = {"sha256": "1" * 64}
    st.save_manifest(tmp_path, m)
    assert not (tmp_path / "MANIFEST.previous.json").exists()
    m["files"]["a.zip"] = {"sha256": "2" * 64}
    st.save_manifest(tmp_path, m)
    prev = json.loads((tmp_path / "MANIFEST.previous.json").read_text())
    assert prev["files"]["a.zip"]["sha256"] == "1" * 64
    assert json.loads((tmp_path / "MANIFEST.json").read_text())["files"]["a.zip"]["sha256"] == "2" * 64


def test_manifest_archives_never_overwrite_each_other(tmp_path, monkeypatch):
    monkeypatch.setenv("IMINT_GIT_SHA", "abc")
    monkeypatch.setattr(st, "_now", lambda: "2026-10-13T05:50:00Z")      # freeze the second
    m = st.load_manifest(tmp_path)
    paths = []
    for digest in ("1" * 64, "2" * 64, "3" * 64):
        m["files"]["a.zip"] = {"sha256": digest}
        st.save_manifest(tmp_path, m)
        paths.append(st.archive_manifest(tmp_path))
    assert [p.name for p in paths] == ["MANIFEST.superseded-20261013T055000Z.json",
                                       "MANIFEST.superseded-20261013T055000Z-1.json",
                                       "MANIFEST.superseded-20261013T055000Z-2.json"]
    assert [json.loads(p.read_text())["files"]["a.zip"]["sha256"][0] for p in paths] == ["1", "2", "3"]
    # two same-second accepted recoveries keep both pre-change records
    rec_a = {"sha256": "a" * 64}
    m["files"]["a.zip"] = rec_a
    st.save_manifest(tmp_path, m)
    st.check_recovered(rec_a, "b" * 64, "a.zip", accept=True, dest=tmp_path)
    m["files"]["a.zip"] = {"sha256": "b" * 64}
    st.save_manifest(tmp_path, m)
    st.check_recovered(m["files"]["a.zip"], "c" * 64, "a.zip", accept=True, dest=tmp_path)
    archived = sorted(tmp_path.glob("MANIFEST.superseded-*.json"))
    assert len(archived) == 5
    assert {json.loads(p.read_text())["files"]["a.zip"]["sha256"][0] for p in archived} >= {"a", "b"}


# ---- Trädhöjd one county at a time, verify before delete (decision 2026-10-19) ----

def _tradhojd_src(tmp_path, value=50):
    import rasterio
    from rasterio.transform import from_origin

    src = tmp_path / "t.tif"
    data = np.full((200, 300), value, dtype=np.int16)
    with rasterio.open(src, "w", driver="GTiff", width=300, height=200, count=1, dtype="int16",
                       crs="EPSG:3006", transform=from_origin(460000, 6270000, 1, 1), nodata=-1) as d:
        d.write(data, 1)
    return src


def test_derivation_records_its_value_range_and_verifies_against_the_source(tmp_path):
    rec = st.derive_tradhojd_10m(str(_tradhojd_src(tmp_path)), tmp_path / "out.tif", "p90")
    assert rec["verified"] and (rec["value_min_dm"], rec["value_max_dm"], rec["valid_cells"]) == (50, 50, 600)
    assert rec["bounds"] == [460000.0, 6269800.0, 460300.0, 6270000.0]


@pytest.mark.parametrize("change, message", [
    ({"bounds": [0.0, 0.0, 1.0, 1.0]}, "differs from the source"),
    ({"width": 31}, "size"),
    ({"value_max_dm": 5000}, "outside the plausible"),
    ({"value_min_dm": -5}, "outside the plausible"),
    ({"valid_cells": 0}, "no valid cell"),
])
def test_verification_refuses_a_derived_raster_that_does_not_match(tmp_path, change, message):
    rec = st.derive_tradhojd_10m(str(_tradhojd_src(tmp_path)), tmp_path / "out.tif", "p90")
    expected = {k: rec[k] for k in ("width", "height", "bounds", "value_min_dm", "value_max_dm", "valid_cells")}
    with pytest.raises(ValueError, match=message):
        st.verify_tradhojd_10m(tmp_path / "out.tif", {**expected, **change})


def test_garbage_values_fail_the_derivation_before_any_delete(tmp_path):
    with pytest.raises(ValueError, match="outside the plausible"):
        st.derive_tradhojd_10m(str(_tradhojd_src(tmp_path, value=20000)), tmp_path / "out.tif", "p90")


class _Index:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return INDEX.encode()


def _staging(monkeypatch, tmp_path, free_gib=1000.0):
    """main() offline: the index page, a fake download writing the zip, a fake verified conversion."""
    monkeypatch.setenv("IMINT_GIT_SHA", "abc")
    monkeypatch.setattr(st, "_request", lambda *a, **k: _Index())
    monkeypatch.setattr(st.shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(0, 0, int(free_gib * 2**30)))
    calls = []

    def download(url, target, expected, **kw):
        calls.append(("download", target.name))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"zip")
        return {}

    def convert(item, zpath, out, work, method):
        calls.append(("convert", item.name))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"tif")
        return {"sha256": "d" * 64, "bytes": 3, "verified": True}

    snapshots = []
    real_save = st.save_manifest

    def save(dest, manifest):
        snapshots.append(json.loads(json.dumps(manifest)))
        real_save(dest, manifest)

    monkeypatch.setattr(st, "download", download)
    monkeypatch.setattr(st, "convert", convert)
    monkeypatch.setattr(st, "save_manifest", save)
    monkeypatch.setattr(st, "build_vrt", lambda dest, vrt, rasters, *a: {"sources": rasters})
    small = zipfile.ZipInfo("Tradhojd01.tif")
    small.compress_type, small.file_size = st.ZIP_DEFLATE, 3
    monkeypatch.setattr(st, "zip_tif_entry", lambda z: small)
    return calls, snapshots


def test_the_derivation_is_on_record_before_the_raw_zip_is_deleted(monkeypatch, tmp_path):
    calls, snaps = _staging(monkeypatch, tmp_path)
    code = st.main(["--dest", str(tmp_path / "m"), "--products", "tradhojd", "--counties", "01",
                    "--drop-raw", "--min-free-gib", "50"])
    assert code == 0 and calls == [("download", "sksTradhojd01.zip"), ("convert", "sksTradhojd01.zip")]
    assert not (tmp_path / "m/raw/sksTradhojd01.zip").exists()
    first_deleted = next(i for i, m in enumerate(snaps) if m["files"].get("sksTradhojd01.zip", {}).get("raw_deleted_at"))
    assert "tradhojd/tradhojd_10m_01.tif" in snaps[first_deleted - 1]["derived"]
    rec = snaps[-1]["files"]["sksTradhojd01.zip"]
    assert rec["sha256"] and rec["bytes"] == 3 and rec["url"].endswith("sksTradhojd01.zip") and rec["downloaded_at"]
    assert snaps[-1]["vrt"]["tradhojd/tradhojd_10m.vrt"]["sources"] == ["tradhojd/tradhojd_10m_01.tif"]


def test_a_free_space_floor_stops_before_the_download(monkeypatch, tmp_path):
    calls, _ = _staging(monkeypatch, tmp_path, free_gib=55.0)        # 6.1 GiB zip would leave < 50
    code = st.main(["--dest", str(tmp_path / "m"), "--products", "tradhojd", "--counties", "01",
                    "--drop-raw", "--min-free-gib", "50"])
    assert code == 3 and calls == []


def test_another_countys_raw_zip_on_disk_refuses_the_next_county(monkeypatch, tmp_path):
    calls, _ = _staging(monkeypatch, tmp_path)
    (tmp_path / "m/raw").mkdir(parents=True)
    (tmp_path / "m/raw/sksTradhojd03.zip").write_bytes(b"left over")
    with pytest.raises(RuntimeError, match="one county at a time"):
        st.main(["--dest", str(tmp_path / "m"), "--products", "tradhojd", "--counties", "01", "--drop-raw"])
    assert calls == []


def test_the_floor_also_covers_inflating_a_deflate64_zip(monkeypatch, tmp_path):
    calls, _ = _staging(monkeypatch, tmp_path, free_gib=60.0)       # the zip itself fits above 50
    big = zipfile.ZipInfo("Tradhojd01.tif")
    big.compress_type, big.file_size = st.ZIP_DEFLATE64, 20 * 2**30   # but inflating it would not
    monkeypatch.setattr(st, "zip_tif_entry", lambda z: big)
    code = st.main(["--dest", str(tmp_path / "m"), "--products", "tradhojd", "--counties", "01",
                    "--drop-raw", "--min-free-gib", "50"])
    assert code == 3 and calls == [("download", "sksTradhojd01.zip")]
    assert (tmp_path / "m/raw/sksTradhojd01.zip").exists()           # kept for the resumed run



def test_a_crash_between_record_and_delete_is_finished_on_resume(monkeypatch, tmp_path):
    """Review B1: derivation recorded, raw zip still on disk -> resume re-verifies, deletes, moves on."""
    dest = tmp_path / "m"
    rec = st.derive_tradhojd_10m(str(_tradhojd_src(tmp_path)), tmp_path / "der.tif", "p90")
    calls, snaps = _staging(monkeypatch, tmp_path)
    (dest / "tradhojd").mkdir(parents=True); (dest / "raw").mkdir()
    shutil.copy(tmp_path / "der.tif", dest / "tradhojd/tradhojd_10m_01.tif")
    (dest / "raw/sksTradhojd01.zip").write_bytes(b"zip")
    st.save_manifest(dest, {"files": {"sksTradhojd01.zip": {"sha256": "a" * 64, "bytes": 3}},
                            "derived": {"tradhojd/tradhojd_10m_01.tif": {
                                **rec, "bytes": (dest / "tradhojd/tradhojd_10m_01.tif").stat().st_size,
                                "sha256": st.sha256_of(dest / "tradhojd/tradhojd_10m_01.tif")}},
                            "vrt": {}, "history": []})
    code = st.main(["--dest", str(dest), "--products", "tradhojd", "--counties", "01", "--drop-raw"])
    assert code == 0 and calls == []                                   # no re-download, no re-derive
    assert not (dest / "raw/sksTradhojd01.zip").exists()
    assert snaps[-1]["files"]["sksTradhojd01.zip"]["raw_deleted_at"]
    assert snaps[-1]["vrt"]["tradhojd/tradhojd_10m.vrt"]["sources"] == ["tradhojd/tradhojd_10m_01.tif"]


def test_a_tampered_derived_raster_is_never_trusted_to_delete_the_raw_zip(monkeypatch, tmp_path):
    dest = tmp_path / "m"
    rec = st.derive_tradhojd_10m(str(_tradhojd_src(tmp_path)), tmp_path / "der.tif", "p90")
    _staging(monkeypatch, tmp_path)
    (dest / "tradhojd").mkdir(parents=True); (dest / "raw").mkdir()
    (dest / "tradhojd/tradhojd_10m_01.tif").write_bytes(b"not the recorded raster")
    (dest / "raw/sksTradhojd01.zip").write_bytes(b"zip")
    st.save_manifest(dest, {"files": {"sksTradhojd01.zip": {"sha256": "a" * 64, "bytes": 3}},
                            "derived": {"tradhojd/tradhojd_10m_01.tif": rec | {"sha256": "b" * 64}},
                            "vrt": {}, "history": []})
    with pytest.raises(RuntimeError, match="sha256 differs from its record"):
        st.main(["--dest", str(dest), "--products", "tradhojd", "--counties", "01", "--drop-raw"])
    assert (dest / "raw/sksTradhojd01.zip").exists()


@pytest.mark.parametrize("leftover", ["sksTradhojd03.zip.part", "sksTradhojd03.zip.upstream-changed"])
def test_the_one_county_guard_also_sees_partial_and_superseded_files(monkeypatch, tmp_path, leftover):
    calls, _ = _staging(monkeypatch, tmp_path)
    (tmp_path / "m/raw").mkdir(parents=True)
    (tmp_path / "m/raw" / leftover).write_bytes(b"x")
    with pytest.raises(RuntimeError, match="one county at a time"):
        st.main(["--dest", str(tmp_path / "m"), "--products", "tradhojd", "--counties", "01", "--drop-raw"])
    assert calls == []


def test_a_held_manifest_lock_refuses_before_any_download(monkeypatch, tmp_path):
    calls, _ = _staging(monkeypatch, tmp_path)
    dest = tmp_path / "m"
    dest.mkdir()
    (dest / st.LOCK_NAME).write_text('{"owner": "drop_skg_open_raw"}')
    with pytest.raises(st.ManifestLocked, match="drop_skg_open_raw"):
        st.main(["--dest", str(dest), "--products", "tradhojd", "--counties", "01", "--drop-raw"])
    assert calls == [] and not (dest / "MANIFEST.json").exists()
    assert (dest / st.LOCK_NAME).exists()                               # another writer's lock is left alone


def test_the_lock_is_held_from_the_first_read_to_the_last_save_then_released(monkeypatch, tmp_path):
    _staging(monkeypatch, tmp_path)
    dest = tmp_path / "m"
    held = []
    real_load, real_save = st.load_manifest, st.save_manifest
    monkeypatch.setattr(st, "load_manifest", lambda d: held.append((d / st.LOCK_NAME).exists()) or real_load(d))
    monkeypatch.setattr(st, "save_manifest", lambda d, m: held.append((d / st.LOCK_NAME).exists()) or real_save(d, m))
    assert st.main(["--dest", str(dest), "--products", "tradhojd", "--counties", "01", "--drop-raw"]) == 0
    assert len(held) > 2 and all(held)
    assert not (dest / st.LOCK_NAME).exists()


def test_a_floor_stop_and_a_crash_both_release_the_lock(monkeypatch, tmp_path):
    _staging(monkeypatch, tmp_path, free_gib=55.0)
    dest = tmp_path / "m"
    assert st.main(["--dest", str(dest), "--products", "tradhojd", "--counties", "01",
                    "--drop-raw", "--min-free-gib", "50"]) == 3
    assert not (dest / st.LOCK_NAME).exists()
    monkeypatch.setattr(st, "download", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt))
    monkeypatch.setattr(st.shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(0, 0, 1000 * 2**30))
    with pytest.raises(KeyboardInterrupt):
        st.main(["--dest", str(dest), "--products", "tradhojd", "--counties", "01", "--drop-raw"])
    assert not (dest / st.LOCK_NAME).exists()


def test_a_dry_run_takes_no_lock(monkeypatch, tmp_path):
    _staging(monkeypatch, tmp_path)
    dest = tmp_path / "m"
    dest.mkdir()
    (dest / st.LOCK_NAME).write_text('{"owner": "other"}')
    assert st.main(["--dest", str(dest), "--products", "tradhojd", "--counties", "01", "--dry-run"]) == 0


def test_sigterm_releases_the_lock_and_restores_the_handler(tmp_path):
    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(SystemExit) as exc:
        with st.manifest_lock(tmp_path, "stage_skg_open"):
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(1)                                               # the handler raises before this ends
    assert exc.value.code == 143
    assert not (tmp_path / st.LOCK_NAME).exists()
    assert signal.getsignal(signal.SIGTERM) is previous


def test_a_pod_stopped_with_sigterm_leaves_no_lock(tmp_path):
    code = ("import sys, time; from pathlib import Path; from scripts import stage_skg_open as st\n"
            "with st.manifest_lock(Path(sys.argv[1]), 'stage_skg_open'):\n    time.sleep(60)\n")
    proc = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)], cwd=Path(st.__file__).parents[1])
    deadline = time.monotonic() + 30
    while not (tmp_path / st.LOCK_NAME).exists():
        assert time.monotonic() < deadline and proc.poll() is None
        time.sleep(0.05)
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=30) == 143
    assert not (tmp_path / st.LOCK_NAME).exists()


def test_a_writer_never_removes_a_lock_that_is_no_longer_its_own(tmp_path):
    lock = tmp_path / st.LOCK_NAME
    with st.manifest_lock(tmp_path, "stage_skg_open"):
        lock.unlink()                                                   # cleared by hand as stale ...
        lock.write_text('{"owner": "drop_skg_open_raw", "token": "other"}')   # ... and taken by another writer
    assert lock.read_text() == '{"owner": "drop_skg_open_raw", "token": "other"}'
