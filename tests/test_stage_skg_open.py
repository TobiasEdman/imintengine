"""Tests for scripts/stage_skg_open.py: index parsing, the staging plan,
the manifest contract, idempotent skipping and the resumable download.
GDAL conversion is exercised on a tiny zip-wrapped GeoTIFF."""
from __future__ import annotations

import hashlib
import http.server
import json
import shutil
import threading
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
    m["derived"]["markfukt/markfukt_01.tif"] = {"bytes": 1}
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
