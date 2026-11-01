"""HR-VPP access probe: layout, Range and window parity against local HTTP and S3-style servers."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
import requests
from rasterio.shutil import copy as rio_copy
from rasterio.transform import from_origin

from scripts import probe_hrvpp_access as probe

PRODUCT = "VPP_2021_S2_T33VWJ-010m_V101_s1_SOSD.tif"
LOCATION = f"s3://hr-vpp-products-vpp-v01-2021/CLMS/Pan-European/Biophysical/VPP/v01/2021/s1/{PRODUCT}"
META = {"metric": "SOSD", "tileId": "T33VWJ", "year": 2021, "season": 1,
        "bounds_4326": [15.0, 59.4, 17.0, 60.4]}


class Handler(SimpleHTTPRequestHandler):
    """Static files; ``mode`` decides how a Range header is treated."""
    mode = "range"
    served = []

    def log_message(self, *args):
        pass

    def send_head(self):
        if self.path.startswith("/dl/"):        # broker answering with a signed redirect
            self.send_response(302)
            self.send_header("Location", f"/files/{self.path[4:]}?X-Amz-Signature=secret")
            self.end_headers()
            return None
        self.path = self.path.split("?")[0]
        spec = self.headers.get("Range")
        if self.command != "GET" or not spec or Handler.mode == "ignore":
            return super().send_head()
        data = open(self.translate_path(self.path), "rb").read()
        start, _, end = spec.removeprefix("bytes=").partition("-")
        start, end = int(start), min(int(end) if end else len(data) - 1, len(data) - 1)
        Handler.served.append(end - start + 1)
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        return io.BytesIO(data[start:end + 1])


class QuietServer(ThreadingHTTPServer):
    def handle_error(self, *args):     # a client closing an unwanted 200 early is expected
        pass


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "www"
    (root / "files").mkdir(parents=True)
    httpd = QuietServer(("127.0.0.1", 0), partial(Handler, directory=str(root)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    Handler.mode, Handler.served = "range", []
    yield root, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def write_vpp(path, *, cog=True, seed=0, size=2048):
    """An HR-VPP-like single-band uint16 raster on a UTM grid."""
    data = np.random.default_rng(seed).integers(0, 400, (1, size, size), dtype=np.uint16)
    profile = dict(driver="GTiff", width=size, height=size, count=1, dtype="uint16",
                   crs="EPSG:32633", transform=from_origin(500000, 6700000, 10, 10),
                   compress="deflate")
    if not cog:
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data)
        return data
    tmp = path.with_suffix(".tmp.tif")
    with rasterio.open(tmp, "w", **profile, tiled=True, blockxsize=512, blockysize=512) as dst:
        dst.write(data)
    rio_copy(tmp, path, driver="COG", BLOCKSIZE=512, COMPRESS="DEFLATE")
    tmp.unlink()
    return data


@pytest.fixture
def cache(tmp_path):
    vpp = tmp_path / "vpp_wekeo"
    vpp.mkdir()
    write_vpp(vpp / PRODUCT)
    older = PRODUCT.replace("2021", "2020")
    write_vpp(vpp / older, seed=1)
    index = {PRODUCT: META, older: {**META, "year": 2020},
             PRODUCT.replace("SOSD", "EOSD"): {**META, "metric": "EOSD"}}
    (vpp / "index.json").write_text(json.dumps(index))
    return vpp


def test_reference_is_the_newest_cached_sosd_unless_named(cache):
    assert probe.pick_reference(cache) == (PRODUCT, META)
    with pytest.raises(SystemExit):     # in the index but not on disk
        probe.pick_reference(cache, PRODUCT.replace("SOSD", "EOSD"))


def test_layout_counts_the_blocks_a_window_touches_and_matches_what_gdal_fetches(cache, server):
    root, base = server
    report = probe.layout(cache / PRODUCT)
    assert report["tiled"] and report["block"] == [512, 512] and report["overviews"]
    assert report["layout_tag"] == "COG" and report["first_ifd_offset"] < 1024
    assert report["window_blocks"] == 4            # centre 512 px of 2048 spans 2 × 2 blocks
    assert 0 < report["window_bytes"] < report["file_bytes"] / 4   # 4 of 16 blocks, + overviews

    (root / "files" / PRODUCT).write_bytes((cache / PRODUCT).read_bytes())
    parity = probe.window_parity(f"/vsicurl/{base}/files/{PRODUCT}", cache / PRODUCT, {})
    assert parity["equal"]
    fetched = sum(Handler.served)
    assert report["window_bytes"] <= fetched <= report["window_bytes"] + 128 * 1024, fetched

    striped = cache / "striped.tif"
    write_vpp(striped, cog=False)
    report = probe.layout(striped)
    assert not report["tiled"] and report["block"][1] == 2048 and report["overviews"] == []


def test_range_probe_needs_206_with_the_cached_bytes(cache, server):
    root, base = server
    reference = cache / PRODUCT
    (root / "files" / PRODUCT).write_bytes(reference.read_bytes())
    session = requests.Session()

    report, final = probe.range_probe(session, f"{base}/files/{PRODUCT}", reference)
    assert report["range_supported"] and final == f"{base}/files/{PRODUCT}"
    assert [c["status"] for c in report["checks"]] == [206, 206]
    assert all(c["matches_cached"] and c["bytes_read"] == 1024 for c in report["checks"])

    Handler.mode = "ignore"                         # 200 with the whole file
    report, _ = probe.range_probe(session, f"{base}/files/{PRODUCT}", reference)
    assert not report["range_supported"]
    assert [c["status"] for c in report["checks"]] == [200, 200]
    assert [c["matches_cached"] for c in report["checks"]] == [True, False]
    assert all(c["bytes_read"] == 1024 for c in report["checks"])

    Handler.mode = "range"                          # a different product at the URL
    (root / "files" / "other.tif").write_bytes((cache / PRODUCT.replace("2021", "2020", 1)).read_bytes())
    report, _ = probe.range_probe(session, f"{base}/files/other.tif", reference)
    assert not report["range_supported"]


def test_redirects_are_followed_and_signatures_never_reported(cache, server):
    root, base = server
    (root / "files" / PRODUCT).write_bytes((cache / PRODUCT).read_bytes())
    report, final = probe.range_probe(requests.Session(), f"{base}/dl/{PRODUCT}", cache / PRODUCT)
    assert report["range_supported"]
    assert report["final_url"] == f"{base}/files/{PRODUCT}" and "secret" in final
    assert "secret" not in json.dumps(report)


class FakeHda:
    """The parts of hda.Client the probe uses, as hda 2.40 shapes them."""

    def __init__(self, url, product=PRODUCT):
        self.url, self.product, self.queries, self.tac = url, product, [], []
        self.session, self.token = requests.Session(), "token"

    def accept_tac(self, dataset):
        self.tac.append(dataset)

    def search(self, query):
        self.queries.append(query)
        feature = {"id": self.product.removesuffix(".tif"),
                   "properties": {"location": LOCATION.replace(PRODUCT, self.product)}}
        return [SimpleNamespace(results=[feature], get_download_urls=lambda: [self.url])]


def test_wekeo_orders_the_cached_product_and_compares_the_window(cache, server):
    root, base = server
    (root / "files" / PRODUCT).write_bytes((cache / PRODUCT).read_bytes())
    client = FakeHda(f"{base}/dl/{PRODUCT}")
    report = probe.wekeo(client, PRODUCT, META, cache / PRODUCT)
    assert client.tac == ["EO:EEA:DAT:CLMS_HRVPP_VPP"]
    assert client.queries[0]["tileId"] == "33VWJ" and client.queries[0]["productType"] == "SOSD"
    assert report["found"] and report["location"] == LOCATION and report["range_supported"]
    assert report["window"] == {"seconds": report["window"]["seconds"], "same_dims": True,
                                "equal": True}
    assert "secret" not in json.dumps(report)

    Handler.mode = "ignore"
    report = probe.wekeo(FakeHda(f"{base}/files/{PRODUCT}"), PRODUCT, META, cache / PRODUCT)
    assert not report["range_supported"] and "window" not in report
    assert probe.wekeo(FakeHda(base, product="VPP_other.tif"), PRODUCT, META,
                       cache / PRODUCT) == {"found": False}


def stac_pages():
    pages = {
        f"{probe.CDSE_STAC}/collections": {
            "collections": [{"id": "sentinel-2-l2a", "title": "Sentinel-2 L2A"}],
            "links": [{"rel": "next", "href": f"{probe.CDSE_STAC}/collections?page=2"}]},
        f"{probe.CDSE_STAC}/collections?page=2": {
            "collections": [{"id": "clms_hrvpp_vpp", "title": "HR-VPP VPP parameters"},
                            {"id": "clms_lsp_global_300m_yearly_v2_cog", "title": "Land Surface Phenology"}],
            "links": []}}

    def get_json(url):
        if url in pages:
            return pages[url]
        assert "datetime=2021-01-01T00:00:00Z/2021-12-31T23:59:59Z" in url and "bbox=15.99," in url
        if "clms_hrvpp_vpp" in url:
            return {"features": [
                {"id": "VPP_2021_S2_T33VWJ-010m_V101_s1_SOSD",
                 "assets": {"data": {"href": f"s3://eodata/CLMS/x/{PRODUCT}",
                                     "alternate": {"https": {"href": "https://dl/x?token=secret"}}}}},
                {"id": "VPP_2021_S2_T33VWK-010m_V101_s1_SOSD", "assets": {}}]}
        return {"features": []}
    return get_json


def test_stac_lists_hr_vpp_collections_and_the_reference_item():
    report = probe.stac_items(stac_pages(), META)
    assert report["collections_total"] == 3
    assert [c["id"] for c in report["matched"]] == ["clms_hrvpp_vpp", "clms_lsp_global_300m_yearly_v2_cog"]
    hrvpp = report["items_for_reference"]["clms_hrvpp_vpp"]
    assert hrvpp["in_box"] == 2 and [i["id"] for i in hrvpp["tile_items"]] == [
        "VPP_2021_S2_T33VWJ-010m_V101_s1_SOSD"]
    assert hrvpp["tile_items"][0]["assets"]["data"] == ["https://dl/x", f"s3://eodata/CLMS/x/{PRODUCT}"]
    assert report["items_for_reference"]["clms_lsp_global_300m_yearly_v2_cog"]["in_box"] == 0
    assert "secret" not in json.dumps(report)


def test_cdse_key_drops_wekeos_bucket_or_falls_back_to_its_known_shape():
    key = f"CLMS/Pan-European/Biophysical/VPP/v01/2021/s1/{PRODUCT}"
    assert probe.cdse_key(LOCATION, PRODUCT, META) == key
    assert probe.cdse_key(None, PRODUCT, META) == key
    assert probe.cdse_key("s3://bucket/other/x.tif", PRODUCT, META) == key


class FakeS3:
    """list_objects_v2 over a set of keys, as boto3 answers it."""

    def __init__(self, keys):
        self.keys, self.calls = keys, 0

    def list_objects_v2(self, Bucket, Prefix, Delimiter=None, MaxKeys=1000):
        assert Bucket == "eodata"
        self.calls += 1
        under = sorted(k for k in self.keys if k.startswith(Prefix))
        if Delimiter:
            children = sorted({Prefix + k[len(Prefix):].split("/")[0] + "/"
                               for k in under if "/" in k[len(Prefix):]})
            return {"CommonPrefixes": [{"Prefix": p} for p in children]}
        return {"Contents": [{"Key": k, "Size": 7} for k in under[:MaxKeys]]}


def test_cdse_walks_the_bucket_and_compares_the_same_key_through_vsis3(cache, server):
    root, base = server
    key = f"CLMS/Pan-European/Biophysical/VPP/v01/2021/s1/{PRODUCT}"
    (root / "eodata" / key).parent.mkdir(parents=True)
    (root / "eodata" / key).write_bytes((cache / PRODUCT).read_bytes())
    s3 = FakeS3({key, "CLMS/Global/Land/x.tif", "CLMS/bio-geophysical/vegetation_phenology/y.tif"})
    env = probe.S3Keys("k", "s", base.removeprefix("http://"), https=False)

    report = probe.cdse(stac_pages(), PRODUCT, META, cache / PRODUCT, LOCATION, s3, env)
    assert report["stac"]["matched"][0]["id"] == "clms_hrvpp_vpp"
    assert report["s3"]["prefixes"] == [
        "CLMS/Pan-European/", "CLMS/bio-geophysical/", "CLMS/Pan-European/Biophysical/",
        "CLMS/bio-geophysical/vegetation_phenology/", "CLMS/Pan-European/Biophysical/VPP/"]
    assert report["s3"]["same_key"]["exists"] and report["s3"]["same_key"]["key"] == key
    assert report["s3"]["window"]["equal"]

    assert probe.cdse(stac_pages(), PRODUCT, META, cache / PRODUCT, None)["s3"] == {
        "skipped": "no CDSE S3 keys (secret cdse-s3)"}
    missing = probe.cdse(stac_pages(), PRODUCT, META, cache / PRODUCT, LOCATION,
                         FakeS3({"CLMS/Global/x.tif"}), env)
    assert not missing["s3"]["same_key"]["exists"] and "window" not in missing["s3"]


def test_cli_writes_the_report_and_records_a_failing_probe(cache, tmp_path, capsys, monkeypatch):
    out = tmp_path / "out" / "report.json"
    assert probe.main(["--vpp-dir", str(cache), "--report", str(out),
                       "--skip-wekeo", "--skip-cdse"]) == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["product"] == PRODUCT and summary["cached_tiled"] and summary["errors"] == {}
    assert json.loads(out.read_text())["layout"]["window_blocks"] == 4

    monkeypatch.setattr(probe, "layout", lambda path: 1 / 0)
    assert probe.main(["--vpp-dir", str(cache), "--report", str(out),
                       "--skip-wekeo", "--skip-cdse"]) == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["errors"] == {"layout": "ZeroDivisionError: division by zero"}


def test_job_pins_the_probe_it_runs():
    import re
    import subprocess
    from pathlib import Path

    yaml = pytest.importorskip("yaml")
    root = Path(__file__).resolve().parents[2]
    job = yaml.safe_load((root / "k8s/hrvpp-access-probe-job.yaml").read_text())
    container = job["spec"]["template"]["spec"]["containers"][0]
    script = container["command"][-1]
    assert "@sha256:" in container["image"]
    assert 'test "$(git rev-parse HEAD)" = "$SHA"' in script
    assert "python3 scripts/probe_hrvpp_access.py" in script and "hda==2.40" in script
    refs = [e["secretRef"] for e in container["envFrom"]]
    assert {"name": "wekeo-creds"} in refs and {"name": "cdse-s3", "optional": True} in refs
    assert not any("cdse-credentials" in json.dumps(e) for e in container.get("env", []))
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    sha = re.search(r"SHA=([0-9a-f]{40})", script)[1]
    git = ["git", "-C", str(root)]
    if subprocess.run([*git, "cat-file", "-e", sha], capture_output=True).returncode:
        pytest.skip(f"{sha} not in this clone")
    assert subprocess.run([*git, "merge-base", "--is-ancestor", sha, "HEAD"]).returncode == 0
    changed = subprocess.run([*git, "diff", "--name-only", sha, "HEAD", "--",
                              "scripts/probe_hrvpp_access.py", "imint/data/wekeo_vpp.py"],
                             capture_output=True, text=True, check=True).stdout.split()
    assert changed == [], f"bump SHA in k8s/hrvpp-access-probe-job.yaml; changed: {changed}"
