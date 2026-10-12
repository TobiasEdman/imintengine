"""Offline tests for the shared Skogsstyrelsen ImageServer download."""
from __future__ import annotations

import http.server
import io
import threading
import urllib.error

import pytest

from imint.training import skg_grunddata, skg_height, skg_imageserver, slu_markfukt
from imint.training.skg_imageserver import SkgEndpointRetiredError, download_export_image

TIFF = b"II*\x00" + b"\x00" * 60
HTML = b"\r\n<!DOCTYPE html><html><title>Skogens karta</title></html>"
PORTAL = "https://www.skogsstyrelsen.se/skogenskarta"
PROXY = "https://kartor.skogsstyrelsen.se/kartor/markfuktighetSLUArcGIS93Rest"
OTHER = "https://images.example.org/arcgis/rest/services/x/ImageServer/exportImage"
KW = dict(product="Test product", open_data_hint="hint.zip", timeout_s=5, retries=2,
          retry_delay_s=0.0)


class _Resp(io.BytesIO):
    def __init__(self, data: bytes, status: int = 200, url: str = ""):
        super().__init__(data)
        self.status, self.url = status, url

    def geturl(self) -> str:
        return self.url


def _http_error(code: int, body: bytes = b"", location: str = "") -> urllib.error.HTTPError:
    headers = {"Location": location} if location else {}
    return urllib.error.HTTPError("http://x", code, "msg", headers, io.BytesIO(body))


@pytest.fixture
def server(monkeypatch):
    """Scripted stand-in for skg_imageserver._open; records every call."""
    calls, script = [], []

    def fake_open(req, timeout_s):
        calls.append(req.full_url)
        item = script.pop(0) if len(script) > 1 else script[0]
        if isinstance(item, urllib.error.HTTPError):
            raise _http_error(item.code, item.fp.getvalue(), item.headers.get("Location", ""))
        if isinstance(item, Exception):
            raise item
        return _Resp(item.getvalue(), item.status, item.url or req.full_url)   # fresh per attempt

    monkeypatch.setattr(skg_imageserver, "_open", fake_open)
    return calls, script


def test_tiff_is_returned(server):
    calls, script = server
    script.append(_Resp(TIFF))
    assert download_export_image(OTHER, {"f": "image"}, **KW) == TIFF
    assert calls == [OTHER + "?f=image"]


@pytest.mark.parametrize("base, failure, status_text", [
    (PROXY, _Resp(HTML, url=PORTAL), "HTTP 200, redirects to " + PORTAL),   # redirect followed to the portal
    (PROXY, _http_error(403, HTML), "HTTP 403"),                            # WAF page from the retired proxy
    (PROXY, _http_error(302, location=PORTAL), "HTTP 302, redirects to " + PORTAL),
], ids=["portal-after-redirect", "proxy-waf-403", "portal-redirect-unfollowed"])
def test_known_retirement_fails_at_once_and_says_why(server, base, failure, status_text):
    calls, script = server
    script.append(failure)
    with pytest.raises(SkgEndpointRetiredError) as exc:
        download_export_image(base, {"f": "image"}, **KW)
    msg = str(exc.value)
    assert len(calls) == 1                                      # a retired endpoint is not retried
    assert base in msg and status_text in msg
    assert skg_imageserver.OPEN_DATA_URL in msg and "hint.zip" in msg and "IM-022" in msg


@pytest.mark.parametrize("failure", [
    _http_error(503, HTML), _http_error(429, HTML), _http_error(502, b""),
    urllib.error.URLError("timed out"),
], ids=["503-html", "429-html", "502", "network"])
def test_transient_failure_is_retried_even_behind_html(server, failure):
    calls, script = server
    script.extend([failure, _Resp(TIFF)])
    assert download_export_image(PROXY, {}, **KW) == TIFF
    assert len(calls) == 2


@pytest.mark.parametrize("base, failure, text", [
    (OTHER, _http_error(503, b"busy"), "HTTP Error 503"),
    (OTHER, _Resp(b'{"error": {"message": "Invalid bbox"}}'), "ImageServer error: Invalid bbox"),
    (OTHER, _Resp(b"\x00\x01garbage"), "Unexpected response"),
    (OTHER, _Resp(HTML), "Unexpected response"),                 # HTML, but not the portal
    (OTHER, _http_error(403, HTML), "HTTP Error 403"),           # HTML 403, but not the retired proxy
], ids=["503", "json-error", "garbage", "html-elsewhere", "403-html-elsewhere"])
def test_other_failures_are_retried_then_raised(server, base, failure, text):
    calls, script = server
    script.append(failure)
    with pytest.raises(RuntimeError, match="Test product fetch failed after 3 attempts") as exc:
        download_export_image(base, {}, **KW)
    assert not isinstance(exc.value, SkgEndpointRetiredError)
    assert text in str(exc.value) and len(calls) == 3


@pytest.fixture
def local_server():
    """Real HTTP server: /moved -> /image.tif (TIFF), /moved-html -> /page (HTML)."""
    hits = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0]
            hits.append(path)
            if path in ("/moved", "/moved-html"):
                self.send_response(302)
                self.send_header("Location", "/image.tif" if path == "/moved" else "/page")
                self.end_headers()
                return
            body = TIFF if path == "/image.tif" else HTML
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", hits
    srv.shutdown()


def test_real_opener_follows_a_redirect_to_a_working_endpoint(local_server):
    root, hits = local_server
    assert download_export_image(root + "/moved", {"f": "image"}, **KW) == TIFF
    assert hits == ["/moved", "/image.tif"]


def test_real_opener_does_not_call_unknown_html_retired(local_server):
    root, hits = local_server
    with pytest.raises(RuntimeError, match="Unexpected response") as exc:
        download_export_image(root + "/moved-html", {}, **KW)
    assert not isinstance(exc.value, SkgEndpointRetiredError)
    assert hits.count("/moved-html") == 3


@pytest.mark.parametrize("fetch, env", [
    (lambda: slu_markfukt.fetch_markfukt_tile(0, 0, 100, 100, size_px=10), "SKG_MARKFUKT_URL"),
    (lambda: skg_grunddata.fetch_volume_tile(0, 0, 100, 100, size_px=10), "SKG_GRUNDDATA_URL"),
    (lambda: skg_height.fetch_height_tile(0, 0, 100, 100, size_px=10), "SKG_HEIGHT_URL"),
], ids=["markfukt", "grunddata", "height"])
def test_each_fetcher_reports_the_retired_proxy(server, monkeypatch, fetch, env):
    calls, script = server
    script.append(_Resp(HTML, url=PORTAL))
    monkeypatch.setenv(env, PROXY)
    for mod in (skg_grunddata, skg_height):
        monkeypatch.setattr(mod, "_IMAGESERVER_URL", None)
    with pytest.raises(SkgEndpointRetiredError, match="kartor.skogsstyrelsen.se"):
        fetch()
    assert len(calls) == 1
