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
PORTAL = "https://www.skogsstyrelsen.se/skogenskarta"
KW = dict(product="Test product", open_data_hint="hint.zip", timeout_s=5, retries=2,
          retry_delay_s=0.0)


class _Resp(io.BytesIO):
    def __init__(self, data: bytes, status: int = 200):
        super().__init__(data)
        self.status = status


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
        if isinstance(item, Exception):
            raise item
        return _Resp(item.getvalue(), item.status)     # a fresh body per attempt

    monkeypatch.setattr(skg_imageserver, "_open", fake_open)
    return calls, script


def test_tiff_is_returned(server):
    calls, script = server
    script.append(_Resp(TIFF))
    assert download_export_image("http://x/ImageServer", {"f": "image"}, **KW) == TIFF
    assert len(calls) == 1 and calls[0] == "http://x/ImageServer?f=image"


@pytest.mark.parametrize("failure", [
    _http_error(302, location=PORTAL),                          # proxy redirects to portal
    _http_error(403, b"\r\n<!DOCTYPE html><title>Web Application Firewall</title>"),
    _Resp(b"<html><body>Skogens karta</body></html>"),           # HTML with 200
], ids=["redirect", "waf-403", "html-200"])
def test_retired_endpoint_fails_at_once_and_says_why(server, failure):
    calls, script = server
    script.append(failure)
    with pytest.raises(SkgEndpointRetiredError) as exc:
        download_export_image("http://x/ImageServer", {"f": "image"}, **KW)
    msg = str(exc.value)
    assert len(calls) == 1                                      # a retired endpoint is not retried
    assert "http://x/ImageServer" in msg and skg_imageserver.OPEN_DATA_URL in msg
    assert "hint.zip" in msg and "IM-022" in msg
    if isinstance(failure, urllib.error.HTTPError):
        assert f"HTTP {failure.code}" in msg
    if failure is not None and getattr(failure, "code", None) == 302:
        assert PORTAL in msg


def test_transient_failure_is_retried(server):
    calls, script = server
    script.extend([urllib.error.URLError("timed out"), _Resp(TIFF)])
    assert download_export_image("http://x/s", {}, **KW) == TIFF
    assert len(calls) == 2


@pytest.mark.parametrize("failure, text", [
    (_http_error(503, b"busy"), "HTTP Error 503"),
    (_Resp(b'{"error": {"message": "Invalid bbox"}}'), "ImageServer error: Invalid bbox"),
    (_Resp(b"\x00\x01garbage"), "Unexpected response"),
])
def test_persistent_failure_raises_after_all_attempts(server, failure, text):
    calls, script = server
    script.append(failure)
    with pytest.raises(RuntimeError, match="Test product fetch failed after 3 attempts") as exc:
        download_export_image("http://x/s", {}, **KW)
    assert not isinstance(exc.value, SkgEndpointRetiredError)
    assert text in str(exc.value) and len(calls) == 3


def test_real_opener_does_not_follow_redirects():
    """Through the real urllib opener: a 302 must surface, not be followed."""
    hits = []

    class Redirect(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(302)
            self.send_header("Location", PORTAL)
            self.end_headers()

        def log_message(self, *args):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Redirect)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_port}/kartor/markfuktighetSLUArcGIS93Rest"
        with pytest.raises(SkgEndpointRetiredError, match="redirects to " + PORTAL):
            download_export_image(url, {"f": "image"}, **KW)
    finally:
        srv.shutdown()
    assert len(hits) == 1


@pytest.mark.parametrize("fetch, env", [
    (lambda: slu_markfukt.fetch_markfukt_tile(0, 0, 100, 100, size_px=10), "SKG_MARKFUKT_URL"),
    (lambda: skg_grunddata.fetch_volume_tile(0, 0, 100, 100, size_px=10), "SKG_GRUNDDATA_URL"),
    (lambda: skg_height.fetch_height_tile(0, 0, 100, 100, size_px=10), "SKG_HEIGHT_URL"),
], ids=["markfukt", "grunddata", "height"])
def test_each_fetcher_reports_the_retired_proxy(server, monkeypatch, fetch, env):
    calls, script = server
    script.append(_http_error(302, location=PORTAL))
    monkeypatch.setenv(env, "http://proxy.invalid/kartor/xArcGIS93Rest")
    for mod in (skg_grunddata, skg_height):
        monkeypatch.setattr(mod, "_IMAGESERVER_URL", None)
    with pytest.raises(SkgEndpointRetiredError, match="proxy.invalid"):
        fetch()
    assert len(calls) == 1
