"""Shared exportImage download for the Skogsstyrelsen ArcGIS ImageServers.

``slu_markfukt``, ``skg_grunddata`` and ``skg_height`` all fetch a GeoTIFF
from an ImageServer ``exportImage`` endpoint. Their default endpoints are
Skogsstyrelsen's public kartportal proxies
(``kartor.skogsstyrelsen.se/kartor/*ArcGIS93Rest``), which were retired in
2026: they now redirect to the Skogens karta web page, or a web-application
firewall answers with an HTML page. urllib followed the redirect, and each
fetcher retried the HTML three times before failing with "Unexpected
response", which hid the cause.

Every answer is classified (redirects are followed within urllib's bound
of 10, so an endpoint that moved to a working image server keeps working):

* GeoTIFF bytes                       -> returned
* HTTP 429 or 5xx, whatever the body  -> retried: busy, not gone
* a known retirement: a redirect that -> ``SkgEndpointRetiredError`` at once,
  ends on the Skogens karta portal,      no retry
  or an HTML 403 from the retired
  proxy host
* anything else (network errors,      -> retried, then ``RuntimeError`` as
  other HTTP errors, other HTML,         before
  ArcGIS JSON errors, unexpected bytes)

The same data is published openly as per-county GeoTIFF zips under
``OPEN_DATA_URL``; existing tiles keep the aux channels they already store.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

USER_AGENT = "ImintEngine/1.0"
OPEN_DATA_URL = "https://geodpags.skogsstyrelsen.se/geodataport/data/"
_TIFF_MAGIC = (b"II*\x00", b"MM\x00*")


class SkgEndpointRetiredError(RuntimeError):
    """The endpoint is a retired Skogsstyrelsen proxy and serves no images."""


_RETIRED_PROXY_HOST = "kartor.skogsstyrelsen.se"
_PORTAL_HOST, _PORTAL_PATH = "www.skogsstyrelsen.se", "/skogenskarta"


def _open(req: urllib.request.Request, timeout_s: float):
    return urllib.request.urlopen(req, timeout=timeout_s)


def _is_html(data: bytes) -> bool:
    return data[:512].lstrip().lower().startswith((b"<!doctype html", b"<html"))


def _is_portal(url: str) -> bool:
    parts = urllib.parse.urlsplit(url)
    return parts.hostname == _PORTAL_HOST and parts.path.startswith(_PORTAL_PATH)


def _retired(product: str, base_url: str, status: int, location: str, hint: str):
    moved = f", redirects to {location}" if location else ""
    return SkgEndpointRetiredError(
        f"{product}: {base_url} no longer serves images (HTTP {status}{moved}). "
        "Skogsstyrelsen retired its public kartportal ImageServer proxies in 2026. "
        f"The data is published as per-county GeoTIFF zips at {OPEN_DATA_URL} "
        f"({hint}); tiles already fetched keep their stored aux channels, and "
        "new tiles need a local mirror of those zips (backlog IM-022)."
    )


def _fetch_once(url: str, base_url: str, timeout_s: float, product: str, hint: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with _open(req, timeout_s) as resp:
            status, data, final_url = resp.status, resp.read(), resp.geturl()
    except urllib.error.HTTPError as e:
        body = e.read() or b""
        location = e.headers.get("Location", "")
        # Only these two answers mean retired; a 429/5xx error page, even in
        # HTML, is a busy server and is retried by the caller.
        if (300 <= e.code < 400 and _is_portal(location)) or (
                e.code == 403 and _is_html(body)
                and urllib.parse.urlsplit(base_url).hostname == _RETIRED_PROXY_HOST):
            raise _retired(product, base_url, e.code, location, hint) from None
        raise
    if data[:4] in _TIFF_MAGIC:
        return data
    if _is_html(data) and _is_portal(final_url):
        raise _retired(product, base_url, status, final_url, hint)
    try:
        err = json.loads(data)
    except (ValueError, UnicodeDecodeError):
        raise RuntimeError(f"Unexpected response ({len(data)} bytes, starts with {data[:20]!r})")
    msg = err.get("error", {}).get("message", str(err)[:200]) if isinstance(err, dict) else str(err)[:200]
    raise RuntimeError(f"ImageServer error: {msg}")


def download_export_image(
    base_url: str,
    params: dict,
    *,
    product: str,
    open_data_hint: str,
    timeout_s: float,
    retries: int,
    retry_delay_s: float,
) -> bytes:
    """GET ``base_url?params`` and return the GeoTIFF bytes.

    Raises ``SkgEndpointRetiredError`` immediately when the endpoint
    redirects or answers with HTML; retries every other failure
    ``retries`` times with a linearly growing delay, then raises
    ``RuntimeError`` naming ``product``.
    """
    url = base_url + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries + 1):
        try:
            return _fetch_once(url, base_url, timeout_s, product, open_data_hint)
        except SkgEndpointRetiredError:
            raise
        except Exception as e:
            if attempt == retries:
                raise RuntimeError(
                    f"{product} fetch failed after {retries + 1} attempts: {e}"
                ) from e
            time.sleep(retry_delay_s * (attempt + 1))
    raise AssertionError("unreachable")
