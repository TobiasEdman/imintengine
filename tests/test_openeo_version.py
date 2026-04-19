"""Tests for openEO version discovery helper in imint.fetch.

The helper protects against silent API-version drift by querying the
server's /.well-known/openeo and picking the highest production
version. This matters because:

    - DES (openeo.digitalearth.se) advertises only api_version=1.1.0
    - CDSE (openeo.dataspace.copernicus.eu) advertises 1.0, 1.1, 1.2, 1.3

Without explicit pinning the Python client auto-negotiates and can
pick different versions across two seemingly-identical backends.
"""
from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import pytest

from imint.fetch import _discover_openeo_endpoint


def _mock_well_known(payload: dict):
    """Return a context-manager-style mock for urllib.request.urlopen."""
    buf = io.BytesIO(json.dumps(payload).encode())

    class _Resp:
        def __enter__(self_inner):
            return buf
        def __exit__(self_inner, *a):
            return False

    return _Resp()


class TestDiscoverOpenEOEndpoint:
    def test_des_only_advertises_1_1(self):
        payload = {"versions": [
            {"url": "https://openeo.digitalearth.se/",
             "production": True, "api_version": "1.1.0"},
        ]}
        with patch("urllib.request.urlopen", return_value=_mock_well_known(payload)):
            got = _discover_openeo_endpoint("https://openeo.digitalearth.se")
        assert got == ("https://openeo.digitalearth.se/", "1.1.0")

    def test_cdse_multi_version_picks_highest_production(self):
        payload = {"versions": [
            {"api_version": "1.0.0", "production": True,
             "url": "https://openeo.dataspace.copernicus.eu/openeo/1.0/"},
            {"api_version": "1.1.0", "production": True,
             "url": "https://openeo.dataspace.copernicus.eu/openeo/1.1/"},
            {"api_version": "1.2.0", "production": True,
             "url": "https://openeo.dataspace.copernicus.eu/openeo/1.2/"},
            # Non-production 1.3 should be ignored when a production exists
            {"api_version": "1.3.0", "production": False,
             "url": "https://openeo.dataspace.copernicus.eu/openeo/1.3/"},
        ]}
        with patch("urllib.request.urlopen", return_value=_mock_well_known(payload)):
            got = _discover_openeo_endpoint("https://openeo.dataspace.copernicus.eu")
        assert got == (
            "https://openeo.dataspace.copernicus.eu/openeo/1.2/", "1.2.0",
        )

    def test_falls_back_to_nonprod_when_no_production(self):
        payload = {"versions": [
            {"api_version": "1.3.0", "production": False,
             "url": "https://example.com/1.3/"},
        ]}
        with patch("urllib.request.urlopen", return_value=_mock_well_known(payload)):
            got = _discover_openeo_endpoint("https://example.com")
        assert got == ("https://example.com/1.3/", "1.3.0")

    def test_http_error_returns_none(self):
        with patch("urllib.request.urlopen", side_effect=OSError("conn refused")):
            got = _discover_openeo_endpoint("https://example.com")
        assert got is None

    def test_malformed_payload_returns_none(self):
        with patch("urllib.request.urlopen",
                   return_value=_mock_well_known({"foo": "bar"})):
            got = _discover_openeo_endpoint("https://example.com")
        assert got is None

    def test_empty_versions_returns_none(self):
        with patch("urllib.request.urlopen",
                   return_value=_mock_well_known({"versions": []})):
            got = _discover_openeo_endpoint("https://example.com")
        assert got is None

    def test_versions_without_url_skipped(self):
        payload = {"versions": [
            {"api_version": "1.1.0", "production": True},  # missing url
            {"api_version": "1.0.0", "production": True,
             "url": "https://example.com/1.0/"},
        ]}
        with patch("urllib.request.urlopen", return_value=_mock_well_known(payload)):
            got = _discover_openeo_endpoint("https://example.com")
        assert got == ("https://example.com/1.0/", "1.0.0")

    def test_version_sort_handles_odd_strings(self):
        payload = {"versions": [
            {"api_version": "garbage", "production": True, "url": "x"},
            {"api_version": "1.1.0", "production": True,
             "url": "https://example.com/1.1/"},
        ]}
        with patch("urllib.request.urlopen", return_value=_mock_well_known(payload)):
            got = _discover_openeo_endpoint("https://example.com")
        # 1.1.0 wins over unparseable
        assert got == ("https://example.com/1.1/", "1.1.0")
