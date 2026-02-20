"""
tests/test_des_connection.py — DES/openEO connection test

Tests the connection to Digital Earth Sweden (DES) via openEO.

Setup (one of):
    1. Token from Web Editor:  python scripts/des_login.py --token "YOUR_TOKEN"
    2. OIDC device flow:       python scripts/des_login.py --device

Run tests:
    pytest tests/test_des_connection.py -v -s

Tests are skipped automatically if no valid authentication is found.
"""
from __future__ import annotations

import os
import sys
import io
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OPENEO_URL = "https://openeo.digitalearth.se"
TOKEN_PATH = os.path.join(os.path.dirname(__file__), "..", ".des_token")


def _connect():
    """Connect and authenticate to DES. Tries saved token, then OIDC."""
    import openeo

    conn = openeo.connect(OPENEO_URL)

    # Try saved EGI OIDC access token first
    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH) as f:
            token = f.read().strip()
        if token:
            conn.authenticate_oidc_access_token(access_token=token, provider_id="egi")
            return conn

    # Fall back to cached OIDC
    conn.authenticate_oidc(provider_id="egi")
    return conn


def _is_authenticated():
    """Check if we have valid authentication."""
    try:
        conn = _connect()
        conn.list_collections()
        return True
    except Exception:
        return False


# Skip all tests if not authenticated
pytestmark = pytest.mark.skipif(
    not _is_authenticated(),
    reason="Not authenticated to DES. Run: python scripts/des_login.py --token YOUR_TOKEN",
)


class TestDESConnection:
    """Basic connectivity tests."""

    def test_connect_and_authenticate(self):
        """Should connect and authenticate."""
        conn = _connect()
        assert conn is not None

    def test_list_collections(self):
        """Should list available Sentinel-2 collections."""
        conn = _connect()
        collections = conn.list_collections()
        collection_ids = [c["id"] for c in collections]

        print(f"\nAvailable collections ({len(collection_ids)}):")
        for cid in sorted(collection_ids):
            print(f"  - {cid}")

        assert "s2_msi_l2a" in collection_ids
        assert "s2_msi_l1c" in collection_ids


class TestDESDataFetch:
    """Test actual data retrieval from DES."""

    SMALL_COORDS = {
        "west": 14.55,
        "south": 56.00,
        "east": 14.60,
        "north": 56.03,
    }
    TEST_DATE = "2022-06-15"

    def test_fetch_rgb(self):
        """Fetch RGB bands (b04, b03, b02) and verify DN→reflectance conversion.

        Note: DES uses lowercase band names (b02, b03, b04, ...).
        Sentinel-2 L2A via DES applies BOA_ADD_OFFSET = 1000:
            reflectance = (DN - 1000) / 10000
        """
        from datetime import datetime, timedelta
        from imint.utils import dn_to_reflectance

        conn = _connect()
        date = datetime.strptime(self.TEST_DATE, "%Y-%m-%d")
        temporal = [date.strftime("%Y-%m-%d"), (date + timedelta(days=1)).strftime("%Y-%m-%d")]

        # DES band names are lowercase
        cube = conn.load_collection(
            collection_id="s2_msi_l2a",
            spatial_extent=self.SMALL_COORDS,
            temporal_extent=temporal,
            bands=["b04", "b03", "b02"],
        )

        data = cube.download(format="gtiff")
        assert len(data) > 0, "Downloaded data is empty"

        import rasterio
        with rasterio.open(io.BytesIO(data)) as src:
            assert src.count >= 3, f"Expected at least 3 bands, got {src.count}"
            dn = src.read()
            refl = dn_to_reflectance(dn)

            print(f"\n  Shape: {src.shape}")
            print(f"  CRS: {src.crs}")
            print(f"  DN range:         [{dn.min()}, {dn.max()}]")
            print(f"  Reflectance range: [{refl.min():.4f}, {refl.max():.4f}]")

            # Reflectance should be in [0, 1] for valid surface pixels
            assert refl.max() <= 1.0, f"Reflectance > 1.0: {refl.max()}"
            assert refl.min() >= 0.0, f"Reflectance < 0.0: {refl.min()}"
            # Typical land reflectance in visible bands: 0.01–0.30
            assert refl.mean() < 0.5, f"Mean reflectance suspiciously high: {refl.mean():.4f}"

    def test_fetch_all_imint_bands(self):
        """Fetch all bands needed by IMINT Engine, handling resolution differences.

        DES Sentinel-2 band resolutions:
          10m: b02, b03, b04, b08
          20m: b05, b06, b07, b8a, b11, b12, scl, cld, snw, wvp, aot

        We must load 10m and 20m bands separately and resample to combine them.
        """
        from datetime import datetime, timedelta

        conn = _connect()
        date = datetime.strptime(self.TEST_DATE, "%Y-%m-%d")
        temporal = [date.strftime("%Y-%m-%d"), (date + timedelta(days=1)).strftime("%Y-%m-%d")]

        bands_10m = ["b02", "b03", "b04", "b08"]
        bands_20m = ["b11"]

        # Load 10m bands
        cube_10m = conn.load_collection(
            collection_id="s2_msi_l2a",
            spatial_extent=self.SMALL_COORDS,
            temporal_extent=temporal,
            bands=bands_10m,
        )

        # Load 20m bands and resample to 10m grid
        cube_20m = conn.load_collection(
            collection_id="s2_msi_l2a",
            spatial_extent=self.SMALL_COORDS,
            temporal_extent=temporal,
            bands=bands_20m,
        )
        cube_20m = cube_20m.resample_cube_spatial(target=cube_10m, method="bilinear")

        # Merge the two cubes
        cube = cube_10m.merge_cubes(cube_20m)

        data = cube.download(format="gtiff")
        assert len(data) > 0

        import rasterio
        with rasterio.open(io.BytesIO(data)) as src:
            total_bands = len(bands_10m) + len(bands_20m)
            print(f"\n  All IMINT bands fetched successfully (with resample)")
            print(f"  Shape: {src.shape}, Bands: {src.count}")
            assert src.count == total_bands, f"Expected {total_bands} bands, got {src.count}"
