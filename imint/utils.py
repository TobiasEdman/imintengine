"""
imint/utils.py — Shared utilities for IMINT Engine

Sentinel-2 data conversion, band name mapping, etc.
"""
from __future__ import annotations

import numpy as np

# ── Data source profiles ─────────────────────────────────────────────────────
# Each source has its own offset and scale for DN → reflectance conversion:
#   reflectance = (DN - offset) / scale
#
# See: https://sentinels.copernicus.eu/web/sentinel/-/copernicus-sentinel-2-major-products-upgrade-702

DATA_SOURCES = {
    "des": {
        "offset": 1000,
        "scale": 10000,
        "description": (
            "Digital Earth Sweden. COGs with PB>=04.00 offset baked in. "
            "Verified: vegetation B04 DN ~1960 → (1960-1000)/10000 = 0.096"
        ),
    },
    "copernicus": {
        "offset": -1000,
        "scale": 10000,
        "description": (
            "Copernicus Data Space (CDSE). Raw L2A with RADIO_ADD_OFFSET=-1000. "
            "reflectance = (DN - (-1000)) / 10000 = (DN + 1000) / 10000"
        ),
    },
    "legacy": {
        "offset": 0,
        "scale": 10000,
        "description": (
            "Pre-PB04.00 data or sources without offset. "
            "reflectance = DN / 10000"
        ),
    },
}

DEFAULT_SOURCE = "des"

# Band name mapping: DES uses lowercase, IMINT Engine uses uppercase internally
DES_TO_IMINT = {
    "b01": "B01",
    "b02": "B02", "b03": "B03", "b04": "B04",
    "b05": "B05", "b06": "B06", "b07": "B07",
    "b08": "B08", "b8a": "B8A", "b09": "B09",
    "b11": "B11", "b12": "B12",
}
IMINT_TO_DES = {v: k for k, v in DES_TO_IMINT.items()}


def dn_to_reflectance(
    dn: np.ndarray,
    clip: bool = True,
    source: str = DEFAULT_SOURCE,
) -> np.ndarray:
    """Convert Sentinel-2 L2A DN values to BOA reflectance.

    Formula: reflectance = (DN - offset) / scale

    The offset and scale depend on the data source:
        - "des":        (DN - 1000) / 10000   (DES bakes PB04.00 offset into COGs)
        - "copernicus": (DN + 1000) / 10000   (raw RADIO_ADD_OFFSET = -1000)
        - "legacy":     DN / 10000            (pre-PB04.00, no offset)

    Args:
        dn: Raw DN array (int or float).
        clip: If True, clip result to [0, 1].
        source: Data source profile name (see DATA_SOURCES).

    Returns:
        Reflectance array as float32 in [0, 1] (if clipped).
    """
    profile = DATA_SOURCES[source]
    reflectance = (dn.astype(np.float32) - profile["offset"]) / profile["scale"]
    if clip:
        reflectance = np.clip(reflectance, 0.0, 1.0)
    return reflectance


def des_to_imint_bands(des_bands: dict) -> dict:
    """Convert DES band dict (lowercase keys) to IMINT format (uppercase keys).

    Args:
        des_bands: Dict like {"b02": array, "b04": array, ...}

    Returns:
        Dict like {"B02": array, "B04": array, ...}
    """
    result = {}
    for des_name, arr in des_bands.items():
        imint_name = DES_TO_IMINT.get(des_name, des_name.upper())
        result[imint_name] = arr
    return result


def bands_to_rgb(
    bands: dict,
    percentile_stretch: bool = True,
    scl: np.ndarray | None = None,
) -> np.ndarray:
    """Convert a band dictionary to an RGB image array.

    Uses B04 (Red), B03 (Green), B02 (Blue). Assumes values are already
    in reflectance [0, 1]. Applies percentile stretch for visualization.

    When *scl* is provided, cloud pixels (SCL classes 8, 9, 10) are masked
    out before computing the stretch percentiles so that bright clouds do
    not compress the dynamic range of the underlying surface.

    Args:
        bands: Dict with at least "B02", "B03", "B04" keys.
        percentile_stretch: If True, stretch to 2nd/98th percentile.
        scl: Optional SCL array (H, W) uint8 with values 0-11.

    Returns:
        RGB array (H, W, 3) float32 in [0, 1].
    """
    fallback = np.zeros((256, 256), dtype=np.float32)
    r = bands.get("B04", fallback)
    g = bands.get("B03", fallback)
    b = bands.get("B02", fallback)
    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)

    if percentile_stretch:
        if scl is not None:
            cloud_mask = np.isin(scl, [8, 9, 10])
            clear_pixels = rgb[~cloud_mask]
            if clear_pixels.size > 0:
                p2, p98 = np.percentile(clear_pixels, [2, 98])
            else:
                p2, p98 = np.percentile(rgb, [2, 98])
        else:
            p2, p98 = np.percentile(rgb, [2, 98])
        rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)

    return rgb
