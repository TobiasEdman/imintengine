"""
imint/utils.py — Shared utilities for IMINT Engine

Sentinel-2 data conversion, band name mapping, etc.
"""
from __future__ import annotations

import numpy as np

# Sentinel-2 L2A Processing Baseline >= 04.00 applies a radiometric offset.
# The RADIO_ADD_OFFSET = -1000 was added during encoding:
#   DN = reflectance * QUANTIFICATION_VALUE + RADIO_ADD_OFFSET
#   DN = reflectance * 10000 + (-1000)
# To decode:
#   reflectance = (DN - RADIO_ADD_OFFSET) / QUANTIFICATION_VALUE
#   reflectance = (DN - (-1000)) / 10000 = (DN + 1000) / 10000
#
# HOWEVER: DES stores COGs with the offset already baked into the DN values,
# meaning the raw values from DES are standard pre-PB04.00 encoding:
#   reflectance = (DN - 1000) / 10000
#
# Verified empirically: vegetation B04 DN ~1960 → (1960-1000)/10000 = 0.096
# which matches expected red reflectance for vegetation (~0.03-0.10).
#
# See: https://sentinels.copernicus.eu/web/sentinel/-/copernicus-sentinel-2-major-products-upgrade-702
BOA_ADD_OFFSET = 1000              # Subtract from DN to get scaled reflectance
QUANTIFICATION_VALUE = 10000       # Scale factor to get [0, 1] reflectance

# Band name mapping: DES uses lowercase, IMINT Engine uses uppercase internally
DES_TO_IMINT = {
    "b02": "B02", "b03": "B03", "b04": "B04",
    "b05": "B05", "b06": "B06", "b07": "B07",
    "b08": "B08", "b8a": "B8A", "b09": "B09",
    "b11": "B11", "b12": "B12",
}
IMINT_TO_DES = {v: k for k, v in DES_TO_IMINT.items()}


def dn_to_reflectance(dn: np.ndarray, clip: bool = True) -> np.ndarray:
    """Convert Sentinel-2 L2A DN values (from DES) to BOA reflectance.

    Formula: reflectance = (DN - BOA_ADD_OFFSET) / QUANTIFICATION_VALUE
             reflectance = (DN - 1000) / 10000

    Verified against DES data:
        Vegetation B04 DN ~1960 → (1960 - 1000) / 10000 = 0.096 ✓
        Expected vegetation red reflectance: ~0.03-0.10

    Args:
        dn: Raw DN array (int or float) from Sentinel-2 L2A via DES.
        clip: If True, clip result to [0, 1].

    Returns:
        Reflectance array as float32 in [0, 1] (if clipped).
    """
    reflectance = (dn.astype(np.float32) - BOA_ADD_OFFSET) / QUANTIFICATION_VALUE
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


def bands_to_rgb(bands: dict, percentile_stretch: bool = True) -> np.ndarray:
    """Convert a band dictionary to an RGB image array.

    Uses B04 (Red), B03 (Green), B02 (Blue). Assumes values are already
    in reflectance [0, 1]. Applies percentile stretch for visualization.

    Args:
        bands: Dict with at least "B02", "B03", "B04" keys.
        percentile_stretch: If True, stretch to 2nd/98th percentile.

    Returns:
        RGB array (H, W, 3) float32 in [0, 1].
    """
    fallback = np.zeros((256, 256), dtype=np.float32)
    r = bands.get("B04", fallback)
    g = bands.get("B03", fallback)
    b = bands.get("B02", fallback)
    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)

    if percentile_stretch:
        p2, p98 = np.percentile(rgb, [2, 98])
        rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)

    return rgb
