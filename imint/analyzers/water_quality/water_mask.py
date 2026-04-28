"""Water mask construction for the WaterQualityAnalyzer.

Two-tier strategy:

1. **Primary** — Sentinel-2 Scene Classification Layer (SCL) class 6 (water).
   Sen2Cor's classifier is well-validated for open water and handles
   clouds/shadows correctly.

2. **Fallback** — MNDWI (Modified Normalized Difference Water Index)
   when SCL is unavailable. Threshold at 0.0 per Xu (2006).

   MNDWI = (B03 − B11) / (B03 + B11)
   Xu (2006), International Journal of Remote Sensing 27: 3025–3033.

The mask excludes clouds (SCL 8, 9, 10) and cloud shadows (SCL 3) when SCL
is provided. For ML retrievals (MDN, C2RCC) and the MCI / NDCI indices,
mask values evaluated as ``True`` mark valid water pixels; ``False`` marks
land, cloud, shadow, or no-data.
"""
from __future__ import annotations

import numpy as np

SCL_WATER = 6
SCL_CLOUD_SHADOW = 3
SCL_CLOUD_MEDIUM = 8
SCL_CLOUD_HIGH = 9
SCL_CIRRUS = 10

_EPS = 1e-10
MNDWI_WATER_THRESHOLD = 0.0  # Xu 2006


def compute_mndwi(green: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Modified Normalized Difference Water Index.

    Args:
        green: B03 surface reflectance, shape (H, W).
        swir: B11 surface reflectance, shape (H, W).

    Returns:
        MNDWI, float32, water > 0.
    """
    return ((green - swir) / (green + swir + _EPS)).astype(np.float32)


def water_mask_from_scl(scl: np.ndarray) -> np.ndarray:
    """Build a water mask from the SCL band, excluding clouds and shadows.

    Args:
        scl: Scene Classification Layer, uint8, shape (H, W).

    Returns:
        Boolean mask, True for valid water pixels.
    """
    contaminated = (
        (scl == SCL_CLOUD_SHADOW)
        | (scl == SCL_CLOUD_MEDIUM)
        | (scl == SCL_CLOUD_HIGH)
        | (scl == SCL_CIRRUS)
    )
    return (scl == SCL_WATER) & ~contaminated


def water_mask_from_mndwi(
    green: np.ndarray,
    swir: np.ndarray,
    threshold: float = MNDWI_WATER_THRESHOLD,
) -> np.ndarray:
    """Build a water mask from MNDWI > threshold (fallback when SCL absent)."""
    mndwi = compute_mndwi(green, swir)
    return mndwi > threshold


def build_water_mask(
    bands: dict[str, np.ndarray],
    scl: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Build the water mask using the best available method.

    Args:
        bands: dict of Sentinel-2 surface reflectance arrays.
        scl: optional SCL band.

    Returns:
        (mask, method) where mask is a (H, W) boolean array and method
        is one of ``"scl"`` or ``"mndwi"``.

    Raises:
        ValueError: if neither SCL nor (B03, B11) are available.
    """
    if scl is not None:
        return water_mask_from_scl(scl), "scl"

    green = bands.get("B03")
    swir = bands.get("B11")
    if green is None or swir is None:
        raise ValueError(
            "Cannot build water mask: SCL absent and B03/B11 missing for MNDWI fallback"
        )
    return water_mask_from_mndwi(green, swir), "mndwi"
