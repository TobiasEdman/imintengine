"""Classical chlorophyll indices for Sentinel-2.

Two deterministic, pure-numpy band-ratio indices used as diagnostic
companions to the AI retrievals (MDN, C2RCC).

NDCI — Normalized Difference Chlorophyll Index
    NDCI = (B05 − B04) / (B05 + B04)
    Mishra & Mishra (2012), Remote Sensing of Environment 117: 394–406.
    Uses the red-edge / red contrast that grows with chlorophyll-a
    absorption at 665 nm. Range ≈ [-1, +1]; bloom water typically > 0.0.

MCI — Maximum Chlorophyll Index
    MCI = B05 − B04 − 0.389 · (B06 − B04)
    Gower, King, Borstad & Brown (2005), International Journal of
    Remote Sensing 26: 2005–2012. Originally for MERIS; ported to
    Sentinel-2 MSI by Binding et al. (2013). Measures the height of
    the 705 nm reflectance peak above the red-edge baseline; positive
    values indicate elevated chlorophyll.

Both indices are unitless. Convert to physical Chl-a (mg/m³) via
empirical regression against in-situ data — out of v1 scope.
"""
from __future__ import annotations

import numpy as np

NDCI_BAND_FACTOR = 0.389  # Linear interpolation weight between B04 and B06 at 705 nm
_EPS = 1e-10


def compute_ndci(red: np.ndarray, red_edge1: np.ndarray) -> np.ndarray:
    """Normalized Difference Chlorophyll Index.

    Args:
        red: B04 surface reflectance, shape (H, W), float32.
        red_edge1: B05 surface reflectance, shape (H, W), float32.

    Returns:
        NDCI, shape (H, W), float32. Values outside [-1, +1] indicate
        non-water or atmospheric contamination.
    """
    return ((red_edge1 - red) / (red_edge1 + red + _EPS)).astype(np.float32)


def compute_mci(
    red: np.ndarray,
    red_edge1: np.ndarray,
    red_edge2: np.ndarray,
) -> np.ndarray:
    """Maximum Chlorophyll Index (Gower et al. 2005).

    Args:
        red: B04 surface reflectance, shape (H, W), float32.
        red_edge1: B05 surface reflectance, shape (H, W), float32.
        red_edge2: B06 surface reflectance, shape (H, W), float32.

    Returns:
        MCI, shape (H, W), float32. Positive values indicate elevated
        chlorophyll; zero corresponds to a flat red-edge spectrum.
    """
    baseline = red + NDCI_BAND_FACTOR * (red_edge2 - red)
    return (red_edge1 - baseline).astype(np.float32)
