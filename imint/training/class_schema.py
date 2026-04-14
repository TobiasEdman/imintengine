"""
imint/training/class_schema.py — NMD to LULC class mappings

Converts raw NMD uint8 class codes to contiguous 19-class training labels
(full NMD Level 2). Use nmd_raster_to_lulc() for label generation, then
nmd19_to_unified() from unified_schema to map to the 23-class training schema.
"""
from __future__ import annotations

import numpy as np

from ..analyzers.nmd import NMD_LEVEL2

# ── 19-class schema (full NMD Level 2) ───────────────────────────────────

LULC_CLASS_NAMES_19: dict[int, str] = {
    0: "background",
    1: "forest_pine",
    2: "forest_spruce",
    3: "forest_deciduous",
    4: "forest_mixed",
    5: "forest_temp_non_forest",
    6: "forest_wetland_pine",
    7: "forest_wetland_spruce",
    8: "forest_wetland_deciduous",
    9: "forest_wetland_mixed",
    10: "forest_wetland_temp",
    11: "open_wetland",
    12: "cropland",
    13: "open_land_bare",
    14: "open_land_vegetated",
    15: "developed_buildings",
    16: "developed_infrastructure",
    17: "developed_roads",
    18: "water_lakes",
    19: "water_sea",
}

_L2_NAME_TO_IDX: dict[str, int] = {v: k for k, v in LULC_CLASS_NAMES_19.items() if k > 0}

# LUT: raw NMD uint8 code → sequential 19-class index (0 = background/unmapped)
_LUT_19 = np.zeros(256, dtype=np.uint8)
for _name, _codes in NMD_LEVEL2.items():
    _idx = _L2_NAME_TO_IDX[_name]
    for _code in _codes:
        _LUT_19[_code] = _idx


# ── Public API ────────────────────────────────────────────────────────────

def nmd_raster_to_lulc(nmd_raster: np.ndarray) -> np.ndarray:
    """Convert raw NMD class codes to sequential 19-class training labels.

    Maps raw NMD uint8 raster codes to contiguous indices 1-19.
    Index 0 = background (unknown or unmapped code).

    Call nmd19_to_unified() from unified_schema to further map to the
    23-class unified training schema.

    Args:
        nmd_raster: (H, W) uint8 array of raw NMD raster codes.

    Returns:
        (H, W) uint8 array with indices 0-19.
    """
    return _LUT_19[nmd_raster.clip(0, 255)]


def get_class_names(num_classes: int = 19) -> dict[int, str]:
    """Return class name mapping.

    Args:
        num_classes: 19 for NMD Level 2 names, 23 for unified schema names.

    Returns:
        Dict mapping class index to name.
    """
    if num_classes == 23:
        from .unified_schema import UNIFIED_CLASSES
        return UNIFIED_CLASSES
    return LULC_CLASS_NAMES_19


def compute_class_weights(
    class_counts: dict[int, int],
    num_classes: int = 19,
    max_weight: float = 10.0,
    ignore_index: int = 0,
    method: str = "sqrt",
) -> np.ndarray:
    """Class weights for cross-entropy loss.

    Args:
        class_counts: Mapping of class index → pixel count.
        num_classes: Total classes including background at index 0.
        max_weight: Upper cap to prevent instability on rare classes.
        ignore_index: Class assigned weight 0 (background, not trained).
        method: Weighting strategy:
            ``"inverse"`` — pure inverse-frequency (total / (C * count)).
                Wide range (~0.3–10x), can cause over-prediction of rare classes.
            ``"sqrt"`` — sqrt of inverse-frequency. Dampened range (~0.5–3.5x).
            ``"effective_number"`` — Cui et al. 2019 (beta=0.999).
                Nearly uniform for large counts — can kill rare-class gradients.

    Returns:
        (num_classes,) float32 weight array.
    """
    counts = np.array([class_counts.get(i, 0) for i in range(num_classes)], dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    raw = counts.sum() / (num_classes * counts)

    if method == "inverse":
        weights = raw
    elif method == "sqrt":
        weights = np.sqrt(raw)
    elif method == "effective_number":
        beta = 0.999
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.mean()
    else:
        raise ValueError(f"Unknown weighting method: {method!r}")

    weights = np.minimum(weights, max_weight)
    weights[ignore_index] = 0.0
    return weights.astype(np.float32)
