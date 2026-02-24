"""
imint/training/class_schema.py — NMD to LULC class mappings

Provides two schemas:
  * 19-class (full NMD Level 2)
  * 10-class (grouped for coarser classification)

The mapping converts raw NMD uint8 class codes into contiguous
integer labels suitable for cross-entropy training.
"""
from __future__ import annotations

import numpy as np

from ..analyzers.nmd import NMD_LEVEL2

# ── 19-class schema (full NMD Level 2) ───────────────────────────────────

# Map NMD Level 2 names to contiguous indices 1-19 (0 = background)
LULC_CLASS_NAMES_19 = {
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

# Reverse: NMD L2 name → training index
_L2_NAME_TO_IDX_19 = {v: k for k, v in LULC_CLASS_NAMES_19.items() if k > 0}

# Build lookup table: raw NMD code → training class index (19-class)
_NMD_CODE_TO_19 = {}
for l2_name, codes in NMD_LEVEL2.items():
    idx = _L2_NAME_TO_IDX_19[l2_name]
    for code in codes:
        _NMD_CODE_TO_19[code] = idx

# Create numpy LUT (codes go up to ~128, use 256 for safety)
_LUT_19 = np.zeros(256, dtype=np.uint8)
for code, idx in _NMD_CODE_TO_19.items():
    _LUT_19[code] = idx


# ── 10-class schema (grouped) ────────────────────────────────────────────

LULC_CLASS_NAMES_10 = {
    0: "background",
    1: "forest_conifer",
    2: "forest_deciduous",
    3: "forest_mixed",
    4: "forest_wetland",
    5: "open_wetland",
    6: "cropland",
    7: "open_land",
    8: "developed",
    9: "water",
}

# Map 19-class indices to 10-class indices
_MAP_19_TO_10 = {
    0: 0,   # background
    1: 1,   # forest_pine → forest_conifer
    2: 1,   # forest_spruce → forest_conifer
    3: 2,   # forest_deciduous
    4: 3,   # forest_mixed
    5: 3,   # forest_temp_non_forest → forest_mixed
    6: 4,   # forest_wetland_pine → forest_wetland
    7: 4,   # forest_wetland_spruce → forest_wetland
    8: 4,   # forest_wetland_deciduous → forest_wetland
    9: 4,   # forest_wetland_mixed → forest_wetland
    10: 4,  # forest_wetland_temp → forest_wetland
    11: 5,  # open_wetland
    12: 6,  # cropland
    13: 7,  # open_land_bare → open_land
    14: 7,  # open_land_vegetated → open_land
    15: 8,  # developed_buildings → developed
    16: 8,  # developed_infrastructure → developed
    17: 8,  # developed_roads → developed
    18: 9,  # water_lakes → water
    19: 9,  # water_sea → water
}

_LUT_19_TO_10 = np.zeros(20, dtype=np.uint8)
for k, v in _MAP_19_TO_10.items():
    _LUT_19_TO_10[k] = v


# ── Public API ────────────────────────────────────────────────────────────

def nmd_raster_to_lulc(
    nmd_raster: np.ndarray,
    num_classes: int = 19,
) -> np.ndarray:
    """Convert raw NMD class codes to contiguous LULC training labels.

    Args:
        nmd_raster: (H, W) uint8 array with raw NMD codes.
        num_classes: 19 for full L2, 10 for grouped.

    Returns:
        (H, W) uint8 array with class indices 0..num_classes.
    """
    labels_19 = _LUT_19[nmd_raster.clip(0, 255)]
    if num_classes == 19:
        return labels_19
    return _LUT_19_TO_10[labels_19.clip(0, 19)]


def get_class_names(num_classes: int = 19) -> dict[int, str]:
    """Return class name mapping for the given schema."""
    if num_classes == 19:
        return LULC_CLASS_NAMES_19
    return LULC_CLASS_NAMES_10


def compute_class_weights(
    class_counts: dict[int, int],
    num_classes: int = 19,
    max_weight: float = 10.0,
    ignore_index: int = 0,
) -> np.ndarray:
    """Compute inverse-frequency class weights for cross-entropy loss.

    Args:
        class_counts: Dict mapping class index to pixel count.
        num_classes: Number of classes (19 or 10).
        max_weight: Cap to prevent instability on rare classes.
        ignore_index: Class index to assign zero weight.

    Returns:
        (num_classes+1,) float32 array of weights (index 0 = ignore).
    """
    n = num_classes + 1  # include background at index 0
    counts = np.array([class_counts.get(i, 0) for i in range(n)], dtype=np.float64)

    # Avoid division by zero
    counts = np.maximum(counts, 1.0)

    # Inverse frequency
    total = counts.sum()
    weights = total / (n * counts)

    # Cap and zero out ignore class
    weights = np.minimum(weights, max_weight)
    weights[ignore_index] = 0.0

    return weights.astype(np.float32)
