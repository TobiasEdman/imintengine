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


# ── 10-class schema (grouped — pine/spruce kept separate) ────────────────

LULC_CLASS_NAMES_10 = {
    0: "background",
    1: "forest_pine",
    2: "forest_spruce",
    3: "forest_deciduous",
    4: "forest_mixed",
    5: "forest_wetland",
    6: "open_wetland",
    7: "cropland",
    8: "open_land",
    9: "developed",
    10: "water",
}

# Map 19-class indices to 10-class indices
_MAP_19_TO_10 = {
    0: 0,    # background
    1: 1,    # forest_pine
    2: 2,    # forest_spruce
    3: 3,    # forest_deciduous
    4: 4,    # forest_mixed
    5: 4,    # forest_temp_non_forest → forest_mixed
    6: 5,    # forest_wetland_pine → forest_wetland
    7: 5,    # forest_wetland_spruce → forest_wetland
    8: 5,    # forest_wetland_deciduous → forest_wetland
    9: 5,    # forest_wetland_mixed → forest_wetland
    10: 5,   # forest_wetland_temp → forest_wetland
    11: 6,   # open_wetland
    12: 7,   # cropland
    13: 8,   # open_land_bare → open_land
    14: 8,   # open_land_vegetated → open_land
    15: 9,   # developed_buildings → developed
    16: 9,   # developed_infrastructure → developed
    17: 9,   # developed_roads → developed
    18: 10,  # water_lakes → water
    19: 10,  # water_sea → water
}

_LUT_19_TO_10 = np.zeros(20, dtype=np.uint8)
for k, v in _MAP_19_TO_10.items():
    _LUT_19_TO_10[k] = v


# ── 12-class schema (10-class + buildings + temp non-forest) ─────────────

LULC_CLASS_NAMES_12 = {
    0: "background",
    1: "forest_pine",
    2: "forest_spruce",
    3: "forest_deciduous",
    4: "forest_mixed",
    5: "forest_temp_non_forest",
    6: "forest_wetland",
    7: "open_wetland",
    8: "cropland",
    9: "open_land",
    10: "developed",           # infrastructure + roads (no buildings)
    11: "buildings",
    12: "water",
}

# Map 19-class indices to 12-class indices
_MAP_19_TO_12 = {
    0: 0,    # background
    1: 1,    # forest_pine
    2: 2,    # forest_spruce
    3: 3,    # forest_deciduous
    4: 4,    # forest_mixed
    5: 5,    # forest_temp_non_forest (own class)
    6: 6,    # forest_wetland_pine → forest_wetland
    7: 6,    # forest_wetland_spruce → forest_wetland
    8: 6,    # forest_wetland_deciduous → forest_wetland
    9: 6,    # forest_wetland_mixed → forest_wetland
    10: 6,   # forest_wetland_temp → forest_wetland
    11: 7,   # open_wetland
    12: 8,   # cropland
    13: 9,   # open_land_bare → open_land
    14: 9,   # open_land_vegetated → open_land
    15: 11,  # developed_buildings → buildings
    16: 10,  # developed_infrastructure → developed
    17: 10,  # developed_roads → developed
    18: 12,  # water_lakes → water
    19: 12,  # water_sea → water
}

_LUT_19_TO_12 = np.zeros(20, dtype=np.uint8)
for k, v in _MAP_19_TO_12.items():
    _LUT_19_TO_12[k] = v


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
    if num_classes == 12:
        return _LUT_19_TO_12[labels_19.clip(0, 19)]
    return _LUT_19_TO_10[labels_19.clip(0, 19)]


def get_class_names(num_classes: int = 19) -> dict[int, str]:
    """Return class name mapping for the given schema."""
    if num_classes == 23:
        from .unified_schema import UNIFIED_CLASSES
        return UNIFIED_CLASSES
    if num_classes == 19:
        return LULC_CLASS_NAMES_19
    if num_classes == 12:
        return LULC_CLASS_NAMES_12
    if num_classes == 10:
        return LULC_CLASS_NAMES_10
    return LULC_CLASS_NAMES_19


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
        (num_classes,) float32 array of weights (index 0 = ignore).
    """
    n = num_classes  # num_classes already includes background at index 0
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
