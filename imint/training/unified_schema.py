"""
imint/training/unified_schema.py — Unified LULC + Crop + Harvest class schema

Merges NMD land cover (10-class) with LPIS crop detail (8 classes)
and SKS harvest data into a single 19-class segmentation schema.

NMD provides background (forest, water, developed, wetland), LPIS
provides crop-specific detail within agricultural pixels, and SKS
provides harvested forest areas.

Usage:
    from imint.training.unified_schema import merge_all, UNIFIED_CLASSES

    # For tiles with NMD + LPIS + SKS harvest:
    unified = merge_all(nmd_10class, lpis_mask, harvest_mask)

    # For crop tiles with both NMD and LPIS:
    unified = merge_nmd_lpis(nmd_10class, lpis_mask)

    # For LULC tiles with only NMD:
    unified = nmd10_to_unified(nmd_10class)
"""
from __future__ import annotations

import numpy as np

# ── Unified Class Schema ──────────────────────────────────────────────────────

NUM_UNIFIED_CLASSES = 19

UNIFIED_CLASSES = {
    0: "bakgrund",          # ignore_index
    # NMD-derived (1-9)
    1: "tallskog",          # NMD 10-class: 1 (forest_pine)
    2: "granskog",          # NMD 10-class: 2 (forest_spruce)
    3: "lövskog",           # NMD 10-class: 3 (forest_deciduous)
    4: "blandskog",         # NMD 10-class: 4 (forest_mixed)
    5: "sumpskog",          # NMD 10-class: 5 (forest_wetland)
    6: "våtmark",           # NMD 10-class: 6 (open_wetland)
    7: "öppen mark",        # NMD 10-class: 8 (open_land)
    8: "bebyggelse",        # NMD 10-class: 9 (developed)
    9: "vatten",            # NMD 10-class: 10 (water)
    # LPIS crop detail (10-17) — replaces NMD class 7 (cropland)
    10: "vete",             # LPIS crop class 1
    11: "korn",             # LPIS crop class 2
    12: "havre",            # LPIS crop class 3
    13: "oljeväxter",       # LPIS crop class 4
    14: "vall",             # LPIS crop class 5
    15: "potatis",          # LPIS crop class 6
    16: "trindsäd",         # LPIS crop class 7
    17: "övrig åker",       # LPIS crop class 8 + unmapped cropland
    # SKS harvest (18)
    18: "hygge",            # SKS utförda avverkningar (harvested forest)
}

UNIFIED_CLASS_NAMES = [UNIFIED_CLASSES[i] for i in range(NUM_UNIFIED_CLASSES)]

# Color palette for visualization (RGB tuples)
UNIFIED_COLORS = {
    0: (0, 0, 0),           # bakgrund
    1: (0, 100, 0),         # tallskog — dark green
    2: (34, 139, 34),       # granskog — forest green
    3: (50, 205, 50),       # lövskog — lime green
    4: (60, 179, 113),      # blandskog — medium sea green
    5: (46, 79, 46),        # sumpskog — dark olive
    6: (139, 90, 43),       # våtmark — brown
    7: (210, 180, 140),     # öppen mark — tan
    8: (255, 0, 0),         # bebyggelse — red
    9: (0, 0, 255),         # vatten — blue
    10: (230, 180, 34),     # vete — gold
    11: (212, 130, 23),     # korn — orange
    12: (255, 255, 100),    # havre — light yellow
    13: (45, 180, 90),      # oljeväxter — green
    14: (100, 200, 100),    # vall — light green
    15: (180, 80, 40),      # potatis — brown
    16: (140, 180, 50),     # trindsäd — olive
    17: (170, 170, 170),    # övrig åker — grey
    18: (180, 120, 60),     # hygge — light brown
}


# ── NMD 10-class → Unified mapping ───────────────────────────────────────────
# NMD 10-class indices from class_schema.py:
#   0=bg, 1=pine, 2=spruce, 3=deciduous, 4=mixed, 5=wetland_forest,
#   6=open_wetland, 7=cropland, 8=open_land, 9=developed, 10=water

_NMD10_TO_UNIFIED = np.zeros(11, dtype=np.uint8)
_NMD10_TO_UNIFIED[0] = 0    # background
_NMD10_TO_UNIFIED[1] = 1    # pine → tallskog
_NMD10_TO_UNIFIED[2] = 2    # spruce → granskog
_NMD10_TO_UNIFIED[3] = 3    # deciduous → lövskog
_NMD10_TO_UNIFIED[4] = 4    # mixed → blandskog
_NMD10_TO_UNIFIED[5] = 5    # wetland forest → sumpskog
_NMD10_TO_UNIFIED[6] = 6    # open wetland → våtmark
_NMD10_TO_UNIFIED[7] = 17   # cropland → övrig åker (default, overridden by LPIS)
_NMD10_TO_UNIFIED[8] = 7    # open land → öppen mark
_NMD10_TO_UNIFIED[9] = 8    # developed → bebyggelse
_NMD10_TO_UNIFIED[10] = 9   # water → vatten

# LPIS crop class → Unified mapping
_LPIS_TO_UNIFIED = np.zeros(9, dtype=np.uint8)
_LPIS_TO_UNIFIED[0] = 0     # no parcel → keep NMD
_LPIS_TO_UNIFIED[1] = 10    # vete
_LPIS_TO_UNIFIED[2] = 11    # korn
_LPIS_TO_UNIFIED[3] = 12    # havre
_LPIS_TO_UNIFIED[4] = 13    # oljeväxter
_LPIS_TO_UNIFIED[5] = 14    # vall
_LPIS_TO_UNIFIED[6] = 15    # potatis
_LPIS_TO_UNIFIED[7] = 16    # trindsäd
_LPIS_TO_UNIFIED[8] = 17    # övrig åker


def nmd10_to_unified(nmd_label: np.ndarray) -> np.ndarray:
    """Convert NMD 10-class labels to unified schema.

    For LULC tiles that have only NMD labels (no LPIS).
    NMD cropland (class 7) becomes 'övrig åker' (class 17).

    Args:
        nmd_label: (H, W) uint8, NMD 10-class indices (0-10)

    Returns:
        (H, W) uint8, unified indices (0-17)
    """
    return _NMD10_TO_UNIFIED[np.clip(nmd_label, 0, 10)]


def merge_nmd_lpis(nmd_label: np.ndarray, lpis_mask: np.ndarray) -> np.ndarray:
    """Merge NMD background labels with LPIS crop detail.

    For crop tiles that have both NMD and LPIS labels.
    - Where LPIS has a crop class (1-8), use the specific crop (10-17)
    - Where NMD has cropland (7) but no LPIS, use 'övrig åker' (17)
    - Everywhere else, use NMD classes (1-9)

    Args:
        nmd_label: (H, W) uint8, NMD 10-class indices (0-10)
        lpis_mask: (H, W) uint8, LPIS crop classes (0-8, 0=no parcel)

    Returns:
        (H, W) uint8, unified indices (0-17)
    """
    # Start with NMD → unified
    unified = nmd10_to_unified(nmd_label)

    # Override with LPIS crop detail where available
    has_lpis = lpis_mask > 0
    unified[has_lpis] = _LPIS_TO_UNIFIED[np.clip(lpis_mask[has_lpis], 0, 8)]

    return unified


def merge_all(
    nmd_label: np.ndarray,
    lpis_mask: np.ndarray | None = None,
    harvest_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Merge NMD + LPIS + SKS harvest into unified 19-class label.

    Priority: LPIS crops > SKS harvest > NMD background.

    Args:
        nmd_label: (H, W) uint8, NMD 10-class (0-10)
        lpis_mask: (H, W) uint8, LPIS crop classes (0-8), or None
        harvest_mask: (H, W) uint8, binary harvest mask (0/1), or None

    Returns:
        (H, W) uint8, unified 19-class (0-18)
    """
    # Start with NMD
    unified = nmd10_to_unified(nmd_label)

    # Overlay LPIS crops where available
    if lpis_mask is not None:
        has_lpis = lpis_mask > 0
        unified[has_lpis] = _LPIS_TO_UNIFIED[np.clip(lpis_mask[has_lpis], 0, 8)]

    # Overlay harvest (only on forest pixels 1-5)
    if harvest_mask is not None:
        is_forest = (unified >= 1) & (unified <= 5)
        unified[is_forest & (harvest_mask > 0)] = 18

    return unified


def get_class_weights(
    class_counts: dict[int, int],
    max_weight: float = 10.0,
) -> np.ndarray:
    """Compute inverse-frequency class weights, capped at max_weight.

    Same strategy as LULC training (class_schema.py).

    Args:
        class_counts: {class_idx: pixel_count}
        max_weight: Maximum weight (default 10× like LULC)

    Returns:
        (NUM_UNIFIED_CLASSES,) float32 weight array
    """
    total = sum(class_counts.values())
    weights = np.ones(NUM_UNIFIED_CLASSES, dtype=np.float32)

    for cls, count in class_counts.items():
        if 0 < cls < NUM_UNIFIED_CLASSES and count > 0:
            w = total / (NUM_UNIFIED_CLASSES * count)
            weights[cls] = min(w, max_weight)

    weights[0] = 0.0  # ignore background
    return weights
