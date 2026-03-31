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

NUM_UNIFIED_CLASSES = 20

UNIFIED_CLASSES = {
    0: "bakgrund",          # ignore_index
    # NMD-derived (1-10)
    1: "tallskog",          # NMD 10-class: 1 (forest_pine)
    2: "granskog",          # NMD 10-class: 2 (forest_spruce)
    3: "lövskog",           # NMD 10-class: 3 (forest_deciduous)
    4: "blandskog",         # NMD 10-class: 4 (forest_mixed)
    5: "sumpskog",          # NMD 10-class: 5 (forest_wetland)
    6: "tillfälligt ej skog",  # NMD raw 5 (temp_non_forest — clearcut regrowth)
    7: "våtmark",           # NMD 10-class: 6 (open_wetland)
    8: "öppen mark",        # NMD 10-class: 8 (open_land)
    9: "bebyggelse",        # NMD 10-class: 9 (developed)
    10: "vatten",           # NMD 10-class: 10 (water)
    # LPIS crop detail (11-18) — replaces NMD class 7 (cropland)
    11: "vete",             # LPIS crop class 1
    12: "korn",             # LPIS crop class 2
    13: "havre",            # LPIS crop class 3
    14: "oljeväxter",       # LPIS crop class 4
    15: "vall",             # LPIS crop class 5
    16: "potatis",          # LPIS crop class 6
    17: "trindsäd",         # LPIS crop class 7
    18: "övrig åker",       # LPIS crop class 8 + unmapped cropland
    # SKS harvest (19)
    19: "hygge",            # SKS utförda avverkningar (harvested forest)
}

UNIFIED_CLASS_NAMES = [UNIFIED_CLASSES[i] for i in range(NUM_UNIFIED_CLASSES)]

# Color palette as flat list (index-aligned with UNIFIED_CLASS_NAMES)
# Color palette for visualization (RGB tuples)
UNIFIED_COLORS = {
    0: (0, 0, 0),           # bakgrund
    1: (0, 100, 0),         # tallskog — dark green
    2: (34, 139, 34),       # granskog — forest green
    3: (50, 205, 50),       # lövskog — lime green
    4: (60, 179, 113),      # blandskog — medium sea green
    5: (46, 79, 46),        # sumpskog — dark olive
    6: (160, 200, 120),     # tillfälligt ej skog — light olive green
    7: (139, 90, 43),       # våtmark — brown
    8: (210, 180, 140),     # öppen mark — tan
    9: (255, 0, 0),         # bebyggelse — red
    10: (0, 0, 255),        # vatten — blue
    11: (230, 180, 34),     # vete — gold
    12: (212, 130, 23),     # korn — orange
    13: (255, 255, 100),    # havre — light yellow
    14: (45, 180, 90),      # oljeväxter — green
    15: (100, 200, 100),    # vall — light green
    16: (180, 80, 40),      # potatis — brown
    17: (140, 180, 50),     # trindsäd — olive
    18: (170, 170, 170),    # övrig åker — grey
    19: (180, 120, 60),     # hygge — light brown
}


# ── NMD raw 19-class → Unified mapping ────────────────────────────────────────
# NMD raw 19-class from class_schema.py (nmd_raster_to_lulc with num_classes=19):
#   0=bg, 1=pine, 2=spruce, 3=deciduous, 4=mixed, 5=temp_non_forest,
#   6-10=wetland_forest variants, 11=open_wetland, 12=cropland,
#   13-14=open_land, 15-17=developed, 18-19=water

_NMD19_TO_UNIFIED = np.zeros(20, dtype=np.uint8)
_NMD19_TO_UNIFIED[0] = 0     # background
_NMD19_TO_UNIFIED[1] = 1     # forest_pine → tallskog
_NMD19_TO_UNIFIED[2] = 2     # forest_spruce → granskog
_NMD19_TO_UNIFIED[3] = 3     # forest_deciduous → lövskog
_NMD19_TO_UNIFIED[4] = 4     # forest_mixed → blandskog
_NMD19_TO_UNIFIED[5] = 6     # forest_temp_non_forest → tillfälligt ej skog (NEW)
_NMD19_TO_UNIFIED[6] = 5     # forest_wetland_pine → sumpskog
_NMD19_TO_UNIFIED[7] = 5     # forest_wetland_spruce → sumpskog
_NMD19_TO_UNIFIED[8] = 5     # forest_wetland_deciduous → sumpskog
_NMD19_TO_UNIFIED[9] = 5     # forest_wetland_mixed → sumpskog
_NMD19_TO_UNIFIED[10] = 5    # forest_wetland_temp → sumpskog
_NMD19_TO_UNIFIED[11] = 7    # open_wetland → våtmark
_NMD19_TO_UNIFIED[12] = 18   # cropland → övrig åker (default, overridden by LPIS)
_NMD19_TO_UNIFIED[13] = 8    # open_land_bare → öppen mark
_NMD19_TO_UNIFIED[14] = 8    # open_land_vegetated → öppen mark
_NMD19_TO_UNIFIED[15] = 9    # developed_buildings → bebyggelse
_NMD19_TO_UNIFIED[16] = 9    # developed_infrastructure → bebyggelse
_NMD19_TO_UNIFIED[17] = 9    # developed_roads → bebyggelse
_NMD19_TO_UNIFIED[18] = 10   # water_lakes → vatten
_NMD19_TO_UNIFIED[19] = 10   # water_sea → vatten

# Backward compat: 10-class → Unified (for tiles with pre-mapped 10-class labels)
_NMD10_TO_UNIFIED = np.zeros(11, dtype=np.uint8)
_NMD10_TO_UNIFIED[0] = 0     # background
_NMD10_TO_UNIFIED[1] = 1     # pine → tallskog
_NMD10_TO_UNIFIED[2] = 2     # spruce → granskog
_NMD10_TO_UNIFIED[3] = 3     # deciduous → lövskog
_NMD10_TO_UNIFIED[4] = 4     # mixed → blandskog (temp_non_forest lost in 10-class)
_NMD10_TO_UNIFIED[5] = 5     # wetland forest → sumpskog
_NMD10_TO_UNIFIED[6] = 7     # open wetland → våtmark
_NMD10_TO_UNIFIED[7] = 18    # cropland → övrig åker
_NMD10_TO_UNIFIED[8] = 8     # open land → öppen mark
_NMD10_TO_UNIFIED[9] = 9     # developed → bebyggelse
_NMD10_TO_UNIFIED[10] = 10   # water → vatten

# LPIS crop class → Unified mapping
_LPIS_TO_UNIFIED = np.zeros(9, dtype=np.uint8)
_LPIS_TO_UNIFIED[0] = 0     # no parcel → keep NMD
_LPIS_TO_UNIFIED[1] = 11    # vete
_LPIS_TO_UNIFIED[2] = 12    # korn
_LPIS_TO_UNIFIED[3] = 13    # havre
_LPIS_TO_UNIFIED[4] = 14    # oljeväxter
_LPIS_TO_UNIFIED[5] = 15    # vall
_LPIS_TO_UNIFIED[6] = 16    # potatis
_LPIS_TO_UNIFIED[7] = 17    # trindsäd
_LPIS_TO_UNIFIED[8] = 18    # övrig åker

# Harvest class index
HARVEST_CLASS = 19

# Forest classes eligible for harvest override
_FOREST_UNIFIED = frozenset({1, 2, 3, 4, 5, 6})  # tallskog..tillfälligt ej skog


def nmd10_to_unified(nmd_label: np.ndarray) -> np.ndarray:
    """Convert NMD 10-class labels to unified schema.

    Note: 10-class input has temp_non_forest merged into mixed.
    Use nmd19_to_unified() for full-resolution NMD labels.

    Args:
        nmd_label: (H, W) uint8, NMD 10-class indices (0-10)

    Returns:
        (H, W) uint8, unified indices (0-18)
    """
    return _NMD10_TO_UNIFIED[np.clip(nmd_label, 0, 10)]


def nmd19_to_unified(nmd_label: np.ndarray) -> np.ndarray:
    """Convert NMD raw 19-class labels to unified schema.

    Preserves temp_non_forest as its own class (6).

    Args:
        nmd_label: (H, W) uint8, NMD raw 19-class indices (0-19)

    Returns:
        (H, W) uint8, unified indices (0-18)
    """
    return _NMD19_TO_UNIFIED[np.clip(nmd_label, 0, 19)]


def merge_nmd_lpis(nmd_label: np.ndarray, lpis_mask: np.ndarray) -> np.ndarray:
    """Merge NMD labels with LPIS crop detail.

    Accepts either 10-class or 19-class NMD input (auto-detected).
    LPIS crop classes override NMD cropland pixels.

    Args:
        nmd_label: (H, W) uint8, NMD indices (0-10 or 0-19)
        lpis_mask: (H, W) uint8, LPIS crop classes (0-8, 0=no parcel)

    Returns:
        (H, W) uint8, unified indices
    """
    # Auto-detect: 19-class if max > 10
    if nmd_label.max() > 10:
        unified = nmd19_to_unified(nmd_label)
    else:
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
    """Merge NMD + LPIS + SKS harvest into unified 20-class label.

    Priority: LPIS crops > SKS harvest > NMD background.

    Args:
        nmd_label: (H, W) uint8, NMD (10-class or raw 19-class)
        lpis_mask: (H, W) uint8, LPIS crop classes (0-8), or None
        harvest_mask: (H, W) uint8, binary harvest mask (0/1), or None

    Returns:
        (H, W) uint8, unified 20-class (0-19)
    """
    # Start with NMD (auto-detect 10 vs 19 class)
    unified = merge_nmd_lpis(nmd_label, lpis_mask) if lpis_mask is not None else (
        nmd19_to_unified(nmd_label) if nmd_label.max() > 10
        else nmd10_to_unified(nmd_label)
    )

    # Overlay harvest (only on forest pixels)
    if harvest_mask is not None:
        is_forest = np.isin(unified, list(_FOREST_UNIFIED))
        unified[is_forest & (harvest_mask > 0)] = HARVEST_CLASS

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


# ── Convenience exports for dashboards/scripts ───────────────────────────────

UNIFIED_COLOR_LIST = [UNIFIED_COLORS[i] for i in range(NUM_UNIFIED_CLASSES)]
"""Index-aligned color list: UNIFIED_COLOR_LIST[cls_id] → (R, G, B)."""


def export_schema_json(path: str | None = None) -> dict:
    """Export unified schema as JSON for dashboards and visualization scripts.

    Returns dict and optionally writes to file. Eliminates the need for
    hardcoded class names/colors in HTML/JS/bash files.

    Args:
        path: Optional file path to write JSON. If None, just returns dict.

    Returns:
        Schema dict with class_names, colors_rgb, colors_css, num_classes.
    """
    import json

    schema = {
        "num_classes": NUM_UNIFIED_CLASSES,
        "class_names": UNIFIED_CLASS_NAMES,
        "colors_rgb": [list(UNIFIED_COLORS[i]) for i in range(NUM_UNIFIED_CLASSES)],
        "colors_css": {
            UNIFIED_CLASS_NAMES[i]: f"rgb({UNIFIED_COLORS[i][0]},{UNIFIED_COLORS[i][1]},{UNIFIED_COLORS[i][2]})"
            for i in range(NUM_UNIFIED_CLASSES)
        },
    }

    if path:
        with open(path, "w") as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)

    return schema
