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

NUM_UNIFIED_CLASSES = 23

UNIFIED_CLASSES = {
    0: "bakgrund",          # ignore_index
    # NMD-derived (1-10)
    1: "tallskog",          # NMD: forest_pine
    2: "granskog",          # NMD: forest_spruce
    3: "lövskog",           # NMD: forest_deciduous
    4: "blandskog",         # NMD: forest_mixed
    5: "sumpskog",          # NMD: forest_wetland (all subtypes)
    6: "tillfälligt ej skog",  # NMD raw 5: clearcut regrowth / young forest
    7: "våtmark",           # NMD: open_wetland
    8: "öppen mark",        # NMD: open_land
    9: "bebyggelse",        # NMD: developed
    10: "vatten",           # NMD: water
    # LPIS crop detail (11-21) — replaces NMD cropland
    # SJV grödkoder: 1=korn(h), 2=korn(v), 3=havre, 4=vete(h), 5=vete(v)
    #                7=rågvete(h), 8=råg, 20-28=oljeväxter, 30-39=trindsäd
    #                45-46=potatis, 47=sockerbetor, 49-50=vall, 52=bete, 80=grönfoder
    11: "vete",             # SJV 4, 5, 29, 307 (höst/vår/rågvete)
    12: "korn",             # SJV 1, 2, 12, 13, 315 (höst/vår/blandsäd)
    13: "havre",            # SJV 3
    14: "oljeväxter",       # SJV 20-28, 38, 40 (raps/rybs/lin/solros)
    15: "slåttervall",      # SJV 49, 50, 57, 58, 59, 62, 302 (vall på åker)
    16: "bete",             # SJV 52, 53, 54, 55, 56, 61, 89, 90, 95 (betesmark)
    17: "potatis",          # SJV 45, 46, 311
    18: "sockerbetor",      # SJV 47, 48
    19: "trindsäd",         # SJV 30-37, 39, 43 (ärter/bönor)
    20: "råg",              # SJV 7, 8, 29, 317 (råg/rågvete)
    21: "övrig åker",       # SJV 9, 60, 74, 77, 80, 81, 85, 87, 88 + unmapped cropland
    # SKS harvest (22)
    22: "hygge",            # SKS utförda avverkningar (harvested forest)
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
    14: (45, 180, 90),      # oljeväxter — bright green
    15: (100, 200, 100),    # slåttervall — light green
    16: (80, 160, 60),      # bete — darker green
    17: (180, 80, 40),      # potatis — brown
    18: (200, 100, 200),    # sockerbetor — purple
    19: (140, 180, 50),     # trindsäd — olive
    20: (190, 150, 80),     # råg — wheat/tan
    21: (170, 170, 170),    # övrig åker — grey
    22: (0, 206, 209),      # hygge — turquoise
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
_NMD19_TO_UNIFIED[12] = 21   # cropland → övrig åker (default, overridden by LPIS)
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
_NMD10_TO_UNIFIED[7] = 21    # cropland → övrig åker (overridden by LPIS)
_NMD10_TO_UNIFIED[8] = 8     # open land → öppen mark
_NMD10_TO_UNIFIED[9] = 9     # developed → bebyggelse
_NMD10_TO_UNIFIED[10] = 10   # water → vatten

# ── SJV grödkod → Unified class mapping ──────────────────────────────────────
# Direct mapping from SJV crop codes (grdkod_mar) to unified class.
# Codes are consistent across years (2018-2026). New codes added from 2022.
# Source: Jordbruksverket grödkodslista 2026 + areal-verifiering mot LPIS.

SJV_TO_UNIFIED = {
    # Vete (unified 11)
    4: 11, 5: 11, 307: 11, 316: 11,          # höstvete, vårvete, speltvete, flerårigt
    # Korn (unified 12)
    1: 12, 2: 12, 12: 12, 13: 12, 315: 12,   # höstkorn, vårkorn, blandsäd
    # Havre (unified 13)
    3: 13, 10: 13, 15: 13,                     # havre, bovete, hirs
    # Oljeväxter (unified 14)
    20: 14, 21: 14, 22: 14, 23: 14, 24: 14,  # raps höst/vår, rybs höst/vår, solros
    25: 14, 26: 14, 27: 14, 28: 14,           # oljeväxtförsök, högerukaraps, vitsenap, oljerättika
    38: 14, 40: 14, 41: 14, 42: 14,           # sojabönor(olja), oljelin, spånadslin, hampa
    85: 14, 86: 14, 87: 14, 88: 14,           # gamla koder (2018): höstraps, vårraps, höstrybs, vårrybs
    # Slåttervall (unified 15)
    49: 15, 50: 15, 57: 15, 58: 15, 59: 15,  # slåttervall, betesvall, frövall
    62: 15, 63: 15, 302: 15, 308: 15,         # klöverfrövall, energigräs, lusern, sötväppling
    6: 15, 301: 15, 300: 15,                   # baljväxt-grovfoder, westerwoldiskt rajgräs, fodermärgkål
    # Bete (unified 16)
    52: 16, 53: 16, 54: 16, 55: 16, 56: 16,  # betesmark, slåtteräng, skogsbete, fäbodbete, alvarbete
    61: 16, 89: 16, 90: 16, 95: 16,           # fäbodbete(gårdsstöd), mosaikbete, gräsfattiga, restaurering
    # Potatis (unified 17)
    45: 17, 46: 17, 311: 17,                   # matpotatis, stärkelsepotatis, färskpotatis
    70: 17, 71: 17, 72: 17,                    # gamla koder: matpotatis, stärkelsepotatis, utsädespotatis
    # Sockerbetor (unified 18)
    47: 18, 48: 18,                             # sockerbetor, foderbetor
    # Trindsäd (unified 19)
    30: 19, 31: 19, 32: 19, 33: 19, 34: 19,  # ärter, konservärter, åkerbönor, sötlupiner, proteingrödor
    35: 19, 36: 19, 37: 19, 39: 19, 43: 19,  # bruna bönor, vicker, kikärter, sojabönor(foder), bönor övr.
    # Råg (unified 20)
    7: 20, 8: 20, 29: 20, 317: 20,            # rågvete höst/vår, råg, flerårigt
    11: 20, 14: 20,                             # spannmålsförsök, kanariefrö → råg/övrigt spannmål
    # Övrig åker (unified 21)
    9: 21, 16: 21,                              # majs, stråsäd till grönfoder
    60: 21, 66: 21, 77: 21, 81: 21,           # träda, anpassad skyddszon, skyddszon, gröngödsling
    74: 21, 79: 21,                             # grönsaksodling, kryddväxter
    80: 21,                                     # grönfoder
    # Ej mappade → övrig åker som default
}

# Default for unmapped SJV codes
_SJV_DEFAULT = 21  # övrig åker

# Harvest class index
HARVEST_CLASS = 22


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


def merge_nmd_sjv(nmd_label: np.ndarray, sjv_codes: np.ndarray) -> np.ndarray:
    """Merge NMD labels with LPIS SJV grödkoder.

    Accepts either 10-class or 19-class NMD input (auto-detected).
    SJV crop codes override NMD where parcels exist (sjv_codes > 0).

    Args:
        nmd_label: (H, W) uint8, NMD indices (0-10 or 0-19)
        sjv_codes: (H, W) uint16, raw SJV grödkoder (0=no parcel)

    Returns:
        (H, W) uint8, unified indices
    """
    # Auto-detect: 19-class if max > 10
    if nmd_label.max() > 10:
        unified = nmd19_to_unified(nmd_label)
    else:
        unified = nmd10_to_unified(nmd_label)

    # Override with SJV crop detail where parcels exist
    has_parcel = sjv_codes > 0
    if has_parcel.any():
        for sjv_code, unified_class in SJV_TO_UNIFIED.items():
            mask = sjv_codes == sjv_code
            if mask.any():
                unified[mask] = unified_class
        # Unmapped SJV codes → övrig åker
        unmapped = has_parcel & ~np.isin(sjv_codes, list(SJV_TO_UNIFIED.keys()))
        unified[unmapped] = _SJV_DEFAULT

    return unified


# Backward compat alias
merge_nmd_lpis = merge_nmd_sjv


def merge_all(
    nmd_label: np.ndarray,
    lpis_mask: np.ndarray | None = None,
    harvest_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Merge NMD + LPIS + SKS harvest into unified label.

    Semantic gating rules (NMD acts as gate for each source):
      Forest (1–6)   + SKS clearcut   → Hygge (22)
      Åker (21)      + LPIS crop       → Crop class (11–21)
      Öppen mark (8) + LPIS crop       → Crop class (11–21)
      Bebyggelse (9)                   → stays as NMD (infra/buildings not overridable)
      Vatten (10)                      → stays as NMD

    Priority within eligible pixels: LPIS > SKS > NMD.

    Args:
        nmd_label: (H, W) uint8, NMD (10-class or raw 19-class)
        lpis_mask: (H, W) uint8 or uint16, LPIS SJV grödkoder or
            old-style crop classes (0-8). Auto-detected by max value.
        harvest_mask: (H, W) uint8, binary harvest mask (0/1), or None

    Returns:
        (H, W) uint8, unified classes (0-{NUM_UNIFIED_CLASSES-1})
    """
    # Step 1: NMD baseline (used both as output and as gate)
    if nmd_label.max() > 10:
        unified = nmd19_to_unified(nmd_label)
    else:
        unified = nmd10_to_unified(nmd_label)

    nmd_base = unified.copy()   # gate reference — never modified

    # Step 2: LPIS crops — where NMD says agricultural (övrig åker=21)
    #          OR open land (öppen mark bar=8).
    #          Infrastructure (9) and buildings (9) are NOT eligible — NMD wins.
    _NMD_AGRI = np.array([21, 8], dtype=np.uint8)
    where_agri = np.isin(nmd_base, _NMD_AGRI)

    if lpis_mask is not None and where_agri.any():
        # Auto-detect: old 0-8 intermediate or raw SJV codes
        if lpis_mask.max() > 10:
            sjv_codes = lpis_mask.astype(np.uint16)
        else:
            _OLD_TO_SJV = {1: 4, 2: 2, 3: 3, 4: 20, 5: 50, 6: 70, 7: 30, 8: 9}
            sjv_codes = np.zeros_like(lpis_mask, dtype=np.uint16)
            for old_cls, sjv_code in _OLD_TO_SJV.items():
                sjv_codes[lpis_mask == old_cls] = sjv_code

        has_parcel = (sjv_codes > 0) & where_agri   # gate: parcel AND NMD=agri
        if has_parcel.any():
            for sjv_code, unified_class in SJV_TO_UNIFIED.items():
                mask = sjv_codes == sjv_code
                if mask.any():
                    unified[mask & where_agri] = unified_class
            unmapped = has_parcel & ~np.isin(sjv_codes, list(SJV_TO_UNIFIED.keys()))
            unified[unmapped] = _SJV_DEFAULT

    # Step 3: SKS harvest — only where NMD says forest (classes 1–6)
    _NMD_FOREST = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint8)
    where_forest = np.isin(nmd_base, _NMD_FOREST)

    if harvest_mask is not None and where_forest.any():
        unified[(harvest_mask > 0) & where_forest] = HARVEST_CLASS

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
