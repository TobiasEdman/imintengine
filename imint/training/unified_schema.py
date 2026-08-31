"""
imint/training/unified_schema.py — Unified LULC + Crop + Harvest class schema

Merges NMD land cover (19-class sequential) with LPIS crop detail and SKS harvest
data into a single 23-class segmentation schema.

NMD provides background (forest, water, developed, wetland). LPIS overrides NMD on
agricultural pixels with crop-specific classes. SKS marks harvested forest (hygge).

Usage:
    from imint.training.unified_schema import merge_all, UNIFIED_CLASSES

    # NMD-only tile:
    unified = nmd19_to_unified(nmd_19class)

    # NMD + LPIS:
    unified = merge_nmd_sjv(nmd_19class, sjv_codes)

    # Full pipeline (NMD + LPIS + SKS):
    unified = merge_all(nmd_19class, lpis_mask, harvest_mask)
"""
from __future__ import annotations

import numpy as np

# ── Unified Class Schema ──────────────────────────────────────────────────────

NUM_UNIFIED_CLASSES = 28

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
    #                45-46=potatis, 47=sockerbetor, 49-50=vall, 52=bete, 9=majs
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
    21: "majs",             # SJV 9 — C4 photosynthesis, spectrally distinct
    # SKS harvest (22)
    22: "hygge",            # SKS utförda avverkningar (harvested forest)
    # NMD2023-only fine classes (23-28) — see nmd2023_to_unified().
    # Only produced when labels are built from the NMD2023 base raster; the
    # NMD2018 base cannot code these (docs/experiments/nmd2023_label_source_retrain.md).
    23: "torvtäkt",         # NMD2023 54 — active peat extraction
    24: "buskdominerad mark",   # NMD2023 421/42xx — shrub-dominated open land
    25: "risdominerad mark",    # NMD2023 422/42xx — dwarf-shrub-dominated
    26: "gräsdominerad mark",   # NMD2023 423/42xx — grass-dominated
    27: "öppen mark utan vegetation",  # NMD2023 411 — bare ground/rock
    # NOTE: NMD2023 glaciär (412) + snöfält (413) → öppen mark (8). A dedicated
    # snö/is class had 0 pixel support under the pure-NMD2023 southern coverage
    # (glaciers/permanent snow are in the far-northern fjäll v2.1 doesn't cover),
    # so it is not carried. Re-introduce if northern coverage is added later.
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
    21: (220, 200,   0),    # majs — corn yellow  #dcc800
    22: (0, 206, 209),      # hygge — turquoise
    23: (140, 25, 100),     # torvtäkt — dark magenta (NMD2023 palette #8c1964)
    24: (171, 200, 166),    # buskdominerad — sage green
    25: (205, 170, 102),    # risdominerad — tan-brown
    26: (255, 210, 126),    # gräsdominerad — light amber
    27: (224, 224, 224),    # öppen mark utan vegetation — light grey
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
_NMD19_TO_UNIFIED[12] = 0    # cropland → background (unknown w/o LPIS parcel)
_NMD19_TO_UNIFIED[13] = 8    # open_land_bare → öppen mark
_NMD19_TO_UNIFIED[14] = 8    # open_land_vegetated → öppen mark
_NMD19_TO_UNIFIED[15] = 9    # developed_buildings → bebyggelse
_NMD19_TO_UNIFIED[16] = 9    # developed_infrastructure → bebyggelse
_NMD19_TO_UNIFIED[17] = 9    # developed_roads → bebyggelse
_NMD19_TO_UNIFIED[18] = 10   # water_lakes → vatten
_NMD19_TO_UNIFIED[19] = 10   # water_sea → vatten

# ── NMD2023 raw code → Unified mapping (direct, uint16) ───────────────────────
# NMD2023 basskikt v2.1 codes differ from NMD2018 for open land (4-digit codes
# > 255) and add fine classes, so it cannot reuse the uint8 nmd_raster_to_lulc
# LUT. Forest codes 111-128 are IDENTICAL to NMD2018 and map to the SAME unified
# classes (so a model finetuned from v8b keeps a consistent forest label space);
# only the NMD2023-new detail (23-28) is introduced. 113 barrblandskog follows
# the existing NMD2018 grouping (→ lövskog) for consistency, not correctness.
# Codes from the shipped legend data/nmd2023/sidecar/NMD2023bas_v2_1.qml.
NMD2023_TO_UNIFIED: dict[int, int] = {
    # Skogsmark på fastmark — same targets as the NMD2018 chain
    111: 1, 112: 2, 113: 3, 114: 4, 115: 3, 116: 3, 117: 3, 118: 6,
    # Skogsmark på våtmark → sumpskog (5)
    121: 5, 122: 5, 123: 5, 124: 5, 125: 5, 126: 5, 127: 5, 128: 5,
    # Låg fjällskog (NMD2023-new base class; mountain birch)
    43: 3,           # på fastmark → lövskog
    23: 5, 230: 5,   # på våtmark / övrig våtmark → sumpskog
    # Åkermark → background (LPIS overrides on parcels)
    3: 0,
    # Öppen våtmark (finindelad) → våtmark (7)
    200: 7, 211: 7, 212: 7, 213: 7, 214: 7, 215: 7, 216: 7, 217: 7, 218: 7,
    221: 7, 222: 7, 223: 7, 224: 7, 225: 7, 226: 7, 227: 7, 228: 7,
    # Anlagd och bebyggd mark
    51: 9, 52: 9, 53: 9,
    54: 23,          # torvtäkt → egen klass 23
    # Vatten
    61: 10, 62: 10,
    # Öppen fastmark — structure broken out, moisture level (4-digit) collapsed
    411: 27,                                   # utan vegetation (bar) → 27
    412: 8, 413: 8,                            # glaciär + snöfält → öppen mark (rare/northern; no own class)
    421: 24, 4211: 24, 4212: 24, 4213: 24,     # buskdominerad → 24
    422: 25, 4221: 25, 4222: 25, 4223: 25,     # risdominerad → 25
    423: 26, 4231: 26, 4232: 26, 4233: 26,     # gräsdominerad → 26
}

_NMD2023_LUT_SIZE = max(NMD2023_TO_UNIFIED) + 1
_NMD2023_TO_UNIFIED = np.zeros(_NMD2023_LUT_SIZE, dtype=np.uint8)
for _code, _uni in NMD2023_TO_UNIFIED.items():
    _NMD2023_TO_UNIFIED[_code] = _uni


def nmd2023_to_unified(nmd_raw: np.ndarray) -> np.ndarray:
    """Map raw NMD2023 basskikt v2.1 codes (uint16) → unified classes (0-28).

    Direct raw→unified (no 19-class intermediate): NMD2023's 4-digit open-land
    codes exceed uint8, and it carries fine classes (23-28) the NMD2018 chain
    has no codes for. Forest codes 111-128 resolve to the SAME unified classes
    as NMD2018 so a model finetuned from v8b keeps a consistent forest label
    space. Unmapped / out-of-range codes → 0 (background), which is also what a
    no-data pixel (raster 0, outside NMD2023's rolling extent) resolves to.
    """
    raw = np.asarray(nmd_raw)
    out = np.zeros(raw.shape, dtype=np.uint8)
    in_range = (raw >= 0) & (raw < _NMD2023_LUT_SIZE)
    out[in_range] = _NMD2023_TO_UNIFIED[raw[in_range].astype(np.int64)]
    return out

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
    # Majs (unified 21) — C4 photosynthesis, unique NIR/NDVI temporal signature
    9: 21,                                      # majs
    # Grönfoder/cut-green crops → slåttervall (unified 15)
    # These are harvested green before maturity — phenology matches slåttervall
    16: 15, 67: 15, 68: 15, 80: 15, 81: 15,   # stråsäd/baljväxt till grönfoder, gröngödsling
    # Träda/fallow → öppen mark (unified 8) — bare or sparse vegetation
    60: 8,
    # Skyddszon (66,77), grönsaksodling (74), kryddväxter (79) fall through to
    # _SJV_DEFAULT=0 (background) — sub-pixel geometry or insufficient spectral contrast
    # Ej mappade → background (0) via _SJV_DEFAULT
}

# Default for unmapped SJV codes — background so unknown cropland does not
# contaminate training with spectrally heterogeneous noise
_SJV_DEFAULT = 0

# Harvest class index
HARVEST_CLASS = 22


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
    """Merge NMD 19-class labels with LPIS SJV grödkoder.

    SJV crop codes override NMD where parcels exist (sjv_codes > 0).
    Input must be 19-class sequential NMD (output of nmd_raster_to_lulc).

    Args:
        nmd_label: (H, W) uint8, NMD 19-class sequential indices (0-19).
        sjv_codes: (H, W) uint16, raw SJV grödkoder (0 = no parcel).

    Returns:
        (H, W) uint8, unified class indices (0-22).
    """
    unified = nmd19_to_unified(nmd_label)
    has_parcel = sjv_codes > 0
    if has_parcel.any():
        sjv_mapped = np.isin(sjv_codes, list(SJV_TO_UNIFIED.keys()))
        for sjv_code, unified_class in SJV_TO_UNIFIED.items():
            mask = sjv_codes == sjv_code
            if mask.any():
                unified[mask] = unified_class
        unified[has_parcel & ~sjv_mapped] = _SJV_DEFAULT
    return unified


def merge_all(
    nmd_label: np.ndarray,
    lpis_mask: np.ndarray | None = None,
    harvest_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Merge NMD 19-class + LPIS + SKS harvest into unified 23-class label.

    Semantic gating rules (raw NMD 19-class input acts as gate for LPIS overlay):
      Forest (NMD 1–6)          + SKS clearcut   → Hygge (22)
      Cropland (NMD 12–14)      + LPIS parcel    → Crop class (9–21)
      Bebyggelse (unified 9)                      → NMD wins, never overridden
      Vatten (unified 10)                         → NMD wins, never overridden

    Priority: LPIS > SKS > NMD.

    Args:
        nmd_label: (H, W) uint8, NMD 19-class sequential indices (0-19).
                   Must be output of nmd_raster_to_lulc() from class_schema.
        lpis_mask: (H, W) uint16, raw SJV grödkoder (0 = no parcel), or None.
        harvest_mask: (H, W) uint8, binary SKS harvest mask (0/1), or None.

    Returns:
        (H, W) uint8, unified class indices (0-{NUM_UNIFIED_CLASSES-1}).
    """
    # Step 1: NMD baseline
    unified = nmd19_to_unified(nmd_label)

    nmd_base = unified.copy()   # gate reference for forest (Step 3) — never modified

    # Step 2: LPIS crops — gate on raw NMD 19-class input, NOT on unified output.
    #   Raw NMD 12=cropland, 13=open_land_bare, 14=open_land_veg are eligible.
    #   Must use raw input: NMD cropland (12) maps to background (0) in unified, so
    #   gating on nmd_base would silently drop all LPIS parcels on cropland.
    #   Developed (15-17) and water (18-19) are NOT eligible — NMD wins.
    _NMD_AGRI_RAW = np.array([12, 13, 14], dtype=np.uint8)
    where_agri = np.isin(nmd_label, _NMD_AGRI_RAW)

    if lpis_mask is not None and where_agri.any():
        sjv_codes = np.asarray(lpis_mask, dtype=np.uint16)
        has_parcel = (sjv_codes > 0) & where_agri
        if has_parcel.any():
            sjv_mapped = np.isin(sjv_codes, list(SJV_TO_UNIFIED.keys()))
            for sjv_code, unified_class in SJV_TO_UNIFIED.items():
                mask = (sjv_codes == sjv_code) & where_agri
                if mask.any():
                    unified[mask] = unified_class
            unified[has_parcel & ~sjv_mapped] = _SJV_DEFAULT

    # Step 3: SKS harvest — only where NMD says forest (classes 1–6)
    _NMD_FOREST = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint8)
    where_forest = np.isin(nmd_base, _NMD_FOREST)

    if harvest_mask is not None and where_forest.any():
        unified[(harvest_mask > 0) & where_forest] = HARVEST_CLASS

    return unified


# Raw NMD2023 codes eligible for an LPIS crop override: åkermark + öppen
# fastmark (bare + shrub/dwarf-shrub/grass). Mirrors merge_all's {12,13,14}
# (cropland + open_bare + open_veg) gate in NMD2018 raw space. Glacier/snow
# (412/413) and öppen våtmark are excluded — no farm parcels there.
_NMD2023_AGRI_RAW = np.array(
    [3, 411, 421, 422, 423, 4211, 4212, 4213, 4221, 4222, 4223, 4231, 4232, 4233],
    dtype=np.uint16,
)


def merge_all_2023(
    nmd2023_raw: np.ndarray,
    lpis_mask: np.ndarray | None = None,
    harvest_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Merge raw NMD2023 codes + LPIS + SKS harvest into the unified 29-class label.

    Same priority and gating as ``merge_all`` (LPIS > SKS > NMD), but the base
    comes from ``nmd2023_to_unified`` (0-28) instead of the NMD2018 19-class
    chain, and the LPIS eligibility gate is expressed in raw NMD2023 codes
    (``_NMD2023_AGRI_RAW``). Forest classes 1-6 stay SKS-clearcut-eligible.

    Args:
        nmd2023_raw: (H, W) uint16 raw NMD2023 basskikt v2.1 codes.
        lpis_mask: (H, W) uint16 raw SJV grödkoder (0 = no parcel), or None.
        harvest_mask: (H, W) uint8 binary SKS harvest mask (0/1), or None.

    Returns:
        (H, W) uint8, unified class indices (0-28).
    """
    raw = np.asarray(nmd2023_raw, dtype=np.uint16)
    unified = nmd2023_to_unified(raw)
    nmd_base = unified.copy()   # forest gate reference — never modified

    # Step 2: LPIS crops — gate on raw NMD2023 agri/open codes.
    where_agri = np.isin(raw, _NMD2023_AGRI_RAW)
    if lpis_mask is not None and where_agri.any():
        sjv_codes = np.asarray(lpis_mask, dtype=np.uint16)
        has_parcel = (sjv_codes > 0) & where_agri
        if has_parcel.any():
            sjv_mapped = np.isin(sjv_codes, list(SJV_TO_UNIFIED.keys()))
            for sjv_code, unified_class in SJV_TO_UNIFIED.items():
                mask = (sjv_codes == sjv_code) & where_agri
                if mask.any():
                    unified[mask] = unified_class
            unified[has_parcel & ~sjv_mapped] = _SJV_DEFAULT

    # Step 3: SKS harvest — only where NMD says forest (unified 1–6).
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
