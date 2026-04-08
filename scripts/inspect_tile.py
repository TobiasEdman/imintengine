#!/usr/bin/env python3
"""
Step-by-step label inspection for one test tile.

Builds the label from scratch in layers and saves PNG images after each step
so you can verify alignment visually before committing to a full rebuild.

Steps:
  1. Satellite RGB (natural colour from tile image)
  2. NMD raw (19-class direct from raster)
  3. NMD → unified schema (remap to 23-class)
  4. + LPIS crop overlay (correct N,E axis handling)
  5. + SKS hygge overlay (E,N — no rot90 needed)
  6. Side-by-side comparison with EXISTING tile label

Usage (on pod):
    python3 scripts/inspect_tile.py \
        --tile /data/unified_v2/tile_491280_6861280.npz \
        --nmd  /data/nmd/nmd2018bas_ogeneraliserad_v1_1.tif \
        --lpis /data/lpis/jordbruksskiften_2022.parquet \
        --sks  /data/sks/utforda_avverkningar.parquet \
        --out  /tmp/inspect
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np
# rasterio imported lazily inside functions that need it (extract_nmd, rasterize_sks)
# so step 1+2 work without system GDAL libs

# ── colour palette (mirrors unified_schema.py) ──────────────────────────────
# Sequential 19-class index colors (used by nmd_label_raw stored in tiles)
# Keys = sequential class indices 1-19 (as built by nmd_raster_to_lulc / _LUT_19)
NMD_SEQ19_COLORS = {
    0:  (30, 30, 30),    # background
    1:  (0, 90, 0),      # tallskog (pine)
    2:  (0, 140, 0),     # granskog (spruce)
    3:  (80, 200, 80),   # lovskog (deciduous)
    4:  (50, 170, 100),  # blandskog (mixed)
    5:  (200, 180, 60),  # tillfälligt ej skog (clearcut/regen)
    6:  (40, 70, 40),    # sumpskog pine
    7:  (46, 79, 46),    # sumpskog spruce
    8:  (56, 89, 56),    # sumpskog deciduous
    9:  (46, 79, 46),    # sumpskog mixed
    10: (56, 89, 56),    # sumpskog temp
    11: (139, 90, 43),   # open wetland (vatmark)
    12: (212, 170, 100), # cropland
    13: (210, 180, 140), # open land bare
    14: (200, 200, 140), # open land vegetated
    15: (255, 60, 60),   # bebyggelse buildings
    16: (200, 0, 0),     # bebyggelse infra
    17: (150, 0, 0),     # bebyggelse roads
    18: (0, 0, 255),     # vatten lakes
    19: (0, 60, 200),    # vatten sea
}

# NMD raw code colors — keyed by ACTUAL NMD raster codes (not sequential)
NMD_RAW_COLORS = {
    0:   (30, 30, 30),    # nodata
    2:   (139, 90, 43),   # open wetland (myrmark)
    3:   (212, 170, 100), # cropland
    41:  (210, 180, 140), # öppen mark, bare
    42:  (200, 200, 140), # öppen mark, vegetated
    51:  (255, 60, 60),   # bebyggelse buildings
    52:  (200, 0, 0),     # bebyggelse infra
    53:  (150, 0, 0),     # bebyggelse roads
    61:  (0, 0, 255),     # vatten lakes
    62:  (0, 60, 200),    # vatten sea
    111: (0, 90, 0),      # tallskog
    112: (0, 140, 0),     # granskog
    113: (80, 200, 80),   # lövskog (deciduous)
    114: (50, 170, 100),  # blandskog (mixed coniferous)
    115: (100, 200, 100), # triviallövskog (birch/aspen/alder)
    116: (50, 180, 60),   # ädellövskog (oak/beech)
    117: (60, 190, 80),   # triviallöv + ädellöv
    118: (200, 180, 60),  # temporärt ej skog (clearcut / young regen)
    121: (46, 79, 46),    # sumpskog — pine
    122: (56, 89, 56),    # sumpskog — spruce
    123: (66, 99, 66),    # sumpskog — deciduous
    124: (56, 89, 56),    # sumpskog — mixed
    125: (66, 99, 66),    # sumpskog — trivial deciduous
    126: (56, 89, 56),    # sumpskog — noble deciduous
    127: (60, 94, 60),    # sumpskog — trivial+noble
    128: (200, 180, 60),  # sumpskog — temp non-forest
    241: (210, 180, 140), # alpine/exploiterad open
    255: (80, 80, 80),    # unclassified
}

UNIFIED_COLORS = {
    # Base classes inherit NMD_SEQ19_COLORS exactly — same hue, same brightness
    0:  (30, 30, 30),
    1:  (0, 90, 0),      # tallskog    — NMD seq19[1]
    2:  (0, 140, 0),     # granskog    — NMD seq19[2]
    3:  (80, 200, 80),   # lövskog     — NMD seq19[3]
    4:  (50, 170, 100),  # blandskog   — NMD seq19[4]
    5:  (46, 79, 46),    # sumpskog    — NMD seq19[6] (darkest wetland green)
    6:  (200, 180, 60),  # tillfälligt ej skog — NMD seq19[5]
    7:  (139, 90, 43),   # våtmark     — NMD seq19[11]
    8:  (210, 180, 140), # öppen mark  — NMD seq19[13]
    9:  (255, 60, 60),   # bebyggelse  — NMD seq19[15]
    10: (0, 0, 255),     # vatten      — NMD seq19[18]
    # LPIS crop classes — build on NMD cropland hue (212,170,100) shifting hue/saturation
    11: (230, 200, 30),  # vete        — golden yellow
    12: (200, 130, 20),  # korn        — amber
    13: (245, 230, 80),  # havre       — pale yellow
    14: (220, 220, 0),   # oljeväxter  — bright yellow-green (rapeseed)
    15: (130, 200, 80),  # slåttervall — fresh green (hay)
    16: (90, 160, 50),   # bete        — medium green (pasture)
    17: (190, 100, 50),  # potatis     — orange-brown
    18: (210, 120, 180), # sockerbetor — pink
    19: (160, 190, 60),  # trindsäd    — yellow-green
    20: (230, 200, 100), # råg         — light amber
    21: (212, 170, 100), # övrig åker  — NMD cropland color exactly
    22: (0, 206, 209),   # hygge       — turquoise (new class, no NMD equivalent)
}

# Sequential 19-class index → unified 23-class
# Used when nmd_label_raw stores sequential indices 1-18 (not raw NMD codes)
_NMD_SEQ19_TO_UNIFIED = np.zeros(20, dtype=np.uint8)
_NMD_SEQ19_TO_UNIFIED[1]  = 1   # forest_pine -> tallskog
_NMD_SEQ19_TO_UNIFIED[2]  = 2   # forest_spruce -> granskog
_NMD_SEQ19_TO_UNIFIED[3]  = 3   # forest_deciduous -> lovskog
_NMD_SEQ19_TO_UNIFIED[4]  = 4   # forest_mixed -> blandskog
_NMD_SEQ19_TO_UNIFIED[5]  = 6   # forest_temp -> tillfälligt ej skog
_NMD_SEQ19_TO_UNIFIED[6]  = 5   # wetland_pine -> sumpskog
_NMD_SEQ19_TO_UNIFIED[7]  = 5   # wetland_spruce -> sumpskog
_NMD_SEQ19_TO_UNIFIED[8]  = 5   # wetland_deciduous -> sumpskog
_NMD_SEQ19_TO_UNIFIED[9]  = 5   # wetland_mixed -> sumpskog
_NMD_SEQ19_TO_UNIFIED[10] = 5   # wetland_temp -> sumpskog
_NMD_SEQ19_TO_UNIFIED[11] = 7   # open_wetland -> vatmark
_NMD_SEQ19_TO_UNIFIED[12] = 21  # cropland -> övrig åker (LPIS overrides)
_NMD_SEQ19_TO_UNIFIED[13] = 8   # open_land_bare -> öppen mark
_NMD_SEQ19_TO_UNIFIED[14] = 8   # open_land_veg -> öppen mark
_NMD_SEQ19_TO_UNIFIED[15] = 9   # buildings -> bebyggelse
_NMD_SEQ19_TO_UNIFIED[16] = 9   # infra -> bebyggelse
_NMD_SEQ19_TO_UNIFIED[17] = 9   # roads -> bebyggelse
_NMD_SEQ19_TO_UNIFIED[18] = 10  # lakes -> vatten
_NMD_SEQ19_TO_UNIFIED[19] = 10  # sea -> vatten

# NMD raw code (actual raster values) → unified 23-class
# Built as a 256-element LUT keyed by actual NMD codes (111=tallskog, etc.)
# Source: imint/analyzers/nmd.py NMD_LEVEL2 + unified_schema mapping
_NMD_RAW_TO_UNIFIED = np.zeros(256, dtype=np.uint8)
# Open wetland
_NMD_RAW_TO_UNIFIED[2]   = 7   # open wetland → våtmark
# Cropland (LPIS will override specific crop classes later)
_NMD_RAW_TO_UNIFIED[3]   = 21  # cropland → övrig åker (default)
# Open land
_NMD_RAW_TO_UNIFIED[41]  = 8   # open_land_bare → öppen mark
_NMD_RAW_TO_UNIFIED[241] = 8   # alpine/exploaterad → öppen mark
_NMD_RAW_TO_UNIFIED[42]  = 8   # open_land_vegetated → öppen mark
# Developed / bebyggelse
_NMD_RAW_TO_UNIFIED[51]  = 9   # buildings → bebyggelse
_NMD_RAW_TO_UNIFIED[52]  = 9   # infra → bebyggelse
_NMD_RAW_TO_UNIFIED[53]  = 9   # roads → bebyggelse
# Water
_NMD_RAW_TO_UNIFIED[61]  = 10  # lakes → vatten
_NMD_RAW_TO_UNIFIED[62]  = 10  # sea → vatten
# Upland forest (codes 111-118)
_NMD_RAW_TO_UNIFIED[111] = 1   # tallskog (pine)
_NMD_RAW_TO_UNIFIED[112] = 2   # granskog (spruce)
_NMD_RAW_TO_UNIFIED[113] = 3   # lövskog (deciduous)
_NMD_RAW_TO_UNIFIED[114] = 4   # blandskog (mixed coniferous)
_NMD_RAW_TO_UNIFIED[115] = 3   # triviallövskog → lövskog
_NMD_RAW_TO_UNIFIED[116] = 3   # ädellövskog → lövskog
_NMD_RAW_TO_UNIFIED[117] = 3   # triviallöv+ädellöv → lövskog
_NMD_RAW_TO_UNIFIED[118] = 6   # temporärt ej skog → tillfälligt ej skog
# Wetland forest (codes 121-128) → sumpskog
for _c in range(121, 128):
    _NMD_RAW_TO_UNIFIED[_c] = 5  # all sumpskog subtypes
_NMD_RAW_TO_UNIFIED[128] = 5    # wetland temp → sumpskog (was clearcut in wetland)


def label_to_rgb(label: np.ndarray, palette: dict) -> np.ndarray:
    h, w = label.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in palette.items():
        mask = label == cls
        if mask.any():
            rgb[mask] = color
    return rgb


PEAK_SUMMER_DOY = 195  # mid-July

def _best_frame(image: np.ndarray, doy: np.ndarray | None = None) -> int:
    """Return index of best summer temporal frame.
    Strategy 1: if doy provided, pick valid frame (doy>0) closest to PEAK_SUMMER_DOY.
    Strategy 2: fallback — max mean (B8A - B04) = NDVI proxy, picks peak vegetation.
    """
    T = image.shape[0] // 6
    if doy is not None and len(doy) >= T:
        valid = [(abs(int(d) - PEAK_SUMMER_DOY), i)
                 for i, d in enumerate(doy[:T]) if int(d) > 0]
        if valid:
            return min(valid)[1]
    # Fallback: max (B8A - B04) per frame — NDVI-like, favours peak summer green
    scores = [image[f * 6 + 3].mean() - image[f * 6 + 2].mean() for f in range(T)]
    return int(np.argmax(scores))


def spectral_to_rgb(image: np.ndarray, frame_idx: int = -2) -> np.ndarray:
    """(T*6, H, W) float32 → (H, W, 3) uint8 natural colour.
    Uses B04(R)/B03(G)/B02(B). frame_idx=-2 → pick brightest (summer) frame.
    Bands per frame: [B02(0), B03(1), B04(2), B8A(3), B11(4), B12(5)]
    """
    T = image.shape[0] // 6
    if frame_idx == -2:
        f = _best_frame(image)
    elif frame_idx < 0:
        f = T - 1
    else:
        f = min(frame_idx, T - 1)
    base = f * 6
    r = image[base + 2].astype(np.float32)  # B04
    g = image[base + 1].astype(np.float32)  # B03
    b = image[base + 0].astype(np.float32)  # B02
    rgb = np.stack([r, g, b], axis=-1)
    for c in range(3):
        p2, p98 = np.percentile(rgb[..., c], (2, 98))
        span = max(p98 - p2, 1e-6)
        rgb[..., c] = np.clip((rgb[..., c] - p2) / span, 0, 1)
    return (rgb * 255).astype(np.uint8), f


def spectral_to_nir(image: np.ndarray, frame_idx: int = -2,
                    doy: np.ndarray | None = None) -> tuple:
    """NIR false-color: R=B8A(NIR), G=B03(Green), B=B04(Red).  GEE bands B8,B3,B4.
    Vegetation = bright red, water = very dark, clearcuts = cyan/light.
    Uses per-band 2-98% percentile stretch — handles L1C atmospheric haze.
    Band order per frame: [B02(0), B03(1), B04(2), B8A(3), B11(4), B12(5)]
    """
    T = image.shape[0] // 6
    if frame_idx == -2:
        f = _best_frame(image, doy)
    elif frame_idx < 0:
        f = T - 1
    else:
        f = min(frame_idx, T - 1)
    base = f * 6
    # Denormalize z-score → DN (Prithvi normalization constants)
    _MEAN = np.array([1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0], np.float32)
    _STD  = np.array([2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0], np.float32)
    nir_dn = image[base + 3].astype(np.float32) * _STD[3] + _MEAN[3]  # B8A → R
    grn_dn = image[base + 1].astype(np.float32) * _STD[1] + _MEAN[1]  # B03 → G
    red_dn = image[base + 2].astype(np.float32) * _STD[2] + _MEAN[2]  # B04 → B
    fc = np.stack([nir_dn, grn_dn, red_dn], axis=-1)  # (H, W, 3)
    # Per-band 2-98% percentile stretch — removes haze offset automatically
    for c in range(3):
        lo = np.percentile(fc[:, :, c], 2)
        hi = np.percentile(fc[:, :, c], 98)
        fc[:, :, c] = np.clip((fc[:, :, c] - lo) / max(hi - lo, 1.0), 0.0, 1.0)
    return (fc * 255).astype(np.uint8), f


def save_png(arr: np.ndarray, path: str, title: str = "") -> None:
    from PIL import Image, ImageDraw
    img = Image.fromarray(arr, "RGB")
    # Scale up 2x for readability
    img = img.resize((arr.shape[1] * 2, arr.shape[0] * 2), Image.NEAREST)
    if title:
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, img.width, 22], fill=(20, 20, 30))
        # ASCII-safe title (PIL default font is latin-1 only)
        safe = title.encode("ascii", errors="replace").decode("ascii")
        draw.text((6, 4), safe, fill=(220, 220, 220))
    img.save(path)
    print(f"  Saved: {path}  ({arr.shape[1]}x{arr.shape[0]}px native)")


def extract_nmd(nmd_path: str, west: float, south: float, east: float, north: float,
                tile_px: int = 256) -> np.ndarray:
    """Extract NMD 19-class raster for bbox [W, S, E, N] in EPSG:3006 (E,N)."""
    import rasterio
    from rasterio.windows import from_bounds as rasterio_from_bounds
    from rasterio.enums import Resampling
    with rasterio.open(nmd_path) as src:
        window = rasterio_from_bounds(west, south, east, north, src.transform)
        nmd = src.read(
            1, window=window,
            out_shape=(tile_px, tile_px),
            resampling=Resampling.nearest,
        )
    print(f"  NMD raw — unique classes: {sorted(np.unique(nmd).tolist())}")
    return nmd.astype(np.uint8)


def extract_lpis(lpis_path: str, west: float, south: float, east: float, north: float,
                 tile_px: int = 256) -> np.ndarray | None:
    """
    Rasterize LPIS SJV grödkoder for a tile bbox.

    LPIS parquets store geometries in EPSG:3006 but with N,E axis order
    (x=Northing, y=Easting). The spatial query and from_bounds must use
    (south, west, north, east) = (N_min, E_min, N_max, E_max) for LPIS,
    then np.rot90(mask, 2).T converts back to standard E,N pixel order.
    """
    try:
        import geopandas as gpd
        from rasterio.transform import from_bounds as rio_from_bounds
        from rasterio.features import rasterize
        from shapely.geometry import box as shapely_box
    except ImportError:
        print("  LPIS: geopandas not available, skipping")
        return None

    # LPIS stores N as x, E as y — bbox filter not supported by this parquet,
    # so load full file and filter with cx (N,E axis order for LPIS).
    query_box = shapely_box(south, west, north, east)  # (N_south, E_west, N_north, E_east)
    lpis_all = gpd.read_parquet(lpis_path)
    # cx indexer: first arg = x range (Northing), second = y range (Easting)
    lpis = lpis_all.cx[south:north, west:east]

    if lpis.empty:
        print("  LPIS: no parcels in bbox")
        return None

    print(f"  LPIS: {len(lpis)} parcels found")
    col = "grdkod_mar" if "grdkod_mar" in lpis.columns else lpis.columns[-1]
    print(f"  LPIS: using crop code column '{col}'")

    # Rasterize in LPIS N,E space: transform uses (south, west, north, east)
    transform = rio_from_bounds(south, west, north, east, tile_px, tile_px)
    shapes = [(geom, int(code)) for geom, code in zip(lpis.geometry, lpis[col]) if code > 0]
    mask = rasterize(shapes, out_shape=(tile_px, tile_px), transform=transform,
                     fill=0, dtype=np.int32)

    # LPIS N,E axis swap: rasterized in (south,west,north,east) space with x=Northing, y=Easting.
    # rot90(2).T converts from that N,E raster space back to standard North-up E,N pixel space.
    mask = np.rot90(mask, 2).T

    unique_codes = sorted(set(mask[mask > 0].tolist()))
    print(f"  LPIS: unique SJV codes after rotation: {unique_codes[:20]}")
    return mask.astype(np.int32)


def extract_sks(sks_path: str, west: float, south: float, east: float, north: float,
                tile_year: int, tile_px: int = 256) -> np.ndarray | None:
    """
    Rasterize SKS harvest polygons for a tile bbox.

    SKS parquets store geometries in standard GIS E,N order (confirmed by
    checking bounds: x in 270k-917k = Easting range for Sweden).
    No axis swap needed. Uses Avvdatum year within [tile_year-5, tile_year].
    """
    try:
        import geopandas as gpd
        from rasterio.transform import from_bounds as rio_from_bounds
        from rasterio.features import rasterize
    except ImportError:
        print("  SKS: geopandas not available, skipping")
        return None

    # SKS uses standard E,N — bbox filter not supported, load full file and filter.
    sks_all = gpd.read_parquet(sks_path)
    # cx indexer: first arg = x range (Easting), second = y range (Northing)
    sks = sks_all.cx[west:east, south:north]

    if sks.empty:
        print("  SKS: no polygons in bbox")
        return None

    print(f"  SKS: {len(sks)} polygons found (before year filter)")

    # Filter by harvest year
    date_col = next((c for c in ["Avvdatum", "avvdatum", "datum", "date"] if c in sks.columns), None)
    if date_col:
        import pandas as pd
        sks[date_col] = pd.to_datetime(sks[date_col], errors="coerce")
        sks = sks[sks[date_col].dt.year.between(tile_year - 5, tile_year)]
        print(f"  SKS: {len(sks)} polygons after year filter [{tile_year-5}–{tile_year}]")
    else:
        print(f"  SKS: no date column found (cols: {list(sks.columns[:10])}), using all")

    if sks.empty:
        return None

    # Rasterize in standard E,N space — no axis swap
    from rasterio.transform import from_bounds as rio_from_bounds
    from rasterio.features import rasterize
    transform = rio_from_bounds(west, south, east, north, tile_px, tile_px)
    shapes = [(geom, 1) for geom in sks.geometry if geom is not None]
    mask = rasterize(shapes, out_shape=(tile_px, tile_px), transform=transform,
                     fill=0, dtype=np.uint8)

    print(f"  SKS: {mask.sum()} harvest pixels rasterized")
    return mask


def build_lpis_colored(lpis_mask: np.ndarray | None, unified: np.ndarray) -> np.ndarray:
    """Return unified label with LPIS crop codes mapped to unified classes."""
    if lpis_mask is None:
        return unified.copy()

    # SJV → unified class map (key codes)
    SJV_TO_UNIFIED = {
        4: 11, 5: 11, 307: 11, 316: 11,        # vete
        1: 12, 2: 12, 12: 12, 13: 12, 315: 12, # korn
        3: 13, 10: 13, 15: 13,                  # havre
        20: 14, 21: 14, 22: 14, 23: 14, 24: 14, 85: 14, 86: 14, 87: 14, 88: 14, # oljeväxter
        49: 15, 50: 15, 57: 15, 58: 15, 302: 15, 308: 15, 6: 15, # slåttervall
        52: 16, 53: 16, 54: 16, 55: 16, 56: 16, 61: 16, 89: 16, 90: 16, # bete
        45: 17, 46: 17, 311: 17, 70: 17, 71: 17, 72: 17,        # potatis
        47: 18, 48: 18,                          # sockerbetor
        30: 19, 31: 19, 32: 19, 33: 19, 34: 19, 35: 19, 36: 19, # trindsäd
        7: 20, 8: 20, 29: 20, 317: 20,           # råg
        9: 21, 60: 21, 77: 21, 81: 21, 80: 21,  # övrig åker
    }
    out = unified.copy()
    has_parcel = lpis_mask > 0
    if has_parcel.any():
        for code, cls in SJV_TO_UNIFIED.items():
            m = lpis_mask == code
            if m.any():
                out[m] = cls
        # unmapped codes → övrig åker
        mapped_codes = set(SJV_TO_UNIFIED.keys())
        unmapped = has_parcel & ~np.isin(lpis_mask, list(mapped_codes))
        out[unmapped] = 21
    return out


def make_comparison(images: list[tuple[np.ndarray, str]], out_path: str) -> None:
    """Save a horizontal strip of labelled images."""
    from PIL import Image, ImageDraw
    cell_h = images[0][0].shape[0] * 2 + 26
    cell_w = images[0][0].shape[1] * 2
    total_w = cell_w * len(images)
    strip = Image.new("RGB", (total_w, cell_h), (20, 20, 30))
    for i, (arr, title) in enumerate(images):
        img = Image.fromarray(arr, "RGB").resize((cell_w, cell_h - 26), Image.NEAREST)
        strip.paste(img, (i * cell_w, 26))
        draw = ImageDraw.Draw(strip)
        draw.rectangle([i * cell_w, 0, (i + 1) * cell_w - 1, 25], fill=(20, 20, 30))
        draw.text((i * cell_w + 6, 4), title, fill=(220, 220, 220))
    strip.save(out_path)
    print(f"  Comparison: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", default="/data/unified_v2/tile_491280_6861280.npz")
    ap.add_argument("--nmd",  default="/data/nmd/nmd2018bas_ogeneraliserad_v1_1.tif")
    ap.add_argument("--lpis", default="/data/lpis/jordbruksskiften_2022.parquet")
    ap.add_argument("--sks",  default="/data/sks/utforda_avverkningar.parquet")
    ap.add_argument("--out",  default="/tmp/inspect")
    ap.add_argument("--step", type=int, default=0,
                    help="0=all steps, 1=NMD only, 2=+LPIS, 3=+SKS, 4=+comparison")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ── Load tile ────────────────────────────────────────────────────────────
    print(f"\nLoading tile: {args.tile}")
    d = np.load(args.tile, allow_pickle=True)
    image = d["image"]          # (T*6, H, W) float32 — already normalised
    existing_label = d["label"] # existing label (may have bugs)
    tile_px = image.shape[-1]   # 256

    # Prefer stored bbox_3006 over filename-derived to avoid mismatches
    if "bbox_3006" in d:
        bbox = d["bbox_3006"].tolist() if hasattr(d["bbox_3006"], "tolist") else list(d["bbox_3006"])
        west, south, east, north = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        print(f"  Bbox from tile file: [{west}, {south}, {east}, {north}]")
    else:
        # Tile filename encodes CENTER easting/northing (not SW corner).
        # E.g. tile_391280_6151280 → center=(391280,6151280), half=1280m
        # → west=390000, south=6150000, east=392560, north=6152560
        stem = Path(args.tile).stem
        parts = stem.split("_")
        center_e, center_n = int(parts[-2]), int(parts[-1])
        half = (tile_px * 10) // 2   # 256*10//2 = 1280 m
        west  = center_e - half
        east  = center_e + half
        south = center_n - half
        north = center_n + half
        print(f"  Bbox from filename (center-based): [{west}, {south}, {east}, {north}]")
    print(f"  Bbox (EPSG:3006 E,N): [{west}, {south}, {east}, {north}]")
    print(f"  All tile keys: {sorted(d.files)}")

    # Detect tile year from dates
    dates = d.get("dates", [])
    doy_arr = np.asarray(d.get("doy", [])).flatten()
    if len(dates) > 0:
        years = [int(str(dt)[:4]) for dt in dates if str(dt)[:4].isdigit()]
        tile_year = max(years) if years else 2022
    else:
        tile_year = 2022
    print(f"  Tile year: {tile_year}")
    print(f"  Dates: {list(dates)}")
    print(f"  DOY:   {list(doy_arr)}")

    # ── Step 1: Satellite RGB ────────────────────────────────────────────────
    print("\n[Step 1] Satellite RGB")
    sat_rgb, best_f = spectral_to_rgb(image)
    sat_nir, nir_f = spectral_to_nir(image, doy=doy_arr)
    print(f"  Best RGB frame:  {best_f}  (DOY {doy_arr[best_f] if best_f < len(doy_arr) else '?'})")
    print(f"  Best NIR frame:  {nir_f}   (DOY {doy_arr[nir_f] if nir_f < len(doy_arr) else '?'})")
    save_png(sat_rgb, f"{args.out}/01_satellite.png",    f"Satellite RGB — frame {best_f}")
    save_png(sat_nir, f"{args.out}/01_satellite_nir.png", f"NIR false-color — frame {nir_f} DOY={doy_arr[nir_f] if nir_f < len(doy_arr) else '?'}")

    if args.step == 1:
        return

    # ── Step 2: NMD raw ───────────────────────────────────────────────────────
    print("\n[Step 2] NMD raw")
    # Use stored nmd_label_raw from tile (aligned with spectral) as primary
    if "nmd_label_raw" in d:
        nmd_raw_stored = np.asarray(d["nmd_label_raw"]).astype(np.uint8)
        print(f"  nmd_label_raw unique values: {sorted(np.unique(nmd_raw_stored).tolist())}")
        nmd_rgb_stored = label_to_rgb(nmd_raw_stored, NMD_SEQ19_COLORS)
        save_png(nmd_rgb_stored, f"{args.out}/02a_nmd_stored.png", "NMD stored (aligned with spectral)")
        nmd_raw = nmd_raw_stored
    else:
        print("  No nmd_label_raw in tile, extracting from TIF...")
        nmd_raw = extract_nmd(args.nmd, west, south, east, north, tile_px)
        nmd_rgb = label_to_rgb(nmd_raw, NMD_RAW_COLORS)
        save_png(nmd_rgb, f"{args.out}/02a_nmd_stored.png", "NMD from TIF")

    # ── Step 2b: Harvest / clearcut overlay ──────────────────────────────────
    print("\n[Step 2b] Harvest mask overlay")
    harvest_mask = np.asarray(d.get("harvest_mask", np.zeros((tile_px, tile_px), np.uint8)))
    harvest_prob = np.asarray(d.get("harvest_probability", np.zeros((tile_px, tile_px), np.float32)))
    n_harvest = int(harvest_mask.sum())
    print(f"  harvest_mask pixels: {n_harvest} ({n_harvest/tile_px**2*100:.1f}%)")
    print(f"  harvest_probability max: {harvest_prob.max():.3f}")

    # SKS rotation bug fixed in enrich_tiles_sks.py — masks now correct as stored.
    harvest_mask_display = harvest_mask
    harvest_prob_display = harvest_prob

    n_harvest_display = int((harvest_mask_display == 1).sum())
    rows_d, cols_d = np.where(harvest_mask_display == 1)
    if len(rows_d):
        print(f"  harvest_mask display (corrected) rows: {rows_d.min()}-{rows_d.max()}, "
              f"cols: {cols_d.min()}-{cols_d.max()}, centroid=({rows_d.mean():.1f},{cols_d.mean():.1f})")

    # Overlay: NIR as background, harvest pixels highlighted yellow (#FFE000)
    # harvest_probability shown as orange gradient where > 0
    harvest_vis = sat_nir.copy()
    # First draw harvest_probability as orange tint (p>0)
    if harvest_prob_display.max() > 0:
        orange = np.array([255, 140, 0], np.uint8)
        prob_mask = harvest_prob_display > 0.05
        alpha = np.clip(harvest_prob_display[prob_mask], 0, 1)
        for c, col in enumerate(orange):
            ch = harvest_vis[:, :, c].astype(np.float32)
            ch[prob_mask] = ch[prob_mask] * (1 - alpha * 0.6) + col * alpha * 0.6
            harvest_vis[:, :, c] = ch.clip(0, 255).astype(np.uint8)
    # Then draw harvest_mask=1 pixels as solid bright yellow
    if n_harvest_display > 0:
        harvest_vis[harvest_mask_display == 1] = [255, 230, 0]
    save_png(harvest_vis, f"{args.out}/02b_harvest.png",
             f"Harvest overlay (yellow=mask n={n_harvest_display}, orange=probability)")

    if args.step == 2:
        return

    # ── Step 3: NMD → unified (no LPIS/SKS yet) ──────────────────────────────
    print("\n[Step 3] NMD -> unified 23-class schema")
    # Use correct LUT based on what nmd_raw contains:
    # - stored nmd_label_raw: sequential indices 1-18 -> use _NMD19_TO_UNIFIED
    # - re-extracted from TIF: raw codes 2,41,111,etc -> use _NMD_RAW_TO_UNIFIED
    if "nmd_label_raw" in d:
        unified_nmd = _NMD_SEQ19_TO_UNIFIED[nmd_raw.clip(0, 19)]  # sequential 1-18
    else:
        unified_nmd = _NMD_RAW_TO_UNIFIED[nmd_raw.astype(np.uint8)]  # raw codes
    unified_rgb = label_to_rgb(unified_nmd, UNIFIED_COLORS)
    save_png(unified_rgb, f"{args.out}/03_unified_nmd_only.png", "Unified (NMD stored)")
    # Also show the existing stored label for direct comparison
    exist_rgb = label_to_rgb(existing_label.astype(np.uint8), UNIFIED_COLORS)
    save_png(exist_rgb, f"{args.out}/03b_existing_label.png", "Existing stored label")

    if args.step == 3:
        return

    # ── Step 4: + LPIS ───────────────────────────────────────────────────────
    print("\n[Step 4] + LPIS crop overlay")
    lpis_mask = extract_lpis(args.lpis, west, south, east, north, tile_px)
    unified_with_lpis = build_lpis_colored(lpis_mask, unified_nmd)
    lpis_rgb = label_to_rgb(unified_with_lpis, UNIFIED_COLORS)
    save_png(lpis_rgb, f"{args.out}/04_unified_nmd_lpis.png", "Unified + LPIS crops")

    if args.step == 4:
        return

    # ── Step 5: + SKS hygge ──────────────────────────────────────────────────
    print("\n[Step 5] + SKS hygge overlay (E,N, no rotation)")
    sks_mask = extract_sks(args.sks, west, south, east, north, tile_year, tile_px)
    unified_full = unified_with_lpis.copy()
    if sks_mask is not None and sks_mask.sum() > 0:
        unified_full[sks_mask > 0] = 22
    full_rgb = label_to_rgb(unified_full, UNIFIED_COLORS)
    save_png(full_rgb, f"{args.out}/05_unified_full.png", "Unified (NMD+LPIS+SKS)")

    # ── Step 6: Compare with existing label ──────────────────────────────────
    print("\n[Step 6] Comparison: satellite / NIR / NMD / +LPIS / +SKS / existing")
    existing_rgb = label_to_rgb(existing_label.astype(np.uint8), UNIFIED_COLORS)
    make_comparison([
        (sat_rgb,       "RGB (natural)"),
        (sat_nir,       "NIR false-color"),
        (nmd_rgb,       "NMD raw"),
        (unified_rgb,   "Unified (NMD)"),
        (lpis_rgb,      "Unified +LPIS"),
        (full_rgb,      "Unified +SKS"),
        (existing_rgb,  "EXISTING label"),
    ], f"{args.out}/06_comparison.png")

    print("\nDone. Files in:", args.out)


if __name__ == "__main__":
    main()
