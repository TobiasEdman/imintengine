"""
imint/exporters/export.py — Output file helpers

Functions for saving analysis results as PNG, GeoTIFF, GeoJSON, and JSON.
Each function is self-contained and can be called independently.
"""
from __future__ import annotations

import os
import json
import cv2
import numpy as np
from PIL import Image
from datetime import datetime


def save_rgb_png(rgb: np.ndarray, path: str) -> str:
    """Save an (H, W, 3) float32 [0,1] RGB array as a PNG file."""
    img = (rgb * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)
    print(f"    saved: {path}")
    return path


def save_change_overlay(rgb: np.ndarray, mask: np.ndarray, path: str) -> str:
    """Overlay detected changes in red on the RGB image and save as PNG."""
    overlay = rgb.copy()
    overlay[mask, 0] = 1.0
    overlay[mask, 1] *= 0.3
    overlay[mask, 2] *= 0.3
    img = (overlay * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)
    print(f"    saved: {path}")
    return path


def save_ndvi_colormap(ndvi: np.ndarray, path: str) -> str:
    """Apply a green-yellow-brown colormap to NDVI and save as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_title("NDVI")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved: {path}")
    return path


def save_regions_geojson(
    regions: list[dict],
    path: str,
    geo=None,
    coords: dict | None = None,
    image_shape: tuple | None = None,
) -> str:
    """Convert pixel-space bounding boxes to GeoJSON polygons.

    GeoJSON requires WGS84 (EPSG:4326) coordinates per RFC 7946.

    If a GeoContext is provided, pixel coords are first transformed to
    the projected CRS (EPSG:3006) via the affine transform, then
    reprojected to WGS84. This gives precise geographic coordinates.

    Otherwise, if WGS84 coords and image_shape are provided, uses
    simple linear interpolation (legacy behaviour).

    Args:
        regions: List of dicts with ``"bbox"`` keys (pixel coordinates).
        path: Output GeoJSON path.
        geo: GeoContext with CRS + transform (preferred).
        coords: WGS84 bounding box (legacy fallback).
        image_shape: Image (H, W, ...) for legacy coord mapping.

    Returns:
        The output file path.
    """
    features = []
    for region in regions:
        bbox = region["bbox"]

        if geo is not None:
            # Pixel → projected CRS via affine transform, then → WGS84
            lon_min, lat_min, lon_max, lat_max = _pixel_bbox_to_wgs84(bbox, geo)
        elif coords and image_shape:
            # Legacy: simple linear interpolation in WGS84
            h, w = image_shape[:2]
            west, south = coords["west"], coords["south"]
            east, north = coords["east"], coords["north"]
            lon_min = west + (bbox["x_min"] / w) * (east - west)
            lon_max = west + (bbox["x_max"] / w) * (east - west)
            lat_min = north - (bbox["y_max"] / h) * (north - south)
            lat_max = north - (bbox["y_min"] / h) * (north - south)
        else:
            lon_min, lon_max = bbox["x_min"], bbox["x_max"]
            lat_min, lat_max = bbox["y_min"], bbox["y_max"]

        polygon = [
            [lon_min, lat_min],
            [lon_max, lat_min],
            [lon_max, lat_max],
            [lon_min, lat_max],
            [lon_min, lat_min],
        ]
        properties = {k: v for k, v in region.items() if k != "bbox"}
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [polygon]},
            "properties": properties,
        })

    geojson = {"type": "FeatureCollection", "features": features}
    with open(path, "w") as f:
        json.dump(geojson, f, indent=2, default=_json_default)
    print(f"    saved: {path}")
    return path


def _pixel_bbox_to_wgs84(bbox: dict, geo) -> tuple[float, float, float, float]:
    """Convert a pixel bounding box to WGS84 coordinates via GeoContext.

    Steps:
        1. Pixel (col, row) → projected CRS via affine transform
        2. Projected CRS → WGS84 via rasterio.warp.transform

    Args:
        bbox: Dict with x_min, y_min, x_max, y_max in pixel coordinates.
        geo: GeoContext with CRS and affine transform.

    Returns:
        (lon_min, lat_min, lon_max, lat_max) in WGS84.
    """
    from rasterio.crs import CRS
    from rasterio.warp import transform

    # Pixel corners → projected coordinates using affine transform
    # transform * (col, row) → (x, y) in projected CRS
    x_min, y_min = geo.transform * (bbox["x_min"], bbox["y_max"])  # bottom-left
    x_max, y_max = geo.transform * (bbox["x_max"], bbox["y_min"])  # top-right

    # Reproject to WGS84
    src_crs = CRS.from_user_input(geo.crs)
    dst_crs = CRS.from_epsg(4326)

    xs = [x_min, x_max]
    ys = [y_min, y_max]
    lons, lats = transform(src_crs, dst_crs, xs, ys)

    return lons[0], lats[0], lons[1], lats[1]


def save_geotiff(
    array: np.ndarray,
    path: str,
    geo=None,
    coords: dict | None = None,
) -> str:
    """Save a 2D array as a GeoTIFF with spatial reference.

    Uses rasterio if available, falls back to plain TIFF via PIL.

    If a GeoContext is provided, its CRS and affine transform are used
    directly (correct EPSG:3006 output). Otherwise falls back to building
    a transform from WGS84 coords (legacy behaviour).

    Args:
        array: 2D or 3D numpy array.
        path: Output file path.
        geo: GeoContext with CRS + transform (preferred).
        coords: WGS84 bounding box (legacy fallback).

    Returns:
        The output file path.
    """
    try:
        import rasterio
        from rasterio.transform import from_bounds

        h, w = array.shape[:2]

        if geo is not None:
            # Use GeoContext — correct EPSG:3006 CRS and affine transform
            crs = geo.crs
            transform = geo.transform
        elif coords:
            crs = "EPSG:4326"
            transform = from_bounds(
                coords["west"], coords["south"],
                coords["east"], coords["north"],
                w, h,
            )
        else:
            crs = "EPSG:4326"
            transform = from_bounds(0, 0, w, h, w, h)

        count = 1 if array.ndim == 2 else array.shape[2]
        with rasterio.open(
            path, "w", driver="GTiff",
            height=h, width=w, count=count,
            dtype=array.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            if array.ndim == 2:
                dst.write(array, 1)
            else:
                for i in range(count):
                    dst.write(array[:, :, i], i + 1)

    except ImportError:
        if array.ndim == 2:
            img = Image.fromarray(array)
        else:
            img = Image.fromarray(array.astype(np.uint8))
        img.save(path)

    print(f"    saved: {path}")
    return path


def save_summary_report(
    results: list,
    date: str | None,
    output_dir: str,
) -> str:
    """Write a JSON summary of all analyzer results. Returns file path."""
    prefix = f"{date}_" if date else ""
    path = os.path.join(output_dir, f"{prefix}imint_summary.json")

    summary = {
        "date": date,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "analyzers": [],
    }

    for r in results:
        entry = {
            "name": r.analyzer,
            "success": r.success,
            "error": r.error,
            "metadata": r.metadata,
        }
        if r.outputs:
            scalar_outputs = {}
            for k, v in r.outputs.items():
                if isinstance(v, (int, float, str, bool, list)):
                    scalar_outputs[k] = v
                elif isinstance(v, dict):
                    filtered = {
                        kk: vv for kk, vv in v.items()
                        if not isinstance(vv, np.ndarray)
                    }
                    if filtered:
                        scalar_outputs[k] = filtered
            entry["outputs"] = scalar_outputs
        summary["analyzers"].append(entry)

    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    return path


def save_nmd_overlay(l2_raster: np.ndarray, path: str) -> str:
    """Render NMD land cover classes as a color-coded PNG overlay.

    Handles both raw NMD codes (e.g. 111, 112, 3, 42, 61, …) and
    pre-remapped sequential codes (0-19).  If max value > 19 the raw
    NMD coding system is assumed and mapped to display colours.

    Args:
        l2_raster: 2D uint8/uint16 array with NMD class codes.
        path: Output PNG path.

    Returns:
        The output file path.
    """
    # Raw NMD Generell (10 m) code → display colour (R, G, B)
    _RAW_NMD_COLORS = {
        # Skog utanför våtmark
        111: (0, 100, 0),       # Tallskog
        112: (34, 139, 34),     # Granskog
        113: (20, 120, 20),     # Tall/gran-blandskog
        114: (60, 179, 113),    # Barrblandskog
        115: (50, 205, 50),     # Triviallövskog
        116: (80, 180, 60),     # Ädellövskog
        117: (100, 200, 80),    # Trivial/ädellövblandskog
        118: (70, 160, 100),    # Barr/lövblandskog
        # Skog på våtmark
        121: (46, 79, 46),      # Tallskog våtmark
        122: (58, 95, 58),      # Granskog våtmark
        123: (50, 85, 50),      # Tall/gran våtmark
        124: (90, 143, 90),     # Barrblandskog våtmark
        125: (74, 127, 74),     # Triviallövskog våtmark
        126: (85, 140, 80),     # Ädellövskog våtmark
        127: (95, 155, 90),     # Lövblandskog våtmark
        128: (80, 130, 80),     # Barr/löv våtmark
        # Öppen mark
        2:   (139, 90, 43),     # Öppen våtmark
        3:   (255, 215, 0),     # Åkermark
        41:  (200, 173, 127),   # Övrig öppen mark utan vegetation
        42:  (210, 180, 140),   # Övrig öppen mark med vegetation
        # Bebyggelse
        51:  (255, 0, 0),       # Exploaterad mark, byggnad
        52:  (255, 69, 0),      # Exploaterad mark, ej byggnad/väg
        53:  (255, 99, 71),     # Exploaterad mark, övrig hårdgjord
        # Vatten
        61:  (0, 0, 255),       # Sjö och vattendrag
        62:  (30, 144, 255),    # Hav
    }

    # Sequential L2 palette (for pre-remapped rasters, indices 0-19)
    _SEQ_PALETTE = np.array([
        [128, 128, 128],  # 0: unclassified
        [0, 100, 0],      # 1: forest_pine
        [34, 139, 34],    # 2: forest_spruce
        [50, 205, 50],    # 3: forest_deciduous
        [60, 179, 113],   # 4: forest_mixed
        [144, 238, 144],  # 5: forest_temp_non_forest
        [46, 79, 46],     # 6: forest_wetland_pine
        [58, 95, 58],     # 7: forest_wetland_spruce
        [74, 127, 74],    # 8: forest_wetland_deciduous
        [90, 143, 90],    # 9: forest_wetland_mixed
        [122, 175, 122],  # 10: forest_wetland_temp
        [139, 90, 43],    # 11: open_wetland
        [255, 215, 0],    # 12: cropland
        [200, 173, 127],  # 13: open_land_bare
        [210, 180, 140],  # 14: open_land_vegetated
        [255, 0, 0],      # 15: developed_buildings
        [255, 69, 0],     # 16: developed_infrastructure
        [255, 99, 71],    # 17: developed_roads
        [0, 0, 255],      # 18: water_lakes
        [30, 144, 255],   # 19: water_sea
    ], dtype=np.uint8)

    max_val = int(l2_raster.max())
    if max_val > 19:
        # Raw NMD codes — map via lookup
        h, w = l2_raster.shape[:2]
        rgb = np.full((h, w, 3), 128, dtype=np.uint8)  # default gray
        for code, color in _RAW_NMD_COLORS.items():
            mask = l2_raster == code
            if mask.any():
                rgb[mask] = color
    else:
        # Pre-remapped sequential codes
        clamped = np.clip(l2_raster, 0, len(_SEQ_PALETTE) - 1)
        rgb = _SEQ_PALETTE[clamped]

    Image.fromarray(rgb).save(path)
    print(f"    saved: {path}")
    return path


# ── NMD Level 1 & 2 palettes and Swedish labels ────────────────────────────

_NMD_L1_PALETTE = {
    0: {"color": "#808080", "rgb": (128, 128, 128), "name": "Oklassificerat"},
    1: {"color": "#228B22", "rgb": (34, 139, 34),   "name": "Skog"},
    2: {"color": "#8B5A2B", "rgb": (139, 90, 43),   "name": "Våtmark"},
    3: {"color": "#FFD700", "rgb": (255, 215, 0),    "name": "Åkermark"},
    4: {"color": "#D2B48C", "rgb": (210, 180, 140),  "name": "Öppen mark"},
    5: {"color": "#FF0000", "rgb": (255, 0, 0),      "name": "Bebyggelse"},
    6: {"color": "#0000FF", "rgb": (0, 0, 255),      "name": "Vatten"},
}

_NMD_L2_PALETTE = {
    "forest_pine":              {"color": "#006400", "name": "Tallskog"},
    "forest_spruce":            {"color": "#228B22", "name": "Granskog"},
    "forest_deciduous":         {"color": "#32CD32", "name": "Lövskog"},
    "forest_mixed":             {"color": "#3CB371", "name": "Blandskog"},
    "forest_temp_non_forest":   {"color": "#90EE90", "name": "Temporärt ej skog"},
    "forest_wetland_pine":      {"color": "#2E4F2E", "name": "Sumpskog tall"},
    "forest_wetland_spruce":    {"color": "#3A5F3A", "name": "Sumpskog gran"},
    "forest_wetland_deciduous": {"color": "#4A7F4A", "name": "Sumpskog löv"},
    "forest_wetland_mixed":     {"color": "#5A8F5A", "name": "Sumpskog bland"},
    "forest_wetland_temp":      {"color": "#7AAF7A", "name": "Sumpskog temp"},
    "open_wetland":             {"color": "#8B5A2B", "name": "Öppen våtmark"},
    "cropland":                 {"color": "#FFD700", "name": "Åkermark"},
    "open_land_bare":           {"color": "#C8AD7F", "name": "Öppen mark, bar"},
    "open_land_vegetated":      {"color": "#D2B48C", "name": "Öppen mark, veg."},
    "developed_buildings":      {"color": "#FF0000", "name": "Byggnader"},
    "developed_infrastructure": {"color": "#FF4500", "name": "Infrastruktur"},
    "developed_roads":          {"color": "#FF6347", "name": "Vägar"},
    "water_lakes":              {"color": "#0000FF", "name": "Sjöar"},
    "water_sea":                {"color": "#1E90FF", "name": "Hav"},
}


def save_nmd_visualization(
    l2_raster: np.ndarray,
    rgb: np.ndarray,
    class_stats: dict,
    cross_ref: dict | None,
    path: str,
) -> str:
    """Create a comprehensive NMD land cover visualization.

    Generates a multi-panel figure:
        Panel 1: RGB satellite image
        Panel 2: NMD Level 2 land cover map with legend
        Panel 3: Land cover distribution (horizontal bar chart, Level 2)
        Panel 4: Cross-reference — mean NDVI/NDWI per LULC class (Level 2)

    Args:
        l2_raster: 2D uint8 array with Level 2 class codes (0-19).
        rgb: (H, W, 3) float32 [0,1] RGB reference image.
        class_stats: Dict with "level1" and "level2" sub-dicts from NMDAnalyzer.
        cross_ref: Cross-reference dict (spectral, change, object per class).
        path: Output PNG path.

    Returns:
        The output file path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec

    has_spectral = cross_ref and "spectral" in cross_ref

    # Layout: 2 rows x 2 columns
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.25)

    # ── Panel 1: RGB ─────────────────────────────────────────────────────
    ax_rgb = fig.add_subplot(gs[0, 0])
    ax_rgb.imshow(rgb.clip(0, 1))
    ax_rgb.set_title("Sentinel-2 RGB", fontsize=13, fontweight="bold")
    ax_rgb.axis("off")

    # ── Panel 2: NMD Level 2 map with legend ─────────────────────────────
    ax_nmd = fig.add_subplot(gs[0, 1])

    # Build L2 palette array: index 0 = unclassified, 1-19 = L2 classes
    l2_palette_list = [(128, 128, 128)]  # 0: unclassified
    l2_keys = list(_NMD_L2_PALETTE.keys())
    for key in l2_keys:
        hex_c = _NMD_L2_PALETTE[key]["color"]
        r, g, b = int(hex_c[1:3], 16), int(hex_c[3:5], 16), int(hex_c[5:7], 16)
        l2_palette_list.append((r, g, b))
    palette_arr = np.array(l2_palette_list, dtype=np.uint8)

    clamped = np.clip(l2_raster, 0, len(palette_arr) - 1)
    nmd_rgb = palette_arr[clamped]

    ax_nmd.imshow(nmd_rgb)
    ax_nmd.set_title("NMD Markt\u00e4cke (Niv\u00e5 2)", fontsize=13, fontweight="bold")
    ax_nmd.axis("off")

    # Legend with L2 class names and fractions
    l2_stats = class_stats.get("level2", {})
    legend_patches = []
    for key in l2_keys:
        info = _NMD_L2_PALETTE[key]
        stats = l2_stats.get(key)
        if stats and stats["fraction"] > 0.001:
            pct = stats["fraction"] * 100
            label = f"{info['name']} ({pct:.1f}%)"
        elif stats:
            label = f"{info['name']} (<0.1%)"
        else:
            continue
        legend_patches.append(mpatches.Patch(color=info["color"], label=label))

    if legend_patches:
        ax_nmd.legend(
            handles=legend_patches, loc="lower left", fontsize=6,
            framealpha=0.9, edgecolor="gray", ncol=2,
        )

    # ── Panel 3: Level 2 bar chart ───────────────────────────────────────
    ax_bars = fig.add_subplot(gs[1, 0])

    # Sort by fraction descending, skip tiny classes
    l2_sorted = sorted(
        [(k, v) for k, v in l2_stats.items() if v["fraction"] > 0.001],
        key=lambda x: x[1]["fraction"],
        reverse=True,
    )

    if l2_sorted:
        names_l2 = []
        fracs_l2 = []
        colors_l2 = []
        for key, stats in l2_sorted:
            info = _NMD_L2_PALETTE.get(key, {"color": "#808080", "name": key})
            names_l2.append(info["name"])
            fracs_l2.append(stats["fraction"] * 100)
            colors_l2.append(info["color"])

        y_pos = range(len(names_l2))
        bars = ax_bars.barh(y_pos, fracs_l2, color=colors_l2,
                            edgecolor="white", linewidth=0.5)
        ax_bars.set_yticks(y_pos)
        ax_bars.set_yticklabels(names_l2, fontsize=9)
        ax_bars.invert_yaxis()
        ax_bars.set_xlabel("Andel (%)", fontsize=10)
        ax_bars.set_title("Markt\u00e4cke \u2014 detaljerade klasser (Niv\u00e5 2)",
                          fontsize=13, fontweight="bold")

        # Value labels on bars
        for bar, frac in zip(bars, fracs_l2):
            if frac > 2:
                ax_bars.text(
                    bar.get_width() - 0.3,
                    bar.get_y() + bar.get_height() / 2,
                    f"{frac:.1f}%", ha="right", va="center", fontsize=8,
                    fontweight="bold", color="white",
                )
            else:
                ax_bars.text(
                    bar.get_width() + 0.3,
                    bar.get_y() + bar.get_height() / 2,
                    f"{frac:.1f}%", ha="left", va="center", fontsize=8,
                    color="black",
                )
        ax_bars.set_xlim(0, max(fracs_l2) * 1.15)
    else:
        ax_bars.text(0.5, 0.5, "Ingen data", ha="center", va="center",
                     fontsize=14)
        ax_bars.axis("off")

    # ── Panel 4: Cross-reference — NDVI/NDWI per LULC class (L2) ────────
    ax_xref = fig.add_subplot(gs[1, 1])

    if has_spectral:
        spectral_xref = cross_ref["spectral"]
        xref_classes = []
        ndvi_vals = []
        ndwi_vals = []

        for key in l2_keys:
            if key in spectral_xref:
                info = _NMD_L2_PALETTE[key]
                xref_classes.append(info["name"])
                ndvi_vals.append(spectral_xref[key].get("mean_ndvi", 0))
                ndwi_vals.append(spectral_xref[key].get("mean_ndwi", 0))

        if xref_classes:
            x = np.arange(len(xref_classes))
            width = 0.35
            ax_xref.bar(x - width / 2, ndvi_vals, width,
                        label="Medel-NDVI", color="#4CAF50", alpha=0.85)
            ax_xref.bar(x + width / 2, ndwi_vals, width,
                        label="Medel-NDWI", color="#2196F3", alpha=0.85)

            ax_xref.set_xticks(x)
            ax_xref.set_xticklabels(xref_classes, fontsize=7,
                                     rotation=45, ha="right")
            ax_xref.set_ylabel("Indexv\u00e4rde", fontsize=10)
            ax_xref.set_title("Spektral korsreferens per markt\u00e4cke (Niv\u00e5 2)",
                              fontsize=13, fontweight="bold")
            ax_xref.legend(fontsize=9, loc="upper right")
            ax_xref.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
            ax_xref.set_ylim(
                min(min(ndwi_vals), -0.6) - 0.1,
                max(max(ndvi_vals), 0.7) + 0.1,
            )

            # Value labels
            for i, (ndvi, ndwi) in enumerate(zip(ndvi_vals, ndwi_vals)):
                ax_xref.text(i - width / 2, ndvi + 0.02, f"{ndvi:.2f}",
                             ha="center", va="bottom", fontsize=6)
                y_ndwi = ndwi - 0.02 if ndwi < 0 else ndwi + 0.02
                va_ndwi = "top" if ndwi < 0 else "bottom"
                ax_xref.text(i + width / 2, y_ndwi, f"{ndwi:.2f}",
                             ha="center", va=va_ndwi, fontsize=6)
        else:
            ax_xref.text(0.5, 0.5, "Ingen spektraldata",
                         ha="center", va="center", fontsize=14)
            ax_xref.axis("off")
    else:
        ax_xref.text(0.5, 0.5, "Ingen korsreferensdata",
                     ha="center", va="center", fontsize=14)
        ax_xref.axis("off")

    fig.suptitle("NMD Markt\u00e4ckeanalys \u2014 Nationellt Markt\u00e4ckedata",
                 fontsize=15, fontweight="bold", y=0.98)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved: {path}")
    return path


def save_prithvi_overlay(
    seg_mask: np.ndarray,
    path: str,
    rgb: np.ndarray | None = None,
    class_names: dict | None = None,
) -> str:
    """Render Prithvi segmentation mask as a color-coded PNG.

    Optionally shows an RGB reference image side-by-side and a legend
    with class names instead of a plain colorbar.

    Args:
        seg_mask: 2D uint8 array with class indices.
        path: Output PNG path.
        rgb: Optional (H, W, 3) float32 [0,1] RGB reference image.
            If provided, renders a 2-panel figure (RGB + segmentation).
        class_names: Optional dict mapping class index to label string.
            E.g. {0: "no_water", 1: "water/flood"}.

    Returns:
        The output file path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    n_classes = int(seg_mask.max()) + 1
    cmap = plt.cm.get_cmap("tab20", max(n_classes, 2))

    if rgb is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(rgb.clip(0, 1))
        axes[0].set_title("RGB")
        axes[0].axis("off")
        ax_seg = axes[1]
    else:
        fig, ax_seg = plt.subplots(1, 1, figsize=(8, 8))

    im = ax_seg.imshow(seg_mask, cmap=cmap, vmin=0, vmax=max(n_classes - 1, 1))
    ax_seg.set_title("Prithvi Segmentation")
    ax_seg.axis("off")

    if class_names:
        patches = []
        for cls_idx in sorted(class_names.keys()):
            color = cmap(cls_idx / max(n_classes - 1, 1))
            label = f"{cls_idx}: {class_names[cls_idx]}"
            patches.append(mpatches.Patch(color=color, label=label))
        ax_seg.legend(handles=patches, loc="lower right", fontsize=9)
    else:
        plt.colorbar(im, ax=ax_seg, fraction=0.046)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved: {path}")
    return path


def save_nmd_stats(
    class_stats: dict,
    cross_ref: dict | None,
    path: str,
) -> str:
    """Save NMD class statistics and cross-reference data as JSON.

    Args:
        class_stats: Dict with "level1" and "level2" sub-dicts.
        cross_ref: Cross-reference dict (spectral, change, object per class).
        path: Output JSON path.

    Returns:
        The output file path.
    """
    data = {
        "class_stats": class_stats,
        "cross_reference": cross_ref or {},
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)
    print(f"    saved: {path}")
    return path


def _pca_feature_map(features_2d: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    """Compute PCA RGB map from embedding features.

    Reduces high-dimensional features to 3 principal components and
    normalizes each to [0, 1] for RGB display.

    Args:
        features_2d: (N_positions, C) feature matrix (e.g. 49 x 1024).
        grid_h: Spatial grid height (e.g. 7).
        grid_w: Spatial grid width (e.g. 7).

    Returns:
        (grid_h, grid_w, 3) float32 array in [0, 1] for RGB display.
    """
    centered = features_2d - features_2d.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    # U columns are principal component scores: (N, min(N,C))
    # Take first 3 PCs and reshape to spatial grid
    n_pcs = min(3, U.shape[1])
    pca_scores = U[:, :n_pcs].reshape(grid_h, grid_w, n_pcs)

    # Pad to 3 channels if fewer than 3 PCs available
    if n_pcs < 3:
        pad = np.full((grid_h, grid_w, 3 - n_pcs), 0.5, dtype=np.float32)
        pca_scores = np.concatenate([pca_scores, pad], axis=-1)

    # Normalize each channel to [0, 1]
    for ch in range(3):
        lo, hi = pca_scores[:, :, ch].min(), pca_scores[:, :, ch].max()
        if hi - lo > 1e-8:
            pca_scores[:, :, ch] = (pca_scores[:, :, ch] - lo) / (hi - lo)
        else:
            pca_scores[:, :, ch] = 0.5

    return pca_scores.astype(np.float32)


def _activation_magnitude(embedding_3d: np.ndarray) -> np.ndarray:
    """Compute per-position L2-norm activation magnitude.

    Args:
        embedding_3d: (C, grid_h, grid_w) feature tensor.

    Returns:
        (grid_h, grid_w) float32 array of L2 norms.
    """
    return np.linalg.norm(embedding_3d, axis=0).astype(np.float32)


def save_prithvi_embedding_viz(
    embedding: np.ndarray,
    rgb: np.ndarray,
    path: str,
) -> str:
    """Visualize Prithvi-EO-2.0 embedding as a 3-panel figure.

    Panels:
        1. RGB reference image (spatial context)
        2. PCA feature map (top 3 PCs mapped to RGB, upscaled)
        3. Activation magnitude heatmap (L2-norm, inferno colormap)

    The 7x7 spatial grid from the ViT encoder is upscaled to match
    the RGB image dimensions using bilinear interpolation.

    Args:
        embedding: (1, C, H_grid, W_grid) float32 embedding tensor,
                   e.g. (1, 1024, 7, 7) from Prithvi-EO-2.0-300M.
        rgb: (H, W, 3) float32 [0,1] RGB reference image.
        path: Output PNG file path.

    Returns:
        The output file path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom

    # Squeeze batch dimension: (1, 1024, 7, 7) -> (1024, 7, 7)
    emb = embedding.squeeze(0)
    c, grid_h, grid_w = emb.shape
    img_h, img_w = rgb.shape[:2]

    # --- PCA feature map ---
    flat = emb.reshape(c, -1).T  # (N_positions, C), e.g. (49, 1024)
    pca_map = _pca_feature_map(flat, grid_h, grid_w)  # (7, 7, 3)

    # Upscale PCA map to image size with bilinear interpolation
    zoom_h = img_h / grid_h
    zoom_w = img_w / grid_w
    pca_up = np.stack([
        zoom(pca_map[:, :, ch], (zoom_h, zoom_w), order=1)
        for ch in range(3)
    ], axis=-1).clip(0, 1)

    # --- Activation magnitude heatmap ---
    magnitude = _activation_magnitude(emb)  # (7, 7)
    mag_up = zoom(magnitude, (zoom_h, zoom_w), order=1)

    # --- Build 3-panel figure ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(rgb.clip(0, 1))
    axes[0].set_title("RGB")
    axes[0].axis("off")

    axes[1].imshow(pca_up)
    axes[1].set_title("PCA Feature Map (PC1-3)")
    axes[1].axis("off")

    im = axes[2].imshow(mag_up, cmap="inferno")
    axes[2].set_title("Activation Magnitude (L2)")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    fig.suptitle("Prithvi-EO-2.0 Embedding", fontsize=14)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved: {path}")
    return path


def save_ndvi_clean_png(ndvi: np.ndarray, path: str) -> str:
    """Save NDVI as a clean color-mapped PNG (no axes/borders) for Leaflet overlay.

    Uses the RdYlGn colormap, mapping [-1, 1] → [0, 1].

    Args:
        ndvi: (H, W) float32 NDVI array in [-1, 1].
        path: Output PNG path.

    Returns:
        The output file path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm

    norm = ((ndvi + 1.0) / 2.0).clip(0, 1)
    cmap = cm.get_cmap("RdYlGn")
    rgba = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(rgba).save(path)
    print(f"    saved: {path}")
    return path


def save_spectral_index_clean_png(
    index_arr: np.ndarray,
    path: str,
    cmap_name: str = "RdYlGn",
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> str:
    """Save a spectral index as a clean color-mapped PNG for Leaflet overlay.

    Args:
        index_arr: (H, W) float32 array.
        path: Output PNG path.
        cmap_name: Matplotlib colormap name.
        vmin: Minimum value for normalization.
        vmax: Maximum value for normalization.

    Returns:
        The output file path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm

    norm = ((index_arr - vmin) / (vmax - vmin + 1e-10)).clip(0, 1)
    cmap = cm.get_cmap(cmap_name)
    rgba = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(rgba).save(path)
    print(f"    saved: {path}")
    return path


def save_prithvi_seg_clean_png(
    seg_mask: np.ndarray,
    path: str,
    class_colors: dict[int, tuple] | None = None,
) -> str:
    """Save Prithvi segmentation as a clean color-coded PNG for Leaflet overlay.

    Args:
        seg_mask: (H, W) uint8 segmentation mask.
        path: Output PNG path.
        class_colors: Optional mapping of class index → (R, G, B) tuple.
            Defaults to burn_scars palette: 0=green, 1=red-orange.

    Returns:
        The output file path.
    """
    if class_colors is None:
        class_colors = {
            0: (34, 139, 34),    # no_burn: green
            1: (255, 69, 0),     # burned: red-orange
        }
    max_class = max(class_colors.keys())
    palette = np.zeros((max_class + 1, 3), dtype=np.uint8)
    for cls_id, color in class_colors.items():
        palette[cls_id] = color
    clamped = np.clip(seg_mask, 0, max_class)
    rgb = palette[clamped]
    Image.fromarray(rgb).save(path)
    print(f"    saved: {path}")
    return path


def save_cot_clean_png(cot_map: np.ndarray, path: str) -> str:
    """Save COT heatmap as a clean PNG for Leaflet overlay.

    Uses the hot_r colormap with a fixed 0–0.05 range matching the DES
    MLP5 model output.  The model's thresholds are 0.015 (thin cloud) and
    0.025 (thick cloud), so 0–0.05 (= 2× thick threshold) gives good
    contrast across the clear → thin → thick transition.

    Args:
        cot_map: (H, W) float32 COT values.
        path: Output PNG path.

    Returns:
        The output file path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm

    COT_VIZ_MAX = 0.05  # 2× thick cloud threshold (0.025)
    norm = (cot_map / COT_VIZ_MAX).clip(0, 1)
    cmap = cm.get_cmap("hot_r")
    rgba = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(rgba).save(path)
    print(f"    saved: {path}")
    return path


def _cot_stretch_range(
    cot_map: np.ndarray,
) -> tuple[float, float]:
    """Compute display range for COT visualization.

    Uses a fixed 0–0.05 range matching the DES MLP5 model output.  The
    model's classification thresholds are 0.015 (thin) and 0.025 (thick),
    so 0–0.05 provides good contrast for cloud-screened scenes where most
    values cluster between 0.003 and 0.025.  P2/P98 percentile stretching
    amplified noise on clean scenes and produced unstable results.

    Returns:
        (vmin, vmax) for color mapping.
    """
    return 0.0, 0.05


def save_cloud_class_clean_png(cloud_class: np.ndarray, path: str) -> str:
    """Save cloud classification as a clean color-coded PNG for Leaflet overlay.

    Palette: 0=clear (blue), 1=thin cloud (amber), 2=thick cloud (red).

    Args:
        cloud_class: (H, W) uint8 cloud classification (0-2).
        path: Output PNG path.

    Returns:
        The output file path.
    """
    palette = np.array([
        [33, 150, 243],   # 0: clear (blue)
        [255, 193, 7],    # 1: thin cloud (amber)
        [244, 67, 54],    # 2: thick cloud (red)
    ], dtype=np.uint8)
    clamped = np.clip(cloud_class, 0, 2)
    rgb = palette[clamped]
    Image.fromarray(rgb).save(path)
    print(f"    saved: {path}")
    return path


def save_change_gradient_png(diff: np.ndarray, path: str) -> str:
    """Save change magnitude as a gradient heatmap PNG for Leaflet overlay.

    Low change → transparent/blue, high change → red.
    Uses the 'hot' colormap normalized to the data range.

    Args:
        diff: (H, W) float32 L2-norm difference between current and baseline.
        path: Output PNG path.

    Returns:
        The output file path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm

    vmax = float(np.percentile(diff[diff > 0], 98)) if (diff > 0).any() else 1.0
    vmax = max(vmax, 1e-6)
    norm = (diff / vmax).clip(0, 1)
    cmap = cm.get_cmap("hot_r")
    rgba = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(rgba).save(path)
    print(f"    saved: {path}")
    return path


def save_dnbr_clean_png(dnbr: np.ndarray, path: str) -> str:
    """Save dNBR as a clean PNG with discrete USGS burn severity classes.

    USGS dNBR classification:
        < -0.25  : High regrowth  (#1a9850)
        -0.25–-0.1: Low regrowth  (#91cf60)
        -0.1–0.1 : Unburned       (#d9ef8b)
        0.1–0.27 : Low severity   (#fee08b)
        0.27–0.44: Moderate-low   (#fdae61)
        0.44–0.66: Moderate-high  (#f46d43)
        > 0.66   : High severity  (#d73027)

    Args:
        dnbr: (H, W) float32 dNBR values (NBR_pre - NBR_post).
        path: Output PNG path.

    Returns:
        The output file path.
    """
    # USGS thresholds and colors (RGB)
    thresholds = [-0.25, -0.1, 0.1, 0.27, 0.44, 0.66]
    colors = np.array([
        [26, 152, 80],    # High regrowth (dark green)
        [145, 207, 96],   # Low regrowth (light green)
        [217, 239, 139],  # Unburned (yellow-green)
        [254, 224, 139],  # Low severity (yellow)
        [253, 174, 97],   # Moderate-low (orange)
        [244, 109, 67],   # Moderate-high (red-orange)
        [215, 48, 39],    # High severity (red)
    ], dtype=np.uint8)

    # Classify each pixel
    class_idx = np.digitize(dnbr, thresholds)  # 0..6
    rgb = colors[class_idx]
    Image.fromarray(rgb).save(path)
    print(f"    saved: {path}")
    return path


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_vessel_overlay(
    rgb: np.ndarray, regions: list[dict], path: str,
) -> str:
    """Save RGB with vessel bounding boxes and count annotation.

    Draws cyan bounding boxes around detected vessels and adds a count
    label in the top-left corner.  Produces a clean PNG suitable for
    the HTML report Leaflet overlay.

    Args:
        rgb: (H, W, 3) float32 [0,1] or uint8 RGB image.
        regions: List of detection dicts with ``bbox`` keys.
        path: Output PNG path.

    Returns:
        The output file path.
    """
    if rgb.dtype != np.uint8:
        img = (rgb * 255).clip(0, 255).astype(np.uint8)
    else:
        img = rgb.copy()

    from PIL import ImageDraw, ImageFont

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    for r in regions:
        bb = r["bbox"]
        x0, y0, x1, y1 = bb["x_min"], bb["y_min"], bb["x_max"], bb["y_max"]
        draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 255), width=2)

    # Count label
    label = f"{len(regions)} vessels"
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([4, 4, 8 + tw, 8 + th], fill=(0, 0, 0, 180))
    draw.text((6, 6), label, fill=(0, 255, 255), font=font)

    pil_img.save(path)
    print(f"    saved: {path}")
    return path


def save_ai2_vessel_overlay(
    rgb: np.ndarray, regions: list[dict], path: str,
) -> str:
    """Save RGB with AI2 vessel detections, heading arrows, speed colours.

    Each vessel is drawn with:
      • Bounding box coloured by speed (blue → yellow → red)
      • Heading arrow from the centre of the box
      • Label showing vessel type + length

    Produces a clean PNG suitable for the HTML report Leaflet overlay.

    Args:
        rgb: (H, W, 3) float32 [0,1] or uint8 RGB image.
        regions: List of detection dicts, each with ``bbox`` and
            optional ``attributes`` dict (length_m, width_m,
            speed_knots, heading_deg, vessel_type, type_confidence).
        path: Output PNG path.

    Returns:
        The output file path.
    """
    import math
    from PIL import ImageDraw, ImageFont

    if rgb.dtype != np.uint8:
        img = (rgb * 255).clip(0, 255).astype(np.uint8)
    else:
        img = rgb.copy()

    pil_img = Image.fromarray(img).convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
        font_big = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = ImageFont.load_default()
        font_big = font

    def _speed_colour(speed_kn: float) -> tuple:
        """Map speed to colour: blue (0 kn) → yellow (5 kn) → red (15+ kn)."""
        if speed_kn <= 0:
            return (80, 160, 255)       # light blue - stationary
        if speed_kn < 5:
            t = speed_kn / 5.0
            r = int(80 + t * (255 - 80))
            g = int(160 + t * (230 - 160))
            b = int(255 - t * 255)
            return (r, g, b)
        if speed_kn < 15:
            t = (speed_kn - 5) / 10.0
            r = 255
            g = int(230 - t * 230)
            b = 0
            return (r, g, b)
        return (255, 0, 0)             # red - fast

    for r in regions:
        bb = r["bbox"]
        x0, y0, x1, y1 = bb["x_min"], bb["y_min"], bb["x_max"], bb["y_max"]
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        attrs = r.get("attributes", {})

        speed = attrs.get("speed_knots", 0)
        colour = _speed_colour(speed)
        colour_a = colour + (220,)  # with alpha

        # Bounding box
        draw.rectangle([x0, y0, x1, y1], outline=colour_a, width=2)

        # Heading arrow
        heading_deg = attrs.get("heading_deg")
        if heading_deg is not None:
            arrow_len = 18
            rad = math.radians(heading_deg)
            # Heading: 0° = East, 90° = North in math coords
            # Convert to image coords (y increases downward)
            dx = arrow_len * math.cos(rad)
            dy = -arrow_len * math.sin(rad)
            ex, ey = cx + dx, cy + dy
            draw.line([(cx, cy), (ex, ey)], fill=colour_a, width=2)
            # Arrow head
            head_len = 5
            head_angle = 0.4  # radians (~23°)
            for sign in (-1, 1):
                hx = ex - head_len * math.cos(rad + sign * head_angle)
                hy = ey + head_len * math.sin(rad + sign * head_angle)
                draw.line([(ex, ey), (hx, hy)], fill=colour_a, width=2)

        # Label: type + length
        vtype = attrs.get("vessel_type", "")
        length = attrs.get("length_m")
        parts = []
        if vtype:
            parts.append(vtype.capitalize())
        if length:
            parts.append(f"{length:.0f}m")
        label = " ".join(parts) if parts else f"{r.get('score', 0):.0%}"

        bbox_t = draw.textbbox((0, 0), label, font=font)
        tw = bbox_t[2] - bbox_t[0]
        th = bbox_t[3] - bbox_t[1]
        # Position label above bbox
        lx = max(0, int(cx - tw / 2))
        ly = max(0, y0 - th - 4)
        draw.rectangle([lx - 1, ly - 1, lx + tw + 1, ly + th + 1],
                        fill=(0, 0, 0, 160))
        draw.text((lx, ly), label, fill=colour_a, font=font)

    # Composite
    pil_img = Image.alpha_composite(pil_img, overlay)

    # Summary label (top-left)
    final = pil_img.convert("RGB")
    draw2 = ImageDraw.Draw(final)
    n = len(regions)
    n_with_attrs = sum(1 for r in regions if r.get("attributes"))
    summary = f"{n} vessels (AI2)"
    if n_with_attrs > 0:
        types = {}
        for r in regions:
            t = r.get("attributes", {}).get("vessel_type", "unknown")
            types[t] = types.get(t, 0) + 1
        top_types = sorted(types.items(), key=lambda x: -x[1])[:3]
        summary += " — " + ", ".join(f"{v}× {k}" for k, v in top_types)

    bbox_s = draw2.textbbox((0, 0), summary, font=font_big)
    sw = bbox_s[2] - bbox_s[0]
    sh = bbox_s[3] - bbox_s[1]
    draw2.rectangle([4, 4, 8 + sw, 8 + sh], fill=(0, 0, 0, 200))
    draw2.text((6, 6), summary, fill=(0, 255, 180), font=font_big)

    final.save(path)
    print(f"    saved: {path}")
    return path


def save_vessel_heatmap_png(
    heatmap: np.ndarray,
    path: str,
    cmap_name: str = "YlOrRd",
) -> str:
    """Save vessel density heatmap as a clean RGBA PNG for Leaflet overlay.

    Normalises the heatmap to [0, 1] and applies a matplotlib colourmap.
    Zero-density pixels are fully transparent so the background layer
    (RGB or sjökort) shows through.

    Args:
        heatmap: (H, W) float32 vessel density array (e.g. Gaussian-smoothed
            detection counts).
        path: Output PNG path.
        cmap_name: Matplotlib colourmap name (default ``"YlOrRd"``).

    Returns:
        The output file path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm

    vmax = heatmap.max()
    if vmax > 0:
        norm = (heatmap / vmax).clip(0, 1)
    else:
        norm = heatmap

    cmap = cm.get_cmap(cmap_name)
    rgba = (cmap(norm) * 255).astype(np.uint8)  # (H, W, 4) RGBA

    # Scale alpha by normalised density so low-value areas fade out
    # smoothly instead of showing a constant yellow wash.
    rgba[:, :, 3] = (norm * 255).astype(np.uint8)

    Image.fromarray(rgba, "RGBA").save(path)
    print(f"    saved: {path}")
    return path


def save_lpis_overlay(
    rgb: np.ndarray,
    lpis_gdf,
    geo,
    path: str,
    fill_color: tuple = (0.0, 0.0, 0.0, 0.0),
    edge_color: tuple = (0.90, 0.07, 0.61),
    edge_width: int = 2,
    predictions: dict | None = None,
) -> str:
    """Render LPIS pasture polygons on top of an RGB image.

    Burns polygon perimeters (and optional fills) onto the satellite image
    using rasterio rasterization.

    When *predictions* is provided, each polygon is coloured by its
    predicted class: green for active grazing, grey for no activity,
    and magenta for polygons without a prediction.

    Args:
        rgb: (H, W, 3) float32 [0, 1] satellite image.
        lpis_gdf: GeoDataFrame with polygon geometries in EPSG:3006.
        geo: GeoContext with ``transform`` and ``shape``.
        path: Output PNG path.
        fill_color: RGBA fill colour for polygon interiors (default: no fill).
        edge_color: RGB edge colour for polygon boundaries (used when
            *predictions* is ``None``).
        edge_width: Boundary line width in pixels.
        predictions: Optional dict mapping polygon_id (str) to a prediction
            object with ``predicted_class`` attribute (1=active, 0=no activity).

    Returns:
        The output file path.
    """
    from rasterio.features import rasterize
    from shapely.geometry import mapping

    _PRED_COLORS = {
        1: (0.2, 0.8, 0.33),   # active grazing → green
        0: (0.53, 0.53, 0.53),  # no activity → grey
    }
    _DEFAULT_COLOR = (0.90, 0.07, 0.61)  # magenta (not analyzed)

    h, w = rgb.shape[:2]
    transform = geo.transform

    geoms = list(lpis_gdf.geometry)
    if not geoms:
        img = (rgb * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img).save(path)
        print(f"    saved: {path} (no LPIS polygons)")
        return path

    out = rgb.copy()

    # Apply fill (alpha blend) — skip if fully transparent
    alpha = fill_color[3] if len(fill_color) > 3 else 0.0
    if alpha > 0:
        fill_mask = rasterize(
            [(mapping(g), 1) for g in geoms],
            out_shape=(h, w),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )
        for c in range(3):
            out[:, :, c] = np.where(
                fill_mask > 0,
                out[:, :, c] * (1 - alpha) + fill_color[c] * alpha,
                out[:, :, c],
            )

    # Rasterize edges per polygon (or all at once if no predictions)
    if edge_width > 0:
        # Group polygons by edge colour
        color_groups: dict[tuple, list] = {}
        for i, (_, feat) in enumerate(lpis_gdf.iterrows()):
            geom = feat.geometry
            if geom is None or geom.is_empty:
                continue
            if predictions:
                bid = str(feat.get("blockid", ""))
                pred = predictions.get(bid)
                cls = pred.predicted_class if pred else -1
                colour = _PRED_COLORS.get(cls, _DEFAULT_COLOR)
            else:
                colour = edge_color
            color_groups.setdefault(colour, []).append(geom)

        for colour, group_geoms in color_groups.items():
            buffered = [g.buffer(edge_width * 10) for g in group_geoms]
            edge_outer = rasterize(
                [(mapping(g), 1) for g in buffered],
                out_shape=(h, w),
                transform=transform,
                fill=0,
                dtype=np.uint8,
            )
            eroded = [g.buffer(-edge_width * 10) for g in group_geoms]
            eroded = [g for g in eroded if not g.is_empty]
            if eroded:
                inner = rasterize(
                    [(mapping(g), 1) for g in eroded],
                    out_shape=(h, w),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8,
                )
            else:
                inner = np.zeros((h, w), dtype=np.uint8)
            fill_rast = rasterize(
                [(mapping(g), 1) for g in group_geoms],
                out_shape=(h, w),
                transform=transform,
                fill=0,
                dtype=np.uint8,
            )
            edge_mask = ((edge_outer > 0) | (fill_rast > 0)) & (inner == 0)
            for c in range(3):
                out[:, :, c] = np.where(edge_mask, colour[c], out[:, :, c])

    img = (out * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)
    print(f"    saved: {path} ({len(geoms)} LPIS polygons)")
    return path


def save_lpis_geojson(lpis_gdf, geo, path: str, img_shape=None,
                      predictions: dict | None = None) -> str:
    """Convert LPIS polygons to pixel-coordinate GeoJSON for Leaflet CRS.Simple.

    Transforms polygon coordinates from their source CRS to pixel space
    suitable for Leaflet CRS.Simple where bounds = ``[[0, 0], [H, W]]``.

    In CRS.Simple the Y-axis points **up** (lat increases upward) but in
    raster pixel space row 0 is the top.  We therefore output GeoJSON
    coordinates as ``[col, H - row]`` so that ``coordsToLatLng(c)`` →
    ``L.latLng(c[1], c[0])`` = ``L.latLng(H - row, col)`` places
    features correctly on top of the image overlay.

    Args:
        lpis_gdf: GeoDataFrame with polygon geometries (any CRS — will
            be reprojected to match *geo.crs* if needed).
        geo: GeoContext with ``crs``, ``transform`` and ``shape``.
        path: Output GeoJSON file path.
        img_shape: ``(H, W)`` tuple.  Falls back to ``geo.shape`` if
            *None*.
        predictions: Optional dict mapping polygon_id (str) to a
            prediction object with ``predicted_class``, ``class_label``,
            and ``confidence`` attributes.

    Returns:
        The output file path.
    """
    import json
    from shapely.geometry import mapping
    from shapely.ops import transform as shp_transform
    from rasterio.transform import AffineTransformer

    # ── Determine image height ──────────────────────────────────────
    if img_shape is not None:
        img_h = int(img_shape[0])
    elif hasattr(geo, "shape") and geo.shape is not None:
        img_h = int(geo.shape[0])
    else:
        raise ValueError("img_shape or geo.shape required for Y-flip")

    # ── Ensure CRS matches the raster transform ─────────────────────
    gdf = lpis_gdf
    target_crs = getattr(geo, "crs", "EPSG:3006")
    if gdf.crs is not None and not gdf.crs.equals(target_crs):
        print(f"    Reprojecting LPIS from {gdf.crs} → {target_crs}")
        gdf = gdf.to_crs(target_crs)

    transformer = AffineTransformer(geo.transform)

    def _to_pixel(x, y, z=None):
        """Convert projected (x, y) → GeoJSON (col, H-row) for Leaflet."""
        row, col = transformer.rowcol(x, y)   # (row, col) from CRS coords
        return (float(col), float(img_h - row))  # GeoJSON [x=col, y=H-row]

    features = []
    for _, feat in gdf.iterrows():
        geom = feat.geometry
        if geom is None or geom.is_empty:
            continue

        # Transform coordinates: projected CRS → pixel space (Y-flipped)
        pixel_geom = shp_transform(_to_pixel, geom)

        props = {}
        for col_name in ["blockid", "agoslag", "areal"]:
            if col_name in gdf.columns:
                val = feat.get(col_name)
                if val is not None:
                    props[col_name] = val if not hasattr(val, 'item') else val.item()

        # Add grazing model predictions if available
        if predictions:
            bid = str(props.get("blockid", ""))
            pred = predictions.get(bid)
            if pred is not None:
                props["predicted_class"] = pred.predicted_class
                props["class_label"] = pred.class_label
                props["confidence"] = round(pred.confidence, 3)

        features.append({
            "type": "Feature",
            "geometry": mapping(pixel_geom),
            "properties": props,
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False)

    print(f"    saved: {path} ({len(features)} features)")
    return path


def save_regions_leaflet_geojson(
    regions: list[dict],
    img_shape: tuple,
    path: str,
) -> str:
    """Convert pixel-space bounding-box regions to Leaflet CRS.Simple GeoJSON.

    Each region dict must contain a ``"bbox"`` key with
    ``{x_min, y_min, x_max, y_max}`` in pixel coordinates.  The output
    GeoJSON uses ``[col, H - row]`` coordinates so that Leaflet
    ``coordsToLatLng(c)`` → ``L.latLng(c[1], c[0])`` places rectangles
    correctly with image bounds ``[[0, 0], [H, W]]``.

    Args:
        regions: List of dicts with ``"bbox"`` plus optional properties
            (``score``, ``label``, ``attributes``, etc.).
        img_shape: ``(H, W)`` image dimensions.
        path: Output GeoJSON file path.

    Returns:
        The output file path.
    """
    import json

    img_h = int(img_shape[0])

    features = []
    for region in regions:
        bbox = region["bbox"]
        x_min, y_min = bbox["x_min"], bbox["y_min"]
        x_max, y_max = bbox["x_max"], bbox["y_max"]

        # Raw pixel coordinates — Leaflet coordsToLatLng handles Y-flip
        polygon = [
            [float(x_min), float(y_min)],
            [float(x_max), float(y_min)],
            [float(x_max), float(y_max)],
            [float(x_min), float(y_max)],
            [float(x_min), float(y_min)],
        ]

        properties = {k: v for k, v in region.items() if k != "bbox"}
        # Ensure JSON-serialisable (numpy scalars → Python)
        for k, v in properties.items():
            if hasattr(v, "item"):
                properties[k] = v.item()

        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [polygon]},
            "properties": properties,
        })

    geojson = {"type": "FeatureCollection", "features": features}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False)

    print(f"    saved: {path} ({len(features)} features)")
    return path


# ──────────────────────────────────────────────────────────────────────
#  Coastline / Shoreline exports
# ──────────────────────────────────────────────────────────────────────

_COAST_CLASS_COLORS = {
    0: (0.08, 0.40, 0.75),   # water — dark blue
    1: (0.39, 0.71, 0.96),   # whitewater — light blue
    2: (0.83, 0.65, 0.42),   # sediment — beige/sand
    3: (0.30, 0.69, 0.31),   # other (land) — green
}


def save_segmentation_clean_png(
    seg_map: np.ndarray, path: str
) -> str:
    """Save a 4-class coastal segmentation map as a coloured PNG.

    Classes: 0=water (blue), 1=whitewater (light blue),
             2=sediment (beige), 3=other/land (green).
    """
    from PIL import Image

    h, w = seg_map.shape
    out = np.zeros((h, w, 3), dtype=np.float32)
    for cls, color in _COAST_CLASS_COLORS.items():
        mask = seg_map == cls
        for c in range(3):
            out[:, :, c][mask] = color[c]

    img = (out * 255).astype(np.uint8)
    Image.fromarray(img).save(path)
    print(f"    saved: {path}")
    return path


def save_shoreline_overlay(
    rgb: np.ndarray,
    shoreline_mask: np.ndarray,
    path: str,
    color: tuple[float, float, float] = (1.0, 0.34, 0.13),
    line_width: int = 2,
) -> str:
    """Save RGB with shoreline edges overlaid as coloured lines.

    Args:
        rgb: (H, W, 3) float32 [0, 1].
        shoreline_mask: (H, W) uint8 binary (255 = shoreline).
        color: overlay colour (default: orange-red).
        line_width: dilation of the shoreline line.
    """
    from PIL import Image

    overlay = rgb.copy()

    # Dilate shoreline for visibility
    if line_width > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (line_width, line_width)
        )
        mask = cv2.dilate(shoreline_mask, kernel)
    else:
        mask = shoreline_mask

    edge = mask > 127
    for c in range(3):
        overlay[:, :, c][edge] = color[c]

    img = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img).save(path)
    print(f"    saved: {path}")
    return path


def save_shoreline_change_png(
    shorelines_per_year: dict[int, np.ndarray],
    rgb_ref: np.ndarray,
    path: str,
) -> str:
    """Save multi-year shoreline change overlay on reference RGB.

    Each year's shoreline is drawn with a colour gradient from yellow
    (oldest) to red (newest).

    Args:
        shorelines_per_year: {year: (H, W) uint8 shoreline mask}.
        rgb_ref: (H, W, 3) float32 [0, 1] reference image.
    """
    from PIL import Image

    overlay = (rgb_ref * 0.6).copy()  # darken background
    years = sorted(shorelines_per_year.keys())
    n = max(len(years) - 1, 1)

    for i, year in enumerate(years):
        t = i / n  # 0 → oldest, 1 → newest
        # Yellow → Red gradient
        r, g, b = 1.0, 1.0 - 0.8 * t, 0.1 * (1.0 - t)
        mask = shorelines_per_year[year]

        # Dilate for visibility
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask = cv2.dilate(mask, kernel)
        edge = mask > 127

        for c_idx, c_val in enumerate([r, g, b]):
            overlay[:, :, c_idx][edge] = c_val

    img = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img).save(path)
    print(f"    saved: {path} ({len(years)} years)")
    return path


def save_coastline_geojson(
    shorelines_per_year: dict[int, list[np.ndarray]],
    geo,
    path: str,
    img_shape: tuple[int, int] | None = None,
    pixel_coords: bool = False,
    smooth_sigma: float = 3.0,
    subsample_step: int = 3,
) -> str:
    """Save shoreline contours as GeoJSON with per-year properties.

    Each contour becomes a LineString feature with year and length_m.

    Args:
        shorelines_per_year: {year: [array (N,2) pixel coords, ...]}.
        geo: GeoContext with transform and CRS.
        path: output JSON path.
        img_shape: (H, W) — required for pixel_coords mode (Y-flip).
        pixel_coords: If True, output pixel coordinates (for Leaflet
            Simple CRS) with Y-flipped for bottom-origin.  If False,
            output WGS84 geographic coordinates.
        smooth_sigma: Gaussian smoothing sigma (0 to disable).
        subsample_step: Keep every Nth point after smoothing (1 = all).
    """
    from scipy.ndimage import gaussian_filter1d

    features = []

    if not pixel_coords:
        from rasterio.transform import Affine
        tf = geo.transform
        if not isinstance(tf, Affine):
            tf = Affine(*tf[:6]) if len(tf) >= 6 else Affine(10, 0, 0, 0, -10, 0)

    img_h = img_shape[0] if img_shape else 0

    for year in sorted(shorelines_per_year.keys()):
        for contour in shorelines_per_year[year]:
            if len(contour) < 3:
                continue

            xs = contour[:, 0].astype(float)
            ys = contour[:, 1].astype(float)

            # Gaussian smoothing
            if smooth_sigma > 0 and len(xs) > 5:
                xs = gaussian_filter1d(xs, sigma=smooth_sigma)
                ys = gaussian_filter1d(ys, sigma=smooth_sigma)

            # Subsample
            if subsample_step > 1:
                xs_s = xs[::subsample_step]
                ys_s = ys[::subsample_step]
                if len(xs) % subsample_step != 1:
                    xs_s = np.append(xs_s, xs[-1])
                    ys_s = np.append(ys_s, ys[-1])
                xs, ys = xs_s, ys_s

            if pixel_coords:
                # Raw pixel coordinates — Leaflet coordsToLatLng handles
                # the Y-flip (row 0 = top → lat = imgH) at render time.
                coords = [
                    [round(float(x), 2), round(float(y), 2)]
                    for x, y in zip(xs, ys)
                ]
                # Length in meters (10 m/pixel)
                length_m = sum(
                    ((coords[j][0] - coords[j-1][0])**2 +
                     (coords[j][1] - coords[j-1][1])**2) ** 0.5 * 10
                    for j in range(1, len(coords))
                )
            else:
                # Convert pixel coords to projected CRS, then to WGS84
                coords_proj = [
                    tf * (float(x), float(y)) for x, y in zip(xs, ys)
                ]
                length_m = sum(
                    ((coords_proj[j][0] - coords_proj[j-1][0])**2 +
                     (coords_proj[j][1] - coords_proj[j-1][1])**2) ** 0.5
                    for j in range(1, len(coords_proj))
                )
                try:
                    from pyproj import Transformer
                    transformer = Transformer.from_crs(
                        geo.crs, "EPSG:4326", always_xy=True
                    )
                    coords = [
                        list(transformer.transform(x, y))
                        for x, y in coords_proj
                    ]
                except Exception:
                    coords = [[x, y] for x, y in coords_proj]

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords,
                },
                "properties": {
                    "year": year,
                    "length_m": round(length_m, 1),
                },
            })

    geojson = {"type": "FeatureCollection", "features": features}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False)

    n_lines = len(features)
    years = sorted(shorelines_per_year.keys())
    print(f"    saved: {path} ({n_lines} lines, years {years[0]}–{years[-1]})")
    return path
