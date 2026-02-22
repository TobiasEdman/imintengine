"""
imint/exporters/export.py — Output file helpers

Functions for saving analysis results as PNG, GeoTIFF, GeoJSON, and JSON.
Each function is self-contained and can be called independently.
"""
from __future__ import annotations

import os
import json
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


def save_nmd_overlay(l1_raster: np.ndarray, path: str) -> str:
    """Render NMD Level 1 land cover classes as a color-coded PNG overlay.

    Color palette:
        0=unclassified (gray), 1=forest (green), 2=wetland (brown),
        3=cropland (gold), 4=open_land (beige), 5=developed (red),
        6=water (blue)

    Args:
        l1_raster: 2D uint8 array with Level 1 class codes (0-6).
        path: Output PNG path.

    Returns:
        The output file path.
    """
    # Level 1 color palette (index = class code)
    palette = np.array([
        [128, 128, 128],  # 0: unclassified
        [34, 139, 34],    # 1: forest
        [139, 90, 43],    # 2: wetland
        [255, 215, 0],    # 3: cropland
        [210, 180, 140],  # 4: open_land
        [255, 0, 0],      # 5: developed
        [0, 0, 255],      # 6: water
    ], dtype=np.uint8)

    # Clamp to valid range
    clamped = np.clip(l1_raster, 0, len(palette) - 1)
    rgb = palette[clamped]

    Image.fromarray(rgb).save(path)
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


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
