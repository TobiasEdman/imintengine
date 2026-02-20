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
    coords: dict | None = None,
    image_shape: tuple | None = None,
) -> str:
    """
    Convert pixel-space bounding boxes to GeoJSON polygons.

    If coords and image_shape are provided, maps pixel coordinates
    to geographic (WGS84) coordinates. Otherwise uses pixel coordinates.
    """
    features = []
    for region in regions:
        bbox = region["bbox"]
        if coords and image_shape:
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


def save_geotiff(array: np.ndarray, path: str, coords: dict | None = None) -> str:
    """
    Save a 2D array as a GeoTIFF with spatial reference.
    Uses rasterio if available, falls back to plain TIFF via PIL.
    """
    try:
        import rasterio
        from rasterio.transform import from_bounds

        h, w = array.shape[:2]
        if coords:
            transform = from_bounds(
                coords["west"], coords["south"],
                coords["east"], coords["north"],
                w, h,
            )
        else:
            transform = from_bounds(0, 0, w, h, w, h)

        count = 1 if array.ndim == 2 else array.shape[2]
        with rasterio.open(
            path, "w", driver="GTiff",
            height=h, width=w, count=count,
            dtype=array.dtype,
            crs="EPSG:4326",
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


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
