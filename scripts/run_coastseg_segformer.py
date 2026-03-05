"""Run CoastSeg SegFormer inference on cached Sentinel-2 kustlinje data.

Loads the SegFormer-B0 model (4-class coastal segmentation) and runs it
on Sentinel-2 RGB for all years (2018–2025), producing segmentation maps,
shoreline vectors, and change analysis — comparable to the index-based
(NDWI/MNDWI + Otsu) pipeline.

Usage:
    .venv/bin/python scripts/run_coastseg_segformer.py
    .venv/bin/python scripts/run_coastseg_segformer.py --year 2025   # single year only

Output:
    outputs/showcase/kustlinje/coastseg_segformer.png         (reference year)
    outputs/showcase/kustlinje/segformer_shoreline_change.png  (multi-year overlay)
    outputs/showcase/kustlinje/segformer_vectors.json          (GeoJSON contours)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


# Band order in cached timeseries
_BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
               "B08", "B8A", "B09", "B11", "B12"]

YEARS = list(range(2018, 2026))

# CoastSeg 4-class colours (same as index-based for fair comparison)
CLASS_COLORS = {
    0: (20 / 255, 102 / 255, 191 / 255),   # deep water — blue
    1: (99 / 255, 181 / 255, 244 / 255),   # whitewater/shallow — light blue
    2: (211 / 255, 165 / 255, 107 / 255),  # sediment — beige
    3: (76 / 255, 175 / 255, 79 / 255),    # land — green
}


def _pick_best_date(dates: list[str], cloud_fractions: list[float]) -> int:
    """Pick the best cloud-free date, preferring summer months."""
    best_idx, best_score = 0, -1.0
    for i, (d, cf) in enumerate(zip(dates, cloud_fractions)):
        month = int(d[5:7])
        season_bonus = 1.0 if 6 <= month <= 8 else 0.5 if month in (5, 9) else 0.0
        score = (1.0 - cf) + season_bonus
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def _build_rgb(snapshot: np.ndarray) -> np.ndarray:
    """Build contrast-stretched RGB from a (12, H, W) band stack."""
    rgb = np.stack([snapshot[3], snapshot[2], snapshot[1]], axis=-1)  # B04,B03,B02
    p2, p98 = np.percentile(rgb, [2, 98])
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-8), 0.0, 1.0).astype(np.float32)
    return rgb


def load_model(weights_path: str):
    """Load TFSegformerForSemanticSegmentation with CoastSeg weights."""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    from transformers import TFSegformerForSemanticSegmentation, SegformerConfig

    config = SegformerConfig(
        num_labels=4,
        num_channels=3,
        hidden_sizes=[32, 64, 160, 256],  # B0
        depths=[2, 2, 2, 2],
        num_attention_heads=[1, 2, 5, 8],
        decoder_hidden_size=256,
        image_size=512,
    )
    model = TFSegformerForSemanticSegmentation(config)

    # Build model
    dummy = np.zeros((1, 3, 512, 512), dtype=np.float32)
    _ = model(pixel_values=tf.constant(dummy), training=False)

    model.load_weights(weights_path)
    print(f"    Loaded SegFormer-B0: {model.count_params():,} parameters")
    return model


def predict_tile(model, rgb_tile: np.ndarray) -> np.ndarray:
    """Run inference on a single 512×512 RGB tile.

    Args:
        model: TFSegformerForSemanticSegmentation
        rgb_tile: (512, 512, 3) float32 in [0, 1]

    Returns:
        seg_map: (512, 512) uint8 — class indices 0–3
    """
    import tensorflow as tf

    # SegFormer expects (B, C, H, W) with pixel values in [0, 255] range
    # The CoastSeg training normalises to [0, 1] — match that
    pixel_values = np.transpose(rgb_tile, (2, 0, 1))[np.newaxis]  # (1, 3, 512, 512)
    pixel_values = pixel_values.astype(np.float32)

    outputs = model(pixel_values=tf.constant(pixel_values), training=False)
    logits = outputs.logits.numpy()  # (1, 4, 128, 128)

    # Upsample logits to 512×512
    logits_tf = tf.constant(logits)
    logits_up = tf.image.resize(
        tf.transpose(logits_tf[0], [1, 2, 0]),  # (128, 128, 4)
        [512, 512],
        method='bilinear',
    )
    seg_map = tf.argmax(logits_up, axis=-1).numpy().astype(np.uint8)  # (512, 512)
    return seg_map


def predict_full_image(model, rgb: np.ndarray, tile_size: int = 512, overlap: int = 64) -> np.ndarray:
    """Run tiled inference on a full image of any size.

    Splits the image into overlapping 512×512 tiles, runs each through
    the model, and stitches back together using the non-overlapping centers.

    Args:
        model: loaded SegFormer model
        rgb: (H, W, 3) float32 in [0, 1]
        tile_size: model input size (512)
        overlap: overlap between tiles

    Returns:
        seg_map: (H, W) uint8
    """
    H, W = rgb.shape[:2]
    step = tile_size - overlap

    # Pad image to at least tile_size in each dimension
    pad_h = max(0, tile_size - H)
    pad_w = max(0, tile_size - W)
    if pad_h > 0 or pad_w > 0:
        rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    H_pad, W_pad = rgb.shape[:2]

    seg_map_padded = np.zeros((H_pad, W_pad), dtype=np.uint8)

    # Compute tile positions
    y_starts = list(range(0, H_pad - tile_size + 1, step))
    if not y_starts:
        y_starts = [0]
    elif y_starts[-1] + tile_size < H_pad:
        y_starts.append(H_pad - tile_size)
    x_starts = list(range(0, W_pad - tile_size + 1, step))
    if not x_starts:
        x_starts = [0]
    elif x_starts[-1] + tile_size < W_pad:
        x_starts.append(W_pad - tile_size)

    total_tiles = len(y_starts) * len(x_starts)
    print(f"    Image {H}×{W} (padded {H_pad}×{W_pad}) → {total_tiles} tiles ({tile_size}×{tile_size}, overlap={overlap})")

    tile_idx = 0
    for y in y_starts:
        for x in x_starts:
            tile = rgb[y:y + tile_size, x:x + tile_size]
            pred = predict_tile(model, tile)

            # Only write the non-overlapping center region
            # (except for edge tiles which fill to the border)
            y_write_start = overlap // 2 if y > 0 else 0
            y_write_end = tile_size - overlap // 2 if y + tile_size < H_pad else tile_size
            x_write_start = overlap // 2 if x > 0 else 0
            x_write_end = tile_size - overlap // 2 if x + tile_size < W_pad else tile_size

            seg_map_padded[
                y + y_write_start:y + y_write_end,
                x + x_write_start:x + x_write_end,
            ] = pred[y_write_start:y_write_end, x_write_start:x_write_end]

            tile_idx += 1
            if tile_idx % 5 == 0 or tile_idx == total_tiles:
                print(f"    Tile {tile_idx}/{total_tiles}")

    # Crop back to original size
    return seg_map_padded[:H, :W]


def save_segmentation_png(seg_map: np.ndarray, path: str):
    """Save 4-class segmentation as coloured PNG."""
    from PIL import Image

    H, W = seg_map.shape
    out = np.zeros((H, W, 3), dtype=np.float32)
    for cls, color in CLASS_COLORS.items():
        mask = seg_map == cls
        for c in range(3):
            out[:, :, c][mask] = color[c]

    img = (out * 255).astype(np.uint8)
    Image.fromarray(img).save(path)
    print(f"    Saved: {path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run CoastSeg SegFormer inference")
    parser.add_argument("--year", type=int, default=None,
                        help="Single year to process (default: all 2018-2025)")
    args = parser.parse_args()

    cache_dir = str(PROJECT_ROOT / "outputs" / "kustlinje_model")
    out_dir = str(PROJECT_ROOT / "outputs" / "showcase" / "kustlinje")
    weights_path = str(PROJECT_ROOT / "imint" / "fm" / "coastseg" / "model_weights.h5")

    years = [args.year] if args.year else YEARS

    print("=" * 60)
    print("  CoastSeg SegFormer — Multi-year Shoreline Analysis")
    print(f"  Years: {years[0]}–{years[-1]}" if len(years) > 1 else f"  Year: {years[0]}")
    print(f"  Output: {out_dir}")
    print("=" * 60)

    # ── Step 1: Load model ───────────────────────────────────────────
    print(f"\n[1] Loading CoastSeg SegFormer model...")
    model = load_model(weights_path)

    # ── Step 2: Load data & run inference per year ───────────────────
    print(f"\n[2] Running inference per year...")
    yearly_rgbs = {}       # year → (H, W, 3) float32
    yearly_seg = {}        # year → (H, W) uint8
    yearly_dates = {}      # year → str

    for year in years:
        cache_npz = os.path.join(cache_dir, f"bbox_timeseries_{year}.npz")
        cache_json = os.path.join(cache_dir, f"bbox_timeseries_{year}_meta.json")

        if not os.path.isfile(cache_npz):
            print(f"    {year}: No cached data, skipping")
            continue

        data = np.load(cache_npz)["data"]
        with open(cache_json) as f:
            meta = json.load(f)

        best_idx = _pick_best_date(meta["dates"], meta["cloud_fractions"])
        date = meta["dates"][best_idx]
        snapshot = data[best_idx]  # (12, H, W)
        rgb = _build_rgb(snapshot)
        yearly_rgbs[year] = rgb
        yearly_dates[year] = date

        print(f"    {year} ({date})...", end=" ", flush=True)
        seg_map = predict_full_image(model, rgb, tile_size=512, overlap=64)
        yearly_seg[year] = seg_map

        total = seg_map.shape[0] * seg_map.shape[1]
        water_frac = (seg_map <= 1).sum() / total
        land_frac = (seg_map == 3).sum() / total
        sed_frac = (seg_map == 2).sum() / total
        print(f"water={water_frac:.1%}, sediment={sed_frac:.1%}, land={land_frac:.1%}")

    if not yearly_seg:
        print("ERROR: No data for any year. Aborting.")
        return

    # ── Step 3: Extract shorelines from SegFormer segmentation ───────
    print(f"\n[3] Extracting shorelines from SegFormer segmentation...")
    from imint.analyzers.shoreline import ShorelineAnalyzer

    analyzer = ShorelineAnalyzer()
    yearly_shorelines = {}   # year → (H, W) uint8 mask
    yearly_contours = {}     # year → [array (N,2)]

    for year in sorted(yearly_seg.keys()):
        print(f"    {year}...", end=" ", flush=True)
        seg_map = yearly_seg[year]
        shoreline = analyzer.extract_shoreline(seg_map)
        contours = analyzer.extract_contours(shoreline, min_length=10)
        yearly_shorelines[year] = shoreline
        yearly_contours[year] = contours
        print(f"{len(contours)} contours")

    # ── Step 4: Save reference year segmentation ─────────────────────
    ref_year = max(yearly_seg.keys())
    print(f"\n[4] Saving outputs...")

    # Reference segmentation PNG
    save_segmentation_png(
        yearly_seg[ref_year],
        os.path.join(out_dir, "coastseg_segformer.png"),
    )

    # ── Step 5: Save multi-year shoreline change overlay ─────────────
    from imint.exporters.export import (
        save_shoreline_change_png,
        save_coastline_geojson,
    )

    if len(yearly_shorelines) > 1:
        ref_rgb = yearly_rgbs[ref_year]
        save_shoreline_change_png(
            yearly_shorelines,
            ref_rgb,
            os.path.join(out_dir, "segformer_shoreline_change.png"),
        )

    # ── Step 6: Save GeoJSON vectors ─────────────────────────────────
    H, W = yearly_seg[ref_year].shape

    # Build a minimal GeoContext for pixel-coords export
    from rasterio.transform import from_origin
    from imint.fetch import GeoContext

    first_meta_file = os.path.join(cache_dir, f"bbox_timeseries_{ref_year}_meta.json")
    with open(first_meta_file) as f:
        first_meta = json.load(f)

    geo = GeoContext(
        crs="EPSG:3006",
        transform=from_origin(
            first_meta["west"], first_meta["north"],
            first_meta.get("pixel_size", 10), first_meta.get("pixel_size", 10),
        ),
        bounds_projected={
            "west": first_meta["west"],
            "south": first_meta["south"],
            "east": first_meta["east"],
            "north": first_meta["north"],
        },
        bounds_wgs84=None,
        shape=(H, W),
    )

    save_coastline_geojson(
        yearly_contours,
        geo,
        os.path.join(out_dir, "segformer_vectors.json"),
        img_shape=(H, W),
        pixel_coords=True,
        smooth_sigma=3.0,
        subsample_step=3,
    )

    # ── Step 7: Per-year statistics ──────────────────────────────────
    print(f"\n[5] Statistics...")
    per_year = {}
    for year in sorted(yearly_seg.keys()):
        seg = yearly_seg[year]
        total = seg.shape[0] * seg.shape[1]
        per_year[year] = {
            "date": yearly_dates[year],
            "water_fraction": round(float((seg <= 1).sum()) / total, 4),
            "deep_water_fraction": round(float((seg == 0).sum()) / total, 4),
            "shallow_water_fraction": round(float((seg == 1).sum()) / total, 4),
            "sediment_fraction": round(float((seg == 2).sum()) / total, 4),
            "land_fraction": round(float((seg == 3).sum()) / total, 4),
            "n_contours": len(yearly_contours[year]),
        }
        print(f"    {year}: water={per_year[year]['water_fraction']:.1%}, "
              f"sediment={per_year[year]['sediment_fraction']:.1%}, "
              f"{per_year[year]['n_contours']} contours")

    meta_out = {
        "method": "CoastSeg SegFormer-B0 (4-class)",
        "model": "imint/fm/coastseg/model_weights.h5",
        "reference_year": ref_year,
        "years_analyzed": sorted(yearly_seg.keys()),
        "per_year": per_year,
    }
    meta_path = os.path.join(out_dir, "segformer_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta_out, f, indent=2)
    print(f"    Saved: {meta_path}")

    print(f"\n{'=' * 60}")
    print(f"  Done! {len(yearly_seg)} years processed")
    files = [f for f in os.listdir(out_dir) if 'segformer' in f]
    print(f"  SegFormer files: {', '.join(sorted(files))}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import functools
    print = functools.partial(print, flush=True)
    main()
