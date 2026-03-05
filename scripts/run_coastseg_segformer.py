"""Run CoastSeg SegFormer inference on cached Sentinel-2 kustlinje data.

Loads the SegFormer-B0 model (4-class coastal segmentation) and runs it
on the reference year RGB, producing a segmentation map that can be
compared with the index-based (NDWI/MNDWI + Otsu) approach.

Usage:
    .venv/bin/python scripts/run_coastseg_segformer.py [--year 2025]

Output:
    outputs/showcase/kustlinje/coastseg_segformer.png
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
    parser.add_argument("--year", type=int, default=2025, help="Year to process")
    args = parser.parse_args()

    cache_dir = str(PROJECT_ROOT / "outputs" / "kustlinje_model")
    out_dir = str(PROJECT_ROOT / "outputs" / "showcase" / "kustlinje")
    weights_path = str(PROJECT_ROOT / "imint" / "fm" / "coastseg" / "model_weights.h5")

    print("=" * 60)
    print("  CoastSeg SegFormer Inference")
    print(f"  Year: {args.year}")
    print("=" * 60)

    # ── Load cached Sentinel-2 data ──────────────────────────────────
    cache_npz = os.path.join(cache_dir, f"bbox_timeseries_{args.year}.npz")
    cache_json = os.path.join(cache_dir, f"bbox_timeseries_{args.year}_meta.json")

    if not os.path.isfile(cache_npz):
        print(f"ERROR: No cached data for {args.year}: {cache_npz}")
        return

    print(f"\n[1] Loading cached data for {args.year}...")
    data = np.load(cache_npz)["data"]
    with open(cache_json) as f:
        meta = json.load(f)

    best_idx = _pick_best_date(meta["dates"], meta["cloud_fractions"])
    date = meta["dates"][best_idx]
    snapshot = data[best_idx]  # (12, H, W)
    print(f"    Best date: {date} (cloud: {meta['cloud_fractions'][best_idx]:.1%})")
    print(f"    Shape: {snapshot.shape}")

    # Build RGB
    rgb = _build_rgb(snapshot)
    H, W = rgb.shape[:2]
    print(f"    RGB: {H}×{W}")

    # ── Load model ───────────────────────────────────────────────────
    print(f"\n[2] Loading CoastSeg SegFormer model...")
    model = load_model(weights_path)

    # ── Run inference ────────────────────────────────────────────────
    print(f"\n[3] Running inference...")
    seg_map = predict_full_image(model, rgb, tile_size=512, overlap=64)

    # Print class distribution
    total = H * W
    class_names = ["Deep water", "Shallow water", "Sediment", "Land"]
    for i, name in enumerate(class_names):
        frac = (seg_map == i).sum() / total
        print(f"    {name}: {frac:.1%}")

    # ── Save output ──────────────────────────────────────────────────
    print(f"\n[4] Saving output...")
    save_segmentation_png(seg_map, os.path.join(out_dir, "coastseg_segformer.png"))

    print(f"\n{'=' * 60}")
    print(f"  Done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import functools
    print = functools.partial(print, flush=True)
    main()
