#!/usr/bin/env python3
"""Generate visual inference previews: RGB / model prediction / NMD ground truth."""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imint.fm.terratorch_loader import _load_prithvi_from_hf
from imint.fm.upernet import PrithviSegmentationModel, get_default_pool_sizes
from imint.training.unified_schema import (
    UNIFIED_CLASSES,
    UNIFIED_COLORS,
    NUM_UNIFIED_CLASSES,
    nmd19_to_unified,
)

from imint.training.unified_dataset import AUX_CHANNEL_NAMES, AUX_NORM

CLASS_NAMES = [UNIFIED_CLASSES.get(i, f"cls_{i}") for i in range(NUM_UNIFIED_CLASSES)]


def label_to_rgb(label: np.ndarray, colors: dict) -> np.ndarray:
    h, w = label.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, c in colors.items():
        mask = label == cls
        if mask.any():
            rgb[mask] = c
    return rgb


def make_summer_rgb(spectral: np.ndarray) -> np.ndarray:
    """Extract summer RGB composite from multitemporal stack."""
    c = spectral.shape[0]
    bands_per_frame = 6
    n_frames = c // bands_per_frame
    # Summer = frame 1 (3-frame) or frame 2 (4-frame)
    summer_idx = min(1, n_frames - 1) if n_frames == 3 else min(2, n_frames - 1)
    base = summer_idx * bands_per_frame
    r = spectral[base + 2]  # B04
    g = spectral[base + 1]  # B03
    b = spectral[base + 0]  # B02
    rgb = np.stack([r, g, b], axis=-1)
    p2, p98 = np.percentile(rgb, [2, 98])
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)
    return (rgb * 255).astype(np.uint8)


def load_model(checkpoint_path: str, device: torch.device):
    """Load PrithviSegmentationModel from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    ckpt_config = ckpt.get("config", {})
    n_aux = ckpt_config.get("n_aux_channels", 0)
    num_frames = ckpt_config.get("num_temporal_frames", 4)
    feature_indices = ckpt_config.get("feature_indices", [2, 5, 8, 11])
    decoder_channels = ckpt_config.get("decoder_channels", 256)

    # Extract state dict first to detect actual num_classes from head
    sd = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    sd = {(k[len("model."):] if k.startswith("model.") else k): v
          for k, v in sd.items()}
    head_key = "head.head.2.bias"
    if head_key in sd:
        num_classes = sd[head_key].shape[0]
    else:
        num_classes = ckpt_config.get("num_classes", 20)

    print(f"  classes={num_classes}, frames={num_frames}, aux={n_aux}")
    backbone = _load_prithvi_from_hf(pretrained=True, num_frames=num_frames)
    model = PrithviSegmentationModel(
        encoder=backbone,
        feature_indices=feature_indices,
        decoder_channels=decoder_channels,
        num_classes=num_classes,
        n_aux_channels=n_aux,
        pool_sizes=get_default_pool_sizes(device),
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  Loaded ({len(missing)} missing, {len(unexpected)} unexpected)")
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="Model checkpoint path. Omit to generate NIR+NMD only (no prediction).")
    parser.add_argument("--tiles-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tile-ids", nargs="+",
                        help="Only process these tile IDs (filename stems, e.g. 45843596)")
    parser.add_argument("--num-tiles", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device) if args.checkpoint else None
    if model is None:
        print("No checkpoint — generating NIR + NMD panels only (skipping prediction)")

    tiles = sorted(glob.glob(os.path.join(args.tiles_dir, "*.npz")))
    if args.tile_ids:
        ids = set(args.tile_ids)
        selected = [t for t in tiles if os.path.basename(t).replace(".npz", "") in ids]
        print(f"Selected {len(selected)} specified tiles")
    else:
        random.seed(args.seed)
        indices = [i * len(tiles) // args.num_tiles for i in range(args.num_tiles)]
        selected = [tiles[i] for i in indices]
        print(f"Selected {len(selected)}/{len(tiles)} tiles")

    results = []
    for i, tile_path in enumerate(selected):
        name = os.path.basename(tile_path).replace(".npz", "")
        print(f"  [{i+1}/{len(selected)}] {name}")
        data = np.load(tile_path, allow_pickle=True)

        spectral = np.array(
            data.get("image", data.get("spectral")), dtype=np.float32
        )

        # Unified ground truth: pre-built label stored directly in tile
        unified_gt = np.asarray(
            data.get("label", np.zeros((256, 256), dtype=np.uint8)), dtype=np.uint8
        )

        # NMD display: use nmd_label_raw (19-class sequential) if available,
        # else fall back to unified label (still meaningful for forest/water/infra)
        nmd_raw = data.get("nmd_label_raw", None)
        nmd_display = (
            nmd19_to_unified(np.asarray(nmd_raw, dtype=np.uint8))
            if nmd_raw is not None
            else unified_gt
        )

        # NIR false-colour: B8A / B04 / B03 → R/G/B (bands index 3/2/1 in PRITHVI_BANDS)
        c = spectral.shape[0]
        bands_per_frame = 6
        n_frames_tile = c // bands_per_frame
        summer_idx = min(1, n_frames_tile - 1) if n_frames_tile == 3 else min(2, n_frames_tile - 1)
        base = summer_idx * bands_per_frame
        # PRITHVI_BANDS = [B02, B03, B04, B8A, B11, B12] → NIR false-colour: B8A=3, B04=2, B03=1
        nir = spectral[base + 3].astype(np.float32)   # B8A
        red = spectral[base + 2].astype(np.float32)   # B04
        grn = spectral[base + 1].astype(np.float32)   # B03
        nir_stack = np.stack([nir, red, grn], axis=-1)
        p2, p98 = np.percentile(nir_stack, [2, 98])
        nir_rgb = np.clip((nir_stack - p2) / (p98 - p2 + 1e-6), 0, 1)
        nir_img = (nir_rgb * 255).astype(np.uint8)

        nmd_rgb = label_to_rgb(nmd_display, UNIFIED_COLORS)
        Image.fromarray(nir_img).save(f"{args.output_dir}/{name}_nir.png")
        Image.fromarray(nmd_rgb).save(f"{args.output_dir}/{name}_nmd.png")

        pred_classes: list[str] = []
        if model is not None:
            model_input = spectral[base:base + bands_per_frame]  # (6, H, W)
            h, w = model_input.shape[1], model_input.shape[2]
            aux_arrays = []
            for ch_name in AUX_CHANNEL_NAMES:
                if ch_name in data:
                    arr = data[ch_name].astype(np.float32)
                    if arr.shape != (h, w):
                        padded = np.zeros((h, w), dtype=np.float32)
                        sh = min(arr.shape[0], h); sw = min(arr.shape[1], w)
                        padded[:sh, :sw] = arr[:sh, :sw]
                        arr = padded
                else:
                    arr = np.zeros((h, w), dtype=np.float32)
                if ch_name in AUX_NORM:
                    mean, std = AUX_NORM[ch_name]
                    arr = (arr - mean) / max(std, 1e-6)
                aux_arrays.append(arr)
            aux_stack = np.stack(aux_arrays, axis=0)
            inp   = torch.from_numpy(model_input).unsqueeze(0).to(device)
            aux_t = torch.from_numpy(aux_stack).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(inp, aux=aux_t)
                logits = out.get("out", list(out.values())[0]) if isinstance(out, dict) else out
                pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
            pred_rgb = label_to_rgb(pred, UNIFIED_COLORS)
            Image.fromarray(pred_rgb).save(f"{args.output_dir}/{name}_pred.png")
            pred_classes = [CLASS_NAMES[c] for c in np.unique(pred) if c < len(CLASS_NAMES)]

        results.append({
            "name": name,
            "pred": pred_classes,
            "gt": [CLASS_NAMES[c] for c in np.unique(unified_gt) if c < len(CLASS_NAMES)],
        })
        print(f"    pred: {results[-1]['pred']}")

    json.dump(results, open(f"{args.output_dir}/results.json", "w"), indent=2)
    print(f"\nDone! {len(results)} tiles → {args.output_dir}/")


if __name__ == "__main__":
    main()
