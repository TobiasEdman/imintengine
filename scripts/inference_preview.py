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
    merge_nmd_lpis,
    nmd10_to_unified,
)

CLASS_NAMES = [UNIFIED_CLASSES.get(i, f"cls_{i}") for i in range(NUM_UNIFIED_CLASSES)]

NMD_COLORS = {
    0: (0, 0, 0), 1: (0, 100, 0), 2: (34, 139, 34), 3: (50, 205, 50),
    4: (60, 179, 113), 5: (46, 79, 46), 6: (139, 90, 43), 7: (200, 150, 50),
    8: (210, 180, 140), 9: (255, 0, 0), 10: (0, 0, 255),
}


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
    num_classes = ckpt_config.get("num_classes", 20)
    num_frames = ckpt_config.get("num_temporal_frames", 4)
    feature_indices = ckpt_config.get("feature_indices", [2, 5, 8, 11])
    decoder_channels = ckpt_config.get("decoder_channels", 256)

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

    sd = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    sd = {(k[len("model."):] if k.startswith("model.") else k): v
          for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  Loaded ({len(missing)} missing, {len(unexpected)} unexpected)")
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tiles-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-tiles", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)

    tiles = sorted(glob.glob(os.path.join(args.tiles_dir, "*.npz")))
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

        nmd_label = data.get("label", None)
        if nmd_label is not None:
            nmd_label = np.array(nmd_label)

        # Build unified ground truth
        lpis = data.get("label_mask", None)
        if nmd_label is not None and nmd_label.ndim == 2:
            if lpis is not None:
                unified_gt = merge_nmd_lpis(nmd_label, np.array(lpis))
            else:
                unified_gt = nmd10_to_unified(nmd_label)
        else:
            unified_gt = np.zeros((256, 256), dtype=np.uint8)

        rgb = make_summer_rgb(spectral)

        # Inference
        inp = torch.from_numpy(spectral).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp)
            if isinstance(out, dict):
                logits = out.get("out", list(out.values())[0])
            else:
                logits = out
            pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

        gt_rgb = label_to_rgb(unified_gt, UNIFIED_COLORS)
        pred_rgb = label_to_rgb(pred, UNIFIED_COLORS)
        nmd_rgb = (
            label_to_rgb(nmd_label, NMD_COLORS)
            if nmd_label is not None and nmd_label.ndim == 2
            else np.zeros_like(rgb)
        )

        Image.fromarray(rgb).save(f"{args.output_dir}/{name}_rgb.png")
        Image.fromarray(gt_rgb).save(f"{args.output_dir}/{name}_truth.png")
        Image.fromarray(pred_rgb).save(f"{args.output_dir}/{name}_pred.png")
        Image.fromarray(nmd_rgb).save(f"{args.output_dir}/{name}_nmd.png")

        results.append({
            "name": name,
            "pred": [CLASS_NAMES[c] for c in np.unique(pred) if c < len(CLASS_NAMES)],
            "gt": [CLASS_NAMES[c] for c in np.unique(unified_gt) if c < len(CLASS_NAMES)],
        })
        print(f"    pred: {results[-1]['pred']}")

    json.dump(results, open(f"{args.output_dir}/results.json", "w"), indent=2)
    print(f"\nDone! {len(results)} tiles → {args.output_dir}/")


if __name__ == "__main__":
    main()
