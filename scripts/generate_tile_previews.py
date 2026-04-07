#!/usr/bin/env python3
"""Generate colored segmentation preview images for 3 representative val tiles.

Runs inference with all available checkpoints and writes tile_previews.json
to the repo root for the training dashboard sidebar.

Usage:
    python scripts/generate_tile_previews.py \
        --data-dir /data/unified_v2 \
        --checkpoint-dir /checkpoints/unified_v5 \
        --output /path/to/tile_previews.json
"""
from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── Color palette ──────────────────────────────────────────────────────────────
from imint.training.unified_schema import (
    UNIFIED_COLORS, UNIFIED_CLASSES, NUM_UNIFIED_CLASSES, HARVEST_CLASS,
)
from imint.training.unified_dataset import UnifiedDataset
from imint.training.config import TrainingConfig

# Build color LUT: (23, 3) uint8
COLOR_LUT = np.zeros((NUM_UNIFIED_CLASSES, 3), dtype=np.uint8)
for i in range(NUM_UNIFIED_CLASSES):
    COLOR_LUT[i] = UNIFIED_COLORS[i]

CLASS_NAMES = [UNIFIED_CLASSES[i] for i in range(NUM_UNIFIED_CLASSES)]


def label_to_rgb(label: np.ndarray) -> np.ndarray:
    """Convert (H, W) integer label to (H, W, 3) uint8 RGB."""
    label_clipped = np.clip(label, 0, NUM_UNIFIED_CLASSES - 1)
    return COLOR_LUT[label_clipped]


def rgb_to_b64(rgb: np.ndarray) -> str:
    """Convert (H, W, 3) uint8 array to base64-encoded PNG string."""
    try:
        from PIL import Image
        img = Image.fromarray(rgb, "RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except ImportError:
        # Fallback: write raw PPM via numpy
        h, w = rgb.shape[:2]
        header = f"P6\n{w} {h}\n255\n".encode()
        raw = header + rgb.tobytes()
        return base64.b64encode(raw).decode()


def find_representative_tiles(ds: UnifiedDataset, n: int = 3) -> list[int]:
    """Find n tiles that together cover hygge, forest confusion, and crops."""
    hygge_idx = None
    forest_idx = None
    crop_idx = None

    for i in range(min(len(ds), 800)):
        lbl = ds[i]["label"].numpy()
        unique = set(lbl[lbl > 0])
        has_hygge = HARVEST_CLASS in unique
        has_forest = bool(unique & {1, 2, 3, 4, 5, 6})
        has_crops = bool(unique & set(range(11, 22)))
        # Pick tile with hygge
        if hygge_idx is None and has_hygge and has_forest:
            hygge_idx = i
        # Pick tile rich in forest confusion classes
        if forest_idx is None and has_forest and not has_hygge and len(unique & {1,2,3,4,5}) >= 3:
            forest_idx = i
        # Pick tile with diverse crops
        if crop_idx is None and has_crops and len(unique & set(range(11,22))) >= 4:
            crop_idx = i
        if hygge_idx is not None and forest_idx is not None and crop_idx is not None:
            break

    # Fall back if any not found
    selected = []
    for idx, label in [(hygge_idx, "hygge+forest"), (forest_idx, "forest mix"), (crop_idx, "crop diversity")]:
        if idx is not None:
            selected.append((idx, label))
    # Pad with first available tiles if needed
    for i in range(min(len(ds), 10)):
        if len(selected) >= n:
            break
        if i not in [x[0] for x in selected]:
            selected.append((i, f"tile_{i}"))
    return selected[:n]


def load_model(ckpt_path: str, cfg: TrainingConfig, device):
    """Load model from checkpoint file."""
    import torch
    from imint.fm.terratorch_loader import _load_prithvi_from_hf
    from imint.fm.upernet import PrithviSegmentationModel, get_default_pool_sizes

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    epoch = ck.get("epoch", "?")

    backbone = _load_prithvi_from_hf(pretrained=True, num_frames=cfg.num_temporal_frames)
    model = PrithviSegmentationModel(
        encoder=backbone,
        num_classes=cfg.num_classes,
        enable_temporal_pooling=cfg.enable_temporal_pooling,
        enable_multilevel_aux=cfg.enable_multilevel_aux,
        n_aux_channels=11,
        pool_sizes=get_default_pool_sizes(device),
    )
    sd = {k.replace("model.", "", 1): v for k, v in ck.get("model_state_dict", ck.get("state_dict", {})).items()}
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    return model, int(epoch)


def run_inference(model, sample: dict, cfg: TrainingConfig, device) -> np.ndarray:
    """Run model inference on one dataset sample. Returns (H, W) prediction."""
    import torch
    img = sample["spectral"].unsqueeze(0).to(device)
    T = img.shape[1] // 6
    img5d = img.view(1, T, 6, img.shape[2], img.shape[3]).permute(0, 2, 1, 3, 4)
    aux_parts = [sample[n].unsqueeze(0).to(device) for n in cfg.enabled_aux_names if n in sample]
    aux = torch.cat(aux_parts, 1) if aux_parts else None
    with torch.no_grad():
        pred = model(img5d, aux=aux).argmax(1).squeeze(0).cpu().numpy()
    return pred


def find_checkpoints(ckpt_dir: str) -> list[tuple[int, str]]:
    """Find all saved checkpoints sorted by epoch (latest first)."""
    ckpt_dir = Path(ckpt_dir)
    checkpoints = []

    # Epoch checkpoints: epoch_NNN.pt
    for p in sorted(ckpt_dir.glob("epoch_*.pt"), reverse=True):
        try:
            ep = int(p.stem.split("_")[1])
            checkpoints.append((ep, str(p)))
        except (IndexError, ValueError):
            pass

    # last_checkpoint.pt
    last = ckpt_dir / "last_checkpoint.pt"
    if last.exists():
        import torch
        ck = torch.load(last, map_location="cpu", weights_only=False)
        ep = ck.get("epoch", 0)
        # Only add if not already in list
        if not any(e == ep for e, _ in checkpoints):
            checkpoints.append((ep, str(last)))

    # Sort latest first, keep at most 5
    checkpoints.sort(key=lambda x: -x[0])
    return checkpoints[:5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/data/unified_v2")
    ap.add_argument("--checkpoint-dir", default="/checkpoints/unified_v5")
    ap.add_argument("--output", default="/workspace/imintengine/tile_previews.json")
    ap.add_argument("--num-tiles", type=int, default=3)
    args = ap.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = TrainingConfig(
        num_classes=23,
        enable_multitemporal=True,
        num_temporal_frames=4,
        enable_temporal_pooling=True,
        enable_multilevel_aux=True,
        loss_type="focal_dice",
    )

    print("Loading val dataset...", flush=True)
    ds = UnifiedDataset(
        lulc_dir=args.data_dir,
        split="val",
        multitemporal=True,
        num_temporal_frames=4,
        augment_override=False,
    )
    print(f"  Val tiles: {len(ds)}", flush=True)

    print("Selecting representative tiles...", flush=True)
    selected = find_representative_tiles(ds, args.num_tiles)
    print(f"  Selected: {selected}", flush=True)

    checkpoints = find_checkpoints(args.checkpoint_dir)
    print(f"  Checkpoints: {[(ep, Path(p).name) for ep, p in checkpoints]}", flush=True)

    if not checkpoints:
        print("No checkpoints found, exiting.", flush=True)
        return

    # Collect samples once
    samples = [(idx, desc, ds[idx]) for idx, desc in selected]

    # Build result structure
    tiles_out = []
    for idx, desc, sample in samples:
        lbl = sample["label"].numpy()
        gt_rgb = label_to_rgb(lbl)
        gt_b64 = rgb_to_b64(gt_rgb)
        tile_name = ds._entries[idx].get("path", f"tile_{idx}")
        tiles_out.append({
            "name": Path(str(tile_name)).name if tile_name else f"tile_{idx}",
            "description": desc,
            "ground_truth_b64": gt_b64,
            "predictions": [],
        })

    # Run inference for each checkpoint (latest first)
    for epoch, ckpt_path in checkpoints:
        print(f"  Loading epoch {epoch} from {Path(ckpt_path).name}...", flush=True)
        try:
            model, actual_epoch = load_model(ckpt_path, cfg, device)
        except Exception as e:
            print(f"    Failed: {e}", flush=True)
            continue

        for i, (idx, desc, sample) in enumerate(samples):
            pred = run_inference(model, sample, cfg, device)
            pred_rgb = label_to_rgb(pred)
            pred_b64 = rgb_to_b64(pred_rgb)

            # Compute per-tile IoU for hygge
            lbl = sample["label"].numpy()
            tp = int(((pred == 22) & (lbl == 22)).sum())
            fp = int(((pred == 22) & (lbl != 22)).sum())
            fn = int(((pred != 22) & (lbl == 22)).sum())
            hygge_iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")

            tiles_out[i]["predictions"].append({
                "epoch": actual_epoch,
                "label": f"Epoch {actual_epoch}",
                "image_b64": pred_b64,
                "hygge_iou": round(hygge_iou, 4) if not (hygge_iou != hygge_iou) else None,
            })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Build color legend
    legend = []
    for i in range(NUM_UNIFIED_CLASSES):
        if i == 0:
            continue
        r, g, b = UNIFIED_COLORS[i]
        legend.append({
            "id": i,
            "name": CLASS_NAMES[i],
            "color": f"rgb({r},{g},{b})",
            "hex": f"#{r:02x}{g:02x}{b:02x}",
            "group": (
                "forest" if i in range(1, 7) else
                "wetland" if i == 7 else
                "open" if i in (8, 21) else
                "developed" if i == 9 else
                "water" if i == 10 else
                "crops" if i in range(11, 22) else
                "harvest"
            ),
        })

    out = {
        "generated_at": int(time.time()),
        "latest_epoch": checkpoints[0][0] if checkpoints else 0,
        "tiles": tiles_out,
        "legend": legend,
    }

    output_path = args.output
    with open(output_path, "w") as f:
        json.dump(out, f)
    print(f"\nWrote {output_path} ({os.path.getsize(output_path)//1024} KB)", flush=True)


if __name__ == "__main__":
    main()
