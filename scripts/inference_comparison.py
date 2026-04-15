#!/usr/bin/env python3
"""Run inference on viz tiles with multiple model checkpoints.

Loads each best_model.pt, runs on the 5 viz tiles, and outputs
a JSON file per model with base64-encoded colored predictions.

Usage (on K8s pod or locally with checkpoints):
    python scripts/inference_comparison.py \
        --viz-dir /data/viz_tiles \
        --checkpoints /checkpoints/unified_v5a/best_model.pt \
                      /checkpoints/unified_v5b/best_model.pt \
                      /checkpoints/unified_v5c/best_model.pt \
        --output-dir /data/viz_tiles/predictions
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imint.training.unified_schema import UNIFIED_CLASSES

# 23-class RGB color palette
CLASS_COLORS = np.array([
    (0, 0, 0),         # 0  bakgrund
    (26, 92, 53),       # 1  tallskog
    (45, 138, 91),      # 2  granskog
    (123, 198, 126),    # 3  lövskog
    (77, 179, 128),     # 4  blandskog
    (107, 142, 90),     # 5  sumpskog
    (201, 223, 110),    # 6  tillfälligt ej skog
    (155, 119, 34),     # 7  våtmark
    (212, 180, 74),     # 8  öppen mark
    (192, 57, 43),      # 9  bebyggelse
    (36, 113, 163),     # 10 vatten
    (232, 184, 0),      # 11 vete
    (212, 120, 10),     # 12 korn
    (240, 208, 96),     # 13 havre
    (212, 198, 0),      # 14 oljeväxter
    (145, 200, 76),     # 15 slåttervall
    (184, 222, 134),    # 16 bete
    (155, 89, 182),     # 17 potatis
    (214, 51, 129),     # 18 sockerbetor
    (224, 112, 32),     # 19 trindsäd
    (139, 32, 32),      # 20 råg
    (220, 200, 0),      # 21 majs
    (0, 168, 198),      # 22 hygge
], dtype=np.uint8)


def pred_to_rgb(pred: np.ndarray) -> np.ndarray:
    """Convert (H, W) class indices to (H, W, 3) RGB."""
    return CLASS_COLORS[np.clip(pred, 0, len(CLASS_COLORS) - 1)]


def rgb_to_b64png(rgb: np.ndarray) -> str:
    """(H, W, 3) uint8 → base64 PNG string."""
    from PIL import Image
    img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def load_model(ckpt_path: str, device):
    """Load a PrithviSegmentationModel from checkpoint."""
    import torch
    from imint.fm.terratorch_loader import _load_prithvi_from_hf
    from imint.fm.upernet import PrithviSegmentationModel, get_default_pool_sizes
    from imint.training.config import TrainingConfig

    cfg = TrainingConfig()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Read model config from checkpoint (img_size, num_frames may differ per run)
    ck_cfg = ck.get("config", {})
    num_frames = ck_cfg.get("num_temporal_frames", cfg.num_temporal_frames)
    n_aux = ck_cfg.get("n_aux_channels", 11)

    # Infer img_size from pos_embed shape in checkpoint
    sd = {k.replace("model.", "", 1): v for k, v in
          ck.get("model_state_dict", ck.get("state_dict", {})).items()}
    pos_embed = sd.get("encoder.pos_embed")
    if pos_embed is not None:
        n_tokens = pos_embed.shape[1] - 1  # subtract CLS token
        n_spatial = n_tokens // max(num_frames, 1)
        grid_size = int(n_spatial ** 0.5)
        img_size = grid_size * 16  # patch_size = 16
    else:
        img_size = cfg.img_size

    backbone = _load_prithvi_from_hf(
        pretrained=False, num_frames=num_frames, img_size=img_size,
    )
    model = PrithviSegmentationModel(
        encoder=backbone,
        feature_indices=cfg.feature_indices,
        decoder_channels=cfg.decoder_channels,
        num_classes=cfg.num_classes,
        enable_temporal_pooling=cfg.enable_temporal_pooling,
        enable_multilevel_aux=cfg.enable_multilevel_aux,
        n_aux_channels=n_aux,
        pool_sizes=get_default_pool_sizes(device, img_size=img_size),
    )
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()

    epoch = ck.get("epoch", "?")
    miou = ck.get("metrics", {}).get("miou", "?")
    return model, epoch, miou, img_size


def run_inference(model, tile_path: str, device, img_size: int = 224):
    """Run inference on a single tile. Returns (H, W) prediction."""
    import torch
    from imint.training.unified_dataset import (
        PRITHVI_MEAN, PRITHVI_STD, N_BANDS,
        AUX_NORM, AUX_LOG_TRANSFORM, AUX_CHANNEL_NAMES,
    )

    data = np.load(tile_path, allow_pickle=True)
    spectral = data.get("spectral", data.get("image")).astype(np.float32)

    # Normalize: reflectance → DN → Prithvi z-score
    n_frames = spectral.shape[0] // N_BANDS
    mean = np.tile(PRITHVI_MEAN.reshape(N_BANDS, 1, 1), (n_frames, 1, 1))
    std = np.tile(PRITHVI_STD.reshape(N_BANDS, 1, 1), (n_frames, 1, 1))
    spectral = (spectral * 10000.0 - mean) / std

    # Center crop to img_size (no crop if tile == img_size)
    _, h, w = spectral.shape
    crop_sz = min(img_size, h, w)
    y0 = (h - crop_sz) // 2
    x0 = (w - crop_sz) // 2
    spectral = spectral[:, y0:y0+crop_sz, x0:x0+crop_sz]

    # Aux channels
    aux_list = []
    for ch_name in AUX_CHANNEL_NAMES:
        if ch_name in data:
            arr = data[ch_name].astype(np.float32)
            arr = arr[y0:y0+crop_sz, x0:x0+crop_sz]
            if ch_name in AUX_LOG_TRANSFORM:
                arr = np.log1p(arr)
            if ch_name in AUX_NORM:
                m, s = AUX_NORM[ch_name]
                arr = (arr - m) / max(s, 1e-6)
            aux_list.append(arr[np.newaxis])
        else:
            aux_list.append(np.zeros((1, crop_sz, crop_sz), dtype=np.float32))

    img = torch.from_numpy(spectral).unsqueeze(0).to(device)
    T = img.shape[1] // 6
    img5d = img.view(1, T, 6, crop_sz, crop_sz).permute(0, 2, 1, 3, 4)

    aux = torch.from_numpy(np.concatenate(aux_list, axis=0)).unsqueeze(0).to(device)

    # Build TL coords
    temporal_coords = None
    location_coords = None
    doy = data.get("doy")
    if doy is not None:
        from imint.training.sampler import _sweref99_to_wgs84
        year = int(data.get("year", data.get("lpis_year", 2022)))
        tc = np.zeros((n_frames, 2), dtype=np.float32)
        tc[:, 0] = float(year)
        tc[:len(doy), 1] = doy[:n_frames].astype(np.float32)
        temporal_coords = torch.from_numpy(tc).unsqueeze(0).to(device)

        easting = float(data.get("easting", 500_000))
        northing = float(data.get("northing", 6_500_000))
        lat, lon = _sweref99_to_wgs84(easting, northing)
        location_coords = torch.from_numpy(
            np.array([[lat, lon]], dtype=np.float32)
        ).to(device)

    with torch.no_grad():
        pred = model(
            img5d, aux=aux,
            temporal_coords=temporal_coords,
            location_coords=location_coords,
        ).argmax(1).squeeze(0).cpu().numpy()
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--viz-dir", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--labels", nargs="*", help="Labels for each checkpoint (e.g. v5a v5b v5c)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    viz_dir = Path(args.viz_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tiles = sorted(viz_dir.glob("*.npz"))
    if not tiles:
        print(f"ERROR: no .npz files in {viz_dir}")
        sys.exit(1)
    print(f"Tiles: {len(tiles)}")

    labels = args.labels or [Path(c).parent.name for c in args.checkpoints]

    for ckpt_path, label in zip(args.checkpoints, labels):
        print(f"\n=== {label}: {ckpt_path} ===")
        model, epoch, miou, model_img_size = load_model(ckpt_path, device)
        print(f"  Loaded: epoch={epoch}, miou={miou}, img_size={model_img_size}")

        result = {"_label": label, "_epoch": str(epoch), "_miou": str(miou)}
        for tile_path in tiles:
            name = tile_path.stem
            print(f"  {name}...", end=" ", flush=True)
            pred = run_inference(model, str(tile_path), device, img_size=model_img_size)
            rgb = pred_to_rgb(pred)
            result[f"{name}_pred"] = rgb_to_b64png(rgb)
            result[f"{name}_shape"] = list(rgb.shape[:2])
            print(f"done ({pred.shape})")

        out_path = out_dir / f"{label}_predictions.json"
        with open(out_path, "w") as f:
            json.dump(result, f)
        print(f"  Wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")

    print("\nAll done.")


if __name__ == "__main__":
    main()
