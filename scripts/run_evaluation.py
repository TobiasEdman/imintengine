#!/usr/bin/env python3
"""Evaluate a trained LULC model on the test split.

Usage:
    python scripts/run_evaluation.py \
        --checkpoint checkpoints/lulc/best_model.pt \
        --data-dir data/lulc_full \
        --split test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    print("ERROR: PyTorch is required. Install with: pip install torch")
    sys.exit(1)

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imint.training.config import TrainingConfig
from imint.training.dataset import LULCDataset
from imint.training.evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate LULC model")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/lulc_full",
        help="Data directory with tiles/ and split files",
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate (default: test)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit samples for quick check",
    )
    # Aux channel flags
    parser.add_argument("--enable-height", action="store_true")
    parser.add_argument("--enable-volume", action="store_true")
    parser.add_argument("--enable-basal-area", action="store_true")
    parser.add_argument("--enable-diameter", action="store_true")
    parser.add_argument("--enable-dem", action="store_true")
    args = parser.parse_args()

    # Build config
    config = TrainingConfig(
        data_dir=args.data_dir,
        enable_height_channel=args.enable_height,
        enable_volume_channel=args.enable_volume,
        enable_basal_area_channel=args.enable_basal_area,
        enable_diameter_channel=args.enable_diameter,
        enable_dem_channel=args.enable_dem,
    )

    # Resolve device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"\n{'='*60}")
    print(f"  LULC Model Evaluation")
    print(f"{'='*60}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Data:       {args.data_dir}")
    print(f"  Split:      {args.split}")
    print(f"  Device:     {device}")
    if args.max_samples:
        print(f"  Max samples: {args.max_samples}")

    # Count aux channels
    aux_names = []
    if config.enable_height_channel:
        aux_names.append("height")
    if config.enable_volume_channel:
        aux_names.append("volume")
    if config.enable_basal_area_channel:
        aux_names.append("basal_area")
    if config.enable_diameter_channel:
        aux_names.append("diameter")
    if config.enable_dem_channel:
        aux_names.append("dem")
    print(f"  Aux channels: {aux_names or 'none'}")
    print(f"{'='*60}\n")

    # Load dataset
    print(f"  Loading {args.split} dataset...")
    dataset = LULCDataset(
        data_dir=args.data_dir,
        split=args.split,
        config=config,
    )
    print(f"  {args.split}: {len(dataset)} tiles")

    # Build model (same as trainer._build_model)
    print(f"  Loading Prithvi backbone...")
    from imint.fm.terratorch_loader import _load_prithvi_from_hf
    from imint.fm.upernet import PrithviSegmentationModel

    backbone = _load_prithvi_from_hf(pretrained=True)

    n_aux = len(aux_names)
    model = PrithviSegmentationModel(
        encoder=backbone,
        num_classes=config.num_classes + 1,
        decoder_channels=config.decoder_channels,
        feature_indices=config.feature_indices,
        dropout=config.dropout,
        n_aux_channels=n_aux,
    )

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Remove "model." prefix if present (Lightning convention)
    clean_state = {}
    for k, v in state_dict.items():
        key = k.replace("model.", "", 1) if k.startswith("model.") else k
        clean_state[key] = v

    model.load_state_dict(clean_state, strict=False)
    model.to(device)
    model.eval()
    print(f"  Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Run evaluation
    print(f"\n  Evaluating on {len(dataset)} tiles...")
    t0 = time.time()
    metrics = evaluate_model(
        model=model,
        dataset=dataset,
        config=config,
        device=device,
        max_samples=args.max_samples,
    )
    elapsed = time.time() - t0

    # Print results
    print(f"\n{'='*60}")
    print(f"  Results ({args.split} split, {len(dataset)} tiles)")
    print(f"{'='*60}")
    print(f"  mIoU:              {metrics['miou']:.4f}")
    print(f"  Overall accuracy:  {metrics['overall_accuracy']:.4f}")
    print(f"  Time:              {elapsed:.1f}s ({elapsed/len(dataset):.2f}s/tile)")
    print()
    print(f"  Per-class IoU:")
    per_class = metrics["per_class_iou"]
    for name, iou in sorted(per_class.items(), key=lambda x: -x[1]):
        bar = "#" * int(iou * 40)
        print(f"    {name:<20s} {iou:.4f}  {bar}")
    print(f"{'='*60}\n")

    # Save results
    out_path = Path(args.data_dir) / f"eval_{args.split}.json"
    result = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "miou": metrics["miou"],
        "overall_accuracy": metrics["overall_accuracy"],
        "per_class_iou": metrics["per_class_iou"],
        "n_tiles": len(dataset),
        "aux_channels": aux_names,
        "elapsed_s": round(elapsed, 1),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
