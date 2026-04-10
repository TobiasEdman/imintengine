#!/usr/bin/env python3
"""
scripts/train_lulc.py — Train LULC classifier on Prithvi + UPerNet

Trains a UPerNet decoder on frozen Prithvi-EO-2.0 backbone using
NMD-derived LULC labels. Produces checkpoints compatible with the
TASK_HEAD_REGISTRY / load_segmentation_model() pipeline.

Usage:
    python scripts/train_lulc.py --data-dir data/lulc_training
    python scripts/train_lulc.py --epochs 2 --batch-size 4 --device mps
    python scripts/train_lulc.py --evaluate-only --checkpoint checkpoints/lulc/best_model.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imint.config.env import load_env
load_env()  # Load config/environments/{IMINT_ENV}.env (default: dev)

from imint.training.config import TrainingConfig
from imint.training.dataset import LULCDataset
from imint.training.trainer import LULCTrainer
from imint.training.evaluate import evaluate_model
from imint.training.class_schema import get_class_names


def main():
    parser = argparse.ArgumentParser(
        description="Train Prithvi LULC segmentation model",
    )

    # Use TrainingConfig defaults for all values
    _defaults = TrainingConfig()

    # Data
    parser.add_argument(
        "--data-dir", type=str, default=_defaults.data_dir,
        help="Directory with training tiles",
    )
    parser.add_argument(
        "--num-classes", type=int, default=_defaults.num_classes,
        help="Number of classes: 19 (full NMD L2) or 10 (grouped)",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=_defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=_defaults.batch_size)
    parser.add_argument("--lr", type=float, default=_defaults.lr)
    parser.add_argument("--weight-decay", type=float, default=_defaults.weight_decay)
    parser.add_argument("--patience", type=int, default=_defaults.early_stopping_patience,
                        help="Early stopping patience")
    parser.add_argument("--num-workers", type=int, default=_defaults.num_workers)

    # Model
    parser.add_argument("--decoder-channels", type=int, default=_defaults.decoder_channels)
    parser.add_argument("--dropout", type=float, default=_defaults.dropout)
    parser.add_argument("--device", type=str, default=_defaults.device,
                        help="Device: cuda, mps, cpu (default: auto)")
    parser.add_argument("--unfreeze-layers", type=int,
                        default=_defaults.unfreeze_backbone_layers,
                        help="Unfreeze last N backbone transformer blocks")
    parser.add_argument("--backbone-lr-factor", type=float,
                        default=_defaults.backbone_lr_factor,
                        help="Backbone LR = lr * this factor")

    # Checkpoint
    parser.add_argument("--checkpoint-dir", type=str,
                        default=_defaults.checkpoint_dir)
    parser.add_argument("--save-every", type=int, default=_defaults.save_every_n_epochs,
                        help="Save checkpoint every N epochs")

    # Loss
    parser.add_argument("--loss-type", type=str, default=_defaults.loss_type,
                        choices=["cross_entropy", "focal"],
                        help="Loss function")
    parser.add_argument("--focal-gamma", type=float, default=_defaults.focal_gamma,
                        help="Focal loss gamma")

    # Early stopping metric
    parser.add_argument("--early-stop-metric", type=str,
                        default=_defaults.early_stop_metric,
                        choices=["miou", "worst_class_iou", "combined"],
                        help="Metric for early stopping")

    # Auxiliary channels
    parser.add_argument("--enable-height", action="store_true",
                        help="Enable tree height aux channel")
    parser.add_argument("--enable-volume", action="store_true",
                        help="Enable timber volume aux channel")
    parser.add_argument("--enable-basal-area", action="store_true",
                        help="Enable basal area aux channel")
    parser.add_argument("--enable-diameter", action="store_true",
                        help="Enable stem diameter aux channel")
    parser.add_argument("--enable-dem", action="store_true",
                        help="Enable DEM terrain elevation aux channel")
    parser.add_argument("--enable-vpp", action="store_true",
                        help="Enable HR-VPP phenology aux channels (5 bands)")
    parser.add_argument("--enable-all-aux", action="store_true",
                        help="Enable all auxiliary channels (incl. VPP)")
    parser.add_argument("--enable-multitemporal", action="store_true",
                        help="Enable multitemporal mode (4 seasonal frames)")
    parser.add_argument("--num-frames", type=int, default=None,
                        help="Override num_temporal_frames (default: 4)")

    # Two-stage training
    parser.add_argument("--freeze-spectral", action="store_true",
                        help="Stage 2: freeze backbone+decoder, train only AuxEncoder")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to checkpoint to load (spectral model for stage 2)")

    # Dashboard
    parser.add_argument("--dashboard", action="store_true",
                        help="Launch live training dashboard in browser")
    parser.add_argument("--dashboard-port", type=int, default=8000,
                        help="Port for dashboard HTTP server (default: 8000)")

    # Background mode
    parser.add_argument("--background", action="store_true",
                        help="Run training as a detached background process")

    # Evaluate-only mode
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Only evaluate an existing checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint for evaluation")

    args = parser.parse_args()

    # ── Background mode: re-exec as detached process ──────────────
    if args.background:
        import subprocess
        cmd = [sys.executable, __file__]
        # Forward all args except --background
        for arg in sys.argv[1:]:
            if arg != "--background":
                cmd.append(arg)
        # Always enable dashboard in background mode
        if "--dashboard" not in cmd:
            cmd.append("--dashboard")

        log_dir = Path(args.data_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "train.log"

        with open(log_file, "w") as lf:
            proc = subprocess.Popen(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(f"\n  Training started in background (PID {proc.pid})")
        print(f"  Log: {log_file}")
        print(f"  Dashboard: http://localhost:{args.dashboard_port}/training_dashboard.html")
        print(f"\n  To stop: kill {proc.pid}")
        return

    # Resolve aux flags
    all_aux = args.enable_all_aux
    config = TrainingConfig(
        data_dir=args.data_dir,
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.patience,
        num_workers=args.num_workers,
        decoder_channels=args.decoder_channels,
        dropout=args.dropout,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        save_every_n_epochs=args.save_every,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        early_stop_metric=args.early_stop_metric,
        unfreeze_backbone_layers=args.unfreeze_layers,
        backbone_lr_factor=args.backbone_lr_factor,
        enable_multitemporal=args.enable_multitemporal,
        num_temporal_frames=args.num_frames or 4,
        enable_height_channel=all_aux or args.enable_height,
        enable_volume_channel=all_aux or args.enable_volume,
        enable_basal_area_channel=all_aux or args.enable_basal_area,
        enable_diameter_channel=all_aux or args.enable_diameter,
        enable_dem_channel=all_aux or args.enable_dem,
        enable_vpp_channels=all_aux or args.enable_vpp,
        freeze_spectral=args.freeze_spectral,
        resume_from_checkpoint=args.resume_from,
    )

    # Load datasets
    print(f"\n  Loading datasets from {config.data_dir}...")
    train_dataset = LULCDataset(config.data_dir, split="train", config=config)
    val_dataset = LULCDataset(config.data_dir, split="val", config=config)
    print(f"  Train: {len(train_dataset)} tiles, Val: {len(val_dataset)} tiles")

    if args.evaluate_only:
        # Evaluate-only mode
        checkpoint_path = args.checkpoint or str(
            Path(config.checkpoint_dir) / "best_model.pt"
        )
        print(f"\n  Evaluating checkpoint: {checkpoint_path}")

        import torch
        from imint.fm.terratorch_loader import _load_prithvi_from_hf
        from imint.fm.upernet import PrithviSegmentationModel

        # Load model
        num_frames = config.num_temporal_frames if config.enable_multitemporal else 1
        backbone = _load_prithvi_from_hf(pretrained=True, num_frames=num_frames)
        model = PrithviSegmentationModel(
            encoder=backbone,
            feature_indices=config.feature_indices,
            decoder_channels=config.decoder_channels,
            num_classes=config.num_classes + 1,
            dropout=config.dropout,
        )

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        # Strip "model." prefix
        clean_sd = {}
        for k, v in state_dict.items():
            key = k[len("model."):] if k.startswith("model.") else k
            clean_sd[key] = v
        model.load_state_dict(clean_sd, strict=False)

        device = torch.device(config.device or "cpu")
        model.to(device)

        # Evaluate on test set if available
        for split_name in ["test", "val"]:
            try:
                ds = LULCDataset(config.data_dir, split=split_name, config=config)
                print(f"\n  Evaluating on {split_name} ({len(ds)} tiles)...")
                metrics = evaluate_model(model, ds, config, device)
                print(f"  mIoU: {metrics['miou']:.4f}")
                print(f"  Overall Accuracy: {metrics['overall_accuracy']:.4f}")
                print(f"  Per-class IoU:")
                for name, iou in metrics["per_class_iou"].items():
                    print(f"    {name:30s} {iou:.4f}")
            except FileNotFoundError:
                print(f"  Split '{split_name}' not found, skipping.")

    else:
        # Training mode
        if args.dashboard:
            from imint.training.dashboard import start_dashboard_server
            start_dashboard_server(
                config.data_dir,
                port=args.dashboard_port,
                open_browser=True,
            )

        trainer = LULCTrainer(config)
        result = trainer.train(train_dataset, val_dataset)

        print(f"\n{'='*60}")
        print(f"  Result:")
        print(f"    Best mIoU:     {result['best_miou']:.4f}")
        print(f"    Best epoch:    {result['best_epoch']}")
        print(f"    Checkpoint:    {result['checkpoint']}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
