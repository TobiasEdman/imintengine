#!/usr/bin/env python3
"""
scripts/train_unified.py — Train unified 19-class segmentation model

Trains a UPerNet decoder on frozen Prithvi-EO-2.0 backbone using the
unified LULC + Crop + Harvest schema (19 classes). Uses UnifiedDataset
which merges NMD land cover, LPIS crop detail, and SKS harvest data
from multiple tile directories.

All 8 auxiliary channels enabled by default:
  height, volume, basal_area, diameter, dem, vpp_sosd, vpp_eosd,
  harvest_probability

Produces checkpoints compatible with the TASK_HEAD_REGISTRY /
load_segmentation_model() pipeline.

Usage:
    # Quick local test
    python scripts/train_unified.py --epochs 2 --batch-size 4 --device mps --limit-tiles 100

    # Full training on H100
    python scripts/train_unified.py --epochs 50 --batch-size 8 --device cuda \
        --data-dirs data/lulc_seasonal/tiles,data/crop_tiles \
        --checkpoint-dir checkpoints/unified

    # Evaluate-only
    python scripts/train_unified.py --evaluate-only --checkpoint checkpoints/unified/best_model.pt
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
from imint.training.unified_dataset import UnifiedDataset
from imint.training.trainer import LULCTrainer
from imint.training.evaluate import evaluate_model
from imint.training.unified_schema import UNIFIED_CLASS_NAMES


# Default data directories (comma-separated)
DEFAULT_DATA_DIRS = "data/lulc_seasonal/tiles,data/crop_tiles"


def main():
    parser = argparse.ArgumentParser(
        description="Train Prithvi unified 19-class segmentation model",
    )

    # Use TrainingConfig defaults for all values
    _defaults = TrainingConfig()

    # Data
    parser.add_argument(
        "--data-dirs", type=str, default=DEFAULT_DATA_DIRS,
        help="Comma-separated directories with training tiles "
             "(default: data/lulc_seasonal/tiles,data/crop_tiles)",
    )
    parser.add_argument(
        "--num-classes", type=int, default=19,
        help="Number of classes (default: 19 for unified schema)",
    )
    parser.add_argument(
        "--limit-tiles", type=int, default=None,
        help="Limit total tiles loaded (for quick local testing)",
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
                        default="checkpoints/unified")
    parser.add_argument("--save-every", type=int, default=_defaults.save_every_n_epochs,
                        help="Save checkpoint every N epochs")

    # Loss
    parser.add_argument("--loss-type", type=str, default=_defaults.loss_type,
                        choices=["cross_entropy", "focal", "focal_dice"],
                        help="Loss function")
    parser.add_argument("--focal-gamma", type=float, default=_defaults.focal_gamma,
                        help="Focal loss gamma")

    # Early stopping metric
    parser.add_argument("--early-stop-metric", type=str,
                        default=_defaults.early_stop_metric,
                        choices=["miou", "worst_class_iou", "combined"],
                        help="Metric for early stopping")

    # Auxiliary channels — all 8 enabled by default for unified training
    # Individual disable flags for when you want to ablate specific channels
    parser.add_argument("--disable-height", action="store_true",
                        help="Disable tree height aux channel")
    parser.add_argument("--disable-volume", action="store_true",
                        help="Disable timber volume aux channel")
    parser.add_argument("--disable-basal-area", action="store_true",
                        help="Disable basal area aux channel")
    parser.add_argument("--disable-diameter", action="store_true",
                        help="Disable stem diameter aux channel")
    parser.add_argument("--disable-dem", action="store_true",
                        help="Disable DEM terrain elevation aux channel")
    parser.add_argument("--disable-vpp", action="store_true",
                        help="Disable HR-VPP phenology aux channels (sosd, eosd)")
    parser.add_argument("--disable-harvest-prob", action="store_true",
                        help="Disable harvest probability aux channel")
    parser.add_argument("--disable-all-aux", action="store_true",
                        help="Disable all auxiliary channels")

    # Temporal
    parser.add_argument("--enable-multitemporal", action="store_true",
                        help="Use 4-frame multi-temporal input (autumn + 3 growing season)")
    parser.add_argument("--num-temporal-frames", type=int, default=4,
                        help="Number of temporal frames (default: 4)")

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

    # Parse comma-separated data directories
    data_dirs = [d.strip() for d in args.data_dirs.split(",") if d.strip()]

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

        log_dir = Path(args.checkpoint_dir)
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

    # Resolve aux flags — all enabled by default, individual disable flags to ablate
    all_disabled = args.disable_all_aux
    config = TrainingConfig(
        data_dir=data_dirs[0],  # Primary dir for TrainingConfig compatibility
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
        # Multi-temporal: 4-frame autumn-first pattern when enabled
        enable_multitemporal=args.enable_multitemporal,
        num_temporal_frames=args.num_temporal_frames if args.enable_multitemporal else 1,
        # 5 skogliga + VPP (5 band) + harvest_probability = 11 aux
        enable_height_channel=True,
        enable_volume_channel=True,
        enable_basal_area_channel=True,
        enable_diameter_channel=True,
        enable_dem_channel=True,
        enable_vpp_channels=True,
        enable_harvest_probability=True,
        freeze_spectral=args.freeze_spectral,
        resume_from_checkpoint=args.resume_from,
    )

    # ── Load datasets ─────────────────────────────────────────────
    print(f"\n  Loading unified datasets from {len(data_dirs)} director{'y' if len(data_dirs) == 1 else 'ies'}:")
    for d in data_dirs:
        print(f"    - {d}")
    print(f"  Num classes: {args.num_classes} (unified schema)")
    print(f"  Class names: {UNIFIED_CLASS_NAMES}")
    if args.limit_tiles:
        print(f"  Tile limit: {args.limit_tiles}")

    # Map data_dirs to lulc_dir / crop_dir
    lulc_dir = data_dirs[0] if len(data_dirs) > 0 else None
    crop_dir = data_dirs[1] if len(data_dirs) > 1 else None

    train_dataset = UnifiedDataset(
        lulc_dir=lulc_dir,
        crop_dir=crop_dir,
        split="train",
        patch_size=config.patch_pixels,
        enable_aux=True,
        multitemporal=config.enable_multitemporal,
        num_temporal_frames=config.num_temporal_frames,
    )
    val_dataset = UnifiedDataset(
        lulc_dir=lulc_dir,
        crop_dir=crop_dir,
        split="val",
        patch_size=config.patch_pixels,
        enable_aux=True,
        multitemporal=config.enable_multitemporal,
        num_temporal_frames=config.num_temporal_frames,
    )
    print(f"  Train: {len(train_dataset)} tiles, Val: {len(val_dataset)} tiles")

    # Print aux channel summary
    aux_names = config.enabled_aux_names
    harvest_enabled = not (all_disabled or args.disable_harvest_prob)
    if harvest_enabled:
        aux_names = aux_names + ("harvest_probability",)
    print(f"  Aux channels ({len(aux_names)}): {', '.join(aux_names)}")

    if args.evaluate_only:
        # Evaluate-only mode
        checkpoint_path = args.checkpoint or str(
            Path(config.checkpoint_dir) / "best_model.pt"
        )
        print(f"\n  Evaluating checkpoint: {checkpoint_path}")

        import torch
        from imint.fm.terratorch_loader import _load_prithvi_from_hf
        from imint.fm.upernet import PrithviSegmentationModel

        # Load model (single-date)
        backbone = _load_prithvi_from_hf(pretrained=True, num_frames=1)
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
                ds = UnifiedDataset(
                    lulc_dir=lulc_dir,
                    crop_dir=crop_dir,
                    split=split_name,
                    patch_size=config.patch_pixels,
                    enable_aux=True,
                )
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
