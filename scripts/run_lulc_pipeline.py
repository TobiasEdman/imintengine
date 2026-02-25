#!/usr/bin/env python3
"""
scripts/run_lulc_pipeline.py — Full LULC training pipeline

Runs data preparation (Sentinel-2 + NMD fetch) followed by model
training in a single command.  Supports background execution and a
live dashboard for monitoring both phases.

Usage:
    # Full pipeline with dashboard:
    DES_USER=testuser DES_PASSWORD=secretpassword \\
      python scripts/run_lulc_pipeline.py \\
        --data-dir data/lulc_training \\
        --grid-spacing 100000 --years 2019 2020 \\
        --epochs 30 --dashboard

    # Background mode (detached process):
    DES_USER=testuser DES_PASSWORD=secretpassword \\
      python scripts/run_lulc_pipeline.py \\
        --data-dir data/lulc_training \\
        --grid-spacing 100000 --years 2019 2020 \\
        --epochs 30 --background

    # Skip data prep (data already exists):
    python scripts/run_lulc_pipeline.py \\
        --data-dir data/lulc_training --skip-prepare --dashboard

    # Only prepare data (no training):
    DES_USER=testuser DES_PASSWORD=secretpassword \\
      python scripts/run_lulc_pipeline.py \\
        --data-dir data/lulc_training --skip-train --dashboard
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imint.training.config import TrainingConfig


def main():
    parser = argparse.ArgumentParser(
        description="Full LULC pipeline: data preparation + training",
    )

    # ── Data preparation ──────────────────────────────────────────
    parser.add_argument(
        "--data-dir", type=str, default="data/lulc_training",
        help="Directory for training data (default: data/lulc_training)",
    )
    parser.add_argument(
        "--grid-spacing", type=int, default=10_000,
        help="Grid spacing in meters (default: 10000, use 100000 for test)",
    )
    parser.add_argument(
        "--years", nargs="+", default=["2019", "2020"],
        help="Years to fetch data from (default: 2019 2020)",
    )
    parser.add_argument(
        "--num-classes", type=int, default=19,
        help="Number of classes: 19 (full NMD L2) or 10 (grouped)",
    )
    parser.add_argument(
        "--cloud-threshold", type=float, default=0.05,
        help="Max cloud+shadow fraction per tile (default: 0.05)",
    )

    # ── Training ──────────────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience (default: 5)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--decoder-channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda, mps, cpu (default: auto)")
    parser.add_argument("--checkpoint-dir", type=str,
                        default="checkpoints/lulc")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--loss-type", type=str, default="cross_entropy",
                        choices=["cross_entropy", "focal"])
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--early-stop-metric", type=str, default="miou",
                        choices=["miou", "worst_class_iou", "combined"])

    # ── Pipeline control ──────────────────────────────────────────
    parser.add_argument("--skip-prepare", action="store_true",
                        help="Skip data preparation (use existing data)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training (only prepare data)")

    # ── Dashboard & background ────────────────────────────────────
    parser.add_argument("--dashboard", action="store_true",
                        help="Launch live dashboard in browser")
    parser.add_argument("--dashboard-port", type=int, default=8000,
                        help="Port for dashboard HTTP server (default: 8000)")
    parser.add_argument("--background", action="store_true",
                        help="Run as detached background process")

    args = parser.parse_args()

    # ── Background mode: re-exec as detached process ──────────────
    if args.background:
        import subprocess
        cmd = [sys.executable, __file__]
        for arg in sys.argv[1:]:
            if arg != "--background":
                cmd.append(arg)
        if "--dashboard" not in cmd:
            cmd.append("--dashboard")

        log_dir = Path(args.data_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "pipeline.log"

        env = dict(__import__('os').environ, PYTHONUNBUFFERED="1")
        with open(log_file, "w") as lf:
            proc = subprocess.Popen(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=env,
            )
        print(f"\n  Pipeline started in background (PID {proc.pid})")
        print(f"  Log: {log_file}")
        print(f"  Dashboard: http://localhost:{args.dashboard_port}/training_dashboard.html")
        print(f"\n  To stop: kill {proc.pid}")
        return

    # ── Build config ──────────────────────────────────────────────
    config = TrainingConfig(
        data_dir=args.data_dir,
        grid_spacing_m=args.grid_spacing,
        years=args.years,
        num_classes=args.num_classes,
        cloud_threshold=args.cloud_threshold,
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
    )

    # ── Start dashboard ───────────────────────────────────────────
    if args.dashboard:
        from imint.training.dashboard import start_dashboard_server
        start_dashboard_server(
            config.data_dir,
            port=args.dashboard_port,
            open_browser=True,
        )

    # ── Phase 1: Data preparation ─────────────────────────────────
    if not args.skip_prepare:
        print("\n" + "=" * 60)
        print("  Phase 1: Data Preparation")
        print("=" * 60)
        from imint.training.prepare_data import prepare_training_data
        prepare_training_data(config)
    else:
        print("\n  Skipping data preparation (--skip-prepare)")

    # ── Phase 2: Training ─────────────────────────────────────────
    if not args.skip_train:
        print("\n" + "=" * 60)
        print("  Phase 2: Training")
        print("=" * 60)

        from imint.training.dataset import LULCDataset
        from imint.training.trainer import LULCTrainer

        print(f"\n  Loading datasets from {config.data_dir}...")
        train_dataset = LULCDataset(config.data_dir, split="train", config=config)
        val_dataset = LULCDataset(config.data_dir, split="val", config=config)
        print(f"  Train: {len(train_dataset)} tiles, Val: {len(val_dataset)} tiles")

        trainer = LULCTrainer(config)
        result = trainer.train(train_dataset, val_dataset)

        print(f"\n{'=' * 60}")
        print(f"  Result:")
        print(f"    Best mIoU:     {result['best_miou']:.4f}")
        print(f"    Best epoch:    {result['best_epoch']}")
        print(f"    Checkpoint:    {result['checkpoint']}")
        print(f"{'=' * 60}")
    else:
        print("\n  Skipping training (--skip-train)")

    print("\n  Pipeline complete.")

    # Keep the process alive so the dashboard server stays running
    if args.dashboard:
        print("  Dashboard still running. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n  Shutting down.")


if __name__ == "__main__":
    main()
