#!/usr/bin/env python3
"""scripts/train_pixel.py — Train Prithvi pixel-context classifier.

Trains a center-pixel classifier using Prithvi-EO-2.0 as a frozen / fine-tuned
backbone.  Each sample is a (T×6, 32, 32) spectral context patch; the label is
the unified 23-class integer at the center pixel.

AUX channel injection (enabled by default):
    The classifier fuses the Prithvi CLS token with a small MLP projection
    of 11 center-pixel auxiliary scalars (tree height, timber volume, basal
    area, stem diameter, DEM, HR-VPP phenology metrics, harvest probability).
    These are z-score normalized identically to UnifiedDataset.  Pass
    ``--no-aux`` to train the spectral-only baseline.

Two-stage training:
    Stage 1 (``--stage1-epochs``, default 5):
        Backbone frozen (except pos_embed).  MLP head, AUX projector and
        positional embeddings are trained.
    Stage 2 (remaining epochs):
        Full fine-tuning.  Backbone LR = ``--backbone-lr-factor`` × head LR
        (default 0.1×).  Discriminative LR prevents catastrophic forgetting.

Usage::

    # Full training on H100 (recommended)
    python scripts/train_pixel.py \\
        --data-dir /data/unified_v2 \\
        --use-frame-2016 \\
        --epochs 35 --stage1-epochs 5 \\
        --batch-size 512 --lr 3e-4 \\
        --checkpoint-dir /checkpoints/pixel_v1 \\
        --device cuda

    # Quick smoke test (3 epochs, 200 tiles)
    python scripts/train_pixel.py \\
        --data-dir /data/unified_v2 \\
        --limit-tiles 200 \\
        --epochs 3 --batch-size 64 \\
        --device cuda

    # Spectral-only baseline (no AUX)
    python scripts/train_pixel.py \\
        --data-dir /data/unified_v2 --no-aux \\
        --checkpoint-dir /checkpoints/pixel_spectral_only

    # Evaluate existing checkpoint
    python scripts/train_pixel.py \\
        --data-dir /data/unified_v2 \\
        --evaluate-only --checkpoint /checkpoints/pixel_v1/best_model.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Ensure repo root on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from imint.training.pixel_dataset import PixelContextDataset, N_AUX
from imint.training.unified_schema import (
    NUM_UNIFIED_CLASSES,
    UNIFIED_CLASS_NAMES,
    get_class_weights,
)
from imint.training.losses import FocalLoss
from imint.fm.pixel_head import PrithviPixelClassifier, build_pixel_classifier, N_AUX_DEFAULT


# ── Argument parser ───────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train Prithvi pixel-context classifier (23-class unified LULC)"
    )

    # Data
    p.add_argument("--data-dir", required=True,
                   help="Directory with unified_v2 .npz tiles")
    p.add_argument("--split-dir", default=None,
                   help="Directory with train.txt / val.txt (optional; auto-split if absent)")
    p.add_argument("--limit-tiles", type=int, default=None,
                   help="Limit number of tiles loaded (for testing)")
    p.add_argument("--use-frame-2016", action="store_true", default=True,
                   help="Prepend 2016 background frame (T=5, default True)")
    p.add_argument("--no-frame-2016", dest="use_frame_2016", action="store_false",
                   help="Disable 2016 background frame (T=4)")
    p.add_argument("--use-aux", action="store_true", default=True,
                   help="Inject center-pixel AUX channels into head (default True)")
    p.add_argument("--no-aux", dest="use_aux", action="store_false",
                   help="Disable AUX injection (spectral-only baseline)")
    p.add_argument("--context-px", type=int, default=32,
                   help="Context window side length in pixels (default: 32)")
    p.add_argument("--samples-per-tile", type=int, default=512,
                   help="Pixel samples per tile in dataset (default: 512)")

    # Training
    p.add_argument("--epochs", type=int, default=35,
                   help="Total training epochs (default: 35)")
    p.add_argument("--stage1-epochs", type=int, default=5,
                   help="Epochs with backbone frozen (default: 5)")
    p.add_argument("--batch-size", type=int, default=512,
                   help="Batch size (default: 512)")
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Head (and classifier) learning rate (default: 3e-4)")
    p.add_argument("--backbone-lr-factor", type=float, default=0.1,
                   help="Backbone LR = lr × factor in Stage 2 (default: 0.1)")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="Weight decay (default: 1e-4)")
    p.add_argument("--focal-gamma", type=float, default=2.0,
                   help="Focal loss gamma (default: 2.0)")
    p.add_argument("--num-workers", type=int, default=8,
                   help="DataLoader workers (default: 8)")
    p.add_argument("--patience", type=int, default=10,
                   help="Early stopping patience (default: 10)")

    # Device
    p.add_argument("--device", default=None,
                   help="cuda / mps / cpu (auto-detect if not set)")

    # Checkpoint
    p.add_argument("--checkpoint-dir", default="checkpoints/pixel",
                   help="Output directory for checkpoints")
    p.add_argument("--resume-from", default=None,
                   help="Resume training from checkpoint")

    # Evaluate-only
    p.add_argument("--evaluate-only", action="store_true")
    p.add_argument("--checkpoint", default=None,
                   help="Checkpoint for --evaluate-only")

    return p


# ── Metrics ───────────────────────────────────────────────────────────────

def compute_metrics(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    num_classes: int = NUM_UNIFIED_CLASSES,
    ignore_index: int = 0,
) -> dict:
    """Compute per-class accuracy and mean accuracy (ignore index 0)."""
    per_class_correct = np.zeros(num_classes, dtype=np.int64)
    per_class_total = np.zeros(num_classes, dtype=np.int64)

    for cls in range(1, num_classes):
        mask = all_labels == cls
        per_class_total[cls] = mask.sum()
        per_class_correct[cls] = (all_preds[mask] == cls).sum()

    per_class_acc = np.where(
        per_class_total > 0,
        per_class_correct / np.maximum(per_class_total, 1),
        np.nan,
    )

    valid = ~np.isnan(per_class_acc[1:])
    mean_acc = float(per_class_acc[1:][valid].mean()) if valid.any() else 0.0
    overall_acc = float(
        (all_preds[all_labels != ignore_index] == all_labels[all_labels != ignore_index]).mean()
        if (all_labels != ignore_index).any() else 0.0
    )

    return {
        "mean_acc": mean_acc,
        "overall_acc": overall_acc,
        "per_class_acc": {
            UNIFIED_CLASS_NAMES[i]: float(per_class_acc[i])
            for i in range(1, num_classes)
            if not np.isnan(per_class_acc[i])
        },
        "per_class_n": {
            UNIFIED_CLASS_NAMES[i]: int(per_class_total[i])
            for i in range(1, num_classes)
        },
    }


# ── Training ──────────────────────────────────────────────────────────────

def _run_epoch(
    model: PrithviPixelClassifier,
    loader: DataLoader,
    criterion,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
    use_aux: bool = True,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run one epoch. Returns (loss, preds, labels).

    When ``use_aux=True`` the DataLoader is expected to yield 3-tuples
    ``(patches, aux_vec, labels)``; otherwise 2-tuples ``(patches, labels)``.
    """
    model.train(train)
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(train):
        for batch in loader:
            if use_aux:
                patches, aux_vec, labels = batch
                aux_vec = aux_vec.to(device, non_blocking=True)
            else:
                patches, labels = batch
                aux_vec = None

            patches = patches.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(patches, aux_vec)  # (B, num_classes)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    preds_arr = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)
    avg_loss = total_loss / max(len(labels_arr), 1)
    return avg_loss, preds_arr, labels_arr


def _build_optimizer(
    model: PrithviPixelClassifier,
    lr: float,
    backbone_lr_factor: float,
    weight_decay: float,
    stage: int,
) -> torch.optim.AdamW:
    """Build optimizer with discriminative LRs for Stage 2."""
    if stage == 1:
        # Only head + pos_embed
        params = [
            {"params": model.head_parameters(), "lr": lr},
            {
                "params": [
                    p for n, p in model.backbone.named_parameters()
                    if "pos_embed" in n and p.requires_grad
                ],
                "lr": lr * 0.5,
            },
        ]
    else:
        # Head at full LR, backbone at reduced LR
        params = [
            {"params": model.head_parameters(), "lr": lr},
            {"params": model.backbone_parameters(), "lr": lr * backbone_lr_factor},
        ]
    return torch.optim.AdamW(params, weight_decay=weight_decay)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"  Device: {device}")

    # ── Discover tiles ────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    all_tiles = sorted(data_dir.glob("*.npz"))
    if args.limit_tiles:
        all_tiles = all_tiles[:args.limit_tiles]
    print(f"  Tiles: {len(all_tiles):,}")

    # ── Splits ────────────────────────────────────────────────────
    if args.split_dir:
        split_dir = Path(args.split_dir)
        def _read_ids(fname):
            p = split_dir / fname
            if not p.exists():
                return set()
            return set(p.read_text().split())

        train_ids = _read_ids("train.txt")
        val_ids = _read_ids("val.txt")

        train_tiles = [t for t in all_tiles if t.stem in train_ids]
        val_tiles = [t for t in all_tiles if t.stem in val_ids]
        if not train_tiles:
            # Fallback: use all tiles for train, empty for val
            train_tiles = all_tiles
            val_tiles = []
    else:
        # 90/10 random split
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(all_tiles))
        split = int(len(all_tiles) * 0.9)
        train_tiles = [all_tiles[i] for i in perm[:split]]
        val_tiles = [all_tiles[i] for i in perm[split:]]

    print(f"  Train tiles: {len(train_tiles):,}, Val tiles: {len(val_tiles):,}")

    # num_frames: 5 if using 2016 frame, else 4
    num_frames = 5 if args.use_frame_2016 else 4
    n_aux = N_AUX if args.use_aux else 0

    # ── Datasets ──────────────────────────────────────────────────
    aux_label = f"+{n_aux}aux" if n_aux > 0 else "spectral-only"
    print(f"\n  Building datasets (context={args.context_px}px, "
          f"frames={num_frames}, {aux_label}, samples/tile={args.samples_per_tile}) …")
    t0 = time.time()

    train_ds = PixelContextDataset(
        train_tiles,
        context_px=args.context_px,
        split="train",
        samples_per_tile=args.samples_per_tile,
        use_frame_2016=args.use_frame_2016,
        enable_aux=args.use_aux,
    )
    val_ds = PixelContextDataset(
        val_tiles,
        context_px=args.context_px,
        split="val",
        samples_per_tile=args.samples_per_tile // 2,
        use_frame_2016=args.use_frame_2016,
        enable_aux=args.use_aux,
    ) if val_tiles else None

    print(f"  Train samples: {len(train_ds):,}, "
          f"Val samples: {len(val_ds):,}  ({time.time()-t0:.0f}s)")

    # ── Model ─────────────────────────────────────────────────────
    print(f"\n  Loading PrithviPixelClassifier "
          f"(T={num_frames}, n_aux={n_aux}, pretrained=True) …")
    t0 = time.time()
    model = PrithviPixelClassifier(
        num_classes=NUM_UNIFIED_CLASSES,
        context_px=args.context_px,
        num_frames=num_frames,
        n_aux=n_aux,
        pretrained=True,
    )
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params/1e6:.1f}M  ({time.time()-t0:.0f}s)")

    # ── Resume ────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=True)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"  Resumed from {args.resume_from} (epoch {start_epoch})")

    # ── Evaluate-only mode ────────────────────────────────────────
    if args.evaluate_only:
        ckpt_path = args.checkpoint or str(
            Path(args.checkpoint_dir) / "best_model.pt"
        )
        print(f"\n  Evaluating {ckpt_path} …")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        model.eval()

        eval_ds = val_ds or train_ds
        loader = DataLoader(
            eval_ds, batch_size=512, num_workers=4, pin_memory=True,
        )
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        _, preds, labels = _run_epoch(
            model, loader, criterion, None, device, train=False, use_aux=args.use_aux,
        )
        metrics = compute_metrics(preds, labels)
        print(f"  Mean Accuracy : {metrics['mean_acc']:.4f}")
        print(f"  Overall Acc   : {metrics['overall_acc']:.4f}")
        print(f"  Per-class accuracy:")
        for name, acc in metrics["per_class_acc"].items():
            n = metrics["per_class_n"].get(name, 0)
            print(f"    {name:20s}  {acc:.3f}  (n={n:,})")
        return

    # ── Class weights (inverse-frequency from training labels) ────
    print("\n  Computing class weights …")
    class_counts = np.zeros(NUM_UNIFIED_CLASSES, dtype=np.int64)
    for _, _, _, cls in train_ds._index:
        class_counts[cls] += 1
    class_weights_np = get_class_weights(
        {i: int(class_counts[i]) for i in range(NUM_UNIFIED_CLASSES)}
    )
    class_weights = torch.tensor(
        list(class_weights_np.values()), dtype=torch.float32, device=device
    )

    criterion = FocalLoss(
        weight=class_weights,
        gamma=args.focal_gamma,
        ignore_index=0,
    )

    # ── DataLoaders ───────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    ) if val_ds else None

    # ── Checkpoint dir ────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────
    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0
    current_stage = 1

    # Stage 1: freeze backbone
    model.freeze_backbone()
    optimizer = _build_optimizer(model, args.lr, args.backbone_lr_factor,
                                  args.weight_decay, stage=1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.stage1_epochs, eta_min=args.lr * 0.1
    )

    print(f"\n{'='*60}")
    print(f"  Training: {args.epochs} epochs total, "
          f"Stage 1 = first {args.stage1_epochs} epochs")
    print(f"  Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"  AUX channels: {n_aux} ({'enabled' if n_aux > 0 else 'disabled'})")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        # Switch to Stage 2
        if epoch == args.stage1_epochs and current_stage == 1:
            print(f"\n  ── Stage 2: unfreezing backbone (epoch {epoch}) ──\n")
            current_stage = 2
            model.unfreeze_backbone()
            optimizer = _build_optimizer(
                model, args.lr, args.backbone_lr_factor,
                args.weight_decay, stage=2,
            )
            stage2_epochs = args.epochs - args.stage1_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=stage2_epochs, eta_min=args.lr * 0.01
            )

        t0 = time.time()
        train_loss, train_preds, train_labels = _run_epoch(
            model, train_loader, criterion, optimizer, device,
            train=True, use_aux=args.use_aux,
        )
        train_metrics = compute_metrics(train_preds, train_labels)
        scheduler.step()

        val_str = ""
        val_acc = 0.0
        if val_loader is not None:
            val_loss, val_preds, val_labels = _run_epoch(
                model, val_loader, criterion, None, device,
                train=False, use_aux=args.use_aux,
            )
            val_metrics = compute_metrics(val_preds, val_labels)
            val_acc = val_metrics["mean_acc"]
            val_str = (
                f"  val_loss={val_loss:.4f}  val_mAcc={val_acc:.4f}"
                f"  val_OA={val_metrics['overall_acc']:.4f}"
            )

        elapsed = time.time() - t0
        head_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch+1:3d}/{args.epochs}  "
            f"loss={train_loss:.4f}  mAcc={train_metrics['mean_acc']:.4f}"
            f"{val_str}  lr={head_lr:.2e}  {elapsed:.0f}s"
        )

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            no_improve = 0
            save_path = ckpt_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mean_acc": val_acc,
                "n_aux": n_aux,
                "num_frames": num_frames,
                "args": vars(args),
            }, save_path)
            print(f"    ↑ New best mAcc={best_val_acc:.4f} → {save_path}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\n  Early stopping (no improvement for {args.patience} epochs)")
                break

        # Periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            periodic_path = ckpt_dir / f"checkpoint_epoch{epoch+1:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_mean_acc": val_acc,
            }, periodic_path)

    # ── Final summary ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best val mAcc: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"  Best model:    {ckpt_dir / 'best_model.pt'}")
    print(f"{'='*60}")

    # ── Final per-class eval on best model ────────────────────────
    if val_loader is not None:
        print("\n  Final per-class accuracy (best model):")
        best_ckpt = torch.load(ckpt_dir / "best_model.pt",
                               map_location=device, weights_only=True)
        model.load_state_dict(best_ckpt["model_state_dict"])
        _, val_preds, val_labels = _run_epoch(
            model, val_loader, criterion, None, device,
            train=False, use_aux=args.use_aux,
        )
        metrics = compute_metrics(val_preds, val_labels)
        for name, acc in metrics["per_class_acc"].items():
            n = metrics["per_class_n"].get(name, 0)
            print(f"    {name:20s}  {acc:.3f}  (n={n:,})")


if __name__ == "__main__":
    main()
