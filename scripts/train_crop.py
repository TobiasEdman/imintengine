#!/usr/bin/env python3
"""
scripts/train_crop.py — Train Swedish crop classifier on Prithvi + UPerNet

Uses the same Prithvi-EO-2.0 backbone + UPerNet decoder as LULC training,
adapted for 8-class crop type classification from LUCAS/LPIS data.

Key differences from train_lulc.py:
  - CropDataset instead of LULCDataset (multitemporal 3×6=18 bands)
  - 8 classes (vete, korn, havre, oljeväxter, vall, potatis, trindsäd, övrig)
  - Focal loss γ=2.0 (same as LULC)
  - Unfreeze last 6 backbone blocks (same as LULC)
  - Classification task (not segmentation) — uses pooled features

Usage:
    python scripts/train_crop.py --data-dir data/crop_tiles
    python scripts/train_crop.py --epochs 50 --batch-size 8 --device cuda
    python scripts/train_crop.py --evaluate-only --checkpoint checkpoints/crop/best_model.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imint.config.env import load_env
load_env()

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
except ImportError:
    raise ImportError("PyTorch required: pip install torch")

from imint.training.crop_dataset import CropDataset, build_crop_sampler
from imint.training.crop_schema import NUM_CLASSES, CLASS_NAMES


# ── Focal Loss (same as LULC) ──────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal loss for classification (from LULC pipeline)."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_factor = (1.0 - p_t) ** self.gamma
        loss = focal_factor * ce_loss
        return loss.mean()


# ── Prithvi Crop Classifier ─────────────────────────────────────────────────

class PrithviCropClassifier(nn.Module):
    """Prithvi backbone + classification head for crop type mapping.

    Uses Prithvi-EO-2.0 as frozen/partially-unfrozen feature extractor,
    then global average pooling + MLP classifier.
    """

    def __init__(
        self,
        num_classes: int = 8,
        num_frames: int = 3,
        backbone_name: str = "prithvi_eo_v2_300m_tl",
        unfreeze_layers: int = 6,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_frames = num_frames

        # Try to load Prithvi backbone via timm
        try:
            import timm
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=True,
                num_frames=num_frames,
                features_only=True,
            )
            # Get feature dimension from backbone
            dummy = torch.randn(1, num_frames * 6, 224, 224)
            with torch.no_grad():
                feats = self.backbone(dummy)
            feat_dim = feats[-1].shape[1]  # Last feature map channels

            # Freeze all, then unfreeze last N blocks
            for param in self.backbone.parameters():
                param.requires_grad = False
            if unfreeze_layers > 0:
                blocks = list(self.backbone.blocks) if hasattr(self.backbone, 'blocks') else []
                for block in blocks[-unfreeze_layers:]:
                    for param in block.parameters():
                        param.requires_grad = True

            self.has_prithvi = True
            print(f"  Prithvi backbone loaded: {backbone_name}")
            print(f"  Feature dim: {feat_dim}, unfrozen: last {unfreeze_layers} blocks")

        except Exception as e:
            print(f"  Prithvi not available ({e}), using CNN backbone")
            self.has_prithvi = False
            feat_dim = 256
            self.backbone = nn.Sequential(
                nn.Conv2d(num_frames * 6, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.Conv2d(256, feat_dim, 3, padding=1), nn.BatchNorm2d(feat_dim), nn.ReLU(),
            )

        # Classification head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_prithvi:
            feats = self.backbone(x)
            x = feats[-1]  # Last feature map
        else:
            x = self.backbone(x)
        x = self.pool(x)
        return self.classifier(x)


# ── Training Loop ────────────────────────────────────────────────────────────

def train(args):
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else
                           "mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}")

    # Dataset
    train_ds = CropDataset(args.data_dir, split="train", patch_size=args.patch_size)
    val_ds = CropDataset(args.data_dir, split="val", patch_size=args.patch_size)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Weighted sampler for class balance
    sampler = build_crop_sampler(train_ds)

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Model
    model = PrithviCropClassifier(
        num_classes=NUM_CLASSES,
        num_frames=3,
        unfreeze_layers=args.unfreeze_layers,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Class weights (capped at 10× like LULC)
    class_counts = train_ds.class_counts()
    total = sum(class_counts.values())
    weights = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    for cls, count in class_counts.items():
        if count > 0:
            w = total / (NUM_CLASSES * count)
            weights[cls] = min(w, 10.0)  # Cap at 10× (same as LULC)
    weights = weights.to(device)
    print(f"Class weights: {[f'{w:.2f}' for w in weights.tolist()]}")

    # Loss — Focal (same as LULC)
    criterion = FocalLoss(gamma=args.focal_gamma, weight=weights)

    # Optimizer — differential LR for backbone vs classifier
    param_groups = []
    if model.has_prithvi:
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        if backbone_params:
            param_groups.append({
                "params": backbone_params,
                "lr": args.lr * args.backbone_lr_factor,
            })
    classifier_params = list(model.classifier.parameters())
    param_groups.append({"params": classifier_params, "lr": args.lr})
    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)

    # Scheduler
    total_steps = len(train_dl) * args.epochs
    warmup_steps = int(total_steps * 0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)

    # Checkpoint dir
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss, correct, total_samples = 0.0, 0, 0
        t0 = time.time()

        for batch in train_dl:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # Warmup
            step = epoch * len(train_dl) + train_dl.dataset.__len__
            if hasattr(scheduler, 'step'):
                scheduler.step()

            train_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total_samples += images.size(0)

        train_acc = correct / max(total_samples, 1)
        avg_loss = train_loss / max(total_samples, 1)
        dt = time.time() - t0

        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == args.epochs - 1:
            model.eval()
            val_correct, val_total = 0, 0
            per_class_correct = [0] * NUM_CLASSES
            per_class_total = [0] * NUM_CLASSES

            with torch.no_grad():
                for batch in val_dl:
                    images = batch["image"].to(device)
                    labels = batch["label"].to(device)
                    logits = model(images)
                    preds = logits.argmax(1)
                    val_correct += (preds == labels).sum().item()
                    val_total += images.size(0)
                    for c in range(NUM_CLASSES):
                        mask = labels == c
                        per_class_correct[c] += (preds[mask] == c).sum().item()
                        per_class_total[c] += mask.sum().item()

            val_acc = val_correct / max(val_total, 1)
            per_class_acc = [
                per_class_correct[c] / max(per_class_total[c], 1)
                for c in range(NUM_CLASSES)
            ]
            worst_class = min(per_class_acc)

            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"loss={avg_loss:.4f} train={train_acc:.3f} "
                  f"val={val_acc:.3f} worst={worst_class:.3f} | {dt:.1f}s")

            # Per-class detail
            for c in range(NUM_CLASSES):
                name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}"
                print(f"  {name:12s}: {per_class_acc[c]:.3f} ({per_class_total[c]} samples)")

            # Save best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "per_class_acc": per_class_acc,
                    "class_names": CLASS_NAMES,
                    "has_prithvi": model.has_prithvi,
                }, ckpt_dir / "best_model.pt")
                print(f"  → Saved best model (val={val_acc:.3f})")
            else:
                patience_counter += 5  # We only validate every 5 epochs

            # Early stopping
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1} (patience={args.patience})")
                break
        else:
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"loss={avg_loss:.4f} train={train_acc:.3f} | {dt:.1f}s")

    print(f"\nBest validation accuracy: {best_val_acc:.3f} (epoch {best_epoch})")

    # Save final
    torch.save(model.state_dict(), ckpt_dir / "final_model.pt")

    # Save training config
    config = {
        "data_dir": args.data_dir,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "focal_gamma": args.focal_gamma,
        "unfreeze_layers": args.unfreeze_layers,
        "backbone_lr_factor": args.backbone_lr_factor,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "device": str(device),
        "has_prithvi": model.has_prithvi,
    }
    with open(ckpt_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {ckpt_dir / 'training_config.json'}")


def main():
    parser = argparse.ArgumentParser(description="Train Prithvi crop classifier")

    # Data
    parser.add_argument("--data-dir", type=str, default="data/crop_tiles")
    parser.add_argument("--patch-size", type=int, default=224)

    # Training (LULC-proven defaults)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=25)

    # Model
    parser.add_argument("--unfreeze-layers", type=int, default=6)
    parser.add_argument("--backbone-lr-factor", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", type=str, default=None)

    # Loss (focal, same as LULC)
    parser.add_argument("--focal-gamma", type=float, default=2.0)

    # Checkpoint
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/crop")

    # Evaluate only
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)

    args = parser.parse_args()

    if args.evaluate_only:
        raise NotImplementedError("Evaluate-only mode not yet implemented")

    train(args)


if __name__ == "__main__":
    main()
