"""
imint/training/trainer.py — LULC training loop

Trains a UPerNet decoder on frozen Prithvi-EO-2.0 backbone using
NMD-derived LULC labels.  Produces checkpoints compatible with
the existing TASK_HEAD_REGISTRY / load_segmentation_model() pipeline.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except ImportError:
    raise ImportError("PyTorch is required. Install with: pip install torch")

from .config import TrainingConfig
from .class_schema import compute_class_weights
from .dataset import LULCDataset, build_weighted_sampler
from .evaluate import evaluate_model


class LULCTrainer:
    """Trains a Prithvi + UPerNet model for multi-class LULC segmentation.

    The Prithvi backbone is frozen.  Only decoder and head parameters
    are optimised.  Checkpoints are saved in Lightning-convention format
    (``model.`` prefix) so they load directly via
    ``load_segmentation_model()`` in ``terratorch_loader.py``.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._resolve_device()
        self.model = self._build_model()
        self._freeze_backbone()
        self.model.to(self.device)
        self._training_log = {
            "config": {},
            "epochs": [],
            "best_epoch": 0,
            "best_metric": 0.0,
            "status": "initialized",
            "started_at": None,
            "updated_at": None,
        }

    # ── Model setup ───────────────────────────────────────────────────

    def _resolve_device(self) -> torch.device:
        if self.config.device:
            return torch.device(self.config.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _build_model(self):
        """Build PrithviSegmentationModel using existing infrastructure."""
        from ..fm.terratorch_loader import _load_prithvi_from_hf
        from ..fm.upernet import PrithviSegmentationModel

        print(f"  Loading Prithvi backbone ({self.config.backbone})...")
        backbone = _load_prithvi_from_hf(pretrained=True)

        model = PrithviSegmentationModel(
            encoder=backbone,
            feature_indices=self.config.feature_indices,
            decoder_channels=self.config.decoder_channels,
            num_classes=self.config.num_classes + 1,  # +1 for background at 0
            dropout=self.config.dropout,
        )
        print(f"  Model built: {self.config.num_classes + 1} classes, "
              f"decoder={self.config.decoder_type}")
        return model

    def _freeze_backbone(self) -> None:
        """Freeze all encoder parameters."""
        frozen = 0
        for param in self.model.encoder.parameters():
            param.requires_grad = False
            frozen += param.numel()

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Frozen: {frozen:,} params | Trainable: {trainable:,} params")

    # ── Training ──────────────────────────────────────────────────────

    def train(
        self,
        train_dataset: LULCDataset,
        val_dataset: LULCDataset,
    ) -> dict:
        """Full training loop with validation and early stopping.

        Args:
            train_dataset: Training LULCDataset.
            val_dataset: Validation LULCDataset.

        Returns:
            Dict with best metrics and checkpoint path.
        """
        cfg = self.config
        ckpt_dir = Path(cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Class weights
        stats_path = Path(cfg.data_dir) / "class_stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            class_counts = {int(k): v for k, v in stats.get("class_counts", {}).items()}
            weights = compute_class_weights(
                class_counts,
                num_classes=cfg.num_classes,
                max_weight=cfg.max_class_weight,
                ignore_index=cfg.ignore_index,
            )
            weights_tensor = torch.from_numpy(weights).to(self.device)
            print(f"  Class weights: {weights.round(2).tolist()}")
        else:
            weights_tensor = None
            print("  WARNING: No class_stats.json — using uniform weights")

        # Loss function
        if cfg.loss_type == "focal":
            from .losses import FocalLoss
            criterion = FocalLoss(
                weight=weights_tensor,
                gamma=cfg.focal_gamma,
                ignore_index=cfg.ignore_index,
            )
            print(f"  Loss: Focal (gamma={cfg.focal_gamma})")
        else:
            criterion = nn.CrossEntropyLoss(
                weight=weights_tensor,
                ignore_index=cfg.ignore_index,
            )
            print(f"  Loss: CrossEntropy")

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        total_steps = cfg.epochs * math.ceil(len(train_dataset) / cfg.batch_size)
        warmup_steps = int(total_steps * cfg.warmup_fraction)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps,
        )

        # DataLoaders
        sampler = build_weighted_sampler(
            train_dataset,
            class_stats_path=str(stats_path) if stats_path.exists() else None,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )

        # Training loop
        best_metric = 0.0
        best_miou = 0.0
        best_epoch = 0
        patience_counter = 0
        step = 0

        print(f"\n{'='*60}")
        print(f"  Training: {cfg.epochs} epochs, batch_size={cfg.batch_size}, "
              f"lr={cfg.lr}, device={self.device}")
        print(f"{'='*60}\n")

        self._init_training_log()

        for epoch in range(1, cfg.epochs + 1):
            # ── Train epoch ───────────────────────────────────────────
            self.model.train()
            # Keep encoder in eval mode (batchnorm/dropout frozen)
            self.model.encoder.eval()

            epoch_loss = 0.0
            n_batches = 0
            t0 = time.time()

            for batch in train_loader:
                images = batch["image"].to(self.device)   # (B, 6, H, W)
                labels = batch["label"].to(self.device)    # (B, H, W)

                # Add temporal dim: (B, 6, H, W) → (B, 6, 1, H, W)
                images_5d = images.unsqueeze(2)

                # Warmup: linear LR ramp
                if step < warmup_steps:
                    lr_scale = (step + 1) / warmup_steps
                    for pg in optimizer.param_groups:
                        pg["lr"] = cfg.lr * lr_scale

                optimizer.zero_grad()
                logits = self.model(images_5d)  # (B, C, H, W)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                if step >= warmup_steps:
                    scheduler.step()

                epoch_loss += loss.item()
                n_batches += 1
                step += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            elapsed = time.time() - t0

            # ── Validate ──────────────────────────────────────────────
            val_metrics = evaluate_model(
                self.model, val_dataset, cfg, self.device,
            )
            val_miou = val_metrics["miou"]
            per_class = val_metrics["per_class_iou"]

            # Compute worst-class IoU
            valid_ious = {k: v for k, v in per_class.items()
                         if not math.isnan(v)}
            if valid_ious:
                worst_class = min(valid_ious, key=valid_ious.get)
                worst_iou = valid_ious[worst_class]
            else:
                worst_class = "N/A"
                worst_iou = 0.0

            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{cfg.epochs} | "
                  f"loss={avg_loss:.4f} | val_mIoU={val_miou:.4f} | "
                  f"worst={worst_iou:.4f} ({worst_class}) | "
                  f"lr={lr_now:.2e} | {elapsed:.0f}s")

            # ── Early stop metric ─────────────────────────────────────
            if cfg.early_stop_metric == "worst_class_iou":
                metric_value = worst_iou
            elif cfg.early_stop_metric == "combined":
                w = cfg.worst_class_weight
                metric_value = (1 - w) * val_miou + w * worst_iou
            else:
                metric_value = val_miou

            # ── Log epoch to JSON ─────────────────────────────────────
            is_new_best = metric_value > best_metric
            self._update_training_log({
                "epoch": epoch,
                "train_loss": round(avg_loss, 6),
                "val_miou": round(val_miou, 6),
                "per_class_iou": {k: round(v, 6) for k, v in per_class.items()
                                  if not math.isnan(v)},
                "worst_class": worst_class,
                "worst_class_iou": round(worst_iou, 6),
                "lr": lr_now,
                "elapsed_s": round(elapsed, 1),
                "metric_value": round(metric_value, 6),
                "is_best": is_new_best,
            })

            # ── Checkpointing ─────────────────────────────────────────
            if is_new_best:
                best_metric = metric_value
                best_miou = val_miou
                best_epoch = epoch
                patience_counter = 0
                self._save_checkpoint(
                    ckpt_dir / "best_model.pt", epoch, val_metrics,
                )
                print(f"    ✓ New best model saved "
                      f"(mIoU={val_miou:.4f}, metric={metric_value:.4f})")
            else:
                patience_counter += 1

            # Per-class IoU table on new best or periodically
            if is_new_best or epoch % cfg.save_every_n_epochs == 0:
                print(f"    Per-class IoU:")
                for name, iou in sorted(valid_ious.items(), key=lambda x: x[1]):
                    bar = "#" * int(iou * 20)
                    print(f"      {name:35s} {iou:.4f}  {bar}")

            if epoch % cfg.save_every_n_epochs == 0:
                self._save_checkpoint(
                    ckpt_dir / f"epoch_{epoch:03d}.pt", epoch, val_metrics,
                )

            # Early stopping
            if patience_counter >= cfg.early_stopping_patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(best={best_epoch}, mIoU={best_miou:.4f})")
                self._training_log["status"] = "stopped"
                self._write_log_file()
                break
        else:
            self._training_log["status"] = "completed"
            self._write_log_file()

        print(f"\n  Training complete. Best mIoU={best_miou:.4f} at epoch {best_epoch}")
        return {
            "best_miou": best_miou,
            "best_epoch": best_epoch,
            "checkpoint": str(ckpt_dir / "best_model.pt"),
        }

    # ── Training log (JSON for dashboard) ─────────────────────────────

    def _init_training_log(self) -> None:
        """Write initial training_log.json before the training loop."""
        cfg = self.config
        self._training_log["config"] = {
            k: v for k, v in asdict(cfg).items()
            if not isinstance(v, (bytes,))
        }
        self._training_log["status"] = "running"
        self._training_log["started_at"] = datetime.now(timezone.utc).isoformat()
        self._write_log_file()

    def _update_training_log(self, epoch_data: dict) -> None:
        """Append epoch metrics and write training_log.json."""
        self._training_log["epochs"].append(epoch_data)
        if epoch_data.get("is_best"):
            self._training_log["best_epoch"] = epoch_data["epoch"]
            self._training_log["best_metric"] = epoch_data["metric_value"]
        self._training_log["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write_log_file()

    def _write_log_file(self) -> None:
        """Atomic write of training_log.json (tmp + rename)."""
        try:
            log_path = Path(self.config.data_dir) / "training_log.json"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = log_path.with_suffix(".json.tmp")
            with open(tmp_path, "w") as f:
                json.dump(self._training_log, f, indent=2, default=str)
            tmp_path.rename(log_path)
        except Exception as e:
            print(f"    WARNING: Failed to write training log: {e}")

    # ── Checkpoint ────────────────────────────────────────────────────

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: dict,
    ) -> None:
        """Save checkpoint in TASK_HEAD_REGISTRY-compatible format.

        Keys are prefixed with ``model.`` to match the Lightning convention
        expected by ``_map_checkpoint_keys()`` in terratorch_loader.py.
        """
        state_dict = {}
        for key, value in self.model.state_dict().items():
            state_dict[f"model.{key}"] = value

        checkpoint = {
            "state_dict": state_dict,
            "epoch": epoch,
            "metrics": {k: v for k, v in metrics.items()
                        if k != "confusion_matrix"},
            "config": {
                "num_classes": self.config.num_classes + 1,
                "decoder_type": self.config.decoder_type,
                "decoder_channels": self.config.decoder_channels,
                "feature_indices": self.config.feature_indices,
                "dropout": self.config.dropout,
            },
        }
        torch.save(checkpoint, path)
