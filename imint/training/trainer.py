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

    def _count_aux_channels(self) -> int:
        """Count how many auxiliary raster channels are enabled."""
        return len(self.config.enabled_aux_names)

    def _collect_aux(
        self,
        batch: dict,
        device: "torch.device",
    ) -> "torch.Tensor | None":
        """Stack enabled aux channels from a batch dict → (B, N, H, W).

        Uses config.enabled_aux_names for canonical channel ordering.
        """
        parts: list["torch.Tensor"] = []
        for name in self.config.enabled_aux_names:
            if name in batch:
                parts.append(batch[name].to(device))  # (B, 1, H, W)
        if not parts:
            return None
        return torch.cat(parts, dim=1)  # (B, N, H, W)

    def _build_model(self):
        """Build PrithviSegmentationModel using existing infrastructure."""
        from ..fm.terratorch_loader import _load_prithvi_from_hf
        from ..fm.upernet import PrithviSegmentationModel, get_default_pool_sizes

        num_frames = self.config.num_temporal_frames if self.config.enable_multitemporal else 1
        print(f"  Loading Prithvi backbone ({self.config.backbone}, "
              f"num_frames={num_frames})...")
        backbone = _load_prithvi_from_hf(pretrained=True, num_frames=num_frames)

        n_aux = self._count_aux_channels()

        model = PrithviSegmentationModel(
            encoder=backbone,
            feature_indices=self.config.feature_indices,
            decoder_channels=self.config.decoder_channels,
            num_classes=self.config.num_classes + 1,  # +1 for background at 0
            dropout=self.config.dropout,
            n_aux_channels=n_aux,
            pool_sizes=get_default_pool_sizes(self.device),
        )
        aux_str = f", aux={n_aux} channels" if n_aux else ""
        print(f"  Model built: {self.config.num_classes + 1} classes, "
              f"decoder={self.config.decoder_type}{aux_str}")
        return model

    def _freeze_backbone(self) -> None:
        """Freeze encoder, then selectively unfreeze last N transformer blocks."""
        # First freeze everything in encoder
        for param in self.model.encoder.parameters():
            param.requires_grad = False

        # Unfreeze last N blocks + final norm
        n_unfreeze = self.config.unfreeze_backbone_layers
        if n_unfreeze > 0 and hasattr(self.model.encoder, "blocks"):
            total_blocks = len(self.model.encoder.blocks)
            start_idx = total_blocks - n_unfreeze
            for i in range(start_idx, total_blocks):
                for param in self.model.encoder.blocks[i].parameters():
                    param.requires_grad = True
            # Also unfreeze final layernorm
            if hasattr(self.model.encoder, "norm"):
                for param in self.model.encoder.norm.parameters():
                    param.requires_grad = True
            print(f"  Unfreezing encoder blocks {start_idx}-{total_blocks-1} "
                  f"+ norm (lr_factor={self.config.backbone_lr_factor})")

        frozen = sum(p.numel() for p in self.model.encoder.parameters()
                     if not p.requires_grad)
        trainable_enc = sum(p.numel() for p in self.model.encoder.parameters()
                           if p.requires_grad)
        trainable_dec = sum(p.numel() for p in self.model.parameters()
                           if p.requires_grad) - trainable_enc
        print(f"  Frozen: {frozen:,} | Trainable encoder: {trainable_enc:,} "
              f"| Trainable decoder: {trainable_dec:,}")

    def _freeze_for_aux_training(self) -> None:
        """Stage 2: freeze backbone + decoder + head, train only AuxEncoder + aux_fusion."""
        for name, param in self.model.named_parameters():
            if "aux_encoder" in name or "aux_fusion" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        print(f"  Stage 2 (aux-only): Frozen: {frozen:,} | Trainable (aux): {trainable:,}")

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
            class_counts_19 = {int(k): v for k, v in stats.get("class_counts", {}).items()}

            # If using grouped classes, aggregate 19-class counts to grouped schema
            if cfg.use_grouped_classes:
                from .class_schema import _MAP_19_TO_10
                class_counts = {}
                for idx_19, count in class_counts_19.items():
                    idx_grouped = _MAP_19_TO_10.get(idx_19, 0)
                    class_counts[idx_grouped] = class_counts.get(idx_grouped, 0) + count
            else:
                class_counts = class_counts_19

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

        # Differential learning rate: backbone gets lower LR
        backbone_params = []
        decoder_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("encoder."):
                backbone_params.append(param)
            else:
                decoder_params.append(param)

        param_groups = [
            {"params": decoder_params, "lr": cfg.lr},
        ]
        if backbone_params:
            param_groups.append({
                "params": backbone_params,
                "lr": cfg.lr * cfg.backbone_lr_factor,
            })
            print(f"  Optimizer: decoder_lr={cfg.lr}, "
                  f"backbone_lr={cfg.lr * cfg.backbone_lr_factor}")
        else:
            print(f"  Optimizer: lr={cfg.lr}")

        optimizer = torch.optim.AdamW(
            param_groups,
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
        train_loss_history = []       # Track recent train losses for convergence
        step = 0
        start_epoch = 1

        print(f"\n{'='*60}")
        print(f"  Training: {cfg.epochs} epochs, batch_size={cfg.batch_size}, "
              f"lr={cfg.lr}, device={self.device}")
        print(f"{'='*60}\n")

        # ── Resume from checkpoint ────────────────────────────────
        resume_path = None
        if cfg.resume_from_checkpoint:
            resume_path = Path(cfg.resume_from_checkpoint)
        else:
            auto_path = ckpt_dir / "last_checkpoint.pt"
            if auto_path.exists():
                resume_path = auto_path

        if resume_path and resume_path.exists():
            if cfg.freeze_spectral:
                # Stage 2: load spectral model, keep aux layers random
                self._load_spectral_checkpoint(resume_path)
                self._freeze_for_aux_training()
                self._init_training_log()
            else:
                state = self._load_resume_checkpoint(
                    resume_path, optimizer, scheduler,
                )
                start_epoch = state["epoch"] + 1
                step = state["step"]
                best_metric = state["best_metric"]
                best_miou = state["best_miou"]
                best_epoch = state["best_epoch"]
                patience_counter = state["patience_counter"]
                train_loss_history = state.get("train_loss_history", [])
        else:
            if cfg.freeze_spectral:
                self._freeze_for_aux_training()
            self._init_training_log()

        for epoch in range(start_epoch, cfg.epochs + 1):
            # ── Train epoch ───────────────────────────────────────────
            self.model.train()
            # Keep encoder in eval mode (batchnorm/dropout frozen)
            self.model.encoder.eval()

            epoch_loss = 0.0
            n_batches = 0
            t0 = time.time()

            for batch in train_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)    # (B, H, W)

                # Reshape to 5D for Prithvi Conv3d: (B, C=6, T, H, W)
                B, CT, H, W = images.shape
                if CT > 6:
                    # Multitemporal: (B, T*6, H, W) → (B, 6, T, H, W)
                    T = CT // 6
                    images_5d = images.view(B, T, 6, H, W).permute(0, 2, 1, 3, 4)
                else:
                    # Single-date: (B, 6, H, W) → (B, 6, 1, H, W)
                    images_5d = images.unsqueeze(2)

                # Collect auxiliary channels (height/volume/etc.)
                aux = self._collect_aux(batch, self.device)

                # Warmup: linear LR ramp
                if step < warmup_steps:
                    lr_scale = (step + 1) / warmup_steps
                    for pg in optimizer.param_groups:
                        pg["lr"] = cfg.lr * lr_scale

                optimizer.zero_grad()
                logits = self.model(images_5d, aux=aux).contiguous()
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

            # ── System metrics ────────────────────────────────────────
            self._write_system_metrics()

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

            # ── Save resume checkpoint (every epoch) ──────────────
            self._save_resume_checkpoint(
                ckpt_dir / "last_checkpoint.pt",
                epoch=epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                best_metric=best_metric,
                best_miou=best_miou,
                best_epoch=best_epoch,
                patience_counter=patience_counter,
                train_loss_history=train_loss_history,
            )

            # Track train loss convergence
            train_loss_history.append(avg_loss)

            # Early stopping: val metric patience OR train loss converged
            val_stop = patience_counter >= cfg.early_stopping_patience
            train_stop = False
            if len(train_loss_history) >= cfg.train_loss_patience:
                recent = train_loss_history[-cfg.train_loss_patience:]
                loss_delta = max(recent) - min(recent)
                if loss_delta < cfg.train_loss_min_delta:
                    train_stop = True

            if val_stop or train_stop:
                reason = ("train loss converged" if train_stop
                          else "val metric plateau")
                print(f"\n  Early stopping at epoch {epoch}: {reason} "
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

    def _write_system_metrics(self) -> None:
        """Write system_metrics.json with system-wide CPU/RAM and GPU usage."""
        try:
            import psutil
        except ImportError:
            return

        vm = psutil.virtual_memory()
        metrics = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": round(vm.percent, 1),
            "memory_used_gb": round(vm.used / (1024**3), 2),
            "memory_total_gb": round(vm.total / (1024**3), 1),
            "device": str(self.device),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        # GPU usage (MPS doesn't expose util, but we can get memory)
        if self.device.type == "cuda":
            import torch
            try:
                metrics["gpu_percent"] = torch.cuda.utilization(self.device)
            except (ModuleNotFoundError, RuntimeError):
                metrics["gpu_percent"] = None  # pynvml not available
            metrics["gpu_memory_used_gb"] = round(
                torch.cuda.memory_allocated(self.device) / (1024**3), 1)
            props = torch.cuda.get_device_properties(self.device)
            total = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
            metrics["gpu_memory_total_gb"] = round(total / (1024**3), 1)
        elif self.device.type == "mps":
            import torch
            metrics["gpu_percent"] = None  # Not exposed by MPS
            metrics["gpu_memory_used_gb"] = round(
                torch.mps.current_allocated_memory() / (1024**3), 2)
            metrics["gpu_memory_total_gb"] = None
        else:
            metrics["gpu_percent"] = None
            metrics["gpu_memory_used_gb"] = None
            metrics["gpu_memory_total_gb"] = None

        # Network delta since pipeline start
        if not hasattr(self, '_net_baseline'):
            self._net_baseline = psutil.net_io_counters()
        net = psutil.net_io_counters()
        metrics["net_sent_mb"] = round(
            (net.bytes_sent - self._net_baseline.bytes_sent) / (1024**2), 1)
        metrics["net_recv_mb"] = round(
            (net.bytes_recv - self._net_baseline.bytes_recv) / (1024**2), 1)

        try:
            path = Path(self.config.data_dir) / "system_metrics.json"
            tmp = path.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump(metrics, f, indent=2)
            tmp.rename(path)
        except Exception:
            pass

    # ── Checkpoint ────────────────────────────────────────────────────

    def _save_resume_checkpoint(
        self,
        path: Path,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        step: int,
        best_metric: float,
        best_miou: float,
        best_epoch: int,
        patience_counter: int,
        train_loss_history: list[float] | None = None,
    ) -> None:
        """Save full training state for resumption after interruption."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": step,
            "best_metric": best_metric,
            "best_miou": best_miou,
            "best_epoch": best_epoch,
            "patience_counter": patience_counter,
            "training_log": self._training_log,
            "train_loss_history": train_loss_history or [],
        }
        tmp = path.with_suffix(".pt.tmp")
        torch.save(checkpoint, tmp)
        tmp.rename(path)

    def _load_spectral_checkpoint(self, path: Path) -> None:
        """Load spectral-only checkpoint for stage-2 aux training.

        Loads encoder + decoder + head weights from a best_model.pt
        checkpoint. Missing aux_encoder/aux_fusion keys are expected
        and left at their random initialization.
        """
        import torch
        print(f"  Loading spectral checkpoint for stage 2: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        # Extract state dict (handles both best_model.pt and resume format)
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                raw_sd = ckpt["state_dict"]
            elif "model_state_dict" in ckpt:
                raw_sd = ckpt["model_state_dict"]
            else:
                raw_sd = ckpt
        else:
            raw_sd = ckpt

        # Strip "model." prefix from Lightning-format checkpoints
        sd = {}
        for k, v in raw_sd.items():
            key = k[len("model."):] if k.startswith("model.") else k
            sd[key] = v

        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        # aux_encoder and aux_fusion are expected to be missing
        real_missing = [k for k in missing
                        if "aux_encoder" not in k and "aux_fusion" not in k]
        if real_missing:
            print(f"  WARNING: {len(real_missing)} unexpected missing keys:")
            for k in real_missing[:10]:
                print(f"    {k}")
        aux_missing = [k for k in missing
                       if "aux_encoder" in k or "aux_fusion" in k]
        if aux_missing:
            print(f"  Aux layers initialized randomly ({len(aux_missing)} params)")
        print(f"  Spectral model loaded successfully")

    def _load_resume_checkpoint(
        self,
        path: Path,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
    ) -> dict:
        """Load full training state from a resume checkpoint.

        Returns:
            Dict with keys: epoch, step, best_metric, best_miou,
            best_epoch, patience_counter.
        """
        print(f"  Resuming from checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "training_log" in ckpt:
            self._training_log = ckpt["training_log"]
        print(f"  Resumed at epoch {ckpt['epoch']} "
              f"(step={ckpt['step']}, best_mIoU={ckpt['best_miou']:.4f})")
        return {
            "epoch": ckpt["epoch"],
            "step": ckpt["step"],
            "best_metric": ckpt["best_metric"],
            "best_miou": ckpt["best_miou"],
            "best_epoch": ckpt["best_epoch"],
            "patience_counter": ckpt["patience_counter"],
            "train_loss_history": ckpt.get("train_loss_history", []),
        }

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
                "n_aux_channels": self._count_aux_channels(),
            },
        }
        torch.save(checkpoint, path)
