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


# ── Safe collate (avoids worker-side shared-memory resize crash) ───────────
#
# PyTorch's default_collate(), when called inside a DataLoader worker process,
# pre-allocates the stacked output tensor in shared memory via:
#
#   storage = elem.storage()._new_shared(numel, device=elem.device)
#   out = elem.new(storage).resize_(len(batch), *list(elem.size()))
#
# On some PyTorch/Linux combinations (notably PyTorch 2.x + Python 3.11 on
# CUDA workers) the mmap-backed shared storage is fixed-size and rejects
# resize_(), raising:
#   RuntimeError: Trying to resize storage that is not resizable
#
# The fix is a custom collate_fn that never calls _new_shared() + resize_().
# It produces identical results to default_collate but always stacks into a
# freshly-allocated tensor.  The tiny extra copy cost is irrelevant at this
# batch size (32×32 patches ≈ 62 MB/batch).
#
def _safe_collate(batch):
    """Like default_collate but skips worker-side shared-memory pre-allocation."""
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        return torch.stack(batch, 0)
    if isinstance(elem, (tuple, list)):
        transposed = list(zip(*batch))
        return type(elem)(_safe_collate(s) for s in transposed)
    # scalars / numpy arrays — fall through to default_collate
    from torch.utils.data.dataloader import default_collate
    return default_collate(batch)

from imint.training.pixel_dataset import PixelContextDataset, TileGroupSampler, N_AUX
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

    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for tile-group shuffle sampler (default: 42)")

    # Performance
    p.add_argument("--amp", action="store_true", default=False,
                   help="Automatic mixed precision (bfloat16) on CUDA — 2-3× speedup on H100")
    p.add_argument("--compile", action="store_true", default=False,
                   help="torch.compile() the model for ~20%% speedup (PyTorch 2.x, adds ~1 min startup)")
    p.add_argument("--log-file", default=None,
                   help="Path to write JSON training log (per-epoch metrics + final summary). "
                        "Written atomically after each epoch. "
                        "E.g. /checkpoints/pixel_v1/train_log.json")
    p.add_argument("--tile-pred-n", type=int, default=0,
                   help="Save colorized prediction/GT images for N representative val tiles "
                        "after every epoch, for dashboard visualization (0 = disabled).")

    return p


# ── Metrics ───────────────────────────────────────────────────────────────

def compute_metrics(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    num_classes: int = NUM_UNIFIED_CLASSES,
    ignore_index: int = 0,
) -> dict:
    """Compute per-class accuracy, per-class IoU, mAcc, mIoU, and OA.

    Uses a confusion matrix for O(N) computation.
    Background class (ignore_index=0) is excluded from mean metrics.

    Returns keys:
        mean_acc      – mean per-class recall  (mAcc)
        mean_iou      – mean per-class IoU     (mIoU)  ← primary metric
        overall_acc   – pixel-level accuracy   (OA)
        per_class_acc – {class_name: float}
        per_class_iou – {class_name: float}
        per_class_n   – {class_name: int}  (true pixels per class)
    """
    # Exclude background and any out-of-range predictions
    valid = (
        (all_labels != ignore_index)
        & (all_labels >= 0) & (all_labels < num_classes)
        & (all_preds  >= 0) & (all_preds  < num_classes)
    )
    v_labels = all_labels[valid].astype(np.int64)
    v_preds  = all_preds[valid].astype(np.int64)

    # Confusion matrix — shape (num_classes, num_classes), rows=true, cols=pred
    confusion = np.bincount(
        num_classes * v_labels + v_preds,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)

    tp = np.diag(confusion)                      # TP[c]
    fn = confusion.sum(axis=1) - tp              # FN[c]
    fp = confusion.sum(axis=0) - tp              # FP[c]
    per_class_total = (tp + fn).astype(np.int64) # true pixels per class

    # Per-class accuracy = recall = TP / (TP + FN)
    per_class_acc = np.where(
        per_class_total > 0,
        tp / np.maximum(per_class_total, 1).astype(np.float64),
        np.nan,
    )

    # Per-class IoU = TP / (TP + FP + FN)
    denom_iou = (tp + fp + fn).astype(np.float64)
    per_class_iou = np.where(
        denom_iou > 0,
        tp / np.maximum(denom_iou, 1),
        np.nan,
    )

    # Mean metrics (exclude background class 0)
    valid_acc = ~np.isnan(per_class_acc[1:])
    mean_acc = float(per_class_acc[1:][valid_acc].mean()) if valid_acc.any() else 0.0

    valid_iou = ~np.isnan(per_class_iou[1:])
    mean_iou = float(per_class_iou[1:][valid_iou].mean()) if valid_iou.any() else 0.0

    # Overall accuracy (fraction of correctly classified valid pixels)
    overall_acc = float(tp[1:].sum() / max(int(valid.sum()), 1))

    return {
        "mean_acc": mean_acc,
        "mean_iou": mean_iou,
        "overall_acc": overall_acc,
        "per_class_acc": {
            UNIFIED_CLASS_NAMES[i]: float(per_class_acc[i])
            for i in range(1, num_classes)
            if not np.isnan(per_class_acc[i])
        },
        "per_class_iou": {
            UNIFIED_CLASS_NAMES[i]: float(per_class_iou[i])
            for i in range(1, num_classes)
            if not np.isnan(per_class_iou[i])
        },
        "per_class_n": {
            UNIFIED_CLASS_NAMES[i]: int(per_class_total[i])
            for i in range(1, num_classes)
        },
    }


# ── Training log helper ───────────────────────────────────────────────────

def _write_training_log(
    log_path: Path,
    config: dict,
    epochs: list[dict],
    summary: dict | None = None,
) -> None:
    """Atomically write training log JSON (tmp-rename so never half-written)."""
    payload: dict = {"config": config, "epochs": epochs}
    if summary is not None:
        payload["summary"] = summary
    tmp = log_path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    tmp.replace(log_path)


# ── Training ──────────────────────────────────────────────────────────────

def _run_epoch(
    model: PrithviPixelClassifier,
    loader: DataLoader,
    criterion,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
    use_aux: bool = True,
    log_every: int = 200,
    use_amp: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run one epoch. Returns (loss, preds, labels).

    When ``use_aux=True`` the DataLoader is expected to yield 3-tuples
    ``(patches, aux_vec, labels)``; otherwise 2-tuples ``(patches, labels)``.
    Prints a one-line progress update every ``log_every`` batches so we
    can see within-epoch activity on long first epochs (cold NFS I/O).

    When ``use_amp=True`` the forward pass runs under ``torch.autocast``
    with ``bfloat16``.  BF16 does NOT require a GradScaler (unlike fp16),
    so gradients are accumulated and applied in fp32.
    """
    import contextlib
    amp_ctx: contextlib.AbstractContextManager = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if use_amp and device.type == "cuda"
        else contextlib.nullcontext()
    )

    model.train(train)
    total_loss = 0.0
    all_preds, all_labels = [], []
    n_batches = len(loader)
    t0 = time.time()

    with torch.set_grad_enabled(train):
        for step, batch in enumerate(loader):
            if use_aux:
                patches, aux_vec, labels = batch
                aux_vec = aux_vec.to(device, non_blocking=True)
            else:
                patches, labels = batch
                aux_vec = None

            patches = patches.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with amp_ctx:
                logits = model(patches, aux_vec)  # (B, num_classes)
                loss = criterion(logits, labels)

            # Backward and optimiser step stay in fp32 (bf16 needs no scaler)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

            if log_every > 0 and (step + 1) % log_every == 0:
                elapsed = time.time() - t0
                ms_per_batch = elapsed / (step + 1) * 1000
                eta = (n_batches - step - 1) * elapsed / (step + 1)
                running_loss = total_loss / max(
                    sum(len(l) for l in all_labels), 1
                )
                phase = "train" if train else "val"
                print(
                    f"    [{phase} {step+1:5d}/{n_batches}]"
                    f"  loss={running_loss:.4f}"
                    f"  {ms_per_batch:.0f}ms/batch"
                    f"  ETA {eta:.0f}s",
                    flush=True,
                )

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


# ── Tile prediction visualization ─────────────────────────────────────────

# RGB color per unified class index (0=background, 1-22=land cover/crop/hygge)
_CLASS_COLORS_RGB: list[tuple[int, int, int]] = [
    (  0,   0,   0),   #  0 background
    ( 22, 101,  52),   #  1 tallskog
    ( 21, 128,  61),   #  2 granskog
    ( 74, 222, 128),   #  3 lövskog
    ( 52, 211, 153),   #  4 blandskog
    ( 19,  78,  74),   #  5 sumpskog
    (163, 230,  53),   #  6 tillfälligt ej skog
    (133,  77,  14),   #  7 våtmark
    (217, 119,   6),   #  8 öppen mark
    (107, 114, 128),   #  9 bebyggelse
    ( 59, 130, 246),   # 10 vatten
    (251, 191,  36),   # 11 vete
    (245, 158,  11),   # 12 korn
    (253, 224,  71),   # 13 havre
    (234, 179,   8),   # 14 oljeväxter
    (132, 204,  22),   # 15 slåttervall
    (134, 239, 172),   # 16 bete
    (168,  85, 247),   # 17 potatis
    (236,  72, 153),   # 18 sockerbetor
    (249, 115,  22),   # 19 trindsäd
    (220,  38,  38),   # 20 råg
    (234,  88,  12),   # 21 övrig åker
    (  6, 182, 212),   # 22 hygge
]


def _colorize_classmap(class_map: np.ndarray) -> np.ndarray:
    """Map (H, W) int class indices → (H, W, 3) uint8 RGB."""
    H, W = class_map.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(_CLASS_COLORS_RGB):
        mask = class_map == cls_idx
        if mask.any():
            rgb[mask] = color
    return rgb


def _infer_tile_grid(
    model,
    tile_path: Path,
    context_px: int,
    use_frame_2016: bool,
    use_aux: bool,
    device: torch.device,
    stride: int = 4,
    batch_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Grid-sampled inference on a full tile.

    Samples every ``stride``-th pixel as a center pixel, extracts its
    context patch, runs inference, and reconstructs a spatial prediction map.

    Returns:
        pred_map: (grid_h, grid_w) int32 — predicted class indices
        gt_map:   (grid_h, grid_w) int32 — ground-truth labels
    """
    import contextlib
    from imint.training.unified_dataset import AUX_CHANNEL_NAMES, AUX_LOG_TRANSFORM, AUX_NORM

    data = np.load(str(tile_path), allow_pickle=False)
    spectral = np.asarray(data.get("spectral", data.get("image")), dtype=np.float32)
    label = np.asarray(data["label"], dtype=np.int32)
    H, W = label.shape
    half = context_px // 2

    rows = np.arange(half, H - half, stride)
    cols = np.arange(half, W - half, stride)
    grid_h, grid_w = len(rows), len(cols)
    N = grid_h * grid_w

    # Build all patches at once
    patches = np.empty((N, (5 if use_frame_2016 else 4) * 6, context_px, context_px), dtype=np.float32)
    has_2016 = use_frame_2016 and int(data.get("has_frame_2016", 0)) == 1
    if has_2016:
        frame_2016 = np.asarray(data["frame_2016"], dtype=np.float32)

    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            r0, r1, c0, c1 = r - half, r + half, c - half, c + half
            patch_base = spectral[:, r0:r1, c0:c1]
            if use_frame_2016:
                if has_2016:
                    patch_bg = frame_2016[:, r0:r1, c0:c1]
                else:
                    patch_bg = np.zeros((6, context_px, context_px), dtype=np.float32)
                patches[i * grid_w + j] = np.concatenate([patch_bg, patch_base], axis=0)
            else:
                patches[i * grid_w + j] = patch_base

    # Build aux vectors
    aux_all = None
    if use_aux:
        from imint.training.pixel_dataset import N_AUX
        aux_all = np.zeros((N, N_AUX), dtype=np.float32)
        for k, ch_name in enumerate(AUX_CHANNEL_NAMES):
            if ch_name not in data:
                continue
            arr = np.asarray(data[ch_name], dtype=np.float32)
            for i, r in enumerate(rows):
                for j, c in enumerate(cols):
                    val = float(arr[min(r, arr.shape[0]-1), min(c, arr.shape[1]-1)])
                    if AUX_LOG_TRANSFORM.get(ch_name, False):
                        val = float(np.log1p(val))
                    mu, sigma = AUX_NORM.get(ch_name, (0.0, 1.0))
                    aux_all[i * grid_w + j, k] = (val - mu) / max(sigma, 1e-8)

    # Inference in batches
    model.eval()
    preds_all = []
    amp_ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
               if device.type == "cuda" else contextlib.nullcontext())
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            bp = torch.tensor(patches[start:end], dtype=torch.float32, device=device)
            ba = (torch.tensor(aux_all[start:end], dtype=torch.float32, device=device)
                  if aux_all is not None else None)
            with amp_ctx:
                logits = model(bp, ba)
            preds_all.append(logits.argmax(dim=1).cpu().numpy())

    pred_flat = np.concatenate(preds_all)
    gt_flat = label[rows[:, None], cols[None, :]].flatten()
    return pred_flat.reshape(grid_h, grid_w).astype(np.int32), \
           gt_flat.reshape(grid_h, grid_w).astype(np.int32)


def _select_representative_tiles(val_tiles: list[Path], n: int) -> list[Path]:
    """Pick up to ``n`` visually diverse tiles from the val set.

    Strategy: scan tiles for dominant class and select the most class-diverse
    subset.  Falls back to the first ``n`` tiles if labels aren't readable.
    """
    scored: list[tuple[int, Path]] = []  # (dominant_class, path)
    for p in val_tiles:
        try:
            d = np.load(str(p), allow_pickle=False)
            lbl = np.asarray(d["label"], dtype=np.int32)
            counts = np.bincount(lbl.ravel(), minlength=23)
            counts[0] = 0  # ignore background
            dominant = int(counts.argmax()) if counts.max() > 0 else -1
            scored.append((dominant, p))
        except Exception:
            continue

    if not scored:
        return val_tiles[:n]

    # One tile per dominant class (to maximize diversity), then fill remainder
    seen_classes: set[int] = set()
    selected: list[Path] = []
    for cls, path in sorted(scored, key=lambda x: x[0]):
        if cls not in seen_classes:
            selected.append(path)
            seen_classes.add(cls)
        if len(selected) >= n:
            break

    # If we still need more tiles, fill with remaining
    if len(selected) < n:
        used = set(selected)
        for _, p in scored:
            if p not in used:
                selected.append(p)
            if len(selected) >= n:
                break

    return selected[:n]


def _save_tile_preds_epoch(
    model,
    pred_tiles: list[Path],
    epoch: int,
    ckpt_dir: Path,
    context_px: int,
    use_frame_2016: bool,
    use_aux: bool,
    device: torch.device,
) -> None:
    """Save colorized prediction PNGs for ``pred_tiles`` at this epoch.

    Directory layout::
        {ckpt_dir}/tile_preds/
            manifest.json
            gt/
                {tile_stem}_gt.png      ← saved once (epoch==1)
            epoch_{N}/
                {tile_stem}_pred.png    ← saved every epoch
    """
    from PIL import Image

    preds_root = ckpt_dir / "tile_preds"
    gt_dir     = preds_root / "gt"
    ep_dir     = preds_root / f"epoch_{epoch}"
    gt_dir.mkdir(parents=True, exist_ok=True)
    ep_dir.mkdir(parents=True, exist_ok=True)

    tile_entries = []
    for tile_path in pred_tiles:
        stem = tile_path.stem
        short = stem[:28] + "…" if len(stem) > 28 else stem

        try:
            pred_map, gt_map = _infer_tile_grid(
                model, tile_path, context_px,
                use_frame_2016, use_aux, device,
            )
        except Exception as e:
            print(f"    [tile_preds] {stem}: skipped ({e})", flush=True)
            continue

        # Save prediction image for this epoch
        pred_rgb = _colorize_classmap(pred_map)
        img_size = max(64, pred_map.shape[0] * 4)  # upscale for visibility
        Image.fromarray(pred_rgb).resize((img_size, img_size), Image.NEAREST).save(
            ep_dir / f"{stem}_pred.png"
        )

        # Save ground truth once (stable across epochs)
        gt_png = gt_dir / f"{stem}_gt.png"
        if not gt_png.exists():
            gt_rgb = _colorize_classmap(gt_map)
            Image.fromarray(gt_rgb).resize((img_size, img_size), Image.NEAREST).save(gt_png)

        tile_entries.append({"name": stem, "short": short})

    # Update manifest atomically
    manifest_path = preds_root / "manifest.json"
    manifest: dict = {"tiles": [], "epochs": []}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            pass

    manifest["tiles"] = tile_entries
    if epoch not in manifest["epochs"]:
        manifest["epochs"].append(epoch)
        manifest["epochs"].sort()

    tmp = manifest_path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    tmp.replace(manifest_path)
    print(f"    [tile_preds] epoch {epoch}: saved {len(tile_entries)} tiles → {ep_dir}", flush=True)


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
    # Exclude *.npz.tmp.npz leftover files from interrupted fetch jobs — these
    # share the *.npz suffix but are incomplete/duplicate tiles that would
    # double-count any tile that has both a real and a .tmp version.
    all_tiles = sorted(p for p in data_dir.glob("*.npz") if ".tmp" not in p.name)
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

    # Select representative val tiles for per-epoch prediction visualization
    pred_tiles: list[Path] = []
    if args.tile_pred_n > 0 and val_tiles:
        pred_tiles = _select_representative_tiles(val_tiles, args.tile_pred_n)
        print(f"  Tile predictions: {len(pred_tiles)} representative val tiles selected")

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

    # cuDNN auto-tuner: safe because all patches are the same size (context_px²)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if args.compile and hasattr(torch, "compile") and device.type == "cuda":
        print("  Compiling model with torch.compile(mode='reduce-overhead') …")
        model = torch.compile(model, mode="reduce-overhead")
        print("  Compile done.")

    # ── Resume ────────────────────────────────────────────────────
    start_epoch = 0
    best_val_acc_resume = 0.0
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=True)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_acc_resume = float(ckpt.get("val_mean_iou", 0.0))
        print(f"  Resumed from {args.resume_from} "
              f"(start_epoch={start_epoch}, best_val_mIoU_so_far={best_val_acc_resume:.4f})")

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
            eval_ds, batch_size=512, num_workers=4,
            pin_memory=(device.type == "cuda"),
            collate_fn=_safe_collate,
        )
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        _, preds, labels = _run_epoch(
            model, loader, criterion, None, device,
            train=False, use_aux=args.use_aux, use_amp=args.amp,
        )
        metrics = compute_metrics(preds, labels)
        print(f"  mIoU          : {metrics['mean_iou']:.4f}")
        print(f"  mAcc          : {metrics['mean_acc']:.4f}")
        print(f"  Overall Acc   : {metrics['overall_acc']:.4f}")
        print(f"  Per-class IoU / Acc:")
        for name in metrics["per_class_iou"]:
            iou = metrics["per_class_iou"][name]
            acc = metrics["per_class_acc"].get(name, float("nan"))
            n   = metrics["per_class_n"].get(name, 0)
            print(f"    {name:22s}  IoU={iou:.3f}  Acc={acc:.3f}  (n={n:,})")
        return

    # ── Class weights (inverse-frequency from training labels) ────
    print("\n  Computing class weights …")
    class_counts = np.zeros(NUM_UNIFIED_CLASSES, dtype=np.int64)
    for _, _, _, cls in train_ds._index:
        if 0 <= cls < NUM_UNIFIED_CLASSES:
            class_counts[cls] += 1
    class_weights_np = get_class_weights(
        {i: int(class_counts[i]) for i in range(NUM_UNIFIED_CLASSES)}
    )
    class_weights = torch.tensor(
        class_weights_np, dtype=torch.float32, device=device
    )

    criterion = FocalLoss(
        weight=class_weights,
        gamma=args.focal_gamma,
        ignore_index=0,
    )

    # ── DataLoaders ───────────────────────────────────────────────
    # TileGroupSampler shuffles at tile granularity (not sample level).
    # The dataset index is sorted by tile path so each worker processes all
    # 512 samples from a tile before moving to the next → ~512× fewer NFS
    # file-opens per epoch vs shuffle=True with random index.
    # _safe_collate bypasses the worker-side _new_shared()+resize_() path
    # that crashes on some PyTorch versions (see _safe_collate docstring).
    train_sampler = TileGroupSampler(train_ds, shuffle=True, seed=args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=_safe_collate,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=_safe_collate,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    ) if val_ds else None

    # ── Checkpoint dir ────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────
    best_val_acc = best_val_acc_resume   # restored from checkpoint (0.0 if fresh run)
    best_epoch = 0
    no_improve = 0

    # ── JSON training log ─────────────────────────────────────────
    log_path: Path | None = Path(args.log_file) if args.log_file else None
    log_epochs: list[dict] = []
    log_config: dict = {
        "run_id": Path(args.checkpoint_dir).name,
        "epochs_planned": args.epochs,
        "stage1_epochs": args.stage1_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "backbone_lr_factor": args.backbone_lr_factor,
        "focal_gamma": args.focal_gamma,
        "use_aux": args.use_aux,
        "amp": args.amp,
        "checkpoint_dir": str(args.checkpoint_dir),
    }
    val_metrics: dict = {}   # keeps last epoch val metrics in scope
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
        train_sampler.set_epoch(epoch)   # re-shuffle tile order each epoch
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
            train=True, use_aux=args.use_aux, use_amp=args.amp,
        )
        train_metrics = compute_metrics(train_preds, train_labels)
        scheduler.step()

        val_str = ""
        val_acc = 0.0
        if val_loader is not None:
            val_loss, val_preds, val_labels = _run_epoch(
                model, val_loader, criterion, None, device,
                train=False, use_aux=args.use_aux, use_amp=args.amp,
            )
            val_metrics = compute_metrics(val_preds, val_labels)
            val_acc = val_metrics["mean_iou"]
            val_str = (
                f"  val_loss={val_loss:.4f}  val_mIoU={val_acc:.4f}"
                f"  val_mAcc={val_metrics['mean_acc']:.4f}"
                f"  val_OA={val_metrics['overall_acc']:.4f}"
            )

        elapsed = time.time() - t0
        head_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch+1:3d}/{args.epochs}  "
            f"loss={train_loss:.4f}  mIoU={train_metrics['mean_iou']:.4f}  mAcc={train_metrics['mean_acc']:.4f}"
            f"{val_str}  lr={head_lr:.2e}  {elapsed:.0f}s"
        )

        # Save best
        _is_new_best = bool(val_acc > best_val_acc)
        if _is_new_best:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            no_improve = 0
            save_path = ckpt_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mean_iou": val_acc,
                "val_mean_acc": val_metrics["mean_acc"],
                "n_aux": n_aux,
                "num_frames": num_frames,
                "args": vars(args),
            }, save_path)
            print(f"    ↑ New best mIoU={best_val_acc:.4f} → {save_path}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\n  Early stopping (no improvement for {args.patience} epochs)")
                break

        # Per-epoch tile predictions for dashboard visualization
        if pred_tiles:
            _save_tile_preds_epoch(
                model, pred_tiles, epoch + 1, ckpt_dir,
                args.context_px, args.use_frame_2016, args.use_aux, device,
            )

        # Append epoch record to JSON log and flush atomically
        if log_path is not None:
            rec: dict = {
                "epoch": epoch + 1,
                "stage": current_stage,
                "train_loss": round(float(train_loss), 6),
                "train_mIoU": round(float(train_metrics["mean_iou"]), 6),
                "train_mAcc": round(float(train_metrics["mean_acc"]), 6),
                "val_loss": round(float(val_loss), 6) if val_loader else None,
                "val_mIoU": round(float(val_acc), 6) if val_loader else None,
                "val_mAcc": round(float(val_metrics.get("mean_acc", 0.0)), 6) if val_loader else None,
                "val_OA":   round(float(val_metrics.get("overall_acc", 0.0)), 6) if val_loader else None,
                "per_class_iou": {k: round(v, 6) for k, v in val_metrics.get("per_class_iou", {}).items()},
                "per_class_acc": {k: round(v, 6) for k, v in val_metrics.get("per_class_acc", {}).items()},
                "per_class_n":   val_metrics.get("per_class_n", {}),
                "lr": round(float(optimizer.param_groups[0]["lr"]), 8),
                "elapsed_s": round(float(time.time() - t0), 1),
                "is_best": _is_new_best,
            }
            log_epochs.append(rec)
            _write_training_log(log_path, log_config, log_epochs)

        # Periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            periodic_path = ckpt_dir / f"checkpoint_epoch{epoch+1:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_mean_iou": val_acc,
                "val_mean_acc": val_metrics.get("mean_acc", 0.0) if val_loader else 0.0,
            }, periodic_path)

    # ── Final summary ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best val mIoU: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"  Best model:    {ckpt_dir / 'best_model.pt'}")
    print(f"{'='*60}")

    # ── Final per-class eval on best model ────────────────────────
    if val_loader is not None:
        print("\n  Final per-class results (best model):")
        print(f"  {'Class':<22}  {'IoU':>6}  {'Acc':>6}  {'N':>8}")
        print(f"  {'-'*22}  {'-'*6}  {'-'*6}  {'-'*8}")
        best_ckpt = torch.load(ckpt_dir / "best_model.pt",
                               map_location=device, weights_only=True)
        model.load_state_dict(best_ckpt["model_state_dict"])
        _, val_preds, val_labels = _run_epoch(
            model, val_loader, criterion, None, device,
            train=False, use_aux=args.use_aux, use_amp=args.amp,
        )
        metrics = compute_metrics(val_preds, val_labels)
        for name in metrics["per_class_iou"]:
            iou = metrics["per_class_iou"][name]
            acc = metrics["per_class_acc"].get(name, float("nan"))
            n   = metrics["per_class_n"].get(name, 0)
            print(f"  {name:<22}  {iou:6.3f}  {acc:6.3f}  {n:>8,}")
        print(f"\n  mIoU={metrics['mean_iou']:.4f}  mAcc={metrics['mean_acc']:.4f}"
              f"  OA={metrics['overall_acc']:.4f}")

        # Write final summary to JSON log
        if log_path is not None:
            summary = {
                "best_epoch": best_epoch,
                "best_val_mIoU": round(best_val_acc, 6),
                "total_epochs_run": len(log_epochs),
                "final_val_mIoU": round(metrics["mean_iou"], 6),
                "final_val_mAcc": round(metrics["mean_acc"], 6),
                "final_val_OA":   round(metrics["overall_acc"], 6),
                "final_per_class_iou": {k: round(v, 6) for k, v in metrics["per_class_iou"].items()},
                "final_per_class_acc": {k: round(v, 6) for k, v in metrics["per_class_acc"].items()},
                "final_per_class_n":   metrics["per_class_n"],
            }
            _write_training_log(log_path, log_config, log_epochs, summary=summary)
            print(f"  Training log → {log_path}")


if __name__ == "__main__":
    main()
