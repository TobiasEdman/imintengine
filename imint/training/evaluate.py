"""
imint/training/evaluate.py — Evaluation metrics for LULC segmentation

Computes per-class IoU, mean IoU, overall accuracy, and confusion matrix.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:
    raise ImportError("PyTorch is required. Install with: pip install torch")

from .class_schema import get_class_names


def compute_miou(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = 0,
) -> dict:
    """Compute per-class IoU and mean IoU.

    Args:
        pred: (N,) or (H, W) predicted class indices.
        target: (N,) or (H, W) ground truth class indices.
        num_classes: Total number of classes (including background).
        ignore_index: Class to ignore in mIoU computation.

    Returns:
        Dict with "miou", "per_class_iou", "overall_accuracy",
        "confusion_matrix".
    """
    pred = pred.flatten()
    target = target.flatten()

    # Confusion matrix
    n = num_classes + 1
    cm = np.zeros((n, n), dtype=np.int64)
    valid = (target >= 0) & (target < n) & (pred >= 0) & (pred < n)
    np.add.at(cm, (target[valid], pred[valid]), 1)

    # Per-class IoU
    class_names = get_class_names(num_classes)
    per_class_iou = {}
    ious = []

    for c in range(n):
        if c == ignore_index:
            continue
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = tp + fp + fn
        if denom > 0:
            iou = tp / denom
        else:
            iou = float("nan")

        name = class_names.get(c, f"class_{c}")
        per_class_iou[name] = round(float(iou), 4)
        if not np.isnan(iou):
            ious.append(iou)

    miou = float(np.mean(ious)) if ious else 0.0

    # Overall accuracy (excluding ignore class)
    mask = target != ignore_index
    if mask.any():
        oa = float((pred[mask] == target[mask]).sum()) / mask.sum()
    else:
        oa = 0.0

    return {
        "miou": round(miou, 4),
        "per_class_iou": per_class_iou,
        "overall_accuracy": round(oa, 4),
        "confusion_matrix": cm,
    }


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataset,
    config,
    device: torch.device,
    max_samples: int | None = None,
) -> dict:
    """Evaluate a segmentation model on a dataset.

    Args:
        model: Prithvi segmentation model in eval mode.
        dataset: LULCDataset instance.
        config: TrainingConfig.
        device: Torch device.
        max_samples: Limit number of samples (for quick checks).

    Returns:
        Dict with mIoU, per-class IoU, overall accuracy.
    """
    model.eval()
    all_preds = []
    all_targets = []

    n_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    # Ordered aux channel names (must match trainer._collect_aux order)
    _AUX_NAMES = ("height", "volume", "basal_area", "diameter", "dem")

    for i in range(n_samples):
        sample = dataset[i]
        image = sample["image"].unsqueeze(0).to(device)  # (1, 6, H, W)
        label = sample["label"].numpy()                    # (H, W)

        # Add temporal dimension: (1, 6, H, W) → (1, 6, 1, H, W)
        image_5d = image.unsqueeze(2)

        # Collect auxiliary channels if present
        aux_parts = []
        for name in _AUX_NAMES:
            if name in sample:
                aux_parts.append(sample[name].unsqueeze(0).to(device))
        aux = torch.cat(aux_parts, dim=1) if aux_parts else None

        logits = model(image_5d, aux=aux).contiguous()  # (1, C, H, W)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)

        all_preds.append(pred)
        all_targets.append(label)

    all_preds = np.concatenate([p.flatten() for p in all_preds])
    all_targets = np.concatenate([t.flatten() for t in all_targets])

    return compute_miou(
        all_preds, all_targets,
        num_classes=config.num_classes,
        ignore_index=config.ignore_index,
    )
