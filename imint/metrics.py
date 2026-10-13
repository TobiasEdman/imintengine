"""NumPy segmentation metrics shared by training and evaluation."""
from __future__ import annotations

import numpy as np
from imint.schema.class_schema import get_class_names
from imint.schema.unified_schema import UNIFIED_CLASSES

# Checkpoint-selection semantics; unversioned historical scores are not comparable.
SEGMENTATION_METRIC_VERSION = "miou-ignore-target-v2"


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
        ignore_index: Target label excluded from confusion, IoU, and accuracy.
            Predictions of this class on other valid targets remain errors.

    Returns:
        Dict with "miou", "per_class_iou", "overall_accuracy",
        "confusion_matrix".
    """
    pred = np.asarray(pred)
    target = np.asarray(target)
    if pred.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    pred = pred.flatten()
    target = target.flatten()

    # Confusion matrix — num_classes already includes background (class 0)
    n = num_classes
    cm = np.zeros((n, n), dtype=np.int64)
    valid = (
        (target != ignore_index)
        & (target >= 0) & (target < n)
    )
    if np.any(target[valid] != np.floor(target[valid])):
        raise ValueError("valid targets must be integer class IDs")
    scored_pred = pred[valid]
    if np.any(~np.isfinite(scored_pred) | (scored_pred < 0)
              | (scored_pred >= n) | (scored_pred != np.floor(scored_pred))):
        raise ValueError("predictions on valid targets must be integer class IDs in range")
    np.add.at(cm, (target[valid].astype(np.int64), scored_pred.astype(np.int64)), 1)

    # Per-class IoU — use unified 23-class names when applicable
    from imint.schema.unified_schema import NUM_UNIFIED_CLASSES
    if num_classes == NUM_UNIFIED_CLASSES:
        class_names = UNIFIED_CLASSES
    else:
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

    # Accuracy uses the same valid pairs as the confusion matrix.
    if valid.any():
        oa = float((pred[valid] == target[valid]).sum()) / valid.sum()
    else:
        oa = 0.0

    return {
        "metric_version": SEGMENTATION_METRIC_VERSION,
        "miou": round(miou, 4),
        "per_class_iou": per_class_iou,
        "overall_accuracy": round(oa, 4),
        "confusion_matrix": cm,
    }
