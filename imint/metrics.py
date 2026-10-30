"""NumPy segmentation metrics shared by training and evaluation."""
from __future__ import annotations

import numpy as np
from imint.schema.class_schema import get_class_names
from imint.schema.unified_schema import UNIFIED_CLASSES

# Checkpoint-selection semantics; unversioned historical scores are not comparable.
SEGMENTATION_METRIC_VERSION = "miou-ignore-target-v2"


def miou_from_confusion(
    cm: np.ndarray,
    ignore_index: int = 0,
) -> tuple[float, dict[int, float]]:
    """Score a (target, prediction) confusion matrix under the v2 contract.

    Row ``ignore_index`` (pixels whose *target* is ignored) is dropped before
    scoring, so a matrix accumulated without the ignore filter is scored the
    same way as one built by :func:`compute_miou`. Column ``ignore_index`` is
    kept: predicting it on a valid target is a false negative for that target.

    Args:
        cm: (n, n) non-negative integer counts, rows = target, cols = pred.
        ignore_index: Target label excluded from scoring.

    Returns:
        ``(miou, per_class)``: unrounded mean IoU over classes with a nonzero
        denominator (0.0 if none), and IoU per class index except
        ``ignore_index`` (NaN where the denominator is zero).
    """
    cm = np.asarray(cm)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"confusion matrix must be square, got {cm.shape}")
    if not np.issubdtype(cm.dtype, np.integer) or (cm < 0).any():
        raise ValueError("confusion matrix must hold non-negative integer counts")
    n = cm.shape[0]
    cm = cm.astype(np.int64)
    if 0 <= ignore_index < n:
        cm[ignore_index, :] = 0

    per_class: dict[int, float] = {}
    ious = []
    for c in range(n):
        if c == ignore_index:
            continue
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else float("nan")
        per_class[c] = iou
        if not np.isnan(iou):
            ious.append(iou)
    miou = float(np.mean(ious)) if ious else 0.0
    return miou, per_class


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

    # Per-class IoU — unified names for the current schema and for models trained
    # before barrblandskog (IDs are append-only, so IDs 0-27 keep their names).
    from imint.schema.unified_schema import NUM_CLASSES_BEFORE_BARRBLANDSKOG, NUM_UNIFIED_CLASSES
    if num_classes in (NUM_CLASSES_BEFORE_BARRBLANDSKOG, NUM_UNIFIED_CLASSES):
        class_names = {c: UNIFIED_CLASSES[c] for c in range(num_classes)}
    else:
        class_names = get_class_names(num_classes)
    miou, ious_by_class = miou_from_confusion(cm, ignore_index)
    per_class_iou = {
        class_names.get(c, f"class_{c}"): round(float(iou), 4)
        for c, iou in ious_by_class.items()
    }

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
