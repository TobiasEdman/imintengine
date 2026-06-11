"""Shared metrics — wraps ``imint.training.evaluate`` + calibration extras.

The existing ``imint.training.evaluate`` already provides per-class IoU,
mean IoU, accuracy, and confusion matrix. This module thinly re-exports
those plus the calibration metrics (ECE, MCE) and the dual-head specific
metrics (AUROC, AUPR) that the segmentation-only evaluator doesn't
cover.

Design rule: **never recompute** what training/evaluate already does.
Wrap and extend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


# ── Re-exports ──────────────────────────────────────────────────────────────


def per_class_iou(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = 0,
) -> dict:
    """Thin re-export of :func:`imint.training.evaluate.compute_miou`."""
    from imint.training.evaluate import compute_miou
    return compute_miou(pred, target, num_classes, ignore_index=ignore_index)


# ── Calibration ─────────────────────────────────────────────────────────────


def expected_calibration_error(
    probs: np.ndarray,
    target: np.ndarray,
    *,
    num_bins: int = 15,
    ignore_index: int = 0,
) -> float:
    """Expected Calibration Error (Guo et al. 2017).

    Bins predictions by their max-class confidence and accumulates the
    population-weighted absolute gap between bin-mean accuracy and
    bin-mean confidence. Returns a scalar in ``[0, 1]``; ``0`` =
    perfectly calibrated.

    Accepts class-axis-first ``(C, ...)`` or class-axis-last
    ``(..., C)`` layouts; detected by which axis matches the unique
    values in ``target``. Pixels where ``target == ignore_index`` are
    excluded — they're typically the background class our schema
    treats as "don't score".
    """
    probs = np.asarray(probs)
    target = np.asarray(target)

    # Move class axis to last for uniform handling.
    if probs.ndim == target.ndim + 1:
        # Assume class axis is the one with the largest size that
        # matches target's max+1 (or just the one not in target's shape).
        target_shape = target.shape
        class_axis_candidates = [
            i for i, s in enumerate(probs.shape) if s not in target_shape
        ]
        if not class_axis_candidates:
            # Fall through to channel-first guess.
            class_axis = 0
        else:
            class_axis = class_axis_candidates[0]
        probs = np.moveaxis(probs, class_axis, -1)

    probs_flat = probs.reshape(-1, probs.shape[-1])
    target_flat = target.reshape(-1)

    valid = target_flat != ignore_index
    probs_flat = probs_flat[valid]
    target_flat = target_flat[valid]
    if probs_flat.size == 0:
        return 0.0

    confidences = probs_flat.max(axis=1)
    predictions = probs_flat.argmax(axis=1)
    accuracies = (predictions == target_flat).astype(np.float32)

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    # np.digitize with right=False gives bins[i] = lo (inclusive) .. hi (exclusive)
    # except the last bin where we want hi inclusive too. Patch with clip.
    bin_idx = np.clip(
        np.digitize(confidences, bin_edges[1:-1], right=False),
        0, num_bins - 1,
    )

    total = float(len(confidences))
    ece = 0.0
    for b in range(num_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (float(mask.sum()) / total) * abs(bin_acc - bin_conf)
    return float(ece)


def reliability_diagram_data(
    probs: np.ndarray,
    target: np.ndarray,
    *,
    num_bins: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_centers, bin_confidence, bin_accuracy) for plotting.

    TODO: produce arrays the report-generator can hand to matplotlib.
    """
    raise NotImplementedError


# ── Dual-head (avverkningsmogen) ────────────────────────────────────────────


def auroc_aupr(
    scores: np.ndarray,
    target_binary: np.ndarray,
) -> tuple[float, float]:
    """Area under ROC + PR for a per-sample binary score.

    ``scores`` is a probability in [0, 1] (e.g. a class-22 / maturity softmax
    sampled at NFI plot pixels, or a sigmoid head); ``target_binary`` is the
    matching {0, 1} truth (e.g. NFI ``Maturityclass`` ≥ 41). Any matching
    shapes work — both are flattened.

    Hand-rolled (no sklearn dependency): sort by score descending, sweep every
    distinct-score threshold, and integrate the ROC (TPR vs FPR) and PR
    (precision vs recall) curves by trapezoid. Returns ``(AUROC, AUPR)``, or
    ``(nan, nan)`` when only one class is present (AUROC is undefined).
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    target = np.asarray(target_binary).ravel().astype(bool)
    if scores.shape != target.shape:
        raise ValueError(f"shape mismatch: scores {scores.shape} vs target {target.shape}")

    n_pos = int(target.sum())
    n_neg = target.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")

    order = np.argsort(-scores, kind="mergesort")  # stable, descending
    y = target[order].astype(np.float64)
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)

    # One operating point per distinct score (collapse ties).
    boundary = np.r_[np.diff(scores[order]) != 0, True]
    tp, fp = tp[boundary], fp[boundary]

    def _trapz(y_vals: np.ndarray, x_vals: np.ndarray) -> float:
        return float(np.sum(np.diff(x_vals) * (y_vals[1:] + y_vals[:-1]) / 2.0))

    tpr = np.r_[0.0, tp / n_pos]
    fpr = np.r_[0.0, fp / n_neg]
    auroc = _trapz(tpr, fpr)

    recall = np.r_[0.0, tp / n_pos]
    precision = tp / (tp + fp)
    precision = np.r_[precision[0], precision]  # PR curve anchored at recall 0
    aupr = _trapz(precision, recall)

    return auroc, aupr


# ── Aggregates ──────────────────────────────────────────────────────────────


@dataclass
class EvalResult:
    """Bundle of metrics produced by a single eval phase.

    Always carries enough metadata that the report generator can
    contextualise the numbers (which split, which model, etc.) without
    side-channels.
    """

    phase: str
    split_name: str
    num_tiles: int
    num_pixels: int
    metrics: dict = field(default_factory=dict)
    per_class: dict = field(default_factory=dict)
    confusion_matrix: np.ndarray | None = None
    notes: dict = field(default_factory=dict)

    def to_jsonable(self) -> dict:
        """Serialise for storage in ``REPORT.md`` JSON fence."""
        out = {
            "phase": self.phase,
            "split_name": self.split_name,
            "num_tiles": self.num_tiles,
            "num_pixels": self.num_pixels,
            "metrics": self.metrics,
            "per_class": self.per_class,
            "notes": self.notes,
        }
        if self.confusion_matrix is not None:
            out["confusion_matrix_shape"] = list(self.confusion_matrix.shape)
            # Full matrix saved separately as .npy to keep markdown small.
        return out


def aggregate_results(results: Iterable[EvalResult]) -> dict:
    """Group results by phase and compute headline numbers.

    The grouped view drives the robustness card at the top of
    ``REPORT.md``. Per-axis deltas (in-dist minus out-of-dist) are
    computed when both arms of a shift were evaluated and the
    ``split_name`` follows the
    ``"<axis>_<variant>_(in|out)_distribution"`` convention.
    """
    by_phase: dict[str, list[EvalResult]] = {}
    for r in results:
        by_phase.setdefault(r.phase, []).append(r)

    out: dict = {"by_phase": {}}
    for phase, items in by_phase.items():
        out["by_phase"][phase] = [r.to_jsonable() for r in items]

    # Robustness deltas — pair in/out arms when present.
    deltas: dict[str, dict] = {}
    by_name: dict[str, EvalResult] = {r.split_name: r for r in results}
    for split_name, r in by_name.items():
        if not split_name.endswith("_in_distribution"):
            continue
        axis_variant = split_name[: -len("_in_distribution")]
        out_split = by_name.get(f"{axis_variant}_out_of_distribution")
        if out_split is None:
            continue
        miou_in = r.metrics.get("mIoU")
        miou_out = out_split.metrics.get("mIoU")
        if miou_in is None or miou_out is None:
            continue
        deltas[axis_variant] = {
            "in_dist_mIoU":   miou_in,
            "out_dist_mIoU":  miou_out,
            "delta_pp":       (miou_in - miou_out) * 100.0,
        }
    out["robustness_deltas"] = deltas
    return out
