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
    """ECE: weighted absolute gap between confidence and accuracy.

    ``probs`` is (..., C) softmax output; ``target`` is (...) integer
    labels. Returns a scalar in [0, 1]. Lower is better-calibrated.

    TODO: bin by max-prob, compute |acc - conf| per bin, weight by bin
    population. Use ``np.histogram`` for efficiency.
    """
    raise NotImplementedError("ECE — to be filled when eval phase 1 runs")


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
    """Area under ROC + PR for the sigmoid dual-head output.

    ``scores`` is the raw sigmoid (in [0, 1]); ``target_binary`` is the
    SKS-derived n_mature_polygons > 0 mask. Returns (AUROC, AUPR).

    TODO: sklearn.metrics if dep available, else hand-rolled trapezoid.
    """
    raise NotImplementedError


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
    """Combine multiple per-split results into the top-level report shape.

    TODO: compute per-axis robustness deltas (in-dist minus out-of-dist)
    and surface them at the top of the report.
    """
    raise NotImplementedError
