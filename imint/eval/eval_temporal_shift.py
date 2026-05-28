"""Phase 2a — temporal distribution shift.

Three configurable variants:

    chronological_holdout: train on years A..B-1, test on years B..C
    next_year:             train on years A..B, test on year B+1 (strictest)
    year_leave_one_out:    train on all-except-Y, test on Y (anomaly scan)

For each variant we re-train the ensemble on the train-side, evaluate
on the test-side, then report:

    in_dist_score:   ensemble's IoU on held-out tiles from train-years
    out_dist_score:  ensemble's IoU on test-year tiles
    delta:           in_dist - out_dist  (drop = robustness penalty)

Per-class delta surfaces which classes are temporally fragile (often
crop classes that change with farming practice; hygge changes with
forestry cycle).
"""
from __future__ import annotations

from pathlib import Path

from .metrics import EvalResult


def run(
    ensemble_train_fn,                       # callable: (train_split) -> checkpoint
    tiles_dir: str,
    splits_root: Path,
    *,
    variant: str = "chronological_holdout",
    train_years: tuple[int, ...] | None = None,
    test_years: tuple[int, ...] | None = None,
    output_dir: Path | None = None,
    device: str = "cuda",
) -> dict[str, EvalResult]:
    """Run a single temporal-shift variant end-to-end.

    Args:
        ensemble_train_fn: Callable that takes a split dict and returns
            a trained-checkpoint path. Lets the orchestrator inject the
            project's existing training entry point without circular
            imports.
        variant: ``"chronological_holdout"`` | ``"next_year"`` |
            ``"year_leave_one_out"``.
        train_years, test_years: Required for the first two variants.
            ``year_leave_one_out`` uses ``test_years=(year,)``.

    Returns:
        ``{"in_distribution": EvalResult, "out_of_distribution":
        EvalResult, "delta_per_class": EvalResult}``.

    TODO: invoke splits.make_temporal_split, hand to train_fn, run
    metrics on both sides, return paired bundle.
    """
    raise NotImplementedError
