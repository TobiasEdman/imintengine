"""Phase 3 — failure-mode analysis (qualitative + structured).

Five analyses on top of the per-tile predictions saved by phase 1:

    worst_tile_panels:        Pick N tiles with lowest per-tile mIoU,
                              build a 4-up panel (RGB / label / pred /
                              error-mask) for visual inspection.
    confusion_off_diagonal:   Identify class pairs with highest
                              confusion. Rank by absolute count and by
                              fraction of either class.
    per_tile_error_cdf:       CDF of per-tile mIoU. Long left tail =
                              many fragile tiles; short tail = few
                              catastrophic failures.
    error_spatial_autocorr:   Moran's I on per-tile error to detect
                              geographic clustering of failures.
    hygge_fp_inspection:      Find dual-head predictions > 0.5 outside
                              any SKS-anmälan polygon. Manual triage
                              of false positives vs missed anmälningar.

This phase is mostly "render artefacts and let a human eyeball them",
so the output is png/markdown rather than pure metrics.
"""
from __future__ import annotations

from pathlib import Path

from .metrics import EvalResult


def run(
    predictions_dir: Path,                   # output of phase 1
    tiles_dir: str,
    sks_polygons_path: Path | None = None,
    *,
    top_n_worst: int = 20,
    output_dir: Path | None = None,
) -> dict[str, EvalResult]:
    """Run all five analyses; each writes its own artefact.

    Returns:
        ``{"worst_tiles": ..., "confusion": ..., "error_cdf": ...,
        "spatial_autocorr": ..., "hygge_fp": ...}``. The ``metrics``
        dict on each result lists which artefact files were written
        so the report generator can link them.

    TODO: implement Moran's I via libpysal if installed, else a hand-
    rolled Geary's C. The hygge FP analysis needs to join predictions
    with SKS parquets which already live on /cephfs/sks/.
    """
    raise NotImplementedError
