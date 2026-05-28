"""Phase 1 — in-distribution sanity check.

Loads the holdout 10% split, runs the ensemble + each baseline, computes
the metric palette from :mod:`imint.eval.metrics`. Produces a single
``EvalResult`` per (model, split) and a comparison table.

Pass criteria — defined in the plan doc — must be checked by the
report generator, not here; this module just produces the numbers.
"""
from __future__ import annotations

from pathlib import Path

from .metrics import EvalResult


def run(
    ensemble_checkpoint: Path,
    tiles_dir: str,
    split_dir: Path,
    *,
    baselines: list[str] | None = None,
    output_dir: Path | None = None,
    device: str = "cuda",
) -> dict[str, EvalResult]:
    """Run the in-distribution evaluation.

    Args:
        ensemble_checkpoint: Path to the trained ensemble .pt / .safetensors.
        tiles_dir: Where the test tiles live (usually /cephfs/unified_v2_512).
        split_dir: Folder with ``test.txt`` listing tile names + manifest.json.
        baselines: Subset of registry keys from
            :data:`imint.eval.baselines.BASELINES`. ``None`` = all four.
        output_dir: Predictions + per-tile metrics dumped here for later
            failure-mode inspection.
        device: ``"cuda"`` for H100 inference, ``"cpu"`` for laptop dry-run.

    Returns:
        ``{"ensemble": EvalResult, "trivial_majority": EvalResult, ...}``
        — one entry per model evaluated.

    TODO: load checkpoint, iterate test split, accumulate confusion
    matrix, compute IoU + F1 + κ + ECE. For each baseline, do the same
    via :mod:`imint.eval.baselines`.
    """
    raise NotImplementedError("phase 1 — fill when ensemble checkpoint lands")
