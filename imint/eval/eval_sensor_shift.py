"""Phase 2c — sensor / atmosphere distribution shift.

We're S2-only, so this is a proxy: we vary the *processing* side of
the sensor pipeline rather than the sensor itself. Four variants:

    l2a_to_l1c:        Train on BOA (L2A), test on TOA (L1C, sen2cor
                       disabled). Atmospheric-correction sensitivity.
    cloud_threshold:   Train on strict SCL (cloud_max=0.10), test on
                       permissive (0.30). Cloud-mask-quality sensitivity.
    drop_slot_0:       Same tiles, slot 0 (autumn y-1) zeroed at eval.
                       Robustness to common refetch failure mode.
    drop_slots_0_3:    Zero both flank frames. Worst-case 2-frame test.

These are "shift" in spirit even though the underlying scenes are the
same — the model sees a different distribution because the input has
been corrupted at a well-defined point in the pipeline.
"""
from __future__ import annotations

from pathlib import Path

from .metrics import EvalResult


def run(
    ensemble_checkpoint: Path,
    tiles_dir: str,
    splits_root: Path,
    *,
    variant: str = "l2a_to_l1c",
    l1c_tiles_dir: str | None = None,        # needed for l2a_to_l1c
    output_dir: Path | None = None,
    device: str = "cuda",
) -> dict[str, EvalResult]:
    """Run a single sensor-shift variant.

    Most variants reuse the in-distribution ensemble checkpoint —
    no re-training. The exception is ``l2a_to_l1c``: ideally we'd
    fine-tune on L1C too for the full pair-wise comparison, but the
    cheaper version just measures the drop from feeding L1C to an
    L2A-trained model. We default to the cheaper version.

    TODO: for drop_slot variants, monkey-patch the dataset loader to
    zero the specified slots; for cloud_threshold, swap the SCL-derived
    mask in the cached tile. For l2a_to_l1c, pair tile names across
    the two directories.
    """
    raise NotImplementedError
