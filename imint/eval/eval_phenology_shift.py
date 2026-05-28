"""Phase 2d — phenology distribution shift.

Sweden's VPP windows (start-of-season, end-of-season) drift year to
year. The standard pipeline pins each tile's frames to its OWN year's
VPP. This module tests what happens when:

    shift_by_days:      Pre-fetch frames using VPP windows shifted by
                        N days (positive = later, simulates climate
                        warming pushing growing season).
    year_swap:          Use year Y's VPP windows but fetch year Z's
                        scenes. Simulates VPP-derivation failure.

Both variants need an offline pre-step that re-fetches tiles with the
modified VPP — they're not pure inference-time tests. Eval-time logic
in this module assumes the pre-fetched tiles already exist in a
parallel directory.
"""
from __future__ import annotations

from pathlib import Path

from .metrics import EvalResult


def run(
    ensemble_checkpoint: Path,
    shifted_tiles_dir: str,                  # output of the offline pre-step
    splits_root: Path,
    *,
    variant: str = "shift_by_days",
    shift_days: int = 30,
    output_dir: Path | None = None,
    device: str = "cuda",
) -> dict[str, EvalResult]:
    """Run a single phenology-shift variant.

    Returns the paired (in_dist, out_dist, delta) bundle. The
    ``in_dist`` arm uses the original tiles directory; the
    ``out_dist`` arm uses ``shifted_tiles_dir``.

    TODO: this is gated on the pre-step that creates shifted tiles.
    Stub the entry point; the orchestrator can detect missing pre-step
    artifacts and skip the variant rather than crash.
    """
    raise NotImplementedError
