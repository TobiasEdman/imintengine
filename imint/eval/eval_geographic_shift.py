"""Phase 2b — geographic distribution shift.

Three axes the project's data supports:

    north_south:   y=6.7e6 (EPSG:3006) splits Norrland from Götaland.
                   Boreal vs nemoral skog confounder.
    coast_inland:  distance-to-coastline. Atmospheric & humidity shift.
    skog_slatt:    NMD forest-fraction per tile. Land-cover mix shift.

Each axis runs the same pair-wise protocol: train one side, test the
other, report delta. Per-class drops surface which classes carry a
strong geographic bias (e.g. tallskog vs granskog north/south split).
"""
from __future__ import annotations

from pathlib import Path

from .metrics import EvalResult


def run(
    ensemble_train_fn,
    tiles_dir: str,
    splits_root: Path,
    *,
    axis: str = "north_south",                # see module docstring
    flip_train_test: bool = False,            # also evaluate the reverse direction
    output_dir: Path | None = None,
    device: str = "cuda",
) -> dict[str, EvalResult]:
    """Run a single geographic-shift axis.

    When ``flip_train_test=True`` the function trains and evaluates
    twice (north→south + south→north) and returns both directions —
    a one-sided drop can come from training-set size rather than the
    shift itself; the symmetric drop is the more honest robustness
    measure.

    TODO: invoke splits.make_geographic_split for both directions,
    train, evaluate, compute deltas, return paired bundle.
    """
    raise NotImplementedError
