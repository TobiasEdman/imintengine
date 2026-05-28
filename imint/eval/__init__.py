"""imint.eval — Robust evaluation suite for the post-training ensemble.

Five-phase evaluation tratt (in-distribution → distribution-shift →
failure-mode → operational → report). Each phase is its own module;
this package only re-exports the public entry points so callers can
import without knowing the layout.

Inspired by the EarthShift framing (Doerksen 2026, RISE seminar):
robustness is measured by *paired* in- vs out-of-distribution scores
along the temporal / geographic / sensor / phenology axes — never as
a single in-distribution number.

Existing infrastructure reused:
    * ``imint.training.evaluate``  — per-class IoU, confusion matrix
    * ``imint.training.dataset``    — tile loader + label transforms
    * ``imint.training.unified_schema`` — 23-class definition
"""
from __future__ import annotations

# Public re-exports. Modules are imported lazily inside the wrappers
# below to keep ``import imint.eval`` cheap (no torch / rasterio at
# package-init time).


def evaluate_in_distribution(*args, **kwargs):  # noqa: D401 — thin wrapper
    """See :func:`imint.eval.eval_in_distribution.run`."""
    from .eval_in_distribution import run
    return run(*args, **kwargs)


def evaluate_temporal_shift(*args, **kwargs):
    """See :func:`imint.eval.eval_temporal_shift.run`."""
    from .eval_temporal_shift import run
    return run(*args, **kwargs)


def evaluate_geographic_shift(*args, **kwargs):
    """See :func:`imint.eval.eval_geographic_shift.run`."""
    from .eval_geographic_shift import run
    return run(*args, **kwargs)


def evaluate_sensor_shift(*args, **kwargs):
    """See :func:`imint.eval.eval_sensor_shift.run`."""
    from .eval_sensor_shift import run
    return run(*args, **kwargs)


def evaluate_phenology_shift(*args, **kwargs):
    """See :func:`imint.eval.eval_phenology_shift.run`."""
    from .eval_phenology_shift import run
    return run(*args, **kwargs)


def evaluate_failure_modes(*args, **kwargs):
    """See :func:`imint.eval.eval_failure_modes.run`."""
    from .eval_failure_modes import run
    return run(*args, **kwargs)


def evaluate_operational(*args, **kwargs):
    """See :func:`imint.eval.eval_operational.run`."""
    from .eval_operational import run
    return run(*args, **kwargs)


def generate_report(*args, **kwargs):
    """See :func:`imint.eval.report.generate`."""
    from .report import generate
    return generate(*args, **kwargs)


__all__ = [
    "evaluate_in_distribution",
    "evaluate_temporal_shift",
    "evaluate_geographic_shift",
    "evaluate_sensor_shift",
    "evaluate_phenology_shift",
    "evaluate_failure_modes",
    "evaluate_operational",
    "generate_report",
]
