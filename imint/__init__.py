"""Satellite intelligence library with capability-specific dependencies.

Importing imint does not initialize models, providers or the analysis engine.
"""
from importlib import import_module

__all__ = [
    "run_job", "coregister_to_reference", "coregister_timeseries",
    "coregister_interframe", "clearest_frame_idx", "estimate_mi_offset",
    "compute_grid_offset", "align_arrays", "subpixel_shift",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = ".engine" if name == "run_job" else ".coregistration"
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
