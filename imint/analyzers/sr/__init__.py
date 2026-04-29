"""Super-resolution model wrappers for the SR showcase.

Each wrapper exposes a uniform :meth:`predict` returning a 4×-upscaled RGB
array shaped ``(H*4, W*4, 3)``, float32 in [0, 1]. The wrappers are *not*
``BaseAnalyzer`` subclasses — SR is an image-to-image transform, not a
scene analysis, so the analyzer interface (which expects
``AnalysisResult.outputs`` dicts) is a bad fit. See ``base.BaseSRModel``.

Registry maps short id → constructor. Used by
``scripts/generate_sr_showcase.py``.
"""
from __future__ import annotations

from .base import BaseSRModel
from .bicubic import BicubicSR
from .sen2sr import SEN2SR
from .ldsr import LDSR
from .difffusr import DiffFuSR
from .sr4rs import SR4RS

MODEL_REGISTRY: dict[str, type[BaseSRModel]] = {
    "bicubic":   BicubicSR,
    "sen2sr":    SEN2SR,
    "ldsr":      LDSR,
    "difffusr":  DiffFuSR,
    "sr4rs":     SR4RS,
}

__all__ = [
    "BaseSRModel",
    "BicubicSR",
    "SEN2SR",
    "LDSR",
    "DiffFuSR",
    "SR4RS",
    "MODEL_REGISTRY",
]
