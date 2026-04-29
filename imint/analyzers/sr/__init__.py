"""Super-resolution model wrappers for the SR showcase.

Each wrapper exposes a uniform :meth:`predict` returning a 4×-upscaled RGB
array shaped ``(H*4, W*4, 3)``, float32 in [0, 1]. The wrappers are *not*
``BaseAnalyzer`` subclasses — SR is an image-to-image transform, not a
scene analysis, so the analyzer interface (which expects
``AnalysisResult.outputs`` dicts) is a bad fit. See ``base.BaseSRModel``.

Registry maps short id → constructor. Used by
``scripts/generate_sr_showcase.py``.

Three open models with weights actually accessible today:
  - bicubic: pure interpolation, no learning. Reference floor.
  - sen2sr:  ESA OpenSR, CNN with hard radiometric constraint. PyPI
             package ``sen2sr``, weights via ``mlstac``.
  - ldsr:    ESA OpenSR latent diffusion. PyPI ``opensr-model``, weights
             on HuggingFace ``simon-donike/RS-SR-LTDF``.

DiffFuSR (arXiv 2506.11764) and SR4RS were considered but excluded — the
former is research code with no PyPI release, the latter is TF-only and
needs a separate container stack. Re-add them when their weights become
pip-installable.
"""
from __future__ import annotations

from .base import BaseSRModel
from .bicubic import BicubicSR
from .sen2sr import SEN2SR
from .ldsr import LDSR

MODEL_REGISTRY: dict[str, type[BaseSRModel]] = {
    "bicubic":   BicubicSR,
    "sen2sr":    SEN2SR,
    "ldsr":      LDSR,
}

__all__ = [
    "BaseSRModel",
    "BicubicSR",
    "SEN2SR",
    "LDSR",
    "MODEL_REGISTRY",
]
