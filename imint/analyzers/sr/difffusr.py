"""DiffFuSR wrapper — two-stage diffusion SR over all 12 S2 bands.

For the RGB-only showcase we use only the first stage (RGB diffusion);
the second band-fusion stage is unused. Best published opensr-test
hallucination score (0.1149 on the v1 benchmark).

Reference: arXiv 2506.11764 (2025).
"""
from __future__ import annotations

import numpy as np

from .base import BaseSRModel


class DiffFuSR(BaseSRModel):
    name = "difffusr"
    scale = 4

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self._model = None
        self._device = self.config.get("device", "cuda")

    def _load(self) -> None:
        import torch
        # Repo currently exposes weights via HF; package name TBD as the
        # repo stabilises. The fallback raises a clear error so the
        # showcase prints "model failed" rather than silently skipping.
        try:
            from difffusr import DiffFuSR_RGB  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "DiffFuSR not installed. See arXiv 2506.11764 for the repo "
                "URL — install via `pip install difffusr` once the package "
                "is on PyPI, or git+ from the published repo."
            ) from e

        ckpt = self.config.get("checkpoint", "difffusr/rgb-4x")
        self._model = DiffFuSR_RGB.from_pretrained(ckpt)
        self._model.to(self._device).eval()
        self._torch = torch

    def _predict(self, rgb_lr: np.ndarray) -> np.ndarray:
        torch = self._torch
        x = torch.from_numpy(rgb_lr).permute(2, 0, 1).unsqueeze(0).to(self._device)
        with torch.no_grad():
            y = self._model(x)
        return y.squeeze(0).permute(1, 2, 0).cpu().numpy()
