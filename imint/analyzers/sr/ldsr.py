"""LDSR-S2 wrapper — ESA OpenSR's latent diffusion 4× SR.

Slower and heavier than SEN2SR but reaches better LPIPS at the cost
of some PSNR. Useful for showing the perception-distortion trade-off
in the showcase.

Reference: Donike et al. 2025, "Trustworthy SR of Multispectral
Sentinel-2 Imagery With Latent Diffusion", IEEE JSTARS.
"""
from __future__ import annotations

import numpy as np

from .base import BaseSRModel


class LDSR(BaseSRModel):
    name = "ldsr"
    scale = 4

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self._model = None
        self._device = self.config.get("device", "cuda")
        self._steps = self.config.get("diffusion_steps", 50)

    def _load(self) -> None:
        import torch
        from opensr_model import LDSR_S2  # type: ignore[import-not-found]

        ckpt = self.config.get("checkpoint", "ESAOpenSR/opensr-model")
        self._model = LDSR_S2.from_pretrained(ckpt)
        self._model.to(self._device).eval()
        self._torch = torch

    def _predict(self, rgb_lr: np.ndarray) -> np.ndarray:
        torch = self._torch
        x = torch.from_numpy(rgb_lr).permute(2, 0, 1).unsqueeze(0).to(self._device)
        with torch.no_grad():
            y = self._model.sample(x, num_steps=self._steps)
        return y.squeeze(0).permute(1, 2, 0).cpu().numpy()
