"""LDSR-S2 wrapper — ESA OpenSR latent diffusion 4× SR.

API verified against https://github.com/ESAOpenSR/opensr-model README:

    pip install opensr-model
    from omegaconf import OmegaConf
    import opensr_model, requests
    config = OmegaConf.load(...)  # config_10m.yaml from repo
    model = opensr_model.SRLatentDiffusion(config, device=device)
    model.load_pretrained(config.ckpt_version)
    out = model.forward(X, sampling_steps=100)   # X: (B, 4, H, W) [0,1]

The default config and weights come from
``simon-donike/RS-SR-LTDF`` on HuggingFace.

Reference: Donike et al. 2025, "Trustworthy SR of Multispectral
Sentinel-2 Imagery With Latent Diffusion", IEEE JSTARS.
"""
from __future__ import annotations

import numpy as np

from .base import BaseSRModel


class LDSR(BaseSRModel):
    name = "ldsr"
    scale = 4

    # Default config URL — pinned to main; bump when the repo cuts a tag.
    DEFAULT_CONFIG_URL = (
        "https://raw.githubusercontent.com/ESAOpenSR/opensr-model/"
        "refs/heads/main/opensr_model/configs/config_10m.yaml"
    )

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self._model = None
        self._device = self.config.get("device", "cuda")
        self._sampling_steps = self.config.get("sampling_steps", 100)
        self._config_url = self.config.get("config_url", self.DEFAULT_CONFIG_URL)

    def _load(self) -> None:
        from io import StringIO
        import requests
        import torch
        from omegaconf import OmegaConf  # type: ignore[import-not-found]
        import opensr_model  # type: ignore[import-not-found]

        resp = requests.get(self._config_url, timeout=30)
        resp.raise_for_status()
        cfg = OmegaConf.load(StringIO(resp.text))

        self._torch = torch
        self._model = opensr_model.SRLatentDiffusion(cfg, device=self._device)
        self._model.load_pretrained(cfg.ckpt_version)

    def _predict(self, rgb_lr: np.ndarray) -> np.ndarray:
        torch = self._torch
        # Same RGB→4-channel adaptation as SEN2SR: duplicate G as NIR proxy.
        # The diffusion model trained on 4 bands; feeding 3 would skew
        # the latent encoder. NIR channel of output is discarded.
        h, w, _ = rgb_lr.shape
        x4 = np.empty((1, 4, h, w), dtype=np.float32)
        x4[0, 0] = rgb_lr[..., 0]
        x4[0, 1] = rgb_lr[..., 1]
        x4[0, 2] = rgb_lr[..., 2]
        x4[0, 3] = rgb_lr[..., 1]
        x = torch.from_numpy(x4).to(self._device)
        with torch.no_grad():
            y = self._model.forward(x, sampling_steps=self._sampling_steps)
        rgb_sr = y[0, :3].permute(1, 2, 0).cpu().numpy()
        return rgb_sr
