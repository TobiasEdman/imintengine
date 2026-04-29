"""SEN2SR wrapper — ESA OpenSR's radiometrically-consistent 4× SR.

API verified against https://github.com/ESAOpenSR/SEN2SR README:

    pip install sen2sr mlstac
    import mlstac

    # Two-step load: download from HuggingFace into a local cache dir,
    # then load via the local path. mlstac.load() does NOT accept URLs.
    mlstac.download(
        file="https://huggingface.co/tacofoundation/sen2sr/"
             "resolve/main/SEN2SRLite/main/mlm.json",
        output_dir=local_dir,
    )
    model = mlstac.load(local_dir).compiled_model(device=device)
    superX = model(X[None]).squeeze(0)   # X: (4, H, W) float32 [0,1]

SEN2SRLite is the lightweight 4× variant; the full model has additional
20m bands. We use Lite for the RGB-only showcase.

Reference: Aybar et al. 2025, "A Radiometrically and Spatially
Consistent Super-Resolution Framework for Sentinel-2", RSE.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .base import BaseSRModel


class SEN2SR(BaseSRModel):
    name = "sen2sr"
    scale = 4

    DEFAULT_HF_URL = (
        "https://huggingface.co/tacofoundation/sen2sr/"
        "resolve/main/SEN2SRLite/main/mlm.json"
    )
    DEFAULT_CACHE_DIR = "outputs/sr_cache/SEN2SRLite"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self._model = None
        self._device = self.config.get("device", "cuda")
        self._hf_url = self.config.get("hf_url", self.DEFAULT_HF_URL)
        self._cache_dir = Path(self.config.get("cache_dir", self.DEFAULT_CACHE_DIR))
        # Same chunked-inference pattern as LDSR. SEN2SRLite was trained
        # on 128×128 input and crashes with CUDA "index out of bounds"
        # when fed the full ~900×600 LR tile in one shot — internal
        # shape assumptions get violated. Chunked + Hann-blended.
        self._chunk_lr = int(self.config.get("chunk_lr", 128))
        self._overlap_lr = int(self.config.get("overlap_lr", 16))

    def _load(self) -> None:
        import torch
        import mlstac  # type: ignore[import-not-found]

        self._torch = torch
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        # mlstac.download is idempotent — skips files already on disk.
        if not (self._cache_dir / "mlm.json").exists():
            mlstac.download(file=self._hf_url, output_dir=str(self._cache_dir))
        self._model = mlstac.load(str(self._cache_dir)).compiled_model(
            device=self._device
        )

    def _hann2d(self, n: int) -> np.ndarray:
        w = np.hanning(n).astype(np.float32)
        w = np.clip(w, 1e-3, 1.0)
        return np.outer(w, w)

    def _forward_chunk(self, rgb_chunk: np.ndarray) -> np.ndarray:
        """Run SEN2SRLite on a single 128×128 RGB chunk → 512×512 SR."""
        torch = self._torch
        h, w, _ = rgb_chunk.shape
        x4 = np.empty((4, h, w), dtype=np.float32)
        x4[0] = rgb_chunk[..., 0]                 # R
        x4[1] = rgb_chunk[..., 1]                 # G
        x4[2] = rgb_chunk[..., 2]                 # B
        x4[3] = rgb_chunk[..., 1]                 # NIR proxy (G)
        x = torch.from_numpy(x4).to(self._device)
        with torch.no_grad():
            y = self._model(x[None]).squeeze(0)
        return y[:3].permute(1, 2, 0).cpu().numpy()

    def _predict(self, rgb_lr: np.ndarray) -> np.ndarray:
        H, W, _ = rgb_lr.shape
        c = self._chunk_lr
        ov = self._overlap_lr
        stride = c - ov
        s = self.scale

        pad_h = (-H) % stride or 0
        pad_w = (-W) % stride or 0
        if pad_h or pad_w:
            rgb = np.pad(rgb_lr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        else:
            rgb = rgb_lr
        Hp, Wp, _ = rgb.shape

        sr_canvas = np.zeros((Hp * s, Wp * s, 3), dtype=np.float32)
        wt_canvas = np.zeros((Hp * s, Wp * s), dtype=np.float32)
        win = self._hann2d(c * s)

        ys = list(range(0, max(1, Hp - c) + 1, stride))
        xs = list(range(0, max(1, Wp - c) + 1, stride))
        if ys[-1] + c < Hp: ys.append(Hp - c)
        if xs[-1] + c < Wp: xs.append(Wp - c)

        for y0 in ys:
            for x0 in xs:
                chunk = rgb[y0:y0 + c, x0:x0 + c]
                sr_chunk = self._forward_chunk(chunk)
                sy0, sx0 = y0 * s, x0 * s
                sr_canvas[sy0:sy0 + c * s, sx0:sx0 + c * s] += sr_chunk * win[..., None]
                wt_canvas[sy0:sy0 + c * s, sx0:sx0 + c * s] += win

        sr_canvas /= wt_canvas[..., None]
        return sr_canvas[: H * s, : W * s]
