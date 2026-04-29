"""Local CPU smoke-test for sen2sr / ldsr — get a real Python traceback
instead of the async CUDA assert we're stuck on in K8s.

Usage:
    .venv-sr/bin/python scripts/_sr_local_smoke.py sen2sr
    .venv-sr/bin/python scripts/_sr_local_smoke.py ldsr

This script bypasses the full DES fetch + generator and just feeds a
single 128×128 chunk of synthetic-but-realistic L2A reflectance into the
model directly. If it crashes here, we see the exact line.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEVICE = "cpu"


def make_test_chunk(seed: int = 0, with_extremes: bool = False) -> np.ndarray:
    """Build a 128×128 RGB chunk that mimics S2 L2A reflectance.

    Real Stockholm tile values land around [0.02, 0.50] for land/water,
    with cloudy specular pixels potentially clipping at 1.0. We
    optionally include a few 1.0 cells to test if those trigger asserts.
    """
    rng = np.random.default_rng(seed)
    # Bulk of values in [0.04, 0.40] like real surface reflectance
    rgb = rng.uniform(0.04, 0.40, size=(128, 128, 3)).astype(np.float32)
    if with_extremes:
        # Plant a 5×5 patch of 1.0 (saturated cloud) and a 5×5 of 0.0
        rgb[10:15, 10:15] = 1.0
        rgb[20:25, 20:25] = 0.0
    return rgb


def test_sen2sr(rgb: np.ndarray) -> None:
    """SEN2SR needs 10 bands [B02, B03, B04, B05, B06, B07, B08, B8A,
    B11, B12]. We synthesise the 7 missing bands by perturbing the RGB
    distribution — the model just needs the right number of channels
    to not crash on internal indexing."""
    print(f"[sen2sr] input rgb shape={rgb.shape}")
    import torch
    import mlstac

    cache = ROOT / "outputs" / "sr_cache" / "SEN2SRLite"
    cache.mkdir(parents=True, exist_ok=True)
    if not (cache / "mlm.json").exists():
        print("[sen2sr] downloading weights to", cache)
        mlstac.download(
            file=("https://huggingface.co/tacofoundation/sen2sr/"
                  "resolve/main/SEN2SRLite/main/mlm.json"),
            output_dir=str(cache),
        )
    model = mlstac.load(str(cache)).compiled_model(device=DEVICE)
    print(f"[sen2sr] model loaded, type={type(model).__name__}")

    rng = np.random.default_rng(1)
    h, w, _ = rgb.shape
    # Build 10-band input: [B02=B, B03=G, B04=R, B05..B07=red-edge proxies,
    # B08=NIR proxy, B8A=NIR proxy, B11=SWIR proxy, B12=SWIR proxy]
    x10 = np.empty((10, h, w), dtype=np.float32)
    x10[0] = rgb[..., 2]                                  # B02
    x10[1] = rgb[..., 1]                                  # B03
    x10[2] = rgb[..., 0]                                  # B04
    for k in range(3, 10):
        x10[k] = rgb[..., 1] + rng.uniform(-0.05, 0.05, (h, w)).astype(np.float32)
    x10 = np.clip(x10, 0.0, 1.0)
    x = torch.from_numpy(x10).to(DEVICE)
    print(f"[sen2sr] tensor shape={tuple(x.shape)}, dtype={x.dtype}, "
          f"range=[{x.min():.3f}, {x.max():.3f}]")

    with torch.no_grad():
        y = model(x[None])
    print(f"[sen2sr] OK — output shape={tuple(y.shape)}, range=[{y.min():.3f}, {y.max():.3f}]")


def test_ldsr(rgb: np.ndarray) -> None:
    print(f"[ldsr] input shape={rgb.shape}, range=[{rgb.min():.3f}, {rgb.max():.3f}]")
    from io import StringIO
    import requests
    import torch
    from omegaconf import OmegaConf
    import opensr_model

    cfg_url = ("https://raw.githubusercontent.com/ESAOpenSR/opensr-model/"
               "refs/heads/main/opensr_model/configs/config_10m.yaml")
    cfg_text = requests.get(cfg_url, timeout=30).text
    cfg = OmegaConf.load(StringIO(cfg_text))
    model = opensr_model.SRLatentDiffusion(cfg, device=DEVICE)
    model.load_pretrained(cfg.ckpt_version)
    print(f"[ldsr] model loaded, type={type(model).__name__}")

    h, w, _ = rgb.shape
    x4 = np.empty((1, 4, h, w), dtype=np.float32)
    x4[0, 0] = rgb[..., 0]
    x4[0, 1] = rgb[..., 1]
    x4[0, 2] = rgb[..., 2]
    x4[0, 3] = rgb[..., 1]
    x = torch.from_numpy(x4).to(DEVICE)

    print(f"[ldsr] tensor shape={tuple(x.shape)}, dtype={x.dtype}")
    with torch.no_grad():
        y = model.forward(x, sampling_steps=10)  # cheap test
    print(f"[ldsr] OK — output shape={tuple(y.shape)}, range=[{y.min():.3f}, {y.max():.3f}]")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "sen2sr"
    extremes = "--extremes" in sys.argv
    rgb = make_test_chunk(with_extremes=extremes)
    if target == "sen2sr":
        test_sen2sr(rgb)
    elif target == "ldsr":
        test_ldsr(rgb)
    else:
        print(f"unknown target: {target}")
        sys.exit(1)
