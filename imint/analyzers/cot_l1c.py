"""
imint/analyzers/cot_l1c.py — Cloud Optical Thickness on L1C TOA reflectance

12-band variant of imint/analyzers/cot.py. The original analyzer uses the
Swedish-Forest-Agency (SFA) 11-band model which drops B01 AND B10 — that
model is intended for L2A surface reflectance (ESA removes B10 when
producing L2A).

This module uses the **12-band** Pirinen COT model — all 13 standard
Sentinel-2 bands EXCEPT B01. It keeps B10 (cirrus), so it operates on
**L1C top-of-atmosphere reflectance** directly. That makes it usable for
scene-quality ranking BEFORE sen2cor (which is the whole point — we want
to rank candidate L1C scenes per tile without paying the sen2cor compute
first).

Provenance:
    Pirinen et al. "Creating and Leveraging a Synthetic Dataset of Cloud
    Optical Thickness Measures for Cloud Detection in MSI", Remote Sensing
    2024. https://github.com/aleksispi/ml-cloud-opt-thick
    Paper §: synthetic dataset simulates *top-of-atmosphere radiances*,
    confirming the 12-band model is the TOA/L1C one.

Weights: imint/fm/cot_models_l1c/cot_mlp5_12band_{00..09}.pt — the
10-MLP ensemble from the repo's "models-trained-synthetic-12bands"
Google-Drive folder (id 1MkqcoxLBb9C1vAUwvHipq5cr6Z7bXIel).

Normalization: the upstream eval script computes input mean/std from
``trainset_smhi.npy`` at runtime and divides the COT regressor target by
``gt_max``. We computed those constants once (2026-05-16) from the SMHI
synthetic trainset and hard-code them below — see COT_MEANS_12 /
COT_STDS_12 / COT_GT_MAX. Reproduce with::

    train = np.load("trainset_smhi.npy")            # (160000, 23)
    spec  = train[:, 2:14]                           # 12 bands, B01 skipped
    COT_MEANS_12 = spec.mean(0); COT_STDS_12 = spec.std(0)
    COT_GT_MAX   = max over train/val/test of column 17  # = 50.0
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .cot import MLP5  # identical architecture, only input_dim differs

# Band order for the 12-band model: all 13 standard S2 bands except B01.
# B10 (cirrus) is KEPT — it only exists in L1C, which is exactly what we
# want to score.
COT_L1C_BAND_ORDER = [
    "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
]

# Input normalization — computed 2026-05-16 from trainset_smhi.npy
# (160 000 rows), columns 2..13 (B01 skipped). See module docstring.
COT_MEANS_12 = torch.tensor([
    0.49675034, 0.47293043, 0.56490300, 0.52927473, 0.65845986,
    0.93623101, 0.90515048, 0.99451205, 0.45604575, 0.07375108,
    0.53309616, 0.43224668,
])
COT_STDS_12 = torch.tensor([
    0.28274442, 0.27778134, 0.28483809, 0.31573642, 0.28173209,
    0.31942519, 0.32981911, 0.36159493, 0.29364748, 0.11409170,
    0.41934613, 0.33355380,
])

# COT regressor target was normalized by this during training; multiply
# the raw model output by it to get COT back in physical units.
COT_GT_MAX = 50.0

# COT thresholds (same as the 11-band analyzer — physical COT units).
DEFAULT_THICK_CLOUD_THRESH = 0.025 * COT_GT_MAX   # ≈ 1.25
DEFAULT_THIN_CLOUD_THRESH = 0.015 * COT_GT_MAX    # ≈ 0.75

_FM_DIR = Path(__file__).resolve().parent.parent / "fm" / "cot_models_l1c"
DEFAULT_MODEL_PATHS = sorted(_FM_DIR.glob("cot_mlp5_12band_*.pt"))


def load_ensemble_l1c(
    model_paths: list[Path] | None = None,
    device: str = "cpu",
) -> list[MLP5]:
    """Load the 10-MLP 12-band ensemble."""
    paths = model_paths if model_paths is not None else DEFAULT_MODEL_PATHS
    if not paths:
        raise FileNotFoundError(
            f"No 12-band COT weights found in {_FM_DIR}. Expected "
            f"cot_mlp5_12band_*.pt — copy them from the repo's "
            f"models-trained-synthetic-12bands Google-Drive folder."
        )
    models = []
    for path in paths:
        model = MLP5(input_dim=12, output_dim=1, apply_relu=True)
        model.load_state_dict(torch.load(str(path), map_location=device))
        model.to(device)
        model.eval()
        models.append(model)
    return models


def cot_inference_l1c(
    bands: dict[str, np.ndarray],
    models: list[MLP5],
    *,
    device: str = "cpu",
    batch_size: int = 65536,
) -> np.ndarray:
    """Run 12-band COT inference on an L1C TOA-reflectance image.

    Args:
        bands: Dict keyed by COT_L1C_BAND_ORDER, each (H, W) float32 in
            TOA reflectance [0, 1] (raw L1C DN / 10000).
        models: Ensemble from ``load_ensemble_l1c``.
        device: Torch device.
        batch_size: Pixels per batch.

    Returns:
        COT map (H, W) float32 in physical units (already denormalized
        by COT_GT_MAX).
    """
    missing = [b for b in COT_L1C_BAND_ORDER if b not in bands]
    if missing:
        raise KeyError(f"cot_inference_l1c: missing bands {missing}")

    h, w = bands[COT_L1C_BAND_ORDER[0]].shape
    img = np.stack([bands[b] for b in COT_L1C_BAND_ORDER], axis=-1)  # (H,W,12)
    pixels = torch.tensor(
        img.reshape(-1, 12), dtype=torch.float32, device=device,
    )

    means = COT_MEANS_12.to(device)
    stds = COT_STDS_12.to(device)
    pixels = (pixels - means) / stds

    n = h * w
    pred_sum = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for model in models:
            pred = np.zeros(n, dtype=np.float32)
            for i in range(0, n, batch_size):
                out = model(pixels[i:i + batch_size])
                pred[i:i + batch_size] = out[:, 0].cpu().numpy()
            pred_sum += pred / len(models)

    # Denormalize: training divided the COT target by COT_GT_MAX.
    return (pred_sum * COT_GT_MAX).reshape(h, w).astype(np.float32)


def cloud_score_l1c(
    bands: dict[str, np.ndarray],
    models: list[MLP5],
    *,
    thresh: float = DEFAULT_THIN_CLOUD_THRESH,
    device: str = "cpu",
) -> dict[str, float]:
    """Per-tile cloud score for ranking candidate L1C scenes.

    Returns a dict with:
        mean_cot   — mean COT over the tile. Threshold-free and strictly
                     monotonic in cloudiness; this is the value the
                     scene-selector ranks on (lower = cleaner).
        cloud_frac — fraction of pixels above ``thresh``. A secondary
                     diagnostic; the absolute threshold is only
                     approximately calibrated (0.015 normalized COT ×
                     COT_GT_MAX), so prefer mean_cot for ranking.

    One inference pass; both scalars come from the same COT map.
    """
    cot = cot_inference_l1c(bands, models, device=device)
    return {
        "mean_cot": float(cot.mean()),
        "cloud_frac": float((cot > thresh).mean()),
    }
