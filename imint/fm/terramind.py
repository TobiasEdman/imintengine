"""
imint/fm/terramind.py — TerraMind foundation model integration (lightweight)

TerraMind (ESA + IBM) is an any-to-any multimodal generative foundation
model for Earth observation. It can generate SAR from optical and vice
versa — a unique capability for data augmentation and gap-filling.

This module provides a thin integration layer. TerraMind is on the backlog
for full integration; this module enables:
  - Availability checking
  - Model loading from HuggingFace
  - SAR↔optical cross-modal generation (when model is available)

TerraMind complements Prithvi and Clay by:
  - Native multimodal: optical + SAR + DEM + LULC in one model
  - Generative: can synthesize missing modalities
  - Large scale: trained on 12M globally distributed samples

Requirements:
    pip install terratorch   # TerraMind builds on TerraTorch

HuggingFace: ibm-granite/granite-geospatial-terramind
"""
from __future__ import annotations

import numpy as np


def check_terramind_available() -> bool:
    """Check if TerraMind dependencies are installed."""
    try:
        import torch  # noqa: F401
        import terratorch  # noqa: F401
        return True
    except ImportError:
        return False


# TerraMind model identifiers
TERRAMIND_HF_REPO = "ibm-granite/granite-geospatial-terramind"

# Supported modalities
MODALITIES = {
    "optical": {
        "bands": ["B02", "B03", "B04", "B8A", "B11", "B12"],
        "input_shape": (6, 224, 224),
    },
    "sar": {
        "bands": ["VV", "VH"],
        "input_shape": (2, 224, 224),
    },
    "dem": {
        "bands": ["elevation"],
        "input_shape": (1, 224, 224),
    },
    "lulc": {
        "bands": ["class"],
        "input_shape": (1, 224, 224),
    },
}


def load_terramind_model(
    device: str = "cpu",
    modality_in: str = "optical",
    modality_out: str = "sar",
):
    """Load TerraMind model for cross-modal generation.

    Args:
        device: Target device.
        modality_in: Input modality ("optical", "sar", "dem", "lulc").
        modality_out: Output modality to generate.

    Returns:
        Tuple of (model, config_dict).

    Raises:
        ImportError: If dependencies are missing.
        ValueError: If modality combination is unsupported.
    """
    if not check_terramind_available():
        raise ImportError(
            "TerraMind requires terratorch. Install with: pip install terratorch\n"
            "Model: ibm-granite/granite-geospatial-terramind"
        )

    if modality_in not in MODALITIES:
        raise ValueError(
            f"Unknown input modality '{modality_in}'. "
            f"Available: {list(MODALITIES.keys())}"
        )
    if modality_out not in MODALITIES:
        raise ValueError(
            f"Unknown output modality '{modality_out}'. "
            f"Available: {list(MODALITIES.keys())}"
        )

    import torch
    from huggingface_hub import hf_hub_download

    # TerraMind uses a conditional generation architecture
    # Load the appropriate encoder-decoder pair
    ckpt_path = hf_hub_download(
        repo_id=TERRAMIND_HF_REPO,
        filename=f"terramind_{modality_in}_to_{modality_out}.pt",
    )

    model = torch.load(ckpt_path, map_location=device)
    model.eval()

    config = {
        "repo_id": TERRAMIND_HF_REPO,
        "modality_in": modality_in,
        "modality_out": modality_out,
        "input_shape": MODALITIES[modality_in]["input_shape"],
        "output_shape": MODALITIES[modality_out]["input_shape"],
        "device": device,
    }

    return model, config


def generate_cross_modal(
    model,
    image: np.ndarray,
    *,
    device: str = "cpu",
) -> np.ndarray:
    """Generate one modality from another using TerraMind.

    Example: optical → SAR (generate synthetic SAR from Sentinel-2)

    Args:
        model: TerraMind model from load_terramind_model().
        image: (C, H, W) float32 input in the source modality.
        device: Compute device.

    Returns:
        (C_out, H, W) float32 generated output in target modality.
    """
    import torch

    tensor = torch.from_numpy(image).unsqueeze(0).float().to(device)

    with torch.no_grad():
        output = model(tensor)

    if isinstance(output, (list, tuple)):
        output = output[0]

    return output.cpu().numpy().squeeze()


def optical_to_sar(
    optical: np.ndarray,
    model=None,
    device: str = "cpu",
) -> np.ndarray:
    """Convenience: generate synthetic SAR from optical Sentinel-2.

    Args:
        optical: (6, H, W) float32 Prithvi-format Sentinel-2.
        model: Pre-loaded TerraMind model (loads if None).
        device: Compute device.

    Returns:
        (2, H, W) float32 synthetic VV/VH SAR.
    """
    if model is None:
        model, _ = load_terramind_model(
            device=device, modality_in="optical", modality_out="sar",
        )
    return generate_cross_modal(model, optical, device=device)


def sar_to_optical(
    sar: np.ndarray,
    model=None,
    device: str = "cpu",
) -> np.ndarray:
    """Convenience: generate synthetic optical from SAR.

    Args:
        sar: (2, H, W) float32 VV/VH SAR.
        model: Pre-loaded TerraMind model (loads if None).
        device: Compute device.

    Returns:
        (6, H, W) float32 synthetic Sentinel-2 (Prithvi bands).
    """
    if model is None:
        model, _ = load_terramind_model(
            device=device, modality_in="sar", modality_out="optical",
        )
    return generate_cross_modal(model, sar, device=device)
