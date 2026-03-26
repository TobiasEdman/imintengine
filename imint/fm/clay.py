"""
imint/fm/clay.py — Clay Foundation Model integration

Lightweight wrapper for the Clay Foundation Model (Apache 2.0),
a self-supervised Vision Transformer trained on Sentinel-1 + Sentinel-2
+ Landsat data. Produces general-purpose geospatial embeddings.

Clay complements Prithvi-EO by offering:
  - Apache 2.0 license (fully permissive, no restrictions)
  - Native Sentinel-1 SAR support (Prithvi is optical-only)
  - Lighter weight (~100M params vs 300M)
  - Good for embedding-based downstream tasks (retrieval, clustering)

This module provides:
  - check_clay_available(): Verify installation
  - load_clay_model(): Load pretrained Clay ViT from HuggingFace
  - extract_clay_embeddings(): Extract feature embeddings from S2/S1 tiles

Requirements:
    pip install clay-model   # or: pip install git+https://github.com/Clay-foundation/model.git

HuggingFace: made-with-clay/Clay
GitHub: https://github.com/Clay-foundation/model
"""
from __future__ import annotations

import numpy as np


def check_clay_available() -> bool:
    """Check if Clay model dependencies are installed."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# Clay model constants
CLAY_HF_REPO = "made-with-clay/Clay"
CLAY_BANDS_S2 = ["B02", "B03", "B04", "B08", "B8A", "B11", "B12"]
CLAY_BANDS_S1 = ["VV", "VH"]

# Clay normalisation (Sentinel-2, reflectance scale)
CLAY_S2_MEAN = np.array(
    [1369.03, 1597.68, 1741.10, 2858.69, 2916.36, 2104.89, 1594.75],
    dtype=np.float32,
)
CLAY_S2_STD = np.array(
    [2026.96, 2011.88, 2146.35, 2138.96, 2003.75, 1500.14, 1204.45],
    dtype=np.float32,
)


def load_clay_model(
    device: str = "cpu",
    pretrained: bool = True,
):
    """Load Clay Foundation Model from HuggingFace.

    Args:
        device: Target device ("cpu", "cuda", "mps").
        pretrained: Load pretrained weights (default True).

    Returns:
        Tuple of (model, metadata_dict).

    Raises:
        ImportError: If torch or Clay dependencies are missing.
    """
    if not check_clay_available():
        raise ImportError(
            "Clay model requires PyTorch. Install with: pip install torch"
        )

    import torch

    try:
        from clay.model import ClayMAE
    except ImportError:
        raise ImportError(
            "Clay model package not found. Install with:\n"
            "  pip install git+https://github.com/Clay-foundation/model.git"
        )

    if pretrained:
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(
            repo_id=CLAY_HF_REPO,
            filename="clay-v1-base.ckpt",
        )
        model = ClayMAE.load_from_checkpoint(
            ckpt_path, map_location=device,
        )
    else:
        model = ClayMAE()

    model = model.to(device)
    model.eval()

    metadata = {
        "repo_id": CLAY_HF_REPO,
        "device": device,
        "pretrained": pretrained,
        "supported_bands_s2": CLAY_BANDS_S2,
        "supported_bands_s1": CLAY_BANDS_S1,
    }

    return model, metadata


def extract_clay_embeddings(
    model,
    image: np.ndarray,
    *,
    bands: list[str] | None = None,
    device: str = "cpu",
) -> np.ndarray:
    """Extract embeddings from a satellite image using Clay.

    Args:
        model: Clay model from load_clay_model().
        image: (C, H, W) float32 reflectance [0, 1] or DN.
        bands: Band names matching image channels.
               Default: CLAY_BANDS_S2 (7 S2 bands).
        device: Compute device.

    Returns:
        (D,) or (N, D) embedding array, where D is the embedding dimension.
    """
    import torch

    if bands is None:
        bands = CLAY_BANDS_S2

    # Normalise
    c, h, w = image.shape
    mean = CLAY_S2_MEAN[:c].reshape(c, 1, 1)
    std = CLAY_S2_STD[:c].reshape(c, 1, 1)

    if image.max() <= 1.0:
        image = image * 10000.0  # reflectance → DN

    image_norm = (image - mean) / std

    # To tensor
    tensor = torch.from_numpy(image_norm).unsqueeze(0).to(device)

    with torch.no_grad():
        # Clay encoder returns patch embeddings
        # Aggregate via global average pooling
        features = model.encoder(tensor)
        if isinstance(features, (list, tuple)):
            features = features[-1]  # Last layer

        # Global average pool: (B, N, D) → (B, D) or (B, D, H, W) → (B, D)
        if features.dim() == 3:
            embedding = features.mean(dim=1)
        elif features.dim() == 4:
            embedding = features.mean(dim=[-2, -1])
        else:
            embedding = features

    return embedding.cpu().numpy().squeeze()


def prithvi_bands_to_clay(bands: dict[str, np.ndarray]) -> np.ndarray:
    """Convert Prithvi 6-band dict to Clay 7-band stack.

    Maps: B02, B03, B04, B8A, B11, B12 → Clay expects B02..B12 + B08.
    B08 is approximated from B8A (similar NIR, different bandwidth).

    Args:
        bands: Dict with Prithvi band names as keys.

    Returns:
        (7, H, W) float32 array in Clay band order.
    """
    b08_approx = bands.get("B08", bands.get("B8A"))
    return np.stack([
        bands["B02"],
        bands["B03"],
        bands["B04"],
        b08_approx,     # B08 (approx from B8A if needed)
        bands["B8A"],
        bands["B11"],
        bands["B12"],
    ], axis=0)
