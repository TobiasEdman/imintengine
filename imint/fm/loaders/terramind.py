"""
imint/fm/loaders/terramind.py — TerraMind v1.0-base loader

TerraMind (IBM + ESA, arXiv:2504.11171) is a multi-modal foundation
model that accepts a dict of {modality: tensor}. Loaded through the
terratorch BACKBONE_REGISTRY so the pretrained weights + band-specific
projection layers are handled by terratorch.

Modalities wired here:
    S2L2A — 6 bands matching Prithvi's order (B02, B03, B04, B8A, B11, B12)
    S1GRD — 2 bands (VV, VH)

To add DEM later, extend ``modalities`` and pass a DEM tensor at forward.

References:
    HF:  https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base
    Code: https://github.com/IBM/terramind
"""
from __future__ import annotations


# Band name aliases required by terratorch's TerraMind band-subsetter.
# Our 6-band spectral order maps to these human-readable names.
_TERRAMIND_S2_BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]


def load_terramind(
    pretrained: bool = True,
    num_frames: int = 1,
    img_size: int = 224,
    variant: str = "terramind_v1_base",
    modalities: list[str] | None = None,
    bands: dict[str, list[str]] | None = None,
    **kwargs,
):
    """Load TerraMind v1 via the terratorch BACKBONE_REGISTRY.

    Args:
        pretrained: Load HF pretrained weights.
        num_frames: Must be 1 — TerraMind is single-date. Batch your
            temporal data before calling.
        img_size: Input resolution. Default 224 (native). Non-default
            sizes rely on pos-embed interpolation inside terratorch.
        variant: Which TerraMind checkpoint. Valid registry keys:
            ``terramind_v1_tiny`` / ``_small`` / ``_base`` / ``_large``,
            plus ``_tim`` suffixed variants for Thinking-in-Modalities.
        modalities: List of modality keys terratorch understands
            (``S2L2A``, ``S2L1C``, ``S1GRD``, ``S1RTC``, ``DEM``, ``RGB``).
            Defaults to ``["S2L2A", "S1GRD"]``.
        bands: Optional dict mapping modality → ordered band-name list
            for subsetting. Default subsets S2L2A to our 6 Prithvi-order
            bands so we can pass the existing ``spectral`` tensor
            without extra fetches.

    Returns:
        nn.Module in eval() mode. Forward call expects a dict input:

            model({"S2L2A": (B, 6, H, W), "S1GRD": (B, 2, H, W)})

        Output shape is ``(B, N_tokens, embed_dim)`` — a token sequence,
        not a spatial feature map. The UPerNet wrapper reshapes to a
        ``(B, embed_dim, grid_h, grid_w)`` spatial grid downstream.
    """
    if num_frames != 1:
        raise ValueError(
            f"TerraMind is single-date (num_frames=1), got {num_frames}. "
            f"Collapse temporal dimension before calling."
        )

    try:
        from terratorch import BACKBONE_REGISTRY
    except ImportError as e:
        raise ImportError(
            "TerraMind requires terratorch. Install with:\n"
            "  pip install terratorch"
        ) from e

    resolved_modalities = modalities or ["S2L2A", "S1GRD"]
    resolved_bands = bands or {"S2L2A": list(_TERRAMIND_S2_BANDS)}

    model = BACKBONE_REGISTRY.build(
        variant,
        pretrained=pretrained,
        modalities=resolved_modalities,
        bands=resolved_bands,
    )
    model.eval()
    return model
