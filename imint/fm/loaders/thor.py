"""
imint/fm/loaders/thor.py — THOR v1.0 loader

THOR: compute-adaptive FlexiViT foundation model for Sentinel-1/2/3
(Norwegian Computing Center / UiT / ESA Φ-lab).

Paper: https://arxiv.org/abs/2601.16011
Weights: https://huggingface.co/FM4CS/THOR-1.0-base
Code: https://github.com/FM4CS/THOR

Loads via the terratorch BACKBONE_REGISTRY using the `thor_terratorch_ext`
extension. Users must `pip install thor-terratorch-ext` separately — we
don't take a hard dependency on it.
"""
from __future__ import annotations


DEFAULT_BANDS = ["BLUE", "GREEN", "RED", "NIR_BROAD", "SWIR_1", "SWIR_2", "VV", "VH"]


def load_thor(
    pretrained: bool = True,
    num_frames: int = 1,
    img_size: int = 224,
    terratorch_name: str = "thor_v1_base",
    model_bands: list[str] | None = None,
    ground_covers: list[int] | None = None,
    flexivit_patch_size_seqs: list[int] | None = None,
    **kwargs,
):
    """Load THOR v1.0 base via terratorch + thor_terratorch_ext.

    Args:
        pretrained: If True, load HF weights (FM4CS/THOR-1.0-base).
        num_frames: THOR is single-date; only 1 is valid.
        img_size: Input resolution — FlexiViT handles any multiple of
            the patch size.
        terratorch_name: Name in terratorch registry.
        model_bands: Ordered list of band names the model consumes.
            Defaults to S2 (B,G,R,NIR,SWIR1,SWIR2) + S1 (VV,VH).
        ground_covers: Ground sampling distances in meters (default 2880
            = 288 pixels × 10 m, one "patch window").
        flexivit_patch_size_seqs: Patch sizes in pixels (default 8).

    Returns:
        nn.Module backbone in eval() mode.
    """
    if num_frames != 1:
        raise ValueError(
            f"THOR is single-date (num_frames=1), got num_frames={num_frames}. "
            f"Collapse temporal dimension before calling."
        )

    try:
        import thor_terratorch_ext  # noqa: F401 — registers THOR in BACKBONE_REGISTRY
    except ImportError as e:
        raise ImportError(
            "THOR requires thor-terratorch-ext. Install with:\n"
            "  pip install git+https://github.com/FM4CS/THOR.git#subdirectory=terratorch_ext"
        ) from e

    from terratorch import BACKBONE_REGISTRY

    model = BACKBONE_REGISTRY.build(
        terratorch_name,
        pretrained=pretrained,
        model_bands=model_bands or DEFAULT_BANDS,
        input_params=dict(
            ground_covers=ground_covers or [2880],
            flexivit_patch_size_seqs=flexivit_patch_size_seqs or [8],
        ),
    )
    model.eval()
    return model
