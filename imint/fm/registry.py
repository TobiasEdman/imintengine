"""
imint/fm/registry.py — Foundation model registry

Declarative spec per model family. Adding a new model = adding a dict
entry, not writing new trainer code.

Usage:
    from imint.fm.registry import build_backbone, MODEL_CONFIGS

    model, spec = build_backbone("prithvi_300m", num_frames=4, img_size=448)
    # spec.embed_dim, spec.feature_indices, etc. available for decoder setup
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal


@dataclass(frozen=True)
class ModelSpec:
    """Declarative specification of a foundation model.

    Attributes:
        name: Registry key (e.g. "prithvi_300m").
        family: Model family — "prithvi", "clay", "terramind", "croma".
        description: Human-readable description.

        embed_dim: Encoder embedding dimension (for decoder sizing).
        depth: Number of transformer blocks.
        feature_indices: 4 block indices for UPerNet multi-scale features.
        patch_size: Patch size in pixels (8 for Clay, 16 for others).

        input_bands: {modality: batch_key} — which dataset keys to consume.
            e.g. {"s2": "spectral"} for Prithvi.
        supports_temporal: True if encoder handles T dimension natively.
        native_num_frames: Allowed frame counts (e.g. (1,2,3,4) for Prithvi).
        supports_coords: True if encoder accepts temporal/location coords.

        loader_fn: Callable that loads the backbone model.
            Signature: (pretrained, num_frames, img_size, **kwargs) -> nn.Module
        loader_kwargs: Family-level defaults for the loader.

        normalizer_family: Key into imint.fm.normalize.NORMALIZERS.
        decoder_compatible: Which decoder architectures work with this model.
    """

    name: str
    family: Literal["prithvi", "clay", "terramind", "croma", "thor"]
    description: str

    embed_dim: int
    depth: int
    feature_indices: tuple[int, ...]
    patch_size: int

    input_bands: dict[str, str]
    supports_temporal: bool
    native_num_frames: tuple[int, ...]
    supports_coords: bool

    loader_fn: Callable[..., Any]
    normalizer_family: str

    loader_kwargs: dict = field(default_factory=dict)
    decoder_compatible: tuple[str, ...] = ("upernet",)


def default_feature_indices(depth: int, n_levels: int = 4) -> tuple[int, ...]:
    """Evenly-spaced feature indices for UPerNet.

    For depth=24 with 4 levels: (5, 11, 17, 23) — Prithvi default.
    For depth=32 with 4 levels: (7, 15, 23, 31) — Prithvi-600M.
    For depth=12 with 4 levels: (2, 5, 8, 11) — TerraMind-B.
    """
    step = depth // n_levels
    return tuple(range(step - 1, depth, step))[:n_levels]


# Lazy loader imports — avoids forcing optional deps at import time
def _load_prithvi(**kwargs):
    from imint.fm.loaders.prithvi import load_prithvi
    return load_prithvi(**kwargs)


def _load_terramind(**kwargs):
    from imint.fm.loaders.terramind import load_terramind
    return load_terramind(**kwargs)


def _load_clay(**kwargs):
    from imint.fm.loaders.clay import load_clay
    return load_clay(**kwargs)


def _load_croma(**kwargs):
    from imint.fm.loaders.croma import load_croma
    return load_croma(**kwargs)


def _load_thor(**kwargs):
    from imint.fm.loaders.thor import load_thor
    return load_thor(**kwargs)


# ── Registry ─────────────────────────────────────────────────────────────────

MODEL_CONFIGS: dict[str, ModelSpec] = {
    "prithvi_300m": ModelSpec(
        name="prithvi_300m",
        family="prithvi",
        description="Prithvi-EO-2.0 300M (temporal-location aware)",
        embed_dim=1024,
        depth=24,
        feature_indices=(5, 11, 17, 23),
        patch_size=16,
        input_bands={"s2": "spectral"},
        supports_temporal=True,
        native_num_frames=(1, 2, 3, 4),
        supports_coords=True,
        loader_fn=_load_prithvi,
        loader_kwargs={
            "hf_repo": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL",
            "weights_filename": "Prithvi_EO_V2_300M_TL.pt",
            "variant": "300m",
        },
        normalizer_family="prithvi",
    ),
    "prithvi_600m": ModelSpec(
        name="prithvi_600m",
        family="prithvi",
        description="Prithvi-EO-2.0 600M (temporal-location aware, patch=14)",
        embed_dim=1280,
        depth=32,
        feature_indices=(7, 15, 23, 31),
        # Prithvi-600M uses patch_size=14 (not 16 like 300M). Callers that
        # size inputs must use a multiple of 14 (e.g. 224, 448, 560).
        patch_size=14,
        input_bands={"s2": "spectral"},
        supports_temporal=True,
        native_num_frames=(1, 2, 3, 4),
        supports_coords=True,
        loader_fn=_load_prithvi,
        loader_kwargs={
            "hf_repo": "ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL",
            "weights_filename": "Prithvi_EO_V2_600M_TL.pt",
            "variant": "600m",
        },
        normalizer_family="prithvi",
    ),
    "terramind_v1_base": ModelSpec(
        name="terramind_v1_base",
        family="terramind",
        description=(
            "TerraMind v1.0-base (any-to-any multi-modal S2+S1 FM, "
            "IBM/ESA, arXiv:2504.11171). ViT-Base, 196 tokens × 768 dim "
            "output at 224×224/patch=16. Consumes dict inputs."
        ),
        embed_dim=768,
        depth=12,
        feature_indices=(2, 5, 8, 11),
        patch_size=16,
        # Modality keys use TerraMind's native naming to minimize surprise
        # when reading dataset wiring; our unified_dataset emits the same
        # tensors as Prithvi S2 + our S1 enrichment.
        input_bands={"S2L2A": "spectral", "S1GRD": "s1_vv_vh"},
        supports_temporal=False,
        native_num_frames=(1,),
        supports_coords=False,
        loader_fn=_load_terramind,
        loader_kwargs={
            "variant": "terramind_v1_base",
            "modalities": ["S2L2A", "S1GRD"],
            "bands": {
                "S2L2A": ["BLUE", "GREEN", "RED", "NIR_NARROW",
                          "SWIR_1", "SWIR_2"],
            },
        },
        normalizer_family="terramind",
    ),
    "clay_v1_5": ModelSpec(
        name="clay_v1_5",
        family="clay",
        description=(
            "Clay v1.5 MAE ViT-L (10-band S2L2A, patch=8, 311M encoder). "
            "Requires rededge enrichment (B05/B06/B07)."
        ),
        embed_dim=1024,
        depth=24,
        feature_indices=(5, 11, 17, 23),
        patch_size=8,
        # Clay consumes a stacked tensor built by
        # imint.fm.loaders.clay.build_s2_clay_tensor from
        # spectral + b08 + rededge. The dataset emits it under the
        # "s2_clay" key when a Clay model is requested.
        input_bands={"s2_clay": "s2_clay"},
        supports_temporal=False,
        native_num_frames=(1,),
        supports_coords=True,
        loader_fn=_load_clay,
        normalizer_family="clay",
    ),
    "croma_base": ModelSpec(
        name="croma_base",
        family="croma",
        description="CROMA (cross-modal S1+S2 contrastive pretraining)",
        embed_dim=768,
        depth=12,
        feature_indices=(2, 5, 8, 11),
        patch_size=8,
        input_bands={"s2_full": "s2_full_12band", "s1": "s1_vv_vh"},
        supports_temporal=False,
        native_num_frames=(1,),
        supports_coords=False,
        loader_fn=_load_croma,
        normalizer_family="croma",
    ),
    "thor_v1_base": ModelSpec(
        name="thor_v1_base",
        family="thor",
        description=(
            "THOR v1.0-base — compute-adaptive FlexiViT for "
            "Sentinel-1/2/3 (NR/UiT/ESA Φ-lab, arXiv:2601.16011)"
        ),
        # Provisional — "base" ViT conventions. Verify against checkpoint.
        embed_dim=768,
        depth=12,
        feature_indices=(2, 5, 8, 11),
        patch_size=8,  # FlexiViT default; adjustable at inference
        input_bands={"s2": "spectral", "s1": "s1_vv_vh"},
        supports_temporal=False,
        native_num_frames=(1,),
        supports_coords=False,
        loader_fn=_load_thor,
        loader_kwargs={
            "terratorch_name": "thor_v1_base",
            "model_bands": ["BLUE", "GREEN", "RED", "NIR_BROAD",
                            "SWIR_1", "SWIR_2", "VV", "VH"],
            "ground_covers": [2880],
            "flexivit_patch_size_seqs": [8],
        },
        normalizer_family="thor",
    ),
}


# Legacy backbone name → registry key
LEGACY_BACKBONE_ALIAS = {
    "prithvi_eo_v2_300m_tl": "prithvi_300m",
    "prithvi_eo_v2_600m_tl": "prithvi_600m",
}


def resolve_backbone_name(
    backbone_name: str | None,
    legacy_backbone: str | None = None,
) -> str:
    """Resolve registry key from config fields.

    Prefers `backbone_name` (new field). Falls back to `backbone` via
    legacy alias map.
    """
    if backbone_name is not None:
        if backbone_name not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown backbone_name: {backbone_name!r}. "
                f"Available: {sorted(MODEL_CONFIGS.keys())}"
            )
        return backbone_name

    if legacy_backbone is not None:
        alias = LEGACY_BACKBONE_ALIAS.get(legacy_backbone)
        if alias is not None:
            return alias
        # Allow direct registry keys in backbone field too
        if legacy_backbone in MODEL_CONFIGS:
            return legacy_backbone

    # Default
    return "prithvi_300m"


def build_backbone(
    name: str,
    *,
    num_frames: int = 1,
    img_size: int = 224,
    pretrained: bool = True,
    **overrides,
):
    """Build a foundation model backbone by registry key.

    Returns:
        (model, spec) tuple. Model is nn.Module; spec is ModelSpec.
    """
    if name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {name!r}. "
            f"Available: {sorted(MODEL_CONFIGS.keys())}"
        )

    spec = MODEL_CONFIGS[name]

    if num_frames not in spec.native_num_frames:
        raise ValueError(
            f"Model {name} supports num_frames={spec.native_num_frames}, "
            f"got {num_frames}. Collapse to 1 for single-date models."
        )

    kwargs = {**spec.loader_kwargs, **overrides}
    model = spec.loader_fn(
        pretrained=pretrained,
        num_frames=num_frames,
        img_size=img_size,
        **kwargs,
    )

    return model, spec
