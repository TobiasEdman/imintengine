"""
imint/fm/loaders/clay.py — Clay v1.5 loader

Clay v1.5 is a masked autoencoder ViT pretrained on many satellite
sensors. For Sentinel-2 L2A it expects 10 bands at 8-pixel patches
with per-band wavelength metadata.

Architecture (per release notes):
    embed_dim = 1024  depth = 24  num_heads = 16
    patch_size = 8    image_size = 256    params = 632M (311M encoder)

Sentinel-2 band list (full Clay spec, 10 bands in this order):
    blue, green, red, rededge1, rededge2, rededge3, nir, nir08, swir16, swir22
    = B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12

Our on-disk layout:
    spectral (6-band Prithvi order):  B02, B03, B04, B8A, B11, B12
    b08 (enrichment):                 B08
    rededge (enrichment):             B05, B06, B07

Total 10 bands matching Clay's full S2L2A spec. If ``rededge`` is missing
(older tile), ``build_s2_clay_tensor`` raises a clear error pointing
at the enrichment script.

Code and weights live in Clay-foundation/model; install with:
    pip install git+https://github.com/Clay-foundation/model.git

References:
    https://clay-foundation.github.io/model/
    https://github.com/Clay-foundation/model
    https://huggingface.co/made-with-clay/Clay/
"""
from __future__ import annotations

import os


# Clay Sentinel-2 band catalog (name → wavelength in micrometres + normalization).
# Values transcribed from Clay-foundation/model/configs/metadata.yaml.
CLAY_S2_BAND_META: dict[str, dict[str, float]] = {
    "blue":      {"wavelength": 0.493, "mean": 1105.0, "std": 1809.0},
    "green":     {"wavelength": 0.560, "mean": 1355.0, "std": 1757.0},
    "red":       {"wavelength": 0.665, "mean": 1552.0, "std": 1888.0},
    "rededge1":  {"wavelength": 0.704, "mean": 1887.0, "std": 1870.0},
    "rededge2":  {"wavelength": 0.740, "mean": 2422.0, "std": 1732.0},
    "rededge3":  {"wavelength": 0.783, "mean": 2630.0, "std": 1697.0},
    "nir":       {"wavelength": 0.842, "mean": 2743.0, "std": 1742.0},
    "nir08":     {"wavelength": 0.865, "mean": 2785.0, "std": 1648.0},
    "swir16":    {"wavelength": 1.610, "mean": 2388.0, "std": 1470.0},
    "swir22":    {"wavelength": 2.190, "mean": 1835.0, "std": 1379.0},
}

# Clay's full 10-band S2L2A spec — matches what the pretrained weights
# expect. This is the default stack order used by build_s2_clay_tensor.
CLAY_S2_BAND_ORDER: tuple[str, ...] = (
    "blue", "green", "red",
    "rededge1", "rededge2", "rededge3",
    "nir", "nir08", "swir16", "swir22",
)

# Kept for backwards compat — use CLAY_S2_BAND_ORDER going forward.
DEFAULT_AVAILABLE_BANDS: tuple[str, ...] = CLAY_S2_BAND_ORDER


def load_clay(
    pretrained: bool = True,
    num_frames: int = 1,
    img_size: int = 256,
    checkpoint_path: str | None = None,
    **kwargs,
):
    """Load Clay v1.5 MAE encoder.

    Args:
        pretrained: Load the released checkpoint. Required for downstream
            use — Clay without pretrained weights is useless.
        num_frames: Clay is single-date; only 1 is valid. Collapse
            temporal stacks before calling.
        img_size: Input resolution (default 256, native). The encoder
            supports arbitrary sizes that are multiples of patch_size=8.
        checkpoint_path: Optional local path to clay-v1.5.ckpt. If None,
            looks in common locations (HF cache, /data/model_cache).

    Returns:
        nn.Module — the Clay encoder (masked autoencoder with
        mask_ratio=0 for feature extraction).

    Raises:
        ImportError: If clay-foundation/model isn't installed.
        FileNotFoundError: If pretrained=True and no checkpoint found.
    """
    if num_frames != 1:
        raise ValueError(
            f"Clay v1.5 is single-date (num_frames=1), got {num_frames}. "
            f"Collapse temporal dimension before calling."
        )

    try:
        from claymodel.module import ClayMAEModule
    except ImportError as e:
        raise ImportError(
            "Clay v1.5 requires the clay-foundation/model package. "
            "Install with:\n"
            "  pip install git+https://github.com/Clay-foundation/model.git"
        ) from e

    # Resolve checkpoint path
    if checkpoint_path is None:
        for candidate in (
            "/data/model_cache/clay-v1.5.ckpt",
            os.path.expanduser("~/.cache/clay/clay-v1.5.ckpt"),
            "/workspace/models/clay-v1.5.ckpt",
        ):
            if os.path.exists(candidate):
                checkpoint_path = candidate
                break

    if pretrained and checkpoint_path is None:
        # Try downloading from HuggingFace Hub
        try:
            from huggingface_hub import hf_hub_download
            checkpoint_path = hf_hub_download(
                "made-with-clay/Clay",
                "clay-v1.5.ckpt",
            )
        except Exception as e:
            raise FileNotFoundError(
                "No Clay v1.5 checkpoint found. Either:\n"
                "  - Download clay-v1.5.ckpt from https://huggingface.co/made-with-clay/Clay\n"
                "    and place it at /data/model_cache/clay-v1.5.ckpt\n"
                "  - Or pass checkpoint_path= explicitly."
            ) from e

    module = ClayMAEModule.load_from_checkpoint(
        checkpoint_path, strict=False,
    )
    module.eval()
    # Return the encoder (what we actually use for feature extraction);
    # the full module also contains the decoder we don't need.
    return module.model


def build_s2_clay_tensor(
    spectral_6band,
    b08,
    rededge=None,
    bands: tuple[str, ...] = CLAY_S2_BAND_ORDER,
):
    """Stack our on-disk tensors into Clay's expected S2 input.

    Our layout:
        spectral_6band (6, H, W) Prithvi order:
            [0]=B02 blue, [1]=B03 green, [2]=B04 red,
            [3]=B8A nir08, [4]=B11 swir16, [5]=B12 swir22
        b08            (H, W) — B08 (nir)
        rededge        (3, H, W) — B05, B06, B07 in that order

    Args:
        spectral_6band: (6, H, W) Prithvi-order spectral for one frame.
        b08: (H, W) B08 broad-NIR for same frame.
        rededge: (3, H, W) B05/B06/B07 for same frame. Required when
            ``bands`` includes rededge1/rededge2/rededge3; may be None
            if caller has explicitly restricted ``bands`` to the 7
            non-rededge bands.
        bands: Ordered list of Clay band names. Defaults to the full
            10-band Clay spec (CLAY_S2_BAND_ORDER).

    Returns:
        (len(bands), H, W) tensor. dtype follows ``spectral_6band``.

    Raises:
        KeyError: Requested band not available given provided inputs.
    """
    import numpy as np

    prithvi_idx = {
        "blue": 0, "green": 1, "red": 2,
        "nir08": 3, "swir16": 4, "swir22": 5,
    }
    rededge_idx = {"rededge1": 0, "rededge2": 1, "rededge3": 2}

    layers = []
    for name in bands:
        if name in prithvi_idx:
            layers.append(spectral_6band[prithvi_idx[name]])
        elif name == "nir":
            layers.append(b08)
        elif name in rededge_idx:
            if rededge is None:
                raise KeyError(
                    f"Clay band {name!r} requires the ``rededge`` tensor "
                    f"(B05/B06/B07). Run scripts/enrich_tiles_rededge.py "
                    f"on this tile first."
                )
            layers.append(rededge[rededge_idx[name]])
        else:
            raise KeyError(
                f"Unknown Clay band {name!r}. Known: {tuple(CLAY_S2_BAND_META)}."
            )

    try:
        import torch
        if isinstance(spectral_6band, torch.Tensor):
            return torch.stack(layers, dim=0)
    except ImportError:
        pass
    return np.stack(layers, axis=0)


def get_clay_wavelengths(bands: tuple[str, ...] = DEFAULT_AVAILABLE_BANDS):
    """Return a 1D torch tensor of per-band wavelengths in µm.

    Clay's dynamic embedding block takes this alongside the input tensor
    so embeddings are built from wavelength, not from band index. See
    Clay-foundation/model/configs/metadata.yaml.
    """
    import torch
    return torch.tensor(
        [CLAY_S2_BAND_META[b]["wavelength"] for b in bands],
        dtype=torch.float32,
    )


def get_clay_norm(bands: tuple[str, ...] = DEFAULT_AVAILABLE_BANDS):
    """Return (mean, std) tensors for the given Clay band subset.

    Clay was trained on DN-scale Sentinel-2 (reflectance × 10000), so
    the normalizer is ``(x * 10000 - mean) / std`` when ``x`` is in
    reflectance [0, 1].
    """
    import torch
    mean = torch.tensor(
        [CLAY_S2_BAND_META[b]["mean"] for b in bands],
        dtype=torch.float32,
    )
    std = torch.tensor(
        [CLAY_S2_BAND_META[b]["std"] for b in bands],
        dtype=torch.float32,
    )
    return mean, std
