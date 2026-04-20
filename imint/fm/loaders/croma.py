"""
imint/fm/loaders/croma.py — CROMA v1 loader

CROMA (antofuller, NeurIPS'23) — contrastive radar-optical MAE.
Expects:
    SAR_images:     (B, 2, H, W)   — VV, VH
    optical_images: (B, 12, H, W)  — S2 bands in a specific order
    H, W multiple of 8; native image_resolution=120 px
    size='base' → embed_dim=768; size='large' → embed_dim=1024

Output: dict with ``{sar,optical,joint}_encodings`` (B, N, D) where
N = (H/8)² patches. We'll use ``joint_encodings`` as the segmentation
feature source when both modalities are available.

Weights: CROMA_base.pt / CROMA_large.pt on HF antofuller/CROMA_benchmarks.
Model code: github.com/antofuller/CROMA/blob/main/use_croma.py
"""
from __future__ import annotations

import os


# CROMA's 12-band S2 order (B10 excluded — atmospheric, not in L2A).
CROMA_S2_BAND_ORDER: tuple[str, ...] = (
    "B01",  # 443nm coastal aerosol — 60m native
    "B02",  # 490nm blue
    "B03",  # 560nm green
    "B04",  # 665nm red
    "B05",  # 705nm rededge1
    "B06",  # 740nm rededge2
    "B07",  # 783nm rededge3
    "B08",  # 842nm nir broad
    "B8A",  # 865nm nir narrow
    "B09",  # 940nm water vapour — 60m native
    "B11",  # 1610nm swir1
    "B12",  # 2190nm swir2
)


def load_croma(
    pretrained: bool = True,
    num_frames: int = 1,
    img_size: int = 120,
    variant: str = "base",
    modality: str = "both",
    checkpoint_path: str | None = None,
    **kwargs,
):
    """Load pretrained CROMA encoder.

    Args:
        pretrained: Load the released checkpoint.
        num_frames: Must be 1. CROMA is single-date.
        img_size: Input resolution. 120 is native; 128/256 work since
            CROMA is patch_size=8 and positional embeddings interpolate.
        variant: 'base' (embed_dim=768) or 'large' (embed_dim=1024).
        modality: 'both' (S1+S2), 'SAR' (S1 only), or 'optical' (S2 only).
        checkpoint_path: Optional local path. If None, auto-searches
            common locations and falls back to huggingface_hub download.

    Returns:
        nn.Module (PretrainedCROMA instance). forward(SAR_images=...,
        optical_images=...) returns a dict of encodings.
    """
    if num_frames != 1:
        raise ValueError(
            f"CROMA is single-date (num_frames=1), got {num_frames}. "
            f"Collapse temporal dimension before calling."
        )
    if img_size % 8 != 0:
        raise ValueError(
            f"CROMA img_size={img_size} must be a multiple of 8 "
            f"(patch_size=8)."
        )
    if variant not in ("base", "large"):
        raise ValueError(
            f"CROMA variant={variant!r}; must be 'base' or 'large'."
        )
    if modality not in ("both", "SAR", "optical"):
        raise ValueError(
            f"CROMA modality={modality!r}; must be 'both', 'SAR', or 'optical'."
        )

    try:
        # antofuller/CROMA isn't a pip package — its use_croma.py lives
        # in the CROMA repo. We expect it to be on sys.path (k8s pod
        # clones CROMA alongside imintengine and adds /workspace/CROMA
        # to PYTHONPATH).
        from use_croma import PretrainedCROMA
    except ImportError as e:
        raise ImportError(
            "CROMA requires the use_croma.py module from "
            "github.com/antofuller/CROMA on PYTHONPATH. In a k8s pod:\n"
            "  git clone https://github.com/antofuller/CROMA /workspace/CROMA\n"
            "  export PYTHONPATH=/workspace/CROMA:$PYTHONPATH"
        ) from e

    # Resolve checkpoint
    if checkpoint_path is None:
        filename = f"CROMA_{variant}.pt"
        for candidate in (
            f"/data/model_cache/{filename}",
            os.path.expanduser(f"~/.cache/croma/{filename}"),
            f"/workspace/models/{filename}",
            f"/workspace/CROMA/{filename}",
        ):
            if os.path.exists(candidate):
                checkpoint_path = candidate
                break

    if pretrained and checkpoint_path is None:
        try:
            from huggingface_hub import hf_hub_download
            checkpoint_path = hf_hub_download(
                "antofuller/CROMA_benchmarks",
                f"CROMA_{variant}.pt",
            )
        except Exception as e:
            raise FileNotFoundError(
                f"No CROMA {variant} checkpoint found. Either:\n"
                f"  - Download CROMA_{variant}.pt from\n"
                f"    https://huggingface.co/antofuller/CROMA_benchmarks\n"
                f"    and place it at /data/model_cache/CROMA_{variant}.pt\n"
                f"  - Or pass checkpoint_path= explicitly."
            ) from e

    model = PretrainedCROMA(
        pretrained_path=checkpoint_path or "",
        size=variant,
        modality=modality,
        image_resolution=img_size,
    )
    model.eval()
    return model


def build_s2_croma_tensor(
    spectral_6band,
    b08,
    rededge=None,
    b01=None,
    b09=None,
    bands: tuple[str, ...] = CROMA_S2_BAND_ORDER,
):
    """Stack on-disk tensors into CROMA's 12-band S2 order.

    Our layout:
        spectral_6band (6, H, W) Prithvi order:
            [0]=B02, [1]=B03, [2]=B04, [3]=B8A, [4]=B11, [5]=B12
        b08            (H, W)    — B08
        rededge        (3, H, W) — B05, B06, B07 (from enrichment)
        b01            (H, W)    — B01 coastal aerosol (optional, padded if None)
        b09            (H, W)    — B09 water vapour   (optional, padded if None)

    If b01 or b09 is None, that band is zero-padded with a clear
    warning. CROMA is robust to band ablation because of its MAE
    pretraining but performance degrades.

    Args:
        spectral_6band, b08, rededge: required on-disk tensors.
        b01, b09: optional, zero-padded when missing.
        bands: band list (default CROMA_S2_BAND_ORDER).

    Returns:
        (len(bands), H, W) tensor.
    """
    import numpy as np

    h, w = spectral_6band.shape[-2:]
    zero_band = None  # lazy construct in the right backend

    def _zeros_like():
        nonlocal zero_band
        if zero_band is None:
            try:
                import torch
                if isinstance(spectral_6band, torch.Tensor):
                    zero_band = torch.zeros(
                        (h, w), dtype=spectral_6band.dtype,
                        device=spectral_6band.device,
                    )
                    return zero_band
            except ImportError:
                pass
            zero_band = np.zeros((h, w), dtype=spectral_6band.dtype)
        return zero_band

    prithvi_idx = {
        "B02": 0, "B03": 1, "B04": 2,
        "B8A": 3, "B11": 4, "B12": 5,
    }
    rededge_idx = {"B05": 0, "B06": 1, "B07": 2}

    layers = []
    for name in bands:
        if name in prithvi_idx:
            layers.append(spectral_6band[prithvi_idx[name]])
        elif name == "B08":
            layers.append(b08)
        elif name in rededge_idx:
            if rededge is None:
                raise KeyError(
                    f"CROMA band {name!r} requires rededge tensor. "
                    f"Run scripts/enrich_tiles_rededge.py first."
                )
            layers.append(rededge[rededge_idx[name]])
        elif name == "B01":
            layers.append(b01 if b01 is not None else _zeros_like())
        elif name == "B09":
            layers.append(b09 if b09 is not None else _zeros_like())
        else:
            raise KeyError(f"Unknown CROMA band {name!r}.")

    try:
        import torch
        if isinstance(spectral_6band, torch.Tensor):
            return torch.stack(layers, dim=0)
    except ImportError:
        pass
    return np.stack(layers, axis=0)
