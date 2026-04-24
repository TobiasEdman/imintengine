"""
imint/fm/loaders/prithvi.py — Prithvi-EO-2.0 loader (300M and 600M)

Extracted from imint/fm/terratorch_loader.py for the registry-based
multi-model architecture. Same behavior, parameterized over HF repo
and config variant.
"""
from __future__ import annotations

import json
import os
import sys


def load_prithvi(
    pretrained: bool = True,
    num_frames: int = 1,
    img_size: int = 224,
    hf_repo: str = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL",
    weights_filename: str = "Prithvi_EO_V2_300M_TL.pt",
    variant: str = "300m",
    **kwargs,
):
    """Load Prithvi-EO-2.0 (300M or 600M) from bundled code + HF weights.

    Uses the PrithviMAE implementation bundled in imint/fm/prithvi_mae/
    and downloads weights from HuggingFace Hub on first call.

    Args:
        pretrained: If True, load HF pretrained weights.
        num_frames: Number of temporal frames (1..4). Pos-embeds are
            recomputed per num_frames, so any of (1,2,3,4) is valid.
        img_size: Input image resolution (224/256/448). Pos-embeds are
            interpolated from the pretrained grid.
        hf_repo: HuggingFace repo ID for the checkpoint.
        weights_filename: Filename inside the HF repo.
        variant: "300m" or "600m". Only affects which bundled config is
            loaded; the model geometry (depth/embed_dim) comes from the
            config file itself.

    Returns:
        PrithviMAE model in eval() mode.
    """
    import torch

    # Add bundled prithvi_mae to sys.path
    fm_dir = os.path.dirname(os.path.dirname(__file__))
    prithvi_dir = os.path.join(fm_dir, "prithvi_mae")
    if prithvi_dir not in sys.path:
        sys.path.insert(0, prithvi_dir)

    from prithvi_mae import PrithviMAE

    # Pick config file for variant (falls back to default config.json)
    config_candidates = [
        f"config_{variant}.json",
        "config.json",
    ]
    config_path = None
    for cand in config_candidates:
        p = os.path.join(prithvi_dir, cand)
        if os.path.exists(p):
            config_path = p
            break
    if config_path is None:
        raise FileNotFoundError(
            f"No Prithvi config found in {prithvi_dir} "
            f"(tried: {config_candidates})"
        )

    with open(config_path) as f:
        config = json.load(f)["pretrained_cfg"]

    model = PrithviMAE(
        img_size=img_size,
        num_frames=num_frames,
        in_chans=config["in_chans"],
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        decoder_embed_dim=config["decoder_embed_dim"],
        decoder_depth=config["decoder_depth"],
        decoder_num_heads=config["decoder_num_heads"],
        mlp_ratio=config["mlp_ratio"],
        coords_encoding=config.get("coords_encoding", []),
        coords_scale_learn=config.get("coords_scale_learn", False),
        patch_size=config["patch_size"],
    )

    if pretrained:
        # Prefer local symlink; otherwise download from HF
        local_weights = os.path.join(prithvi_dir, weights_filename)
        if os.path.exists(local_weights):
            weights_path = local_weights
        else:
            from huggingface_hub import hf_hub_download
            weights_path = hf_hub_download(hf_repo, weights_filename)

        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        # pos_embed depends on (num_frames, img_size) — drop and let model recompute
        for k in list(state_dict.keys()):
            if "pos_embed" in k:
                del state_dict[k]
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model
