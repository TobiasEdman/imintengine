"""
imint/fm/terratorch_loader.py — Shared foundation model loading utilities

Provides common infrastructure for foundation model analyzers
(Prithvi-EO-2.0, THOR, etc.).

Loading priority:
  1. TerraTorch (pip install terratorch) — if available
  2. Direct HuggingFace loading — uses prithvi_mae.py bundled in imint/fm/

All torch imports are lazy (inside functions) to keep the core IMINT Engine
lightweight when FM analyzers are disabled.
"""
from __future__ import annotations

import os
import json
import math
import numpy as np


def check_terratorch_available() -> bool:
    """Check if terratorch and torch are importable.

    Returns:
        True if both terratorch and torch are available, False otherwise.
    """
    try:
        import torch  # noqa: F401
        import terratorch  # noqa: F401
        return True
    except ImportError:
        return False


def check_torch_available() -> bool:
    """Check if torch is importable (terratorch not required).

    Returns:
        True if torch is available, False otherwise.
    """
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def get_device(preferred: str | None = None) -> str:
    """Detect the best available compute device.

    Priority: explicit preference → CUDA → MPS (Apple Silicon) → CPU.

    Args:
        preferred: Optional device string ("cuda", "mps", "cpu").
            If provided and available, used directly.

    Returns:
        Device string suitable for ``torch.device()``.
    """
    import torch

    if preferred is not None:
        preferred = preferred.lower()
        if preferred == "cuda" and torch.cuda.is_available():
            return "cuda"
        if preferred == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if preferred == "cpu":
            return "cpu"
        # Fall through to auto-detect if preferred device not available

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_backbone(model_name: str, pretrained: bool = True):
    """Load a foundation model backbone.

    Tries TerraTorch first; falls back to direct HuggingFace loading
    for Prithvi models.

    Args:
        model_name: Registry key, e.g. "prithvi_eo_v2_300m_tl".
        pretrained: Whether to load pretrained weights.

    Returns:
        A PyTorch nn.Module (the backbone model).

    Raises:
        ImportError: If neither terratorch nor torch is installed.
    """
    # Try TerraTorch first
    try:
        from terratorch.registry import BACKBONE_REGISTRY
        model = BACKBONE_REGISTRY.build(model_name, pretrained=pretrained)
        model.eval()
        return model
    except ImportError:
        pass

    # Fallback: direct HuggingFace loading for Prithvi
    if "prithvi" in model_name.lower():
        return _load_prithvi_from_hf(pretrained=pretrained)

    raise ImportError(
        "terratorch is required for non-Prithvi foundation models. "
        "Install with: pip install terratorch"
    )


def _load_prithvi_from_hf(pretrained: bool = True, num_frames: int = 1):
    """Load Prithvi-EO-2.0 directly from bundled code + HuggingFace weights.

    Uses prithvi_mae.py bundled in imint/fm/prithvi_mae/ and downloads
    weights from HuggingFace Hub.

    Args:
        pretrained: Whether to load pretrained weights.
        num_frames: Number of temporal frames (1 for single-date, 4 for
            multitemporal seasonal). Positional embeddings are recomputed
            for the target num_frames.

    Returns:
        PrithviMAE model ready for inference.
    """
    import sys
    import torch

    # Add the bundled prithvi_mae module to sys.path
    prithvi_dir = os.path.join(os.path.dirname(__file__), "prithvi_mae")
    if prithvi_dir not in sys.path:
        sys.path.insert(0, prithvi_dir)

    from prithvi_mae import PrithviMAE

    # Load config
    config_path = os.path.join(prithvi_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)["pretrained_cfg"]

    # Create model with specified num_frames
    model = PrithviMAE(
        img_size=config["img_size"],
        num_frames=num_frames,
        in_chans=config["in_chans"],
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        decoder_embed_dim=config["decoder_embed_dim"],
        decoder_depth=config["decoder_depth"],
        decoder_num_heads=config["decoder_num_heads"],
        mlp_ratio=config["mlp_ratio"],
        coords_encoding=[],
        patch_size=config["patch_size"],
    )

    if pretrained:
        # Try local symlink first, then download from HuggingFace
        local_weights = os.path.join(prithvi_dir, "Prithvi_EO_V2_300M_TL.pt")
        if os.path.exists(local_weights):
            weights_path = local_weights
        else:
            from huggingface_hub import hf_hub_download
            weights_path = hf_hub_download(
                "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL",
                "Prithvi_EO_V2_300M_TL.pt",
            )

        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        # pos_embed depends on num_frames and must be recomputed
        for k in list(state_dict.keys()):
            if "pos_embed" in k:
                del state_dict[k]
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model


def get_prithvi_config() -> dict:
    """Load and return the Prithvi model config.

    Returns:
        Dict with model configuration including bands, mean, std, etc.
    """
    config_path = os.path.join(
        os.path.dirname(__file__), "prithvi_mae", "config.json"
    )
    with open(config_path) as f:
        return json.load(f)["pretrained_cfg"]


def extract_encoder_features(model, tensor):
    """Extract encoder feature embeddings from a PrithviMAE model.

    Runs a forward pass with mask_ratio=0 (no masking) and extracts
    the encoder output via a forward hook. Returns a spatial feature map.

    Args:
        model: PrithviMAE model.
        tensor: Input tensor of shape (B, C, T, H, W).

    Returns:
        numpy array of shape (B, embed_dim, grid_h, grid_w).
    """
    import torch

    encoder_output = {}

    def hook_fn(module, input, output):
        encoder_output["output"] = output

    # Hook the encoder's final layer norm
    handle = model.encoder.norm.register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            model(tensor, mask_ratio=0.0)

        latent = encoder_output["output"]
        # Remove CLS token: (B, N+1, D) → (B, N, D)
        spatial = latent[:, 1:, :]
        n_tokens = spatial.shape[1]
        embed_dim = spatial.shape[2]

        # Reshape to spatial grid
        side = int(math.sqrt(n_tokens))
        if side * side == n_tokens:
            feature_map = spatial.reshape(-1, side, side, embed_dim)
            feature_map = feature_map.permute(0, 3, 1, 2)  # (B, D, H, W)
        else:
            # Non-square grid — keep as token sequence
            feature_map = spatial.permute(0, 2, 1).unsqueeze(-1)  # (B, D, N, 1)

        return feature_map.cpu().numpy()

    finally:
        handle.remove()


# ── Task head registry ───────────────────────────────────────────────────────

# Known fine-tuned Prithvi segmentation models on HuggingFace.
# Each entry defines how to build the complete segmentation model.
TASK_HEAD_REGISTRY = {
    "sen1floods11": {
        "repo_id": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
        "filename": "Prithvi-EO-V2-300M-TL-Sen1Floods11.pt",
        "num_classes": 2,
        "feature_indices": [5, 11, 17, 23],
        "decoder_channels": 256,
        "dropout": 0.1,
        "class_names": {0: "no_water", 1: "water/flood"},
        "description": "Flood segmentation (Sen1Floods11 dataset)",
    },
    "burn_scars": {
        "repo_id": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars",
        "filename": "Prithvi_EO_V2_300M_BurnScars.pt",
        "num_classes": 2,
        "feature_indices": [5, 11, 17, 23],
        "decoder_type": "unet",
        "dropout": 0.1,
        "class_names": {0: "no_burn", 1: "burned"},
        "description": "Burn scar segmentation (HLS Burn Scars dataset)",
    },
    "nmd_lulc": {
        "local_path": "checkpoints/lulc/best_model.pt",
        "num_classes": 20,  # 19 NMD L2 classes + background at index 0
        "feature_indices": [5, 11, 17, 23],
        "decoder_channels": 256,
        "decoder_type": "upernet",
        "dropout": 0.1,
        "class_names": {
            0: "background", 1: "forest_pine", 2: "forest_spruce",
            3: "forest_deciduous", 4: "forest_mixed",
            5: "forest_temp_non_forest", 6: "forest_wetland_pine",
            7: "forest_wetland_spruce", 8: "forest_wetland_deciduous",
            9: "forest_wetland_mixed", 10: "forest_wetland_temp",
            11: "open_wetland", 12: "cropland",
            13: "open_land_bare", 14: "open_land_vegetated",
            15: "developed_buildings", 16: "developed_infrastructure",
            17: "developed_roads", 18: "water_lakes", 19: "water_sea",
        },
        "description": "LULC classification (NMD Level 2, Sweden)",
    },
}


def list_task_heads() -> dict:
    """Return the task head registry for inspection.

    Returns:
        Dict mapping task head names to their config dicts.
    """
    return TASK_HEAD_REGISTRY.copy()


def load_segmentation_model(
    task_head: str,
    device: str = "cpu",
):
    """Load a complete Prithvi segmentation model.

    Selects the correct decoder architecture based on ``decoder_type``
    in the task head config:
        - ``"upernet"`` (default): UPerNet decoder (Sen1Floods11)
        - ``"unet"``: UNet decoder (BurnScars)

    Downloads the fine-tuned checkpoint from HuggingFace and maps its
    weights into the model.

    Args:
        task_head: Name from TASK_HEAD_REGISTRY (e.g. "sen1floods11").
        device: Target device ("cpu", "cuda", "mps").

    Returns:
        Tuple of (segmentation_model, task_config_dict).

    Raises:
        ValueError: If task_head is not in the registry.
        ImportError: If torch is not installed.
    """
    import torch
    from .upernet import PrithviSegmentationModel, PrithviUNetSegmentationModel

    if task_head not in TASK_HEAD_REGISTRY:
        available = ", ".join(sorted(TASK_HEAD_REGISTRY.keys()))
        raise ValueError(
            f"Unknown task head '{task_head}'. "
            f"Available: {available}"
        )

    config = TASK_HEAD_REGISTRY[task_head]
    decoder_type = config.get("decoder_type", "upernet")

    # 1. Load backbone (pretrained from HF)
    backbone = _load_prithvi_from_hf(pretrained=True)

    # 2. Build segmentation model based on decoder type
    if decoder_type == "unet":
        seg_model = PrithviUNetSegmentationModel(
            encoder=backbone,
            feature_indices=config["feature_indices"],
            num_classes=config["num_classes"],
            dropout=config["dropout"],
        )
    else:
        seg_model = PrithviSegmentationModel(
            encoder=backbone,
            feature_indices=config["feature_indices"],
            decoder_channels=config.get("decoder_channels", 256),
            num_classes=config["num_classes"],
            dropout=config["dropout"],
        )

    # 3. Load fine-tuned checkpoint (local_path or HuggingFace)
    if "local_path" in config:
        checkpoint_path = config["local_path"]
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Local checkpoint not found: {checkpoint_path}. "
                f"Train the model first with scripts/train_lulc.py"
            )
        print(f"    Loading local checkpoint: {checkpoint_path}")
    else:
        from huggingface_hub import hf_hub_download
        checkpoint_path = hf_hub_download(
            config["repo_id"],
            config["filename"],
        )
        print(f"    Loading checkpoint: {config['filename']}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract state dict from Lightning checkpoint
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint.state_dict() if hasattr(checkpoint, "state_dict") else None

    if state_dict is not None:
        # Map checkpoint keys to our model
        mapped_sd = _map_checkpoint_keys(state_dict, seg_model)
        missing, unexpected = seg_model.load_state_dict(mapped_sd, strict=False)

        # Filter out expected missing keys (pos_embed recomputed by model)
        real_missing = [k for k in missing if "pos_embed" not in k]
        if real_missing:
            print(f"    Warning: {len(real_missing)} missing keys (may affect accuracy)")
            for k in real_missing[:10]:
                print(f"      {k}")
        if unexpected:
            print(f"    Info: {len(unexpected)} unexpected keys (ignored)")

    seg_model = seg_model.to(device)
    seg_model.eval()
    return seg_model, config


def _map_checkpoint_keys(state_dict: dict, target_model) -> dict:
    """Map TerraTorch/Lightning checkpoint keys to our model keys.

    The checkpoint uses ``model.`` prefix (Lightning wrapping):
        model.encoder.* → our encoder.*
        model.decoder.* → our decoder.*
        model.neck.*    → our neck.*     (UNet variant)
        model.head.*    → our head.*

    Our segmentation models match this layout exactly, so the mapping
    is just stripping the ``model.`` prefix.

    Returns:
        Mapped state dict ready for load_state_dict().
    """
    mapped = {}
    target_keys = set(target_model.state_dict().keys())

    for key, value in state_dict.items():
        # Strip Lightning "model." prefix
        if key.startswith("model."):
            new_key = key[len("model."):]
        else:
            new_key = key

        # Skip non-model keys (optimizer, scheduler, etc.)
        if not any(new_key.startswith(p) for p in
                   ("encoder.", "decoder.", "neck.", "head.")):
            continue

        if new_key in target_keys:
            mapped[new_key] = value

    return mapped


def bands_to_tensor(
    bands_dict: dict[str, np.ndarray],
    band_order: list[str],
    device: str = "cpu",
):
    """Convert a dict of band arrays to a batched PyTorch tensor.

    Stacks the specified bands in the given order and creates a tensor
    with shape ``(1, C, H, W)`` suitable for model inference.

    Args:
        bands_dict: Dict mapping band name (e.g. "B02") to 2D numpy array.
        band_order: List of band names specifying stacking order.
            E.g. ["B02", "B03", "B04", "B8A", "B11", "B12"].
        device: Target device string ("cpu", "cuda", "mps").

    Returns:
        torch.Tensor of shape (1, C, H, W) with float32 dtype.

    Raises:
        KeyError: If a required band is missing from bands_dict.
    """
    import torch

    arrays = []
    for band_name in band_order:
        if band_name not in bands_dict:
            raise KeyError(
                f"Required band '{band_name}' not found in bands_dict. "
                f"Available bands: {sorted(bands_dict.keys())}"
            )
        arrays.append(bands_dict[band_name].astype(np.float32))

    # Stack: (C, H, W)
    stacked = np.stack(arrays, axis=0)
    # Add batch dimension: (1, C, H, W)
    tensor = torch.from_numpy(stacked).unsqueeze(0)
    return tensor.to(device)
