"""Prithvi-EO-2.0 pixel classifier head.

Wraps the Prithvi ViT encoder as a backbone for single-pixel classification
using a small (context_px × context_px) temporal context window.

Architecture:
    Spectral path:
        Input  : (B, T*6, ctx, ctx) float32, e.g. (B, 30, 32, 32) for T=5
        Reshape: (B, 6, T, ctx, ctx) for Prithvi
        Encoder: Prithvi-EO-2.0 PrithviViT → 24 transformer blocks
        Readout: [CLS] token from block[-1]: (B, embed_dim)

    AUX path (when n_aux > 0):
        Input  : (B, N_AUX) float32 center-pixel auxiliary scalars
                 (height, volume, DEM, VPP, harvest_probability — z-scored)
        Project: Linear(N_AUX → aux_proj_dim) → GELU          → (B, 64)

    Fusion + Head:
        cat([cls_token, aux_feat])                              → (B, 1088)
        Linear(1088 → 512) → GELU → Dropout(0.2) → Linear(512 → 23)
        Output : (B, 23) logits

Temporal embedding extension (T=4 → T=5):
    The pretrained backbone uses learned positional embeddings that encode
    both spatial patch position and temporal frame index.  When ``num_frames``
    is set to 5 instead of the pretrained 4, ``_load_prithvi_from_hf``
    already removes the stale ``pos_embed`` keys and re-initialises them
    via ``initialize_weights()``.  The encoder then learns T=5 positional
    embeddings from scratch during Stage 1 fine-tuning (head-only, backbone
    frozen) and adapts the full backbone in Stage 2.

Two-stage training (recommended):
    Stage 1 (5 epochs): backbone frozen, train head + aux_proj + pos_embed
    Stage 2 (30 epochs): all layers; backbone LR = 0.1 × head LR

Usage::

    from imint.fm.pixel_head import PrithviPixelClassifier

    model = PrithviPixelClassifier(num_classes=23, context_px=32, num_frames=5)
    model.freeze_backbone()          # Stage 1
    # ... train head + aux_proj for 5 epochs ...
    model.unfreeze_backbone()        # Stage 2
    # ... fine-tune full model for 30 epochs ...
"""
from __future__ import annotations

import torch
import torch.nn as nn

from imint.training.unified_schema import NUM_UNIFIED_CLASSES

# Default AUX dimensionality — matches len(AUX_CHANNEL_NAMES) in unified_dataset
N_AUX_DEFAULT = 11   # height, volume, basal_area, diameter, dem,
                     # vpp_sosd, vpp_eosd, vpp_length, vpp_maxv, vpp_minv,
                     # harvest_probability
AUX_PROJ_DIM = 64    # hidden dim of the lightweight AUX projector


class PrithviPixelClassifier(nn.Module):
    """Center-pixel classifier using Prithvi-EO-2.0 as frozen/fine-tuned backbone.

    The spectral CLS token is fused with a projected center-pixel AUX vector
    before the final MLP head.  Pass ``n_aux=0`` to disable AUX entirely and
    recover the spectral-only architecture.

    Args:
        num_classes: Number of output classes (default: 23).
        context_px: Side length of the square context crop (default: 32).
            Must be divisible by the backbone patch size (16).
        num_frames: Number of temporal frames in the input (default: 5).
            Set to 4 if no 2016 background frame is available.
        embed_dim: Prithvi encoder embedding dimension (default: 1024 for 300M).
        mlp_hidden: Hidden dimension of the MLP classification head (default: 512).
        dropout: Dropout probability in the classification head (default: 0.2).
        pretrained: Whether to load pretrained Prithvi weights (default: True).
        n_aux: Number of auxiliary scalar channels fed to the AUX projector
            (default: 11).  Set to 0 to disable AUX injection.
        aux_proj_dim: Output dimension of the AUX projector (default: 64).
    """

    def __init__(
        self,
        num_classes: int = NUM_UNIFIED_CLASSES,
        context_px: int = 32,
        num_frames: int = 5,
        embed_dim: int = 1024,
        mlp_hidden: int = 512,
        dropout: float = 0.2,
        pretrained: bool = True,
        n_aux: int = N_AUX_DEFAULT,
        aux_proj_dim: int = AUX_PROJ_DIM,
    ) -> None:
        super().__init__()

        if context_px % 16 != 0:
            raise ValueError(
                f"context_px ({context_px}) must be divisible by the Prithvi "
                f"patch size (16).  Use 32, 48, 64, …"
            )

        self.num_frames = num_frames
        self.context_px = context_px
        self.embed_dim = embed_dim
        self.n_aux = n_aux
        self.aux_proj_dim = aux_proj_dim if n_aux > 0 else 0

        # ── Load Prithvi encoder ──────────────────────────────────────
        # _load_prithvi_from_hf drops pos_embed keys and re-initialises
        # for the requested num_frames.  All other pretrained weights are
        # preserved (patch_embed, transformer blocks, norm).
        from imint.fm.terratorch_loader import _load_prithvi_from_hf
        mae = _load_prithvi_from_hf(pretrained=pretrained, num_frames=num_frames)
        self.backbone = mae.encoder   # PrithviViT

        # ── AUX projector (lightweight, trained in both stages) ───────
        # Projects the N_AUX center-pixel scalars to a 64-dim feature
        # vector that is concatenated with the Prithvi CLS token before
        # the classification head.
        if n_aux > 0:
            self.aux_proj = nn.Sequential(
                nn.Linear(n_aux, aux_proj_dim),
                nn.GELU(),
            )
            nn.init.trunc_normal_(self.aux_proj[0].weight, std=0.02)
            nn.init.zeros_(self.aux_proj[0].bias)
        else:
            self.aux_proj = None

        # ── MLP classification head ───────────────────────────────────
        head_in = embed_dim + self.aux_proj_dim  # 1024 + 64 = 1088 (or 1024)
        self.head = nn.Sequential(
            nn.Linear(head_in, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_classes),
        )
        nn.init.trunc_normal_(self.head[0].weight, std=0.02)
        nn.init.zeros_(self.head[0].bias)
        nn.init.trunc_normal_(self.head[3].weight, std=0.02)
        nn.init.zeros_(self.head[3].bias)

    # ── Parameter group helpers (used by training script) ────────────

    def freeze_backbone(self) -> None:
        """Freeze backbone (all parameters except pos_embed for Stage 1).

        The AUX projector and classification head remain trainable.
        """
        for name, p in self.backbone.named_parameters():
            if "pos_embed" in name:
                # Allow pos_embed to adapt even in Stage 1 (T=4→T=5)
                p.requires_grad = True
            else:
                p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze entire backbone for Stage 2 fine-tuning."""
        for p in self.backbone.parameters():
            p.requires_grad = True

    def backbone_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def head_parameters(self) -> list[nn.Parameter]:
        """Parameters of the head + AUX projector (always trained)."""
        params = list(self.head.parameters())
        if self.aux_proj is not None:
            params += list(self.aux_proj.parameters())
        return params

    # ── Forward pass ─────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        aux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(B, T*6, H, W)`` float32 spectral context patches.
               Channel layout: ``[frame0_band0..5, frame1_band0..5, …]``
               Band order per frame: B02, B03, B04, B8A, B11, B12.
               Frame order: 2016-summer, autumn yr-1, spring, peak, harvest.
            aux: ``(B, n_aux)`` float32 center-pixel auxiliary scalars,
               z-score normalized (same pipeline as UnifiedDataset).
               If ``None`` and ``n_aux > 0``, a zero tensor is used (i.e.
               all auxiliary channels treated as missing).

        Returns:
            ``(B, num_classes)`` raw logits.
        """
        B = x.shape[0]
        T = self.num_frames

        # Reshape: (B, T*6, H, W) → (B, 6, T, H, W)
        # x is laid out as [frame0_bands(6), frame1_bands(6), …]
        x = x.view(B, T, 6, self.context_px, self.context_px)  # (B, T, 6, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()              # (B, 6, T, H, W)

        # Prithvi encoder: returns list of 24 per-block feature tensors
        # Each: (B, N_tokens+1, embed_dim)  — CLS token at position 0
        block_features = self.backbone.forward_features(x)
        cls_token = block_features[-1][:, 0, :]  # (B, embed_dim)

        # ── AUX fusion ────────────────────────────────────────────────
        if self.aux_proj is not None:
            if aux is None:
                aux = torch.zeros(B, self.n_aux, device=x.device, dtype=cls_token.dtype)
            aux_feat = self.aux_proj(aux)                           # (B, aux_proj_dim)
            cls_token = torch.cat([cls_token, aux_feat], dim=-1)    # (B, 1088)

        return self.head(cls_token)  # (B, num_classes)


# ── Factory helper ────────────────────────────────────────────────────────

def build_pixel_classifier(
    num_frames: int = 5,
    num_classes: int = NUM_UNIFIED_CLASSES,
    context_px: int = 32,
    n_aux: int = N_AUX_DEFAULT,
    pretrained: bool = True,
    checkpoint_path: str | None = None,
    device: str | torch.device = "cpu",
) -> PrithviPixelClassifier:
    """Build (and optionally load checkpoint for) a PrithviPixelClassifier.

    Args:
        num_frames: 5 if using 2016 background frame, else 4.
        num_classes: 23 for the unified Swedish LULC schema.
        context_px: Context window side length in pixels.
        n_aux: Number of AUX channels (11 default; 0 to disable).
        pretrained: Load Prithvi pretrained weights.
        checkpoint_path: Path to a saved classifier checkpoint (.pt).
        device: Target device.

    Returns:
        PrithviPixelClassifier ready for inference or fine-tuning.
    """
    model = PrithviPixelClassifier(
        num_classes=num_classes,
        context_px=context_px,
        num_frames=num_frames,
        n_aux=n_aux,
        pretrained=pretrained,
    )

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        # Strip "model." prefix if present (Lightning wrapping)
        state = {k.removeprefix("model."): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  [pixel_head] missing keys: {missing[:5]}")
        if unexpected:
            print(f"  [pixel_head] unexpected keys: {unexpected[:5]}")

    model = model.to(device)
    return model
