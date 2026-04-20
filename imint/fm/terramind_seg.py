"""
imint/fm/terramind_seg.py — TerraMind segmentation wrapper.

TerraMind's encoder returns a token sequence ``(B, N, D)`` where N is
the number of patches (for base: 14×14 = 196 patches at 224 px) and
D = 768 (embed_dim). UPerNet expects per-block spatial feature maps
from multiple transformer depths, which TerraMind doesn't expose as a
first-class API. For the first ensemble member we use a **linear
probe** style segmentation head: reshape the single output token
sequence to a spatial grid, apply a lightweight conv stack with a
learnable up-projection, then bilinear upsample to the input
resolution.

This is the pattern most "frozen FM + linear head" papers evaluate
with. If mIoU is clearly below Prithvi, we can swap in a full
multi-level UPerNet by hooking into TerraMind's transformer blocks.

Usage:
    from imint.fm.terramind_seg import TerraMindSegmentationModel
    model = TerraMindSegmentationModel(
        encoder=terramind_backbone,  # from registry.build_backbone
        num_classes=23,
        img_size=224,
        embed_dim=768,
        patch_size=16,
        n_aux_channels=0,
    )
    # Forward takes a dict of modality tensors + optional aux channels
    logits = model({
        "S2L2A": (B, 6, H, W),
        "S1GRD": (B, 2, H, W),
    }, aux=None)
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class TerraMindSegmentationModel(nn.Module):
    """Linear-probe segmentation head on top of a TerraMind encoder.

    Architecture:
        1. encoder({modality: (B, C, H, W)}) → tokens (B, N, D)
        2. reshape tokens → spatial (B, D, grid_h, grid_w)
        3. 2× ConvTranspose upsample (D → D//2 → D//4)
        4. 3×3 Conv → D//8
        5. Optional gated fusion with (B, n_aux, H, W) auxiliary channels
        6. 1×1 Conv → num_classes
        7. Bilinear resize to input resolution

    Args:
        encoder: TerraMind backbone loaded via registry.build_backbone.
        num_classes: Output segmentation classes (default 23 = unified schema).
        img_size: Input spatial resolution (must match what encoder was
            built with). Default 224.
        embed_dim: Encoder embedding dimension (default 768 for base).
        patch_size: ViT patch size (default 16). Grid size = img_size /
            patch_size.
        n_aux_channels: Number of auxiliary raster channels (DEM, VPP,
            etc). Concatenated at input resolution before the final
            1×1 conv. 0 = no aux.
        dropout: Dropout applied before the classifier.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int = 23,
        img_size: int = 224,
        embed_dim: int = 768,
        patch_size: int = 16,
        n_aux_channels: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.n_aux_channels = n_aux_channels

        grid = img_size // patch_size
        if grid * patch_size != img_size:
            raise ValueError(
                f"img_size={img_size} must be divisible by patch_size={patch_size}"
            )
        self.grid_size = grid

        # Progressive upsampling: (B, 768, 14, 14) → (B, 384, 28, 28)
        # → (B, 192, 56, 56) → (B, 96, 112, 112) → via bilinear to input.
        c1 = embed_dim             # 768
        c2 = embed_dim // 2        # 384
        c3 = embed_dim // 4        # 192
        c4 = embed_dim // 8        # 96

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(c1, c2, kernel_size=2, stride=2),
            nn.BatchNorm2d(c2), nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(c2, c3, kernel_size=2, stride=2),
            nn.BatchNorm2d(c3), nn.GELU(),
        )
        self.smooth = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c4), nn.GELU(),
        )

        # Optional aux branch — concat-fuse at the final resolution.
        if n_aux_channels > 0:
            self.aux_proj = nn.Sequential(
                nn.Conv2d(n_aux_channels, c4, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c4), nn.GELU(),
            )
            classifier_in = c4 * 2
        else:
            self.aux_proj = None
            classifier_in = c4

        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(classifier_in, num_classes, kernel_size=1)

    def _tokens_to_spatial(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, N, D) → (B, D, grid, grid). Validates N matches grid²."""
        B, N, D = tokens.shape
        expected = self.grid_size * self.grid_size
        if N != expected:
            # TerraMind may return (B, N+1, D) with a CLS token prepended
            if N == expected + 1:
                tokens = tokens[:, 1:, :]  # drop CLS
                N = tokens.shape[1]
            else:
                raise ValueError(
                    f"TerraMind returned {N} tokens; expected {expected} "
                    f"(grid {self.grid_size}²) for img_size={self.img_size} "
                    f"patch_size={self.patch_size}. Did you pass inputs at "
                    f"a non-native resolution?"
                )
        if D != self.embed_dim:
            raise ValueError(
                f"TerraMind returned embed_dim={D}; expected {self.embed_dim}. "
                f"Check the registry spec matches the actual checkpoint."
            )
        return tokens.transpose(1, 2).reshape(B, D, self.grid_size, self.grid_size)

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        aux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            inputs: Dict of modality → tensor, e.g.
                ``{"S2L2A": (B, 6, H, W), "S1GRD": (B, 2, H, W)}``.
                Shapes must match what TerraMind was built with (typ. 224).
            aux: Optional (B, n_aux, H, W) auxiliary raster channels,
                concatenated at the final resolution before classifier.

        Returns:
            (B, num_classes, H, W) logits at input resolution.
        """
        if not isinstance(inputs, dict):
            raise TypeError(
                f"TerraMind forward expects a dict of modality tensors, "
                f"got {type(inputs).__name__}."
            )

        # Pull input resolution from any tensor in the dict
        any_t = next(iter(inputs.values()))
        input_h, input_w = any_t.shape[-2:]

        tokens = self.encoder(inputs)  # (B, N, D) or (B, N+1, D)
        feat = self._tokens_to_spatial(tokens)   # (B, D, grid, grid)

        feat = self.up1(feat)                     # (B, D/2, 2*grid, 2*grid)
        feat = self.up2(feat)                     # (B, D/4, 4*grid, 4*grid)
        feat = self.smooth(feat)                  # (B, D/8, ...)

        # Upsample to input resolution before (optional) aux fusion
        feat = F.interpolate(
            feat, size=(input_h, input_w), mode="bilinear", align_corners=True,
        )

        if self.aux_proj is not None and aux is not None:
            aux_feat = self.aux_proj(aux)
            feat = torch.cat([feat, aux_feat], dim=1)

        feat = self.dropout(feat)
        logits = self.classifier(feat)
        return logits
