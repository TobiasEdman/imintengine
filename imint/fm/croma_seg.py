"""
imint/fm/croma_seg.py — CROMA segmentation wrapper.

CROMA forward returns a dict:
    SAR_encodings:      (B, N, D) — S1 patches
    optical_encodings:  (B, N, D) — S2 patches
    joint_encodings:    (B, N, D) — fused (modality='both' only)
    plus *_GAP globally-pooled vectors we ignore.

For dense segmentation we use ``joint_encodings`` when both modalities
are available, falling back to ``optical_encodings`` otherwise. N =
(img_size/8)² tokens. Linear-probe head follows the same pattern as
TerraMindSegmentationModel / ClaySegmentationModel.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CromaSegmentationModel(nn.Module):
    """Linear-probe segmentation head on top of CROMA encodings.

    Args:
        encoder: PretrainedCROMA instance from ``load_croma()``.
        num_classes: Output classes.
        img_size: Input resolution. Must be multiple of patch_size=8.
        patch_size: 8 (CROMA-base and CROMA-large both).
        embed_dim: 768 (base) or 1024 (large).
        modality: Which encoding to use: 'joint' (S1+S2 fused),
            'optical' (S2 only), or 'sar' (S1 only).
        n_aux_channels: Optional aux raster channels fused at output res.
        dropout: Dropout before classifier.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int = 23,
        img_size: int = 120,
        patch_size: int = 8,
        embed_dim: int = 768,
        modality: str = "joint",
        n_aux_channels: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.modality = modality
        self.n_aux_channels = n_aux_channels

        grid = img_size // patch_size
        if grid * patch_size != img_size:
            raise ValueError(
                f"img_size={img_size} must be divisible by patch_size={patch_size}"
            )
        self.grid_size = grid
        self.expected_n_patches = grid * grid

        if modality not in ("joint", "optical", "sar"):
            raise ValueError(
                f"modality={modality!r}; must be 'joint', 'optical', or 'sar'."
            )

        # Progressive upsample
        c1 = embed_dim
        c2 = embed_dim // 2
        c3 = embed_dim // 4
        c4 = embed_dim // 8

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

    def _pick_encoding(self, enc_dict: dict) -> torch.Tensor:
        """Select the token sequence per ``self.modality``, with a
        graceful fallback when joint encodings are unavailable."""
        key_map = {
            "joint": "joint_encodings",
            "optical": "optical_encodings",
            "sar": "SAR_encodings",
        }
        primary = key_map[self.modality]
        if primary in enc_dict:
            return enc_dict[primary]
        # Fallbacks
        if self.modality == "joint":
            for fb in ("optical_encodings", "SAR_encodings"):
                if fb in enc_dict:
                    return enc_dict[fb]
        raise KeyError(
            f"CROMA encoder did not return any usable encoding. "
            f"Wanted {primary!r}; got keys {list(enc_dict)}."
        )

    def _tokens_to_spatial(self, tokens: torch.Tensor) -> torch.Tensor:
        B, N, D = tokens.shape
        if N != self.expected_n_patches:
            raise ValueError(
                f"CROMA returned {N} tokens; expected "
                f"{self.expected_n_patches} (grid {self.grid_size}²). "
                f"Check img_size/patch_size."
            )
        if D != self.embed_dim:
            raise ValueError(
                f"CROMA returned embed_dim={D}; expected {self.embed_dim}."
            )
        return tokens.transpose(1, 2).reshape(
            B, D, self.grid_size, self.grid_size,
        )

    def forward(
        self,
        sar: torch.Tensor | None = None,
        optical: torch.Tensor | None = None,
        aux: torch.Tensor | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Per-pixel class logits from CROMA encodings.

        Args:
            sar: (B, 2, H, W) S1 VV/VH. Required for modality in
                {'sar', 'joint'}.
            optical: (B, 12, H, W) S2 12-band stack from
                build_s2_croma_tensor. Required for modality in
                {'optical', 'joint'}.
            aux: Optional (B, n_aux, Ho, Wo) at the desired output res.
            output_size: Optional (H, W) for the output logits. Defaults
                to the input sar/optical tensor's (H, W).

        Returns:
            (B, num_classes, Ho, Wo) logits.
        """
        if self.modality in ("joint", "sar") and sar is None:
            raise ValueError(f"CROMA modality={self.modality!r} requires 'sar'.")
        if self.modality in ("joint", "optical") and optical is None:
            raise ValueError(f"CROMA modality={self.modality!r} requires 'optical'.")

        ref = optical if optical is not None else sar
        if output_size is None:
            output_size = tuple(ref.shape[-2:])

        enc = self.encoder(SAR_images=sar, optical_images=optical)
        tokens = self._pick_encoding(enc)
        feat = self._tokens_to_spatial(tokens)  # (B, D, grid, grid)

        feat = self.up1(feat)
        feat = self.up2(feat)
        feat = self.smooth(feat)

        feat = F.interpolate(
            feat, size=output_size, mode="bilinear", align_corners=True,
        )

        if self.aux_proj is not None and aux is not None:
            if aux.shape[-2:] != output_size:
                aux = F.interpolate(
                    aux, size=output_size, mode="bilinear", align_corners=True,
                )
            aux_feat = self.aux_proj(aux)
            feat = torch.cat([feat, aux_feat], dim=1)

        feat = self.dropout(feat)
        return self.classifier(feat)
