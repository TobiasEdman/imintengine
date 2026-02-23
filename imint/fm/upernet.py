"""
imint/fm/upernet.py — Segmentation decoders matching TerraTorch checkpoints

Implements two decoder architectures used by TerraTorch fine-tuned
Prithvi models so that checkpoints can be loaded directly:

1. UPerNet (Sen1Floods11): PSP + FPN decoder
   model.encoder  → PrithviViT backbone
   model.decoder  → UperNet (psp_modules, bottleneck, lateral/fpn_convs, fpn1/fpn2)
   model.head     → Segmentation head

2. UNet (BurnScars): Progressive upsampling decoder
   model.encoder  → PrithviViT backbone
   model.neck     → Scale modules (fpn1/fpn2)
   model.decoder  → UNet blocks (conv1+conv2 per level)
   model.head     → Segmentation head

References:
    - Xiao et al., "Unified Perceptual Parsing for Scene Understanding" (2018)
    - TerraTorch: github.com/IBM/terratorch
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu(nn.Module):
    """Conv2d + BatchNorm2d + ReLU block matching TerraTorch naming.

    State dict keys: conv.weight, norm.weight, norm.bias, norm.running_mean, etc.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1,
                 bias: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=bias)
        self.norm = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.norm(self.conv(x)))


class TerraTorchUperNetDecoder(nn.Module):
    """UperNet decoder matching TerraTorch checkpoint key layout.

    This decoder expects multi-scale feature maps that have already been
    rescaled by scale modules (fpn1/fpn2). The channel dimensions after
    scaling are: [256, 512, 1024, 1024] for 4 levels.

    Architecture:
        psp_modules — Pyramid Pooling on deepest level (level 3)
        bottleneck  — PSP bottleneck (in_channels + 4*256 → 256)
        lateral_convs — 1×1 reduce channels per level [0,1,2]
        fpn_convs — 3×3 smoothing per level [0,1,2]
        fpn_bottleneck — fuse all 4 levels (4*256 → 256)

    Args:
        in_channels: Channel dims for each scale level after fpn scaling.
            Default [256, 512, 1024, 1024] for Prithvi 300M with fpn1/fpn2.
        decoder_channels: Internal channel dim (default 256).
        pool_sizes: PSP pool sizes (default (1, 2, 3, 6)).
    """

    def __init__(
        self,
        in_channels: list[int] = None,
        decoder_channels: int = 256,
        pool_sizes: tuple[int, ...] = (1, 2, 3, 6),
    ):
        super().__init__()
        if in_channels is None:
            in_channels = [256, 512, 1024, 1024]

        self.num_levels = len(in_channels)

        # PSP modules on deepest level
        self.psp_modules = nn.ModuleList()
        for pool_size in pool_sizes:
            self.psp_modules.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                ConvBnRelu(in_channels[-1], decoder_channels, kernel=1, padding=0),
            ))

        # PSP bottleneck: input_channels[-1] + 4 * 256 → 256
        psp_concat_channels = in_channels[-1] + decoder_channels * len(pool_sizes)
        self.bottleneck = ConvBnRelu(psp_concat_channels, decoder_channels)

        # Lateral convolutions (1×1) for levels [0, 1, 2]
        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_levels - 1):
            self.lateral_convs.append(
                ConvBnRelu(in_channels[i], decoder_channels, kernel=1, padding=0)
            )

        # FPN convolutions (3×3) for levels [0, 1, 2]
        self.fpn_convs = nn.ModuleList()
        for i in range(self.num_levels - 1):
            self.fpn_convs.append(
                ConvBnRelu(decoder_channels, decoder_channels)
            )

        # FPN bottleneck: concat all levels (4 × 256 → 256)
        self.fpn_bottleneck = ConvBnRelu(
            decoder_channels * self.num_levels, decoder_channels,
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Args:
            features: Multi-scale feature maps [level_0, ..., level_N].
                Level 0 = highest resolution, level N = deepest.

        Returns:
            (B, 256, H_0, W_0) fused feature map.
        """
        # PSP on deepest feature
        deepest = features[-1]
        h, w = deepest.shape[2:]
        psp_outs = [deepest]
        for psp_module in self.psp_modules:
            pooled = psp_module(deepest)
            pooled = F.interpolate(pooled, size=(h, w), mode="bilinear", align_corners=True)
            psp_outs.append(pooled)
        psp_out = self.bottleneck(torch.cat(psp_outs, dim=1))

        # FPN: top-down path
        fpn_outs = [psp_out]
        for i in range(self.num_levels - 2, -1, -1):
            lateral = self.lateral_convs[i](features[i])
            target_h, target_w = lateral.shape[2:]
            upsampled = F.interpolate(
                fpn_outs[0], size=(target_h, target_w),
                mode="bilinear", align_corners=True,
            )
            fpn_out = self.fpn_convs[i](lateral + upsampled)
            fpn_outs.insert(0, fpn_out)

        # Resize all to highest resolution and concatenate
        target_h, target_w = fpn_outs[0].shape[2:]
        resized = []
        for out in fpn_outs:
            if out.shape[2:] != (target_h, target_w):
                out = F.interpolate(
                    out, size=(target_h, target_w),
                    mode="bilinear", align_corners=True,
                )
            resized.append(out)

        return self.fpn_bottleneck(torch.cat(resized, dim=1))


class SegmentationHead(nn.Module):
    """Segmentation classification head matching TerraTorch layout.

    TerraTorch structure: head.head = Sequential(Identity, Dropout2d, Conv2d)
    Checkpoint keys: head.head.2.weight, head.head.2.bias
    """

    def __init__(self, in_channels: int = 256, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Identity(),       # index 0 — placeholder
            nn.Dropout2d(dropout),  # index 1
            nn.Conv2d(in_channels, num_classes, 1),  # index 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class PrithviSegmentationModel(nn.Module):
    """Complete Prithvi segmentation model matching TerraTorch checkpoint layout.

    State dict layout (matching checkpoint):
        encoder.*       → PrithviViT backbone
        decoder.*       → UperNet decoder
        head.*          → Segmentation head

    The model uses:
        1. PrithviViT.forward_features() → per-block token embeddings
        2. SelectIndices [5,11,17,23] → 4 selected feature levels
        3. prepare_features_for_image_model() → reshape tokens to spatial maps
        4. Scale modules (fpn1/fpn2) → multi-scale features [256, 512, 1024, 1024]
        5. UperNet decoder → fused 256-channel feature map
        6. Head → per-pixel class logits

    Args:
        encoder: PrithviViT backbone model (or PrithviMAE — uses .encoder).
        feature_indices: Transformer block indices to extract features from.
        decoder_channels: UperNet internal channels.
        num_classes: Number of segmentation classes.
        dropout: Head dropout rate.
    """

    def __init__(
        self,
        encoder,
        feature_indices: list[int] = (5, 11, 17, 23),
        decoder_channels: int = 256,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_indices = list(feature_indices)

        # Get the ViT encoder from PrithviMAE if needed
        if hasattr(encoder, "encoder"):
            self.encoder = encoder.encoder
        else:
            self.encoder = encoder

        self.embed_dim = self.encoder.embed_dim  # 1024 for 300M

        # Scale modules: create multi-scale from ViT's uniform-scale features
        # fpn1: 1024 → 512 (2× upsample) → 256 (2× upsample) → level 0
        # fpn2: 1024 → 512 (2× upsample) → level 1
        # level 2: 1024 (pass-through)
        # level 3: 1024 (pass-through, goes to PSP)
        self.decoder = nn.Module()
        # We need to register sub-modules manually to match checkpoint paths

        # fpn1: ConvTranspose2d(1024→512, 2×2, stride 2) + BN + ReLU
        #        + ConvTranspose2d(512→256, 2×2, stride 2)
        self.decoder.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, self.embed_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(self.embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(self.embed_dim // 2, self.embed_dim // 4, kernel_size=2, stride=2),
        )

        # fpn2: ConvTranspose2d(1024→512, 2×2, stride 2)
        self.decoder.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, self.embed_dim // 2, kernel_size=2, stride=2),
        )

        # UperNet decoder expects channels [256, 512, 1024, 1024]
        scale_channels = [
            self.embed_dim // 4,  # 256 (after fpn1)
            self.embed_dim // 2,  # 512 (after fpn2)
            self.embed_dim,       # 1024 (pass-through)
            self.embed_dim,       # 1024 (deepest, goes to PSP)
        ]

        # PSP modules
        pool_sizes = (1, 2, 3, 6)
        self.decoder.psp_modules = nn.ModuleList()
        for pool_size in pool_sizes:
            self.decoder.psp_modules.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                ConvBnRelu(scale_channels[-1], decoder_channels, kernel=1, padding=0),
            ))

        # PSP bottleneck
        psp_concat_ch = scale_channels[-1] + decoder_channels * len(pool_sizes)
        self.decoder.bottleneck = ConvBnRelu(psp_concat_ch, decoder_channels)

        # Lateral convolutions (levels 0, 1, 2)
        self.decoder.lateral_convs = nn.ModuleList()
        for i in range(len(scale_channels) - 1):
            self.decoder.lateral_convs.append(
                ConvBnRelu(scale_channels[i], decoder_channels, kernel=1, padding=0)
            )

        # FPN convolutions (levels 0, 1, 2)
        self.decoder.fpn_convs = nn.ModuleList()
        for i in range(len(scale_channels) - 1):
            self.decoder.fpn_convs.append(
                ConvBnRelu(decoder_channels, decoder_channels)
            )

        # FPN bottleneck (concat 4 × 256 → 256)
        self.decoder.fpn_bottleneck = ConvBnRelu(
            decoder_channels * len(scale_channels), decoder_channels,
        )

        # Segmentation head
        self.head = SegmentationHead(decoder_channels, num_classes, dropout)

    def _extract_multi_scale_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract and rescale features from selected transformer blocks.

        Returns list of [level_0 (highest res), ..., level_3 (lowest/deepest)].
        """
        # Get all block outputs: list of (B, N_tokens+1, embed_dim)
        all_features = self.encoder.forward_features(x)

        # Select features at specified indices
        selected = [all_features[i] for i in self.feature_indices]

        # Reshape tokens → spatial maps using backbone's method
        # Result: list of (B, embed_dim, grid_h, grid_w)
        spatial = self.encoder.prepare_features_for_image_model(selected)

        # Apply scale modules to create multi-scale
        # spatial[0] = level 0 features → fpn1 (1024→256, 4× upsample)
        # spatial[1] = level 1 features → fpn2 (1024→512, 2× upsample)
        # spatial[2] = level 2 features → pass-through (1024)
        # spatial[3] = level 3 features → pass-through (1024, goes to PSP)
        scaled = [
            self.decoder.fpn1(spatial[0]),   # (B, 256, 4*grid_h, 4*grid_w)
            self.decoder.fpn2(spatial[1]),   # (B, 512, 2*grid_h, 2*grid_w)
            spatial[2],                      # (B, 1024, grid_h, grid_w)
            spatial[3],                      # (B, 1024, grid_h, grid_w)
        ]

        return scaled

    def _decode(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Run UperNet decode on multi-scale features.

        Returns (B, 256, H_0, W_0) feature map.
        """
        # PSP on deepest
        deepest = features[-1]
        h, w = deepest.shape[2:]
        psp_outs = [deepest]
        for psp_module in self.decoder.psp_modules:
            pooled = psp_module(deepest)
            pooled = F.interpolate(pooled, size=(h, w), mode="bilinear", align_corners=True)
            psp_outs.append(pooled)
        psp_out = self.decoder.bottleneck(torch.cat(psp_outs, dim=1))

        # FPN top-down
        n = len(features)
        fpn_outs = [psp_out]
        for i in range(n - 2, -1, -1):
            lateral = self.decoder.lateral_convs[i](features[i])
            target_h, target_w = lateral.shape[2:]
            upsampled = F.interpolate(
                fpn_outs[0], size=(target_h, target_w),
                mode="bilinear", align_corners=True,
            )
            fpn_out = self.decoder.fpn_convs[i](lateral + upsampled)
            fpn_outs.insert(0, fpn_out)

        # Resize all to highest resolution
        target_h, target_w = fpn_outs[0].shape[2:]
        resized = []
        for out in fpn_outs:
            if out.shape[2:] != (target_h, target_w):
                out = F.interpolate(
                    out, size=(target_h, target_w),
                    mode="bilinear", align_corners=True,
                )
            resized.append(out)

        return self.decoder.fpn_bottleneck(torch.cat(resized, dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: encoder → scale → UperNet → head.

        Args:
            x: (B, C, H, W) or (B, C, T, H, W) input tensor.

        Returns:
            (B, num_classes, H, W) logits at input resolution.
        """
        if x.dim() == 4:
            input_h, input_w = x.shape[2:]
        else:
            input_h, input_w = x.shape[3:]

        features = self._extract_multi_scale_features(x)
        decoded = self._decode(features)
        logits = self.head(decoded)

        # Upsample to input resolution
        if logits.shape[2:] != (input_h, input_w):
            logits = F.interpolate(
                logits, size=(input_h, input_w),
                mode="bilinear", align_corners=True,
            )

        return logits


# ── UNet decoder (BurnScars checkpoint) ──────────────────────────────────────


class UNetDecoderBlock(nn.Module):
    """Single UNet decoder block matching smp's DecoderBlock layout.

    Checkpoint layout per block:
        blocks.{i}.conv1.0.weight  (Conv2d)
        blocks.{i}.conv1.1.*       (BatchNorm2d)
        blocks.{i}.conv2.0.weight  (Conv2d)
        blocks.{i}.conv2.1.*       (BatchNorm2d)

    When skip is provided, input is concat(upsampled, skip).
    When skip is None, input is just upsampled (no skip connection).
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.skip_ch = skip_ch
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor,
                skip: torch.Tensor | None = None) -> torch.Tensor:
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class TerraTorchUNetDecoder(nn.Module):
    """UNet decoder matching BurnScars TerraTorch/smp checkpoint layout.

    The checkpoint path is ``decoder.decoder.blocks.{i}`` — note the
    double ``decoder.`` prefix from TerraTorch wrapping.

    Architecture for 4 encoder levels [256, 512, 1024, 1024] (shallow→deep):
        blocks.0: up(1024) + skip[-2]=1024 → concat(1024+1024)=2048 → 512
        blocks.1: up(512) + skip[-3]=512  → concat(512+512)=1024 → 256
        blocks.2: up(256) + skip[-4]=256  → concat(256+256)=512 → 128
        blocks.3: up(128), no skip        → 128 → 64

    N encoder levels produce N-1 skip connections. The last block has
    no skip connection.
    """

    def __init__(self, encoder_channels: list[int] = None,
                 decoder_channels: list[int] = None):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [256, 512, 1024, 1024]
        if decoder_channels is None:
            decoder_channels = [512, 256, 128, 64]

        n_blocks = len(decoder_channels)
        # Skips: from second-deepest to shallowest
        skip_channels = list(reversed(encoder_channels[:-1]))
        # Pad with 0 for blocks without skip connections
        while len(skip_channels) < n_blocks:
            skip_channels.append(0)

        self.blocks = nn.ModuleList()
        in_ch = encoder_channels[-1]  # deepest = 1024
        for i in range(n_blocks):
            self.blocks.append(UNetDecoderBlock(
                in_ch=in_ch,
                skip_ch=skip_channels[i],
                out_ch=decoder_channels[i],
            ))
            in_ch = decoder_channels[i]

        self.out_channels = decoder_channels[-1]

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Args:
            features: [level_0 (highest res), ..., level_N (deepest)].

        Returns:
            (B, out_channels, H, W) decoded feature map.
        """
        n = len(features)
        x = features[-1]  # deepest

        for i, block in enumerate(self.blocks):
            skip_idx = n - 2 - i
            skip = features[skip_idx] if skip_idx >= 0 else None
            x = block(x, skip)

        return x


class PrithviUNetSegmentationModel(nn.Module):
    """Prithvi segmentation model with UNet decoder (BurnScars architecture).

    State dict layout (matching BurnScars checkpoint):
        encoder.*           → PrithviViT backbone
        neck.2.fpn1/fpn2    → Scale modules
        decoder.decoder.*   → UNet decoder blocks
        head.*              → Segmentation head

    The key difference from UPerNet variant:
        - Scale modules live under ``neck.2`` instead of ``decoder``
        - Decoder is UNet (progressive upsampling) not UPerNet (PSP+FPN)
        - Output channels = 64 (not 256)
    """

    def __init__(
        self,
        encoder,
        feature_indices: list[int] = (5, 11, 17, 23),
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_indices = list(feature_indices)

        if hasattr(encoder, "encoder"):
            self.encoder = encoder.encoder
        else:
            self.encoder = encoder

        self.embed_dim = self.encoder.embed_dim  # 1024 for 300M

        # Scale modules under neck.2 (matching checkpoint path)
        self.neck = nn.ModuleList([
            nn.Identity(),  # neck.0 placeholder
            nn.Identity(),  # neck.1 placeholder
            nn.Module(),    # neck.2 holds fpn1/fpn2
        ])

        # fpn1: ConvTranspose2d(1024→512) + BN + GELU + ConvTranspose2d(512→256)
        self.neck[2].fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, self.embed_dim // 2,
                               kernel_size=2, stride=2),
            nn.BatchNorm2d(self.embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(self.embed_dim // 2, self.embed_dim // 4,
                               kernel_size=2, stride=2),
        )

        # fpn2: ConvTranspose2d(1024→512)
        self.neck[2].fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, self.embed_dim // 2,
                               kernel_size=2, stride=2),
        )

        # Multi-scale channels after scaling
        scale_channels = [
            self.embed_dim // 4,  # 256 (after fpn1)
            self.embed_dim // 2,  # 512 (after fpn2)
            self.embed_dim,       # 1024
            self.embed_dim,       # 1024
        ]

        # UNet decoder
        self.decoder = nn.Module()
        self.decoder.decoder = TerraTorchUNetDecoder(
            encoder_channels=scale_channels,
            decoder_channels=[512, 256, 128, 64],
        )

        # Segmentation head (64 channels from UNet output)
        head_channels = self.decoder.decoder.out_channels
        self.head = SegmentationHead(head_channels, num_classes, dropout)

    def _extract_multi_scale_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract and rescale features from selected transformer blocks."""
        all_features = self.encoder.forward_features(x)
        selected = [all_features[i] for i in self.feature_indices]
        spatial = self.encoder.prepare_features_for_image_model(selected)

        scaled = [
            self.neck[2].fpn1(spatial[0]),   # (B, 256, 4*gh, 4*gw)
            self.neck[2].fpn2(spatial[1]),   # (B, 512, 2*gh, 2*gw)
            spatial[2],                      # (B, 1024, gh, gw)
            spatial[3],                      # (B, 1024, gh, gw)
        ]
        return scaled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward: encoder → neck → UNet decoder → head."""
        if x.dim() == 4:
            input_h, input_w = x.shape[2:]
        else:
            input_h, input_w = x.shape[3:]

        features = self._extract_multi_scale_features(x)
        decoded = self.decoder.decoder(features)
        logits = self.head(decoded)

        if logits.shape[2:] != (input_h, input_w):
            logits = F.interpolate(
                logits, size=(input_h, input_w),
                mode="bilinear", align_corners=True,
            )
        return logits
