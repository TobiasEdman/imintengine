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


def get_default_pool_sizes(
    device: str | torch.device | None = None,
    img_size: int = 224,
    patch_size: int = 16,
) -> tuple[int, ...]:
    """Return PSP pool sizes appropriate for feature map dimensions.

    The deepest PSP feature map has size ``img_size // patch_size``.
    Pool sizes must evenly divide this for AdaptiveAvgPool2d (especially
    on MPS). Supports patch_size=8 (Clay/CROMA), 14 (Prithvi-600M) and
    16 (Prithvi-300M/TerraMind/THOR).

    Args:
        device: Target device string or torch.device.
        img_size: Training input resolution (224, 256, or 448).
        patch_size: ViT patch size (8, 14, or 16).

    Returns:
        Tuple of pool sizes for PSP modules.
    """
    fm = img_size // patch_size  # feature map spatial size after patch embed

    if fm >= 56:       # 448px/8 → 56×56 (Clay)
        return (1, 2, 4, 7, 14, 28)
    if fm >= 32:       # 448px/14 → 32×32 (Prithvi-600M)
        return (1, 2, 4, 8, 16)
    if fm >= 28:       # 448px/16 → 28×28, 224px/8 → 28×28
        return (1, 2, 4, 7, 14)
    if fm >= 16:       # 256px/16 → 16×16, 224px/14 → 16×16
        return (1, 2, 4, 8)
    # 224px/16 → 14×14
    if device is not None and ("cuda" in str(device) or "cpu" in str(device)):
        return (1, 2, 3, 6)  # not all divide 14 cleanly but works on CUDA
    return (1, 2, 7, 14)      # MPS-safe: all divide 14


class ConvBnRelu(nn.Module):
    """Conv2d + BatchNorm2d + ReLU block matching TerraTorch naming.

    State dict keys: conv.weight, norm.weight, norm.bias, norm.running_mean, etc.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1,
                 bias: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=bias)
        self.norm = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.norm(self.conv(x.contiguous())))


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
        pool_sizes: tuple[int, ...] = (1, 2, 7, 14),
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
            pooled = F.interpolate(
                pooled, size=(h, w), mode="bilinear", align_corners=True,
            ).contiguous()
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
            ).contiguous()
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
                ).contiguous()
            resized.append(out)

        return self.fpn_bottleneck(torch.cat(resized, dim=1))


class AuxEncoder(nn.Module):
    """Lightweight CNN encoder for auxiliary raster channels (legacy late-fusion).

    Processes (B, N, H, W) auxiliary channels (e.g. tree height, timber
    volume, basal area, DEM) into (B, out_ch, H, W) feature maps at full
    input resolution.  Used for late fusion with the decoder output.
    """

    def __init__(self, in_channels: int, out_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            ConvBnRelu(in_channels, out_channels, kernel=3, padding=1),
            ConvBnRelu(out_channels, out_channels, kernel=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LiDARBranch(nn.Module):
    """Dual-branch CNN encoder for LiDAR/auxiliary channels.

    Produces a feature map that is fused at multiple decoder levels
    via GatedFusion, instead of only at the final layer.

    Architecture: 2× Conv(3×3) + BN + ReLU at full input resolution.
    No downsampling — spatial alignment is handled by F.interpolate
    at each FPN level in _decode().

    Args:
        in_channels: Number of auxiliary input channels.
        out_channels: Feature dimension (default 64).
    """

    def __init__(self, in_channels: int, out_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            ConvBnRelu(in_channels, 32, kernel=3, padding=1),
            ConvBnRelu(32, out_channels, kernel=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedFusion(nn.Module):
    """Gated fusion between spectral (S2) and auxiliary (LiDAR) features.

    Learns a per-pixel gate that blends spectral and auxiliary features:
        fused = S2 + gate * (aux_proj - S2)

    This residual form is more stable than direct interpolation because
    the gradient flows directly through S2 even when the gate is near 0.

    Args:
        s2_channels: Channel dim of spectral features (typically 256).
        aux_channels: Channel dim of auxiliary features (typically 64).
    """

    def __init__(self, s2_channels: int = 256, aux_channels: int = 64):
        super().__init__()
        # Project aux to match S2 channels
        self.aux_proj = ConvBnRelu(aux_channels, s2_channels, kernel=1, padding=0)
        # Gate: takes concatenated [S2, aux_proj] → sigmoid weight
        self.gate = nn.Sequential(
            nn.Conv2d(s2_channels * 2, s2_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, s2: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """Fuse spectral and auxiliary features.

        Args:
            s2: (B, C, H, W) spectral decoder features.
            aux: (B, C_aux, H, W) auxiliary features (resized to match s2).

        Returns:
            (B, C, H, W) fused features.
        """
        aux_p = self.aux_proj(aux)
        weight = self.gate(torch.cat([s2, aux_p], dim=1))
        return s2 + weight * (aux_p - s2)


class TemporalPooling(nn.Module):
    """Temporal mean+max pooling for multi-temporal feature aggregation.

    Reshapes (B, T*C, H, W) → (B, T, C, H, W), computes mean and max
    over T, concatenates → (B, 2*C, H, W).

    This provides explicit temporal aggregation while preserving both
    the average signal (mean) and peak responses (max) across frames.

    Args:
        embed_dim: Per-frame channel dimension (e.g. 1024 for Prithvi-300M).
        num_frames: Number of temporal frames T.
    """

    def __init__(self, embed_dim: int, num_frames: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool temporal frames.

        Args:
            x: (B, T*embed_dim, H, W)

        Returns:
            (B, 2*embed_dim, H, W) concatenated mean+max pooled features.
        """
        B, _, H, W = x.shape
        # Reshape: (B, T*C, H, W) → (B, T, C, H, W)
        x = x.view(B, self.num_frames, self.embed_dim, H, W)
        mean_pool = x.mean(dim=1)   # (B, C, H, W)
        max_pool = x.max(dim=1)[0]  # (B, C, H, W)
        return torch.cat([mean_pool, max_pool], dim=1)  # (B, 2*C, H, W)


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
        n_aux_channels: int = 0,
        pool_sizes: tuple[int, ...] | None = None,
        enable_temporal_pooling: bool = True,
        enable_multilevel_aux: bool = True,
    ):
        super().__init__()
        self.feature_indices = list(feature_indices)
        self.n_aux_channels = n_aux_channels
        self.enable_temporal_pooling = enable_temporal_pooling
        self.enable_multilevel_aux = enable_multilevel_aux

        # Get the ViT encoder from PrithviMAE if needed
        if hasattr(encoder, "encoder"):
            self.encoder = encoder.encoder
        else:
            self.encoder = encoder

        self.embed_dim = self.encoder.embed_dim  # 1024 for 300M

        # Compute effective temporal dimension
        pe = self.encoder.patch_embed
        effective_t = pe.input_size[0] // pe.patch_size[0]  # e.g. 4/1=4
        self.effective_t = effective_t
        raw_feature_dim = self.embed_dim * effective_t  # 1024*4=4096

        # Temporal pooling: mean+max → 2*embed_dim channels (halves raw_feature_dim)
        if enable_temporal_pooling and effective_t > 1:
            self.temporal_pool = TemporalPooling(self.embed_dim, effective_t)
            self.feature_dim = self.embed_dim * 2  # 2048 (mean+max)
        else:
            self.temporal_pool = None
            self.feature_dim = raw_feature_dim  # 4096 (legacy)

        # Scale modules: create multi-scale from ViT's uniform-scale features
        self.decoder = nn.Module()

        # fpn1: feature_dim → feature_dim//2 (2× up) → feature_dim//4 (2× up)
        self.decoder.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.feature_dim, self.feature_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(self.feature_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(self.feature_dim // 2, self.feature_dim // 4, kernel_size=2, stride=2),
        )

        # fpn2: feature_dim → feature_dim//2 (2× up)
        self.decoder.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.feature_dim, self.feature_dim // 2, kernel_size=2, stride=2),
        )

        # UperNet decoder channel sizes
        scale_channels = [
            self.feature_dim // 4,  # after fpn1
            self.feature_dim // 2,  # after fpn2
            self.feature_dim,       # pass-through
            self.feature_dim,       # deepest → PSP
        ]

        # PSP modules
        if pool_sizes is None:
            pool_sizes = get_default_pool_sizes()
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

        # Auxiliary fusion: mid-level (gated per FPN level) or legacy late fusion
        if n_aux_channels > 0 and enable_multilevel_aux:
            # Mid-level gated fusion at each FPN level + PSP level
            self.lidar_branch = LiDARBranch(n_aux_channels, out_channels=64)
            n_levels = len(scale_channels)
            self.gated_fusions = nn.ModuleList([
                GatedFusion(decoder_channels, 64) for _ in range(n_levels)
            ])
            # No late fusion — mid-level handles everything
            self.aux_encoder = None
            self.aux_fusion = None
        elif n_aux_channels > 0:
            # Legacy late fusion (backward compat)
            self.lidar_branch = None
            self.gated_fusions = None
            self.aux_encoder = AuxEncoder(n_aux_channels, out_channels=64)
            self.aux_fusion = ConvBnRelu(
                decoder_channels + 64, decoder_channels,
                kernel=1, padding=0,
            )
        else:
            self.lidar_branch = None
            self.gated_fusions = None
            self.aux_encoder = None
            self.aux_fusion = None

    def _extract_multi_scale_features(
        self,
        x: torch.Tensor,
        temporal_coords: torch.Tensor | None = None,
        location_coords: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Extract and rescale features from selected transformer blocks.

        When temporal pooling is enabled, applies mean+max pooling over the
        temporal dimension before scaling, reducing T*embed_dim → 2*embed_dim.

        Returns list of [level_0 (highest res), ..., level_3 (lowest/deepest)].
        """
        # Get all block outputs: list of (B, N_tokens+1, embed_dim)
        all_features = self.encoder.forward_features(
            x, temporal_coords=temporal_coords, location_coords=location_coords,
        )

        # Select features at specified indices
        selected = [all_features[i] for i in self.feature_indices]

        # Reshape tokens → spatial maps: list of (B, T*embed_dim, gh, gw)
        spatial = self.encoder.prepare_features_for_image_model(selected)

        # Temporal pooling: (B, T*embed_dim, H, W) → (B, 2*embed_dim, H, W)
        if self.temporal_pool is not None:
            spatial = [self.temporal_pool(feat) for feat in spatial]

        # Apply scale modules to create multi-scale
        scaled = [
            self.decoder.fpn1(spatial[0]),   # (B, feature_dim//4, 4*gh, 4*gw)
            self.decoder.fpn2(spatial[1]),   # (B, feature_dim//2, 2*gh, 2*gw)
            spatial[2],                      # (B, feature_dim, gh, gw)
            spatial[3],                      # (B, feature_dim, gh, gw)
        ]

        return scaled

    def _decode(
        self,
        features: list[torch.Tensor],
        aux_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run UperNet decode on multi-scale features with optional aux fusion.

        Args:
            features: Multi-scale feature maps from _extract_multi_scale_features.
            aux_feat: Optional (B, 64, H, W) LiDAR branch features at input
                resolution. If provided and gated_fusions is set, fuses at
                each FPN level via GatedFusion.

        Returns (B, 256, H_0, W_0) feature map.
        """
        # PSP on deepest
        deepest = features[-1]
        h, w = deepest.shape[2:]
        psp_outs = [deepest]
        for psp_module in self.decoder.psp_modules:
            pooled = psp_module(deepest)
            pooled = F.interpolate(
                pooled, size=(h, w), mode="bilinear", align_corners=True,
            ).contiguous()
            psp_outs.append(pooled)
        psp_out = self.decoder.bottleneck(torch.cat(psp_outs, dim=1))

        # Mid-level aux fusion at PSP level (level 3)
        if self.gated_fusions is not None and aux_feat is not None:
            aux_resized = F.interpolate(
                aux_feat, size=(h, w), mode="bilinear", align_corners=True,
            ).contiguous()
            psp_out = self.gated_fusions[-1](psp_out, aux_resized)

        # FPN top-down
        n = len(features)
        fpn_outs = [psp_out]
        for i in range(n - 2, -1, -1):
            lateral = self.decoder.lateral_convs[i](features[i])
            target_h, target_w = lateral.shape[2:]
            upsampled = F.interpolate(
                fpn_outs[0], size=(target_h, target_w),
                mode="bilinear", align_corners=True,
            ).contiguous()
            fpn_out = self.decoder.fpn_convs[i](lateral + upsampled)

            # Mid-level aux fusion at this FPN level
            if self.gated_fusions is not None and aux_feat is not None:
                aux_resized = F.interpolate(
                    aux_feat, size=(target_h, target_w),
                    mode="bilinear", align_corners=True,
                ).contiguous()
                fpn_out = self.gated_fusions[i](fpn_out, aux_resized)

            fpn_outs.insert(0, fpn_out)

        # Resize all to highest resolution
        target_h, target_w = fpn_outs[0].shape[2:]
        resized = []
        for out in fpn_outs:
            if out.shape[2:] != (target_h, target_w):
                out = F.interpolate(
                    out, size=(target_h, target_w),
                    mode="bilinear", align_corners=True,
                ).contiguous()
            resized.append(out)

        return self.decoder.fpn_bottleneck(torch.cat(resized, dim=1))

    def forward(
        self,
        x: torch.Tensor,
        aux: torch.Tensor | None = None,
        temporal_coords: torch.Tensor | None = None,
        location_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Full forward pass: encoder → temporal pool → scale → UperNet → head.

        Args:
            x: (B, C, H, W) or (B, C, T, H, W) input tensor.
            aux: Optional (B, N, H, W) auxiliary raster channels
                (e.g. height, volume, basal area).
            temporal_coords: Optional (B, T, 2) float32 [year, doy] per frame.
            location_coords: Optional (B, 2) float32 [lat, lon] in WGS84.

        Returns:
            (B, num_classes, H, W) logits at input resolution.
        """
        if x.dim() == 4:
            input_h, input_w = x.shape[2:]
        else:
            input_h, input_w = x.shape[3:]

        features = self._extract_multi_scale_features(
            x, temporal_coords=temporal_coords, location_coords=location_coords,
        )

        # Mid-level fusion: LiDAR branch → gated fusion at each FPN level
        if self.lidar_branch is not None and aux is not None:
            aux_feat = self.lidar_branch(aux)  # (B, 64, H, W)
            decoded = self._decode(features, aux_feat=aux_feat)
        # Legacy late fusion
        elif self.aux_encoder is not None and aux is not None:
            decoded = self._decode(features)
            decoded = F.interpolate(
                decoded, size=(input_h, input_w),
                mode="bilinear", align_corners=True,
            ).contiguous()
            aux_enc = self.aux_encoder(aux)
            decoded = self.aux_fusion(
                torch.cat([decoded, aux_enc], dim=1))
            logits = self.head(decoded)
            return logits
        else:
            decoded = self._decode(features)

        # Final head + upsample to input resolution
        logits = self.head(decoded)
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
            nn.ReLU(inplace=False),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor,
                skip: torch.Tensor | None = None) -> torch.Tensor:
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=True).contiguous()
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x.contiguous())
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


# ── Foundation model segmentation factory (registry-aware) ───────────────────

# Alias: FMSegmentationModel for forward-compat naming. The Prithvi model
# is the reference implementation. TerraMind/Clay/CROMA will get family-
# specific wrappers in Fas 3-5; they can import this alias now.
FMSegmentationModel = PrithviSegmentationModel


def build_segmentation_from_spec(
    spec,
    *,
    encoder,
    num_classes: int,
    img_size: int = 224,
    decoder_channels: int = 256,
    dropout: float = 0.1,
    n_aux_channels: int = 0,
    enable_temporal_pooling: bool = True,
    enable_multilevel_aux: bool = True,
    device: str | torch.device | None = None,
):
    """Build a segmentation model from a ModelSpec + encoder.

    Uses spec.feature_indices and spec.patch_size to configure the
    UPerNet decoder correctly for the given backbone family.

    Currently routes all Prithvi-family specs to PrithviSegmentationModel.
    Non-Prithvi families raise NotImplementedError until their wrappers
    land in Fas 3-5.

    Args:
        spec: imint.fm.registry.ModelSpec.
        encoder: The backbone model from spec.loader_fn(...).
        num_classes: Output classes.
        img_size: Input resolution — used to pick safe PSP pool sizes.
        decoder_channels: UPerNet internal channels.
        dropout: Segmentation head dropout.
        n_aux_channels: Auxiliary raster channels (DEM, VPP, etc.).
        enable_temporal_pooling: Use mean+max over temporal frames.
        enable_multilevel_aux: Use gated mid-level aux fusion.
        device: Used to pick MPS-safe pool sizes.

    Returns:
        nn.Module ready for training/inference at input resolution.
    """
    pool_sizes = get_default_pool_sizes(
        device=device, img_size=img_size, patch_size=spec.patch_size,
    )

    if spec.family in ("prithvi",):
        return PrithviSegmentationModel(
            encoder=encoder,
            feature_indices=spec.feature_indices,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            dropout=dropout,
            n_aux_channels=n_aux_channels,
            pool_sizes=pool_sizes,
            enable_temporal_pooling=enable_temporal_pooling,
            enable_multilevel_aux=enable_multilevel_aux,
        )

    if spec.family == "terramind":
        # TerraMind returns a token sequence, not per-block spatial
        # features. Use a linear-probe-style head for the first ensemble
        # member (see imint/fm/terramind_seg.py for rationale).
        from imint.fm.terramind_seg import TerraMindSegmentationModel
        return TerraMindSegmentationModel(
            encoder=encoder,
            num_classes=num_classes,
            img_size=img_size,
            embed_dim=spec.embed_dim,
            patch_size=spec.patch_size,
            n_aux_channels=n_aux_channels,
            dropout=dropout,
        )

    raise NotImplementedError(
        f"Segmentation wrapper for family={spec.family!r} not implemented yet. "
        f"Prithvi + TerraMind available. Clay/CROMA/THOR/TESSERA land in Fas 4-5c."
    )
