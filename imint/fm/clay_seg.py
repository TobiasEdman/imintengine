"""
imint/fm/clay_seg.py — Clay v1.5 segmentation wrapper.

Clay's documented ``model.encoder(chips, timestamps, wavelengths)`` returns
a pooled ``(B, 1024)`` embedding per image — useful for classification
and retrieval, but not for per-pixel segmentation.

For dense prediction we hook into the encoder's final transformer block
BEFORE the pooling to grab the (B, N+1, D) token sequence. At the
native Clay config (256 px input, patch_size=8) this gives N = 32*32 =
1024 patch tokens plus one CLS token. We drop the CLS, reshape the
rest to a (B, D=1024, 32, 32) spatial feature map, then run a linear-
probe head to predict per-pixel class logits.

Architecture (same philosophy as TerraMindSegmentationModel):
    1. encoder hook → pre-pool tokens (B, N+1, 1024)
    2. drop CLS, reshape → (B, 1024, grid, grid)
    3. 2x ConvTranspose up (1024→512, 512→256)
    4. 3x3 smooth → 128
    5. optional aux fusion
    6. 1x1 classifier → num_classes
    7. bilinear resize to input resolution

If mIoU disappoints, swap in a multi-level hook (hook blocks 5/11/17/23)
to feed a real UPerNet.

Usage:
    from imint.fm.clay_seg import ClaySegmentationModel
    model = ClaySegmentationModel(
        encoder=clay_encoder,        # from loader.load_clay
        num_classes=23,
        img_size=256,
        patch_size=8,
        embed_dim=1024,
    )
    logits = model(
        chips=s2_clay_tensor,        # (B, 10, H, W)
        wavelengths=wls,             # (B, 10) in nanometers
        timestamps=timestamps,       # (B, 4): [week, hour, lat, lon]
        aux=None,
    )
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClaySegmentationModel(nn.Module):
    """Hook-based linear-probe segmentation head on top of a Clay encoder.

    Args:
        encoder: Clay encoder from ``load_clay()``. Expected to be a
            ViT with a ``.blocks`` (or equivalent) module list.
        num_classes: Output classes.
        img_size: Input image size. Must be divisible by patch_size.
        patch_size: ViT patch size (Clay v1.5 uses 8).
        embed_dim: Encoder hidden size (Clay v1.5: 1024).
        n_aux_channels: Auxiliary raster channels, concatenated at
            output resolution before the classifier. 0 = no aux.
        dropout: Dropout before classifier.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int = 23,
        img_size: int = 256,
        patch_size: int = 8,
        embed_dim: int = 1024,
        n_aux_channels: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.n_aux_channels = n_aux_channels

        grid = img_size // patch_size
        if grid * patch_size != img_size:
            raise ValueError(
                f"img_size={img_size} must be divisible by patch_size={patch_size}"
            )
        self.grid_size = grid
        self.expected_n_patches = grid * grid  # 32×32 = 1024 at native

        # Progressive up-sampling (token spatial map → input resolution)
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

    def _extract_tokens(
        self,
        chips: torch.Tensor,
        timestamps: torch.Tensor,
        wavelengths: torch.Tensor,
    ) -> torch.Tensor:
        """Return the pre-pool token sequence (B, N+1, D).

        Clay's ``encoder.forward`` pools and returns (B, D). We hook the
        final transformer block's output to grab the un-pooled tokens.
        """
        captured: dict[str, torch.Tensor] = {}

        # Identify the last transformer block. Clay MAE ViT structures
        # vary across versions; try the common attribute names.
        blocks = None
        for attr in ("blocks", "transformer", "layers"):
            if hasattr(self.encoder, attr):
                blocks = getattr(self.encoder, attr)
                break
        if blocks is None:
            raise AttributeError(
                "Clay encoder has no .blocks/.transformer/.layers — "
                "cannot hook final transformer block. Inspect the "
                "encoder structure with print(encoder) and wire the "
                "correct attribute."
            )

        last_block = blocks[-1]

        def hook(module, inputs, output):
            # Some blocks return (tokens,) tuple; normalize to tensor
            t = output[0] if isinstance(output, tuple) else output
            captured["tokens"] = t

        handle = last_block.register_forward_hook(hook)
        try:
            with torch.no_grad():
                _ = self.encoder(chips, timestamps, wavelengths)
        finally:
            handle.remove()

        if "tokens" not in captured:
            raise RuntimeError(
                "Forward hook did not capture tokens from Clay's last "
                "transformer block. Encoder internals may have changed."
            )
        return captured["tokens"]

    def _tokens_to_spatial(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, N+1, D) or (B, N, D) → (B, D, grid, grid)."""
        B, N, D = tokens.shape
        if N == self.expected_n_patches + 1:
            tokens = tokens[:, 1:, :]  # drop CLS
        elif N != self.expected_n_patches:
            raise ValueError(
                f"Clay encoder returned {N} tokens; expected "
                f"{self.expected_n_patches} (grid {self.grid_size}²) or "
                f"{self.expected_n_patches + 1} (with CLS) for "
                f"img_size={self.img_size} patch_size={self.patch_size}. "
                f"Verify chips were passed at native resolution."
            )
        if D != self.embed_dim:
            raise ValueError(
                f"Clay encoder returned embed_dim={D}; expected {self.embed_dim}."
            )
        return tokens.transpose(1, 2).reshape(
            B, D, self.grid_size, self.grid_size,
        )

    def forward(
        self,
        chips: torch.Tensor,
        timestamps: torch.Tensor,
        wavelengths: torch.Tensor,
        aux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-pixel class logits from Clay encoder features.

        Args:
            chips: (B, n_bands, H, W) stacked Sentinel-2 tensor from
                ``build_s2_clay_tensor``. n_bands=10 for the default
                Clay spec; Clay's dynamic embedding block accepts
                variable band counts as long as ``wavelengths`` matches.
            timestamps: (B, 4) tensor [week, hour, lat, lon]. Pass
                zeros if time/location unknown.
            wavelengths: (B, n_bands) tensor of per-band central
                wavelength in nanometers.
            aux: Optional (B, n_aux, H, W) auxiliary raster channels.

        Returns:
            (B, num_classes, H, W) logits at input resolution.
        """
        input_h, input_w = chips.shape[-2:]

        tokens = self._extract_tokens(chips, timestamps, wavelengths)
        feat = self._tokens_to_spatial(tokens)  # (B, D, grid, grid)

        feat = self.up1(feat)
        feat = self.up2(feat)
        feat = self.smooth(feat)

        feat = F.interpolate(
            feat, size=(input_h, input_w), mode="bilinear", align_corners=True,
        )

        if self.aux_proj is not None and aux is not None:
            aux_feat = self.aux_proj(aux)
            feat = torch.cat([feat, aux_feat], dim=1)

        feat = self.dropout(feat)
        logits = self.classifier(feat)
        return logits
