"""
imint/fm/tessera_seg.py — TESSERA segmentation head.

TESSERA provides a 128-D embedding per pixel already at native 10 m
GSD via the pre-computed annual embeddings in each tile's .npz.
No encoder forward pass is needed; we just run a small segmentation
head directly on the (128, H, W) embedding tensor.

Why keep it small: the embedding itself is the 311M-parameter
encoder's output, squeezed to 128 dims. A heavy decoder would
overfit on our small training set and defeat the whole point of
TESSERA (cheap, label-efficient transfer).

Architecture:
    1. 1x1 Conv(128 → 256)                  (expand, share context)
    2. 3x3 Conv(256 → 128)                  (spatial smoothing)
    3. Optional aux fusion (concat)
    4. Dropout
    5. 1x1 Conv(128 → num_classes)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TesseraSegmentationModel(nn.Module):
    """Shallow pixel-wise classifier on TESSERA 128-D embeddings.

    Args:
        encoder: Passthrough module from ``load_tessera()``. The
            ``forward(embeddings)`` multiplies by a learnable scale
            to keep the graph trainable, but does no heavy work.
        num_classes: Output classes (default 23).
        embed_dim: TESSERA embedding dim (128).
        hidden: Hidden channel count for the small head (default 256).
        n_aux_channels: Optional aux raster channels fused at output res.
        dropout: Dropout before classifier.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int = 23,
        embed_dim: int = 128,
        hidden: int = 256,
        n_aux_channels: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.n_aux_channels = n_aux_channels

        self.expand = nn.Sequential(
            nn.Conv2d(embed_dim, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden), nn.GELU(),
        )
        self.smooth = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden // 2), nn.GELU(),
        )

        final_in = hidden // 2
        if n_aux_channels > 0:
            self.aux_proj = nn.Sequential(
                nn.Conv2d(n_aux_channels, hidden // 2, kernel_size=3,
                          padding=1, bias=False),
                nn.BatchNorm2d(hidden // 2), nn.GELU(),
            )
            final_in = hidden
        else:
            self.aux_proj = None

        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(final_in, num_classes, kernel_size=1)

    def forward(
        self,
        embeddings: torch.Tensor,
        aux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-pixel class logits.

        Args:
            embeddings: (B, 128, H, W) TESSERA embedding tensor.
                Typically read from the ``tessera`` key in each .npz.
                Accepts fp16 — internally promoted to fp32 by the
                BatchNorm layers.
            aux: Optional (B, n_aux, H, W) auxiliary raster channels.

        Returns:
            (B, num_classes, H, W) logits.
        """
        if embeddings.dim() != 4 or embeddings.shape[1] != self.embed_dim:
            raise ValueError(
                f"Expected (B, {self.embed_dim}, H, W) embedding tensor; "
                f"got shape {tuple(embeddings.shape)}."
            )

        x = self.encoder(embeddings.float())   # passthrough + learnable scale
        x = self.expand(x)
        x = self.smooth(x)

        if self.aux_proj is not None and aux is not None:
            a = self.aux_proj(aux)
            x = torch.cat([x, a], dim=1)

        x = self.dropout(x)
        return self.classifier(x)
