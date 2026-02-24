"""
imint/training/losses.py — Custom loss functions for LULC segmentation

Provides FocalLoss as an alternative to standard CrossEntropyLoss,
which focuses training on hard/rare examples by down-weighting
easy, well-classified pixels.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise ImportError("PyTorch is required. Install with: pip install torch")


class FocalLoss(nn.Module):
    """Focal Loss for multi-class segmentation.

    Adds ``(1 - p_t)^gamma`` modulation to cross-entropy, focusing
    training on hard/rare examples.  When ``gamma=0``, this reduces
    to standard weighted cross-entropy.

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    Args:
        weight: Optional per-class weight tensor (same semantics as
            ``torch.nn.CrossEntropyLoss.weight``).
        gamma: Focusing parameter.  Higher values focus more on hard
            examples.  Typical range: 1.0 - 3.0.  Default: 2.0.
        ignore_index: Class index to ignore in loss computation.
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 2.0,
        ignore_index: int = 0,
    ):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.register_buffer("weight", weight)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: (B, C, H, W) raw class scores.
            targets: (B, H, W) integer class labels.

        Returns:
            Scalar loss tensor.
        """
        # Standard cross-entropy per pixel (no reduction)
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )  # (B, H, W)

        # p_t = probability of the correct class
        # Using the identity: p_t = exp(-ce_loss) for numerical stability
        p_t = torch.exp(-ce_loss)

        # Focal modulation: (1 - p_t)^gamma
        focal_factor = torch.clamp(1.0 - p_t, min=1e-6) ** self.gamma

        # Apply focal modulation
        focal_loss = focal_factor * ce_loss

        # Mask out ignored pixels
        valid_mask = targets != self.ignore_index
        if valid_mask.any():
            return focal_loss[valid_mask].mean()
        return focal_loss.mean()
