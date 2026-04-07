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


class DiceLoss(nn.Module):
    """Soft Dice Loss for multi-class segmentation.

    Computes per-class soft dice and averages across valid classes.
    Particularly effective for classes with high inter-class confusion
    (e.g. forest types) because it directly optimizes overlap.

    Args:
        ignore_index: Class index to exclude from loss computation.
        smooth: Smoothing constant to avoid division by zero.
    """

    def __init__(self, ignore_index: int = 0, smooth: float = 1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute soft dice loss.

        Args:
            logits: (B, C, H, W) raw class scores.
            targets: (B, H, W) integer class labels.

        Returns:
            Scalar loss tensor (1 - mean_dice).
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        # One-hot encode targets: (B, H, W) → (B, C, H, W)
        targets_oh = F.one_hot(
            targets.clamp(0, num_classes - 1).long(), num_classes
        ).permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Mask out ignored pixels
        valid_mask = (targets != self.ignore_index).unsqueeze(1).float()
        probs = probs * valid_mask
        targets_oh = targets_oh * valid_mask

        # Per-class dice
        dims = (0, 2, 3)  # sum over batch, H, W
        intersection = (probs * targets_oh).sum(dims)
        cardinality = probs.sum(dims) + targets_oh.sum(dims)
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Average over non-ignored classes that have ground truth
        valid_classes = []
        for c in range(num_classes):
            if c == self.ignore_index:
                continue
            if targets_oh[:, c].sum() > 0:
                valid_classes.append(dice_per_class[c])

        if valid_classes:
            mean_dice = torch.stack(valid_classes).mean()
        else:
            mean_dice = dice_per_class[1:].mean()

        return 1.0 - mean_dice


class CombinedLoss(nn.Module):
    """Focal + Dice combined loss.

    Args:
        focal: FocalLoss instance.
        dice: DiceLoss instance.
        focal_weight: Weight for focal loss (default 0.5).
        dice_weight: Weight for dice loss (default 0.5).
    """

    def __init__(
        self,
        focal: FocalLoss,
        dice: DiceLoss,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
    ):
        super().__init__()
        self.focal = focal
        self.dice = dice
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor,
    ) -> torch.Tensor:
        return (self.focal_weight * self.focal(logits, targets)
                + self.dice_weight * self.dice(logits, targets))
