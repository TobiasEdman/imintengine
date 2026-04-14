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


def parcel_area_to_pixel_weights(
    area_ha: "torch.Tensor",
    mmu_ha: float = 0.25,
    max_weight: float = 4.0,
) -> "torch.Tensor":
    """Per-pixel loss weights derived from LPIS parcel area.

    Ensures small parcels receive proportionally higher gradient so the model
    learns to delineate sub-hectare field boundaries, not just bulk crop types.

    Weight function:
        area == 0  (NMD non-crop / background)  →  1.0  (uniform weight)
        0 < area < mmu_ha  (sub-MMU crop)        →  max_weight  (highest emphasis)
        area >= mmu_ha     (normal crop)          →  clip(1/√area, 1.0, max_weight)

    NMD non-crop pixels (forest, water, urban, etc.) have area_ha==0 because
    they carry no LPIS parcel. Defaulting to 1.0 ensures the model trains on
    all 23 classes, not just LPIS crop classes.  The ignore_index in FocalLoss /
    DiceLoss handles true background (label==0) independently of pixel_weight.

    At 0.25 ha: w = 1/√0.25 = 2.0
    At 1.00 ha: w = 1/√1.00 = 1.0  (floor — large parcels carry standard weight)
    At 0.05 ha: w = max_weight = 4.0

    Args:
        area_ha:    (B, H, W) or (H, W) float32 tensor of parcel area per pixel.
        mmu_ha:     Minimum Mapping Unit in hectares (default 0.25, matching NMD).
        max_weight: Maximum weight cap for sub-MMU pixels (default 4.0).

    Returns:
        Float32 tensor of same shape as area_ha.
    """
    weight = torch.ones_like(area_ha)   # default 1.0 for NMD/non-LPIS pixels

    crop_mask = area_ha > 0.0
    sub_mmu   = crop_mask & (area_ha < mmu_ha)
    normal    = area_ha >= mmu_ha

    weight[sub_mmu] = max_weight
    weight[normal]  = torch.clamp(
        1.0 / torch.sqrt(area_ha[normal]),
        min=1.0, max=max_weight,
    )
    return weight


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
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        pixel_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits:       (B, C, H, W) raw class scores.
            targets:      (B, H, W) integer class labels.
            pixel_weight: Optional (B, H, W) float32 per-pixel weights.
                          Use parcel_area_to_pixel_weights() to derive from
                          parcel area so small parcels get higher emphasis.

        Returns:
            Scalar loss tensor.
        """
        # Standard cross-entropy per pixel with label smoothing
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )  # (B, H, W)

        # p_t = probability of the correct class
        # Using the identity: p_t = exp(-ce_loss) for numerical stability
        p_t = torch.exp(-ce_loss)

        # Focal modulation: (1 - p_t)^gamma
        focal_factor = torch.clamp(1.0 - p_t, min=1e-6) ** self.gamma

        # Apply focal modulation
        focal_loss = focal_factor * ce_loss

        # Per-pixel area weighting (emphasises small parcels)
        if pixel_weight is not None:
            focal_loss = focal_loss * pixel_weight.to(focal_loss)

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
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        pixel_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute soft dice loss.

        Args:
            logits:       (B, C, H, W) raw class scores.
            targets:      (B, H, W) integer class labels.
            pixel_weight: Optional (B, H, W) float32 per-pixel weights.

        Returns:
            Scalar loss tensor (1 - mean_dice).
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        # One-hot encode targets: (B, H, W) → (B, C, H, W)
        targets_oh = F.one_hot(
            targets.clamp(0, num_classes - 1).long(), num_classes
        ).permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Combined mask: ignore_index pixels AND per-pixel area weights
        valid_mask = (targets != self.ignore_index).float()
        if pixel_weight is not None:
            valid_mask = valid_mask * pixel_weight.to(valid_mask)
        valid_mask = valid_mask.unsqueeze(1)   # (B, 1, H, W)
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
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        pixel_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return (
            self.focal_weight * self.focal(logits, targets, pixel_weight)
            + self.dice_weight * self.dice(logits, targets, pixel_weight)
        )
