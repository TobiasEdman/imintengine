"""Tests for Lovász-softmax loss and extended CombinedLoss."""
import pytest
import torch

from imint.training.losses import (
    _lovasz_grad,
    LovaszSoftmaxLoss,
    FocalLoss,
    DiceLoss,
    CombinedLoss,
)


class TestLovaszGrad:

    def test_grad_shape_and_finite(self):
        """Lovász grad output should be same length and finite."""
        gt = torch.ones(10)
        grad = _lovasz_grad(gt)
        assert grad.shape == (10,)
        assert torch.isfinite(grad).all()

    def test_output_shape(self):
        gt = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0])
        grad = _lovasz_grad(gt)
        assert grad.shape == (5,)

    def test_grad_values_non_negative(self):
        """Lovász gradient should be non-negative."""
        gt = torch.tensor([1.0, 0.0, 1.0, 0.0])
        grad = _lovasz_grad(gt)
        assert (grad >= -1e-6).all()


class TestLovaszSoftmaxLoss:

    def test_returns_scalar(self):
        loss_fn = LovaszSoftmaxLoss(ignore_index=0)
        logits = torch.randn(2, 5, 16, 16)
        targets = torch.randint(0, 5, (2, 16, 16))
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_perfect_prediction_near_zero(self):
        """When model predicts correct class with high confidence, loss ≈ 0."""
        loss_fn = LovaszSoftmaxLoss(ignore_index=0)
        # 3 classes, all pixels are class 1
        targets = torch.ones(1, 8, 8, dtype=torch.long)
        logits = torch.zeros(1, 3, 8, 8)
        logits[:, 1, :, :] = 100.0  # very confident class 1
        loss = loss_fn(logits, targets)
        assert loss.item() < 0.01

    def test_ignore_index_excluded(self):
        """Pixels with ignore_index should not contribute to loss."""
        loss_fn = LovaszSoftmaxLoss(ignore_index=0)
        # All pixels are background (ignore_index=0)
        targets = torch.zeros(1, 8, 8, dtype=torch.long)
        logits = torch.randn(1, 5, 8, 8)
        loss = loss_fn(logits, targets)
        assert loss.item() == 0.0

    def test_backward_no_nan(self):
        """Gradients should be finite, no NaN or Inf."""
        loss_fn = LovaszSoftmaxLoss(ignore_index=0)
        logits = torch.randn(2, 5, 16, 16, requires_grad=True)
        targets = torch.randint(1, 5, (2, 16, 16))  # no background
        loss = loss_fn(logits, targets)
        loss.backward()
        assert torch.isfinite(logits.grad).all()

    def test_loss_decreases_with_better_prediction(self):
        """Loss should be lower when predictions are closer to target."""
        loss_fn = LovaszSoftmaxLoss(ignore_index=0)
        targets = torch.ones(1, 16, 16, dtype=torch.long) * 2  # all class 2

        # Bad prediction (class 1)
        bad_logits = torch.zeros(1, 5, 16, 16)
        bad_logits[:, 1, :, :] = 10.0
        bad_loss = loss_fn(bad_logits, targets)

        # Good prediction (class 2)
        good_logits = torch.zeros(1, 5, 16, 16)
        good_logits[:, 2, :, :] = 10.0
        good_loss = loss_fn(good_logits, targets)

        assert good_loss.item() < bad_loss.item()


class TestCombinedLossWithLovasz:

    def test_combined_without_lovasz_backward_compat(self):
        """CombinedLoss without lovász should behave as before."""
        focal = FocalLoss(gamma=2.0, ignore_index=0)
        dice = DiceLoss(ignore_index=0)
        criterion = CombinedLoss(focal, dice, focal_weight=0.5, dice_weight=0.5)
        logits = torch.randn(2, 5, 16, 16, requires_grad=True)
        targets = torch.randint(1, 5, (2, 16, 16))
        loss = criterion(logits, targets)
        loss.backward()
        assert loss.dim() == 0
        assert torch.isfinite(logits.grad).all()

    def test_combined_with_lovasz(self):
        """CombinedLoss with Lovász term should work end-to-end."""
        focal = FocalLoss(gamma=2.0, ignore_index=0)
        dice = DiceLoss(ignore_index=0)
        lovasz = LovaszSoftmaxLoss(ignore_index=0)
        criterion = CombinedLoss(
            focal, dice, lovasz=lovasz,
            focal_weight=0.35, dice_weight=0.35, lovasz_weight=0.3,
        )
        logits = torch.randn(2, 5, 16, 16, requires_grad=True)
        targets = torch.randint(1, 5, (2, 16, 16))
        loss = criterion(logits, targets)
        loss.backward()
        assert loss.dim() == 0
        assert torch.isfinite(logits.grad).all()

    def test_lovasz_weight_zero_equals_no_lovasz(self):
        """lovasz_weight=0 should give same result as no lovász."""
        focal = FocalLoss(gamma=2.0, ignore_index=0)
        dice = DiceLoss(ignore_index=0)
        lovasz = LovaszSoftmaxLoss(ignore_index=0)

        torch.manual_seed(42)
        logits = torch.randn(1, 5, 16, 16)
        targets = torch.randint(1, 5, (1, 16, 16))

        c_without = CombinedLoss(focal, dice, focal_weight=0.5, dice_weight=0.5)
        c_with_zero = CombinedLoss(
            focal, dice, lovasz=lovasz,
            focal_weight=0.5, dice_weight=0.5, lovasz_weight=0.0,
        )

        l1 = c_without(logits, targets)
        l2 = c_with_zero(logits, targets)
        assert torch.allclose(l1, l2, atol=1e-6)
