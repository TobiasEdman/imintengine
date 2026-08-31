"""Regression tests for the strict-deterministic UPerNet spatial path."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from imint.fm.upernet import (
    DeterministicAdaptiveAvgPool2d,
    _resize_bilinear,
)


def _set_determinism(enabled: bool, warn_only: bool) -> None:
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def test_strict_bilinear_matrix_path_matches_pytorch_output_and_gradient():
    old_enabled = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    source = torch.randn(2, 3, 7, 5)
    expected_input = source.clone().requires_grad_(True)
    expected = F.interpolate(
        expected_input, size=(13, 11), mode="bilinear", align_corners=True,
    )
    expected.square().sum().backward()

    try:
        _set_determinism(True, False)
        actual_input = source.clone().requires_grad_(True)
        actual = _resize_bilinear(
            actual_input, (13, 11), align_corners=True,
        )
        actual.square().sum().backward()
    finally:
        _set_determinism(old_enabled, old_warn_only)

    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(
        actual_input.grad, expected_input.grad, rtol=2e-6, atol=2e-6,
    )


def test_strict_adaptive_pool_matrix_path_matches_pytorch_output_and_gradient():
    old_enabled = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    source = torch.randn(2, 3, 11, 9)
    expected_input = source.clone().requires_grad_(True)
    expected = F.adaptive_avg_pool2d(expected_input, (4, 3))
    expected.square().sum().backward()

    try:
        _set_determinism(True, False)
        actual_input = source.clone().requires_grad_(True)
        actual = DeterministicAdaptiveAvgPool2d((4, 3))(actual_input)
        actual.square().sum().backward()
    finally:
        _set_determinism(old_enabled, old_warn_only)

    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(
        actual_input.grad, expected_input.grad, rtol=2e-6, atol=2e-6,
    )
