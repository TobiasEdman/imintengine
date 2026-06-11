"""Tests for the hand-rolled auroc_aupr (imint/eval/metrics.py).

Checks the textbook anchors (perfect / inverted / random) plus tie handling
and the one-class guard, so the metric can be trusted when scoring NFI
maturity against sampled model probabilities.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from imint.eval.metrics import auroc_aupr


def test_perfect_separation():
    auroc, aupr = auroc_aupr([0.9, 0.8, 0.3, 0.2], [1, 1, 0, 0])
    assert auroc == pytest.approx(1.0)
    assert aupr == pytest.approx(1.0)


def test_perfectly_inverted():
    auroc, _ = auroc_aupr([0.9, 0.8, 0.3, 0.2], [0, 0, 1, 1])
    assert auroc == pytest.approx(0.0)


def test_random_diagonal_with_ties():
    # Tied scores with balanced labels collapse to the chance diagonal.
    auroc, _ = auroc_aupr([1, 1, 0, 0], [1, 0, 1, 0])
    assert auroc == pytest.approx(0.5)


def test_known_intermediate():
    # Sorted desc: 0.9(1) 0.8(0) 0.7(1) 0.6(0) → AUROC 0.75 (worked by hand).
    auroc, _ = auroc_aupr([0.9, 0.8, 0.7, 0.6], [1, 0, 1, 0])
    assert auroc == pytest.approx(0.75)


def test_single_class_returns_nan():
    a1, p1 = auroc_aupr([0.5, 0.6, 0.7], [1, 1, 1])
    a2, p2 = auroc_aupr([0.5, 0.6, 0.7], [0, 0, 0])
    assert all(math.isnan(v) for v in (a1, p1, a2, p2))


def test_invariant_to_monotonic_score_scaling():
    # AUROC is rank-based: any strictly increasing transform leaves it fixed.
    rng = np.random.default_rng(0)
    scores = rng.random(200)
    target = (rng.random(200) < scores).astype(int)  # correlated labels
    base, _ = auroc_aupr(scores, target)
    scaled, _ = auroc_aupr(scores * 3.0 + 1.0, target)
    assert base == pytest.approx(scaled)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        auroc_aupr([0.1, 0.2, 0.3], [0, 1])
