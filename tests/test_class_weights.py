"""Tests for compute_class_weights with all weighting methods."""
import numpy as np
import pytest

from imint.training.class_schema import compute_class_weights


# Realistic 23-class pixel counts (approximating v5 schema distribution)
SAMPLE_COUNTS = {
    0: 50_000_000,   # bakgrund
    1: 4_000_000,    # tallskog
    2: 2_500_000,    # granskog
    3: 6_400_000,    # lövskog
    4: 1_900_000,    # blandskog
    5: 1_000_000,    # sumpskog
    6: 2_300_000,    # tillfälligt ej skog
    7: 2_200_000,    # våtmark
    8: 3_400_000,    # öppen mark
    9: 900_000,      # bebyggelse
    10: 3_000_000,   # vatten
    11: 485_000,     # vete
    12: 354_000,     # korn
    13: 156_000,     # havre
    14: 124_000,     # oljeväxter
    15: 700_000,     # slåttervall
    16: 168_000,     # bete
    17: 44_000,      # potatis
    18: 24_000,      # sockerbetor
    19: 76_000,      # trindsäd
    20: 45_000,      # råg
    21: 1_042_000,   # majs
    22: 562_000,     # hygge
}
NUM_CLASSES = 23


class TestWeightingMethods:
    """Test all three weighting strategies produce valid, distinct outputs."""

    @pytest.mark.parametrize("method", ["inverse", "sqrt", "effective_number"])
    def test_returns_correct_shape(self, method):
        w = compute_class_weights(SAMPLE_COUNTS, NUM_CLASSES, method=method)
        assert w.shape == (NUM_CLASSES,)
        assert w.dtype == np.float32

    @pytest.mark.parametrize("method", ["inverse", "sqrt", "effective_number"])
    def test_ignore_index_is_zero(self, method):
        w = compute_class_weights(SAMPLE_COUNTS, NUM_CLASSES, ignore_index=0, method=method)
        assert w[0] == 0.0

    @pytest.mark.parametrize("method", ["inverse", "sqrt", "effective_number"])
    def test_all_weights_non_negative(self, method):
        w = compute_class_weights(SAMPLE_COUNTS, NUM_CLASSES, method=method)
        assert (w >= 0).all()

    @pytest.mark.parametrize("method", ["inverse", "sqrt", "effective_number"])
    def test_max_weight_cap_respected(self, method):
        for cap in [3.0, 5.0, 10.0]:
            w = compute_class_weights(SAMPLE_COUNTS, NUM_CLASSES, max_weight=cap, method=method)
            assert w.max() <= cap + 1e-6, f"{method} exceeded cap {cap}: max={w.max()}"

    @pytest.mark.parametrize("method", ["inverse", "sqrt"])
    def test_rare_classes_get_higher_weight(self, method):
        w = compute_class_weights(SAMPLE_COUNTS, NUM_CLASSES, max_weight=100.0, method=method)
        # sockerbetor (18, 24k pixels) should weigh more than tallskog (1, 4M pixels)
        assert w[18] > w[1], f"{method}: sockerbetor ({w[18]:.3f}) should > tallskog ({w[1]:.3f})"

    def test_effective_number_nearly_uniform_at_large_counts(self):
        """With millions of pixels, effective_number produces near-uniform weights.
        This is a known property — confirmed by v5b where crops died."""
        w = compute_class_weights(SAMPLE_COUNTS, NUM_CLASSES, max_weight=100.0, method="effective_number")
        active = w[1:]
        ratio = active.max() / active.min()
        assert ratio < 2.0, f"effective_number should be nearly uniform, got ratio {ratio:.1f}"

    def test_inverse_has_widest_range(self):
        """Inverse-freq should produce the widest weight spread."""
        w_inv = compute_class_weights(SAMPLE_COUNTS, NUM_CLASSES, max_weight=100.0, method="inverse")
        w_sqrt = compute_class_weights(SAMPLE_COUNTS, NUM_CLASSES, max_weight=100.0, method="sqrt")
        w_eff = compute_class_weights(SAMPLE_COUNTS, NUM_CLASSES, max_weight=100.0, method="effective_number")
        active = slice(1, None)  # skip ignore_index=0
        range_inv = w_inv[active].max() / w_inv[active].min()
        range_sqrt = w_sqrt[active].max() / w_sqrt[active].min()
        range_eff = w_eff[active].max() / w_eff[active].min()
        assert range_inv > range_sqrt, f"inverse range ({range_inv:.1f}) should > sqrt ({range_sqrt:.1f})"
        assert range_sqrt > range_eff, f"sqrt range ({range_sqrt:.1f}) should > effective ({range_eff:.1f})"

    def test_sqrt_is_sqrt_of_inverse(self):
        """Sqrt weights == sqrt(inverse weights) when cap is high enough."""
        cap = 1000.0  # high cap so neither method clips
        w_inv = compute_class_weights(SAMPLE_COUNTS, NUM_CLASSES, max_weight=cap, method="inverse")
        w_sqrt = compute_class_weights(SAMPLE_COUNTS, NUM_CLASSES, max_weight=cap, method="sqrt")
        np.testing.assert_allclose(w_sqrt[1:], np.sqrt(w_inv[1:]), rtol=1e-5)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown weighting method"):
            compute_class_weights(SAMPLE_COUNTS, NUM_CLASSES, method="bogus")

    def test_empty_counts_no_crash(self):
        """All counts zero should still produce valid weights."""
        for method in ["inverse", "sqrt", "effective_number"]:
            w = compute_class_weights({}, NUM_CLASSES, method=method)
            assert w.shape == (NUM_CLASSES,)
            assert np.isfinite(w).all()

    def test_single_class_dominant(self):
        """One class with 99.9% of pixels should get low weight."""
        counts = {0: 0, 1: 100_000_000, 2: 100}
        for method in ["inverse", "sqrt", "effective_number"]:
            w = compute_class_weights(counts, 3, max_weight=50.0, method=method)
            assert w[2] > w[1], f"{method}: rare class should weigh more"


class TestWeightRanges:
    """Verify expected weight ranges for each method with realistic data."""

    def test_inverse_range(self):
        w = compute_class_weights(SAMPLE_COUNTS, NUM_CLASSES, max_weight=10.0, method="inverse")
        active = w[1:]  # skip background
        assert active.min() > 0.1, f"inverse min too low: {active.min():.3f}"
        assert active.max() <= 10.0

    def test_sqrt_range(self):
        w = compute_class_weights(SAMPLE_COUNTS, NUM_CLASSES, max_weight=5.0, method="sqrt")
        active = w[1:]
        assert active.min() > 0.3, f"sqrt min too low: {active.min():.3f}"
        assert active.max() <= 5.0

    def test_effective_number_range(self):
        w = compute_class_weights(SAMPLE_COUNTS, NUM_CLASSES, max_weight=5.0, method="effective_number")
        active = w[1:]
        assert active.min() > 0.01, f"effective min too low: {active.min():.3f}"
        assert active.max() <= 5.0
