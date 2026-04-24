"""Tests for imint.fm.normalize — per-model input normalizers."""
from __future__ import annotations

import pytest
import torch

from imint.fm.normalize import (
    NORMALIZERS,
    ClayNormalizer,
    CromaNormalizer,
    PrithviNormalizer,
    TerraMindNormalizer,
    ThorNormalizer,
)


class TestPrithviNormalizer:
    def test_5d_shape_preserved(self):
        n = PrithviNormalizer()
        x = torch.rand(2, 6, 4, 32, 32)
        assert n(x).shape == x.shape

    def test_4d_shape_preserved(self):
        n = PrithviNormalizer()
        x = torch.rand(2, 6, 32, 32)
        assert n(x).shape == x.shape

    def test_mean_std_registered_as_buffers(self):
        n = PrithviNormalizer()
        buf_names = {name for name, _ in n.named_buffers()}
        assert "mean" in buf_names
        assert "std" in buf_names

    def test_device_move(self):
        n = PrithviNormalizer()
        assert n.mean.device.type == "cpu"
        # No actual CUDA/MPS hardware assumption — just check .to() doesn't crash
        n2 = n.to("cpu")
        assert n2.mean.device.type == "cpu"

    def test_normalization_formula(self):
        """(x*10000 - mean) / std — verify on a known point."""
        n = PrithviNormalizer()
        # Constant input of 0.1 reflectance → DN = 1000
        x = torch.full((1, 6, 1, 1, 1), 0.1)
        out = n(x)
        # Band 0: (1000 - 1087) / 2248 ≈ -0.0387
        expected = (1000.0 - 1087.0) / 2248.0
        assert abs(out[0, 0, 0, 0, 0].item() - expected) < 1e-4


class TestClayNormalizer:
    def test_shape(self):
        n = ClayNormalizer()
        x = torch.rand(2, 7, 32, 32)
        assert n(x).shape == x.shape

    def test_mean_std_shape(self):
        n = ClayNormalizer()
        assert n.mean.shape == (1, 7, 1, 1)
        assert n.std.shape == (1, 7, 1, 1)


class TestTerraMindNormalizer:
    def test_s2_only(self):
        n = TerraMindNormalizer()
        out = n({"s2": torch.rand(2, 6, 32, 32)})
        assert "s2" in out
        assert "s1" not in out
        assert out["s2"].shape == (2, 6, 32, 32)

    def test_all_modalities(self):
        n = TerraMindNormalizer()
        inputs = {
            "s2": torch.rand(2, 6, 32, 32),
            "s1": torch.rand(2, 2, 32, 32) * 0.1,
            "dem": torch.rand(2, 1, 32, 32) * 500,
        }
        out = n(inputs)
        assert set(out.keys()) == {"s2", "s1", "dem"}

    def test_s2_temporal_5d(self):
        n = TerraMindNormalizer()
        out = n({"s2": torch.rand(2, 6, 4, 32, 32)})
        assert out["s2"].shape == (2, 6, 4, 32, 32)

    def test_s1_log_scale(self):
        """S1 should apply 10*log10() — verify with known input."""
        n = TerraMindNormalizer()
        # Linear σ⁰ = 0.01 → dB = -20
        x = torch.full((1, 2, 1, 1), 0.01)
        out = n({"s1": x})["s1"]
        # Band 0: (-20 - (-15)) / 5 = -1.0
        expected = (-20.0 - (-15.0)) / 5.0
        assert abs(out[0, 0, 0, 0].item() - expected) < 1e-3


class TestCromaNormalizer:
    def test_s2_12band(self):
        n = CromaNormalizer()
        out = n({"s2_full": torch.rand(2, 12, 32, 32)})
        assert out["s2_full"].shape == (2, 12, 32, 32)

    def test_s1(self):
        n = CromaNormalizer()
        out = n({"s1": torch.rand(2, 2, 32, 32) * 0.1})
        assert out["s1"].shape == (2, 2, 32, 32)


class TestThorNormalizer:
    def test_passthrough_tensor(self):
        n = ThorNormalizer()
        x = torch.rand(2, 8, 32, 32)
        assert torch.equal(n(x), x)

    def test_passthrough_dict(self):
        n = ThorNormalizer()
        d = {"s2": torch.rand(1, 6, 8, 8)}
        assert n(d) is d


class TestNormalizerRegistry:
    @pytest.mark.parametrize("family", sorted(NORMALIZERS.keys()))
    def test_all_instantiable(self, family):
        inst = NORMALIZERS[family]()
        assert isinstance(inst, torch.nn.Module)

    def test_registry_coverage_matches_model_families(self):
        from imint.fm.registry import MODEL_CONFIGS
        needed = {spec.normalizer_family for spec in MODEL_CONFIGS.values()}
        assert needed.issubset(NORMALIZERS.keys())
