"""Checkpoint reconstruction must not fetch a second set of FM weights."""

from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")


def _package(monkeypatch, name: str) -> types.ModuleType:
    package = types.ModuleType(name)
    package.__path__ = []
    monkeypatch.setitem(sys.modules, name, package)
    return package


def test_clay_uninitialized_mode_builds_only_the_large_encoder(monkeypatch):
    """The rung checkpoint supplies weights; no 5 GB Clay load or teacher."""
    _package(monkeypatch, "claymodel")
    clay_module = types.ModuleType("claymodel.module")
    clay_model = types.ModuleType("claymodel.model")
    calls: list[dict] = []

    class ClayMAEModule:
        @classmethod
        def load_from_checkpoint(cls, *args, **kwargs):
            raise AssertionError("pretrained checkpoint load must not run")

    class Encoder(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            calls.append(kwargs)
            self.sentinel = torch.nn.Parameter(torch.ones(1))

    clay_module.ClayMAEModule = ClayMAEModule
    clay_model.Encoder = Encoder
    monkeypatch.setitem(sys.modules, "claymodel.module", clay_module)
    monkeypatch.setitem(sys.modules, "claymodel.model", clay_model)

    from imint.fm.loaders.clay import load_clay

    model = load_clay(pretrained=False, num_frames=1, img_size=504)

    assert calls == [{
        "mask_ratio": 0.0,
        "patch_size": 8,
        "shuffle": False,
        "dim": 1024,
        "depth": 24,
        "heads": 16,
        "dim_head": 64,
        "mlp_ratio": 4,
    }]
    assert isinstance(model.encoder, Encoder)
    assert model.training is False
    # The loader contributes the first ``encoder`` component.  The outer
    # segmentation model contributes the second, matching rung checkpoints.
    assert list(model.state_dict()) == ["encoder.sentinel"]

    class SegmentationShell(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

    assert list(SegmentationShell(model).state_dict()) == [
        "encoder.encoder.sentinel",
    ]
    assert not hasattr(model, "teacher")
    assert not hasattr(model, "decoder")


def test_croma_uninitialized_mode_preserves_upstream_module_graph(monkeypatch):
    """Upstream eagerly torch.loads even for an empty path; bypass only I/O."""
    use_croma = types.ModuleType("use_croma")
    vit_calls: list[dict] = []
    cross_calls: list[dict] = []
    bias_calls: list[dict] = []

    class PretrainedCROMA(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            raise AssertionError("eager upstream torch.load path must not run")

    class ViT(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            vit_calls.append(kwargs)
            self.sentinel = torch.nn.Parameter(torch.ones(1))

    class BaseTransformerCrossAttn(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            cross_calls.append(kwargs)
            self.sentinel = torch.nn.Parameter(torch.ones(1))

    def get_2dalibi(**kwargs):
        bias_calls.append(kwargs)
        return torch.zeros(1)

    use_croma.PretrainedCROMA = PretrainedCROMA
    use_croma.ViT = ViT
    use_croma.BaseTransformerCrossAttn = BaseTransformerCrossAttn
    use_croma.get_2dalibi = get_2dalibi
    monkeypatch.setitem(sys.modules, "use_croma", use_croma)

    from imint.fm.loaders.croma import load_croma

    model = load_croma(
        pretrained=False,
        num_frames=1,
        img_size=504,
        variant="base",
        modality="both",
    )

    assert model.encoder_dim == 768
    assert model.encoder_depth == 12
    assert model.num_patches == 63 * 63
    assert bias_calls == [{"num_heads": 16, "num_patches": 63 * 63}]
    assert vit_calls == [
        {"dim": 768, "depth": 6, "in_channels": 2},
        {"dim": 768, "depth": 12, "in_channels": 12},
    ]
    assert cross_calls == [{"dim": 768, "depth": 6, "num_heads": 16}]
    keys = set(model.state_dict())
    assert {
        "s1_encoder.sentinel",
        "GAP_FFN_s1.0.weight",
        "s2_encoder.sentinel",
        "GAP_FFN_s2.0.weight",
        "cross_encoder.sentinel",
    } <= keys
    assert model.training is False
