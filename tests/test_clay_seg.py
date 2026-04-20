"""Tests for imint.fm.clay_seg.ClaySegmentationModel.

Clay's real encoder needs the `claymodel` package and a 1.25 GB
checkpoint, so tests use a fake encoder that mimics Clay's structure:
a `.blocks` ModuleList where the last block's output is the token
sequence the wrapper needs to hook.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from imint.fm.clay_seg import ClaySegmentationModel


class _FakeClayBlock(nn.Module):
    """Pass-through identity — lets the forward hook capture whatever
    token tensor was passed in, as a real transformer block would
    return."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _FakeClayEncoder(nn.Module):
    """Mimics Clay's ClayMAEModule.model.encoder interface.

    - Has ``.blocks`` ModuleList so the wrapper can hook the last one.
    - Forward returns (B, embed_dim) — the pooled vector, matching
      Clay's real API. The wrapper ignores this return value and uses
      the hook instead.

    The forward runs a sequence:
        patch_embed (image→tokens)
        prepend CLS token
        pass through each block (we capture the last block's output)
        pool (mean over patch tokens) → (B, embed_dim)
    """

    def __init__(self, patch_size: int = 8, embed_dim: int = 1024,
                 include_cls: bool = True, n_bands: int = 10):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.include_cls = include_cls
        self.n_bands = n_bands
        # Conv-based patch embedder: (B, C, H, W) → (B, D, H/ps, W/ps)
        self.patch_embed = nn.Conv2d(
            n_bands, embed_dim, kernel_size=patch_size, stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([_FakeClayBlock(embed_dim) for _ in range(4)])

    def forward(self, chips: torch.Tensor, timestamps: torch.Tensor,
                wavelengths: torch.Tensor) -> torch.Tensor:
        B = chips.shape[0]
        # Patch embed → (B, D, Gh, Gw) then flatten → (B, Gh*Gw, D)
        x = self.patch_embed(chips)
        tokens = x.flatten(2).transpose(1, 2)

        if self.include_cls:
            cls = self.cls_token.expand(B, 1, self.embed_dim)
            tokens = torch.cat([cls, tokens], dim=1)

        for block in self.blocks:
            tokens = block(tokens)

        # Pool patch tokens → (B, D), matching Clay's real pooled return
        if self.include_cls:
            patch_tokens = tokens[:, 1:, :]
        else:
            patch_tokens = tokens
        return patch_tokens.mean(dim=1)


class TestClaySegmentationForward:
    def test_output_shape_native_config(self):
        enc = _FakeClayEncoder(patch_size=8, embed_dim=1024, include_cls=True)
        model = ClaySegmentationModel(
            encoder=enc, num_classes=23,
            img_size=256, patch_size=8, embed_dim=1024,
        )
        chips = torch.randn(2, 10, 256, 256)
        ts = torch.zeros(2, 4)
        wl = torch.full((2, 10), 500.0)
        out = model(chips, ts, wl)
        assert out.shape == (2, 23, 256, 256)

    def test_output_shape_without_cls(self):
        enc = _FakeClayEncoder(patch_size=8, embed_dim=1024, include_cls=False)
        model = ClaySegmentationModel(
            encoder=enc, num_classes=23,
            img_size=256, patch_size=8, embed_dim=1024,
        )
        chips = torch.randn(1, 10, 256, 256)
        ts = torch.zeros(1, 4)
        wl = torch.full((1, 10), 500.0)
        out = model(chips, ts, wl)
        assert out.shape == (1, 23, 256, 256)

    def test_wrong_embed_dim_raises(self):
        enc = _FakeClayEncoder(patch_size=8, embed_dim=768, include_cls=True)
        model = ClaySegmentationModel(
            encoder=enc, num_classes=23,
            img_size=256, patch_size=8, embed_dim=1024,  # expects 1024
        )
        chips = torch.randn(1, 10, 256, 256)
        with pytest.raises(ValueError, match="embed_dim"):
            model(chips, torch.zeros(1, 4), torch.full((1, 10), 500.0))

    def test_indivisible_img_size_raises(self):
        enc = _FakeClayEncoder(patch_size=8, embed_dim=1024, include_cls=True)
        with pytest.raises(ValueError, match="divisible"):
            ClaySegmentationModel(
                encoder=enc, num_classes=23,
                img_size=250, patch_size=8, embed_dim=1024,
            )

    def test_encoder_without_blocks_raises(self):
        class NoBlocks(nn.Module):
            def forward(self, *args, **kwargs):
                return torch.zeros(1, 1024)
        model = ClaySegmentationModel(
            encoder=NoBlocks(), num_classes=23,
            img_size=256, patch_size=8, embed_dim=1024,
        )
        chips = torch.randn(1, 10, 256, 256)
        with pytest.raises(AttributeError, match="blocks"):
            model(chips, torch.zeros(1, 4), torch.full((1, 10), 500.0))


class TestClaySegmentationAux:
    def test_aux_wired(self):
        enc = _FakeClayEncoder(patch_size=8, embed_dim=1024, include_cls=True)
        model = ClaySegmentationModel(
            encoder=enc, num_classes=23,
            img_size=256, patch_size=8, embed_dim=1024,
            n_aux_channels=5,
        )
        chips = torch.randn(1, 10, 256, 256)
        aux = torch.randn(1, 5, 256, 256)
        out = model(
            chips, torch.zeros(1, 4), torch.full((1, 10), 500.0),
            aux=aux,
        )
        assert out.shape == (1, 23, 256, 256)

    def test_no_aux(self):
        enc = _FakeClayEncoder(patch_size=8, embed_dim=1024, include_cls=True)
        model = ClaySegmentationModel(
            encoder=enc, num_classes=23,
            img_size=256, patch_size=8, embed_dim=1024,
            n_aux_channels=0,
        )
        assert model.aux_proj is None
        chips = torch.randn(1, 10, 256, 256)
        out = model(chips, torch.zeros(1, 4), torch.full((1, 10), 500.0))
        assert out.shape == (1, 23, 256, 256)


@pytest.mark.parametrize("img_size", [128, 256, 512])
class TestClaySegmentationResolutions:
    def test_various_resolutions(self, img_size):
        enc = _FakeClayEncoder(
            patch_size=8, embed_dim=1024, include_cls=True,
        )
        model = ClaySegmentationModel(
            encoder=enc, num_classes=23,
            img_size=img_size, patch_size=8, embed_dim=1024,
        )
        chips = torch.randn(1, 10, img_size, img_size)
        out = model(chips, torch.zeros(1, 4), torch.full((1, 10), 500.0))
        assert out.shape == (1, 23, img_size, img_size)
