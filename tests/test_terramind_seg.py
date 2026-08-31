"""Tests for imint.fm.terramind_seg.TerraMindSegmentationModel.

Uses a fake encoder that returns a (B, N, D) token sequence, so these
tests run without terratorch / TerraMind weights — they exercise the
wrapper's reshape/upsample/classifier logic and input validation.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from imint.fm.terramind_seg import TerraMindSegmentationModel


class _FakeEncoder(nn.Module):
    """Returns tokens of configurable shape. Accepts dict input."""

    def __init__(self, n_tokens: int = 196, embed_dim: int = 768,
                 add_cls: bool = False):
        super().__init__()
        self.n_tokens = n_tokens
        self.embed_dim = embed_dim
        self.add_cls = add_cls
        # A trainable param so requires_grad propagates meaningfully
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, inputs: dict) -> torch.Tensor:
        any_t = next(iter(inputs.values()))
        B = any_t.shape[0]
        n = self.n_tokens + (1 if self.add_cls else 0)
        return torch.randn(B, n, self.embed_dim) * self.scale


class TestTerraMindSegmentationForward:
    def test_output_shape_matches_input(self):
        enc = _FakeEncoder(n_tokens=196, embed_dim=768)
        model = TerraMindSegmentationModel(
            encoder=enc, num_classes=23,
            img_size=224, embed_dim=768, patch_size=16,
        )
        x = {"S2L2A": torch.randn(2, 6, 224, 224),
             "S1GRD": torch.randn(2, 2, 224, 224)}
        out = model(x)
        assert out.shape == (2, 23, 224, 224)

    def test_list_of_block_tokens_uses_last(self):
        """terratorch's real TerraMind backbone returns a LIST of per-block
        (B, N, D) token sequences — not a single tensor. The wrapper must
        take the last block. (Verified on-cluster: 12 tensors of (1,961,768)
        at 496px/patch16.)"""
        class _ListEncoder(nn.Module):
            def __init__(self, n_tokens, embed_dim, depth=12):
                super().__init__()
                self.n_tokens = n_tokens
                self.embed_dim = embed_dim
                self.depth = depth
                self.scale = nn.Parameter(torch.ones(1))

            def forward(self, inputs: dict):
                B = next(iter(inputs.values())).shape[0]
                return [torch.randn(B, self.n_tokens, self.embed_dim) * self.scale
                        for _ in range(self.depth)]

        enc = _ListEncoder(n_tokens=196, embed_dim=768)
        model = TerraMindSegmentationModel(
            encoder=enc, num_classes=23,
            img_size=224, embed_dim=768, patch_size=16,
        )
        x = {"S2L2A": torch.randn(2, 6, 224, 224),
             "S1GRD": torch.randn(2, 2, 224, 224)}
        out = model(x)
        assert out.shape == (2, 23, 224, 224)

    def test_cls_token_dropped(self):
        # B=2: the UPerNet head has BatchNorm (PSP pool_size=1 → 1×1 maps);
        # training-mode BN needs >1 sample. Real training uses B=8; B=1 only
        # occurs at eval where BN uses running stats. Test the realistic path.
        enc = _FakeEncoder(n_tokens=196, embed_dim=768, add_cls=True)
        model = TerraMindSegmentationModel(
            encoder=enc, num_classes=23,
            img_size=224, embed_dim=768, patch_size=16,
        )
        x = {"S2L2A": torch.randn(2, 6, 224, 224)}
        out = model(x)
        assert out.shape == (2, 23, 224, 224)

    def test_wrong_token_count_raises(self):
        enc = _FakeEncoder(n_tokens=400, embed_dim=768)  # not 196 or 197
        model = TerraMindSegmentationModel(
            encoder=enc, num_classes=23,
            img_size=224, embed_dim=768, patch_size=16,
        )
        x = {"S2L2A": torch.randn(2, 6, 224, 224)}
        with pytest.raises(ValueError, match="400 tokens"):
            model(x)

    def test_wrong_embed_dim_raises(self):
        enc = _FakeEncoder(n_tokens=196, embed_dim=512)
        model = TerraMindSegmentationModel(
            encoder=enc, num_classes=23,
            img_size=224, embed_dim=768, patch_size=16,
        )
        x = {"S2L2A": torch.randn(2, 6, 224, 224)}
        with pytest.raises(ValueError, match="embed_dim"):
            model(x)

    def test_non_dict_input_raises(self):
        enc = _FakeEncoder(n_tokens=196, embed_dim=768)
        model = TerraMindSegmentationModel(
            encoder=enc, num_classes=23,
            img_size=224, embed_dim=768, patch_size=16,
        )
        with pytest.raises(TypeError, match="dict"):
            model(torch.randn(1, 6, 224, 224))

    def test_indivisible_img_size_raises(self):
        enc = _FakeEncoder(n_tokens=196, embed_dim=768)
        with pytest.raises(ValueError, match="divisible"):
            TerraMindSegmentationModel(
                encoder=enc, num_classes=23,
                img_size=225, embed_dim=768, patch_size=16,
            )


class TestTerraMindSegmentationAux:
    def test_aux_channels_wired(self):
        # Aux now flows through the UPerNet head's gated mid-level fusion
        # (decoder_head.lidar_branch), not a top-level aux_proj concat.
        enc = _FakeEncoder(n_tokens=196, embed_dim=768)
        model = TerraMindSegmentationModel(
            encoder=enc, num_classes=23,
            img_size=224, embed_dim=768, patch_size=16,
            n_aux_channels=5,
        )
        assert model.decoder_head.lidar_branch is not None
        x = {"S2L2A": torch.randn(2, 6, 224, 224)}
        aux = torch.randn(2, 5, 224, 224)
        out = model(x, aux=aux)
        assert out.shape == (2, 23, 224, 224)

    def test_no_aux_channels(self):
        enc = _FakeEncoder(n_tokens=196, embed_dim=768)
        model = TerraMindSegmentationModel(
            encoder=enc, num_classes=23,
            img_size=224, embed_dim=768, patch_size=16,
            n_aux_channels=0,
        )
        assert model.decoder_head.lidar_branch is None
        x = {"S2L2A": torch.randn(2, 6, 224, 224)}
        out = model(x)
        assert out.shape == (2, 23, 224, 224)

    def test_aux_none_with_aux_channels_skipped(self):
        """Passing aux=None when the model has an aux branch must skip the
        branch gracefully (UPerNet gated fusion is only applied when aux is
        given), not crash — inference-time flexibility."""
        enc = _FakeEncoder(n_tokens=196, embed_dim=768)
        model = TerraMindSegmentationModel(
            encoder=enc, num_classes=23,
            img_size=224, embed_dim=768, patch_size=16,
            n_aux_channels=5,
        )
        x = {"S2L2A": torch.randn(2, 6, 224, 224)}
        out = model(x, aux=None)  # gated fusion skipped, no crash
        assert out.shape == (2, 23, 224, 224)


@pytest.mark.parametrize("img_size,patch_size", [
    (224, 16), (448, 16), (256, 16), (224, 14),
])
class TestTerraMindSegmentationResolutions:
    def test_various_resolutions(self, img_size, patch_size):
        grid = img_size // patch_size
        n_tokens = grid * grid
        enc = _FakeEncoder(n_tokens=n_tokens, embed_dim=768)
        model = TerraMindSegmentationModel(
            encoder=enc, num_classes=23,
            img_size=img_size, embed_dim=768, patch_size=patch_size,
        )
        x = {"S2L2A": torch.randn(2, 6, img_size, img_size)}
        out = model(x)
        assert out.shape == (2, 23, img_size, img_size)
