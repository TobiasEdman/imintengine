"""Tests for imint.fm.tessera_seg.TesseraSegmentationModel and
imint.fm.loaders.tessera.load_tessera.

TESSERA has no encoder at our train time — the "loader" is just a
learnable-scale passthrough. The head consumes the pre-baked
(B, 128, H, W) embedding tensor directly.
"""
from __future__ import annotations

import pytest
import torch

from imint.fm.loaders.tessera import load_tessera, _TesseraPassthrough
from imint.fm.tessera_seg import TesseraSegmentationModel


class TestLoadTessera:
    def test_returns_passthrough(self):
        enc = load_tessera()
        assert isinstance(enc, _TesseraPassthrough)

    def test_passthrough_preserves_shape(self):
        enc = load_tessera()
        x = torch.randn(2, 128, 32, 32)
        y = enc(x)
        assert y.shape == x.shape

    def test_passthrough_is_learnable(self):
        enc = load_tessera()
        params = list(enc.parameters())
        assert len(params) == 1  # the scale param
        assert params[0].requires_grad

    def test_num_frames_must_be_1(self):
        with pytest.raises(ValueError, match="annual"):
            load_tessera(num_frames=4)


class TestTesseraSegmentationForward:
    def test_output_shape(self):
        enc = load_tessera()
        model = TesseraSegmentationModel(
            encoder=enc, num_classes=23, embed_dim=128,
        )
        emb = torch.randn(2, 128, 512, 512)
        out = model(emb)
        assert out.shape == (2, 23, 512, 512)

    def test_accepts_fp16(self):
        enc = load_tessera()
        model = TesseraSegmentationModel(
            encoder=enc, num_classes=23, embed_dim=128,
        )
        emb = torch.randn(1, 128, 64, 64, dtype=torch.float16)
        out = model(emb)
        assert out.shape == (1, 23, 64, 64)

    def test_wrong_embed_dim_raises(self):
        enc = load_tessera()
        model = TesseraSegmentationModel(
            encoder=enc, num_classes=23, embed_dim=128,
        )
        emb = torch.randn(1, 256, 32, 32)  # wrong channels
        with pytest.raises(ValueError, match="128"):
            model(emb)

    def test_wrong_rank_raises(self):
        enc = load_tessera()
        model = TesseraSegmentationModel(
            encoder=enc, num_classes=23, embed_dim=128,
        )
        emb = torch.randn(128, 32, 32)  # missing batch dim
        with pytest.raises(ValueError):
            model(emb)

    def test_aux_wired(self):
        enc = load_tessera()
        model = TesseraSegmentationModel(
            encoder=enc, num_classes=23, embed_dim=128,
            n_aux_channels=5,
        )
        emb = torch.randn(1, 128, 64, 64)
        aux = torch.randn(1, 5, 64, 64)
        out = model(emb, aux=aux)
        assert out.shape == (1, 23, 64, 64)

    def test_no_aux(self):
        enc = load_tessera()
        model = TesseraSegmentationModel(
            encoder=enc, num_classes=23, embed_dim=128,
            n_aux_channels=0,
        )
        assert model.aux_proj is None
        emb = torch.randn(1, 128, 64, 64)
        out = model(emb)
        assert out.shape == (1, 23, 64, 64)

    @pytest.mark.parametrize("h,w", [(256, 256), (512, 512), (224, 224)])
    def test_various_resolutions(self, h, w):
        enc = load_tessera()
        model = TesseraSegmentationModel(
            encoder=enc, num_classes=23, embed_dim=128,
        )
        emb = torch.randn(1, 128, h, w)
        out = model(emb)
        assert out.shape == (1, 23, h, w)


class TestRegistryIntegration:
    def test_tessera_in_registry(self):
        from imint.fm.registry import MODEL_CONFIGS
        assert "tessera_v1" in MODEL_CONFIGS
        spec = MODEL_CONFIGS["tessera_v1"]
        assert spec.family == "tessera"
        assert spec.embed_dim == 128

    def test_tessera_build_backbone(self):
        from imint.fm.registry import build_backbone
        model, spec = build_backbone("tessera_v1", num_frames=1, pretrained=False)
        assert isinstance(model, _TesseraPassthrough)
        assert spec.embed_dim == 128

    def test_tessera_build_segmentation(self):
        from imint.fm.registry import build_backbone, MODEL_CONFIGS
        from imint.fm.upernet import build_segmentation_from_spec
        model, spec = build_backbone("tessera_v1", num_frames=1, pretrained=False)
        seg = build_segmentation_from_spec(
            spec, encoder=model, num_classes=23, img_size=512,
        )
        assert isinstance(seg, TesseraSegmentationModel)
        emb = torch.randn(1, 128, 256, 256)
        out = seg(emb)
        assert out.shape == (1, 23, 256, 256)
